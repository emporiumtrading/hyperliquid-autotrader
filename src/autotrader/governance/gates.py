"""Governance gates: candidate must beat baseline on utility and respect risk constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GateConfig:
    """Thresholds and flags that control the promotion gate evaluation.

    All numeric thresholds represent *absolute* constraints unless stated
    otherwise.  The ``utility_improvement_threshold`` is a *relative*
    improvement requirement: the candidate's utility must exceed the
    baseline's utility by at least this amount.
    """

    utility_improvement_threshold: float = 0.05
    max_drawdown_pct: float = 0.20
    max_cvar_95: float = 0.10
    min_sharpe: float = 0.5
    min_profit_factor: float = 1.2
    min_win_rate: float = 0.35
    require_walkforward: bool = True
    require_robustness: bool = True
    robustness_pass_rate: float = 0.7


def load_gate_config(cfg: dict) -> GateConfig:
    """Extract a :class:`GateConfig` from a top-level configuration dict.

    Looks for a ``"governance"`` (or ``"gates"``) sub-dictionary.  Any keys
    that match :class:`GateConfig` field names are forwarded; unrecognised
    keys are silently ignored so that the config can evolve without breaking
    older code.

    Parameters
    ----------
    cfg:
        Top-level application configuration dictionary.

    Returns
    -------
    GateConfig
        Populated gate configuration.
    """
    section = cfg.get("governance", cfg.get("gates", {}))
    known_fields = {f.name for f in GateConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in section.items() if k in known_fields}
    return GateConfig(**filtered)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_metric(report: dict, *keys: str, default: float = 0.0) -> float:
    """Walk *report* looking for the first matching key path.

    Supports nested lookup: it first tries ``report[key]``, then
    ``report["metrics_summary"][key]``, then ``report["metrics"][key]``.
    """
    for key in keys:
        # Direct top-level
        if key in report:
            val = report[key]
            if isinstance(val, (int, float)):
                return float(val)

        # Nested under metrics_summary or metrics
        for container_key in ("metrics_summary", "metrics"):
            container = report.get(container_key, {})
            if isinstance(container, dict) and key in container:
                val = container[key]
                if isinstance(val, (int, float)):
                    return float(val)

    return default


def _make_gate(
    name: str,
    passed: bool,
    candidate_value: float,
    threshold: float,
    baseline_value: float | None = None,
) -> dict:
    """Build a single gate result dictionary."""
    gate: dict[str, Any] = {
        "passed": passed,
        "candidate_value": candidate_value,
        "threshold": threshold,
    }
    if baseline_value is not None:
        gate["baseline_value"] = baseline_value
    return gate


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def evaluate_candidate(
    candidate_report: dict,
    baseline_report: dict | None,
    config: GateConfig | None = None,
) -> dict:
    """Run the full governance gate evaluation.

    Parameters
    ----------
    candidate_report:
        Metrics / report dictionary for the candidate strategy run.
    baseline_report:
        Metrics / report dictionary for the current baseline.  Pass ``None``
        when no baseline exists yet (first-time promotion).
    config:
        Gate thresholds.  Falls back to :class:`GateConfig` defaults when
        ``None``.

    Returns
    -------
    dict
        Evaluation result with keys ``passed``, ``reasons``, ``gates``,
        ``delta``, and ``recommendation``.
    """
    if config is None:
        config = GateConfig()

    gates: dict[str, dict] = {}
    reasons: list[str] = []
    delta: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Fast-path: no baseline means automatic promotion (first baseline)
    # ------------------------------------------------------------------
    if baseline_report is None:
        logger.info("gates.evaluate.no_baseline", result="auto_pass")
        return {
            "passed": True,
            "reasons": [],
            "gates": {},
            "delta": {},
            "recommendation": "promote",
        }

    # ------------------------------------------------------------------
    # 1. Utility improvement
    # ------------------------------------------------------------------
    candidate_utility = _get_metric(candidate_report, "utility", "total_pnl", "net_pnl")
    baseline_utility = _get_metric(baseline_report, "utility", "total_pnl", "net_pnl")
    required_utility = baseline_utility + config.utility_improvement_threshold
    utility_passed = candidate_utility > required_utility
    gates["utility_improvement"] = _make_gate(
        "utility_improvement",
        utility_passed,
        candidate_utility,
        required_utility,
        baseline_utility,
    )
    delta["utility"] = candidate_utility - baseline_utility
    if not utility_passed:
        reasons.append(
            f"Utility {candidate_utility:.4f} does not exceed "
            f"baseline {baseline_utility:.4f} + threshold "
            f"{config.utility_improvement_threshold:.4f} = {required_utility:.4f}"
        )

    # ------------------------------------------------------------------
    # 2. Max drawdown
    # ------------------------------------------------------------------
    candidate_dd = abs(_get_metric(candidate_report, "max_drawdown", "max_drawdown_pct"))
    baseline_dd = abs(_get_metric(baseline_report, "max_drawdown", "max_drawdown_pct"))
    dd_passed = candidate_dd <= config.max_drawdown_pct
    gates["max_drawdown"] = _make_gate(
        "max_drawdown",
        dd_passed,
        candidate_dd,
        config.max_drawdown_pct,
        baseline_dd,
    )
    delta["max_drawdown"] = candidate_dd - baseline_dd
    if not dd_passed:
        reasons.append(
            f"Max drawdown {candidate_dd:.4f} exceeds limit {config.max_drawdown_pct:.4f}"
        )

    # ------------------------------------------------------------------
    # 3. CVaR 95
    # ------------------------------------------------------------------
    candidate_cvar = abs(_get_metric(candidate_report, "cvar_95", "cvar95"))
    baseline_cvar = abs(_get_metric(baseline_report, "cvar_95", "cvar95"))
    cvar_passed = candidate_cvar <= config.max_cvar_95
    gates["cvar_95"] = _make_gate(
        "cvar_95",
        cvar_passed,
        candidate_cvar,
        config.max_cvar_95,
        baseline_cvar,
    )
    delta["cvar_95"] = candidate_cvar - baseline_cvar
    if not cvar_passed:
        reasons.append(f"CVaR-95 {candidate_cvar:.4f} exceeds limit {config.max_cvar_95:.4f}")

    # ------------------------------------------------------------------
    # 4. Sharpe ratio
    # ------------------------------------------------------------------
    candidate_sharpe = _get_metric(candidate_report, "sharpe", "sharpe_ratio")
    baseline_sharpe = _get_metric(baseline_report, "sharpe", "sharpe_ratio")
    sharpe_passed = candidate_sharpe >= config.min_sharpe
    gates["sharpe"] = _make_gate(
        "sharpe",
        sharpe_passed,
        candidate_sharpe,
        config.min_sharpe,
        baseline_sharpe,
    )
    delta["sharpe"] = candidate_sharpe - baseline_sharpe
    if not sharpe_passed:
        reasons.append(f"Sharpe {candidate_sharpe:.4f} below minimum {config.min_sharpe:.4f}")

    # ------------------------------------------------------------------
    # 5. Profit factor
    # ------------------------------------------------------------------
    candidate_pf = _get_metric(candidate_report, "profit_factor")
    baseline_pf = _get_metric(baseline_report, "profit_factor")
    pf_passed = candidate_pf >= config.min_profit_factor
    gates["profit_factor"] = _make_gate(
        "profit_factor",
        pf_passed,
        candidate_pf,
        config.min_profit_factor,
        baseline_pf,
    )
    delta["profit_factor"] = candidate_pf - baseline_pf
    if not pf_passed:
        reasons.append(
            f"Profit factor {candidate_pf:.4f} below minimum {config.min_profit_factor:.4f}"
        )

    # ------------------------------------------------------------------
    # 6. Win rate
    # ------------------------------------------------------------------
    candidate_wr = _get_metric(candidate_report, "win_rate")
    baseline_wr = _get_metric(baseline_report, "win_rate")
    wr_passed = candidate_wr >= config.min_win_rate
    gates["win_rate"] = _make_gate(
        "win_rate",
        wr_passed,
        candidate_wr,
        config.min_win_rate,
        baseline_wr,
    )
    delta["win_rate"] = candidate_wr - baseline_wr
    if not wr_passed:
        reasons.append(f"Win rate {candidate_wr:.4f} below minimum {config.min_win_rate:.4f}")

    # ------------------------------------------------------------------
    # 7. Walk-forward validation (if required)
    # ------------------------------------------------------------------
    if config.require_walkforward:
        wf_results = candidate_report.get("walkforward", candidate_report.get("walkforward_result"))
        if wf_results is None:
            wf_passed = False
            reasons.append("Walk-forward results required but not provided")
            gates["walkforward"] = _make_gate("walkforward", False, 0.0, 1.0)
        else:
            # Check that out-of-sample is profitable
            oos_pnl = 0.0
            if isinstance(wf_results, dict):
                oos_pnl = float(wf_results.get("oos_pnl", wf_results.get("oos_total_pnl", 0.0)))
                # Also accept a list of fold results
                folds = wf_results.get("folds", [])
                if folds and oos_pnl == 0.0:
                    oos_pnl = sum(float(f.get("oos_pnl", f.get("pnl", 0.0))) for f in folds)
            wf_passed = oos_pnl > 0.0
            gates["walkforward"] = _make_gate("walkforward", wf_passed, oos_pnl, 0.0)
            if not wf_passed:
                reasons.append(f"Walk-forward OOS PnL {oos_pnl:.4f} is not profitable")

    # ------------------------------------------------------------------
    # 8. Robustness (if required)
    # ------------------------------------------------------------------
    if config.require_robustness:
        robustness = candidate_report.get("robustness", candidate_report.get("robustness_result"))
        if robustness is None:
            rob_passed = False
            reasons.append("Robustness results required but not provided")
            gates["robustness"] = _make_gate("robustness", False, 0.0, config.robustness_pass_rate)
        else:
            # Robustness result should contain a pass_rate or passed field
            if isinstance(robustness, dict):
                pass_rate = float(
                    robustness.get(
                        "pass_rate",
                        robustness.get("survival_rate", 0.0),
                    )
                )
                # If a boolean 'passed' field exists, use it as a fallback
                if pass_rate == 0.0 and robustness.get("passed", False):
                    pass_rate = 1.0
            else:
                pass_rate = 0.0

            rob_passed = pass_rate >= config.robustness_pass_rate
            gates["robustness"] = _make_gate(
                "robustness",
                rob_passed,
                pass_rate,
                config.robustness_pass_rate,
            )
            if not rob_passed:
                reasons.append(
                    f"Robustness pass rate {pass_rate:.2%} below "
                    f"threshold {config.robustness_pass_rate:.2%}"
                )

    # ------------------------------------------------------------------
    # Overall verdict
    # ------------------------------------------------------------------
    all_passed = all(g["passed"] for g in gates.values())

    # Recommendation logic:
    # - "promote" if all gates pass
    # - "reject" if critical gates (utility, drawdown, sharpe) fail
    # - "review" for borderline cases (only minor gates failed)
    if all_passed:
        recommendation = "promote"
    else:
        critical_gates = {"utility_improvement", "max_drawdown", "sharpe"}
        critical_failures = [
            name for name in critical_gates if name in gates and not gates[name]["passed"]
        ]
        if critical_failures:
            recommendation = "reject"
        else:
            recommendation = "review"

    logger.info(
        "gates.evaluate.done",
        passed=all_passed,
        recommendation=recommendation,
        failed_gates=[k for k, v in gates.items() if not v["passed"]],
    )

    return {
        "passed": all_passed,
        "reasons": reasons,
        "gates": gates,
        "delta": delta,
        "recommendation": recommendation,
    }
