"""Approval workflow for strategy promotion.

Coordinates the full promote-or-reject pipeline:

1. Load the current baseline from the :class:`BaselineRegistry`.
2. Run governance gate evaluation via :func:`evaluate_candidate`.
3. If approved, create a new baseline entry and persist it.
4. Support emergency rollback to a previous version.
"""

from __future__ import annotations

import structlog

from autotrader.governance.gates import GateConfig, evaluate_candidate
from autotrader.governance.registry import BaselineRegistry
from autotrader.utils.time import now_ms

logger = structlog.get_logger(__name__)


class ApprovalWorkflow:
    """Orchestrates candidate evaluation and baseline promotion.

    Parameters
    ----------
    registry:
        The :class:`BaselineRegistry` used for persistence.
    gate_config:
        Optional :class:`GateConfig` overrides.  Falls back to defaults
        if ``None``.
    """

    def __init__(
        self,
        registry: BaselineRegistry,
        gate_config: GateConfig | None = None,
    ) -> None:
        self.registry = registry
        self.gate_config = gate_config or GateConfig()

    # ------------------------------------------------------------------
    # Evaluate & promote
    # ------------------------------------------------------------------

    def evaluate_and_promote(
        self,
        candidate_report: dict,
        walkforward_result: dict | None = None,
        robustness_result: dict | None = None,
    ) -> dict:
        """Evaluate a candidate and promote it if it passes all gates.

        The method enriches ``candidate_report`` with the optional
        walk-forward and robustness results before evaluation so that the
        gate logic can inspect them.

        Parameters
        ----------
        candidate_report:
            Metrics / report dictionary for the candidate strategy run.
        walkforward_result:
            Walk-forward test results (optional, injected into the report
            under the ``"walkforward"`` key if not already present).
        robustness_result:
            Robustness test results (optional, injected under the
            ``"robustness"`` key if not already present).

        Returns
        -------
        dict
            Result with keys:

            * ``promoted`` (bool) -- whether the candidate was promoted.
            * ``baseline_version`` (int | None) -- new version if promoted.
            * ``history_path`` (str | None) -- path to the history snapshot.
            * ``gate_result`` (dict) -- full output of :func:`evaluate_candidate`.
            * ``reasons`` (list[str]) -- failure reasons (empty on success).
            * ``timestamp`` (int) -- evaluation time in ms.
        """
        # Inject optional results into the report so gates can see them
        enriched = dict(candidate_report)
        if walkforward_result is not None and "walkforward" not in enriched:
            enriched["walkforward"] = walkforward_result
        if robustness_result is not None and "robustness" not in enriched:
            enriched["robustness"] = robustness_result

        # Load current baseline (may be None for first-time promotion)
        current_baseline = self.registry.load_current()
        baseline_report: dict | None = None
        if current_baseline is not None:
            baseline_report = current_baseline.get("metrics_summary", current_baseline)

        # Run gate evaluation
        gate_result = evaluate_candidate(enriched, baseline_report, self.gate_config)

        timestamp = now_ms()

        if gate_result["passed"]:
            # Build new baseline entry
            strategy_name = enriched.get(
                "strategy_name",
                enriched.get("strategy", "unknown"),
            )
            strategy_config = enriched.get("strategy_config", enriched.get("config", {}))
            metrics_summary = _extract_metrics_summary(enriched)
            dataset_hash = enriched.get("dataset_hash", "")
            git_commit = enriched.get("git_commit", "")
            run_id = enriched.get("run_id", "")

            new_baseline = self.registry.create_baseline(
                strategy_name=strategy_name,
                strategy_config=strategy_config,
                metrics=metrics_summary,
                dataset_hash=dataset_hash,
                git_commit=git_commit,
                run_id=run_id,
            )
            history_path = self.registry.save_current(new_baseline)

            logger.info(
                "approval.promoted",
                version=new_baseline["version"],
                strategy_name=strategy_name,
                history_path=history_path,
            )

            return {
                "promoted": True,
                "baseline_version": new_baseline["version"],
                "history_path": history_path,
                "gate_result": gate_result,
                "reasons": [],
                "timestamp": timestamp,
            }
        else:
            logger.info(
                "approval.rejected",
                reasons=gate_result["reasons"],
                recommendation=gate_result["recommendation"],
            )

            return {
                "promoted": False,
                "baseline_version": None,
                "history_path": None,
                "gate_result": gate_result,
                "reasons": gate_result["reasons"],
                "timestamp": timestamp,
            }

    # ------------------------------------------------------------------
    # Emergency rollback
    # ------------------------------------------------------------------

    def emergency_rollback(self, version: int | None = None) -> dict:
        """Roll back to a previous baseline version.

        Parameters
        ----------
        version:
            Specific version to restore.  If ``None``, the most recent
            previous version is used.

        Returns
        -------
        dict
            Result with keys:

            * ``rolled_back`` (bool) -- whether rollback succeeded.
            * ``old_version`` (int | None) -- the version that was active.
            * ``new_version`` (int | None) -- the restored version.
            * ``baseline`` (dict | None) -- the restored baseline dict.
            * ``error`` (str | None) -- error message on failure.
        """
        old_baseline = self.registry.load_current()
        old_version = old_baseline.get("version") if old_baseline else None

        try:
            restored = self.registry.rollback(version=version)
        except ValueError as exc:
            logger.error(
                "approval.rollback.failed",
                error=str(exc),
                requested_version=version,
            )
            return {
                "rolled_back": False,
                "old_version": old_version,
                "new_version": None,
                "baseline": None,
                "error": str(exc),
            }

        new_version = restored.get("version")
        logger.info(
            "approval.rollback.ok",
            old_version=old_version,
            new_version=new_version,
        )

        return {
            "rolled_back": True,
            "old_version": old_version,
            "new_version": new_version,
            "baseline": restored,
            "error": None,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_metrics_summary(report: dict) -> dict:
    """Pull a flat metrics dictionary from a candidate report.

    Looks for ``metrics_summary``, ``metrics``, or falls back to picking
    known metric keys from the top level.
    """
    if "metrics_summary" in report and isinstance(report["metrics_summary"], dict):
        return dict(report["metrics_summary"])
    if "metrics" in report and isinstance(report["metrics"], dict):
        return dict(report["metrics"])

    # Fallback: extract known metric keys from the top level
    known_keys = {
        "utility",
        "total_pnl",
        "net_pnl",
        "sharpe",
        "sharpe_ratio",
        "max_drawdown",
        "max_drawdown_pct",
        "cvar_95",
        "cvar95",
        "profit_factor",
        "win_rate",
        "total_trades",
        "num_trades",
    }
    return {k: v for k, v in report.items() if k in known_keys}
