"""Performance drift detection.

Monitors the divergence between expected (backtested / predicted) behaviour
and realised live performance across three dimensions:

1. **PnL drift** -- realised PnL significantly worse than expected (z-score).
2. **Slippage drift** -- realised slippage exceeds a multiple of expected.
3. **Regime mismatch** -- predicted regime does not match observed outcome.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

import structlog

from autotrader.utils.time import MS_PER_DAY, now_ms

logger = structlog.get_logger(__name__)


class DriftDetector:
    """Accumulates live observations and checks for performance drift.

    Parameters
    ----------
    config:
        Optional configuration dictionary.  Recognised keys:

        * ``lookback_days`` (int) -- observation window in days (default 7).
        * ``slippage_drift_threshold`` (float) -- multiplier of expected
          slippage above which slippage drift is flagged (default 2.0).
        * ``pnl_drift_threshold`` (float) -- z-score below which PnL
          drift is flagged.  Negative means "worse than expected" (default
          -2.0).
        * ``regime_mismatch_threshold`` (float) -- mismatch rate above
          which regime accuracy drift is flagged (default 0.3 = 30%).
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.lookback_days: int = int(cfg.get("lookback_days", 7))
        self.slippage_drift_threshold: float = float(cfg.get("slippage_drift_threshold", 2.0))
        self.pnl_drift_threshold: float = float(cfg.get("pnl_drift_threshold", -2.0))
        self.regime_mismatch_threshold: float = float(cfg.get("regime_mismatch_threshold", 0.3))
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def add_observation(
        self,
        timestamp_ms: int,
        expected_pnl: float,
        realized_pnl: float,
        expected_slippage: float,
        realized_slippage: float,
        predicted_regime: str,
        actual_outcome: str,
    ) -> None:
        """Record a single observation for later drift analysis.

        Parameters
        ----------
        timestamp_ms:
            Observation timestamp in milliseconds since epoch.
        expected_pnl:
            The PnL predicted by the model / backtest for this period.
        realized_pnl:
            The actual PnL observed in live trading.
        expected_slippage:
            The slippage estimate from the cost model.
        realized_slippage:
            The actual slippage incurred.
        predicted_regime:
            The regime label predicted by the classifier.
        actual_outcome:
            The regime label that was actually observed.
        """
        self.history.append(
            {
                "timestamp_ms": timestamp_ms,
                "expected_pnl": expected_pnl,
                "realized_pnl": realized_pnl,
                "expected_slippage": expected_slippage,
                "realized_slippage": realized_slippage,
                "predicted_regime": predicted_regime,
                "actual_outcome": actual_outcome,
            }
        )

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _recent_observations(self) -> list[dict]:
        """Return observations within the lookback window."""
        cutoff_ms = now_ms() - self.lookback_days * MS_PER_DAY
        return [o for o in self.history if o["timestamp_ms"] >= cutoff_ms]

    def check_drift(self) -> dict:
        """Analyse recent observations for performance drift.

        Returns
        -------
        dict
            Result with keys:

            * ``drifting`` (bool) -- whether any drift signal fired.
            * ``signals`` (list[str]) -- human-readable descriptions.
            * ``severity`` (str) -- ``"none"``, ``"warning"``, or
              ``"critical"``.
            * ``recommended_action`` (str) -- suggested response.
            * ``details`` (dict) -- per-check numeric details.
        """
        recent = self._recent_observations()
        signals: list[str] = []
        details: dict[str, Any] = {}

        if not recent:
            return {
                "drifting": False,
                "signals": [],
                "severity": "none",
                "recommended_action": "continue",
                "details": {"observation_count": 0},
            }

        details["observation_count"] = len(recent)

        # ---- PnL drift (z-score of realised vs expected) ----
        pnl_diffs = [o["realized_pnl"] - o["expected_pnl"] for o in recent]
        pnl_z = _z_score(pnl_diffs)
        details["pnl_z_score"] = pnl_z
        details["pnl_mean_diff"] = statistics.mean(pnl_diffs) if pnl_diffs else 0.0

        pnl_drifting = pnl_z <= self.pnl_drift_threshold
        if pnl_drifting:
            signals.append(
                f"PnL drift detected: z-score {pnl_z:.2f} "
                f"(threshold {self.pnl_drift_threshold:.2f})"
            )

        # ---- Slippage drift ----
        slippage_ratios: list[float] = []
        for o in recent:
            expected = abs(o["expected_slippage"])
            realized = abs(o["realized_slippage"])
            if expected > 0:
                slippage_ratios.append(realized / expected)
            elif realized > 0:
                # Expected zero but got positive slippage -- treat as infinite
                slippage_ratios.append(float("inf"))
            # If both zero, no slippage to compare

        if slippage_ratios:
            avg_slippage_ratio = (
                statistics.mean([r for r in slippage_ratios if math.isfinite(r)])
                if any(math.isfinite(r) for r in slippage_ratios)
                else float("inf")
            )
        else:
            avg_slippage_ratio = 0.0

        details["avg_slippage_ratio"] = avg_slippage_ratio

        slippage_drifting = avg_slippage_ratio > self.slippage_drift_threshold
        if slippage_drifting:
            signals.append(
                f"Slippage drift detected: avg ratio {avg_slippage_ratio:.2f}x "
                f"(threshold {self.slippage_drift_threshold:.2f}x)"
            )

        # ---- Regime accuracy ----
        total_regime_checks = len(recent)
        mismatches = sum(1 for o in recent if o["predicted_regime"] != o["actual_outcome"])
        mismatch_rate = mismatches / total_regime_checks if total_regime_checks else 0.0
        details["regime_mismatch_rate"] = mismatch_rate
        details["regime_mismatches"] = mismatches
        details["regime_total"] = total_regime_checks

        regime_drifting = mismatch_rate > self.regime_mismatch_threshold
        if regime_drifting:
            signals.append(
                f"Regime mismatch drift: {mismatch_rate:.1%} mismatch rate "
                f"(threshold {self.regime_mismatch_threshold:.1%})"
            )

        # ---- Severity and recommendation ----
        drifting = bool(signals)
        num_signals = len(signals)

        if not drifting:
            severity = "none"
            recommended_action = "continue"
        elif num_signals >= 2 or pnl_drifting:
            severity = "critical"
            recommended_action = "halt_trading_and_review"
        else:
            severity = "warning"
            recommended_action = "reduce_exposure"

        logger.info(
            "drift.check",
            drifting=drifting,
            severity=severity,
            signals=signals,
        )

        return {
            "drifting": drifting,
            "signals": signals,
            "severity": severity,
            "recommended_action": recommended_action,
            "details": details,
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all recorded observations."""
        self.history.clear()
        logger.info("drift.reset")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _z_score(values: list[float]) -> float:
    """Compute the z-score of the mean of *values*.

    Returns 0.0 when the sample is too small.  When the standard deviation
    is zero (all values identical), returns ``-inf`` if the mean is
    negative, ``+inf`` if positive, or ``0.0`` if the mean is exactly zero.
    A consistently negative mean with no variance is the strongest possible
    signal of underperformance.
    """
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    if stdev == 0.0:
        # All values identical -- if consistently negative, treat as
        # infinitely significant negative drift.
        if mean < 0:
            return float("-inf")
        elif mean > 0:
            return float("inf")
        return 0.0
    # z-score of the sample mean
    n = len(values)
    return mean / (stdev / math.sqrt(n))
