"""Ensemble strategy selector.

Routes the current market context to the set of strategies that are
applicable for the detected regime, collects their signals, and returns
the highest-confidence non-flat signal.
"""

from __future__ import annotations

from autotrader.hl.types import Signal
from autotrader.strategies.base import BaseStrategy, MarketContext, flat_signal
from autotrader.strategies.crash_meanrev import CrashMeanRevStrategy
from autotrader.strategies.funding_extremes import FundingExtremesStrategy
from autotrader.strategies.range_meanrev import RangeMeanRevStrategy
from autotrader.strategies.squeeze_breakout import SqueezeBreakoutStrategy
from autotrader.strategies.trend_breakout import TrendBreakoutStrategy
from autotrader.strategies.vol_expansion import VolExpansionStrategy


def _build_default_strategies() -> list[BaseStrategy]:
    """Instantiate the six built-in strategies with default configs."""
    return [
        TrendBreakoutStrategy(),
        RangeMeanRevStrategy(),
        VolExpansionStrategy(),
        FundingExtremesStrategy(),
        SqueezeBreakoutStrategy(),
        CrashMeanRevStrategy(),
    ]


class EnsembleStrategy:
    """Select and combine signals from multiple sub-strategies.

    Parameters
    ----------
    strategies : list[BaseStrategy] | None
        Concrete strategy instances.  If ``None``, the four default
        strategies are instantiated automatically.
    config : dict | None
        Optional configuration.  Supports a ``"regime_weights"`` key mapping
        ``{regime: {strategy_name: weight}}``.  When not provided every
        applicable strategy has equal weight.
    """

    name: str = "ensemble"

    def __init__(
        self,
        strategies: list[BaseStrategy] | None = None,
        config: dict | None = None,
    ) -> None:
        self.strategies: list[BaseStrategy] = (
            strategies if strategies is not None else _build_default_strategies()
        )

        cfg = config or {}
        # regime_weights: { "TREND": { "trend_breakout": 1.0, ... }, ... }
        self.regime_weights: dict[str, dict[str, float]] = cfg.get("regime_weights", {})

    # ------------------------------------------------------------------

    def select_strategies(self, regime: str) -> list[BaseStrategy]:
        """Return the subset of strategies applicable to *regime*."""
        return [s for s in self.strategies if regime in s.applicable_regimes()]

    # ------------------------------------------------------------------

    def _get_weight(self, regime: str, strategy_name: str) -> float:
        """Look up the weight for a strategy in a given regime.

        Falls back to 1.0 (equal weight) if no explicit mapping exists.
        """
        regime_map = self.regime_weights.get(regime, {})
        return float(regime_map.get(strategy_name, 1.0))

    # ------------------------------------------------------------------

    def compute_signal(self, ctx: MarketContext) -> Signal:
        """Run applicable strategies and return the best signal.

        Selection logic:
        1. Collect strategies applicable to the current regime.
        2. Compute each strategy's signal.
        3. Discard flat signals and those that fail ``invariants_ok``.
        4. If one non-flat signal remains, return it.
        5. If multiple remain, pick the one with the highest
           *weight-adjusted* confidence.
        6. If none remain, return a flat signal.
        """
        applicable = self.select_strategies(ctx.regime)

        if not applicable:
            return flat_signal()

        candidates: list[tuple[Signal, float]] = []  # (signal, weighted_conf)

        for strategy in applicable:
            try:
                sig = strategy.compute_signal(ctx)
            except Exception:
                # Strategy raised -- skip gracefully rather than crashing
                # the entire ensemble.
                continue

            if sig.side == "flat":
                continue

            if not strategy.invariants_ok(sig):
                continue

            weight = self._get_weight(ctx.regime, strategy.name)
            weighted_confidence = sig.confidence * weight
            candidates.append((sig, weighted_confidence))

        if not candidates:
            return flat_signal()

        # Pick highest weighted confidence
        best_signal, _best_wc = max(candidates, key=lambda t: t[1])
        return best_signal


# ------------------------------------------------------------------
# Module-level convenience function (backwards-compatible)
# ------------------------------------------------------------------


def combine(signals: list[dict], weights: list[float]) -> dict:
    """Weighted signal combination (dict-based, backwards-compatible).

    Each element of *signals* is a dict with at least ``"side"`` and
    ``"confidence"`` keys.  *weights* is a parallel list of floats.

    Returns the signal dict whose weighted confidence is highest.
    If all signals are flat or the lists are empty, returns a flat dict.

    Parameters
    ----------
    signals : list[dict]
        List of signal dictionaries.
    weights : list[float]
        Parallel list of weight multipliers.

    Returns
    -------
    dict
        The best signal by weighted confidence, or a flat signal dict.
    """
    flat: dict = {
        "side": "flat",
        "entry": None,
        "stop": None,
        "take_profit": None,
        "confidence": 0.0,
        "metadata": {},
    }

    if not signals or not weights:
        return flat

    best_signal: dict = flat
    best_weighted: float = -1.0

    for sig, w in zip(signals, weights):
        side = sig.get("side", "flat")
        if side == "flat":
            continue
        conf = float(sig.get("confidence", 0.0))
        weighted = conf * float(w)
        if weighted > best_weighted:
            best_weighted = weighted
            best_signal = sig

    return best_signal
