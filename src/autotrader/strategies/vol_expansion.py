"""Volatility Expansion Strategy -- designed for the VOLATILE_BREAKOUT regime.

Detects a transition from compressed to expanding volatility (Bollinger Band
squeeze breakout) confirmed by above-average volume, then enters in the
breakout direction with wide ATR-based stops and targets.
"""

from __future__ import annotations

from autotrader.hl.types import Signal
from autotrader.strategies.base import BaseStrategy, MarketContext, flat_signal

_DEFAULT_CONFIG: dict = {
    "bb_squeeze_percentile": 0.2,
    "bb_expansion_percentile": 0.7,
    "volume_multiple": 1.5,
    "atr_multiplier_stop": 2.5,
    "atr_multiplier_tp": 4.0,
    "min_confidence": 0.3,
}


class VolExpansionStrategy(BaseStrategy):
    """Trade volatility expansion breakouts after a squeeze.

    Parameters
    ----------
    config : dict | None
        Override any default parameter by key name.
    """

    name: str = "vol_expansion"

    def __init__(self, config: dict | None = None) -> None:
        merged = {**_DEFAULT_CONFIG}
        if config:
            merged.update(config)

        self.bb_squeeze_percentile: float = float(merged["bb_squeeze_percentile"])
        self.bb_expansion_percentile: float = float(merged["bb_expansion_percentile"])
        self.volume_multiple: float = float(merged["volume_multiple"])
        self.atr_multiplier_stop: float = float(merged["atr_multiplier_stop"])
        self.atr_multiplier_tp: float = float(merged["atr_multiplier_tp"])
        self.min_confidence: float = float(merged["min_confidence"])

    # ------------------------------------------------------------------

    def applicable_regimes(self) -> list[str]:
        return ["VOLATILE_BREAKOUT"]

    # ------------------------------------------------------------------

    @staticmethod
    def _safe_feat(features: dict, key: str) -> float | None:
        val = features.get(key)
        if val is None:
            return None
        try:
            fval = float(val)
        except (TypeError, ValueError):
            return None
        if fval != fval:
            return None
        return fval

    # ------------------------------------------------------------------

    def compute_signal(self, ctx: MarketContext) -> Signal:
        """Generate a volatility-expansion breakout signal.

        Conditions:
        1. Prior compression: recent BB width percentile was < squeeze_percentile.
        2. Current expansion: BB width percentile > expansion_percentile.
        3. Volume confirmation: current volume > volume_multiple * volume_sma.
        4. Direction determined by price vs middle BB + MA slope.
        """
        candles = ctx.candles
        features = ctx.features

        if candles.empty or len(candles) < 2:
            return flat_signal()

        close = float(candles["close"].iloc[-1])

        # ------------------------------------------------------------------
        # Required features
        # ------------------------------------------------------------------
        bb_width_pct = self._safe_feat(features, "bb_width_pct")
        atr_val = self._safe_feat(features, "atr")
        bb_middle = self._safe_feat(features, "bb_middle")
        ma_slope_val = self._safe_feat(features, "ma_slope")
        volume_sma_val = self._safe_feat(features, "volume_sma")

        if any(v is None for v in (bb_width_pct, atr_val, bb_middle)):
            return flat_signal()

        assert bb_width_pct is not None
        assert atr_val is not None
        assert bb_middle is not None

        if atr_val <= 0.0:
            return flat_signal()

        # ------------------------------------------------------------------
        # Check for prior compression
        # ------------------------------------------------------------------
        # Look for a recent bar (within last 5 candles) where BB width pct
        # was below the squeeze threshold.  We check the feature history if
        # available, or fall back to checking the prev_bb_width_pct feature.
        had_prior_squeeze = False
        prev_bb_width_pct = self._safe_feat(features, "prev_bb_width_pct")

        if prev_bb_width_pct is not None and prev_bb_width_pct < self.bb_squeeze_percentile:
            had_prior_squeeze = True

        # Also check bb_width_pct_history (list of recent values) if provided
        pct_history = features.get("bb_width_pct_history")
        if isinstance(pct_history, (list, tuple)):
            for val in pct_history:
                try:
                    if float(val) < self.bb_squeeze_percentile:
                        had_prior_squeeze = True
                        break
                except (TypeError, ValueError):
                    continue

        # If we cannot confirm prior squeeze from features, use a heuristic:
        # if current expansion is very strong (>0.85) we still consider it
        # valid, as we may have just missed the squeeze in the feature window.
        if not had_prior_squeeze and bb_width_pct > 0.85:
            had_prior_squeeze = True

        # ------------------------------------------------------------------
        # Current expansion check
        # ------------------------------------------------------------------
        if not had_prior_squeeze:
            return flat_signal()

        if bb_width_pct < self.bb_expansion_percentile:
            return flat_signal()

        # ------------------------------------------------------------------
        # Volume confirmation
        # ------------------------------------------------------------------
        volume_confirmed = True  # default if volume data unavailable
        if volume_sma_val is not None and volume_sma_val > 0.0:
            current_volume = float(candles["volume"].iloc[-1])
            if current_volume < self.volume_multiple * volume_sma_val:
                volume_confirmed = False

        # ------------------------------------------------------------------
        # Direction: price vs middle BB + MA slope
        # ------------------------------------------------------------------
        price_above_mid = close > bb_middle
        slope_positive = ma_slope_val is not None and ma_slope_val > 0.0
        slope_negative = ma_slope_val is not None and ma_slope_val < 0.0

        side: str = "flat"

        if price_above_mid:
            side = "long"
        elif not price_above_mid:
            side = "short"

        # Use MA slope as a tiebreaker / confirmation
        if side == "long" and slope_negative:
            # Conflicting signals -- reduce confidence later
            pass
        elif side == "short" and slope_positive:
            pass

        if side == "flat":
            return flat_signal()

        # ------------------------------------------------------------------
        # Confidence
        # ------------------------------------------------------------------
        # Components: expansion strength, volume confirmation, slope alignment
        expansion_strength = (
            min(
                (bb_width_pct - self.bb_expansion_percentile)
                / (1.0 - self.bb_expansion_percentile + 1e-9),
                1.0,
            )
            * 0.35
        )

        vol_bonus = 0.25 if volume_confirmed else 0.0

        slope_alignment = 0.0
        if ma_slope_val is not None:
            if (side == "long" and ma_slope_val > 0.0) or (side == "short" and ma_slope_val < 0.0):
                slope_alignment = 0.20
            elif (side == "long" and ma_slope_val < 0.0) or (
                side == "short" and ma_slope_val > 0.0
            ):
                slope_alignment = -0.10  # penalty for conflicting slope

        base = 0.15
        confidence = base + expansion_strength + vol_bonus + slope_alignment
        confidence = max(min(confidence, 1.0), 0.0)

        if confidence < self.min_confidence:
            return flat_signal()

        # ------------------------------------------------------------------
        # Price levels -- wide stops/targets for vol expansion
        # ------------------------------------------------------------------
        entry = close

        if side == "long":
            stop = entry - self.atr_multiplier_stop * atr_val
            take_profit = entry + self.atr_multiplier_tp * atr_val
        else:
            stop = entry + self.atr_multiplier_stop * atr_val
            take_profit = entry - self.atr_multiplier_tp * atr_val

        signal = Signal(
            side=side,
            entry=round(entry, 8),
            stop=round(stop, 8),
            take_profit=round(take_profit, 8),
            confidence=round(confidence, 4),
            metadata={
                "strategy": self.name,
                "bb_width_pct": bb_width_pct,
                "atr": atr_val,
                "bb_middle": bb_middle,
                "ma_slope": ma_slope_val,
                "volume_confirmed": volume_confirmed,
                "had_prior_squeeze": had_prior_squeeze,
                "regime": ctx.regime,
            },
        )

        if not self.invariants_ok(signal):
            return flat_signal()

        return signal
