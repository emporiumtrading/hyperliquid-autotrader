"""Range Mean Reversion Strategy -- designed for the RANGE regime.

Buys at the lower Bollinger Band when RSI is oversold, sells at the upper
band when RSI is overbought.  Only active when ADX is low (no trend).
Targets the middle band (mean reversion) with tight ATR-based stops.
"""

from __future__ import annotations

from autotrader.hl.types import Signal
from autotrader.strategies.base import BaseStrategy, MarketContext, flat_signal

_DEFAULT_CONFIG: dict = {
    "bb_period": 20,
    "bb_std": 2.0,
    "rsi_period": 14,
    "rsi_oversold": 30.0,
    "rsi_overbought": 70.0,
    "atr_multiplier_stop": 1.5,
    "max_adx": 25.0,
    "min_confidence": 0.3,
}


class RangeMeanRevStrategy(BaseStrategy):
    """Mean-revert at Bollinger Band extremes in ranging markets.

    Parameters
    ----------
    config : dict | None
        Override any default parameter by key name.
    """

    name: str = "range_meanrev"

    def __init__(self, config: dict | None = None) -> None:
        merged = {**_DEFAULT_CONFIG}
        if config:
            merged.update(config)

        self.bb_period: int = int(merged["bb_period"])
        self.bb_std: float = float(merged["bb_std"])
        self.rsi_period: int = int(merged["rsi_period"])
        self.rsi_oversold: float = float(merged["rsi_oversold"])
        self.rsi_overbought: float = float(merged["rsi_overbought"])
        self.atr_multiplier_stop: float = float(merged["atr_multiplier_stop"])
        self.max_adx: float = float(merged["max_adx"])
        self.min_confidence: float = float(merged["min_confidence"])

    # ------------------------------------------------------------------

    def applicable_regimes(self) -> list[str]:
        return ["RANGE"]

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
        """Generate a mean-reversion signal at BB extremes.

        Long: close near/below lower BB + RSI oversold -> target middle BB.
        Short: close near/above upper BB + RSI overbought -> target middle BB.
        """
        candles = ctx.candles
        features = ctx.features

        if candles.empty:
            return flat_signal()

        close = float(candles["close"].iloc[-1])

        # Required features
        adx_val = self._safe_feat(features, "adx")
        rsi_val = self._safe_feat(features, "rsi")
        atr_val = self._safe_feat(features, "atr")
        bb_upper = self._safe_feat(features, "bb_upper")
        bb_lower = self._safe_feat(features, "bb_lower")
        bb_middle = self._safe_feat(features, "bb_middle")

        if any(v is None for v in (adx_val, rsi_val, atr_val, bb_upper, bb_lower, bb_middle)):
            return flat_signal()

        # Type narrowing for mypy (already checked None above)
        assert adx_val is not None
        assert rsi_val is not None
        assert atr_val is not None
        assert bb_upper is not None
        assert bb_lower is not None
        assert bb_middle is not None

        if atr_val <= 0.0:
            return flat_signal()

        # Only trade low-ADX (ranging) environments
        if adx_val > self.max_adx:
            return flat_signal()

        # Bollinger Band width (used for proximity calculations)
        bb_width = bb_upper - bb_lower
        if bb_width <= 0.0:
            return flat_signal()

        # ------------------------------------------------------------------
        # Determine direction
        # ------------------------------------------------------------------
        side: str = "flat"
        confidence: float = 0.0

        # How far price is from each band, as a fraction of BB width
        dist_to_lower = (close - bb_lower) / bb_width
        dist_to_upper = (bb_upper - close) / bb_width

        # LONG: close near or below lower band + RSI oversold
        if dist_to_lower <= 0.15 and rsi_val < self.rsi_oversold:
            side = "long"
            # Confidence based on distance from band + RSI extremity
            band_proximity = max(1.0 - dist_to_lower / 0.15, 0.0) * 0.40
            rsi_extremity = max((self.rsi_oversold - rsi_val) / self.rsi_oversold, 0.0) * 0.35
            adx_component = max((self.max_adx - adx_val) / self.max_adx, 0.0) * 0.15
            base = 0.10
            confidence = base + band_proximity + rsi_extremity + adx_component

        # SHORT: close near or above upper band + RSI overbought
        elif dist_to_upper <= 0.15 and rsi_val > self.rsi_overbought:
            side = "short"
            band_proximity = max(1.0 - dist_to_upper / 0.15, 0.0) * 0.40
            rsi_extremity = (
                max((rsi_val - self.rsi_overbought) / (100.0 - self.rsi_overbought), 0.0) * 0.35
            )
            adx_component = max((self.max_adx - adx_val) / self.max_adx, 0.0) * 0.15
            base = 0.10
            confidence = base + band_proximity + rsi_extremity + adx_component

        if side == "flat":
            return flat_signal()

        confidence = max(min(confidence, 1.0), 0.0)
        if confidence < self.min_confidence:
            return flat_signal()

        # ------------------------------------------------------------------
        # Price levels -- target the middle band (mean reversion)
        # ------------------------------------------------------------------
        entry = close

        if side == "long":
            stop = entry - self.atr_multiplier_stop * atr_val
            take_profit = bb_middle  # revert to mean
            # Ensure TP is actually above entry (middle BB should be above
            # lower BB, but guard against edge cases).
            if take_profit <= entry:
                take_profit = entry + atr_val
        else:
            stop = entry + self.atr_multiplier_stop * atr_val
            take_profit = bb_middle
            if take_profit >= entry:
                take_profit = entry - atr_val

        signal = Signal(
            side=side,
            entry=round(entry, 8),
            stop=round(stop, 8),
            take_profit=round(take_profit, 8),
            confidence=round(confidence, 4),
            metadata={
                "strategy": self.name,
                "adx": adx_val,
                "rsi": rsi_val,
                "atr": atr_val,
                "bb_upper": bb_upper,
                "bb_lower": bb_lower,
                "bb_middle": bb_middle,
                "dist_to_lower": round(dist_to_lower, 4),
                "dist_to_upper": round(dist_to_upper, 4),
                "regime": ctx.regime,
            },
        )

        if not self.invariants_ok(signal):
            return flat_signal()

        return signal
