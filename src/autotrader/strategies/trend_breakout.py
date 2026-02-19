"""Trend Breakout Strategy -- designed for the TREND regime.

Looks for pullbacks in an established trend (high ADX) or breakouts above /
below a lookback-period price channel, then enters in the trend direction
with ATR-based stops and take-profit levels.
"""

from __future__ import annotations

from autotrader.hl.types import Signal
from autotrader.strategies.base import BaseStrategy, MarketContext, flat_signal

_DEFAULT_CONFIG: dict = {
    "lookback": 20,
    "atr_multiplier_stop": 2.0,
    "atr_multiplier_tp": 3.0,
    "min_adx": 25.0,
    "trend_ma_period": 50,
    "pullback_rsi_long": 40.0,
    "pullback_rsi_short": 60.0,
    "min_confidence": 0.3,
}


class TrendBreakoutStrategy(BaseStrategy):
    """Enter with the trend on pullbacks or channel breakouts.

    Parameters
    ----------
    config : dict | None
        Override any default parameter by key name.
    """

    name: str = "trend_breakout"

    def __init__(self, config: dict | None = None) -> None:
        merged = {**_DEFAULT_CONFIG}
        if config:
            merged.update(config)

        self.lookback: int = int(merged["lookback"])
        self.atr_multiplier_stop: float = float(merged["atr_multiplier_stop"])
        self.atr_multiplier_tp: float = float(merged["atr_multiplier_tp"])
        self.min_adx: float = float(merged["min_adx"])
        self.trend_ma_period: int = int(merged["trend_ma_period"])
        self.pullback_rsi_long: float = float(merged["pullback_rsi_long"])
        self.pullback_rsi_short: float = float(merged["pullback_rsi_short"])
        self.min_confidence: float = float(merged["min_confidence"])

    # ------------------------------------------------------------------

    def applicable_regimes(self) -> list[str]:
        return ["TREND"]

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
        """Generate a trend-following signal.

        Long setup:
            * Price is above the trend MA (uptrend confirmed).
            * RSI has pulled back below ``pullback_rsi_long`` **or**
              price is breaking above the ``lookback``-period high.
        Short setup:
            * Mirror conditions.
        """
        candles = ctx.candles
        features = ctx.features

        if candles.empty or len(candles) < self.lookback:
            return flat_signal()

        # Last candle values
        close = float(candles["close"].iloc[-1])
        high = float(candles["high"].iloc[-1])
        low = float(candles["low"].iloc[-1])

        # Features
        adx_val = self._safe_feat(features, "adx")
        rsi_val = self._safe_feat(features, "rsi")
        atr_val = self._safe_feat(features, "atr")
        ma_key = f"sma_{self.trend_ma_period}"
        ma_val = self._safe_feat(features, ma_key)

        # Fallback: also try generic "sma" key
        if ma_val is None:
            ma_val = self._safe_feat(features, "sma")

        # Require core features
        if adx_val is None or rsi_val is None or atr_val is None or ma_val is None:
            return flat_signal()

        if atr_val <= 0.0:
            return flat_signal()

        # ADX must show trend
        if adx_val < self.min_adx:
            return flat_signal()

        # Channel highs/lows over lookback
        lookback_high = float(candles["high"].iloc[-self.lookback :].max())
        lookback_low = float(candles["low"].iloc[-self.lookback :].min())

        # ------------------------------------------------------------------
        # Determine direction
        # ------------------------------------------------------------------
        side: str = "flat"
        confidence: float = 0.0

        # LONG conditions
        is_uptrend = close > ma_val
        pullback_long = rsi_val < self.pullback_rsi_long
        breakout_long = high >= lookback_high

        # SHORT conditions
        is_downtrend = close < ma_val
        pullback_short = rsi_val > self.pullback_rsi_short
        breakout_short = low <= lookback_low

        if is_uptrend and (pullback_long or breakout_long):
            side = "long"
            # Confidence components
            adx_component = min((adx_val - self.min_adx) / 25.0, 1.0) * 0.40
            trend_alignment = min(abs(close - ma_val) / (atr_val * 2.0), 1.0) * 0.30
            trigger_bonus = 0.20 if breakout_long else 0.10
            base = 0.10
            confidence = base + adx_component + trend_alignment + trigger_bonus

        elif is_downtrend and (pullback_short or breakout_short):
            side = "short"
            adx_component = min((adx_val - self.min_adx) / 25.0, 1.0) * 0.40
            trend_alignment = min(abs(close - ma_val) / (atr_val * 2.0), 1.0) * 0.30
            trigger_bonus = 0.20 if breakout_short else 0.10
            base = 0.10
            confidence = base + adx_component + trend_alignment + trigger_bonus

        if side == "flat":
            return flat_signal()

        confidence = max(min(confidence, 1.0), 0.0)
        if confidence < self.min_confidence:
            return flat_signal()

        # ------------------------------------------------------------------
        # Price levels
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
                "adx": adx_val,
                "rsi": rsi_val,
                "atr": atr_val,
                "ma": ma_val,
                "lookback_high": lookback_high,
                "lookback_low": lookback_low,
                "regime": ctx.regime,
            },
        )

        if not self.invariants_ok(signal):
            return flat_signal()

        return signal
