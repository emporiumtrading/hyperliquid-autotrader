"""Crash Mean Reversion Strategy -- designed for the MEAN_REVERT_CRASH regime.

During high-volatility crash conditions, identifies exhaustion points where
extreme moves are likely to mean-revert.  Uses extreme RSI readings, long
wick ratios (indicating rejection), elevated realized volatility, and
Bollinger Band breaches to catch turning points with wider stops appropriate
for the volatile environment.
"""

from __future__ import annotations

from autotrader.hl.types import Signal
from autotrader.strategies.base import BaseStrategy, MarketContext, flat_signal

_DEFAULT_CONFIG: dict = {
    "rsi_period": 14,
    "rsi_oversold": 20.0,
    "rsi_overbought": 80.0,
    "bb_period": 20,
    "bb_std": 2.0,
    "atr_multiplier_stop": 3.0,
    "atr_multiplier_tp": 2.5,
    "min_wick_ratio": 0.6,
    "min_realized_vol": 0.5,
    "min_confidence": 0.3,
}


class CrashMeanRevStrategy(BaseStrategy):
    """Mean-revert at exhaustion points during crash / high-vol regimes.

    Parameters
    ----------
    config : dict | None
        Override any default parameter by key name.
    """

    name: str = "crash_meanrev"

    def __init__(self, config: dict | None = None) -> None:
        merged = {**_DEFAULT_CONFIG}
        if config:
            merged.update(config)

        self.rsi_period: int = int(merged["rsi_period"])
        self.rsi_oversold: float = float(merged["rsi_oversold"])
        self.rsi_overbought: float = float(merged["rsi_overbought"])
        self.bb_period: int = int(merged["bb_period"])
        self.bb_std: float = float(merged["bb_std"])
        self.atr_multiplier_stop: float = float(merged["atr_multiplier_stop"])
        self.atr_multiplier_tp: float = float(merged["atr_multiplier_tp"])
        self.min_wick_ratio: float = float(merged["min_wick_ratio"])
        self.min_realized_vol: float = float(merged["min_realized_vol"])
        self.min_confidence: float = float(merged["min_confidence"])

    # ------------------------------------------------------------------

    def applicable_regimes(self) -> list[str]:
        return ["MEAN_REVERT_CRASH"]

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
        """Generate a mean-reversion signal at exhaustion extremes.

        Long (catching falling knife): close below lower BB + RSI < 20 +
        high wick ratio (buying rejection) + elevated realized vol.

        Short (fading spike): close above upper BB + RSI > 80 +
        high wick ratio (selling rejection) + elevated realized vol.
        """
        candles = ctx.candles
        features = ctx.features

        if candles.empty:
            return flat_signal()

        close = float(candles["close"].iloc[-1])

        # Required features
        rsi_val = self._safe_feat(features, "rsi")
        atr_val = self._safe_feat(features, "atr")
        wick_ratio = self._safe_feat(features, "wick_ratio")
        realized_vol = self._safe_feat(features, "realized_vol")
        bb_upper = self._safe_feat(features, "bb_upper")
        bb_lower = self._safe_feat(features, "bb_lower")
        bb_middle = self._safe_feat(features, "bb_middle")

        if any(
            v is None
            for v in (rsi_val, atr_val, wick_ratio, realized_vol, bb_upper, bb_lower, bb_middle)
        ):
            return flat_signal()

        # Type narrowing for mypy (already checked None above)
        assert rsi_val is not None
        assert atr_val is not None
        assert wick_ratio is not None
        assert realized_vol is not None
        assert bb_upper is not None
        assert bb_lower is not None
        assert bb_middle is not None

        if atr_val <= 0.0:
            return flat_signal()

        # Must be in a high-vol environment
        if realized_vol < self.min_realized_vol:
            return flat_signal()

        # Must show rejection wicks
        if wick_ratio < self.min_wick_ratio:
            return flat_signal()

        # Bollinger Band width (used for confidence calculations)
        bb_width = bb_upper - bb_lower
        if bb_width <= 0.0:
            return flat_signal()

        # ------------------------------------------------------------------
        # Determine direction
        # ------------------------------------------------------------------
        side: str = "flat"
        confidence: float = 0.0

        # LONG: catching a falling knife -- price below lower BB + extreme oversold
        if close < bb_lower and rsi_val < self.rsi_oversold:
            side = "long"
            # Confidence based on RSI extremity, wick ratio, and BB breach depth
            rsi_extremity = max((self.rsi_oversold - rsi_val) / self.rsi_oversold, 0.0) * 0.35
            wick_component = (
                max(min((wick_ratio - self.min_wick_ratio) / (1.0 - self.min_wick_ratio), 1.0), 0.0)
                * 0.30
            )
            bb_breach = max(min((bb_lower - close) / bb_width, 1.0), 0.0) * 0.25
            base = 0.10
            confidence = base + rsi_extremity + wick_component + bb_breach

        # SHORT: fading a spike -- price above upper BB + extreme overbought
        elif close > bb_upper and rsi_val > self.rsi_overbought:
            side = "short"
            rsi_extremity = (
                max((rsi_val - self.rsi_overbought) / (100.0 - self.rsi_overbought), 0.0) * 0.35
            )
            wick_component = (
                max(min((wick_ratio - self.min_wick_ratio) / (1.0 - self.min_wick_ratio), 1.0), 0.0)
                * 0.30
            )
            bb_breach = max(min((close - bb_upper) / bb_width, 1.0), 0.0) * 0.25
            base = 0.10
            confidence = base + rsi_extremity + wick_component + bb_breach

        if side == "flat":
            return flat_signal()

        confidence = max(min(confidence, 1.0), 0.0)
        if confidence < self.min_confidence:
            return flat_signal()

        # ------------------------------------------------------------------
        # Price levels -- wider stops for volatile environment, modest TP
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
                "rsi": rsi_val,
                "atr": atr_val,
                "wick_ratio": wick_ratio,
                "realized_vol": realized_vol,
                "bb_upper": bb_upper,
                "bb_lower": bb_lower,
                "bb_middle": bb_middle,
                "regime": ctx.regime,
            },
        )

        if not self.invariants_ok(signal):
            return flat_signal()

        return signal
