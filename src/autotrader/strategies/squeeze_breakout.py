"""Squeeze Breakout Strategy -- designed for the SQUEEZE_RISK regime.

When Bollinger Band width is extremely compressed (squeeze), places trades
anticipating the eventual breakout direction.  Uses price position relative
to the middle band and MA slope to determine direction, with volume readiness
as confirmation.  Tight ATR-based stops (early in the move) and wide targets
(breakouts can run far).
"""

from __future__ import annotations

from autotrader.hl.types import Signal
from autotrader.strategies.base import BaseStrategy, MarketContext, flat_signal

_DEFAULT_CONFIG: dict = {
    "bb_squeeze_percentile": 0.15,
    "volume_readiness_factor": 0.8,
    "atr_multiplier_stop": 1.5,
    "atr_multiplier_tp": 4.0,
    "min_confidence": 0.3,
}


class SqueezeBreakoutStrategy(BaseStrategy):
    """Trade anticipated breakouts from Bollinger Band squeezes.

    Parameters
    ----------
    config : dict | None
        Override any default parameter by key name.
    """

    name: str = "squeeze_breakout"

    def __init__(self, config: dict | None = None) -> None:
        merged = {**_DEFAULT_CONFIG}
        if config:
            merged.update(config)

        self.bb_squeeze_percentile: float = float(merged["bb_squeeze_percentile"])
        self.volume_readiness_factor: float = float(merged["volume_readiness_factor"])
        self.atr_multiplier_stop: float = float(merged["atr_multiplier_stop"])
        self.atr_multiplier_tp: float = float(merged["atr_multiplier_tp"])
        self.min_confidence: float = float(merged["min_confidence"])

    # ------------------------------------------------------------------

    def applicable_regimes(self) -> list[str]:
        return ["SQUEEZE_RISK"]

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
        """Generate a squeeze-breakout signal.

        Conditions:
        1. BB width percentile < squeeze threshold (extreme compression).
        2. Direction: price above/below middle BB + MA slope alignment.
        3. Volume readiness: current volume >= volume_readiness_factor * volume_sma.
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
        bb_middle = self._safe_feat(features, "bb_middle")
        ma_slope_val = self._safe_feat(features, "ma_slope")
        atr_val = self._safe_feat(features, "atr")
        volume_sma_val = self._safe_feat(features, "volume_sma")

        if any(v is None for v in (bb_width_pct, bb_middle, ma_slope_val, atr_val)):
            return flat_signal()

        assert bb_width_pct is not None
        assert bb_middle is not None
        assert ma_slope_val is not None
        assert atr_val is not None

        if atr_val <= 0.0:
            return flat_signal()

        # ------------------------------------------------------------------
        # Squeeze check: BB width must be extremely compressed
        # ------------------------------------------------------------------
        if bb_width_pct >= self.bb_squeeze_percentile:
            return flat_signal()

        # ------------------------------------------------------------------
        # Volume readiness
        # ------------------------------------------------------------------
        volume_ready = True  # default if volume data unavailable
        current_volume = float(candles["volume"].iloc[-1])

        if volume_sma_val is not None and volume_sma_val > 0.0:
            if current_volume < self.volume_readiness_factor * volume_sma_val:
                volume_ready = False

        if not volume_ready:
            return flat_signal()

        # ------------------------------------------------------------------
        # Direction: price vs middle BB + MA slope alignment
        # ------------------------------------------------------------------
        price_above_mid = close > bb_middle
        slope_positive = ma_slope_val > 0.0
        slope_negative = ma_slope_val < 0.0

        side: str = "flat"

        if price_above_mid and slope_positive:
            side = "long"
        elif not price_above_mid and slope_negative:
            side = "short"

        if side == "flat":
            return flat_signal()

        # ------------------------------------------------------------------
        # Confidence
        # ------------------------------------------------------------------
        # Components:
        #   1. Squeeze compression (lower bb_width_pct = more potential)
        #   2. Volume readiness (higher volume relative to SMA = better)
        #   3. MA slope alignment strength

        # Compression score: how deep into the squeeze we are
        # At pct=0 we get full score, at pct=squeeze_threshold we get 0.
        compression_score = (
            max(1.0 - bb_width_pct / (self.bb_squeeze_percentile + 1e-9), 0.0) * 0.35
        )

        # Volume readiness score
        vol_score = 0.0
        if volume_sma_val is not None and volume_sma_val > 0.0:
            vol_ratio = current_volume / volume_sma_val
            # Scale: at 0.8x we get minimal score, at 1.5x+ we get full score
            vol_score = min(
                max((vol_ratio - self.volume_readiness_factor) / (1.5 - self.volume_readiness_factor + 1e-9), 0.0),
                1.0,
            ) * 0.30
        else:
            vol_score = 0.15  # partial credit when volume data unavailable

        # MA slope strength
        slope_score = min(abs(ma_slope_val) / (atr_val * 0.1 + 1e-9), 1.0) * 0.20

        base = 0.15
        confidence = base + compression_score + vol_score + slope_score
        confidence = max(min(confidence, 1.0), 0.0)

        if confidence < self.min_confidence:
            return flat_signal()

        # ------------------------------------------------------------------
        # Price levels -- tight stop, wide target for early breakout entry
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
                "bb_middle": bb_middle,
                "ma_slope": ma_slope_val,
                "atr": atr_val,
                "volume_ready": volume_ready,
                "current_volume": current_volume,
                "volume_sma": volume_sma_val,
                "regime": ctx.regime,
            },
        )

        if not self.invariants_ok(signal):
            return flat_signal()

        return signal
