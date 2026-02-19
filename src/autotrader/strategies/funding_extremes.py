"""Funding Rate Fade Strategy -- active across multiple regimes.

Fades extreme funding rates combined with price exhaustion.  When funding is
extremely positive (crowded longs) and RSI shows exhaustion, we short.  When
funding is extremely negative (crowded shorts) and RSI shows exhaustion, we
go long.  This is a contrarian / mean-reversion play on positioning.
"""

from __future__ import annotations

from autotrader.hl.types import Signal
from autotrader.strategies.base import BaseStrategy, MarketContext, flat_signal

_DEFAULT_CONFIG: dict = {
    "funding_zscore_threshold": 2.0,
    "funding_percentile_threshold": 0.9,
    "rsi_exhaustion_long": 25.0,
    "rsi_exhaustion_short": 75.0,
    "atr_multiplier_stop": 2.0,
    "atr_multiplier_tp": 3.0,
    "min_confidence": 0.4,
}


class FundingExtremesStrategy(BaseStrategy):
    """Fade extreme funding with price exhaustion confirmation.

    Parameters
    ----------
    config : dict | None
        Override any default parameter by key name.
    """

    name: str = "funding_extremes"

    def __init__(self, config: dict | None = None) -> None:
        merged = {**_DEFAULT_CONFIG}
        if config:
            merged.update(config)

        self.funding_zscore_threshold: float = float(merged["funding_zscore_threshold"])
        self.funding_percentile_threshold: float = float(merged["funding_percentile_threshold"])
        self.rsi_exhaustion_long: float = float(merged["rsi_exhaustion_long"])
        self.rsi_exhaustion_short: float = float(merged["rsi_exhaustion_short"])
        self.atr_multiplier_stop: float = float(merged["atr_multiplier_stop"])
        self.atr_multiplier_tp: float = float(merged["atr_multiplier_tp"])
        self.min_confidence: float = float(merged["min_confidence"])

    # ------------------------------------------------------------------

    def applicable_regimes(self) -> list[str]:
        return ["TREND", "RANGE", "VOLATILE_BREAKOUT", "MEAN_REVERT_CRASH"]

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
        """Generate a funding-fade signal.

        SHORT when:
            * funding_zscore > threshold (extremely positive funding)
            * funding_percentile > threshold (top decile)
            * RSI > rsi_exhaustion_short (price exhaustion / overbought)

        LONG when:
            * funding_zscore < -threshold (extremely negative funding)
            * funding_percentile < (1 - threshold) (bottom decile)
            * RSI < rsi_exhaustion_long (price exhaustion / oversold)
        """
        candles = ctx.candles
        features = ctx.features

        if candles.empty:
            return flat_signal()

        close = float(candles["close"].iloc[-1])

        # ------------------------------------------------------------------
        # Required features
        # ------------------------------------------------------------------
        funding_zscore = self._safe_feat(features, "funding_zscore")
        funding_pct = self._safe_feat(features, "funding_percentile")
        rsi_val = self._safe_feat(features, "rsi")
        atr_val = self._safe_feat(features, "atr")

        # Need at minimum the z-score and RSI
        if funding_zscore is None or rsi_val is None or atr_val is None:
            return flat_signal()

        if atr_val <= 0.0:
            return flat_signal()

        # ------------------------------------------------------------------
        # Determine direction
        # ------------------------------------------------------------------
        side: str = "flat"
        confidence: float = 0.0

        # SHORT: fade crowded longs (extreme positive funding + price exhaustion)
        funding_extremely_positive = funding_zscore > self.funding_zscore_threshold
        pct_extremely_positive = (
            funding_pct is not None and funding_pct > self.funding_percentile_threshold
        )
        price_exhaustion_short = rsi_val > self.rsi_exhaustion_short

        # LONG: fade crowded shorts (extreme negative funding + price exhaustion)
        funding_extremely_negative = funding_zscore < -self.funding_zscore_threshold
        pct_extremely_negative = funding_pct is not None and funding_pct < (
            1.0 - self.funding_percentile_threshold
        )
        price_exhaustion_long = rsi_val < self.rsi_exhaustion_long

        if funding_extremely_positive and price_exhaustion_short:
            side = "short"

            # Confidence: funding z-score strength + percentile confirmation + RSI
            zscore_component = (
                min((abs(funding_zscore) - self.funding_zscore_threshold) / 2.0, 1.0) * 0.35
            )
            pct_bonus = 0.20 if pct_extremely_positive else 0.0
            rsi_component = (
                min(
                    (rsi_val - self.rsi_exhaustion_short)
                    / (100.0 - self.rsi_exhaustion_short + 1e-9),
                    1.0,
                )
                * 0.25
            )
            base = 0.15
            confidence = base + zscore_component + pct_bonus + rsi_component

        elif funding_extremely_negative and price_exhaustion_long:
            side = "long"

            zscore_component = (
                min((abs(funding_zscore) - self.funding_zscore_threshold) / 2.0, 1.0) * 0.35
            )
            pct_bonus = 0.20 if pct_extremely_negative else 0.0
            rsi_component = (
                min(
                    (self.rsi_exhaustion_long - rsi_val) / (self.rsi_exhaustion_long + 1e-9),
                    1.0,
                )
                * 0.25
            )
            base = 0.15
            confidence = base + zscore_component + pct_bonus + rsi_component

        if side == "flat":
            return flat_signal()

        confidence = max(min(confidence, 1.0), 0.0)
        if confidence < self.min_confidence:
            return flat_signal()

        # ------------------------------------------------------------------
        # Price levels -- tight stops, medium targets
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
                "funding_zscore": funding_zscore,
                "funding_percentile": funding_pct,
                "rsi": rsi_val,
                "atr": atr_val,
                "regime": ctx.regime,
            },
        )

        if not self.invariants_ok(signal):
            return flat_signal()

        return signal
