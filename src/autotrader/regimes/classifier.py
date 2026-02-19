"""Regime classifier that detects market conditions from pre-computed features.

Regime labels
-------------
- ``TREND``             -- Strong directional move (high ADX, hurst > 0.6).
- ``RANGE``             -- Sideways, mean-reverting price action.
- ``VOLATILE_BREAKOUT`` -- Volatility expansion after compression.
- ``MEAN_REVERT_CRASH`` -- Extreme move likely to revert (high vol + RSI extreme).
- ``SQUEEZE_RISK``      -- Very compressed volatility; breakout imminent.
- ``UNKNOWN``           -- Insufficient data or no regime scores above threshold.
"""

from __future__ import annotations

from typing import Sequence

# All valid regime labels (exported for reference by other modules).
REGIME_LABELS: Sequence[str] = (
    "TREND",
    "RANGE",
    "VOLATILE_BREAKOUT",
    "MEAN_REVERT_CRASH",
    "SQUEEZE_RISK",
    "UNKNOWN",
)

_DEFAULT_THRESHOLDS: dict = {
    "adx_trend_threshold": 25.0,
    "adx_range_threshold": 20.0,
    "vol_expansion_threshold": 1.5,
    "vol_compression_threshold": 0.5,
    "hurst_trend_threshold": 0.6,
    "hurst_range_threshold": 0.4,
}


class RegimeClassifier:
    """Score-based regime classifier.

    Parameters
    ----------
    config : dict | None
        Override any of the default thresholds by passing a dictionary whose
        keys match the threshold names listed in ``_DEFAULT_THRESHOLDS``.
    """

    def __init__(self, config: dict | None = None) -> None:
        merged = {**_DEFAULT_THRESHOLDS}
        if config:
            merged.update(config)

        self.adx_trend_threshold: float = float(merged["adx_trend_threshold"])
        self.adx_range_threshold: float = float(merged["adx_range_threshold"])
        self.vol_expansion_threshold: float = float(merged["vol_expansion_threshold"])
        self.vol_compression_threshold: float = float(merged["vol_compression_threshold"])
        self.hurst_trend_threshold: float = float(merged["hurst_trend_threshold"])
        self.hurst_range_threshold: float = float(merged["hurst_range_threshold"])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_get(features: dict, key: str, default: float | None = None) -> float | None:
        """Retrieve a numeric value from *features*; return *default* on miss/NaN."""
        val = features.get(key, default)
        if val is None:
            return default
        try:
            fval = float(val)
        except (TypeError, ValueError):
            return default
        # Treat NaN as missing.
        if fval != fval:  # fast NaN check
            return default
        return fval

    # ------------------------------------------------------------------
    # Scoring functions (one per regime)
    # ------------------------------------------------------------------

    def _score_trend(self, adx: float, hurst: float, ma_slope: float, rsi: float) -> float:
        score = 0.0

        # ADX contribution (0-0.35)
        if adx > self.adx_trend_threshold:
            score += 0.15 + 0.20 * min((adx - self.adx_trend_threshold) / 25.0, 1.0)

        # Hurst contribution (0-0.25)
        if hurst > self.hurst_trend_threshold:
            score += 0.10 + 0.15 * min((hurst - self.hurst_trend_threshold) / 0.4, 1.0)

        # MA slope magnitude contribution (0-0.20)
        abs_slope = abs(ma_slope)
        if abs_slope > 0.0:
            # Normalise slope: a slope of 1.0 (1 unit per bar) maps to full score
            score += 0.20 * min(abs_slope / 1.0, 1.0)

        # RSI away from 50 contribution (0-0.20)
        rsi_deviation = abs(rsi - 50.0)
        if rsi_deviation > 15.0:
            score += 0.10 + 0.10 * min((rsi_deviation - 15.0) / 25.0, 1.0)

        return score

    def _score_range(self, adx: float, hurst: float, bb_width_pct: float, rsi: float) -> float:
        score = 0.0

        # Low ADX (0-0.30)
        if adx < self.adx_range_threshold:
            score += 0.15 + 0.15 * min((self.adx_range_threshold - adx) / 20.0, 1.0)

        # Low hurst (0-0.25)
        if hurst < self.hurst_range_threshold:
            score += 0.10 + 0.15 * min((self.hurst_range_threshold - hurst) / 0.4, 1.0)

        # Low BB width percentile (0-0.25)
        if bb_width_pct < 0.3:
            score += 0.10 + 0.15 * min((0.3 - bb_width_pct) / 0.3, 1.0)

        # RSI near 50 (0-0.20)
        if 35.0 <= rsi <= 65.0:
            closeness = 1.0 - abs(rsi - 50.0) / 15.0
            score += 0.20 * max(closeness, 0.0)

        return score

    def _score_volatile_breakout(
        self,
        adx: float,
        bb_width_pct: float,
        vol_ratio: float,
        realized_vol: float,
    ) -> float:
        score = 0.0

        # BB width expansion (0-0.35)
        if bb_width_pct > 0.8:
            score += 0.20 + 0.15 * min((bb_width_pct - 0.8) / 0.2, 1.0)

        # Vol ratio spike (0-0.30)
        if vol_ratio > self.vol_expansion_threshold:
            score += 0.15 + 0.15 * min((vol_ratio - self.vol_expansion_threshold) / 1.5, 1.0)

        # ADX rising helps confirm directional breakout (0-0.20)
        if adx > self.adx_range_threshold:
            score += 0.10 + 0.10 * min((adx - self.adx_range_threshold) / 20.0, 1.0)

        # High realized vol (0-0.15)
        if realized_vol > 0.5:
            score += 0.15 * min(realized_vol / 2.0, 1.0)

        return score

    def _score_squeeze_risk(self, bb_width_pct: float, realized_vol: float) -> float:
        score = 0.0

        # Very compressed BB width (0-0.50)
        if bb_width_pct < 0.15:
            score += 0.25 + 0.25 * min((0.15 - bb_width_pct) / 0.15, 1.0)

        # Low realized vol (0-0.30)
        if realized_vol < 0.3:
            score += 0.15 + 0.15 * min((0.3 - realized_vol) / 0.3, 1.0)

        # Combined: both conditions present => bonus (0-0.20)
        if bb_width_pct < 0.15 and realized_vol < 0.3:
            score += 0.20

        return score

    def _score_mean_revert_crash(self, realized_vol: float, rsi: float, wick_ratio: float) -> float:
        score = 0.0

        # High vol (0-0.30)
        if realized_vol > 0.8:
            score += 0.15 + 0.15 * min((realized_vol - 0.8) / 1.2, 1.0)

        # Extreme RSI (0-0.35)
        if rsi < 20.0:
            score += 0.20 + 0.15 * min((20.0 - rsi) / 20.0, 1.0)
        elif rsi > 80.0:
            score += 0.20 + 0.15 * min((rsi - 80.0) / 20.0, 1.0)

        # Significant wick ratio (rejection candles) (0-0.25)
        if wick_ratio > 1.5:
            score += 0.15 + 0.10 * min((wick_ratio - 1.5) / 3.0, 1.0)

        # Bonus if both vol and RSI are extreme (0-0.10)
        if realized_vol > 0.8 and (rsi < 20.0 or rsi > 80.0):
            score += 0.10

        return score

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, features: dict) -> tuple[str, float]:
        """Classify the current market regime from a features dictionary.

        Parameters
        ----------
        features : dict
            Must contain numeric values for keys such as ``adx``,
            ``bb_width_pct``, ``hurst``, ``realized_vol``, ``vol_ratio``,
            ``rsi``, ``ma_slope``, and optionally ``wick_ratio``.

        Returns
        -------
        (regime_label, confidence) : tuple[str, float]

        Notes
        -----
        The PRD also specifies ``expected_slippage_bps`` in the regime
        output.  This value depends on L2 book data which the classifier
        does not receive directly.  Use :meth:`classify_with_book` for
        the full output including slippage estimation.
        """
        # Extract features with safe defaults
        adx = self._safe_get(features, "adx")
        bb_width_pct = self._safe_get(features, "bb_width_pct")
        hurst = self._safe_get(features, "hurst")
        realized_vol = self._safe_get(features, "realized_vol")
        vol_ratio = self._safe_get(features, "vol_ratio", 1.0)
        rsi = self._safe_get(features, "rsi")
        ma_slope = self._safe_get(features, "ma_slope", 0.0)
        wick_ratio = self._safe_get(features, "wick_ratio", 1.0)

        # Need at least ADX and RSI and one volatility feature to produce a
        # meaningful classification.
        if adx is None or rsi is None or (bb_width_pct is None and realized_vol is None):
            return ("UNKNOWN", 0.0)

        # Fill remaining None values with neutral defaults so scoring works.
        if bb_width_pct is None:
            bb_width_pct = 0.5
        if realized_vol is None:
            realized_vol = 0.5
        if hurst is None:
            hurst = 0.5
        if vol_ratio is None:
            vol_ratio = 1.0
        if ma_slope is None:
            ma_slope = 0.0
        if wick_ratio is None:
            wick_ratio = 1.0

        # Compute scores
        scores: dict[str, float] = {
            "TREND": self._score_trend(adx, hurst, ma_slope, rsi),
            "RANGE": self._score_range(adx, hurst, bb_width_pct, rsi),
            "VOLATILE_BREAKOUT": self._score_volatile_breakout(
                adx, bb_width_pct, vol_ratio, realized_vol
            ),
            "SQUEEZE_RISK": self._score_squeeze_risk(bb_width_pct, realized_vol),
            "MEAN_REVERT_CRASH": self._score_mean_revert_crash(realized_vol, rsi, wick_ratio),
        }

        max_regime = max(scores, key=scores.__getitem__)
        max_score = scores[max_regime]

        total_score = sum(scores.values())
        if total_score == 0.0 or max_score == 0.0:
            return ("UNKNOWN", 0.0)

        confidence = max_score / total_score

        if confidence < 0.3:
            return ("UNKNOWN", confidence)

        return (max_regime, round(confidence, 4))

    def classify_with_book(
        self,
        features: dict,
        book: dict | None = None,
    ) -> dict:
        """Classify regime and return full output including slippage estimate.

        This is the PRD-compliant version that returns all three fields:
        ``regime``, ``confidence``, and ``expected_slippage_bps``.

        Parameters
        ----------
        features : dict
            Feature dictionary (same as :meth:`classify`).
        book : dict | None
            L2 book snapshot from :func:`snapshot_l2`, with keys like
            ``spread_bps``, ``bid_depth_usd``, ``ask_depth_usd``.

        Returns
        -------
        dict
            ``{"regime": str, "confidence": float, "expected_slippage_bps": float}``
        """
        regime, confidence = self.classify(features)

        # Estimate expected slippage from L2 book data
        expected_slippage_bps = 1.0  # base minimum
        if book is not None:
            spread_bps = book.get("spread_bps", 2.0)
            bid_depth = book.get("bid_depth_usd", 0.0)
            ask_depth = book.get("ask_depth_usd", 0.0)
            total_depth = bid_depth + ask_depth

            # Half-spread component
            expected_slippage_bps = spread_bps / 2.0

            # Depth penalty: less depth = more slippage
            if total_depth > 0 and total_depth < 500_000:
                depth_penalty = (500_000 - total_depth) / 500_000 * 2.0
                expected_slippage_bps += depth_penalty

            # Volatility penalty: high vol regimes tend to have wider fills
            if regime in ("VOLATILE_BREAKOUT", "MEAN_REVERT_CRASH"):
                expected_slippage_bps *= 1.5

        return {
            "regime": regime,
            "confidence": confidence,
            "expected_slippage_bps": round(expected_slippage_bps, 2),
        }


# ------------------------------------------------------------------
# Module-level convenience function (backwards-compatible)
# ------------------------------------------------------------------


def classify(features: dict) -> str:
    """Classify market regime and return only the label string.

    Creates a default :class:`RegimeClassifier` and delegates to its
    ``classify`` method.  Provided for backward compatibility.
    """
    label, _confidence = RegimeClassifier().classify(features)
    return label
