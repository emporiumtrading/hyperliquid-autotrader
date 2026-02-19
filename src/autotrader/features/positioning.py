"""Positioning and flow features derived from funding rates and open interest."""

from __future__ import annotations

import numpy as np
import pandas as pd


def funding_rate_zscore(funding_rates: pd.Series, period: int = 100) -> pd.Series:
    """Z-score of the funding rate over a rolling window.

    zscore = (current - rolling_mean) / rolling_std
    """
    rolling_mean = funding_rates.rolling(window=period, min_periods=period).mean()
    rolling_std = funding_rates.rolling(window=period, min_periods=period).std(ddof=1)

    zscore = (funding_rates - rolling_mean) / rolling_std
    zscore = zscore.replace([np.inf, -np.inf], np.nan)

    return zscore


def funding_rate_percentile(funding_rates: pd.Series, period: int = 100) -> pd.Series:
    """Rolling percentile rank of the funding rate.

    Returns values in [0, 1] representing the fraction of values in the
    rolling window that are less than the current value.
    """

    def _pct_rank(window: pd.Series) -> float:
        if len(window) < 2:
            return np.nan
        current = window.iloc[-1]
        count_below = (window.iloc[:-1] < current).sum()
        return count_below / (len(window) - 1)

    return funding_rates.rolling(window=period, min_periods=period).apply(_pct_rank, raw=False)


def oi_change_pct(open_interest: pd.Series, period: int = 1) -> pd.Series:
    """Percent change in open interest over *period* bars.

    Returns fractional change (e.g. 0.05 = 5% increase).
    """
    change = open_interest.pct_change(periods=period)
    change = change.replace([np.inf, -np.inf], np.nan)
    return change


def oi_zscore(open_interest: pd.Series, period: int = 50) -> pd.Series:
    """Z-score of open-interest changes over a rolling window.

    First computes the 1-bar percent change of OI, then calculates its z-score.
    """
    oi_pct = open_interest.pct_change(periods=1)
    oi_pct = oi_pct.replace([np.inf, -np.inf], np.nan)

    rolling_mean = oi_pct.rolling(window=period, min_periods=period).mean()
    rolling_std = oi_pct.rolling(window=period, min_periods=period).std(ddof=1)

    zscore = (oi_pct - rolling_mean) / rolling_std
    zscore = zscore.replace([np.inf, -np.inf], np.nan)

    return zscore


def long_short_ratio(funding_rate: pd.Series) -> pd.Series:
    """Proxy for the long/short ratio derived from funding rate.

    Positive funding => more longs paying shorts => ratio > 1.
    Negative funding => more shorts paying longs => ratio < 1.

    We convert funding rate to a ratio via:
        ratio = 1 + funding_rate * scale_factor

    where scale_factor = 1000 provides a reasonable mapping
    (e.g. 0.01% funding -> 1.1 ratio).  The result is clipped to [0.1, 10]
    to avoid extreme values.
    """
    scale_factor = 1000.0
    ratio = 1.0 + funding_rate * scale_factor
    return ratio.clip(lower=0.1, upper=10.0)
