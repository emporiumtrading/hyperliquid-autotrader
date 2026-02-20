"""Microstructure features from order book and trade data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_spread_bps(bid_px: float, ask_px: float) -> float:
    """Compute bid-ask spread in basis points.

    spread_bps = (ask - bid) / mid * 10_000
    Returns 0.0 if mid is zero.
    """
    mid = (bid_px + ask_px) / 2.0
    if mid == 0.0:
        return 0.0
    return (ask_px - bid_px) / mid * 10_000.0


def compute_mid_price(bid_px: float, ask_px: float) -> float:
    """Compute mid price as the average of best bid and best ask."""
    return (bid_px + ask_px) / 2.0


def compute_depth_usd(levels: list[tuple[float, float]], n_levels: int = 5) -> float:
    """Sum of price * size for the top *n_levels* of one side of the book.

    Each element of *levels* is (price, size).  Only the first *n_levels*
    entries are considered.

    Returns the total notional depth in USD.
    """
    total = 0.0
    for i, (px, sz) in enumerate(levels):
        if i >= n_levels:
            break
        total += px * sz
    return total


def compute_book_imbalance(bid_depth: float, ask_depth: float) -> float:
    """Order-book imbalance: (bid_depth - ask_depth) / (bid_depth + ask_depth).

    Returns a value in [-1, 1].
    Returns 0.0 if both sides are zero (empty book).
    """
    total = bid_depth + ask_depth
    if total == 0.0:
        return 0.0
    return (bid_depth - ask_depth) / total


def compute_vwap(prices: pd.Series, volumes: pd.Series, period: int = 20) -> pd.Series:
    """Volume-weighted average price over a rolling window.

    VWAP = sum(price * volume) / sum(volume) over *period* bars.
    """
    pv = prices * volumes
    sum_pv = pv.rolling(window=period, min_periods=period).sum()
    sum_v = volumes.rolling(window=period, min_periods=period).sum()

    vwap = sum_pv / sum_v
    # Handle case where cumulative volume is zero
    vwap = vwap.replace([np.inf, -np.inf], np.nan)

    return vwap


def compute_volume_profile(close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """Relative volume: current volume divided by its SMA.

    Values > 1.0 indicate above-average activity; < 1.0 below-average.
    """
    vol_sma = volume.rolling(window=period, min_periods=period).mean()
    rel_vol = volume / vol_sma
    rel_vol = rel_vol.replace([np.inf, -np.inf], np.nan)
    return rel_vol


def compute_trade_intensity(
    volume: pd.Series,
    period: int = 20,
    spike_threshold: float = 2.0,
) -> pd.DataFrame:
    """Compute trade intensity metrics from bar volume data.

    Returns a DataFrame with columns:

    * ``intensity`` -- rolling trade count / volume normalised by its
      SMA.  Values > 1.0 indicate above-average activity.
    * ``acceleration`` -- bar-over-bar change of ``intensity``,
      capturing whether activity is *increasing* (positive) or
      *waning* (negative).
    * ``is_spike`` -- boolean flag when ``intensity`` exceeds
      *spike_threshold* (default 2.0), useful as a filter for
      breakout / capitulation detection.

    Parameters
    ----------
    volume : pd.Series
        Per-bar volume (trade count or notional).
    period : int
        Rolling window for the baseline SMA (default 20).
    spike_threshold : float
        Multiple of the SMA above which a bar is flagged as a spike.
    """
    vol_sma = volume.rolling(window=period, min_periods=period).mean()

    intensity = volume / vol_sma
    intensity = intensity.replace([np.inf, -np.inf], np.nan)

    acceleration = intensity.diff()

    is_spike = intensity >= spike_threshold

    return pd.DataFrame(
        {
            "intensity": intensity,
            "acceleration": acceleration,
            "is_spike": is_spike,
        },
        index=volume.index,
    )
