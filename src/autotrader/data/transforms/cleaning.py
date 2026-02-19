"""Data cleaning utilities -- gap detection, outlier removal, OHLCV validation.

All functions operate on pandas DataFrames that follow the candle schema
(``timestamp_ms``, ``open``, ``high``, ``low``, ``close``, ``volume``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from autotrader.utils.time import interval_ms


def detect_gaps(
    df: pd.DataFrame,
    interval: str,
    max_gap_multiple: float = 2.0,
) -> pd.DataFrame:
    """Detect timestamp gaps that exceed *max_gap_multiple* times the bar interval.

    Parameters
    ----------
    df:
        Candle DataFrame with a ``timestamp_ms`` column.
    interval:
        Expected bar interval (e.g. ``"15m"``).
    max_gap_multiple:
        A gap is flagged when the time between consecutive bars exceeds
        ``max_gap_multiple * interval_ms(interval)``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``start_ms``, ``end_ms``, and ``gap_bars``
        for each detected gap.  ``gap_bars`` is the gap duration expressed
        as an integer multiple of the bar interval.  Returns an empty
        DataFrame with the correct columns if no gaps are found.
    """
    empty = pd.DataFrame(columns=["start_ms", "end_ms", "gap_bars"])

    if df.empty or len(df) < 2:
        return empty

    sorted_df = df.sort_values("timestamp_ms").reset_index(drop=True)
    bar_ms = interval_ms(interval)
    threshold = max_gap_multiple * bar_ms

    ts = sorted_df["timestamp_ms"].values
    diffs = np.diff(ts)

    gap_mask = diffs > threshold
    if not gap_mask.any():
        return empty

    gap_indices = np.where(gap_mask)[0]

    rows: list[dict] = []
    for idx in gap_indices:
        start = int(ts[idx])
        end = int(ts[idx + 1])
        gap_bars = int(round((end - start) / bar_ms))
        rows.append({"start_ms": start, "end_ms": end, "gap_bars": gap_bars})

    return pd.DataFrame(rows)


def remove_outliers(
    df: pd.DataFrame,
    column: str = "close",
    z_threshold: float = 5.0,
) -> pd.DataFrame:
    """Remove rows where the z-score of *column* exceeds *z_threshold*.

    Parameters
    ----------
    df:
        Input DataFrame (not modified in-place).
    column:
        Column to compute z-scores on.
    z_threshold:
        Absolute z-score cutoff.  Rows exceeding this are dropped.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with outliers removed, index reset.
    """
    if df.empty or column not in df.columns:
        return df.copy()

    values = df[column].astype(float)
    mean = values.mean()
    std = values.std()

    if std == 0 or np.isnan(std):
        # All values are identical or insufficient data -- nothing to remove
        return df.copy()

    z_scores = ((values - mean) / std).abs()
    mask = z_scores <= z_threshold

    return df.loc[mask].reset_index(drop=True)


def validate_ohlcv(df: pd.DataFrame) -> list[str]:
    """Validate OHLCV data integrity and return a list of error descriptions.

    Checks performed on every row:

    1. ``high >= open`` and ``high >= close``
    2. ``low <= open`` and ``low <= close``
    3. ``volume >= 0``

    Parameters
    ----------
    df:
        Candle DataFrame.

    Returns
    -------
    list[str]
        Human-readable error messages.  Empty list means the data is valid.
    """
    if df.empty:
        return []

    errors: list[str] = []

    # Vectorised checks for performance
    high_lt_open = df["high"] < df["open"]
    high_lt_close = df["high"] < df["close"]
    low_gt_open = df["low"] > df["open"]
    low_gt_close = df["low"] > df["close"]
    neg_volume = df["volume"] < 0

    for idx in df.index[high_lt_open]:
        ts = df.at[idx, "timestamp_ms"]
        errors.append(
            f"Row {idx} (ts={ts}): high ({df.at[idx, 'high']}) < open ({df.at[idx, 'open']})"
        )

    for idx in df.index[high_lt_close]:
        ts = df.at[idx, "timestamp_ms"]
        errors.append(
            f"Row {idx} (ts={ts}): high ({df.at[idx, 'high']}) < close ({df.at[idx, 'close']})"
        )

    for idx in df.index[low_gt_open]:
        ts = df.at[idx, "timestamp_ms"]
        errors.append(
            f"Row {idx} (ts={ts}): low ({df.at[idx, 'low']}) > open ({df.at[idx, 'open']})"
        )

    for idx in df.index[low_gt_close]:
        ts = df.at[idx, "timestamp_ms"]
        errors.append(
            f"Row {idx} (ts={ts}): low ({df.at[idx, 'low']}) > close ({df.at[idx, 'close']})"
        )

    for idx in df.index[neg_volume]:
        ts = df.at[idx, "timestamp_ms"]
        errors.append(f"Row {idx} (ts={ts}): volume ({df.at[idx, 'volume']}) < 0")

    return errors
