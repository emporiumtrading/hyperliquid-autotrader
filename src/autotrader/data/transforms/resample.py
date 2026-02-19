"""OHLCV timeframe resampling.

Converts candle data from a finer bar interval to a coarser one using
standard aggregation rules (first open, max high, min low, last close,
sum volume).
"""

from __future__ import annotations

import pandas as pd

from autotrader.utils.time import interval_ms


def resample_ohlcv(
    df: pd.DataFrame,
    source_interval: str,
    target_interval: str,
) -> pd.DataFrame:
    """Resample OHLCV data from *source_interval* to *target_interval*.

    Parameters
    ----------
    df:
        Input candle DataFrame with column ``timestamp_ms`` (int64) and the
        standard OHLCV columns.
    source_interval:
        Current bar interval (e.g. ``"1m"``).  Used for validation only.
    target_interval:
        Desired output bar interval (e.g. ``"15m"``).

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame with ``timestamp_ms`` as an int64 column
        (not an index) and the standard OHLCV columns.

    Raises
    ------
    ValueError
        If *target_interval* is not a whole multiple of *source_interval*.
    """
    if df.empty:
        return df.copy()

    source_ms = interval_ms(source_interval)
    target_ms = interval_ms(target_interval)

    if target_ms < source_ms:
        raise ValueError(
            f"target_interval ({target_interval}, {target_ms}ms) must be "
            f">= source_interval ({source_interval}, {source_ms}ms)"
        )
    if target_ms % source_ms != 0:
        raise ValueError(
            f"target_interval ({target_interval}) is not a whole multiple "
            f"of source_interval ({source_interval})"
        )

    # Convert timestamp_ms to a DatetimeIndex for pandas resampling
    work = df.copy()
    work["_dt"] = pd.to_datetime(work["timestamp_ms"], unit="ms", utc=True)
    work = work.set_index("_dt").sort_index()

    # Build a pandas-compatible frequency string from the target interval
    freq = _interval_to_pandas_freq(target_interval)

    # Standard OHLCV aggregation
    agg = work.resample(freq, closed="left", label="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    # Drop incomplete bars that result from NaN (e.g. partial last bar)
    agg = agg.dropna(subset=["open"])

    # Reconstruct timestamp_ms from the resampled DatetimeIndex
    agg["timestamp_ms"] = (agg.index.astype("int64") // 10**6).astype("int64")

    result = agg[["timestamp_ms", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
    return result


def _interval_to_pandas_freq(interval: str) -> str:
    """Convert a bar-interval string to a pandas-compatible frequency alias.

    Examples
    --------
    >>> _interval_to_pandas_freq("1m")
    '1min'
    >>> _interval_to_pandas_freq("4h")
    '4h'
    >>> _interval_to_pandas_freq("1d")
    '1D'
    """
    if len(interval) < 2:
        raise ValueError(f"Invalid interval: {interval!r}")

    amount = interval[:-1]
    unit = interval[-1]

    mapping = {
        "m": "min",
        "h": "h",
        "d": "D",
        "w": "W",
        "M": "MS",  # Month Start
    }

    pd_unit = mapping.get(unit)
    if pd_unit is None:
        raise ValueError(f"Unsupported interval unit {unit!r} in {interval!r}")

    return f"{amount}{pd_unit}"
