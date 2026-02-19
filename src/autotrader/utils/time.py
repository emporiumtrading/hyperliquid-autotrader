"""Time utilities for UTC timestamps, bar intervals, and bar alignment."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Final

# ---------------------------------------------------------------------------
# Timeframe constants
# ---------------------------------------------------------------------------

TIMEFRAME_1M: Final[str] = "1m"
TIMEFRAME_5M: Final[str] = "5m"
TIMEFRAME_15M: Final[str] = "15m"
TIMEFRAME_30M: Final[str] = "30m"
TIMEFRAME_1H: Final[str] = "1h"
TIMEFRAME_4H: Final[str] = "4h"
TIMEFRAME_1D: Final[str] = "1d"
TIMEFRAME_1W: Final[str] = "1w"
TIMEFRAME_1MO: Final[str] = "1M"

# Mapping from bar‑interval string to its approximate timedelta.
# "1M" uses 30 days as an approximation.
_INTERVAL_MAP: Final[dict[str, timedelta]] = {
    "1m": timedelta(minutes=1),
    "3m": timedelta(minutes=3),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "2h": timedelta(hours=2),
    "4h": timedelta(hours=4),
    "6h": timedelta(hours=6),
    "8h": timedelta(hours=8),
    "12h": timedelta(hours=12),
    "1d": timedelta(days=1),
    "3d": timedelta(days=3),
    "1w": timedelta(weeks=1),
    "1M": timedelta(days=30),
}

# Milliseconds per second – avoids magic numbers elsewhere.
MS_PER_SEC: Final[int] = 1_000
MS_PER_MIN: Final[int] = 60 * MS_PER_SEC
MS_PER_HOUR: Final[int] = 60 * MS_PER_MIN
MS_PER_DAY: Final[int] = 24 * MS_PER_HOUR

# ---------------------------------------------------------------------------
# UTC timestamp helpers
# ---------------------------------------------------------------------------


def now_ms() -> int:
    """Return the current UTC time as milliseconds since the Unix epoch."""
    return int(datetime.now(timezone.utc).timestamp() * MS_PER_SEC)


def ms_to_datetime(ms: int) -> datetime:
    """Convert milliseconds since the Unix epoch to a timezone-aware UTC datetime."""
    return datetime.fromtimestamp(ms / MS_PER_SEC, tz=timezone.utc)


def datetime_to_ms(dt: datetime) -> int:
    """Convert a datetime to milliseconds since the Unix epoch.

    If *dt* is naive (no tzinfo), it is assumed to be UTC.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * MS_PER_SEC)


# ---------------------------------------------------------------------------
# Bar interval parsing
# ---------------------------------------------------------------------------


def parse_interval(interval: str) -> timedelta:
    """Parse a bar-interval string (e.g. ``"15m"``, ``"4h"``, ``"1d"``) into a
    :class:`~datetime.timedelta`.

    Supported suffixes: ``m`` (minutes), ``h`` (hours), ``d`` (days),
    ``w`` (weeks), ``M`` (months, approximated as 30 days).

    Raises
    ------
    ValueError
        If *interval* is not recognised.
    """
    td = _INTERVAL_MAP.get(interval)
    if td is not None:
        return td

    # Fallback: try to parse "<number><unit>" generically.
    if len(interval) < 2:
        raise ValueError(f"Unknown bar interval: {interval!r}")

    unit = interval[-1]
    try:
        amount = int(interval[:-1])
    except ValueError:
        raise ValueError(f"Unknown bar interval: {interval!r}") from None

    if unit == "m":
        return timedelta(minutes=amount)
    if unit == "h":
        return timedelta(hours=amount)
    if unit == "d":
        return timedelta(days=amount)
    if unit == "w":
        return timedelta(weeks=amount)
    if unit == "M":
        return timedelta(days=amount * 30)

    raise ValueError(f"Unknown bar interval unit {unit!r} in {interval!r}")


def interval_ms(interval: str) -> int:
    """Return the bar-interval duration in milliseconds."""
    return int(parse_interval(interval).total_seconds() * MS_PER_SEC)


# ---------------------------------------------------------------------------
# Bar alignment
# ---------------------------------------------------------------------------


def align_timestamp_ms(ts_ms: int, interval: str) -> int:
    """Align (floor) a millisecond timestamp to the start of its bar boundary.

    For sub-day intervals the alignment is relative to the Unix epoch
    (midnight UTC 1970-01-01).  For ``"1w"`` the alignment is to Monday
    00:00 UTC (epoch day 0 was a Thursday, so we offset by 3 days).  For
    ``"1M"`` the alignment is to the first day of the month.

    Parameters
    ----------
    ts_ms:
        Millisecond timestamp to align.
    interval:
        Bar-interval string such as ``"15m"`` or ``"1d"``.

    Returns
    -------
    int
        Aligned millisecond timestamp.
    """
    if interval == "1M":
        dt = ms_to_datetime(ts_ms)
        aligned = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return datetime_to_ms(aligned)

    if interval == "1w":
        dt = ms_to_datetime(ts_ms)
        # Shift to Monday 00:00 UTC of the same ISO week.
        days_since_monday = dt.weekday()  # Monday == 0
        monday = dt - timedelta(days=days_since_monday)
        aligned = monday.replace(hour=0, minute=0, second=0, microsecond=0)
        return datetime_to_ms(aligned)

    bar_ms = interval_ms(interval)
    return (ts_ms // bar_ms) * bar_ms
