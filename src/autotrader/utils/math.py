"""Math utilities for financial calculations."""

from __future__ import annotations

import math
from typing import Sequence

# ---------------------------------------------------------------------------
# Safe arithmetic
# ---------------------------------------------------------------------------


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide *numerator* by *denominator*, returning *default* when
    *denominator* is zero or the result is not finite.

    >>> safe_div(10, 2)
    5.0
    >>> safe_div(1, 0)
    0.0
    >>> safe_div(1, 0, default=float('nan'))
    nan
    """
    if denominator == 0.0:
        return default
    result = numerator / denominator
    if not math.isfinite(result):
        return default
    return result


def pct_change(old: float, new: float) -> float:
    """Return the percentage change from *old* to *new*.

    The result is expressed as a fraction (0.05 means +5 %).
    Returns 0.0 when *old* is zero.

    >>> pct_change(100, 105)
    0.05
    >>> pct_change(0, 10)
    0.0
    """
    return safe_div(new - old, old, default=0.0)


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to the range [*lo*, *hi*].

    >>> clamp(5, 0, 10)
    5
    >>> clamp(-3, 0, 10)
    0
    >>> clamp(15, 0, 10)
    10
    """
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Rolling / windowed operations
# ---------------------------------------------------------------------------


def ema(values: Sequence[float], span: int) -> list[float]:
    """Compute the exponential moving average over *values* with the given
    *span* (period).

    Uses the standard smoothing factor ``alpha = 2 / (span + 1)``.  The first
    element of the returned list equals the first value (seed).

    Parameters
    ----------
    values:
        Input series.  Must contain at least one element.
    span:
        EMA period.  Must be >= 1.

    Returns
    -------
    list[float]
        A list the same length as *values* containing EMA values.

    Raises
    ------
    ValueError
        If *values* is empty or *span* < 1.
    """
    if not values:
        raise ValueError("values must be non-empty")
    if span < 1:
        raise ValueError(f"span must be >= 1, got {span}")

    alpha = 2.0 / (span + 1)
    result: list[float] = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1.0 - alpha) * result[-1])
    return result


def rolling_max(values: Sequence[float], window: int) -> list[float]:
    """Compute the rolling maximum over *values* with the given *window* size.

    For indices ``i < window - 1`` the rolling max is computed over
    ``values[0 : i + 1]`` (expanding window until the full window is
    reached).

    Parameters
    ----------
    values:
        Input series.
    window:
        Window size.  Must be >= 1.

    Returns
    -------
    list[float]
        A list the same length as *values*.

    Raises
    ------
    ValueError
        If *values* is empty or *window* < 1.
    """
    if not values:
        raise ValueError("values must be non-empty")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    n = len(values)
    result: list[float] = []
    for i in range(n):
        start = max(0, i - window + 1)
        result.append(max(values[start : i + 1]))
    return result


def rolling_min(values: Sequence[float], window: int) -> list[float]:
    """Compute the rolling minimum over *values* with the given *window* size.

    Follows the same expanding-window semantics as :func:`rolling_max`.

    Parameters
    ----------
    values:
        Input series.
    window:
        Window size.  Must be >= 1.

    Returns
    -------
    list[float]
        A list the same length as *values*.

    Raises
    ------
    ValueError
        If *values* is empty or *window* < 1.
    """
    if not values:
        raise ValueError("values must be non-empty")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    n = len(values)
    result: list[float] = []
    for i in range(n):
        start = max(0, i - window + 1)
        result.append(min(values[start : i + 1]))
    return result


# ---------------------------------------------------------------------------
# Annualisation
# ---------------------------------------------------------------------------


def annualize_returns(
    daily_return: float,
    trading_days: int = 365,
) -> float:
    """Convert a daily return to an annualised return.

    Uses compound growth: ``(1 + daily_return) ** trading_days - 1``.

    Parameters
    ----------
    daily_return:
        The per-day return expressed as a fraction (e.g. 0.001 for 0.1 %).
    trading_days:
        Number of trading days per year.  Defaults to 365 (crypto markets
        trade around the clock).

    Returns
    -------
    float
        Annualised return as a fraction.
    """
    return (1.0 + daily_return) ** trading_days - 1.0
