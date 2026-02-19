"""Systematic leverage selection.

Computes leverage from stop distance and liquidation buffer, optionally
haircut by recent volatility, and validates that stop-loss prices sit
safely above (or below) the liquidation price.
"""

from __future__ import annotations

import math

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Dynamic maintenance margin (Hyperliquid tiered schedule)
# ---------------------------------------------------------------------------
# Each tier: (max_notional_usd, maint_margin_pct)
# Source: Hyperliquid documentation -- maintenance margin tiers.
_MM_TIERS: list[tuple[float, float]] = [
    (100_000, 0.005),       # <= $100k: 0.5%
    (500_000, 0.01),        # <= $500k: 1.0%
    (1_000_000, 0.02),      # <= $1M:   2.0%
    (5_000_000, 0.03),      # <= $5M:   3.0%
    (float("inf"), 0.05),   # > $5M:    5.0%
]


def dynamic_maint_margin(notional_usd: float) -> float:
    """Look up the maintenance margin rate for a position size.

    Uses Hyperliquid's tiered maintenance margin schedule.

    Parameters
    ----------
    notional_usd:
        Absolute position notional value in USD.

    Returns
    -------
    float
        Maintenance margin fraction (e.g. 0.005 for 0.5%).
    """
    abs_notional = abs(notional_usd)
    for max_notional, mm_pct in _MM_TIERS:
        if abs_notional <= max_notional:
            return mm_pct
    return _MM_TIERS[-1][1]


# ---------------------------------------------------------------------------
# Leverage selection
# ---------------------------------------------------------------------------


def select_leverage(
    stop_distance_pct: float,
    liquidation_buffer_pct: float = 0.25,
    max_leverage: float = 20.0,
    volatility_pct: float | None = None,
    vol_haircut: float = 0.5,
) -> float:
    """Compute the appropriate leverage for a trade.

    Parameters
    ----------
    stop_distance_pct:
        Fractional distance from entry to stop (e.g. 0.02 for 2 %).
    liquidation_buffer_pct:
        Extra margin of safety between stop and liquidation price.
    max_leverage:
        Hard ceiling on leverage.
    volatility_pct:
        Recent annualised or rolling volatility as a fraction (e.g. 0.05).
        When provided, the base leverage is reduced via *vol_haircut*.
    vol_haircut:
        Fraction of leverage to remove at full volatility.  The haircut
        scales linearly from 0 at ``volatility_pct == 0`` up to
        *vol_haircut* at ``volatility_pct >= 0.10``.

    Returns
    -------
    float
        Leverage rounded **down** to the nearest 0.5, clamped to
        ``[1.0, max_leverage]``.
    """
    denominator = stop_distance_pct + liquidation_buffer_pct
    if denominator <= 0:
        logger.warning(
            "leverage_denominator_zero",
            stop_distance_pct=stop_distance_pct,
            liquidation_buffer_pct=liquidation_buffer_pct,
        )
        return 1.0

    lev = 1.0 / denominator

    # --- volatility haircut ---
    if volatility_pct is not None and volatility_pct > 0:
        vol_ratio = min(1.0, volatility_pct / 0.1)
        lev = lev * (1.0 - vol_haircut * vol_ratio)

    # --- clamp ---
    lev = max(1.0, min(lev, max_leverage))

    # --- round down to nearest 0.5 ---
    lev = math.floor(lev * 2.0) / 2.0

    # After rounding we might have gone below 1.0
    lev = max(1.0, lev)

    logger.debug(
        "leverage_selected",
        leverage=lev,
        stop_distance_pct=stop_distance_pct,
        volatility_pct=volatility_pct,
    )
    return lev


# ---------------------------------------------------------------------------
# Stop / liquidation helpers
# ---------------------------------------------------------------------------


def compute_stop_distance_pct(entry: float, stop: float) -> float:
    """Return the fractional distance between *entry* and *stop*.

    >>> compute_stop_distance_pct(100.0, 98.0)
    0.02
    """
    if entry <= 0:
        return 0.0
    return abs(entry - stop) / entry


def compute_liquidation_price(
    entry: float,
    leverage: float,
    is_long: bool,
    maint_margin_pct: float | None = None,
    notional_usd: float | None = None,
) -> float:
    """Estimate the liquidation price for a position.

    Parameters
    ----------
    entry:
        Entry price.
    leverage:
        Selected leverage.
    is_long:
        ``True`` for a long position, ``False`` for short.
    maint_margin_pct:
        Maintenance margin requirement as a fraction of position value.

    Returns
    -------
    float
        Estimated liquidation price.  Always >= 0 for longs.
    """
    if leverage <= 0:
        leverage = 1.0

    # Resolve maintenance margin: use dynamic schedule if not explicitly given
    if maint_margin_pct is None:
        if notional_usd is not None:
            maint_margin_pct = dynamic_maint_margin(notional_usd)
        else:
            maint_margin_pct = dynamic_maint_margin(entry * leverage)

    # HL liquidation derivation:
    #   Long: (liq - entry)*size + initial_margin - liq*size*mm = 0
    #         liq*(1 - mm) = entry*(1 - 1/lev)  =>  liq = entry*(1 - 1/lev)/(1 - mm)
    #   Short: (entry - liq)*size + initial_margin - liq*size*mm = 0
    #          liq*(1 + mm) = entry*(1 + 1/lev)  =>  liq = entry*(1 + 1/lev)/(1 + mm)
    if is_long:
        denom = 1.0 - maint_margin_pct
        if denom <= 0:
            return 0.0
        liq = entry * (1.0 - 1.0 / leverage) / denom
        return max(0.0, liq)
    else:
        denom = 1.0 + maint_margin_pct
        liq = entry * (1.0 + 1.0 / leverage) / denom
        return liq


def validate_stop_vs_liquidation(
    entry: float,
    stop: float,
    leverage: float,
    is_long: bool,
    buffer_pct: float = 0.25,
) -> tuple[bool, str]:
    """Check that the stop-loss sits safely away from the liquidation price.

    For **longs** the stop must be *above* ``liquidation_price * (1 + buffer_pct)``.
    For **shorts** the stop must be *below* ``liquidation_price * (1 - buffer_pct)``.

    Returns
    -------
    tuple[bool, str]
        ``(valid, explanation)`` where *valid* is ``True`` when the stop is
        safely positioned.
    """
    liq_px = compute_liquidation_price(entry, leverage, is_long)

    if is_long:
        threshold = liq_px * (1.0 + buffer_pct)
        if stop >= threshold:
            msg = (
                f"Stop {stop:.6f} is safely above liquidation buffer "
                f"{threshold:.6f} (liq={liq_px:.6f}, buffer={buffer_pct:.0%})"
            )
            return True, msg
        else:
            msg = (
                f"Stop {stop:.6f} is too close to liquidation price "
                f"{liq_px:.6f}; needs to be above {threshold:.6f} "
                f"(buffer={buffer_pct:.0%})"
            )
            logger.warning(
                "stop_too_close_to_liquidation",
                stop=stop,
                liq_px=liq_px,
                threshold=threshold,
                is_long=is_long,
            )
            return False, msg
    else:
        threshold = liq_px * (1.0 - buffer_pct)
        if stop <= threshold:
            msg = (
                f"Stop {stop:.6f} is safely below liquidation buffer "
                f"{threshold:.6f} (liq={liq_px:.6f}, buffer={buffer_pct:.0%})"
            )
            return True, msg
        else:
            msg = (
                f"Stop {stop:.6f} is too close to liquidation price "
                f"{liq_px:.6f}; needs to be below {threshold:.6f} "
                f"(buffer={buffer_pct:.0%})"
            )
            logger.warning(
                "stop_too_close_to_liquidation",
                stop=stop,
                liq_px=liq_px,
                threshold=threshold,
                is_long=is_long,
            )
            return False, msg
