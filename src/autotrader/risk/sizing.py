"""Position sizing.

Computes the notional and coin size for a trade using risk-per-trade
parameters, applies Kelly criterion scaling, and adjusts risk budgets
based on the detected market regime.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Regime multipliers
# ---------------------------------------------------------------------------

_REGIME_MULTIPLIERS: dict[str, float] = {
    "TREND": 1.0,
    "RANGE": 0.7,
    "VOLATILE_BREAKOUT": 0.8,
    "SQUEEZE_RISK": 0.3,
    "MEAN_REVERT_CRASH": 0.5,
    "UNKNOWN": 0.2,
}


# ---------------------------------------------------------------------------
# Core position sizing
# ---------------------------------------------------------------------------


def compute_position_size(
    equity: float,
    risk_per_trade_pct: float,
    entry: float,
    stop: float,
    leverage: float = 1.0,
    max_position_pct: float = 0.25,
) -> tuple[float, float]:
    """Compute position size in coins and notional USD.

    Parameters
    ----------
    equity:
        Account equity in USD.
    risk_per_trade_pct:
        Fraction of equity to risk (e.g. 0.01 for 1 %).
    entry:
        Intended entry price.
    stop:
        Stop-loss price.
    leverage:
        Selected leverage multiplier (>= 1.0).
    max_position_pct:
        Maximum fraction of equity allowed in a single position (before
        leverage).

    Returns
    -------
    tuple[float, float]
        ``(size_in_coins, final_notional_usd)``.
    """
    if entry <= 0:
        logger.warning("sizing_invalid_entry", entry=entry)
        return 0.0, 0.0

    stop_distance_pct = abs(entry - stop) / entry
    if stop_distance_pct <= 0:
        logger.warning("sizing_zero_stop_distance", entry=entry, stop=stop)
        return 0.0, 0.0

    risk_amount = equity * risk_per_trade_pct
    position_notional = risk_amount / stop_distance_pct
    max_notional = equity * max_position_pct * leverage
    final_notional = min(position_notional, max_notional)
    size_in_coins = final_notional / entry

    logger.debug(
        "position_sized",
        size_coins=size_in_coins,
        notional_usd=final_notional,
        risk_amount=risk_amount,
        stop_distance_pct=stop_distance_pct,
    )
    return size_in_coins, final_notional


# ---------------------------------------------------------------------------
# Kelly criterion
# ---------------------------------------------------------------------------


def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """Compute the Kelly fraction for position sizing.

    Uses the standard Kelly criterion formula::

        f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win

    The result is clamped to ``[0, 0.25]`` (quarter-Kelly maximum) to
    limit risk of over-sizing in practice.

    Parameters
    ----------
    win_rate:
        Historical win probability in ``[0, 1]``.
    avg_win:
        Average winning trade return (positive number).
    avg_loss:
        Average losing trade return (positive number representing loss
        magnitude).

    Returns
    -------
    float
        Kelly fraction in ``[0, 0.25]``.
    """
    if avg_win <= 0:
        return 0.0

    f = (win_rate * avg_win - (1.0 - win_rate) * avg_loss) / avg_win
    f = max(0.0, min(f, 0.25))

    logger.debug(
        "kelly_fraction_computed",
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        kelly_f=f,
    )
    return f


# ---------------------------------------------------------------------------
# Regime-based risk scaling
# ---------------------------------------------------------------------------


def scale_risk_by_regime(
    base_risk_pct: float,
    regime: str,
    regime_confidence: float,
) -> float:
    """Scale the per-trade risk budget based on detected market regime.

    Parameters
    ----------
    base_risk_pct:
        Base risk percentage (e.g. 0.01 for 1 %).
    regime:
        Current market regime label (e.g. ``"TREND"``, ``"RANGE"``).
    regime_confidence:
        Confidence in the regime classification in ``[0, 1]``.

    Returns
    -------
    float
        Adjusted risk percentage.
    """
    multiplier = _REGIME_MULTIPLIERS.get(regime.upper(), _REGIME_MULTIPLIERS["UNKNOWN"])
    # Scale by confidence: at confidence=0 we use half the multiplier,
    # at confidence=1 we use the full multiplier.
    confidence_scale = 0.5 + 0.5 * max(0.0, min(1.0, regime_confidence))
    result = base_risk_pct * multiplier * confidence_scale

    logger.debug(
        "risk_scaled_by_regime",
        base_risk_pct=base_risk_pct,
        regime=regime,
        regime_confidence=regime_confidence,
        multiplier=multiplier,
        result=result,
    )
    return result
