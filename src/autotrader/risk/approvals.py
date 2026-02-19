"""Pre-trade risk approval gate.

Orchestrates the full approval pipeline: portfolio checks, leverage
selection, liquidation validation, regime-scaled sizing, and per-trade
constraint checks.  Returns a single decision dict consumed by the
execution layer.
"""

from __future__ import annotations

import structlog

from autotrader.hl.types import Signal
from autotrader.risk.constraints import (
    RiskConfig,
    RiskState,
    check_portfolio,
    check_trade,
)
from autotrader.risk.leverage import (
    compute_stop_distance_pct,
    select_leverage,
    validate_stop_vs_liquidation,
)
from autotrader.risk.sizing import (
    compute_position_size,
    scale_risk_by_regime,
)

logger = structlog.get_logger(__name__)


def approve_trade(
    signal: Signal,
    risk_state: RiskState,
    risk_config: RiskConfig,
    regime: str = "UNKNOWN",
    regime_confidence: float = 0.0,
    volatility_pct: float | None = None,
) -> dict:
    """Run the full pre-trade approval pipeline.

    Parameters
    ----------
    signal:
        The trading signal produced by a strategy.
    risk_state:
        Current portfolio risk snapshot.
    risk_config:
        Immutable risk-management parameters.
    regime:
        Detected market regime label.
    regime_confidence:
        Confidence in the regime classification (0-1).
    volatility_pct:
        Recent volatility as a decimal fraction.

    Returns
    -------
    dict
        Decision payload with keys:

        - ``approved`` (bool) -- whether the trade passed all gates.
        - ``reasons`` (list[str]) -- rejection/warning messages.
        - ``leverage`` (float) -- selected leverage (0.0 if rejected).
        - ``size_coins`` (float) -- position size in coins.
        - ``size_usd`` (float) -- position notional in USD.
        - ``risk_pct`` (float) -- regime-scaled risk fraction.
        - ``stop_distance_pct`` (float) -- fractional stop distance.
        - ``expected_rr`` (float) -- estimated reward-to-risk ratio.
    """
    reasons: list[str] = []

    # Defaults for the result dict (returned on early rejection).
    result: dict = {
        "approved": False,
        "reasons": reasons,
        "leverage": 0.0,
        "size_coins": 0.0,
        "size_usd": 0.0,
        "risk_pct": 0.0,
        "stop_distance_pct": 0.0,
        "expected_rr": 0.0,
    }

    # ------------------------------------------------------------------
    # 1. Signal must not be flat / must have required fields
    # ------------------------------------------------------------------
    if signal.side not in ("buy", "sell"):
        reasons.append(f"Signal side '{signal.side}' is not actionable")
        logger.info("trade_rejected", reasons=reasons)
        return result

    if signal.entry is None or signal.entry <= 0:
        reasons.append("Signal has no valid entry price")
        logger.info("trade_rejected", reasons=reasons)
        return result

    if signal.stop is None or signal.stop <= 0:
        reasons.append("Signal has no valid stop price")
        logger.info("trade_rejected", reasons=reasons)
        return result

    entry = signal.entry
    stop = signal.stop
    is_long = signal.side == "buy"

    # ------------------------------------------------------------------
    # 2. Portfolio-level constraints
    # ------------------------------------------------------------------
    portfolio_ok, portfolio_violations = check_portfolio(risk_state, risk_config)
    if not portfolio_ok:
        reasons.extend(portfolio_violations)
        logger.info("trade_rejected_portfolio", reasons=reasons)
        return result

    # ------------------------------------------------------------------
    # 3. Compute stop distance
    # ------------------------------------------------------------------
    stop_distance_pct = compute_stop_distance_pct(entry, stop)
    if stop_distance_pct <= 0:
        reasons.append("Stop distance is zero or negative")
        logger.info("trade_rejected", reasons=reasons)
        return result
    result["stop_distance_pct"] = stop_distance_pct

    # ------------------------------------------------------------------
    # 4. Select leverage
    # ------------------------------------------------------------------
    leverage = select_leverage(
        stop_distance_pct=stop_distance_pct,
        liquidation_buffer_pct=risk_config.liquidation_buffer_pct,
        max_leverage=risk_config.max_leverage,
        volatility_pct=volatility_pct,
    )
    result["leverage"] = leverage

    # ------------------------------------------------------------------
    # 5. Validate stop vs liquidation
    # ------------------------------------------------------------------
    stop_valid, stop_msg = validate_stop_vs_liquidation(
        entry=entry,
        stop=stop,
        leverage=leverage,
        is_long=is_long,
        buffer_pct=risk_config.liquidation_buffer_pct,
    )
    if not stop_valid:
        reasons.append(stop_msg)
        logger.info("trade_rejected_liquidation", reasons=reasons)
        return result

    # ------------------------------------------------------------------
    # 6. Compute risk per trade (scaled by regime)
    # ------------------------------------------------------------------
    base_risk_pct = risk_config.equity_risk_per_trade_max
    risk_pct = scale_risk_by_regime(base_risk_pct, regime, regime_confidence)
    # Clamp within configured min/max bounds
    risk_pct = max(risk_config.equity_risk_per_trade_min, risk_pct)
    risk_pct = min(risk_config.equity_risk_per_trade_max, risk_pct)
    result["risk_pct"] = risk_pct

    # ------------------------------------------------------------------
    # 7. Compute position size
    # ------------------------------------------------------------------
    size_coins, size_usd = compute_position_size(
        equity=risk_state.equity,
        risk_per_trade_pct=risk_pct,
        entry=entry,
        stop=stop,
        leverage=leverage,
        max_position_pct=risk_config.max_position_pct,
    )
    if size_coins <= 0 or size_usd <= 0:
        reasons.append("Computed position size is zero")
        logger.info("trade_rejected_sizing", reasons=reasons)
        return result
    result["size_coins"] = size_coins
    result["size_usd"] = size_usd

    # ------------------------------------------------------------------
    # 7b. Compute expected reward:risk if take_profit is available
    # ------------------------------------------------------------------
    expected_rr = 0.0
    if signal.take_profit is not None and signal.take_profit > 0:
        reward_distance = abs(signal.take_profit - entry)
        risk_distance = abs(entry - stop)
        if risk_distance > 0:
            expected_rr = reward_distance / risk_distance
    result["expected_rr"] = expected_rr

    # Check minimum R:R if we could compute it
    if expected_rr > 0 and expected_rr < risk_config.min_expected_rr:
        reasons.append(
            f"Expected R:R {expected_rr:.2f} below minimum " f"{risk_config.min_expected_rr:.2f}"
        )

    # ------------------------------------------------------------------
    # 8. Per-trade constraint check
    # ------------------------------------------------------------------
    trade_ok, trade_violations = check_trade(
        size_usd=size_usd,
        leverage=leverage,
        entry=entry,
        stop=stop,
        state=risk_state,
        config=risk_config,
    )
    if not trade_ok:
        reasons.extend(trade_violations)

    # ------------------------------------------------------------------
    # 9. Final decision
    # ------------------------------------------------------------------
    approved = len(reasons) == 0
    result["approved"] = approved
    result["reasons"] = reasons

    if approved:
        logger.info(
            "trade_approved",
            side=signal.side,
            entry=entry,
            stop=stop,
            leverage=leverage,
            size_coins=size_coins,
            size_usd=size_usd,
            risk_pct=risk_pct,
            expected_rr=expected_rr,
            regime=regime,
        )
    else:
        logger.info(
            "trade_rejected",
            side=signal.side,
            entry=entry,
            stop=stop,
            reasons=reasons,
        )

    return result
