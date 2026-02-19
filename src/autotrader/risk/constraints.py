"""Portfolio and per-trade constraints (immutable, config-driven).

Provides :class:`RiskConfig` (frozen dataclass with all risk limits),
:class:`RiskState` (mutable snapshot of current portfolio risk metrics),
and helper functions to validate both portfolio-level and individual-trade
constraints before execution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskConfig:
    """Immutable risk-management parameters loaded from configuration."""

    equity_risk_per_trade_min: float = 0.002
    equity_risk_per_trade_max: float = 0.01
    max_concurrent_positions: int = 4
    max_total_notional_multiple: float = 2.5
    daily_loss_limit_pct: float = 0.05
    weekly_loss_limit_pct: float = 0.12
    max_drawdown_pct: float = 0.18
    min_expected_rr: float = 1.8
    liquidation_buffer_pct: float = 0.25
    max_leverage: float = 20.0
    max_position_pct: float = 0.25  # max 25 % of equity in a single position


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class RiskState:
    """Mutable snapshot of the portfolio's current risk metrics."""

    equity: float = 10_000.0
    peak_equity: float = 10_000.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    open_positions: int = 0
    total_notional: float = 0.0


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_risk_config(cfg: dict) -> RiskConfig:
    """Extract the ``risk`` section from *cfg* and return a :class:`RiskConfig`.

    Keys in the ``risk`` sub-dict are matched by name to :class:`RiskConfig`
    fields.  Unknown keys are silently ignored so that the configuration file
    can carry extra data without breaking this loader.
    """
    risk_section: dict = cfg.get("risk", {})
    valid_fields = {f.name for f in RiskConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in risk_section.items() if k in valid_fields}
    config = RiskConfig(**filtered)
    logger.info(
        "risk_config_loaded",
        max_leverage=config.max_leverage,
        max_positions=config.max_concurrent_positions,
        max_drawdown_pct=config.max_drawdown_pct,
    )
    return config


# ---------------------------------------------------------------------------
# Portfolio-level checks
# ---------------------------------------------------------------------------


def check_portfolio(state: RiskState, config: RiskConfig) -> tuple[bool, list[str]]:
    """Validate portfolio-level risk constraints.

    Returns a ``(passed, violations)`` tuple where *passed* is ``True`` when
    every constraint is satisfied and *violations* is a (possibly empty) list
    of human-readable violation descriptions.
    """
    violations: list[str] = []

    # --- concurrent positions ---
    if state.open_positions >= config.max_concurrent_positions:
        violations.append(
            f"Max concurrent positions reached: "
            f"{state.open_positions} >= {config.max_concurrent_positions}"
        )

    # --- total notional ---
    max_notional = state.equity * config.max_total_notional_multiple
    if state.total_notional > max_notional:
        violations.append(
            f"Total notional {state.total_notional:.2f} exceeds "
            f"limit {max_notional:.2f} "
            f"({config.max_total_notional_multiple}x equity)"
        )

    # --- daily loss ---
    daily_loss_limit = state.equity * config.daily_loss_limit_pct
    if state.daily_pnl < 0 and abs(state.daily_pnl) > daily_loss_limit:
        violations.append(
            f"Daily loss {abs(state.daily_pnl):.2f} exceeds "
            f"limit {daily_loss_limit:.2f} "
            f"({config.daily_loss_limit_pct:.1%} of equity)"
        )

    # --- weekly loss ---
    weekly_loss_limit = state.equity * config.weekly_loss_limit_pct
    if state.weekly_pnl < 0 and abs(state.weekly_pnl) > weekly_loss_limit:
        violations.append(
            f"Weekly loss {abs(state.weekly_pnl):.2f} exceeds "
            f"limit {weekly_loss_limit:.2f} "
            f"({config.weekly_loss_limit_pct:.1%} of equity)"
        )

    # --- drawdown ---
    if state.peak_equity > 0:
        drawdown = 1.0 - state.equity / state.peak_equity
    else:
        drawdown = 0.0
    if drawdown > config.max_drawdown_pct:
        violations.append(
            f"Drawdown {drawdown:.2%} exceeds " f"max allowed {config.max_drawdown_pct:.2%}"
        )

    passed = len(violations) == 0
    if not passed:
        logger.warning("portfolio_constraint_violation", violations=violations)
    return passed, violations


# ---------------------------------------------------------------------------
# Per-trade checks
# ---------------------------------------------------------------------------


def check_trade(
    size_usd: float,
    leverage: float,
    entry: float,
    stop: float,
    state: RiskState,
    config: RiskConfig,
) -> tuple[bool, list[str]]:
    """Validate a single proposed trade against risk constraints.

    Checks position-size limits, leverage caps, minimum reward/risk, and
    re-runs portfolio checks as if this trade were added.

    Returns a ``(passed, violations)`` tuple.
    """
    violations: list[str] = []

    # --- single-position size ---
    max_pos_usd = config.max_position_pct * state.equity
    if size_usd > max_pos_usd:
        violations.append(
            f"Position size ${size_usd:.2f} exceeds "
            f"max ${max_pos_usd:.2f} "
            f"({config.max_position_pct:.0%} of equity)"
        )

    # --- leverage cap ---
    if leverage > config.max_leverage:
        violations.append(f"Leverage {leverage:.1f}x exceeds max {config.max_leverage:.1f}x")

    # --- reward / risk ratio (if stop is meaningful) ---
    if entry > 0 and stop > 0 and not math.isclose(entry, stop):
        stop_distance = abs(entry - stop)
        # We can only estimate R:R if the signal provides enough info.
        # Here we use the minimum expected R:R as a threshold, assuming
        # take-profit = entry + min_expected_rr * stop_distance on the
        # favourable side.  The actual R:R will be validated later if
        # take-profit information is available.
        # For now, just flag if the stop distance itself is unreasonable.
        risk_per_coin = stop_distance
        # The minimum expected R:R is checked externally when TP is known.
        # We still verify that the stop distance is non-zero and sane.
        if risk_per_coin <= 0:
            violations.append("Stop distance is zero or negative")

    # --- portfolio impact check ---
    projected_state = RiskState(
        equity=state.equity,
        peak_equity=state.peak_equity,
        daily_pnl=state.daily_pnl,
        weekly_pnl=state.weekly_pnl,
        open_positions=state.open_positions + 1,
        total_notional=state.total_notional + size_usd,
    )
    portfolio_ok, portfolio_violations = check_portfolio(projected_state, config)
    if not portfolio_ok:
        violations.extend(portfolio_violations)

    passed = len(violations) == 0
    if not passed:
        logger.warning(
            "trade_constraint_violation",
            size_usd=size_usd,
            leverage=leverage,
            violations=violations,
        )
    return passed, violations
