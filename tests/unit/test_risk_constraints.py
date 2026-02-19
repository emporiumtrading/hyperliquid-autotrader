"""Tests for risk constraints: config defaults, loading, portfolio and trade checks."""

import pytest

from autotrader.risk.constraints import (
    RiskConfig,
    RiskState,
    check_portfolio,
    check_trade,
    load_risk_config,
)

# -----------------------------------------------------------------------
# RiskConfig defaults
# -----------------------------------------------------------------------


class TestRiskConfigDefaults:
    def test_risk_config_defaults(self):
        """Verify all default values on a freshly constructed RiskConfig."""
        cfg = RiskConfig()
        assert cfg.equity_risk_per_trade_min == 0.002
        assert cfg.equity_risk_per_trade_max == 0.01
        assert cfg.max_concurrent_positions == 4
        assert cfg.max_total_notional_multiple == 2.5
        assert cfg.daily_loss_limit_pct == 0.05
        assert cfg.weekly_loss_limit_pct == 0.12
        assert cfg.max_drawdown_pct == 0.18
        assert cfg.min_expected_rr == 1.8
        assert cfg.liquidation_buffer_pct == 0.25
        assert cfg.max_leverage == 20.0
        assert cfg.max_position_pct == 0.25

    def test_risk_config_is_frozen(self):
        """RiskConfig is frozen -- attributes cannot be reassigned."""
        cfg = RiskConfig()
        with pytest.raises(AttributeError):
            cfg.max_leverage = 50.0  # type: ignore[misc]


# -----------------------------------------------------------------------
# load_risk_config
# -----------------------------------------------------------------------


class TestLoadRiskConfig:
    def test_load_risk_config(self):
        """Parse a RiskConfig from a configuration dict with a 'risk' section."""
        cfg_dict = {
            "risk": {
                "max_leverage": 10.0,
                "max_concurrent_positions": 6,
                "daily_loss_limit_pct": 0.03,
            },
            "other_section": {"irrelevant": True},
        }
        config = load_risk_config(cfg_dict)

        assert isinstance(config, RiskConfig)
        assert config.max_leverage == 10.0
        assert config.max_concurrent_positions == 6
        assert config.daily_loss_limit_pct == 0.03
        # Fields not in the dict should use defaults
        assert config.max_drawdown_pct == 0.18
        assert config.liquidation_buffer_pct == 0.25

    def test_load_risk_config_empty(self):
        """An empty dict produces a RiskConfig with all defaults."""
        config = load_risk_config({})
        assert config == RiskConfig()

    def test_load_risk_config_ignores_unknown_keys(self):
        """Unknown keys in the risk section are silently ignored."""
        cfg_dict = {
            "risk": {
                "max_leverage": 15.0,
                "totally_unknown_key": "should_be_ignored",
            }
        }
        config = load_risk_config(cfg_dict)
        assert config.max_leverage == 15.0
        assert not hasattr(config, "totally_unknown_key")


# -----------------------------------------------------------------------
# check_portfolio
# -----------------------------------------------------------------------


class TestCheckPortfolio:
    def test_check_portfolio_passes(self):
        """A healthy portfolio state with no violations passes all checks."""
        config = RiskConfig()
        state = RiskState(
            equity=10_000.0,
            peak_equity=10_000.0,
            daily_pnl=100.0,  # positive daily PnL
            weekly_pnl=500.0,  # positive weekly PnL
            open_positions=2,  # below max of 4
            total_notional=5_000.0,  # below 2.5 * 10000 = 25000
        )
        passed, violations = check_portfolio(state, config)
        assert passed is True
        assert violations == []

    def test_check_portfolio_max_positions(self):
        """Fails when open_positions >= max_concurrent_positions."""
        config = RiskConfig(max_concurrent_positions=3)
        state = RiskState(
            equity=10_000.0,
            peak_equity=10_000.0,
            daily_pnl=0.0,
            weekly_pnl=0.0,
            open_positions=3,  # equals the limit
            total_notional=1_000.0,
        )
        passed, violations = check_portfolio(state, config)
        assert passed is False
        assert len(violations) >= 1
        assert any("positions" in v.lower() for v in violations)

    def test_check_portfolio_daily_loss(self):
        """Fails when daily loss exceeds the daily_loss_limit_pct."""
        config = RiskConfig(daily_loss_limit_pct=0.05)
        state = RiskState(
            equity=10_000.0,
            peak_equity=10_000.0,
            daily_pnl=-600.0,  # 6% of equity > 5% limit
            weekly_pnl=0.0,
            open_positions=1,
            total_notional=1_000.0,
        )
        passed, violations = check_portfolio(state, config)
        assert passed is False
        assert any("daily" in v.lower() for v in violations)

    def test_check_portfolio_drawdown(self):
        """Fails when current drawdown exceeds max_drawdown_pct."""
        config = RiskConfig(max_drawdown_pct=0.18)
        # Equity dropped from 10000 to 8000 => 20% drawdown > 18% limit
        state = RiskState(
            equity=8_000.0,
            peak_equity=10_000.0,
            daily_pnl=0.0,
            weekly_pnl=0.0,
            open_positions=1,
            total_notional=1_000.0,
        )
        passed, violations = check_portfolio(state, config)
        assert passed is False
        assert any("drawdown" in v.lower() for v in violations)

    def test_check_portfolio_weekly_loss(self):
        """Fails when weekly loss exceeds the weekly_loss_limit_pct."""
        config = RiskConfig(weekly_loss_limit_pct=0.12)
        state = RiskState(
            equity=10_000.0,
            peak_equity=10_000.0,
            daily_pnl=0.0,
            weekly_pnl=-1_500.0,  # 15% > 12% limit
            open_positions=1,
            total_notional=1_000.0,
        )
        passed, violations = check_portfolio(state, config)
        assert passed is False
        assert any("weekly" in v.lower() for v in violations)

    def test_check_portfolio_total_notional(self):
        """Fails when total notional exceeds max_total_notional_multiple * equity."""
        config = RiskConfig(max_total_notional_multiple=2.5)
        state = RiskState(
            equity=10_000.0,
            peak_equity=10_000.0,
            daily_pnl=0.0,
            weekly_pnl=0.0,
            open_positions=1,
            total_notional=30_000.0,  # 3x equity > 2.5x limit
        )
        passed, violations = check_portfolio(state, config)
        assert passed is False
        assert any("notional" in v.lower() for v in violations)


# -----------------------------------------------------------------------
# check_trade
# -----------------------------------------------------------------------


class TestCheckTrade:
    def test_check_trade_passes(self):
        """A valid trade with acceptable size, leverage, and portfolio impact passes."""
        config = RiskConfig()
        state = RiskState(
            equity=10_000.0,
            peak_equity=10_000.0,
            daily_pnl=0.0,
            weekly_pnl=0.0,
            open_positions=1,
            total_notional=5_000.0,
        )
        # size=2000 < 25% of 10000=2500, leverage=5 < 20, entry=100, stop=95
        passed, violations = check_trade(
            size_usd=2000.0,
            leverage=5.0,
            entry=100.0,
            stop=95.0,
            state=state,
            config=config,
        )
        assert passed is True
        assert violations == []

    def test_check_trade_size_limit(self):
        """Rejects a trade whose size exceeds max_position_pct * equity."""
        config = RiskConfig(max_position_pct=0.25)
        state = RiskState(equity=10_000.0, peak_equity=10_000.0)
        # 3000 > 25% of 10000 = 2500
        passed, violations = check_trade(
            size_usd=3_000.0,
            leverage=5.0,
            entry=100.0,
            stop=95.0,
            state=state,
            config=config,
        )
        assert passed is False
        assert any("position size" in v.lower() for v in violations)

    def test_check_trade_leverage_limit(self):
        """Rejects a trade whose leverage exceeds max_leverage."""
        config = RiskConfig(max_leverage=20.0)
        state = RiskState(equity=10_000.0, peak_equity=10_000.0)
        passed, violations = check_trade(
            size_usd=500.0,
            leverage=25.0,  # exceeds 20x
            entry=100.0,
            stop=95.0,
            state=state,
            config=config,
        )
        assert passed is False
        assert any("leverage" in v.lower() for v in violations)

    def test_check_trade_portfolio_impact(self):
        """Rejects a trade that would push portfolio over max positions."""
        config = RiskConfig(max_concurrent_positions=2)
        state = RiskState(
            equity=10_000.0,
            peak_equity=10_000.0,
            open_positions=1,  # adding one more -> 2 = limit
            total_notional=1_000.0,
        )
        # The projected state will have open_positions=2 >= max_concurrent_positions=2
        passed, violations = check_trade(
            size_usd=500.0,
            leverage=2.0,
            entry=100.0,
            stop=95.0,
            state=state,
            config=config,
        )
        assert passed is False
        assert any("positions" in v.lower() for v in violations)
