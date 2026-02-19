"""Tests for autotrader.backtest.cost_model -- fee, slippage, and funding calculations."""

from __future__ import annotations

import pytest

from autotrader.backtest.cost_model import CostConfig, CostModel


@pytest.fixture
def model() -> CostModel:
    """Return a CostModel with default Hyperliquid parameters."""
    return CostModel()


@pytest.fixture
def config() -> CostConfig:
    """Return a default CostConfig for direct inspection."""
    return CostConfig()


class TestEntryCost:
    def test_entry_cost_maker(self, model: CostModel, config: CostConfig):
        """Maker entry fee = notional * maker_fee_bps / 10_000."""
        notional = 100_000.0
        cost = model.compute_entry_cost(notional, is_maker=True)
        expected = notional * config.maker_fee_bps / 10_000.0
        assert pytest.approx(cost, abs=1e-6) == expected
        assert cost > 0

    def test_entry_cost_taker(self, model: CostModel, config: CostConfig):
        """Taker entry fee = notional * taker_fee_bps / 10_000."""
        notional = 100_000.0
        cost = model.compute_entry_cost(notional, is_maker=False)
        expected = notional * config.taker_fee_bps / 10_000.0
        assert pytest.approx(cost, abs=1e-6) == expected
        # Taker is more expensive than maker
        maker_cost = model.compute_entry_cost(notional, is_maker=True)
        assert cost > maker_cost


class TestSlippage:
    def test_slippage_increases_with_size(self, model: CostModel):
        """Larger notional sizes should incur more slippage."""
        small = model.compute_slippage(10_000.0, spread_bps=1.0)
        large = model.compute_slippage(500_000.0, spread_bps=1.0)
        assert large > small
        # Both should be positive
        assert small > 0
        assert large > 0

        # The per-dollar slippage should also increase (size-dependent component)
        per_dollar_small = small / 10_000.0
        per_dollar_large = large / 500_000.0
        assert per_dollar_large > per_dollar_small


class TestFundingCost:
    def test_funding_cost_positive(self, model: CostModel):
        """Positive funding rate with a long position = paying funding."""
        # notional=100000, rate=0.0001 per hour, held 24 hours
        # funding_interval_hours=1.0 => 24 payments
        # cost = 24 * 100000 * 0.0001 = 240
        cost = model.compute_funding_cost(notional=100_000.0, funding_rate=0.0001, hours_held=24.0)
        assert pytest.approx(cost, abs=1e-6) == 240.0
        assert cost > 0

    def test_funding_cost_negative(self, model: CostModel):
        """Negative notional (short) with positive funding rate = receiving funding."""
        cost = model.compute_funding_cost(notional=-100_000.0, funding_rate=0.0001, hours_held=24.0)
        # -100000 * 0.0001 * 24 = -240 (received)
        assert pytest.approx(cost, abs=1e-6) == -240.0
        assert cost < 0


class TestTotalTradeCost:
    def test_total_trade_cost(self, model: CostModel):
        """All components should add up to the total."""
        notional = 50_000.0
        funding_rate = 0.00005
        hours_held = 8.0
        spread_bps = 1.5

        result = model.total_trade_cost(
            notional=notional,
            funding_rate=funding_rate,
            hours_held=hours_held,
            spread_bps=spread_bps,
        )

        # Verify it's a dict with required keys
        assert set(result.keys()) == {"entry_fee", "exit_fee", "slippage", "funding", "total"}

        # Verify sum
        component_sum = (
            result["entry_fee"] + result["exit_fee"] + result["slippage"] + result["funding"]
        )
        assert pytest.approx(result["total"], abs=1e-6) == component_sum

        # All fee components positive
        assert result["entry_fee"] > 0
        assert result["exit_fee"] > 0
        assert result["slippage"] > 0
