"""Tests for autotrader.risk.leverage -- leverage selection and liquidation helpers."""

from __future__ import annotations

import pytest

from autotrader.risk.leverage import (
    compute_liquidation_price,
    compute_stop_distance_pct,
    select_leverage,
    validate_stop_vs_liquidation,
)


class TestSelectLeverage:
    def test_leverage_from_stop(self):
        """Basic leverage computation: lev = 1 / (stop_distance + buffer)."""
        # stop_distance_pct=0.02, buffer=0.25 => denominator=0.27 => lev=3.7037
        # Rounded down to nearest 0.5 => 3.5
        lev = select_leverage(
            stop_distance_pct=0.02,
            liquidation_buffer_pct=0.25,
            max_leverage=20.0,
        )
        assert lev == 3.5

    def test_leverage_max_clamp(self):
        """Leverage is clamped to max_leverage."""
        # Very small stop distance => huge raw leverage, must be clamped
        lev = select_leverage(
            stop_distance_pct=0.001,
            liquidation_buffer_pct=0.001,
            max_leverage=5.0,
        )
        assert lev == 5.0

    def test_leverage_min_clamp(self):
        """Leverage should never go below 1.0."""
        # Very large stop distance => raw leverage < 1
        lev = select_leverage(
            stop_distance_pct=0.9,
            liquidation_buffer_pct=0.25,
            max_leverage=20.0,
        )
        assert lev >= 1.0

    def test_leverage_with_volatility(self):
        """Volatility haircut should reduce leverage compared to no-vol."""
        lev_no_vol = select_leverage(
            stop_distance_pct=0.02,
            liquidation_buffer_pct=0.25,
            max_leverage=20.0,
            volatility_pct=None,
        )
        lev_with_vol = select_leverage(
            stop_distance_pct=0.02,
            liquidation_buffer_pct=0.25,
            max_leverage=20.0,
            volatility_pct=0.08,
            vol_haircut=0.5,
        )
        assert lev_with_vol <= lev_no_vol
        # With 8% vol, vol_ratio = 0.08/0.1 = 0.8
        # lev *= (1 - 0.5*0.8) = 0.6 => 3.7037*0.6 = 2.222, floor => 2.0
        assert lev_with_vol == 2.0


class TestStopDistance:
    def test_stop_distance(self):
        """compute_stop_distance_pct(100, 98) should return 0.02."""
        result = compute_stop_distance_pct(100.0, 98.0)
        assert pytest.approx(result, abs=1e-9) == 0.02

        # Works for shorts too (stop above entry)
        result_short = compute_stop_distance_pct(100.0, 102.0)
        assert pytest.approx(result_short, abs=1e-9) == 0.02


class TestLiquidationPrice:
    def test_liquidation_price_long(self):
        """Correct liquidation price for a long position.

        HL formula: liq = entry * (1 - 1/lev) / (1 - mm)
        entry=100, leverage=10, maint=0.005
        => liq = 100 * 0.9 / 0.995 = 90.45226...
        """
        liq = compute_liquidation_price(
            entry=100.0, leverage=10.0, is_long=True, maint_margin_pct=0.005
        )
        expected = 100.0 * (1.0 - 1.0 / 10.0) / (1.0 - 0.005)
        assert pytest.approx(liq, abs=1e-6) == expected

    def test_liquidation_price_short(self):
        """Correct liquidation price for a short position.

        HL formula: liq = entry * (1 + 1/lev) / (1 + mm)
        entry=100, leverage=10, maint=0.005
        => liq = 100 * 1.1 / 1.005 = 109.45274...
        """
        liq = compute_liquidation_price(
            entry=100.0, leverage=10.0, is_long=False, maint_margin_pct=0.005
        )
        expected = 100.0 * (1.0 + 1.0 / 10.0) / (1.0 + 0.005)
        assert pytest.approx(liq, abs=1e-6) == expected


class TestValidateStopVsLiquidation:
    def test_validate_stop_vs_liquidation_valid(self):
        """A stop well above liquidation price should pass for longs."""
        # entry=100, leverage=3, is_long=True
        # liq = 100*(1 - 1/3) / (1 - 0.005) = 100*0.6667/0.995 = 66.99
        # threshold = 66.99 * 1.25 = 83.74
        # stop=95 >= 83.74 => should PASS
        valid, msg = validate_stop_vs_liquidation(
            entry=100.0,
            stop=95.0,
            leverage=3.0,
            is_long=True,
            buffer_pct=0.25,
        )
        assert valid is True
        assert "safely" in msg.lower()

    def test_validate_stop_vs_liquidation_invalid(self):
        """A stop too close to liquidation should fail."""
        # entry=100, leverage=5, is_long=True
        # liq = 100*(1 - 0.2) / (1 - 0.005) = 80.40
        # threshold = 80.40 * 1.25 = 100.50
        # stop=82 < 100.50 => FAIL
        valid, msg = validate_stop_vs_liquidation(
            entry=100.0,
            stop=82.0,
            leverage=5.0,
            is_long=True,
            buffer_pct=0.25,
        )
        assert valid is False
        assert "too close" in msg.lower()
