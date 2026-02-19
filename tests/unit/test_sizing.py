"""Tests for autotrader.risk.sizing -- position sizing, Kelly, regime scaling."""

from __future__ import annotations

import pytest

from autotrader.risk.sizing import (
    compute_position_size,
    kelly_fraction,
    scale_risk_by_regime,
)


class TestComputePositionSize:
    def test_compute_position_size_basic(self):
        """Basic sizing: risk_amount / stop_distance, capped by max_position."""
        # equity=10000, risk_per_trade=0.01 => risk_amount=100
        # entry=100, stop=98 => stop_distance_pct = 0.02
        # position_notional = 100 / 0.02 = 5000
        # max_notional = 10000 * 0.25 * 1.0 = 2500
        # final = min(5000, 2500) = 2500
        # size_coins = 2500 / 100 = 25
        coins, notional = compute_position_size(
            equity=10_000.0,
            risk_per_trade_pct=0.01,
            entry=100.0,
            stop=98.0,
            leverage=1.0,
            max_position_pct=0.25,
        )
        assert pytest.approx(notional, abs=0.01) == 2500.0
        assert pytest.approx(coins, abs=0.01) == 25.0

    def test_compute_position_size_capped(self):
        """Position is capped by max_position_pct of equity * leverage."""
        # equity=10000, risk=0.05, entry=100, stop=99.5 => stop_dist=0.005
        # risk_amount = 500, position_notional = 500 / 0.005 = 100000
        # max_notional = 10000 * 0.10 * 2.0 = 2000  (with small max_pos and leverage)
        # final = 2000
        coins, notional = compute_position_size(
            equity=10_000.0,
            risk_per_trade_pct=0.05,
            entry=100.0,
            stop=99.5,
            leverage=2.0,
            max_position_pct=0.10,
        )
        assert notional == pytest.approx(2000.0, abs=0.01)
        assert coins == pytest.approx(20.0, abs=0.01)

    def test_compute_position_size_zero_stop(self):
        """Zero stop distance (entry == stop) returns (0, 0)."""
        coins, notional = compute_position_size(
            equity=10_000.0,
            risk_per_trade_pct=0.01,
            entry=100.0,
            stop=100.0,
            leverage=1.0,
            max_position_pct=0.25,
        )
        assert coins == 0.0
        assert notional == 0.0


class TestKellyFraction:
    def test_kelly_fraction_positive(self):
        """Good stats should give positive Kelly fraction."""
        # win_rate=0.6, avg_win=2.0, avg_loss=1.0
        # f = (0.6*2 - 0.4*1) / 2 = (1.2 - 0.4)/2 = 0.4
        # clamped to 0.25
        f = kelly_fraction(win_rate=0.6, avg_win=2.0, avg_loss=1.0)
        assert f > 0.0

    def test_kelly_fraction_capped(self):
        """Kelly fraction is capped at 0.25."""
        # Very favorable stats that would yield f > 0.25
        f = kelly_fraction(win_rate=0.9, avg_win=5.0, avg_loss=0.5)
        assert f == 0.25

    def test_kelly_fraction_negative(self):
        """Losing stats should return 0."""
        # win_rate=0.2, avg_win=1.0, avg_loss=3.0
        # f = (0.2*1 - 0.8*3) / 1 = (0.2 - 2.4)/1 = -2.2 => clamped to 0
        f = kelly_fraction(win_rate=0.2, avg_win=1.0, avg_loss=3.0)
        assert f == 0.0


class TestScaleRiskByRegime:
    def test_scale_risk_trend(self):
        """TREND regime with full confidence gets the full base risk (multiplier=1.0)."""
        result = scale_risk_by_regime(
            base_risk_pct=0.01,
            regime="TREND",
            regime_confidence=1.0,
        )
        # multiplier=1.0, confidence_scale = 0.5 + 0.5*1.0 = 1.0
        # result = 0.01 * 1.0 * 1.0 = 0.01
        assert pytest.approx(result, abs=1e-9) == 0.01

    def test_scale_risk_unknown(self):
        """UNKNOWN regime gets reduced risk (multiplier=0.2)."""
        result = scale_risk_by_regime(
            base_risk_pct=0.01,
            regime="UNKNOWN",
            regime_confidence=1.0,
        )
        # multiplier=0.2, confidence_scale = 1.0
        # result = 0.01 * 0.2 * 1.0 = 0.002
        assert pytest.approx(result, abs=1e-9) == 0.002
