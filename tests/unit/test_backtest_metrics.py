"""Tests for autotrader.backtest.metrics -- metric calculations from equity curves and trades."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autotrader.backtest.metrics import (
    BacktestMetrics,
    compute_drawdown_series,
    compute_metrics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_equity_curve(start: float = 10_000.0, n: int = 100, growth: float = 0.002):
    """Generate a simple growing equity curve with some noise."""
    rng = np.random.RandomState(7)
    returns = growth + rng.randn(n) * 0.005
    prices = start * np.cumprod(1 + returns)
    prices = np.insert(prices, 0, start)  # prepend initial equity
    return pd.Series(prices, dtype=float)


def _simple_trades(n: int = 20):
    """Generate a DataFrame of n synthetic trades with realistic columns."""
    rng = np.random.RandomState(42)
    pnl = rng.normal(50, 100, size=n)  # some winners, some losers
    sides = rng.choice(["long", "short"], size=n)
    fees = np.abs(rng.uniform(1, 5, size=n))
    slippage = np.abs(rng.uniform(0.5, 2, size=n))
    funding = np.abs(rng.uniform(0, 1, size=n))
    holding = rng.randint(1, 50, size=n)

    return pd.DataFrame(
        {
            "entry_time": range(n),
            "exit_time": range(n, 2 * n),
            "side": sides,
            "entry_px": 100.0,
            "exit_px": 100.0 + pnl / 10,
            "size": 10.0,
            "pnl": pnl,
            "fees": fees,
            "slippage": slippage,
            "funding": funding,
            "holding_bars": holding,
        }
    )


# ===========================================================================
# Tests
# ===========================================================================


class TestComputeMetricsBasic:
    def test_compute_metrics_basic(self):
        """Basic metrics from a simple equity curve and trades."""
        equity = _simple_equity_curve()
        trades = _simple_trades()
        m = compute_metrics(equity, trades)

        assert isinstance(m, BacktestMetrics)
        assert m.total_return != 0.0
        assert m.total_trades == 20
        assert m.long_trades + m.short_trades == 20
        assert 0.0 <= m.win_rate <= 1.0
        assert m.max_drawdown >= 0.0
        assert m.total_fees > 0.0
        assert m.total_slippage > 0.0


class TestDrawdownSeries:
    def test_drawdown_series(self):
        """Drawdown should be 0 at peaks and negative during declines."""
        equity = pd.Series([100.0, 110.0, 105.0, 115.0, 100.0])
        dd = compute_drawdown_series(equity)

        assert len(dd) == len(equity)
        # At index 0 (initial), running max = 100, dd = 0
        assert dd.iloc[0] == 0.0
        # At index 1 (peak 110), dd = 0
        assert dd.iloc[1] == 0.0
        # At index 2 (105, down from 110), dd = (105-110)/110 = -0.04545...
        assert dd.iloc[2] < 0.0
        assert pytest.approx(dd.iloc[2], abs=1e-4) == (105.0 - 110.0) / 110.0
        # At index 3 (new peak 115), dd = 0
        assert dd.iloc[3] == 0.0
        # At index 4 (100, down from 115), dd = (100-115)/115
        assert dd.iloc[4] < 0.0
        assert pytest.approx(dd.iloc[4], abs=1e-4) == (100.0 - 115.0) / 115.0


class TestSharpePositive:
    def test_sharpe_ratio_positive(self):
        """Positive returns should give a positive Sharpe ratio."""
        # Strongly positive equity curve
        equity = pd.Series([10000.0 + i * 50 for i in range(200)], dtype=float)
        trades = _simple_trades()
        m = compute_metrics(equity, trades)
        assert m.sharpe_ratio > 0.0


class TestNoTrades:
    def test_no_trades(self):
        """Empty trades DataFrame should not crash; metrics still populated from equity."""
        equity = _simple_equity_curve()
        empty_trades = pd.DataFrame(
            columns=[
                "entry_time",
                "exit_time",
                "side",
                "entry_px",
                "exit_px",
                "size",
                "pnl",
                "fees",
                "slippage",
                "funding",
                "holding_bars",
            ]
        )
        m = compute_metrics(equity, empty_trades)
        assert m.total_trades == 0
        assert m.total_return != 0.0  # equity curve still changes
        assert m.max_drawdown >= 0.0


class TestUtilityFormula:
    def test_utility_formula(self):
        """Utility = CAGR - lambda*DD - mu*CVaR - nu*turnover."""
        equity = _simple_equity_curve(n=200)
        trades = _simple_trades(n=30)

        lambda_dd = 1.0
        mu_cvar = 0.5
        nu_turnover = 0.1

        m = compute_metrics(
            equity,
            trades,
            config={
                "lambda_dd": lambda_dd,
                "mu_cvar": mu_cvar,
                "nu_turnover": nu_turnover,
            },
        )

        # Manually recompute utility from the metrics fields
        initial_equity = equity.iloc[0]
        turnover_cost = (m.total_fees + m.total_slippage) / max(initial_equity, 1e-12)
        expected_utility = (
            m.cagr - lambda_dd * m.max_drawdown - mu_cvar * m.cvar_95 - nu_turnover * turnover_cost
        )
        assert pytest.approx(m.utility, abs=1e-6) == expected_utility
