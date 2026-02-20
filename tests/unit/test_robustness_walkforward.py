"""Tests for robustness and walk-forward testing modules."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from autotrader.backtest.robustness import (
    RobustnessTest,
    _get_numeric_params,
    _perturb_params,
    _set_numeric_params,
)
from autotrader.backtest.walkforward import (
    WalkForwardConfig,
    WalkForwardTest,
    _chain_equity_curves,
)
from autotrader.strategies.trend_breakout import TrendBreakoutStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trending_candles(n: int = 500, start: float = 100.0) -> pd.DataFrame:
    """Generate trending OHLCV data for backtest testing."""
    rng = np.random.default_rng(42)
    trend = np.linspace(0, 20, n) + np.cumsum(rng.normal(0, 0.3, n))
    close = start + trend
    open_ = np.roll(close, 1)
    open_[0] = start
    high = np.maximum(open_, close) + rng.uniform(0.5, 2.0, n)
    low = np.minimum(open_, close) - rng.uniform(0.5, 2.0, n)
    volume = rng.uniform(500, 5000, n)

    return pd.DataFrame(
        {
            "timestamp_ms": [1_000_000 + i * 900_000 for i in range(n)],  # 15m bars
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _backtest_config() -> dict:
    return {
        "initial_equity": 10_000.0,
        "warmup_bars": 50,
        "risk": {"max_leverage": 5.0, "max_concurrent_positions": 2},
    }


# ---------------------------------------------------------------------------
# Parameter perturbation helpers
# ---------------------------------------------------------------------------


class TestGetNumericParams:
    def test_extracts_float_params(self):
        strategy = TrendBreakoutStrategy()
        params = _get_numeric_params(strategy)
        assert len(params) > 0
        assert all(isinstance(v, float) for v in params.values())

    def test_skips_private_and_name(self):
        strategy = TrendBreakoutStrategy()
        params = _get_numeric_params(strategy)
        assert "name" not in params
        assert all(not k.startswith("_") for k in params)


class TestPerturbParams:
    def test_perturbed_values_differ(self):
        params = {"a": 10.0, "b": 20.0, "c": 30.0}
        rng = np.random.default_rng(42)
        perturbed = _perturb_params(params, 0.2, rng)
        assert set(perturbed.keys()) == set(params.keys())
        # At least one value should differ
        assert any(abs(perturbed[k] - params[k]) > 1e-6 for k in params)

    def test_positive_values_stay_positive(self):
        params = {"x": 0.01}  # small positive
        rng = np.random.default_rng(99)
        perturbed = _perturb_params(params, 0.2, rng)
        assert perturbed["x"] > 0


class TestSetNumericParams:
    def test_set_params(self):
        strategy = TrendBreakoutStrategy()
        original = _get_numeric_params(strategy)
        modified = {k: v * 1.5 for k, v in original.items()}
        _set_numeric_params(strategy, modified)
        for k, v in modified.items():
            actual = getattr(strategy, k)
            if isinstance(actual, int):
                assert actual == max(1, int(round(v)))
            else:
                assert abs(actual - v) < 1e-6


# ---------------------------------------------------------------------------
# RobustnessTest.monte_carlo_equity
# ---------------------------------------------------------------------------


class TestMonteCarloEquity:
    def test_monte_carlo_returns_structure(self):
        rt = RobustnessTest({"n_monte_carlo": 10, "mc_max_dd_pct": 0.50})
        trades = pd.DataFrame({"pnl": [50.0, -20.0, 30.0, -10.0, 40.0, -5.0, 25.0]})
        result = rt.monte_carlo_equity(trades)
        assert "simulations" in result
        assert "median_dd" in result
        assert "p95_dd" in result
        assert "p99_dd" in result
        assert "passed" in result
        assert len(result["simulations"]) == 10

    def test_monte_carlo_passed_with_low_dd(self):
        """Profitable trades with small losses should pass."""
        rt = RobustnessTest({"n_monte_carlo": 20, "mc_max_dd_pct": 0.50})
        trades = pd.DataFrame({"pnl": [100.0, 50.0, -10.0, 80.0, -5.0] * 3})
        result = rt.monte_carlo_equity(trades)
        assert result["passed"] is True
        assert result["p95_dd"] < 0.50

    def test_monte_carlo_fails_with_large_losses(self):
        """Large losses should produce high drawdowns."""
        rt = RobustnessTest({"n_monte_carlo": 20, "mc_max_dd_pct": 0.05})
        trades = pd.DataFrame({"pnl": [-500.0, -400.0, 100.0, -300.0, 50.0]})
        result = rt.monte_carlo_equity(trades)
        assert result["passed"] is False
        assert result["p95_dd"] > 0.05

    def test_monte_carlo_empty_trades(self):
        """Empty trades produce empty result with passed=False."""
        rt = RobustnessTest()
        result = rt.monte_carlo_equity(pd.DataFrame())
        assert result["passed"] is False
        assert result["simulations"] == []

    def test_monte_carlo_with_fees(self):
        """Monte Carlo accounts for fees."""
        rt = RobustnessTest({"n_monte_carlo": 5})
        trades = pd.DataFrame({
            "pnl": [100.0, -20.0, 50.0],
            "fees": [2.0, 2.0, 2.0],
        })
        result = rt.monte_carlo_equity(trades)
        assert len(result["simulations"]) == 5


# ---------------------------------------------------------------------------
# RobustnessTest.param_perturbation
# ---------------------------------------------------------------------------


class TestParamPerturbation:
    def test_perturbation_runs(self):
        """Perturbation test completes with valid inputs."""
        rt = RobustnessTest({"n_perturbations": 3, "perturbation_pct": 0.1})
        candles = _make_trending_candles(n=200)
        strategy = TrendBreakoutStrategy()
        config = _backtest_config()
        result = rt.param_perturbation(candles, strategy, config)
        assert "pass_rate" in result
        assert "results" in result
        assert len(result["results"]) == 3

    def test_perturbation_pass_rate_range(self):
        """Pass rate should be between 0 and 1."""
        rt = RobustnessTest({"n_perturbations": 5})
        candles = _make_trending_candles(n=200)
        strategy = TrendBreakoutStrategy()
        config = _backtest_config()
        result = rt.param_perturbation(candles, strategy, config)
        assert 0.0 <= result["pass_rate"] <= 1.0


# ---------------------------------------------------------------------------
# WalkForwardConfig
# ---------------------------------------------------------------------------


class TestWalkForwardConfig:
    def test_defaults(self):
        cfg = WalkForwardConfig()
        assert cfg.train_bars == 2000
        assert cfg.test_bars == 500
        assert cfg.step_bars == 500
        assert cfg.min_train_bars == 1000


# ---------------------------------------------------------------------------
# WalkForwardTest.generate_folds
# ---------------------------------------------------------------------------


class TestGenerateFolds:
    def test_basic_folds(self):
        wf = WalkForwardTest(WalkForwardConfig(train_bars=100, test_bars=50, step_bars=50, min_train_bars=50))
        folds = wf.generate_folds(total_bars=300)
        assert len(folds) >= 2
        for train_start, train_end, test_start, test_end in folds:
            assert train_end == test_start  # no gap
            assert train_end - train_start == 100
            assert test_end - test_start <= 50

    def test_no_folds_insufficient_data(self):
        wf = WalkForwardTest(WalkForwardConfig(train_bars=100, test_bars=50, min_train_bars=100))
        folds = wf.generate_folds(total_bars=50)  # not enough for even one fold
        assert folds == []

    def test_single_fold(self):
        wf = WalkForwardTest(WalkForwardConfig(train_bars=100, test_bars=50, step_bars=50, min_train_bars=100))
        # 150 bars: fold (0,100,100,150) works; fold (50,150,150,???) has no test bars
        folds = wf.generate_folds(total_bars=150)
        assert len(folds) == 1

    def test_step_controls_overlap(self):
        """Smaller step => more overlapping folds."""
        wf_big = WalkForwardTest(WalkForwardConfig(train_bars=100, test_bars=50, step_bars=50, min_train_bars=50))
        wf_small = WalkForwardTest(WalkForwardConfig(train_bars=100, test_bars=50, step_bars=25, min_train_bars=50))
        folds_big = wf_big.generate_folds(total_bars=500)
        folds_small = wf_small.generate_folds(total_bars=500)
        assert len(folds_small) >= len(folds_big)


# ---------------------------------------------------------------------------
# _chain_equity_curves
# ---------------------------------------------------------------------------


class TestChainEquityCurves:
    def test_empty_pieces(self):
        result = _chain_equity_curves([], initial_equity=10_000.0)
        assert result.empty

    def test_single_piece(self):
        piece = pd.Series([10_000.0, 10_100.0, 10_200.0])
        result = _chain_equity_curves([piece], initial_equity=10_000.0)
        assert len(result) == 3
        assert abs(result.iloc[0] - 10_000.0) < 0.01

    def test_chaining_continuity(self):
        """Second fold should start where first fold ended."""
        piece1 = pd.Series([10_000.0, 11_000.0])  # +10%
        piece2 = pd.Series([10_000.0, 10_500.0])  # +5% from its own start
        result = _chain_equity_curves([piece1, piece2], initial_equity=10_000.0)
        assert len(result) == 4
        # fold 2 starts at 11000
        assert abs(result.iloc[2] - 11_000.0) < 0.01
        # fold 2 ends at 11000 * 1.05 = 11550
        assert abs(result.iloc[3] - 11_550.0) < 1.0

    def test_chaining_preserves_drawdown(self):
        """Drawdown in fold 2 should be reflected in combined curve."""
        piece1 = pd.Series([10_000.0, 12_000.0])
        piece2 = pd.Series([10_000.0, 8_000.0])  # -20%
        result = _chain_equity_curves([piece1, piece2], initial_equity=10_000.0)
        # fold 2 ends at 12000 * 0.8 = 9600
        assert result.iloc[-1] < 10_000.0


# ---------------------------------------------------------------------------
# WalkForwardTest.run (integration)
# ---------------------------------------------------------------------------


class TestWalkForwardRun:
    def test_run_produces_result(self):
        """Walk-forward run produces expected result keys."""
        candles = _make_trending_candles(n=400)
        strategy = TrendBreakoutStrategy()
        config = _backtest_config()
        wf = WalkForwardTest(
            WalkForwardConfig(train_bars=100, test_bars=50, step_bars=50, min_train_bars=50)
        )
        result = wf.run(candles, strategy, config)
        assert "folds" in result
        assert "oos_metrics" in result
        assert "fold_metrics" in result
        assert "config" in result
        assert len(result["folds"]) > 0

    def test_run_no_folds(self):
        """Returns empty result when data is too short."""
        candles = _make_trending_candles(n=30)
        strategy = TrendBreakoutStrategy()
        config = _backtest_config()
        wf = WalkForwardTest(WalkForwardConfig(train_bars=100, test_bars=50))
        result = wf.run(candles, strategy, config)
        assert result["folds"] == []
