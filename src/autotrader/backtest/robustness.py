"""Robustness testing via parameter perturbation and Monte Carlo equity analysis.

Validates that a strategy's edge is not fragile -- i.e. that it survives
small parameter changes and that its trade distribution produces acceptable
drawdowns under random reordering.
"""

from __future__ import annotations

import copy
from dataclasses import asdict

import numpy as np
import pandas as pd
import structlog

from autotrader.backtest.engine import BacktestEngine
from autotrader.backtest.metrics import BacktestMetrics
from autotrader.strategies.base import BaseStrategy

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_numeric_params(strategy: BaseStrategy) -> dict[str, float]:
    """Extract numeric (float/int) instance parameters from a strategy.

    Skips private/dunder attributes and the ``name`` attribute.
    """
    params: dict[str, float] = {}
    for attr_name in vars(strategy):
        if attr_name.startswith("_"):
            continue
        if attr_name == "name":
            continue
        val = getattr(strategy, attr_name)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            params[attr_name] = float(val)
    return params


def _set_numeric_params(strategy: BaseStrategy, params: dict[str, float]) -> None:
    """Set numeric parameters on a strategy instance."""
    for attr_name, value in params.items():
        original = getattr(strategy, attr_name, None)
        if isinstance(original, int):
            setattr(strategy, attr_name, max(1, int(round(value))))
        else:
            setattr(strategy, attr_name, value)


def _perturb_params(
    params: dict[str, float],
    perturbation_pct: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Randomly perturb each parameter by up to +/- perturbation_pct.

    Parameters that must remain positive are clamped to a small epsilon.
    """
    perturbed: dict[str, float] = {}
    for key, value in params.items():
        # Uniform perturbation in [-pct, +pct]
        factor = 1.0 + rng.uniform(-perturbation_pct, perturbation_pct)
        new_val = value * factor
        # Clamp to avoid degenerate values
        if value > 0:
            new_val = max(new_val, 1e-6)
        elif value < 0:
            new_val = min(new_val, -1e-6)
        perturbed[key] = new_val
    return perturbed


# ---------------------------------------------------------------------------
# Robustness test class
# ---------------------------------------------------------------------------


class RobustnessTest:
    """Parameter perturbation and Monte Carlo robustness analysis.

    Parameters
    ----------
    config : dict | None
        Configuration overrides:

        - ``n_perturbations`` (int): Number of perturbed backtests
          (default 10).
        - ``perturbation_pct`` (float): Max parameter variation as a
          fraction (default 0.2 = 20 %).
        - ``n_monte_carlo`` (int): Number of Monte Carlo equity
          simulations (default 50).
        - ``min_pass_rate`` (float): Minimum fraction of perturbations
          that must be profitable (default 0.7).
        - ``mc_max_dd_pct`` (float): Maximum acceptable 95th-percentile
          drawdown in Monte Carlo (default 0.30 = 30 %).
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.n_perturbations: int = int(cfg.get("n_perturbations", 10))
        self.perturbation_pct: float = float(cfg.get("perturbation_pct", 0.2))
        self.n_monte_carlo: int = int(cfg.get("n_monte_carlo", 50))
        self.min_pass_rate: float = float(cfg.get("min_pass_rate", 0.7))
        self.mc_max_dd_pct: float = float(cfg.get("mc_max_dd_pct", 0.30))

    # ------------------------------------------------------------------
    # Parameter perturbation
    # ------------------------------------------------------------------

    def param_perturbation(
        self,
        candles: pd.DataFrame,
        strategy: BaseStrategy,
        backtest_config: dict,
        symbol: str = "ETH",
        timeframe: str = "15m",
    ) -> dict:
        """Run backtests with randomly perturbed strategy parameters.

        For each perturbation:
        1. Deep-copy the strategy.
        2. Perturb all numeric parameters by +/- ``perturbation_pct``.
        3. Run a full backtest.
        4. Record the resulting metrics.

        A perturbation *passes* if the total return is positive.

        Parameters
        ----------
        candles : pd.DataFrame
            Candle data with OHLCV columns.
        strategy : BaseStrategy
            The baseline strategy to perturb.
        backtest_config : dict
            Configuration for :class:`BacktestEngine`.
        symbol : str
            Asset symbol.
        timeframe : str
            Candle timeframe string.

        Returns
        -------
        dict
            ``{results, pass_rate, passed, mean_utility, std_utility}``.
        """
        rng = np.random.default_rng(seed=42)
        base_params = _get_numeric_params(strategy)

        if not base_params:
            logger.warning("no_numeric_params_found", strategy=strategy.name)
            return {
                "results": [],
                "pass_rate": 0.0,
                "passed": False,
                "mean_utility": 0.0,
                "std_utility": 0.0,
            }

        logger.info(
            "perturbation_start",
            n_perturbations=self.n_perturbations,
            perturbation_pct=self.perturbation_pct,
            params=list(base_params.keys()),
        )

        results: list[dict] = []
        utilities: list[float] = []
        n_profitable = 0

        for i in range(self.n_perturbations):
            # Deep copy and perturb
            perturbed_strategy = copy.deepcopy(strategy)
            perturbed_params = _perturb_params(base_params, self.perturbation_pct, rng)
            _set_numeric_params(perturbed_strategy, perturbed_params)

            # Build an ensemble wrapping the perturbed strategy if the
            # original was a single strategy; otherwise use it directly.
            engine = BacktestEngine(
                config=backtest_config,
                strategy=perturbed_strategy,
            )

            try:
                result = engine.run(candles, symbol=symbol, timeframe=timeframe)
                metrics: BacktestMetrics = result["metrics"]
            except Exception as exc:
                logger.warning("perturbation_failed", perturbation=i, error=str(exc))
                metrics = BacktestMetrics()

            metrics_dict = asdict(metrics)
            metrics_dict["perturbation_index"] = i
            metrics_dict["perturbed_params"] = perturbed_params
            results.append(metrics_dict)

            utilities.append(metrics.utility)
            if metrics.total_return > 0:
                n_profitable += 1

        pass_rate = n_profitable / max(self.n_perturbations, 1)
        mean_utility = float(np.mean(utilities)) if utilities else 0.0
        std_utility = float(np.std(utilities)) if utilities else 0.0
        passed = pass_rate >= self.min_pass_rate

        logger.info(
            "perturbation_complete",
            pass_rate=f"{pass_rate:.2f}",
            passed=passed,
            mean_utility=f"{mean_utility:.4f}",
        )

        return {
            "results": results,
            "pass_rate": round(pass_rate, 4),
            "passed": passed,
            "mean_utility": round(mean_utility, 6),
            "std_utility": round(std_utility, 6),
        }

    # ------------------------------------------------------------------
    # Monte Carlo equity simulation
    # ------------------------------------------------------------------

    def monte_carlo_equity(
        self,
        trades: pd.DataFrame,
        n_simulations: int | None = None,
    ) -> dict:
        """Reshuffle trade PnLs to estimate drawdown distribution.

        Creates ``n_simulations`` synthetic equity curves by randomly
        permuting the order of trade PnLs.  This reveals how sensitive the
        equity curve shape (especially max drawdown) is to trade ordering.

        Parameters
        ----------
        trades : pd.DataFrame
            Trade ledger with at least a ``pnl`` column.
        n_simulations : int | None
            Override the configured number of simulations.

        Returns
        -------
        dict
            ``{simulations, median_dd, p95_dd, p99_dd, passed}``.
        """
        n_sims = n_simulations if n_simulations is not None else self.n_monte_carlo

        if trades.empty or "pnl" not in trades.columns:
            logger.warning("monte_carlo_no_trades")
            return {
                "simulations": [],
                "median_dd": 0.0,
                "p95_dd": 0.0,
                "p99_dd": 0.0,
                "passed": False,
            }

        pnl_values = trades["pnl"].astype(float).values
        fee_values = (
            trades["fees"].astype(float).values
            if "fees" in trades.columns
            else np.zeros(len(pnl_values))
        )
        slippage_values = (
            trades["slippage"].astype(float).values
            if "slippage" in trades.columns
            else np.zeros(len(pnl_values))
        )
        funding_values = (
            trades["funding"].astype(float).values
            if "funding" in trades.columns
            else np.zeros(len(pnl_values))
        )

        # Net PnL per trade
        net_pnl = pnl_values - fee_values - slippage_values - funding_values

        rng = np.random.default_rng(seed=123)

        logger.info(
            "monte_carlo_start",
            n_simulations=n_sims,
            n_trades=len(net_pnl),
        )

        sim_results: list[dict] = []
        max_drawdowns: list[float] = []

        initial_equity = 10_000.0

        for sim_idx in range(n_sims):
            # Randomly permute trade PnLs
            shuffled = rng.permutation(net_pnl)

            # Build equity curve
            equity = np.empty(len(shuffled) + 1, dtype=float)
            equity[0] = initial_equity
            for j, pnl_val in enumerate(shuffled):
                equity[j + 1] = equity[j] + pnl_val

            # Compute max drawdown
            running_max = np.maximum.accumulate(equity)
            drawdowns = (equity - running_max) / np.where(running_max > 0, running_max, 1.0)
            max_dd = float(abs(drawdowns.min()))

            # Final return
            total_return = (equity[-1] - initial_equity) / initial_equity

            sim_results.append(
                {
                    "simulation": sim_idx,
                    "total_return": round(total_return, 6),
                    "max_drawdown": round(max_dd, 6),
                    "final_equity": round(float(equity[-1]), 2),
                }
            )
            max_drawdowns.append(max_dd)

        max_drawdowns_arr = np.array(max_drawdowns)
        median_dd = float(np.median(max_drawdowns_arr))
        p95_dd = float(np.percentile(max_drawdowns_arr, 95))
        p99_dd = float(np.percentile(max_drawdowns_arr, 99))

        # Pass if the 95th percentile drawdown is within acceptable limits
        passed = p95_dd <= self.mc_max_dd_pct

        logger.info(
            "monte_carlo_complete",
            median_dd=f"{median_dd:.4f}",
            p95_dd=f"{p95_dd:.4f}",
            p99_dd=f"{p99_dd:.4f}",
            passed=passed,
        )

        return {
            "simulations": sim_results,
            "median_dd": round(median_dd, 6),
            "p95_dd": round(p95_dd, 6),
            "p99_dd": round(p99_dd, 6),
            "passed": passed,
        }

    # ------------------------------------------------------------------
    # Full robustness check
    # ------------------------------------------------------------------

    def full_robustness_check(
        self,
        candles: pd.DataFrame,
        strategy: BaseStrategy,
        backtest_config: dict,
        symbol: str = "ETH",
        timeframe: str = "15m",
    ) -> dict:
        """Run both perturbation and Monte Carlo tests.

        The baseline backtest is run first to obtain the trade ledger for
        Monte Carlo analysis.  Then parameter perturbation is executed.

        Parameters
        ----------
        candles : pd.DataFrame
            Candle data with OHLCV columns.
        strategy : BaseStrategy
            The strategy to test.
        backtest_config : dict
            Configuration for :class:`BacktestEngine`.
        symbol : str
            Asset symbol.
        timeframe : str
            Candle timeframe string.

        Returns
        -------
        dict
            ``{baseline_metrics, perturbation, monte_carlo, passed}``.
        """
        logger.info("full_robustness_start")

        # 1. Run baseline backtest
        engine = BacktestEngine(
            config=backtest_config,
            strategy=strategy,
        )
        baseline_result = engine.run(candles, symbol=symbol, timeframe=timeframe)
        baseline_metrics = baseline_result["metrics"]

        # 2. Monte Carlo on baseline trades
        mc_result = self.monte_carlo_equity(baseline_result["trades"])

        # 3. Parameter perturbation
        perturb_result = self.param_perturbation(
            candles, strategy, backtest_config, symbol=symbol, timeframe=timeframe
        )

        # Overall pass: both tests must pass
        overall_passed = perturb_result["passed"] and mc_result["passed"]

        logger.info(
            "full_robustness_complete",
            perturbation_passed=perturb_result["passed"],
            monte_carlo_passed=mc_result["passed"],
            overall_passed=overall_passed,
        )

        return {
            "baseline_metrics": asdict(baseline_metrics),
            "perturbation": perturb_result,
            "monte_carlo": mc_result,
            "passed": overall_passed,
        }
