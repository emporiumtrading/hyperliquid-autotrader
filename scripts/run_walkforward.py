#!/usr/bin/env python3
"""Run walk-forward out-of-sample testing.

Usage
-----
    python scripts/run_walkforward.py --symbol ETH --timeframe 15m --train-bars 2000 --test-bars 500
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict

from autotrader.backtest.reporting import generate_report
from autotrader.backtest.walkforward import WalkForwardConfig, WalkForwardTest
from autotrader.monitoring.logger import setup_logging, get_logger
from autotrader.store.parquet import ParquetStore
from autotrader.strategies.base import BaseStrategy
from autotrader.strategies.ensemble import EnsembleStrategy
from autotrader.strategies.funding_extremes import FundingExtremesStrategy
from autotrader.strategies.range_meanrev import RangeMeanRevStrategy
from autotrader.strategies.trend_breakout import TrendBreakoutStrategy
from autotrader.strategies.vol_expansion import VolExpansionStrategy
from autotrader.utils.config import load_config
from autotrader.utils.serialization import save_json


_STRATEGY_MAP: dict[str, type] = {
    "ensemble": EnsembleStrategy,
    "trend_breakout": TrendBreakoutStrategy,
    "range_meanrev": RangeMeanRevStrategy,
    "vol_expansion": VolExpansionStrategy,
    "funding_extremes": FundingExtremesStrategy,
}


def _create_strategy(name: str) -> BaseStrategy | EnsembleStrategy:
    """Instantiate a strategy by name."""
    cls = _STRATEGY_MAP.get(name, EnsembleStrategy)
    return cls()


def main() -> int:
    """Run walk-forward testing and print OOS summary.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(description="Run walk-forward test")
    parser.add_argument(
        "--config",
        type=str,
        default="config/base.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="ETH",
        help="Trading symbol (default: ETH)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="15m",
        help="Candle timeframe (default: 15m)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="ensemble",
        choices=list(_STRATEGY_MAP.keys()),
        help="Strategy to test (default: ensemble)",
    )
    parser.add_argument(
        "--train-bars",
        type=int,
        default=2000,
        help="Training window size in bars (default: 2000)",
    )
    parser.add_argument(
        "--test-bars",
        type=int,
        default=500,
        help="Test window size in bars (default: 500)",
    )
    parser.add_argument(
        "--step-bars",
        type=int,
        default=500,
        help="Step size between folds in bars (default: 500)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for results output",
    )
    args = parser.parse_args()

    setup_logging()
    logger = get_logger("walkforward")

    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        logger.error("config_not_found", path=args.config)
        return 1
    except Exception as exc:
        logger.error("config_load_failed", path=args.config, error=str(exc))
        return 1

    # Load candles from store
    store = ParquetStore(cfg.get("store_dir", "data/processed"))
    candles = store.read_candles(
        args.symbol, args.timeframe, 0, int(time.time() * 1000),
    )
    if candles.empty:
        logger.error(
            "no_candles", symbol=args.symbol, timeframe=args.timeframe,
        )
        print(
            f"No candles found for {args.symbol}/{args.timeframe}. "
            f"Run bootstrap_history.py first."
        )
        return 1

    logger.info(
        "walkforward_setup",
        symbol=args.symbol,
        timeframe=args.timeframe,
        total_bars=len(candles),
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        step_bars=args.step_bars,
    )

    # Create strategy and walk-forward config
    strategy = _create_strategy(args.strategy)
    wf_config = WalkForwardConfig(
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        step_bars=args.step_bars,
    )
    wf_test = WalkForwardTest(config=wf_config)

    # Run walk-forward test
    result = wf_test.run(
        candles=candles,
        strategy=strategy,
        backtest_config=cfg,
        symbol=args.symbol,
        timeframe=args.timeframe,
    )

    # Determine output directory
    output_dir = args.output_dir or f"reports/walkforward/{args.symbol}_{args.timeframe}"

    # Save OOS results summary
    oos_metrics = result["oos_metrics"]
    fold_count = len(result["folds"])

    # Build a summary dict for persistence
    summary = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "strategy": args.strategy,
        "walkforward_config": asdict(wf_config),
        "n_folds": fold_count,
        "oos_metrics": asdict(oos_metrics),
        "fold_metrics": [asdict(fm) for fm in result["fold_metrics"]],
    }
    import os
    os.makedirs(output_dir, exist_ok=True)
    save_json(summary, f"{output_dir}/walkforward_report.json")

    # Save OOS equity curve if available
    if not result["oos_equity_curve"].empty:
        import pandas as pd
        eq_df = pd.DataFrame({
            "timestamp": result["oos_equity_curve"].index,
            "equity": result["oos_equity_curve"].values,
        })
        eq_df.to_csv(f"{output_dir}/oos_equity_curve.csv", index=False)

    # Print summary
    print(f"\nWalk-Forward Results: {args.symbol}/{args.timeframe}")
    print(f"Strategy: {args.strategy}")
    print(f"Folds: {fold_count}")
    print(f"OOS Total Return: {oos_metrics.total_return:.2%}")
    print(f"OOS Sharpe Ratio: {oos_metrics.sharpe_ratio:.2f}")
    print(f"OOS Max Drawdown: {oos_metrics.max_drawdown:.2%}")
    print(f"OOS Total Trades: {oos_metrics.total_trades}")
    print(f"OOS Win Rate: {oos_metrics.win_rate:.1%}")
    print(f"OOS Profit Factor: {oos_metrics.profit_factor:.2f}")
    print(f"OOS Utility: {oos_metrics.utility:.4f}")

    # Per-fold summary
    print("\nPer-fold OOS metrics:")
    for i, fm in enumerate(result["fold_metrics"]):
        print(
            f"  Fold {i}: return={fm.total_return:.2%}  "
            f"sharpe={fm.sharpe_ratio:.2f}  "
            f"dd={fm.max_drawdown:.2%}  "
            f"trades={fm.total_trades}"
        )

    print(f"\nReport saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
