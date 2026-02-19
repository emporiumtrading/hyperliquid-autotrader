#!/usr/bin/env python3
"""Run a backtest against stored candle data.

Usage
-----
    python scripts/run_backtest.py --symbol ETH --timeframe 15m --strategy ensemble
"""
from __future__ import annotations

import argparse
import sys
import time

from autotrader.backtest.engine import BacktestEngine
from autotrader.backtest.reporting import generate_report
from autotrader.monitoring.logger import setup_logging, get_logger
from autotrader.store.parquet import ParquetStore
from autotrader.strategies.base import BaseStrategy
from autotrader.strategies.ensemble import EnsembleStrategy
from autotrader.strategies.funding_extremes import FundingExtremesStrategy
from autotrader.strategies.range_meanrev import RangeMeanRevStrategy
from autotrader.strategies.trend_breakout import TrendBreakoutStrategy
from autotrader.strategies.vol_expansion import VolExpansionStrategy
from autotrader.utils.config import load_config


_STRATEGY_MAP: dict[str, type] = {
    "ensemble": EnsembleStrategy,
    "trend_breakout": TrendBreakoutStrategy,
    "range_meanrev": RangeMeanRevStrategy,
    "vol_expansion": VolExpansionStrategy,
    "funding_extremes": FundingExtremesStrategy,
}


def _create_strategy(name: str) -> BaseStrategy | EnsembleStrategy:
    """Instantiate a strategy by name.

    Parameters
    ----------
    name : str
        One of the keys in ``_STRATEGY_MAP``.

    Returns
    -------
    BaseStrategy | EnsembleStrategy
        A fresh strategy instance.
    """
    cls = _STRATEGY_MAP.get(name, EnsembleStrategy)
    return cls()


def main() -> int:
    """Run a backtest and print summary metrics.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(description="Run backtest")
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
        help="Strategy to backtest (default: ensemble)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for report output (default: reports/backtests/<run_id>)",
    )
    args = parser.parse_args()

    setup_logging()
    logger = get_logger("backtest")

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

    # Create strategy
    strategy = _create_strategy(args.strategy)

    # Run backtest
    engine = BacktestEngine(cfg, strategy=strategy)
    result = engine.run(candles, symbol=args.symbol, timeframe=args.timeframe)

    # Generate report
    output_dir = (
        args.output_dir or f"reports/backtests/{result['run_id']}"
    )
    generate_report(result, output_dir=output_dir)

    # Print summary
    m = result["metrics"]
    print(f"\nBacktest Results: {args.symbol}/{args.timeframe}")
    print(f"Strategy: {args.strategy}")
    print(f"Total Trades: {m.total_trades}")
    print(f"Win Rate: {m.win_rate:.1%}")
    print(f"Total Return: {m.total_return:.2%}")
    print(f"Sharpe Ratio: {m.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {m.max_drawdown:.2%}")
    print(f"Profit Factor: {m.profit_factor:.2f}")
    print(f"Utility: {m.utility:.4f}")
    print(f"Report saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
