#!/usr/bin/env python3
"""Bootstrap historical candle data from Hyperliquid into the ParquetStore.

Usage
-----
    python scripts/bootstrap_history.py --coins ETH BTC --intervals 15m 1h 4h --num-candles 5000
"""
from __future__ import annotations

import argparse
import sys
import time

from autotrader.data.collectors.candles import bootstrap_candles
from autotrader.hl.client import create_client
from autotrader.monitoring.logger import setup_logging, get_logger
from autotrader.store.dataset_hash import create_manifest, save_manifest
from autotrader.store.parquet import ParquetStore
from autotrader.utils.config import load_config


def main() -> int:
    """Bootstrap candle history from Hyperliquid into the local store.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(
        description="Bootstrap candle history from Hyperliquid",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/base.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--coins",
        nargs="+",
        default=["ETH", "BTC"],
        help="Coins to bootstrap (default: ETH BTC)",
    )
    parser.add_argument(
        "--intervals",
        nargs="+",
        default=["15m", "1h", "4h"],
        help="Candle intervals (default: 15m 1h 4h)",
    )
    parser.add_argument(
        "--num-candles",
        type=int,
        default=5000,
        help="Number of candles per coin/interval (default: 5000)",
    )
    args = parser.parse_args()

    setup_logging()
    logger = get_logger("bootstrap")

    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        logger.error("config_not_found", path=args.config)
        return 1
    except Exception as exc:
        logger.error("config_load_failed", path=args.config, error=str(exc))
        return 1

    client = create_client(cfg.get("hyperliquid", {}))
    store = ParquetStore(cfg.get("store_dir", "data/processed"))

    total = 0
    for coin in args.coins:
        for interval in args.intervals:
            logger.info("bootstrapping", coin=coin, interval=interval)
            try:
                n = bootstrap_candles(
                    client, store, coin, interval, args.num_candles,
                )
                total += n
                logger.info(
                    "bootstrapped", coin=coin, interval=interval, rows=n,
                )
            except Exception as exc:
                logger.error(
                    "bootstrap_failed",
                    coin=coin,
                    interval=interval,
                    error=str(exc),
                )
                return 1

    # Create and save dataset manifest
    manifest = create_manifest(
        args.coins,
        args.intervals,
        start_ms=0,
        end_ms=int(time.time() * 1000),
    )
    hash_val = save_manifest(manifest, "data/manifests/bootstrap.json")
    logger.info(
        "bootstrap_complete",
        total_rows=total,
        dataset_hash=hash_val,
    )
    print(f"Bootstrap complete: {total} rows, hash={hash_val}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
