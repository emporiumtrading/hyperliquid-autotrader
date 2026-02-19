#!/usr/bin/env python3
"""Run the trading system in paper mode (no real orders sent).

Usage
-----
    python scripts/paper_trade.py --config config/base.yaml
"""
from __future__ import annotations

import argparse
import sys

from autotrader.hl.nonces import init as init_nonces
from autotrader.monitoring.alerts import init as init_alerts
from autotrader.monitoring.logger import setup_logging, get_logger
from autotrader.runtime.kill_switch import init as init_kill_switch
from autotrader.runtime.scheduler import TradingScheduler
from autotrader.runtime.startup_checks import run_startup_checks
from autotrader.utils.config import load_config


def main() -> int:
    """Start the paper trading loop.

    Returns
    -------
    int
        Exit code: 0 on clean shutdown, 1 on failure.
    """
    parser = argparse.ArgumentParser(
        description="Paper trade (no real orders)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/base.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Max trading loop iterations (for testing)",
    )
    args = parser.parse_args()

    setup_logging()
    logger = get_logger("paper_trade")

    # Load config with env=paper override
    try:
        cfg = load_config(args.config, overrides={"env": "paper"})
    except FileNotFoundError:
        logger.error("config_not_found", path=args.config)
        return 1
    except Exception as exc:
        logger.error("config_load_failed", path=args.config, error=str(exc))
        return 1

    # Initialize subsystems
    init_alerts(cfg.get("observability", {}).get("alert_webhook_url"))
    init_nonces(cfg.get("nonce_path", "data/nonces.json"))
    init_kill_switch(cfg.get("kill_switch_path", "data/kill_switch.json"))

    # Run startup checks
    errors = run_startup_checks(cfg)
    if errors:
        for e in errors:
            logger.error("startup_check_failed", error=e)
        return 1

    logger.info("starting_paper_trader")
    scheduler = TradingScheduler(cfg)

    try:
        scheduler.run_loop(max_iterations=args.max_iterations)
    except KeyboardInterrupt:
        logger.info("shutdown_requested")
        scheduler.shutdown()
    except Exception as exc:
        logger.error("paper_trade_error", error=str(exc), exc_info=True)
        return 1

    logger.info("paper_trade_shutdown_complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
