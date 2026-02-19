#!/usr/bin/env python3
"""Run the trading system in live mode (real orders).

Refuses to start unless a promoted baseline exists in the governance registry.
Do not use without first passing governance gates and canary validation.

Usage
-----
    python scripts/live_trade.py --config config/live.yaml
"""
from __future__ import annotations

import argparse
import sys

from autotrader.governance.registry import BaselineRegistry
from autotrader.hl.nonces import init as init_nonces
from autotrader.monitoring.alerts import init as init_alerts
from autotrader.monitoring.logger import setup_logging, get_logger
from autotrader.runtime.kill_switch import init as init_kill_switch
from autotrader.runtime.scheduler import TradingScheduler
from autotrader.runtime.startup_checks import run_startup_checks
from autotrader.utils.config import load_config


def main() -> int:
    """Start the live trading loop.

    Returns
    -------
    int
        Exit code: 0 on clean shutdown, 1 on failure.
    """
    parser = argparse.ArgumentParser(
        description="Live trade (real orders -- requires approved baseline)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/live.yaml",
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
    logger = get_logger("live_trade")

    # Load config with env=live override
    try:
        cfg = load_config(args.config, overrides={"env": "live"})
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

    # Verify baseline exists before proceeding
    baselines_dir = cfg.get("baselines_dir", "artifacts/baselines")
    registry = BaselineRegistry(baselines_dir=baselines_dir)
    baseline = registry.load_current()
    if not baseline:
        logger.error(
            "no_baseline",
            msg="Live mode requires an approved baseline. "
            "Run promote_candidate.py first.",
        )
        print(
            "ERROR: No approved baseline found. "
            "Live trading requires a promoted baseline.\n"
            "Run promote_candidate.py to evaluate and promote a candidate."
        )
        return 1

    logger.info(
        "baseline_loaded",
        version=baseline.get("version"),
        strategy=baseline.get("strategy_name"),
        created_at=baseline.get("created_at"),
    )

    # Run startup checks
    errors = run_startup_checks(cfg)
    if errors:
        for e in errors:
            logger.error("startup_check_failed", error=e)
        return 1

    logger.info(
        "starting_live_trader",
        baseline_version=baseline.get("version"),
    )
    scheduler = TradingScheduler(cfg)

    try:
        scheduler.run_loop(max_iterations=args.max_iterations)
    except KeyboardInterrupt:
        logger.info("shutdown_requested")
        scheduler.shutdown()
    except Exception as exc:
        logger.error("live_trade_error", error=str(exc), exc_info=True)
        return 1

    logger.info("live_trade_shutdown_complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
