"""Main entrypoint for the Hyperliquid Autonomous Trader.

Parses CLI arguments, initialises all subsystems, runs startup checks,
and dispatches to the appropriate trading mode (paper, canary, or live).
"""

from __future__ import annotations

import argparse
import signal
import sys

from autotrader.governance.registry import BaselineRegistry
from autotrader.hl.nonces import init as init_nonces
from autotrader.monitoring.alerts import init as init_alerts
from autotrader.monitoring.audit import init as init_audit
from autotrader.monitoring.logger import get_logger, setup_logging
from autotrader.runtime.kill_switch import init as init_kill_switch
from autotrader.runtime.scheduler import TradingScheduler
from autotrader.runtime.startup_checks import run_startup_checks
from autotrader.utils.config import load_config


def main() -> int:
    """Parse arguments, initialise subsystems, and run the trading loop.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(description="Hyperliquid Autonomous Trader")
    parser.add_argument(
        "--env",
        choices=["paper", "canary", "live"],
        default="paper",
        help="Trading environment (default: paper)",
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

    # Setup structured logging
    setup_logging()
    logger = get_logger("main")

    # Load config with env override
    try:
        cfg = load_config(args.config, overrides={"env": args.env})
    except FileNotFoundError:
        logger.error("config_not_found", path=args.config)
        return 1
    except Exception as exc:
        logger.error("config_load_failed", path=args.config, error=str(exc))
        return 1

    # Initialize subsystems
    init_alerts(cfg.get("observability", {}).get("alert_webhook_url"))
    init_audit(cfg.get("audit_path", "logs/audit.jsonl"))
    init_nonces(cfg.get("nonce_path", "data/nonces.json"))
    init_kill_switch(cfg.get("kill_switch_path", "data/kill_switch.json"))

    # Run startup checks
    errors = run_startup_checks(cfg)
    if errors:
        for e in errors:
            logger.error("startup_check_failed", error=e)
        return 1

    # Determine effective environment
    env = cfg.get("env", args.env)

    if env == "live":
        # Live mode requires an approved baseline
        registry = BaselineRegistry()
        baseline = registry.load_current()
        if not baseline:
            logger.error(
                "no_baseline",
                msg="Live mode requires an approved baseline",
            )
            return 1
        logger.info(
            "starting_live_trader",
            baseline_version=baseline.get("version"),
        )
    elif env in ("paper", "canary"):
        logger.info("starting_trader", env=env)
    else:
        logger.error("unknown_env", env=env)
        return 1

    scheduler = TradingScheduler(cfg)

    # Register SIGTERM handler so containers / systemd can shut down
    # gracefully instead of getting an unhandled signal.
    def _sigterm_handler(signum: int, frame: object) -> None:
        logger.info("sigterm_received", signal=signum)
        scheduler.shutdown()

    signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        scheduler.run_loop(max_iterations=args.max_iterations)
    except KeyboardInterrupt:
        logger.info("shutdown_requested")
        scheduler.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
