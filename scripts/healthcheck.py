#!/usr/bin/env python3
"""Health check: run startup checks and report results.

Designed to be used as a liveness/readiness probe or pre-deploy verification.
Exit code 0 means all checks pass; exit code 1 means one or more failed.

Usage
-----
    python scripts/healthcheck.py --config config/base.yaml
"""
from __future__ import annotations

import argparse
import sys

from autotrader.monitoring.logger import setup_logging, get_logger
from autotrader.runtime.startup_checks import run_startup_checks
from autotrader.utils.config import load_config


def main() -> int:
    """Run all startup checks and print results.

    Returns
    -------
    int
        Exit code: 0 if all checks pass, 1 if any fail.
    """
    parser = argparse.ArgumentParser(
        description="Health check (blocks trading if any check fails)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/base.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Only print failures (suppress OK messages)",
    )
    args = parser.parse_args()

    setup_logging()
    logger = get_logger("healthcheck")

    # Load config
    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        print(f"FAIL: Config file not found: {args.config}")
        return 1
    except Exception as exc:
        print(f"FAIL: Config load error: {exc}")
        return 1

    if not args.quiet:
        print(f"Running health checks with config: {args.config}")

    # Run all startup checks
    errors = run_startup_checks(cfg)

    if errors:
        print(f"\nHealth check FAILED ({len(errors)} error(s)):")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
            logger.error("healthcheck_failed", check_number=i, error=error)
        return 1
    else:
        if not args.quiet:
            print("\nAll health checks passed.")
        logger.info("healthcheck_ok")
        return 0


if __name__ == "__main__":
    sys.exit(main())
