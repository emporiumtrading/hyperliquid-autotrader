#!/usr/bin/env python3
"""Emergency rollback to a previous baseline version.

Usage
-----
    python scripts/rollback.py                  # rollback to previous version
    python scripts/rollback.py --version 3      # rollback to specific version
"""
from __future__ import annotations

import argparse
import sys

from autotrader.governance.approvals import ApprovalWorkflow
from autotrader.governance.registry import BaselineRegistry
from autotrader.monitoring.alerts import init as init_alerts, send as send_alert
from autotrader.monitoring.logger import setup_logging, get_logger
from autotrader.utils.config import load_config


def main() -> int:
    """Roll back to a previous baseline version.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(
        description="Emergency rollback to a previous baseline version",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Specific version to restore (default: previous version)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/base.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--baselines-dir",
        type=str,
        default=None,
        help="Baselines directory (default: artifacts/baselines)",
    )
    args = parser.parse_args()

    setup_logging()
    logger = get_logger("rollback")

    # Load config (non-fatal if missing)
    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, Exception):
        cfg = {}

    # Initialize alerts so rollback notifications are sent
    init_alerts(cfg.get("observability", {}).get("alert_webhook_url"))

    # Setup registry and workflow
    baselines_dir = args.baselines_dir or cfg.get(
        "baselines_dir", "artifacts/baselines",
    )
    registry = BaselineRegistry(baselines_dir=baselines_dir)
    workflow = ApprovalWorkflow(registry=registry)

    # Show current state
    current = registry.load_current()
    if current:
        print(f"Current baseline: version {current.get('version')}")
        print(f"  Strategy: {current.get('strategy_name')}")
        print(f"  Created:  {current.get('created_at')}")
    else:
        print("No current baseline found.")

    # Show available history
    history = registry.list_history()
    if history:
        print(f"\nAvailable history versions: {[h.get('version') for h in history]}")
    else:
        print("\nNo history available for rollback.")
        return 1

    # Perform rollback
    if args.version is not None:
        print(f"\nRolling back to version {args.version}...")
    else:
        print("\nRolling back to previous version...")

    result = workflow.emergency_rollback(version=args.version)

    if result["rolled_back"]:
        print(f"\nRollback successful!")
        print(f"  Old version: {result['old_version']}")
        print(f"  New version: {result['new_version']}")
        print(f"  Strategy:    {result['baseline'].get('strategy_name', 'N/A')}")

        # Send alert about the rollback
        send_alert(
            "warning",
            "Baseline Rollback Executed",
            f"Rolled back from v{result['old_version']} to v{result['new_version']}",
            data={
                "old_version": result["old_version"],
                "new_version": result["new_version"],
            },
        )

        logger.info(
            "rollback_complete",
            old_version=result["old_version"],
            new_version=result["new_version"],
        )
        return 0
    else:
        print(f"\nRollback failed: {result['error']}")
        logger.error(
            "rollback_failed",
            error=result["error"],
            requested_version=args.version,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
