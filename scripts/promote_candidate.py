#!/usr/bin/env python3
"""Evaluate a candidate report and promote it to the current baseline if it passes gates.

Usage
-----
    python scripts/promote_candidate.py --report-path reports/backtests/<run_id>/report.json
"""
from __future__ import annotations

import argparse
import sys

from autotrader.governance.approvals import ApprovalWorkflow
from autotrader.governance.gates import load_gate_config
from autotrader.governance.registry import BaselineRegistry
from autotrader.monitoring.logger import setup_logging, get_logger
from autotrader.utils.config import load_config
from autotrader.utils.serialization import load_json


def main() -> int:
    """Evaluate a candidate report against governance gates and promote if passed.

    Returns
    -------
    int
        Exit code: 0 on promotion, 1 on rejection or error.
    """
    parser = argparse.ArgumentParser(
        description="Promote candidate to current baseline",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        required=True,
        help="Path to candidate report JSON (from run_backtest or run_walkforward)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/base.yaml",
        help="Config file path (for gate thresholds)",
    )
    parser.add_argument(
        "--baselines-dir",
        type=str,
        default=None,
        help="Baselines directory (default: artifacts/baselines)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Promote even if no baseline exists (first-time setup)",
    )
    args = parser.parse_args()

    setup_logging()
    logger = get_logger("promote")

    # Load config for gate thresholds
    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        logger.warning("config_not_found", path=args.config)
        cfg = {}
    except Exception as exc:
        logger.warning("config_load_failed", path=args.config, error=str(exc))
        cfg = {}

    # Load the candidate report
    try:
        candidate_report = load_json(args.report_path)
    except FileNotFoundError:
        logger.error("report_not_found", path=args.report_path)
        print(f"ERROR: Report file not found: {args.report_path}")
        return 1
    except Exception as exc:
        logger.error(
            "report_load_failed", path=args.report_path, error=str(exc),
        )
        print(f"ERROR: Failed to load report: {exc}")
        return 1

    # Setup registry and workflow
    baselines_dir = args.baselines_dir or cfg.get(
        "baselines_dir", "artifacts/baselines",
    )
    registry = BaselineRegistry(baselines_dir=baselines_dir)
    gate_config = load_gate_config(cfg)
    workflow = ApprovalWorkflow(registry=registry, gate_config=gate_config)

    logger.info(
        "evaluating_candidate",
        report_path=args.report_path,
        baselines_dir=baselines_dir,
    )

    # Run evaluation and promotion
    result = workflow.evaluate_and_promote(candidate_report)

    # Print results
    print("\n=== Candidate Evaluation ===")
    print(f"Report: {args.report_path}")
    print(f"Promoted: {result['promoted']}")

    if result["promoted"]:
        print(f"New baseline version: {result['baseline_version']}")
        print(f"History snapshot: {result['history_path']}")
        logger.info(
            "candidate_promoted",
            version=result["baseline_version"],
            history_path=result["history_path"],
        )
    else:
        print(f"Recommendation: {result['gate_result'].get('recommendation', 'reject')}")
        if result["reasons"]:
            print("\nRejection reasons:")
            for reason in result["reasons"]:
                print(f"  - {reason}")
        logger.info(
            "candidate_rejected",
            reasons=result["reasons"],
        )

    # Print gate details
    gates = result["gate_result"].get("gates", {})
    if gates:
        print("\nGate Results:")
        for gate_name, gate_info in gates.items():
            status = "PASS" if gate_info["passed"] else "FAIL"
            candidate_val = gate_info.get("candidate_value", "N/A")
            threshold = gate_info.get("threshold", "N/A")
            baseline_val = gate_info.get("baseline_value", "N/A")
            print(
                f"  [{status}] {gate_name}: "
                f"candidate={candidate_val}, "
                f"threshold={threshold}, "
                f"baseline={baseline_val}"
            )

    return 0 if result["promoted"] else 1


if __name__ == "__main__":
    sys.exit(main())
