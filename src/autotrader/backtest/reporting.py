"""Backtest report generation and comparison.

Creates structured report dictionaries from backtest results and can
compare two reports (candidate vs baseline) to evaluate strategy changes.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
import structlog

from autotrader.backtest.metrics import BacktestMetrics
from autotrader.utils.serialization import save_json

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(result: dict, output_dir: str | None = None) -> dict:
    """Create a comprehensive report from a backtest engine result.

    Parameters
    ----------
    result : dict
        Output of :meth:`BacktestEngine.run`.  Expected keys:
        ``run_id``, ``symbol``, ``timeframe``, ``trades`` (DataFrame),
        ``equity_curve`` (Series), ``metrics`` (:class:`BacktestMetrics`),
        ``regime_history`` (list), ``config`` (dict).
    output_dir : str | None
        If provided, write ``report.json``, ``trades.csv``, and
        ``equity_curve.csv`` to this directory.

    Returns
    -------
    dict
        Structured report dictionary.
    """
    run_id: str = result.get("run_id", "unknown")
    symbol: str = result.get("symbol", "")
    timeframe: str = result.get("timeframe", "")
    metrics: BacktestMetrics = result.get("metrics", BacktestMetrics())
    trades_df: pd.DataFrame = result.get("trades", pd.DataFrame())
    equity_curve: pd.Series = result.get("equity_curve", pd.Series(dtype=float))
    regime_history: list = result.get("regime_history", [])
    config: dict = result.get("config", {})

    # Metrics dict
    metrics_dict = asdict(metrics)

    # Trade summary
    trade_summary: dict = {
        "total_trades": metrics.total_trades,
        "long_trades": metrics.long_trades,
        "short_trades": metrics.short_trades,
        "win_rate": round(metrics.win_rate, 4),
        "avg_win": round(metrics.avg_win, 4),
        "avg_loss": round(metrics.avg_loss, 4),
        "avg_rr": round(metrics.avg_rr, 4),
        "profit_factor": (
            round(metrics.profit_factor, 4) if metrics.profit_factor != float("inf") else "inf"
        ),
        "avg_holding_bars": metrics.avg_holding_bars,
    }

    # Exit reason breakdown
    if not trades_df.empty and "exit_reason" in trades_df.columns:
        exit_counts = trades_df["exit_reason"].value_counts().to_dict()
        trade_summary["exit_reasons"] = {str(k): int(v) for k, v in exit_counts.items()}
    else:
        trade_summary["exit_reasons"] = {}

    # Regime summary
    regime_summary: dict = {}
    if regime_history:
        regime_labels = [entry[1] for entry in regime_history]
        total_bars = len(regime_labels)
        unique_regimes = set(regime_labels)
        for regime in sorted(unique_regimes):
            count = regime_labels.count(regime)
            regime_summary[regime] = {
                "bars": count,
                "pct": round(count / max(total_bars, 1), 4),
            }

    # Assemble report
    report: dict = {
        "run_id": run_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "metrics": metrics_dict,
        "trade_summary": trade_summary,
        "regime_summary": regime_summary,
        "config": config,
    }

    # Persist to disk if requested
    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # report.json
        save_json(report, out_path / "report.json")

        # trades.csv
        if not trades_df.empty:
            trades_df.to_csv(out_path / "trades.csv", index=False)
        else:
            # Write header-only file
            pd.DataFrame(
                columns=[
                    "trade_id",
                    "symbol",
                    "side",
                    "entry_time",
                    "exit_time",
                    "entry_px",
                    "exit_px",
                    "size",
                    "notional",
                    "leverage",
                    "pnl",
                    "fees",
                    "slippage",
                    "funding",
                    "holding_bars",
                    "exit_reason",
                ]
            ).to_csv(out_path / "trades.csv", index=False)

        # equity_curve.csv
        if not equity_curve.empty:
            eq_df = pd.DataFrame({"timestamp": equity_curve.index, "equity": equity_curve.values})
            eq_df.to_csv(out_path / "equity_curve.csv", index=False)
        else:
            pd.DataFrame(columns=["timestamp", "equity"]).to_csv(
                out_path / "equity_curve.csv", index=False
            )

        logger.info("report_saved", output_dir=str(out_path), run_id=run_id)

    return report


# ---------------------------------------------------------------------------
# Report comparison
# ---------------------------------------------------------------------------


def compare_reports(candidate: dict, baseline: dict) -> dict:
    """Compare a candidate report against a baseline.

    Parameters
    ----------
    candidate : dict
        Report dict from :func:`generate_report` (the new/proposed run).
    baseline : dict
        Report dict from :func:`generate_report` (the reference run).

    Returns
    -------
    dict
        ``{metric_name: {candidate, baseline, delta, pct_change}}`` for
        every numeric metric, plus a top-level ``"candidate_wins"`` bool
        based on the utility metric.
    """
    cand_metrics: dict = candidate.get("metrics", {})
    base_metrics: dict = baseline.get("metrics", {})

    comparison: dict = {}

    # Collect all metric keys from both dicts
    all_keys = set(cand_metrics.keys()) | set(base_metrics.keys())

    for key in sorted(all_keys):
        cand_val = cand_metrics.get(key)
        base_val = base_metrics.get(key)

        # Only compare numeric values
        if not _is_numeric(cand_val) or not _is_numeric(base_val):
            continue

        cand_f = float(cand_val)
        base_f = float(base_val)
        delta = cand_f - base_f

        if abs(base_f) > 1e-12:
            pct_change = delta / abs(base_f)
        else:
            pct_change = 0.0 if abs(delta) < 1e-12 else float("inf")

        comparison[key] = {
            "candidate": round(cand_f, 6),
            "baseline": round(base_f, 6),
            "delta": round(delta, 6),
            "pct_change": round(pct_change, 6),
        }

    # Determine overall winner by utility
    cand_utility = float(cand_metrics.get("utility", 0.0))
    base_utility = float(base_metrics.get("utility", 0.0))
    comparison["candidate_wins"] = cand_utility > base_utility

    return comparison


def _is_numeric(val) -> bool:
    """Return ``True`` if *val* can be safely cast to float."""
    if val is None:
        return False
    try:
        f = float(val)
        # Reject NaN / inf
        if f != f:
            return False
        return True
    except (TypeError, ValueError):
        return False
