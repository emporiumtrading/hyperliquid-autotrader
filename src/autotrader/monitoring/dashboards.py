"""Dashboard definitions, text rendering, and Prometheus export."""

from __future__ import annotations

import math
from typing import Any

TRADING_DASHBOARD: dict[str, Any] = {
    "title": "Hyperliquid AutoTrader",
    "description": "Real-time trading dashboard for the Hyperliquid auto-trader.",
    "refresh_interval": "5s",
    "panels": [
        # ---- PnL ----
        {
            "title": "Realised PnL (cumulative)",
            "metric": "pnl_realised_total",
            "type": "counter",
            "description": "Cumulative realised profit and loss in USD.",
        },
        {
            "title": "Unrealised PnL",
            "metric": "pnl_unrealised",
            "type": "gauge",
            "description": "Mark-to-market unrealised PnL across all open positions.",
        },
        {
            "title": "Total PnL",
            "metric": "pnl_total",
            "type": "gauge",
            "description": "Realised + unrealised PnL.",
        },
        # ---- Positions ----
        {
            "title": "Open Positions Count",
            "metric": "positions_open_count",
            "type": "gauge",
            "description": "Number of currently open positions.",
        },
        {
            "title": "Total Notional Exposure",
            "metric": "positions_notional_total",
            "type": "gauge",
            "description": "Sum of absolute notional values of all open positions (USD).",
        },
        {
            "title": "Net Exposure",
            "metric": "positions_net_exposure",
            "type": "gauge",
            "description": "Signed net notional across all positions (long positive).",
        },
        # ---- Orders ----
        {
            "title": "Orders Placed",
            "metric": "orders_placed_total",
            "type": "counter",
            "description": "Total number of orders submitted to the exchange.",
        },
        {
            "title": "Orders Cancelled",
            "metric": "orders_cancelled_total",
            "type": "counter",
            "description": "Total number of orders cancelled.",
        },
        {
            "title": "Order Rejection Rate",
            "metric": "orders_rejected_total",
            "type": "counter",
            "description": "Total number of orders rejected by the exchange.",
        },
        # ---- Fills ----
        {
            "title": "Fills Count",
            "metric": "fills_total",
            "type": "counter",
            "description": "Total number of trade fills received.",
        },
        {
            "title": "Fill Notional Volume",
            "metric": "fills_notional_volume",
            "type": "counter",
            "description": "Cumulative filled notional volume in USD.",
        },
        {
            "title": "Fill Slippage (bps)",
            "metric": "fills_slippage_bps",
            "type": "histogram",
            "description": "Distribution of execution slippage in basis points.",
        },
        # ---- Latency ----
        {
            "title": "Order Latency (ms)",
            "metric": "latency_order_ms",
            "type": "histogram",
            "description": "Round-trip order placement latency in milliseconds.",
        },
        {
            "title": "Data Feed Latency (ms)",
            "metric": "latency_feed_ms",
            "type": "histogram",
            "description": "Latency of market-data feed updates.",
        },
        {
            "title": "Signal Computation Latency (ms)",
            "metric": "latency_signal_ms",
            "type": "histogram",
            "description": "Time to compute trading signals per cycle.",
        },
        # ---- Risk ----
        {
            "title": "Risk Utilisation (%)",
            "metric": "risk_utilisation_pct",
            "type": "gauge",
            "description": "Current risk budget utilisation as a percentage (0-100).",
        },
        {
            "title": "Margin Usage (%)",
            "metric": "risk_margin_usage_pct",
            "type": "gauge",
            "description": "Exchange margin usage percentage.",
        },
        {
            "title": "Max Position Limit Used (%)",
            "metric": "risk_position_limit_pct",
            "type": "gauge",
            "description": "Largest single position as a percentage of the position limit.",
        },
        # ---- Regime ----
        {
            "title": "Current Market Regime",
            "metric": "regime_current",
            "type": "gauge",
            "description": (
                "Numeric code for the detected market regime "
                "(e.g. 0=unknown, 1=trending, 2=mean-reverting, 3=volatile)."
            ),
        },
        {
            "title": "Regime Confidence",
            "metric": "regime_confidence",
            "type": "gauge",
            "description": "Confidence score of the regime classifier (0-1).",
        },
        # ---- Drawdown ----
        {
            "title": "Current Drawdown (%)",
            "metric": "drawdown_current_pct",
            "type": "gauge",
            "description": "Current drawdown from peak equity as a percentage.",
        },
        {
            "title": "Max Drawdown (%)",
            "metric": "drawdown_max_pct",
            "type": "gauge",
            "description": "Maximum drawdown experienced since start.",
        },
        {
            "title": "Drawdown Duration (bars)",
            "metric": "drawdown_duration_bars",
            "type": "gauge",
            "description": "Number of bars since the last equity high-water mark.",
        },
    ],
}


def get_dashboard_config() -> dict[str, Any]:
    """Return the full trading dashboard definition.

    This can be serialised to JSON and consumed by a Grafana provisioning
    pipeline or a custom web UI.
    """
    return dict(TRADING_DASHBOARD)


# ---------------------------------------------------------------------------
# Plain-text terminal dashboard
# ---------------------------------------------------------------------------

_SEPARATOR = "-" * 60
_HEADER_CHAR = "="


def _format_value(value: float | int) -> str:
    """Format a numeric value for display, handling NaN gracefully."""
    if isinstance(value, float):
        if math.isnan(value):
            return "N/A"
        if value == float("inf") or value == float("-inf"):
            return str(value)
        # Use commas for large numbers; keep two decimals.
        if abs(value) >= 1_000:
            return f"{value:,.2f}"
        return f"{value:.4f}"
    return str(value)


def _resolve_panel_value(
    panel: dict[str, Any],
    metrics_snapshot: dict[str, Any],
) -> str:
    """Look up the panel's metric in the snapshot and return a display string."""
    metric = panel["metric"]
    ptype = panel["type"]

    counters: dict[str, float] = metrics_snapshot.get("counters", {})
    gauges: dict[str, float] = metrics_snapshot.get("gauges", {})
    histograms: dict[str, dict[str, Any]] = metrics_snapshot.get("histograms", {})

    if ptype == "counter":
        value = counters.get(metric, 0.0)
        return _format_value(value)

    if ptype == "gauge":
        value = gauges.get(metric, 0.0)
        return _format_value(value)

    if ptype == "histogram":
        stats = histograms.get(metric, {})
        if not stats or stats.get("count", 0) == 0:
            return "no observations"
        parts = [
            f"count={stats['count']}",
            f"mean={_format_value(stats.get('mean', float('nan')))}",
            f"p50={_format_value(stats.get('p50', float('nan')))}",
            f"p95={_format_value(stats.get('p95', float('nan')))}",
            f"p99={_format_value(stats.get('p99', float('nan')))}",
        ]
        return "  ".join(parts)

    return "?"


def render_text_dashboard(metrics_snapshot: dict[str, Any]) -> str:
    """Render a plain-text terminal dashboard from a metrics snapshot.

    Parameters
    ----------
    metrics_snapshot:
        A dict as returned by :meth:`MetricsCollector.snapshot` with keys
        ``counters``, ``gauges``, and ``histograms``.

    Returns
    -------
    str
        A multi-line string suitable for printing to a terminal.
    """
    title = TRADING_DASHBOARD["title"]
    width = max(60, len(title) + 4)
    header_line = _HEADER_CHAR * width

    lines: list[str] = [
        header_line,
        f"  {title}",
        header_line,
        "",
    ]

    for panel in TRADING_DASHBOARD["panels"]:
        value_str = _resolve_panel_value(panel, metrics_snapshot)
        lines.append(f"  {panel['title']:<40s} {value_str}")

    lines.append("")
    lines.append(_SEPARATOR)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prometheus text exposition format
# ---------------------------------------------------------------------------

def _prom_sanitise(name: str) -> str:
    """Ensure a metric name is valid for Prometheus (lowercase, underscores)."""
    return name.replace(".", "_").replace("-", "_").lower()


def export_prometheus(metrics_snapshot: dict[str, Any]) -> str:
    """Convert a metrics snapshot to Prometheus text exposition format.

    Produces ``# HELP``, ``# TYPE``, and metric value lines for every metric
    defined in :data:`TRADING_DASHBOARD` that is present in the snapshot.

    Parameters
    ----------
    metrics_snapshot:
        A dict as returned by :meth:`MetricsCollector.snapshot`.

    Returns
    -------
    str
        Prometheus text exposition output (``text/plain; version=0.0.4``).
    """
    counters: dict[str, float] = metrics_snapshot.get("counters", {})
    gauges: dict[str, float] = metrics_snapshot.get("gauges", {})
    histograms: dict[str, dict[str, Any]] = metrics_snapshot.get("histograms", {})

    lines: list[str] = []

    for panel in TRADING_DASHBOARD["panels"]:
        metric = _prom_sanitise(panel["metric"])
        ptype = panel["type"]
        description = panel.get("description", "")

        if ptype == "counter":
            value = counters.get(panel["metric"], 0.0)
            lines.append(f"# HELP {metric} {description}")
            lines.append(f"# TYPE {metric} counter")
            lines.append(f"{metric} {value}")
            lines.append("")

        elif ptype == "gauge":
            value = gauges.get(panel["metric"], 0.0)
            lines.append(f"# HELP {metric} {description}")
            lines.append(f"# TYPE {metric} gauge")
            lines.append(f"{metric} {value}")
            lines.append("")

        elif ptype == "histogram":
            stats = histograms.get(panel["metric"], {})
            count = stats.get("count", 0)
            total = stats.get("sum", 0.0)

            lines.append(f"# HELP {metric} {description}")
            lines.append(f"# TYPE {metric} summary")
            if count > 0:
                for quantile, key in [("0.5", "p50"), ("0.95", "p95"), ("0.99", "p99")]:
                    qval = stats.get(key, float("nan"))
                    if not math.isnan(qval):
                        lines.append(f'{metric}{{quantile="{quantile}"}} {qval}')
            lines.append(f"{metric}_count {count}")
            lines.append(f"{metric}_sum {total}")
            lines.append("")

    return "\n".join(lines)

