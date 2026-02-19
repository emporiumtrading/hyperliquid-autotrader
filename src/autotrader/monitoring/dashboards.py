"""Dashboard definitions (data structures for Grafana-style dashboards)."""

from __future__ import annotations

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
