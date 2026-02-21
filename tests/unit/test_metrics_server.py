"""Tests for Prometheus metrics HTTP server and text exposition."""

from __future__ import annotations

import json
import time
import urllib.request

import pytest

from autotrader.monitoring.dashboards import (
    TRADING_DASHBOARD,
    export_prometheus,
    render_text_dashboard,
    _format_value,
    _prom_sanitise,
)
from autotrader.monitoring.metrics import MetricsCollector, metrics
from autotrader.monitoring.metrics_server import MetricsServer


# ---------------------------------------------------------------------------
# export_prometheus
# ---------------------------------------------------------------------------


class TestExportPrometheus:
    def test_empty_snapshot(self):
        """Empty metrics produce valid Prometheus output with zero values."""
        snapshot = {"counters": {}, "gauges": {}, "histograms": {}}
        output = export_prometheus(snapshot)
        assert isinstance(output, str)
        assert "# HELP" in output
        assert "# TYPE" in output

    def test_counter_in_output(self):
        """Counter metrics appear in Prometheus output."""
        snapshot = {
            "counters": {"pnl_realised_total": 42.5},
            "gauges": {},
            "histograms": {},
        }
        output = export_prometheus(snapshot)
        assert "pnl_realised_total 42.5" in output
        assert "# TYPE pnl_realised_total counter" in output

    def test_gauge_in_output(self):
        """Gauge metrics appear in Prometheus output."""
        snapshot = {
            "counters": {},
            "gauges": {"pnl_unrealised": -15.3},
            "histograms": {},
        }
        output = export_prometheus(snapshot)
        assert "pnl_unrealised -15.3" in output
        assert "# TYPE pnl_unrealised gauge" in output

    def test_histogram_in_output(self):
        """Histogram metrics produce summary lines with quantiles."""
        snapshot = {
            "counters": {},
            "gauges": {},
            "histograms": {
                "fills_slippage_bps": {
                    "count": 10,
                    "sum": 25.0,
                    "mean": 2.5,
                    "p50": 2.0,
                    "p95": 4.5,
                    "p99": 5.0,
                    "min": 0.5,
                    "max": 5.0,
                }
            },
        }
        output = export_prometheus(snapshot)
        assert "# TYPE fills_slippage_bps summary" in output
        assert 'fills_slippage_bps{quantile="0.5"} 2.0' in output
        assert 'fills_slippage_bps{quantile="0.95"} 4.5' in output
        assert "fills_slippage_bps_count 10" in output
        assert "fills_slippage_bps_sum 25.0" in output

    def test_all_dashboard_panels_present(self):
        """Every panel in the dashboard config produces output."""
        collector = MetricsCollector()
        collector.inc_counter("pnl_realised_total", 100.0)
        collector.set_gauge("pnl_unrealised", -5.0)
        collector.observe("fills_slippage_bps", 1.5)
        snapshot = collector.snapshot()
        output = export_prometheus(snapshot)
        for panel in TRADING_DASHBOARD["panels"]:
            metric = _prom_sanitise(panel["metric"])
            assert metric in output, f"Panel metric {metric} missing from output"


# ---------------------------------------------------------------------------
# render_text_dashboard
# ---------------------------------------------------------------------------


class TestRenderTextDashboard:
    def test_renders_title(self):
        snapshot = {"counters": {}, "gauges": {}, "histograms": {}}
        output = render_text_dashboard(snapshot)
        assert "Hyperliquid AutoTrader" in output

    def test_renders_all_panels(self):
        snapshot = {"counters": {}, "gauges": {}, "histograms": {}}
        output = render_text_dashboard(snapshot)
        for panel in TRADING_DASHBOARD["panels"]:
            assert panel["title"] in output

    def test_histogram_no_observations(self):
        snapshot = {"counters": {}, "gauges": {}, "histograms": {}}
        output = render_text_dashboard(snapshot)
        assert "no observations" in output


# ---------------------------------------------------------------------------
# _format_value
# ---------------------------------------------------------------------------


class TestFormatValue:
    def test_nan(self):
        assert _format_value(float("nan")) == "N/A"

    def test_inf(self):
        assert "inf" in _format_value(float("inf")).lower()

    def test_large_number(self):
        result = _format_value(1_234_567.89)
        assert "," in result  # comma-separated

    def test_small_number(self):
        result = _format_value(0.1234)
        assert "0.1234" in result

    def test_int(self):
        assert _format_value(42) == "42"


# ---------------------------------------------------------------------------
# _prom_sanitise
# ---------------------------------------------------------------------------


class TestPromSanitise:
    def test_dots_to_underscores(self):
        assert _prom_sanitise("my.metric.name") == "my_metric_name"

    def test_dashes_to_underscores(self):
        assert _prom_sanitise("my-metric") == "my_metric"

    def test_lowercase(self):
        assert _prom_sanitise("MyMetric") == "mymetric"


# ---------------------------------------------------------------------------
# MetricsServer
# ---------------------------------------------------------------------------


class TestMetricsServer:
    def test_start_stop(self):
        """Server starts and stops without error."""
        server = MetricsServer(port=19109)
        server.start()
        assert server.is_running
        server.stop()
        time.sleep(0.1)
        assert not server.is_running

    def test_double_start(self):
        """Calling start twice does not error."""
        server = MetricsServer(port=19110)
        server.start()
        server.start()  # should be a no-op
        assert server.is_running
        server.stop()

    def test_stop_without_start(self):
        """Stopping an unstarted server does not error."""
        server = MetricsServer(port=19111)
        server.stop()  # should be a no-op

    def test_metrics_endpoint(self):
        """GET /metrics returns Prometheus text."""
        # Seed some metrics
        metrics.reset()
        metrics.inc_counter("test_counter", 5.0)
        metrics.set_gauge("test_gauge", 99.0)

        server = MetricsServer(port=19112)
        server.start()
        try:
            time.sleep(0.1)
            req = urllib.request.Request("http://127.0.0.1:19112/metrics")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode("utf-8")
                assert resp.status == 200
                content_type = resp.headers.get("Content-Type", "")
                assert "text/plain" in content_type
                # Prometheus output should contain HELP/TYPE lines
                assert "# HELP" in body or "# TYPE" in body
        finally:
            server.stop()
            metrics.reset()

    def test_health_endpoint(self):
        """GET /health returns JSON OK."""
        server = MetricsServer(port=19113)
        server.start()
        try:
            time.sleep(0.1)
            req = urllib.request.Request("http://127.0.0.1:19113/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode("utf-8")
                assert resp.status == 200
                data = json.loads(body)
                assert data["status"] == "ok"
        finally:
            server.stop()

    def test_404_on_unknown_path(self):
        """GET on unknown path returns 404."""
        server = MetricsServer(port=19114)
        server.start()
        try:
            time.sleep(0.1)
            req = urllib.request.Request("http://127.0.0.1:19114/unknown")
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(req, timeout=5)
            assert exc_info.value.code == 404
        finally:
            server.stop()


# ---------------------------------------------------------------------------
# MetricsCollector integration
# ---------------------------------------------------------------------------


class TestMetricsCollectorIntegration:
    def test_snapshot_round_trip(self):
        """Snapshot captures all metric types and can be exported."""
        collector = MetricsCollector()
        collector.inc_counter("a", 1.0)
        collector.inc_counter("a", 2.0)
        collector.set_gauge("b", 42.0)
        collector.observe("c", 10.0)
        collector.observe("c", 20.0)

        snap = collector.snapshot()
        assert snap["counters"]["a"] == 3.0
        assert snap["gauges"]["b"] == 42.0
        assert snap["histograms"]["c"]["count"] == 2
        assert snap["histograms"]["c"]["mean"] == 15.0

        # Export to Prometheus format without error
        output = export_prometheus(snap)
        assert isinstance(output, str)

    def test_reset_clears_all(self):
        """Reset clears counters, gauges, and histograms."""
        collector = MetricsCollector()
        collector.inc_counter("x")
        collector.set_gauge("y", 1.0)
        collector.observe("z", 5.0)
        collector.reset()

        assert collector.get_counter("x") == 0.0
        assert collector.get_gauge("y") == 0.0
        stats = collector.get_histogram_stats("z")
        assert stats["count"] == 0
