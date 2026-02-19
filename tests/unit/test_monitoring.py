"""Tests for autotrader.monitoring -- metrics collector, alerts, and logging."""

from __future__ import annotations

import math

import pytest

from autotrader.monitoring.alerts import AlertManager
from autotrader.monitoring.logger import setup_logging
from autotrader.monitoring.metrics import MetricsCollector

# ---------------------------------------------------------------------------
# MetricsCollector tests
# ---------------------------------------------------------------------------


class TestMetricsCounter:
    def test_metrics_counter(self):
        """Counter increments correctly."""
        m = MetricsCollector()
        assert m.get_counter("test_counter") == 0.0

        m.inc_counter("test_counter")
        assert m.get_counter("test_counter") == 1.0

        m.inc_counter("test_counter", 5.0)
        assert m.get_counter("test_counter") == 6.0

        # Different counter is independent
        assert m.get_counter("other_counter") == 0.0


class TestMetricsGauge:
    def test_metrics_gauge(self):
        """Gauge sets and overwrites values."""
        m = MetricsCollector()
        assert m.get_gauge("test_gauge") == 0.0

        m.set_gauge("test_gauge", 42.5)
        assert m.get_gauge("test_gauge") == 42.5

        m.set_gauge("test_gauge", -10.0)
        assert m.get_gauge("test_gauge") == -10.0

        # Different gauge
        m.set_gauge("equity", 10_000.0)
        assert m.get_gauge("equity") == 10_000.0
        assert m.get_gauge("test_gauge") == -10.0


class TestMetricsHistogram:
    def test_metrics_histogram(self):
        """Histogram tracks observations and computes stats."""
        m = MetricsCollector()

        # No observations yet
        stats = m.get_histogram_stats("latency")
        assert stats["count"] == 0
        assert math.isnan(stats["mean"])

        # Add observations
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        for v in values:
            m.observe("latency", v)

        stats = m.get_histogram_stats("latency")
        assert stats["count"] == 10
        assert pytest.approx(stats["sum"], abs=1e-6) == 55.0
        assert pytest.approx(stats["mean"], abs=1e-6) == 5.5
        assert pytest.approx(stats["min"], abs=1e-6) == 1.0
        assert pytest.approx(stats["max"], abs=1e-6) == 10.0
        assert stats["p50"] == pytest.approx(5.5, abs=0.5)  # median


class TestMetricsSnapshot:
    def test_metrics_snapshot(self):
        """Snapshot returns all counters, gauges, and histogram summaries."""
        m = MetricsCollector()

        m.inc_counter("orders_total", 10)
        m.set_gauge("equity", 9_500.0)
        m.observe("fill_time_ms", 120.0)
        m.observe("fill_time_ms", 80.0)

        snap = m.snapshot()

        assert "counters" in snap
        assert "gauges" in snap
        assert "histograms" in snap

        assert snap["counters"]["orders_total"] == 10.0
        assert snap["gauges"]["equity"] == 9_500.0
        assert snap["histograms"]["fill_time_ms"]["count"] == 2
        assert pytest.approx(snap["histograms"]["fill_time_ms"]["mean"], abs=1e-6) == 100.0


# ---------------------------------------------------------------------------
# AlertManager tests
# ---------------------------------------------------------------------------


class TestAlertManagerLogOnly:
    def test_alert_manager_log_only(self):
        """Alert with no webhook URL logs and returns True."""
        mgr = AlertManager(webhook_url=None)
        result = mgr.send(
            severity="warning",
            title="Test Alert",
            message="This is a test alert",
            data={"key": "value"},
        )
        assert result is True

        # Convenience methods also work
        assert mgr.critical("Crit", "msg") is True
        assert mgr.warning("Warn", "msg") is True
        assert mgr.info("Info", "msg") is True


# ---------------------------------------------------------------------------
# Logging setup tests
# ---------------------------------------------------------------------------


class TestSetupLogging:
    def test_setup_logging(self):
        """setup_logging should run without errors for various levels."""
        # Should not raise
        setup_logging(level="DEBUG")
        setup_logging(level="INFO")
        setup_logging(level="WARNING")
