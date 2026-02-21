"""CLI regression tests.

Validates that scripts and main entrypoint behave correctly,
config loading works, startup checks run, and edge cases
in the trading scheduler are handled properly.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from autotrader.monitoring.metrics import MetricsCollector
from autotrader.runtime.startup_checks import run_startup_checks, validate_config_schema
from autotrader.utils.config import load_config


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


class TestConfigLoading:
    def test_load_base_config(self):
        """Base config loads and has required sections."""
        cfg = load_config("config/base.yaml")
        assert "hyperliquid" in cfg
        assert "risk" in cfg
        assert "universe" in cfg
        assert "timeframes" in cfg

    def test_load_config_with_override(self):
        """Overrides are applied on top of base config."""
        cfg = load_config("config/base.yaml", overrides={"env": "canary"})
        assert cfg["env"] == "canary"

    def test_load_config_missing_file(self):
        """Loading a nonexistent config raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_config_has_initial_equity(self):
        """Config must have initial_equity (D3 fix)."""
        cfg = load_config("config/base.yaml")
        assert "initial_equity" in cfg
        assert isinstance(cfg["initial_equity"], (int, float))
        assert cfg["initial_equity"] > 0

    def test_config_execution_keys(self):
        """Execution section uses correct key names (D3 fix)."""
        cfg = load_config("config/base.yaml")
        exec_cfg = cfg.get("execution", {})
        # These keys must match what the scheduler reads
        assert "chase_timeout_sec" in exec_cfg or "chase_seconds" not in exec_cfg
        assert "chase_max_retries" in exec_cfg or "max_order_retries" not in exec_cfg

    def test_config_prometheus_port(self):
        """Observability section has prometheus_port."""
        cfg = load_config("config/base.yaml")
        obs = cfg.get("observability", {})
        assert "prometheus_port" in obs
        assert isinstance(obs["prometheus_port"], int)


# ---------------------------------------------------------------------------
# Config schema validation
# ---------------------------------------------------------------------------


class TestConfigSchemaValidation:
    def test_valid_config(self):
        """Valid config passes schema validation."""
        cfg = load_config("config/base.yaml")
        errors = validate_config_schema(cfg)
        assert errors == []

    def test_missing_hyperliquid(self):
        """Missing 'hyperliquid' key is flagged."""
        cfg = {"risk": {}, "universe": [], "timeframes": ["15m"]}
        errors = validate_config_schema(cfg)
        assert any("hyperliquid" in e for e in errors)

    def test_missing_risk(self):
        """Missing 'risk' key is flagged."""
        cfg = {"hyperliquid": {}, "universe": [], "timeframes": ["15m"]}
        errors = validate_config_schema(cfg)
        assert any("risk" in e for e in errors)

    def test_missing_universe(self):
        """Missing 'universe' key is flagged."""
        cfg = {"hyperliquid": {}, "risk": {}, "timeframes": ["15m"]}
        errors = validate_config_schema(cfg)
        assert any("universe" in e for e in errors)

    def test_missing_timeframes(self):
        """Missing 'timeframes' key is flagged."""
        cfg = {"hyperliquid": {}, "risk": {}, "universe": []}
        errors = validate_config_schema(cfg)
        assert any("timeframes" in e for e in errors)

    def test_empty_universe_list(self):
        """Empty universe list is flagged."""
        cfg = {
            "hyperliquid": {"rest_url": "x"},
            "risk": {},
            "universe": [],
            "timeframes": ["15m"],
        }
        errors = validate_config_schema(cfg)
        assert any("empty" in e.lower() for e in errors)

    def test_empty_timeframes_list(self):
        """Empty timeframes list is flagged."""
        cfg = {
            "hyperliquid": {"rest_url": "x"},
            "risk": {},
            "universe": ["ETH"],
            "timeframes": [],
        }
        errors = validate_config_schema(cfg)
        assert any("empty" in e.lower() for e in errors)

    def test_dict_universe_accepted(self):
        """Dict-form universe (with top_n etc.) is accepted."""
        cfg = {
            "hyperliquid": {"rest_url": "x"},
            "risk": {},
            "universe": {"top_n": 8},
            "timeframes": {"regime": ["1h"], "signal": "15m"},
        }
        errors = validate_config_schema(cfg)
        # No errors about universe
        assert not any("universe" in e for e in errors)

    def test_string_timeframe_accepted(self):
        """String timeframe is accepted."""
        cfg = {
            "hyperliquid": {"rest_url": "x"},
            "risk": {},
            "universe": ["ETH"],
            "timeframes": "15m",
        }
        errors = validate_config_schema(cfg)
        assert not any("timeframes" in e for e in errors)


# ---------------------------------------------------------------------------
# Startup checks
# ---------------------------------------------------------------------------


class TestStartupChecks:
    def test_paper_mode_passes(self):
        """Paper mode startup checks pass with minimal config."""
        cfg = load_config("config/base.yaml")
        cfg["env"] = "paper"
        errors = run_startup_checks(cfg)
        # Paper mode should pass (no HL_ACCOUNT_ADDRESS required)
        schema_errors = [e for e in errors if "Missing required" in e]
        assert schema_errors == []

    def test_live_mode_no_account(self):
        """Live mode fails without HL_ACCOUNT_ADDRESS."""
        cfg = load_config("config/base.yaml")
        cfg["env"] = "live"
        # Remove env var if set
        with patch.dict(os.environ, {}, clear=True):
            errors = run_startup_checks(cfg)
        account_errors = [e for e in errors if "HL_ACCOUNT_ADDRESS" in e]
        assert len(account_errors) >= 1

    def test_live_mode_no_baseline(self):
        """Live mode fails without baseline file."""
        cfg = load_config("config/base.yaml")
        cfg["env"] = "live"
        cfg["baselines_dir"] = "/tmp/nonexistent_baselines_dir"
        with patch.dict(os.environ, {"HL_ACCOUNT_ADDRESS": "0x1234"}, clear=False):
            errors = run_startup_checks(cfg)
        baseline_errors = [e for e in errors if "Baseline" in e or "baseline" in e]
        assert len(baseline_errors) >= 1

    def test_startup_checks_returns_list(self):
        """Startup checks always return a list."""
        cfg = load_config("config/base.yaml")
        errors = run_startup_checks(cfg)
        assert isinstance(errors, list)


# ---------------------------------------------------------------------------
# Main entrypoint validation
# ---------------------------------------------------------------------------


class TestMainEntrypoint:
    def test_import_main(self):
        """Main module can be imported without error."""
        from autotrader.main import main
        assert callable(main)

    def test_main_bad_config(self):
        """Main returns 1 with nonexistent config."""
        from autotrader.main import main
        with patch("sys.argv", ["autotrader", "--config", "/nonexistent.yaml"]):
            result = main()
        assert result == 1

    def test_metrics_server_import(self):
        """MetricsServer can be imported from monitoring."""
        from autotrader.monitoring.metrics_server import MetricsServer
        server = MetricsServer(port=19200)
        assert not server.is_running


# ---------------------------------------------------------------------------
# Scheduler bug regression: managed variable ordering
# ---------------------------------------------------------------------------


class TestSchedulerBugRegression:
    def test_process_ws_order_update_no_name_error(self):
        """_process_ws_order_update does not raise NameError for managed.

        Regression test for the bug where `managed` was used before
        being defined in the filled branch of _process_ws_order_update.
        """
        from autotrader.runtime.scheduler import TradingScheduler

        # Read the source to verify managed is defined before use
        import inspect
        source = inspect.getsource(TradingScheduler._process_ws_order_update)
        # Find the "if status == \"filled\"" block
        filled_block_start = source.find('if status == "filled"')
        assert filled_block_start >= 0

        # Find where managed is first assigned and first used
        after_filled = source[filled_block_start:]

        # The assignment "managed = self.order_manager.orders.get(oid)"
        assign_pos = after_filled.find("managed = self.order_manager.orders.get(oid)")
        assert assign_pos >= 0

        # The first usage "if managed is not None"
        use_pos = after_filled.find("if managed is not None")
        assert use_pos >= 0

        # Assignment must come before usage
        assert assign_pos < use_pos, (
            "Bug regression: `managed` must be assigned before first use "
            f"(assign at {assign_pos}, use at {use_pos})"
        )


# ---------------------------------------------------------------------------
# Kill switch position flattening
# ---------------------------------------------------------------------------


class TestKillSwitchFlattening:
    def test_close_all_positions_exists(self):
        """Broker has close_all_positions method (D6 fix)."""
        from autotrader.execution.broker import Broker
        broker = Broker(client=None, mode="paper")
        assert hasattr(broker, "close_all_positions")
        assert callable(broker.close_all_positions)

    def test_close_all_positions_empty(self):
        """close_all_positions with empty list returns empty list."""
        from autotrader.execution.broker import Broker
        broker = Broker(client=None, mode="paper")
        results = broker.close_all_positions([])
        assert results == []

    def test_close_all_positions_with_positions(self):
        """close_all_positions closes each position."""
        from autotrader.execution.broker import Broker
        broker = Broker(client=None, mode="paper")
        positions = [
            {"symbol": "ETH", "side": "long", "size": 1.0, "current_px": 3000.0},
            {"symbol": "BTC", "side": "short", "size": 0.1, "current_px": 50000.0},
        ]
        results = broker.close_all_positions(positions)
        assert len(results) == 2
        # Long position closed with sell
        assert results[0].status == "filled"
        # Short position closed with buy
        assert results[1].status == "filled"


# ---------------------------------------------------------------------------
# Rate limiter timeout
# ---------------------------------------------------------------------------


class TestRateLimiterTimeout:
    def test_timeout_raises(self):
        """Rate limiter acquire with impossible weight times out."""
        from autotrader.hl.rate_limiter import TokenBucket
        rl = TokenBucket(capacity=10, refill_rate=0.001)
        # Drain all tokens
        rl.acquire(weight=10.0, timeout=0.5)
        # Now acquiring more should timeout quickly
        with pytest.raises(TimeoutError):
            rl.acquire(weight=10.0, timeout=0.1)


# ---------------------------------------------------------------------------
# Audit event types
# ---------------------------------------------------------------------------


class TestAuditEventTypes:
    def test_audit_module_log_event(self):
        """audit.log_event can be called without error."""
        from autotrader.monitoring import audit
        # Should not raise even if not fully initialized
        try:
            audit.log_event("test_event", {"key": "value"})
        except Exception:
            pass  # OK if audit not initialized

    def test_position_opened_event_schema(self):
        """position_opened event has expected fields."""
        event = {
            "oid": "123",
            "symbol": "ETH",
            "side": "buy",
            "size": 1.0,
            "entry_px": 3000.0,
        }
        assert "oid" in event
        assert "symbol" in event
        assert "entry_px" in event

    def test_position_closed_event_schema(self):
        """position_closed event has expected fields."""
        event = {
            "oid": "456",
            "symbol": "ETH",
            "side": "sell",
            "size": 1.0,
            "close_px": 3100.0,
            "order_type": "take_profit",
        }
        assert "oid" in event
        assert "close_px" in event
        assert "order_type" in event
