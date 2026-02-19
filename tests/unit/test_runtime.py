"""Tests for autotrader.runtime -- config validation, startup checks, kill switch."""

from __future__ import annotations

import json

from autotrader.risk.constraints import RiskConfig, RiskState
from autotrader.runtime.kill_switch import KillSwitch
from autotrader.runtime.startup_checks import validate_config_schema

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_config() -> dict:
    """Return a config dict that passes schema validation."""
    return {
        "hyperliquid": {"account_address": "0xabc"},
        "risk": {
            "max_drawdown_pct": 0.18,
            "max_leverage": 20.0,
        },
        "universe": ["ETH", "BTC"],
        "timeframes": ["15m"],
    }


# ===========================================================================
# Config schema validation
# ===========================================================================


class TestValidateConfigSchemaValid:
    def test_validate_config_schema_valid(self):
        """A valid config with all required sections passes."""
        config = _valid_config()
        errors = validate_config_schema(config)
        assert errors == []


class TestValidateConfigSchemaMissing:
    def test_validate_config_schema_missing(self):
        """Missing required sections are reported as errors."""
        # Missing "risk" and "universe"
        config = {
            "hyperliquid": {"account_address": "0xabc"},
            "timeframes": ["15m"],
        }
        errors = validate_config_schema(config)
        assert len(errors) >= 2
        error_text = " ".join(errors)
        assert "risk" in error_text.lower()
        assert "universe" in error_text.lower()

        # Completely empty config
        errors_empty = validate_config_schema({})
        assert len(errors_empty) >= 4  # all four required keys missing


class TestStartupChecksPaper:
    def test_startup_checks_paper(self):
        """Paper mode with valid config passes the schema validation at least."""
        config = _valid_config()
        config["env"] = "paper"
        errors = validate_config_schema(config)
        # Schema validation should pass (full startup checks need exchange, etc.)
        assert errors == []


# ===========================================================================
# Kill switch
# ===========================================================================


class TestKillSwitchTriggerReset:
    def test_kill_switch_trigger_reset(self, tmp_path):
        """Trigger and reset cycle works correctly."""
        state_file = str(tmp_path / "ks.json")
        ks = KillSwitch(state_path=state_file)

        # Initially not triggered
        assert ks.is_triggered() is False
        assert ks.trigger_reason() == ""

        # Trigger
        ks.trigger("test reason")
        assert ks.is_triggered() is True
        assert ks.trigger_reason() == "test reason"
        assert ks.trigger_time() > 0

        # Reset
        ks.reset()
        assert ks.is_triggered() is False
        assert ks.trigger_reason() == ""
        assert ks.trigger_time() == 0


class TestKillSwitchPersistence:
    def test_kill_switch_persistence(self, tmp_path):
        """Triggered state persists to file and is loaded by a new instance."""
        state_file = str(tmp_path / "ks.json")

        # Create and trigger
        ks1 = KillSwitch(state_path=state_file)
        ks1.trigger("persistent reason")
        assert ks1.is_triggered() is True

        # Create new instance from same file
        ks2 = KillSwitch(state_path=state_file)
        assert ks2.is_triggered() is True
        assert ks2.trigger_reason() == "persistent reason"

        # Verify the JSON file contents
        with open(state_file) as f:
            data = json.load(f)
        assert data["triggered"] is True
        assert data["reason"] == "persistent reason"

        # Reset via second instance
        ks2.reset()

        # Third instance should see cleared state
        ks3 = KillSwitch(state_path=state_file)
        assert ks3.is_triggered() is False


class TestKillSwitchCheckDrawdown:
    def test_kill_switch_check_drawdown(self, tmp_path):
        """Drawdown exceeding max_drawdown_pct triggers the kill switch."""
        state_file = str(tmp_path / "ks.json")
        ks = KillSwitch(state_path=state_file)

        risk_config = RiskConfig(
            max_drawdown_pct=0.10,
            daily_loss_limit_pct=0.05,
            weekly_loss_limit_pct=0.12,
        )

        # Normal state -- no trigger
        normal_state = RiskState(
            equity=9_500.0,
            peak_equity=10_000.0,
            daily_pnl=-100.0,
            weekly_pnl=-200.0,
        )
        result_normal = ks.check_conditions(normal_state, risk_config)
        # drawdown = 1 - 9500/10000 = 0.05 < 0.10 => should be False
        assert result_normal is False
        assert ks.is_triggered() is False

        # Breached state -- drawdown > 10%
        bad_state = RiskState(
            equity=8_800.0,
            peak_equity=10_000.0,
            daily_pnl=-50.0,
            weekly_pnl=-100.0,
        )
        result_bad = ks.check_conditions(bad_state, risk_config)
        # drawdown = 1 - 8800/10000 = 0.12 > 0.10 => should trigger
        assert result_bad is True
        assert ks.is_triggered() is True
        assert "drawdown" in ks.trigger_reason().lower()
