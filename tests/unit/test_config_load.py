"""Tests for config loading, deep merge, and env-var substitution."""

from pathlib import Path

import pytest

from autotrader.utils.config import _substitute_env, deep_merge, load_config

# -----------------------------------------------------------------------
# deep_merge
# -----------------------------------------------------------------------


class TestDeepMerge:
    def test_deep_merge(self):
        """Flat-key override: override replaces base values, adds new keys."""
        base = {"a": 1, "b": 2, "c": 3}
        override = {"b": 20, "d": 40}
        result = deep_merge(base, override)

        assert result["a"] == 1  # unchanged from base
        assert result["b"] == 20  # overridden
        assert result["c"] == 3  # unchanged from base
        assert result["d"] == 40  # new key from override
        # Originals are not mutated
        assert base["b"] == 2
        assert "d" not in base

    def test_deep_merge_nested(self):
        """Nested dicts are recursively merged rather than replaced wholesale."""
        base = {
            "level1": {
                "keep": "original",
                "overwrite": "old",
                "nested": {"deep_keep": True, "deep_overwrite": 0},
            },
            "top_only": 99,
        }
        override = {
            "level1": {
                "overwrite": "new",
                "nested": {"deep_overwrite": 1, "deep_new": "hello"},
                "added": "fresh",
            }
        }
        result = deep_merge(base, override)

        assert result["top_only"] == 99
        assert result["level1"]["keep"] == "original"
        assert result["level1"]["overwrite"] == "new"
        assert result["level1"]["added"] == "fresh"
        assert result["level1"]["nested"]["deep_keep"] is True
        assert result["level1"]["nested"]["deep_overwrite"] == 1
        assert result["level1"]["nested"]["deep_new"] == "hello"

        # Originals untouched
        assert base["level1"]["overwrite"] == "old"
        assert "added" not in base["level1"]


# -----------------------------------------------------------------------
# load_config
# -----------------------------------------------------------------------


class TestLoadConfig:
    def test_load_config(self, monkeypatch):
        """Load config/base.yaml after setting required env vars; verify keys."""
        # Set env vars that base.yaml references with ${...} patterns
        monkeypatch.setenv("HL_ACCOUNT_ADDRESS", "0xTEST_ADDRESS")
        monkeypatch.setenv("HL_API_WALLET_PRIVATE_KEY", "0xTEST_KEY")
        monkeypatch.setenv("POSTGRES_DSN", "postgresql://test:test@localhost/test")
        # AUTOTRADER_ENV not set -> defaults to "dev"
        monkeypatch.delenv("AUTOTRADER_ENV", raising=False)

        config_path = Path("/home/user/hyperliquid-autotrader/config/base.yaml")
        cfg = load_config(config_path)

        # Top-level keys present
        assert "hyperliquid" in cfg
        assert "risk" in cfg
        assert "universe" in cfg
        assert "timeframes" in cfg
        assert "execution" in cfg
        assert "governance" in cfg
        assert "observability" in cfg
        assert "storage" in cfg

        # Env vars were substituted
        assert cfg["hyperliquid"]["account_address"] == "0xTEST_ADDRESS"
        assert cfg["hyperliquid"]["api_wallet_private_key"] == "0xTEST_KEY"
        assert cfg["storage"]["postgres_dsn"] == "postgresql://test:test@localhost/test"

        # Numeric values preserved
        assert cfg["risk"]["max_concurrent_positions"] == 4
        assert cfg["risk"]["daily_loss_limit_pct"] == 0.05

    def test_load_config_with_overrides(self, monkeypatch):
        """Runtime overrides are deep-merged on top of loaded config."""
        monkeypatch.setenv("HL_ACCOUNT_ADDRESS", "0xADDR")
        monkeypatch.setenv("HL_API_WALLET_PRIVATE_KEY", "0xKEY")
        monkeypatch.setenv("POSTGRES_DSN", "")
        monkeypatch.delenv("AUTOTRADER_ENV", raising=False)

        config_path = Path("/home/user/hyperliquid-autotrader/config/base.yaml")
        overrides = {
            "risk": {
                "max_concurrent_positions": 10,
                "custom_field": "injected",
            },
            "new_section": {"key": "value"},
        }
        cfg = load_config(config_path, overrides=overrides)

        # Overridden value
        assert cfg["risk"]["max_concurrent_positions"] == 10
        # New key injected inside existing section
        assert cfg["risk"]["custom_field"] == "injected"
        # Pre-existing key NOT in overrides still present
        assert cfg["risk"]["daily_loss_limit_pct"] == 0.05
        # Entirely new section
        assert cfg["new_section"]["key"] == "value"


# -----------------------------------------------------------------------
# Environment variable substitution
# -----------------------------------------------------------------------


class TestEnvSubstitution:
    def test_env_substitution_simple(self, monkeypatch):
        """${VAR} is replaced with the value of VAR from the environment."""
        monkeypatch.setenv("MY_TEST_VAR", "hello_world")
        result = _substitute_env("prefix_${MY_TEST_VAR}_suffix")
        assert result == "prefix_hello_world_suffix"

    def test_env_substitution_default(self, monkeypatch):
        """${VAR:default} uses the default when VAR is unset."""
        monkeypatch.delenv("NONEXISTENT_VAR_1234", raising=False)
        result = _substitute_env("${NONEXISTENT_VAR_1234:fallback_value}")
        assert result == "fallback_value"

    def test_env_substitution_default_overridden(self, monkeypatch):
        """When the env var IS set, its value overrides the default."""
        monkeypatch.setenv("MY_SET_VAR", "real_val")
        result = _substitute_env("${MY_SET_VAR:default_val}")
        assert result == "real_val"

    def test_env_substitution_empty_default(self, monkeypatch):
        """${VAR:} means the default is an empty string."""
        monkeypatch.delenv("MISSING_VAR_XYZ", raising=False)
        result = _substitute_env("start_${MISSING_VAR_XYZ:}_end")
        assert result == "start__end"

    def test_env_substitution_missing_no_default_raises(self, monkeypatch):
        """${VAR} with no default and VAR unset raises KeyError."""
        monkeypatch.delenv("ABSOLUTELY_MISSING", raising=False)
        with pytest.raises(KeyError, match="ABSOLUTELY_MISSING"):
            _substitute_env("${ABSOLUTELY_MISSING}")

    def test_env_substitution_in_nested_dict(self, monkeypatch):
        """Substitution recurses into dicts and lists."""
        monkeypatch.setenv("INNER", "replaced")
        data = {
            "a": "${INNER}",
            "b": [1, "${INNER}", {"c": "val_${INNER}"}],
            "d": 42,
        }
        result = _substitute_env(data)
        assert result["a"] == "replaced"
        assert result["b"][0] == 1
        assert result["b"][1] == "replaced"
        assert result["b"][2]["c"] == "val_replaced"
        assert result["d"] == 42

    def test_env_substitution_multiple_vars_in_one_string(self, monkeypatch):
        """Multiple ${VAR} tokens in a single string are all expanded."""
        monkeypatch.setenv("HOST", "localhost")
        monkeypatch.setenv("PORT", "5432")
        result = _substitute_env("postgres://${HOST}:${PORT}/db")
        assert result == "postgres://localhost:5432/db"
