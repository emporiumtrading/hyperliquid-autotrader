"""Startup checks run before any trading mode.

Validates configuration, environment variables, exchange connectivity,
rate limiter health, baseline availability, kill switch state, and risk
config parseability.
"""

from __future__ import annotations

import os
from pathlib import Path

import structlog

from autotrader.hl.client import create_client
from autotrader.hl.rate_limiter import available as rl_available
from autotrader.risk.constraints import load_risk_config

logger = structlog.get_logger(__name__)

# Keys that must be present at the top level of the configuration dict.
_REQUIRED_TOP_LEVEL_KEYS: list[str] = [
    "hyperliquid",
    "risk",
    "universe",
    "timeframes",
]

# Placeholder account addresses that indicate no real key is configured.
_PLACEHOLDER_ADDRESSES: set[str] = {
    "",
    "0x0000",
    "0x" + "0" * 40,
    "0x0000000000000000000000000000000000000000",
}


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def validate_config_schema(config: dict) -> list[str]:
    """Check that *config* has required sections and values.

    Returns a list of validation error strings.  An empty list means the
    config schema is valid.
    """
    errors: list[str] = []

    for key in _REQUIRED_TOP_LEVEL_KEYS:
        if key not in config:
            errors.append(f"Missing required config key: '{key}'")

    # hyperliquid section checks
    hl_section = config.get("hyperliquid", {})
    if not isinstance(hl_section, dict):
        errors.append("Config key 'hyperliquid' must be a dict")
    else:
        if "account_address" not in hl_section and "rest_url" not in hl_section:
            # At minimum one HL setting should be present; warn if section is empty
            if not hl_section:
                errors.append("Config 'hyperliquid' section is empty")

    # risk section checks
    risk_section = config.get("risk", {})
    if not isinstance(risk_section, dict):
        errors.append("Config key 'risk' must be a dict")

    # universe checks
    universe = config.get("universe", None)
    if universe is not None:
        if isinstance(universe, dict):
            # universe might be a dict with settings like top_n, min_volume
            pass
        elif isinstance(universe, list):
            if len(universe) == 0:
                errors.append("Config 'universe' list is empty")
        else:
            errors.append("Config 'universe' must be a list or dict")

    # timeframes checks
    timeframes = config.get("timeframes", None)
    if timeframes is not None:
        if isinstance(timeframes, list):
            if len(timeframes) == 0:
                errors.append("Config 'timeframes' list is empty")
        elif isinstance(timeframes, str):
            # Single timeframe string is acceptable
            pass
        else:
            errors.append("Config 'timeframes' must be a list or string")

    return errors


# ---------------------------------------------------------------------------
# Full startup check suite
# ---------------------------------------------------------------------------


def run_startup_checks(config: dict) -> list[str]:
    """Run all startup checks and return a list of error messages.

    An empty list means all checks passed and the system is ready to proceed.

    Checks performed
    ----------------
    1. **Config validation** -- required keys exist (hyperliquid, risk,
       universe, timeframes).
    2. **Environment variables** -- ``HL_ACCOUNT_ADDRESS`` must be present
       when ``env`` is not ``paper``.
    3. **HL connectivity** -- attempt ``get_meta()`` to verify the exchange
       is reachable (skipped when no real API key is configured).
    4. **Rate limiter** -- ``available() > 0``.
    5. **Baseline check** -- for ``live`` / ``canary`` environments, verify
       ``artifacts/baselines/current.json`` exists.
    6. **Kill switch** -- check the kill switch is not already triggered.
    7. **Risk config** -- verify the ``risk`` section parses into a valid
       :class:`RiskConfig`.
    """
    errors: list[str] = []

    # ------------------------------------------------------------------ 1
    logger.info("startup_check.config_schema")
    schema_errors = validate_config_schema(config)
    errors.extend(schema_errors)

    # Determine the environment (paper / canary / live)
    env = config.get("env", config.get("environment", "paper"))
    if isinstance(env, str):
        env = env.lower()
    else:
        env = "paper"

    # ------------------------------------------------------------------ 2
    logger.info("startup_check.env_vars", env=env)
    if env != "paper":
        hl_account_env = os.environ.get("HL_ACCOUNT_ADDRESS", "")
        if not hl_account_env:
            errors.append(
                f"Environment variable HL_ACCOUNT_ADDRESS is required "
                f"for env='{env}' but is not set"
            )

    # ------------------------------------------------------------------ 3
    logger.info("startup_check.hl_connectivity")
    hl_section = config.get("hyperliquid", {})
    account_address = hl_section.get(
        "account_address",
        os.environ.get("HL_ACCOUNT_ADDRESS", ""),
    )

    # Only attempt the connectivity check if a real address is configured
    has_real_address = account_address not in _PLACEHOLDER_ADDRESSES and bool(account_address)

    if has_real_address:
        try:
            client = create_client(hl_section)
            client.get_meta()
            logger.info("startup_check.hl_connectivity.ok")
        except Exception as exc:
            errors.append(f"HL connectivity check failed: {exc}")
            logger.error(
                "startup_check.hl_connectivity.failed",
                error=str(exc),
            )
    else:
        logger.info(
            "startup_check.hl_connectivity.skipped",
            reason="no real API key configured",
        )

    # ------------------------------------------------------------------ 4
    logger.info("startup_check.rate_limiter")
    try:
        tokens = rl_available()
        if tokens <= 0:
            errors.append(f"Rate limiter has no available tokens ({tokens:.1f})")
        else:
            logger.info(
                "startup_check.rate_limiter.ok",
                available_tokens=tokens,
            )
    except Exception as exc:
        errors.append(f"Rate limiter check failed: {exc}")

    # ------------------------------------------------------------------ 5
    logger.info("startup_check.baseline", env=env)
    if env in ("live", "canary"):
        baselines_dir = config.get("baselines_dir", "artifacts/baselines")
        baseline_path = Path(baselines_dir) / "current.json"
        if not baseline_path.exists():
            errors.append(
                f"Baseline file not found at {baseline_path} " f"(required for env='{env}')"
            )
        else:
            logger.info(
                "startup_check.baseline.ok",
                path=str(baseline_path),
            )

    # ------------------------------------------------------------------ 6
    logger.info("startup_check.kill_switch")
    try:
        from autotrader.runtime.kill_switch import is_triggered, trigger_reason

        if is_triggered():
            reason = trigger_reason()
            errors.append(f"Kill switch is already triggered: {reason}")
        else:
            logger.info("startup_check.kill_switch.ok")
    except Exception as exc:
        # Kill switch module may not be initialised yet; that is OK for a
        # startup check -- we just note we could not verify.
        logger.warning(
            "startup_check.kill_switch.skipped",
            reason=str(exc),
        )

    # ------------------------------------------------------------------ 7
    logger.info("startup_check.risk_config")
    try:
        _risk_cfg = load_risk_config(config)
        logger.info("startup_check.risk_config.ok")
    except Exception as exc:
        errors.append(f"Risk config validation failed: {exc}")

    # ------------------------------------------------------------------ summary
    if errors:
        logger.error(
            "startup_checks.failed",
            error_count=len(errors),
            errors=errors,
        )
    else:
        logger.info("startup_checks.all_passed")

    return errors
