"""Config loading with YAML support, environment variable substitution, and deep merge."""

from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

# Pattern to match ${ENV_VAR} or ${ENV_VAR:default_value}
_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")


# ---------------------------------------------------------------------------
# Environment variable substitution
# ---------------------------------------------------------------------------


def _substitute_env(value: Any) -> Any:
    """Recursively walk *value* and replace ``${ENV_VAR}`` patterns with the
    corresponding environment variable.  Supports a default via
    ``${ENV_VAR:default}``.

    If *value* is a ``str``, every ``${...}`` token is expanded.  If the
    environment variable is unset and no default is given, a :class:`KeyError`
    is raised.

    Non-string scalars (int, float, bool, None) are returned as-is.  Dicts
    and lists are traversed recursively.
    """
    if isinstance(value, str):

        def _replace(match: re.Match) -> str:
            var_name = match.group(1)
            default = match.group(2)  # None if no default was specified
            env_val = os.environ.get(var_name)
            if env_val is not None:
                return env_val
            if default is not None:
                return default
            raise KeyError(
                f"Environment variable {var_name!r} is not set and no "
                f"default was provided (referenced as ${{{var_name}}})"
            )

        return _ENV_PATTERN.sub(_replace, value)

    if isinstance(value, dict):
        return {k: _substitute_env(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_substitute_env(item) for item in value]

    return value


# ---------------------------------------------------------------------------
# Deep merge
# ---------------------------------------------------------------------------


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively deep-merge *override* into *base* and return a **new** dict.

    - Dict values are merged recursively.
    - All other types in *override* replace the corresponding key in *base*.
    - Neither *base* nor *override* are mutated.

    Parameters
    ----------
    base:
        The base dictionary.
    override:
        Dictionary whose values take precedence.

    Returns
    -------
    dict
        A new merged dictionary.
    """
    merged = deepcopy(base)
    for key, over_val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(over_val, dict):
            merged[key] = deep_merge(merged[key], over_val)
        else:
            merged[key] = deepcopy(over_val)
    return merged


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    """Load a YAML configuration file, substitute environment variables, and
    optionally apply runtime overrides.

    Workflow
    --------
    1. Read the YAML file at *path*.
    2. If the top-level dict contains an ``_env`` key whose value matches the
       ``AUTOTRADER_ENV`` environment variable (or ``"dev"`` by default), the
       dict under ``environments.<env>`` is deep-merged onto the base config.
       The ``environments`` and ``_env`` keys are then removed.
    3. Every string value containing ``${VAR}`` or ``${VAR:default}`` is
       replaced with the corresponding environment variable.
    4. If *overrides* is provided it is deep-merged on top.

    Parameters
    ----------
    path:
        Path to a YAML config file.
    overrides:
        Optional dict of runtime overrides to deep-merge last.

    Returns
    -------
    dict
        The fully resolved configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    KeyError
        If an ``${ENV_VAR}`` reference cannot be resolved.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise TypeError(f"Expected top-level YAML dict, got {type(raw).__name__}")

    # --- environment-specific merge ---
    env_name = raw.pop("_env", None) or os.environ.get("AUTOTRADER_ENV", "dev")
    environments: dict = raw.pop("environments", {})
    if isinstance(environments, dict) and env_name in environments:
        env_block = environments[env_name]
        if isinstance(env_block, dict):
            raw = deep_merge(raw, env_block)

    # --- env-var substitution ---
    raw = _substitute_env(raw)  # type: ignore[assignment]

    # --- runtime overrides ---
    if overrides:
        raw = deep_merge(raw, overrides)

    return raw
