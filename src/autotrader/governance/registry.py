"""Baseline registry: load/save current.json and history."""

from __future__ import annotations

from pathlib import Path

import structlog

from autotrader.utils.ids import generate_run_id
from autotrader.utils.serialization import load_json, save_json
from autotrader.utils.time import ms_to_datetime, now_ms

logger = structlog.get_logger(__name__)


class BaselineRegistry:
    """Manages the current baseline and its version history.

    The registry stores:
    * ``current.json`` -- the active baseline that candidates are compared against.
    * ``history/<timestamp>.json`` -- immutable snapshots for every promoted baseline.
    """

    def __init__(self, baselines_dir: str = "artifacts/baselines") -> None:
        self.baselines_dir = Path(baselines_dir)
        self.current_path = self.baselines_dir / "current.json"
        self.history_dir = self.baselines_dir / "history"

    # ------------------------------------------------------------------
    # Current baseline
    # ------------------------------------------------------------------

    def load_current(self) -> dict | None:
        """Load *current.json* and return its contents.

        Returns ``None`` if the file does not exist (i.e. no baseline has
        been established yet).
        """
        if not self.current_path.exists():
            logger.info("baseline.load_current.not_found", path=str(self.current_path))
            return None
        try:
            data = load_json(self.current_path)
            logger.info("baseline.load_current.ok", version=data.get("version"))
            return data
        except Exception:
            logger.exception("baseline.load_current.error", path=str(self.current_path))
            return None

    def save_current(self, baseline: dict) -> str:
        """Persist *baseline* as the active baseline and archive a history copy.

        Parameters
        ----------
        baseline:
            The baseline dictionary (as returned by :meth:`create_baseline`).

        Returns
        -------
        str
            The path to the history snapshot file.
        """
        # Ensure directories exist
        self.baselines_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

        # Save current
        save_json(baseline, self.current_path)

        # Archive to history with a timestamp-based filename
        ts = ms_to_datetime(now_ms()).strftime("%Y%m%dT%H%M%S")
        version = baseline.get("version", 0)
        history_filename = f"{ts}_v{version}.json"
        history_path = self.history_dir / history_filename
        save_json(baseline, history_path)

        logger.info(
            "baseline.save_current.ok",
            version=version,
            history_path=str(history_path),
        )
        return str(history_path)

    # ------------------------------------------------------------------
    # Baseline creation
    # ------------------------------------------------------------------

    def create_baseline(
        self,
        strategy_name: str,
        strategy_config: dict,
        metrics: dict,
        dataset_hash: str,
        git_commit: str = "",
        run_id: str = "",
    ) -> dict:
        """Build a baseline dictionary with an auto-incrementing version.

        The version number is derived from the current baseline: if one
        exists its version is incremented by one, otherwise we start at 1.

        Parameters
        ----------
        strategy_name:
            Human-readable strategy identifier (e.g. ``"trend_breakout"``).
        strategy_config:
            Full strategy configuration dictionary.
        metrics:
            Summary metrics from the backtest/evaluation.
        dataset_hash:
            SHA-256 hash of the dataset manifest used for evaluation.
        git_commit:
            Optional git commit SHA for traceability.
        run_id:
            Optional run identifier; generated automatically if empty.

        Returns
        -------
        dict
            The baseline document ready for :meth:`save_current`.
        """
        current = self.load_current()
        if current is not None:
            version = current.get("version", 0) + 1
        else:
            version = 1

        if not run_id:
            run_id = generate_run_id()

        baseline = {
            "version": version,
            "created_at": now_ms(),
            "strategy_name": strategy_name,
            "strategy_config": strategy_config,
            "metrics_summary": metrics,
            "dataset_hash": dataset_hash,
            "git_commit": git_commit,
            "run_id": run_id,
        }

        logger.info(
            "baseline.create",
            version=version,
            strategy_name=strategy_name,
            run_id=run_id,
        )
        return baseline

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def list_history(self) -> list[dict]:
        """Load all history snapshots and return them sorted by ``created_at``.

        Returns
        -------
        list[dict]
            List of baseline dictionaries in chronological order (oldest first).
        """
        if not self.history_dir.exists():
            return []

        entries: list[dict] = []
        for path in sorted(self.history_dir.glob("*.json")):
            try:
                entry = load_json(path)
                entries.append(entry)
            except Exception:
                logger.warning("baseline.history.skip_corrupt", path=str(path))
                continue

        # Sort by created_at (millisecond timestamp)
        entries.sort(key=lambda e: e.get("created_at", 0))
        return entries

    def rollback(self, version: int | None = None) -> dict:
        """Restore a previous baseline version.

        Parameters
        ----------
        version:
            The version number to restore.  If ``None``, the version
            immediately before the current one is used.

        Returns
        -------
        dict
            The restored baseline dictionary.

        Raises
        ------
        ValueError
            If the requested version cannot be found in history or there
            is no previous version to roll back to.
        """
        history = self.list_history()
        if not history:
            raise ValueError("No history available for rollback")

        if version is not None:
            # Find the exact version
            target = None
            for entry in history:
                if entry.get("version") == version:
                    target = entry
                    break
            if target is None:
                available = [e.get("version") for e in history]
                raise ValueError(
                    f"Version {version} not found in history. " f"Available versions: {available}"
                )
        else:
            # Find the version before the current one
            current = self.load_current()
            if current is None:
                raise ValueError("No current baseline to roll back from")
            current_version = current.get("version", 0)
            target = None
            for entry in reversed(history):
                if entry.get("version", 0) < current_version:
                    target = entry
                    break
            if target is None:
                raise ValueError(
                    f"No version prior to current version {current_version} " f"found in history"
                )

        # Save the target as the new current
        self.baselines_dir.mkdir(parents=True, exist_ok=True)
        save_json(target, self.current_path)

        logger.info(
            "baseline.rollback.ok",
            restored_version=target.get("version"),
        )
        return target


# ---------------------------------------------------------------------------
# Module-level convenience functions (backwards compatibility)
# ---------------------------------------------------------------------------

_default_registry: BaselineRegistry | None = None


def _get_registry(path: str) -> BaselineRegistry:
    """Return a registry instance for the given path."""
    global _default_registry
    if _default_registry is None or str(_default_registry.baselines_dir) != path:
        _default_registry = BaselineRegistry(baselines_dir=path)
    return _default_registry


def load_current(path: str = "artifacts/baselines") -> dict:
    """Load the current baseline from *path*.

    Returns an empty dict when no baseline exists (preserving the original
    function signature that always returns a dict).
    """
    registry = _get_registry(path)
    result = registry.load_current()
    return result if result is not None else {}


def save_current(path: str = "artifacts/baselines", data: dict = {}) -> None:
    """Save *data* as the current baseline at *path*."""
    registry = _get_registry(path)
    registry.save_current(data)
