"""Persistent audit trail for order and trade events.

Writes structured JSONL (one JSON object per line) to a file on disk.
Every write is flushed immediately so that events survive unexpected exits.
All operations are thread-safe.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any


# Allowed event types for validation.
EVENT_TYPES: frozenset[str] = frozenset(
    {
        "order_placed",
        "order_filled",
        "order_cancelled",
        "position_opened",
        "position_closed",
        "kill_switch_triggered",
        "drift_detected",
    }
)


class AuditTrail:
    """Append-only JSONL audit log for trading events.

    Each record is a single JSON line containing at minimum:

    * ``timestamp_ms`` -- Unix epoch in milliseconds.
    * ``event_type`` -- One of :data:`EVENT_TYPES`.
    * ``data`` -- Arbitrary event-specific payload.

    Parameters
    ----------
    path:
        Filesystem path where the JSONL file will be created / appended to.
        Parent directories are created automatically if they do not exist.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()

        # Ensure the parent directory exists.
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        # Open the file in append mode so that existing entries are preserved.
        self._file = open(path, "a", encoding="utf-8")  # noqa: SIM115

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_event(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """Write a single audit record.

        Parameters
        ----------
        event_type:
            Must be one of :data:`EVENT_TYPES`.
        data:
            Optional dictionary of event-specific fields merged into the
            record.

        Raises
        ------
        ValueError
            If *event_type* is not recognised.
        """
        if event_type not in EVENT_TYPES:
            raise ValueError(
                f"Unknown event_type {event_type!r}. "
                f"Must be one of {sorted(EVENT_TYPES)}."
            )

        record: dict[str, Any] = {
            "timestamp_ms": _now_ms(),
            "event_type": event_type,
        }
        if data:
            record["data"] = data

        line = json.dumps(record, separators=(",", ":")) + "\n"

        with self._lock:
            self._file.write(line)
            self._file.flush()
            os.fsync(self._file.fileno())

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        with self._lock:
            self._file.flush()
            self._file.close()

    @property
    def path(self) -> str:
        """Return the filesystem path of the audit file."""
        return self._path

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> AuditTrail:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_ms() -> int:
    """Return the current UTC time as milliseconds since the Unix epoch."""
    return int(time.time() * 1000)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_trail: AuditTrail | None = None


def init(path: str) -> None:
    """Initialise the module-level :class:`AuditTrail` singleton.

    Parameters
    ----------
    path:
        Filesystem path for the JSONL audit file.
    """
    global _trail  # noqa: PLW0603
    _trail = AuditTrail(path=path)


def log_event(event_type: str, data: dict[str, Any] | None = None) -> None:
    """Write an audit record via the module-level singleton.

    If :func:`init` has not been called yet, a default audit trail is created
    at ``./audit_trail.jsonl`` in the current working directory.
    """
    global _trail  # noqa: PLW0603
    if _trail is None:
        _trail = AuditTrail(path="audit_trail.jsonl")
    _trail.log_event(event_type, data)
