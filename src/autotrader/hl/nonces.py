"""Persistent nonce manager for Hyperliquid exchange requests.

Nonces must be strictly monotonically increasing and close to the current
wall-clock time in milliseconds.  This module guarantees monotonicity by
tracking the last issued nonce and persisting it to disk so that restarts
do not accidentally reuse a value.

Usage
-----
    from autotrader.hl import nonces

    nonces.init("data/nonces.json")   # optional; default path used otherwise
    nonce = nonces.get_next()
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time


class NonceManager:
    """Thread-safe, file-backed nonce generator.

    Parameters
    ----------
    persist_path : str
        Path to a JSON file where the last nonce is stored.  Parent
        directories are created automatically if they do not exist.
    """

    def __init__(self, persist_path: str = "data/nonces.json") -> None:
        self._persist_path = persist_path
        self._lock = threading.Lock()
        self._last_nonce: int = self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load(self) -> int:
        """Load the last nonce from the persistence file.

        Returns ``0`` when the file does not exist or cannot be read.
        """
        try:
            with open(self._persist_path, "r") as fh:
                data = json.load(fh)
                return int(data.get("last_nonce", 0))
        except (FileNotFoundError, json.JSONDecodeError, ValueError, KeyError):
            return 0

    def _save(self, nonce: int) -> None:
        """Atomically persist *nonce* to the backing file.

        Writes to a temporary file in the same directory then renames it
        so that a crash mid-write never leaves a corrupt file.
        """
        dir_name = os.path.dirname(self._persist_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        # Write to a temp file then atomically rename
        fd, tmp_path = tempfile.mkstemp(
            dir=dir_name or ".",
            prefix=".nonce_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump({"last_nonce": nonce}, fh)
            os.replace(tmp_path, self._persist_path)
        except BaseException:
            # Clean up the temp file on any failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_next(self) -> int:
        """Return the next valid nonce and persist it.

        The nonce is ``max(last_nonce + 1, current_time_ms)`` which keeps
        values close to real time while guaranteeing strict monotonicity.
        """
        with self._lock:
            current_time_ms = int(time.time() * 1000)
            nonce = max(self._last_nonce + 1, current_time_ms)
            self._last_nonce = nonce
            self._save(nonce)
            return nonce


# ----------------------------------------------------------------------
# Module-level singleton & convenience functions
# ----------------------------------------------------------------------

_manager: NonceManager | None = None


def init(persist_path: str = "data/nonces.json") -> None:
    """Initialise (or reinitialise) the module-level NonceManager."""
    global _manager
    _manager = NonceManager(persist_path=persist_path)


def get_next() -> int:
    """Return the next nonce from the module-level manager.

    Lazily initialises the manager with default settings if ``init()``
    has not been called.
    """
    global _manager
    if _manager is None:
        _manager = NonceManager()
    return _manager.get_next()
