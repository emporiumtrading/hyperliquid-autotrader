"""Serialization helpers for JSON with support for datetime, numpy, and pandas types."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Custom JSON encoder
# ---------------------------------------------------------------------------


class _ExtendedEncoder(json.JSONEncoder):
    """JSON encoder that handles common scientific-Python and datetime types."""

    def default(self, o: Any) -> Any:
        # datetime / date
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, date):
            return o.isoformat()

        # numpy scalar types (imported lazily to avoid hard dep at module level)
        try:
            import numpy as np

            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.bool_):
                return bool(o)
        except ImportError:
            pass

        # pandas types
        try:
            import pandas as pd

            if isinstance(o, pd.Timestamp):
                return o.isoformat()
            if isinstance(o, pd.Timedelta):
                return o.total_seconds()
            if isinstance(o, (pd.Series, pd.DataFrame)):
                return o.to_dict()
            if pd.isna(o):
                return None
        except (ImportError, TypeError, ValueError):
            pass

        # dataclasses
        try:
            import dataclasses

            if dataclasses.is_dataclass(o) and not isinstance(o, type):
                return dataclasses.asdict(o)
        except ImportError:
            pass

        # Fallback: convert to string rather than raising TypeError.
        try:
            return super().default(o)
        except TypeError:
            return str(o)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def to_json(obj: Any, *, indent: int | None = 2) -> str:
    """Serialise *obj* to a JSON string.

    Handles :class:`~datetime.datetime`, numpy scalars / arrays, pandas
    Timestamps / Series / DataFrames, and frozen dataclasses.

    Parameters
    ----------
    obj:
        The object to serialise.
    indent:
        Pretty-print indentation.  Pass ``None`` for compact output.

    Returns
    -------
    str
        JSON string.
    """
    return json.dumps(obj, cls=_ExtendedEncoder, indent=indent)


def from_json(s: str) -> Any:
    """Deserialise a JSON string into Python objects.

    Parameters
    ----------
    s:
        JSON-encoded string.

    Returns
    -------
    Any
        Parsed Python object (dict, list, str, int, float, bool, or None).
    """
    return json.loads(s)


def save_json(obj: Any, path: str | Path, *, indent: int | None = 2) -> None:
    """Serialise *obj* and write it to *path* atomically.

    The file is first written to a temporary file in the same directory, then
    renamed into place.  This avoids partial/corrupt writes on crash.

    Parameters
    ----------
    obj:
        The object to serialise.
    path:
        Destination file path.
    indent:
        Pretty-print indentation.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = to_json(obj, indent=indent)

    # Write to temp file in the same directory, then atomic rename.
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        # Clean up temp file on any error.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_json(path: str | Path) -> Any:
    """Read and deserialise a JSON file.

    Parameters
    ----------
    path:
        Path to the JSON file.

    Returns
    -------
    Any
        Parsed Python object.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    json.JSONDecodeError
        If the file contains invalid JSON.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)
