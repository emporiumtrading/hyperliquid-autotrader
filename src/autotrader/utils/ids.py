"""ID generation for runs, orders, and trades."""

from __future__ import annotations

import secrets
import uuid
from datetime import datetime, timezone


def generate_run_id() -> str:
    """Generate a unique run identifier.

    Format: ``run_YYYYMMDD_HHMMSS_xxxx`` where *xxxx* is 4 random hex
    characters.

    >>> rid = generate_run_id()
    >>> rid.startswith("run_")
    True
    >>> len(rid)
    24
    """
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d_%H%M%S")
    rand = secrets.token_hex(2)  # 2 bytes -> 4 hex chars
    return f"run_{ts}_{rand}"


def generate_order_id() -> str:
    """Generate a unique order identifier.

    Format: ``ord_<12 hex chars>`` derived from a UUID4.

    >>> oid = generate_order_id()
    >>> oid.startswith("ord_")
    True
    >>> len(oid)
    16
    """
    return f"ord_{uuid.uuid4().hex[:12]}"


def generate_trade_id() -> str:
    """Generate a unique trade identifier.

    Format: ``trd_<12 hex chars>`` derived from a UUID4.

    >>> tid = generate_trade_id()
    >>> tid.startswith("trd_")
    True
    >>> len(tid)
    16
    """
    return f"trd_{uuid.uuid4().hex[:12]}"
