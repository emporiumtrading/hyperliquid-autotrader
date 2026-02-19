"""Deterministic dataset hashing for reproducibility.

Every dataset used in backtesting or training is uniquely identified by a
*manifest* -- a small JSON document describing which symbols, timeframes,
and time range were included.  :func:`compute_hash` produces a SHA-256
digest of a canonically-encoded manifest so that results can be traced back
to the exact input data.
"""

from __future__ import annotations

import hashlib
import json

from autotrader.utils.serialization import load_json, save_json


def compute_hash(manifest: dict) -> str:
    """Compute a deterministic SHA-256 hex digest for *manifest*.

    Keys are sorted recursively to ensure the same logical manifest always
    produces the same hash regardless of insertion order.

    Parameters
    ----------
    manifest:
        Dictionary with keys such as ``symbols``, ``timeframes``,
        ``start_ms``, ``end_ms``, ``source``.

    Returns
    -------
    str
        64-character lowercase hex digest.
    """
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def create_manifest(
    symbols: list[str],
    timeframes: list[str],
    start_ms: int,
    end_ms: int,
    source: str = "hyperliquid",
) -> dict:
    """Create a normalised manifest dictionary.

    Symbols and timeframes are sorted alphabetically so that different
    orderings of the same inputs produce the same manifest (and therefore
    the same hash).

    Parameters
    ----------
    symbols:
        List of asset symbols (e.g. ``["ETH", "BTC"]``).
    timeframes:
        List of bar intervals (e.g. ``["1h", "15m"]``).
    start_ms:
        Start timestamp in milliseconds.
    end_ms:
        End timestamp in milliseconds.
    source:
        Data source identifier.

    Returns
    -------
    dict
        Manifest dictionary ready for hashing or persistence.
    """
    return {
        "symbols": sorted(symbols),
        "timeframes": sorted(timeframes),
        "start_ms": start_ms,
        "end_ms": end_ms,
        "source": source,
    }


def save_manifest(manifest: dict, path: str) -> str:
    """Persist *manifest* to a JSON file and return its hash.

    Parameters
    ----------
    manifest:
        Manifest dictionary.
    path:
        Destination file path.

    Returns
    -------
    str
        SHA-256 hex digest of the manifest.
    """
    digest = compute_hash(manifest)
    # Attach the hash inside the saved file for convenience
    payload = {**manifest, "_hash": digest}
    save_json(payload, path)
    return digest


def load_manifest(path: str) -> tuple[dict, str]:
    """Load a manifest from disk and verify / compute its hash.

    Parameters
    ----------
    path:
        Path to the manifest JSON file.

    Returns
    -------
    tuple[dict, str]
        ``(manifest, hash)`` where *manifest* is the dict (without the
        stored ``_hash`` key) and *hash* is the freshly computed digest.
    """
    payload = load_json(path)
    # Remove the stored hash before computing so the digest is stable
    stored = dict(payload)
    stored.pop("_hash", None)
    digest = compute_hash(stored)
    return stored, digest
