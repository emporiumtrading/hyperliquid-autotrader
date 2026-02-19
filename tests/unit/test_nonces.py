"""Tests for the persistent nonce manager."""

import json
import os
import time

import pytest

from autotrader.hl.nonces import NonceManager


@pytest.fixture
def nonce_path(tmp_path):
    """Return a temporary file path for nonce persistence."""
    return str(tmp_path / "test_nonces.json")


class TestNonceManager:
    def test_get_next_monotonic(self, nonce_path):
        """Multiple calls to get_next return strictly increasing nonces."""
        mgr = NonceManager(persist_path=nonce_path)
        nonces = [mgr.get_next() for _ in range(50)]

        # Every nonce must be strictly greater than the previous one
        for i in range(1, len(nonces)):
            assert nonces[i] > nonces[i - 1], (
                f"Nonce at index {i} ({nonces[i]}) is not strictly greater "
                f"than nonce at index {i-1} ({nonces[i-1]})"
            )

    def test_persistence(self, nonce_path):
        """A new NonceManager with the same path continues from the last nonce."""
        mgr1 = NonceManager(persist_path=nonce_path)
        mgr1.get_next()
        mgr1.get_next()
        last_from_first = mgr1.get_next()

        # Create a brand-new manager pointing to the same file
        mgr2 = NonceManager(persist_path=nonce_path)
        n4 = mgr2.get_next()

        # Must be strictly greater than the last nonce from the first manager
        assert n4 > last_from_first, (
            f"Nonce from second manager ({n4}) should be > "
            f"last nonce from first manager ({last_from_first})"
        )

    def test_time_based(self, nonce_path):
        """Nonces should be close to the current time in milliseconds."""
        mgr = NonceManager(persist_path=nonce_path)
        before_ms = int(time.time() * 1000)
        nonce = mgr.get_next()
        after_ms = int(time.time() * 1000)

        # The nonce is max(last+1, current_time_ms), so it should be
        # within a small window of current time (allow 2 seconds tolerance).
        assert (
            nonce >= before_ms - 1
        ), f"Nonce {nonce} is too far in the past (before_ms={before_ms})"
        assert (
            nonce <= after_ms + 2000
        ), f"Nonce {nonce} is too far in the future (after_ms={after_ms})"

    def test_file_created(self, nonce_path):
        """The nonce file is created on the first get_next call."""
        assert not os.path.exists(nonce_path)
        mgr = NonceManager(persist_path=nonce_path)
        mgr.get_next()
        assert os.path.exists(nonce_path)

        with open(nonce_path, "r") as f:
            data = json.load(f)
        assert "last_nonce" in data
        assert isinstance(data["last_nonce"], int)

    def test_missing_file_starts_at_zero(self, nonce_path):
        """When the persistence file does not exist, _load returns 0."""
        mgr = NonceManager(persist_path=nonce_path)
        # First nonce should be based on current time (since last is 0)
        n = mgr.get_next()
        now_ms = int(time.time() * 1000)
        assert n >= now_ms - 2000  # within 2 seconds
