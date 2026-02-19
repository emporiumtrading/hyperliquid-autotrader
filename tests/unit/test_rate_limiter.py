"""Tests for the token-bucket rate limiter."""

import threading
import time

import pytest

from autotrader.hl.rate_limiter import TokenBucket


class TestTokenBucket:
    def test_initial_capacity(self):
        """Default bucket starts with 1200 tokens (matching HL rate limit)."""
        bucket = TokenBucket()
        available = bucket.available()
        assert available == pytest.approx(1200.0, abs=1.0)

    def test_acquire_reduces_tokens(self):
        """Acquiring tokens reduces the available count by the requested weight."""
        bucket = TokenBucket(capacity=100.0, refill_rate=0.0)  # no refill
        initial = bucket.available()
        bucket.acquire(10)
        after = bucket.available()
        assert after == pytest.approx(initial - 10.0, abs=0.5)

    def test_try_acquire_success(self):
        """try_acquire returns True and consumes tokens when capacity allows."""
        bucket = TokenBucket(capacity=100.0, refill_rate=0.0)
        result = bucket.try_acquire(50.0)
        assert result is True
        assert bucket.available() == pytest.approx(50.0, abs=0.5)

    def test_try_acquire_failure(self):
        """try_acquire returns False when insufficient tokens are available."""
        bucket = TokenBucket(capacity=10.0, refill_rate=0.0)
        # Drain the bucket
        bucket.acquire(10.0)
        result = bucket.try_acquire(1.0)
        assert result is False
        # Available stays at ~0
        assert bucket.available() == pytest.approx(0.0, abs=0.5)

    def test_refill(self):
        """After sleeping, tokens refill according to refill_rate."""
        # Use a bucket with a very fast refill: 1000 tokens/sec
        bucket = TokenBucket(capacity=100.0, refill_rate=1000.0)
        # Drain completely
        bucket.acquire(100.0)
        assert bucket.available() < 5.0  # nearly zero

        # Sleep 0.05s => should refill ~50 tokens (1000 * 0.05)
        time.sleep(0.06)
        refilled = bucket.available()
        assert refilled >= 40.0  # conservative lower bound
        assert refilled <= 100.0  # capped at capacity

    def test_acquire_blocks(self):
        """acquire() blocks when insufficient tokens exist (verified with a small fast bucket)."""
        # Tiny bucket: capacity 5, refill 100/sec => refills fast
        bucket = TokenBucket(capacity=5.0, refill_rate=100.0)
        bucket.acquire(5.0)  # drain it

        blocked = threading.Event()
        completed = threading.Event()

        def blocking_acquire():
            blocked.set()
            bucket.acquire(3.0)  # needs 3 tokens, currently ~0
            completed.set()

        t = threading.Thread(target=blocking_acquire, daemon=True)
        t.start()

        blocked.wait(timeout=1.0)
        # Give it a moment -- the thread should eventually complete
        # since refill_rate=100/s means 3 tokens in 0.03s
        completed.wait(timeout=1.0)
        assert completed.is_set(), "acquire() should have completed after token refill"
        t.join(timeout=1.0)

    def test_acquire_exceeding_capacity_raises(self):
        """Requesting more tokens than the bucket capacity raises ValueError."""
        bucket = TokenBucket(capacity=10.0)
        with pytest.raises(ValueError, match="exceeds bucket capacity"):
            bucket.acquire(20.0)

    def test_custom_capacity(self):
        """A bucket with custom capacity reports that capacity on creation."""
        bucket = TokenBucket(capacity=500.0, refill_rate=10.0)
        assert bucket.available() == pytest.approx(500.0, abs=1.0)
