"""Weight-based token bucket rate limiter for Hyperliquid REST API.

HL enforces ~1200 weight units per minute.  This module provides a thread-safe
token bucket that refills at 20 tokens/second (= 1200/min) and blocks callers
when the bucket is empty.

Usage
-----
    from autotrader.hl.rate_limiter import acquire

    acquire(weight=10)          # blocks until 10 tokens available
    ok = try_acquire(weight=5)  # non-blocking check
    print(available())          # current token count
"""

from __future__ import annotations

import threading
import time


class TokenBucket:
    """Thread-safe token bucket rate limiter.

    Parameters
    ----------
    capacity : float
        Maximum number of tokens the bucket can hold.  Defaults to 1200
        (matching the HL per-minute weight budget).
    refill_rate : float
        Tokens added per second.  Defaults to 20.0 (1200 / 60).
    """

    def __init__(
        self,
        capacity: float = 1200.0,
        refill_rate: float = 20.0,
    ) -> None:
        self._capacity = capacity
        self._refill_rate = refill_rate
        self._tokens: float = capacity
        self._last_refill: float = time.monotonic()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refill(self) -> None:
        """Recalculate available tokens based on elapsed time.

        Must be called while ``_lock`` is held.
        """
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed > 0:
            self._tokens = min(
                self._capacity,
                self._tokens + elapsed * self._refill_rate,
            )
            self._last_refill = now

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire(self, weight: float = 1.0, timeout: float = 60.0) -> None:
        """Block until *weight* tokens are available, then consume them.

        Parameters
        ----------
        weight : float
            Number of tokens (API weight units) to consume.
        timeout : float
            Maximum seconds to wait.  Raises :class:`TimeoutError` if the
            budget cannot be acquired within this period.  Defaults to 60 s.

        Raises
        ------
        ValueError
            If *weight* exceeds the bucket capacity (would never be satisfied).
        TimeoutError
            If tokens are not available within *timeout* seconds.
        """
        if weight > self._capacity:
            raise ValueError(f"Requested weight {weight} exceeds bucket capacity {self._capacity}")

        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= weight:
                    self._tokens -= weight
                    return
                # Calculate how long we need to wait for enough tokens
                deficit = weight - self._tokens
                wait_time = deficit / self._refill_rate

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Rate limit acquire timed out after {timeout}s "
                    f"waiting for {weight} tokens"
                )
            # Sleep outside the lock so other threads can proceed
            time.sleep(min(wait_time, remaining))

    def try_acquire(self, weight: float = 1.0) -> bool:
        """Try to consume *weight* tokens without blocking.

        Returns
        -------
        bool
            ``True`` if tokens were consumed, ``False`` otherwise.
        """
        with self._lock:
            self._refill()
            if self._tokens >= weight:
                self._tokens -= weight
                return True
            return False

    def available(self) -> float:
        """Return the current number of available tokens."""
        with self._lock:
            self._refill()
            return self._tokens


# ----------------------------------------------------------------------
# Module-level singleton & convenience functions
# ----------------------------------------------------------------------

_default_bucket: TokenBucket = TokenBucket()


def acquire(weight: float = 1.0) -> None:
    """Block until *weight* tokens are available (module-level singleton)."""
    _default_bucket.acquire(weight)


def try_acquire(weight: float = 1.0) -> bool:
    """Non-blocking acquire from the module-level singleton."""
    return _default_bucket.try_acquire(weight)


def available() -> float:
    """Return available tokens from the module-level singleton."""
    return _default_bucket.available()
