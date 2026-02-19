"""In-memory metrics collection with optional Prometheus-style export."""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any

import numpy as np


class MetricsCollector:
    """Thread-safe in-memory metrics store.

    Supports three metric types:
    * **counters** -- monotonically increasing values (e.g. total trades).
    * **gauges** -- point-in-time values (e.g. current PnL).
    * **histograms** -- observed sample lists for statistical queries.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def inc_counter(self, name: str, value: float = 1.0) -> None:
        """Increment a counter by *value* (default 1)."""
        with self._lock:
            self._counters[name] += value

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge to an absolute *value*."""
        with self._lock:
            self._gauges[name] = value

    def observe(self, name: str, value: float) -> None:
        """Record an observation in a histogram."""
        with self._lock:
            self._histograms[name].append(value)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_counter(self, name: str) -> float:
        """Return the current counter value (0.0 if not yet incremented)."""
        with self._lock:
            return self._counters.get(name, 0.0)

    def get_gauge(self, name: str) -> float:
        """Return the current gauge value (0.0 if not set)."""
        with self._lock:
            return self._gauges.get(name, 0.0)

    def get_histogram_stats(self, name: str) -> dict[str, Any]:
        """Return summary statistics for a histogram.

        Keys: count, sum, mean, p50, p95, p99, min, max.
        Returns zeros / NaNs when no observations have been recorded.
        """
        with self._lock:
            samples = list(self._histograms.get(name, []))

        if not samples:
            return {
                "count": 0,
                "sum": 0.0,
                "mean": float("nan"),
                "p50": float("nan"),
                "p95": float("nan"),
                "p99": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
            }

        arr = np.array(samples, dtype=np.float64)
        return {
            "count": len(arr),
            "sum": float(np.sum(arr)),
            "mean": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    # ------------------------------------------------------------------
    # Snapshot / Reset
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Return a full copy of all metric state.

        The returned dict has keys ``counters``, ``gauges``, ``histograms``.
        Histograms are returned as summary stats (not raw samples).
        """
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            histogram_names = list(self._histograms.keys())

        histogram_stats = {name: self.get_histogram_stats(name) for name in histogram_names}

        return {
            "counters": counters,
            "gauges": gauges,
            "histograms": histogram_stats,
        }

    def reset(self) -> None:
        """Clear all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

metrics = MetricsCollector()
