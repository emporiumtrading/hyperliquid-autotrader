"""Tests for correlation-based exposure clustering in ExposureTracker."""

from __future__ import annotations

import numpy as np
import pytest

from autotrader.risk.exposure import (
    ExposureTracker,
    Position,
    ReturnTracker,
    _correlation_clusters,
)


# ---------------------------------------------------------------------------
# ReturnTracker
# ---------------------------------------------------------------------------


class TestReturnTracker:
    def test_record_and_retrieve(self):
        """Recorded returns are available in the correlation matrix."""
        tracker = ReturnTracker(lookback=50, min_obs=5)
        for i in range(10):
            tracker.record("ETH", float(i) * 0.01)
            tracker.record("BTC", float(i) * 0.012)

        corr = tracker.correlation_matrix(["ETH", "BTC"])
        assert corr is not None
        assert corr.shape == (2, 2)
        # Diagonal should be 1.0
        np.testing.assert_allclose(corr[0, 0], 1.0, atol=1e-6)
        np.testing.assert_allclose(corr[1, 1], 1.0, atol=1e-6)

    def test_returns_none_when_too_few_observations(self):
        """Returns None when fewer than min_obs observations exist."""
        tracker = ReturnTracker(lookback=50, min_obs=20)
        for i in range(5):
            tracker.record("ETH", 0.01)
            tracker.record("BTC", 0.02)

        assert tracker.correlation_matrix(["ETH", "BTC"]) is None

    def test_returns_none_when_single_symbol(self):
        """Need at least 2 symbols for a correlation matrix."""
        tracker = ReturnTracker(lookback=50, min_obs=3)
        for i in range(10):
            tracker.record("ETH", 0.01)

        assert tracker.correlation_matrix(["ETH"]) is None

    def test_lookback_caps_window(self):
        """Deque should cap at lookback length."""
        tracker = ReturnTracker(lookback=5, min_obs=3)
        for i in range(20):
            tracker.record("ETH", float(i))

        assert len(tracker._returns["ETH"]) == 5

    def test_highly_correlated_series(self):
        """Two identical return series should have correlation ~1.0."""
        tracker = ReturnTracker(lookback=100, min_obs=10)
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 50)
        for r in returns:
            tracker.record("A", float(r))
            tracker.record("B", float(r))  # identical

        corr = tracker.correlation_matrix(["A", "B"])
        assert corr is not None
        np.testing.assert_allclose(corr[0, 1], 1.0, atol=1e-6)

    def test_uncorrelated_series(self):
        """Independent random series should have low correlation."""
        tracker = ReturnTracker(lookback=500, min_obs=10)
        rng = np.random.default_rng(123)
        for _ in range(400):
            tracker.record("X", float(rng.normal(0, 1)))
            tracker.record("Y", float(rng.normal(0, 1)))

        corr = tracker.correlation_matrix(["X", "Y"])
        assert corr is not None
        assert abs(corr[0, 1]) < 0.15  # should be near zero


# ---------------------------------------------------------------------------
# _correlation_clusters
# ---------------------------------------------------------------------------


class TestCorrelationClusters:
    def test_two_correlated_symbols(self):
        """Two highly correlated symbols form a single cluster."""
        corr = np.array([[1.0, 0.8], [0.8, 1.0]])
        clusters = _correlation_clusters(["A", "B"], corr, threshold=0.5)
        assert len(clusters) == 1
        assert set(clusters[0]) == {"A", "B"}

    def test_two_uncorrelated_symbols(self):
        """Two uncorrelated symbols form separate clusters."""
        corr = np.array([[1.0, 0.1], [0.1, 1.0]])
        clusters = _correlation_clusters(["A", "B"], corr, threshold=0.5)
        assert len(clusters) == 2

    def test_three_symbols_partial_correlation(self):
        """A and B correlated, C independent -> 2 clusters."""
        corr = np.array([
            [1.0, 0.7, 0.1],
            [0.7, 1.0, 0.2],
            [0.1, 0.2, 1.0],
        ])
        clusters = _correlation_clusters(["A", "B", "C"], corr, threshold=0.5)
        assert len(clusters) == 2
        # A and B should be in the same cluster
        for c in clusters:
            if "A" in c:
                assert "B" in c
                assert "C" not in c

    def test_negative_correlation_clusters(self):
        """Negative correlation exceeding threshold also clusters."""
        corr = np.array([[1.0, -0.8], [-0.8, 1.0]])
        clusters = _correlation_clusters(["A", "B"], corr, threshold=0.5)
        assert len(clusters) == 1  # |rho| = 0.8 > 0.5


# ---------------------------------------------------------------------------
# ExposureTracker.max_correlated_cluster_exposure with returns
# ---------------------------------------------------------------------------


class TestExposureTrackerCorrelation:
    def _make_tracker_with_positions(self) -> ExposureTracker:
        """Create a tracker with 3 positions and inject correlated returns."""
        tracker = ExposureTracker(
            corr_lookback=60, corr_min_obs=10, corr_threshold=0.5
        )
        tracker.add_position(
            Position("ETH", "long", 10.0, 2000.0, 2000.0, 5.0)
        )
        tracker.add_position(
            Position("BTC", "long", 1.0, 40000.0, 40000.0, 3.0)
        )
        tracker.add_position(
            Position("SOL", "short", 100.0, 100.0, 100.0, 4.0)
        )
        return tracker

    def test_fallback_heuristic_when_no_return_data(self):
        """Without return data, falls back to directional heuristic."""
        tracker = self._make_tracker_with_positions()
        # 2 longs vs 1 short — should use directional fallback
        exceeded, pct = tracker.max_correlated_cluster_exposure(
            max_cluster_pct=0.6
        )
        # Long notional = 20000 + 40000 = 60000
        # Short notional = 10000
        # Gross = 70000, long_pct = 60000/70000 ≈ 0.857
        assert exceeded is True
        assert pct > 0.8

    def test_correlation_used_when_returns_available(self):
        """When returns are recorded, correlation matrix is used."""
        tracker = ExposureTracker(
            corr_lookback=60, corr_min_obs=5, corr_threshold=0.5
        )
        tracker.add_position(
            Position("A", "long", 10.0, 100.0, 100.0, 2.0)
        )
        tracker.add_position(
            Position("B", "long", 10.0, 100.0, 100.0, 2.0)
        )
        # Record highly correlated returns
        rng = np.random.default_rng(7)
        for _ in range(20):
            r = float(rng.normal(0, 0.01))
            tracker.record_return("A", r)
            tracker.record_return("B", r + float(rng.normal(0, 0.001)))

        # Both in one cluster = 100% of gross
        exceeded, pct = tracker.max_correlated_cluster_exposure(
            max_cluster_pct=0.6
        )
        assert exceeded is True
        assert pct > 0.9

    def test_uncorrelated_positions_pass(self):
        """Uncorrelated positions should not trigger cluster breach."""
        tracker = ExposureTracker(
            corr_lookback=60, corr_min_obs=5, corr_threshold=0.5
        )
        tracker.add_position(
            Position("A", "long", 10.0, 100.0, 100.0, 2.0)
        )
        tracker.add_position(
            Position("B", "long", 10.0, 100.0, 100.0, 2.0)
        )
        # Record independent returns
        rng = np.random.default_rng(42)
        for _ in range(30):
            tracker.record_return("A", float(rng.normal(0, 0.01)))
            tracker.record_return("B", float(rng.normal(0, 0.01)))

        # Each in its own cluster = 50% each, under 60% threshold
        exceeded, pct = tracker.max_correlated_cluster_exposure(
            max_cluster_pct=0.6
        )
        assert exceeded is False
        assert pct <= 0.6

    def test_update_price_records_returns(self):
        """update_price should automatically record returns."""
        tracker = ExposureTracker(
            corr_lookback=60, corr_min_obs=3, corr_threshold=0.5
        )
        tracker.add_position(
            Position("ETH", "long", 10.0, 2000.0, 2000.0, 5.0)
        )
        # Simulate price updates
        for px in [2010.0, 2020.0, 2015.0, 2025.0, 2030.0]:
            tracker.update_price("ETH", px)

        # Should have recorded returns
        assert len(tracker._return_tracker._returns.get("ETH", [])) == 5
