"""Tests for autotrader.features.microstructure -- order book and trade features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from autotrader.features.microstructure import (
    compute_book_imbalance,
    compute_depth_usd,
    compute_mid_price,
    compute_spread_bps,
    compute_trade_intensity,
    compute_volume_profile,
    compute_vwap,
)


# ---------------------------------------------------------------------------
# Existing functions (regression)
# ---------------------------------------------------------------------------


class TestSpread:
    def test_spread_bps_basic(self):
        spread = compute_spread_bps(99.0, 101.0)
        # (101-99)/100 * 10000 = 200 bps
        assert abs(spread - 200.0) < 0.01

    def test_spread_zero_mid(self):
        assert compute_spread_bps(0.0, 0.0) == 0.0


class TestMidPrice:
    def test_mid(self):
        assert compute_mid_price(100.0, 102.0) == 101.0


class TestDepth:
    def test_depth_usd(self):
        levels = [(100.0, 5.0), (99.0, 10.0), (98.0, 15.0)]
        depth = compute_depth_usd(levels, n_levels=2)
        # 100*5 + 99*10 = 500 + 990 = 1490
        assert abs(depth - 1490.0) < 0.01


class TestBookImbalance:
    def test_balanced(self):
        assert compute_book_imbalance(100.0, 100.0) == 0.0

    def test_bid_heavy(self):
        imb = compute_book_imbalance(200.0, 100.0)
        assert abs(imb - 1 / 3) < 0.01

    def test_empty_book(self):
        assert compute_book_imbalance(0.0, 0.0) == 0.0


class TestVWAP:
    def test_vwap_constant_price(self):
        prices = pd.Series([100.0] * 30)
        volumes = pd.Series([10.0] * 30)
        vwap = compute_vwap(prices, volumes, period=20)
        # VWAP of constant prices should equal the price
        assert abs(vwap.iloc[-1] - 100.0) < 0.01


class TestVolumeProfile:
    def test_above_average(self):
        volume = pd.Series([10.0] * 25 + [30.0])
        close = pd.Series([100.0] * 26)
        rel = compute_volume_profile(close, volume, period=20)
        # Last bar has 3x average volume
        assert rel.iloc[-1] > 2.5


# ---------------------------------------------------------------------------
# Trade Intensity (new W5 feature)
# ---------------------------------------------------------------------------


class TestTradeIntensity:
    def _make_volume(self, n: int = 50, base: float = 100.0) -> pd.Series:
        rng = np.random.default_rng(42)
        return pd.Series(rng.normal(base, base * 0.1, n).clip(1))

    def test_returns_dataframe_with_correct_columns(self):
        vol = self._make_volume()
        result = compute_trade_intensity(vol, period=20)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"intensity", "acceleration", "is_spike"}
        assert len(result) == len(vol)

    def test_intensity_near_one_for_stable_volume(self):
        """Constant volume should produce intensity near 1.0."""
        vol = pd.Series([100.0] * 40)
        result = compute_trade_intensity(vol, period=20)
        # After warmup, intensity should be exactly 1.0
        assert abs(result["intensity"].iloc[-1] - 1.0) < 0.01

    def test_spike_detection(self):
        """A volume spike should be flagged."""
        volumes = [100.0] * 30 + [300.0]
        vol = pd.Series(volumes)
        result = compute_trade_intensity(vol, period=20, spike_threshold=2.0)
        # Last bar is 3x the SMA -> should be flagged
        assert bool(result["is_spike"].iloc[-1]) is True
        assert result["intensity"].iloc[-1] > 2.0

    def test_no_spike_for_normal_volume(self):
        """Normal volume should not be flagged."""
        vol = pd.Series([100.0] * 30)
        result = compute_trade_intensity(vol, period=20, spike_threshold=2.0)
        assert bool(result["is_spike"].iloc[-1]) is False

    def test_acceleration_positive_on_increasing_volume(self):
        """Acceleration should be positive when volume jumps up."""
        # Stable baseline then sudden increase
        volumes = [100.0] * 25 + [200.0, 250.0, 300.0]
        vol = pd.Series([float(v) for v in volumes])
        result = compute_trade_intensity(vol, period=20)
        # The bars with increasing volume should show positive acceleration
        recent_acc = result["acceleration"].dropna().iloc[-3:]
        assert (recent_acc > 0).all()

    def test_acceleration_negative_on_decreasing_volume(self):
        """Acceleration should be negative when volume decreases."""
        volumes = list(range(100, 50, -1))
        vol = pd.Series([float(v) for v in volumes])
        result = compute_trade_intensity(vol, period=20)
        recent_acc = result["acceleration"].dropna().iloc[-5:]
        assert (recent_acc < 0).all()

    def test_nan_during_warmup(self):
        """First `period` bars should be NaN (insufficient data)."""
        vol = pd.Series([100.0] * 30)
        result = compute_trade_intensity(vol, period=20)
        assert result["intensity"].iloc[:19].isna().all()
        assert result["intensity"].iloc[19:].notna().all()
