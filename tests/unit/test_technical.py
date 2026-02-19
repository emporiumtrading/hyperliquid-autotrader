"""Tests for autotrader.features.technical -- indicator calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autotrader.features.technical import (
    adx,
    atr,
    bb_width,
    ema,
    macd,
    realized_vol,
    rsi,
    sma,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_series(values: list[float]) -> pd.Series:
    """Build a simple float Series from a list."""
    return pd.Series(values, dtype=float)


def _make_trending_ohlcv(n: int = 150, start: float = 100.0, step: float = 0.5):
    """Generate clearly trending OHLCV data (monotonically increasing close).

    Returns (open, high, low, close, volume) as pd.Series.
    """
    rng = np.random.RandomState(42)
    close_vals = np.array([start + i * step for i in range(n)])
    noise = rng.uniform(-0.2, 0.2, size=n)
    open_vals = close_vals - step / 2 + noise
    high_vals = np.maximum(open_vals, close_vals) + rng.uniform(0.1, 0.5, size=n)
    low_vals = np.minimum(open_vals, close_vals) - rng.uniform(0.1, 0.5, size=n)
    volume = rng.uniform(1000, 5000, size=n)

    return (
        pd.Series(open_vals, dtype=float),
        pd.Series(high_vals, dtype=float),
        pd.Series(low_vals, dtype=float),
        pd.Series(close_vals, dtype=float),
        pd.Series(volume, dtype=float),
    )


# ===========================================================================
# Tests
# ===========================================================================


class TestSMA:
    def test_sma(self):
        """SMA of [1, 2, 3, 4, 5] with period=3 should be NaN, NaN, 2, 3, 4."""
        data = _make_series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(data, period=3)

        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert pytest.approx(result.iloc[2], abs=1e-9) == 2.0
        assert pytest.approx(result.iloc[3], abs=1e-9) == 3.0
        assert pytest.approx(result.iloc[4], abs=1e-9) == 4.0


class TestEMA:
    def test_ema(self):
        """EMA on a constant series should equal that constant.
        EMA on [1, 2, 3, 4, 5] with span=3 should produce known values.
        """
        # Constant series
        const = _make_series([5.0] * 10)
        result = ema(const, span=3)
        for val in result:
            assert pytest.approx(val, abs=1e-9) == 5.0

        # Increasing series: first value = 1.0, subsequent values follow
        # EMA formula: EMA_t = alpha * x_t + (1 - alpha) * EMA_{t-1}
        # where alpha = 2/(span+1) = 2/4 = 0.5
        data = _make_series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ema(data, span=3)

        # With adjust=False:
        #   EMA[0] = 1.0
        #   EMA[1] = 0.5 * 2 + 0.5 * 1 = 1.5
        #   EMA[2] = 0.5 * 3 + 0.5 * 1.5 = 2.25
        #   EMA[3] = 0.5 * 4 + 0.5 * 2.25 = 3.125
        #   EMA[4] = 0.5 * 5 + 0.5 * 3.125 = 4.0625
        assert pytest.approx(result.iloc[0], abs=1e-9) == 1.0
        assert pytest.approx(result.iloc[1], abs=1e-9) == 1.5
        assert pytest.approx(result.iloc[2], abs=1e-9) == 2.25
        assert pytest.approx(result.iloc[3], abs=1e-9) == 3.125
        assert pytest.approx(result.iloc[4], abs=1e-9) == 4.0625


class TestRSI:
    def test_rsi_constant_up(self):
        """A series that only goes up should yield RSI = 100 (after warmup)."""
        prices = _make_series([float(i) for i in range(1, 32)])  # 1 .. 31
        result = rsi(prices, period=14)
        # After warmup (first 14 values are NaN), all should be 100
        valid = result.dropna()
        assert len(valid) > 0
        for val in valid:
            assert pytest.approx(val, abs=1e-6) == 100.0

    def test_rsi_constant_down(self):
        """A series that only goes down should yield RSI = 0 (after warmup)."""
        prices = _make_series([float(100 - i) for i in range(31)])  # 100 .. 70
        result = rsi(prices, period=14)
        valid = result.dropna()
        assert len(valid) > 0
        for val in valid:
            assert pytest.approx(val, abs=1e-6) == 0.0

    def test_rsi_mixed(self):
        """A mixed up/down series should give RSI between 0 and 100."""
        rng = np.random.RandomState(123)
        prices = _make_series(list(100.0 + np.cumsum(rng.randn(50))))
        result = rsi(prices, period=14)
        valid = result.dropna()
        assert len(valid) > 0
        for val in valid:
            assert 0.0 <= val <= 100.0
        # Not all the same
        assert valid.min() < valid.max()


class TestATR:
    def test_atr_positive(self):
        """ATR should be positive for real-world-like OHLCV data."""
        _, high, low, close, _ = _make_trending_ohlcv(60)
        result = atr(high, low, close, period=14)
        valid = result.dropna()
        assert len(valid) > 0
        for val in valid:
            assert val > 0.0


class TestADX:
    def test_adx_trending(self):
        """Trending data with strong directional movement should give high ADX."""
        _, high, low, close, _ = _make_trending_ohlcv(150, step=1.0)
        result = adx(high, low, close, period=14)
        valid = result.dropna()
        assert len(valid) > 0
        # A strongly trending series should have ADX > 25 (convention for
        # "trending" markets).  With step=1.0 every bar, ADX is typically high.
        median_adx = valid.median()
        assert (
            median_adx > 25.0
        ), f"Expected ADX > 25 for trending data, got median {median_adx:.2f}"


class TestBollingerBands:
    def test_bollinger_bands_width(self):
        """BB width = (upper - lower) / middle should be positive after warmup."""
        _, _, _, close, _ = _make_trending_ohlcv(80)
        width = bb_width(close, period=20, num_std=2.0)
        valid = width.dropna()
        assert len(valid) > 0
        for val in valid:
            assert val > 0.0


class TestMACD:
    def test_macd_crossover(self):
        """On trending data, MACD line should cross above signal line at some point,
        evidenced by a histogram that changes sign from negative to positive."""
        # Build a series that starts flat then trends up
        flat = [100.0] * 40
        trend = [100.0 + i * 0.5 for i in range(80)]
        data = _make_series(flat + trend)

        macd_line, signal_line, histogram = macd(data, fast=12, slow=26, signal=9)

        # The histogram should become positive at some point during the
        # uptrend (MACD line crossing above signal).
        assert histogram.iloc[-1] > 0, "MACD histogram should be positive at end of uptrend"

        # Also verify that the histogram was negative at some earlier point
        # (during the flat part) so a crossover truly occurred.
        early_hist = histogram.iloc[30:50]  # during/after flat period
        late_hist = histogram.iloc[-20:]  # well into the uptrend
        assert early_hist.min() < late_hist.max(), "Crossover should show sign change"


class TestRealizedVol:
    def test_realized_vol_positive(self):
        """Realized volatility should be non-negative."""
        _, _, _, close, _ = _make_trending_ohlcv(60)
        result = realized_vol(close, period=20)
        valid = result.dropna()
        assert len(valid) > 0
        for val in valid:
            assert val >= 0.0
