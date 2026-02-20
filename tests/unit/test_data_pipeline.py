"""Tests for data pipeline: cleaning, resampling, OHLCV validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autotrader.data.transforms.cleaning import detect_gaps, remove_outliers, validate_ohlcv
from autotrader.data.transforms.resample import resample_ohlcv, _interval_to_pandas_freq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candles(
    n: int = 50,
    interval_ms: int = 60_000,
    start_ms: int = 1_000_000_000,
    base_px: float = 100.0,
) -> pd.DataFrame:
    """Generate a valid OHLCV DataFrame with *n* bars."""
    rng = np.random.default_rng(42)
    timestamps = [start_ms + i * interval_ms for i in range(n)]
    closes = base_px + np.cumsum(rng.normal(0, 0.5, n))
    opens = np.roll(closes, 1)
    opens[0] = base_px
    highs = np.maximum(opens, closes) + rng.uniform(0.1, 1.0, n)
    lows = np.minimum(opens, closes) - rng.uniform(0.1, 1.0, n)
    volumes = rng.uniform(100, 1000, n)

    return pd.DataFrame(
        {
            "timestamp_ms": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


# ---------------------------------------------------------------------------
# detect_gaps
# ---------------------------------------------------------------------------


class TestDetectGaps:
    def test_no_gaps(self):
        """Contiguous data has no gaps."""
        df = _make_candles(n=20, interval_ms=60_000)
        gaps = detect_gaps(df, interval="1m")
        assert gaps.empty

    def test_single_gap(self):
        """Multiple missing bars create a gap above 2x threshold."""
        df = _make_candles(n=20, interval_ms=60_000)
        # Remove 3 bars to create a gap of 4 intervals (>2x threshold)
        df = df.drop(index=[10, 11, 12]).reset_index(drop=True)
        gaps = detect_gaps(df, interval="1m")
        assert len(gaps) == 1
        assert gaps.iloc[0]["gap_bars"] == 4

    def test_empty_dataframe(self):
        """Empty DataFrame produces no gaps."""
        df = pd.DataFrame(columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
        gaps = detect_gaps(df, interval="1m")
        assert gaps.empty

    def test_single_row(self):
        """Single row produces no gaps."""
        df = _make_candles(n=1)
        gaps = detect_gaps(df, interval="1m")
        assert gaps.empty

    def test_custom_max_gap_multiple(self):
        """Custom max_gap_multiple controls sensitivity."""
        df = _make_candles(n=10, interval_ms=60_000)
        # Remove one bar -- gap is 2x interval
        df = df.drop(index=5).reset_index(drop=True)
        # With multiple=3.0, this gap should NOT be flagged
        gaps = detect_gaps(df, interval="1m", max_gap_multiple=3.0)
        assert gaps.empty
        # With multiple=1.5, it SHOULD be flagged
        gaps = detect_gaps(df, interval="1m", max_gap_multiple=1.5)
        assert len(gaps) == 1


# ---------------------------------------------------------------------------
# remove_outliers
# ---------------------------------------------------------------------------


class TestRemoveOutliers:
    def test_no_outliers(self):
        """Normal data is preserved."""
        df = _make_candles(n=30)
        result = remove_outliers(df, column="close", z_threshold=5.0)
        assert len(result) == len(df)

    def test_extreme_outlier_removed(self):
        """An extreme value is removed."""
        df = _make_candles(n=30)
        df.loc[15, "close"] = 99999.0  # extreme outlier
        result = remove_outliers(df, column="close", z_threshold=3.0)
        assert len(result) < len(df)

    def test_empty_dataframe(self):
        """Empty DataFrame returns empty."""
        df = pd.DataFrame(columns=["timestamp_ms", "close"])
        result = remove_outliers(df)
        assert result.empty

    def test_missing_column(self):
        """Missing column returns original DataFrame."""
        df = _make_candles(n=10)
        result = remove_outliers(df, column="nonexistent")
        assert len(result) == len(df)

    def test_zero_std(self):
        """Constant values (std=0) returns all rows."""
        df = pd.DataFrame({"timestamp_ms": range(10), "close": [50.0] * 10})
        result = remove_outliers(df, column="close")
        assert len(result) == 10


# ---------------------------------------------------------------------------
# validate_ohlcv
# ---------------------------------------------------------------------------


class TestValidateOHLCV:
    def test_valid_data(self):
        """Valid OHLCV data produces no errors."""
        df = _make_candles(n=20)
        errors = validate_ohlcv(df)
        assert errors == []

    def test_high_lt_open(self):
        """Detects high < open."""
        df = _make_candles(n=5)
        df.loc[2, "high"] = df.loc[2, "open"] - 1.0
        errors = validate_ohlcv(df)
        assert len(errors) >= 1
        assert "high" in errors[0].lower()

    def test_high_lt_close(self):
        """Detects high < close."""
        df = _make_candles(n=5)
        df.loc[2, "high"] = df.loc[2, "close"] - 1.0
        errors = validate_ohlcv(df)
        assert len(errors) >= 1

    def test_low_gt_open(self):
        """Detects low > open."""
        df = _make_candles(n=5)
        df.loc[2, "low"] = df.loc[2, "open"] + 1.0
        errors = validate_ohlcv(df)
        assert len(errors) >= 1
        assert "low" in errors[0].lower()

    def test_low_gt_close(self):
        """Detects low > close."""
        df = _make_candles(n=5)
        df.loc[2, "low"] = df.loc[2, "close"] + 1.0
        errors = validate_ohlcv(df)
        assert len(errors) >= 1

    def test_negative_volume(self):
        """Detects negative volume."""
        df = _make_candles(n=5)
        df.loc[3, "volume"] = -100.0
        errors = validate_ohlcv(df)
        assert len(errors) >= 1
        assert "volume" in errors[0].lower()

    def test_empty_dataframe(self):
        """Empty DataFrame produces no errors."""
        df = pd.DataFrame(columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
        errors = validate_ohlcv(df)
        assert errors == []

    def test_multiple_errors(self):
        """Multiple violations produce multiple errors."""
        df = _make_candles(n=5)
        df.loc[1, "high"] = df.loc[1, "open"] - 1.0  # high < open
        df.loc[3, "volume"] = -10.0  # negative volume
        errors = validate_ohlcv(df)
        assert len(errors) >= 2


# ---------------------------------------------------------------------------
# resample_ohlcv
# ---------------------------------------------------------------------------


class TestResampleOHLCV:
    def test_1m_to_5m(self):
        """Resample 1m bars to 5m bars."""
        df = _make_candles(n=100, interval_ms=60_000)
        result = resample_ohlcv(df, source_interval="1m", target_interval="5m")
        # Allow for partial last bar producing one extra
        assert len(result) >= 20
        assert result["volume"].iloc[0] > 0

    def test_1m_to_15m(self):
        """Resample 1m bars to 15m bars."""
        df = _make_candles(n=150, interval_ms=60_000)
        result = resample_ohlcv(df, source_interval="1m", target_interval="15m")
        assert len(result) >= 10

    def test_ohlcv_aggregation_rules(self):
        """Verify correct aggregation: first/max/min/last/sum."""
        df = pd.DataFrame(
            {
                "timestamp_ms": [0, 60_000, 120_000, 180_000, 240_000],
                "open": [10.0, 11.0, 12.0, 13.0, 14.0],
                "high": [15.0, 16.0, 17.0, 18.0, 19.0],
                "low": [5.0, 6.0, 7.0, 8.0, 9.0],
                "close": [11.0, 12.0, 13.0, 14.0, 15.0],
                "volume": [100.0, 200.0, 300.0, 400.0, 500.0],
            }
        )
        result = resample_ohlcv(df, source_interval="1m", target_interval="5m")
        assert len(result) == 1
        row = result.iloc[0]
        assert row["open"] == 10.0  # first
        assert row["high"] == 19.0  # max
        assert row["low"] == 5.0  # min
        assert row["close"] == 15.0  # last
        assert row["volume"] == 1500.0  # sum

    def test_target_lt_source_raises(self):
        """Target smaller than source raises ValueError."""
        df = _make_candles(n=10, interval_ms=300_000)  # 5m bars
        with pytest.raises(ValueError, match="must be >="):
            resample_ohlcv(df, source_interval="5m", target_interval="1m")

    def test_target_equals_source_no_change(self):
        """Resampling to the same interval returns same bar count."""
        df = _make_candles(n=10, interval_ms=60_000)
        result = resample_ohlcv(df, source_interval="1m", target_interval="1m")
        assert len(result) == len(df)

    def test_empty_dataframe(self):
        """Empty DataFrame returns empty."""
        df = pd.DataFrame(columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
        result = resample_ohlcv(df, source_interval="1m", target_interval="5m")
        assert result.empty

    def test_timestamp_ms_column_preserved(self):
        """Output has int64 timestamp_ms column."""
        df = _make_candles(n=10, interval_ms=60_000)
        result = resample_ohlcv(df, source_interval="1m", target_interval="5m")
        assert "timestamp_ms" in result.columns
        assert result["timestamp_ms"].dtype in (np.int64, "int64")


# ---------------------------------------------------------------------------
# _interval_to_pandas_freq
# ---------------------------------------------------------------------------


class TestIntervalToPandasFreq:
    def test_minutes(self):
        assert _interval_to_pandas_freq("1m") == "1min"
        assert _interval_to_pandas_freq("15m") == "15min"

    def test_hours(self):
        assert _interval_to_pandas_freq("1h") == "1h"
        assert _interval_to_pandas_freq("4h") == "4h"

    def test_days(self):
        assert _interval_to_pandas_freq("1d") == "1D"

    def test_invalid_short(self):
        with pytest.raises(ValueError):
            _interval_to_pandas_freq("m")

    def test_unsupported_unit(self):
        with pytest.raises(ValueError):
            _interval_to_pandas_freq("1x")
