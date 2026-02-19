"""Tests for individual strategy signal generation with synthetic MarketContext data.

Each strategy is tested for:
1. Returning flat for non-applicable regimes.
2. Producing a LONG signal with appropriate features.
3. Producing a SHORT signal with appropriate features.
4. Invariants passing on all produced signals.
5. Returning flat when features are missing or insufficient.
"""

import numpy as np
import pandas as pd
import pytest

from autotrader.hl.types import Signal
from autotrader.strategies.base import BaseStrategy, MarketContext, flat_signal
from autotrader.strategies.trend_breakout import TrendBreakoutStrategy
from autotrader.strategies.range_meanrev import RangeMeanRevStrategy
from autotrader.strategies.vol_expansion import VolExpansionStrategy
from autotrader.strategies.funding_extremes import FundingExtremesStrategy
from autotrader.strategies.squeeze_breakout import SqueezeBreakoutStrategy
from autotrader.strategies.crash_meanrev import CrashMeanRevStrategy

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _make_candles(
    n: int = 100,
    base_price: float = 100.0,
    seed: int = 42,
    trend: float = 0.05,
) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame.

    Parameters
    ----------
    n : int
        Number of candles.
    base_price : float
        Starting price level.
    seed : int
        Random seed for reproducibility.
    trend : float
        Per-bar drift applied to close prices.
    """
    rng = np.random.default_rng(seed)
    timestamps = list(range(1_000_000, 1_000_000 + n * 60_000, 60_000))
    closes = base_price + np.cumsum(rng.normal(trend, 0.5, size=n))
    highs = closes + rng.uniform(0.1, 1.0, size=n)
    lows = closes - rng.uniform(0.1, 1.0, size=n)
    opens = closes + rng.normal(0.0, 0.3, size=n)
    volumes = rng.uniform(100, 1000, size=n)

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


def _make_candles_with_last(
    last_close: float,
    last_high: float | None = None,
    last_low: float | None = None,
    last_open: float | None = None,
    last_volume: float = 500.0,
    n: int = 100,
    base_price: float = 100.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate candles but override the last candle's values for precise control."""
    df = _make_candles(n=n, base_price=base_price, seed=seed)
    if last_high is None:
        last_high = last_close + 0.5
    if last_low is None:
        last_low = last_close - 0.5
    if last_open is None:
        last_open = last_close - 0.1

    df.loc[df.index[-1], "close"] = last_close
    df.loc[df.index[-1], "high"] = last_high
    df.loc[df.index[-1], "low"] = last_low
    df.loc[df.index[-1], "open"] = last_open
    df.loc[df.index[-1], "volume"] = last_volume
    return df


def _build_market_context(
    regime: str = "TREND",
    features: dict | None = None,
    candles: pd.DataFrame | None = None,
    funding_rate: float | None = None,
    open_interest: float | None = None,
) -> MarketContext:
    """Build a MarketContext with sensible defaults."""
    if candles is None:
        candles = _make_candles()
    if features is None:
        features = {}
    return MarketContext(
        symbol="ETH",
        timeframe="15m",
        candles=candles,
        features=features,
        regime=regime,
        regime_confidence=0.8,
        funding_rate=funding_rate,
        open_interest=open_interest,
        account_equity=10_000.0,
    )


def _assert_flat(signal: Signal) -> None:
    """Assert a signal is flat."""
    assert signal.side == "flat"
    assert signal.confidence == 0.0


def _assert_valid_long(signal: Signal, strategy: BaseStrategy) -> None:
    """Assert a signal is a valid long with passing invariants."""
    assert signal.side == "long"
    assert signal.entry is not None
    assert signal.stop is not None
    assert signal.take_profit is not None
    assert signal.stop < signal.entry, "Long stop must be below entry"
    assert signal.take_profit > signal.entry, "Long TP must be above entry"
    assert 0.0 < signal.confidence <= 1.0
    assert strategy.invariants_ok(signal) is True


def _assert_valid_short(signal: Signal, strategy: BaseStrategy) -> None:
    """Assert a signal is a valid short with passing invariants."""
    assert signal.side == "short"
    assert signal.entry is not None
    assert signal.stop is not None
    assert signal.take_profit is not None
    assert signal.stop > signal.entry, "Short stop must be above entry"
    assert signal.take_profit < signal.entry, "Short TP must be below entry"
    assert 0.0 < signal.confidence <= 1.0
    assert strategy.invariants_ok(signal) is True


# =======================================================================
# TrendBreakoutStrategy
# =======================================================================


class TestTrendBreakoutStrategy:
    """Tests for the TrendBreakoutStrategy."""

    def setup_method(self):
        self.strategy = TrendBreakoutStrategy()

    # ---- 1. Non-applicable regimes return flat ----

    @pytest.mark.parametrize(
        "regime",
        ["RANGE", "VOLATILE_BREAKOUT", "SQUEEZE_RISK", "MEAN_REVERT_CRASH", "UNKNOWN"],
    )
    def test_flat_for_non_applicable_regime_features(self, regime: str):
        """Even with valid features, the strategy should conceptually only
        apply to TREND.  Here we test that the strategy itself still produces
        a signal (it does not check regime internally), but the ensemble would
        filter it.  We verify applicable_regimes is correct."""
        assert regime not in self.strategy.applicable_regimes()
        assert self.strategy.applicable_regimes() == ["TREND"]

    # ---- 2. LONG signal with appropriate features ----

    def test_long_signal_pullback(self):
        """In an uptrend (close > SMA), RSI pulled back below 40, ADX strong."""
        last_close = 110.0
        candles = _make_candles_with_last(
            last_close=last_close,
            last_high=111.0,
            last_low=109.0,
            n=100,
            base_price=100.0,
        )
        features = {
            "adx": 35.0,          # > 25 (min_adx)
            "rsi": 32.0,          # < 40 (pullback_rsi_long)
            "atr": 2.0,           # positive ATR
            "sma_50": 105.0,      # close (110) > sma_50 -> uptrend
        }
        ctx = _build_market_context(regime="TREND", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_valid_long(signal, self.strategy)
        assert signal.metadata.get("strategy") == "trend_breakout"

    def test_long_signal_breakout(self):
        """Close above lookback high triggers a breakout long."""
        # Build candles where the lookback high is below the last candle high
        candles = _make_candles(n=100, base_price=100.0, seed=42)
        lookback_high = float(candles["high"].iloc[-20:].max())
        # Set last candle high above lookback high to trigger breakout
        candles.loc[candles.index[-1], "high"] = lookback_high + 1.0
        candles.loc[candles.index[-1], "close"] = lookback_high + 0.5
        last_close = float(candles["close"].iloc[-1])

        features = {
            "adx": 30.0,          # > 25
            "rsi": 55.0,          # not pulled back, but breakout should trigger
            "atr": 2.0,
            "sma_50": last_close - 5.0,  # close > sma -> uptrend
        }
        ctx = _build_market_context(regime="TREND", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_valid_long(signal, self.strategy)

    # ---- 3. SHORT signal with appropriate features ----

    def test_short_signal_pullback(self):
        """In a downtrend (close < SMA), RSI above 60, ADX strong."""
        last_close = 95.0
        candles = _make_candles_with_last(
            last_close=last_close,
            last_high=96.0,
            last_low=94.0,
            n=100,
            base_price=100.0,
        )
        features = {
            "adx": 35.0,          # > 25
            "rsi": 68.0,          # > 60 (pullback_rsi_short)
            "atr": 2.0,
            "sma_50": 105.0,      # close (95) < sma_50 -> downtrend
        }
        ctx = _build_market_context(regime="TREND", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_valid_short(signal, self.strategy)
        assert signal.metadata.get("strategy") == "trend_breakout"

    def test_short_signal_breakout(self):
        """Close below lookback low triggers a breakout short."""
        candles = _make_candles(n=100, base_price=100.0, seed=42)
        lookback_low = float(candles["low"].iloc[-20:].min())
        # Set last candle low below lookback low for breakout
        candles.loc[candles.index[-1], "low"] = lookback_low - 1.0
        candles.loc[candles.index[-1], "close"] = lookback_low - 0.5
        last_close = float(candles["close"].iloc[-1])

        features = {
            "adx": 30.0,
            "rsi": 50.0,          # not pulled back, but breakout triggers
            "atr": 2.0,
            "sma_50": last_close + 5.0,  # close < sma -> downtrend
        }
        ctx = _build_market_context(regime="TREND", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_valid_short(signal, self.strategy)

    # ---- 4. Invariants on produced signals ----

    def test_invariants_on_all_signals(self):
        """Run the strategy across multiple feature sets and verify invariants."""
        configs = [
            # Long pullback
            {"adx": 40.0, "rsi": 30.0, "atr": 3.0, "sma_50": 90.0},
            # Short pullback
            {"adx": 45.0, "rsi": 70.0, "atr": 3.0, "sma_50": 120.0},
            # Weak ADX -> flat
            {"adx": 15.0, "rsi": 30.0, "atr": 3.0, "sma_50": 90.0},
        ]
        for features in configs:
            candles = _make_candles_with_last(last_close=100.0, n=100, base_price=100.0)
            ctx = _build_market_context(regime="TREND", features=features, candles=candles)
            signal = self.strategy.compute_signal(ctx)
            assert self.strategy.invariants_ok(signal) is True

    # ---- 5. Flat when features missing / insufficient ----

    def test_flat_when_no_features(self):
        """Returns flat when features dict is empty."""
        ctx = _build_market_context(regime="TREND", features={})
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_adx_missing(self):
        """Returns flat when ADX is missing."""
        features = {"rsi": 30.0, "atr": 2.0, "sma_50": 90.0}
        ctx = _build_market_context(regime="TREND", features=features)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_rsi_missing(self):
        """Returns flat when RSI is missing."""
        features = {"adx": 35.0, "atr": 2.0, "sma_50": 90.0}
        ctx = _build_market_context(regime="TREND", features=features)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_atr_missing(self):
        """Returns flat when ATR is missing."""
        features = {"adx": 35.0, "rsi": 30.0, "sma_50": 90.0}
        ctx = _build_market_context(regime="TREND", features=features)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_sma_missing(self):
        """Returns flat when both sma_50 and sma are missing."""
        features = {"adx": 35.0, "rsi": 30.0, "atr": 2.0}
        ctx = _build_market_context(regime="TREND", features=features)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_atr_zero(self):
        """Returns flat when ATR is zero."""
        features = {"adx": 35.0, "rsi": 30.0, "atr": 0.0, "sma_50": 90.0}
        ctx = _build_market_context(regime="TREND", features=features)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_adx_below_threshold(self):
        """Returns flat when ADX is below min_adx (25)."""
        last_close = 110.0
        candles = _make_candles_with_last(last_close=last_close, n=100, base_price=100.0)
        features = {"adx": 20.0, "rsi": 30.0, "atr": 2.0, "sma_50": 105.0}
        ctx = _build_market_context(regime="TREND", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_candles_empty(self):
        """Returns flat when candles DataFrame is empty."""
        candles = pd.DataFrame(columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
        features = {"adx": 35.0, "rsi": 30.0, "atr": 2.0, "sma_50": 90.0}
        ctx = _build_market_context(regime="TREND", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_candles_too_few(self):
        """Returns flat when fewer candles than lookback period."""
        candles = _make_candles(n=5, base_price=100.0)  # default lookback is 20
        features = {"adx": 35.0, "rsi": 30.0, "atr": 2.0, "sma_50": 90.0}
        ctx = _build_market_context(regime="TREND", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_uses_generic_sma_fallback(self):
        """Falls back to 'sma' key when 'sma_50' is absent."""
        last_close = 110.0
        candles = _make_candles_with_last(last_close=last_close, n=100, base_price=100.0)
        features = {
            "adx": 35.0,
            "rsi": 32.0,
            "atr": 2.0,
            "sma": 105.0,  # generic fallback key
        }
        ctx = _build_market_context(regime="TREND", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_valid_long(signal, self.strategy)

    def test_nan_feature_treated_as_missing(self):
        """NaN values in features should be treated as missing -> flat."""
        features = {"adx": float("nan"), "rsi": 30.0, "atr": 2.0, "sma_50": 90.0}
        ctx = _build_market_context(regime="TREND", features=features)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)


# =======================================================================
# RangeMeanRevStrategy
# =======================================================================


class TestRangeMeanRevStrategy:
    """Tests for the RangeMeanRevStrategy."""

    def setup_method(self):
        self.strategy = RangeMeanRevStrategy()

    # ---- 1. Non-applicable regimes ----

    @pytest.mark.parametrize(
        "regime",
        ["TREND", "VOLATILE_BREAKOUT", "SQUEEZE_RISK", "MEAN_REVERT_CRASH", "UNKNOWN"],
    )
    def test_applicable_regimes(self, regime: str):
        """Strategy only applies to RANGE."""
        assert regime not in self.strategy.applicable_regimes()
        assert self.strategy.applicable_regimes() == ["RANGE"]

    # ---- 2. LONG signal ----

    def test_long_signal_near_lower_bb(self):
        """Close near lower BB with RSI oversold triggers a long."""
        last_close = 95.0
        candles = _make_candles_with_last(last_close=last_close, n=100, base_price=100.0)
        bb_lower = 95.1   # close is below lower BB
        bb_upper = 105.0
        bb_middle = 100.0
        # dist_to_lower = (95.0 - 95.1) / (105.0 - 95.1) = -0.01 which is <= 0.15
        features = {
            "adx": 15.0,           # < 25 (max_adx)
            "rsi": 22.0,           # < 30 (rsi_oversold)
            "atr": 2.0,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_middle": bb_middle,
        }
        ctx = _build_market_context(regime="RANGE", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_valid_long(signal, self.strategy)
        assert signal.metadata.get("strategy") == "range_meanrev"

    def test_long_signal_at_lower_bb_exact(self):
        """Close exactly at lower BB with RSI oversold triggers long."""
        bb_lower = 95.0
        bb_upper = 105.0
        bb_middle = 100.0
        last_close = bb_lower  # dist_to_lower = 0.0 -> <= 0.15
        candles = _make_candles_with_last(last_close=last_close, n=100, base_price=100.0)
        features = {
            "adx": 10.0,
            "rsi": 20.0,
            "atr": 2.0,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_middle": bb_middle,
        }
        ctx = _build_market_context(regime="RANGE", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_valid_long(signal, self.strategy)

    # ---- 3. SHORT signal ----

    def test_short_signal_near_upper_bb(self):
        """Close near upper BB with RSI overbought triggers a short."""
        last_close = 104.9
        candles = _make_candles_with_last(last_close=last_close, n=100, base_price=100.0)
        bb_lower = 95.0
        bb_upper = 105.0
        bb_middle = 100.0
        # dist_to_upper = (105.0 - 104.9) / (105.0 - 95.0) = 0.01 -> <= 0.15
        features = {
            "adx": 15.0,
            "rsi": 78.0,           # > 70 (rsi_overbought)
            "atr": 2.0,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_middle": bb_middle,
        }
        ctx = _build_market_context(regime="RANGE", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_valid_short(signal, self.strategy)
        assert signal.metadata.get("strategy") == "range_meanrev"

    # ---- 4. Invariants pass on all signals ----

    def test_invariants_on_long_and_short(self):
        """Verify invariants pass for both long and short signals."""
        # Long scenario
        candles_long = _make_candles_with_last(last_close=95.0, n=100, base_price=100.0)
        feat_long = {
            "adx": 12.0, "rsi": 18.0, "atr": 2.0,
            "bb_upper": 106.0, "bb_lower": 95.5, "bb_middle": 100.0,
        }
        ctx_long = _build_market_context(regime="RANGE", features=feat_long, candles=candles_long)
        sig_long = self.strategy.compute_signal(ctx_long)
        assert self.strategy.invariants_ok(sig_long) is True

        # Short scenario
        candles_short = _make_candles_with_last(last_close=105.5, n=100, base_price=100.0)
        feat_short = {
            "adx": 12.0, "rsi": 82.0, "atr": 2.0,
            "bb_upper": 106.0, "bb_lower": 95.0, "bb_middle": 100.0,
        }
        ctx_short = _build_market_context(regime="RANGE", features=feat_short, candles=candles_short)
        sig_short = self.strategy.compute_signal(ctx_short)
        assert self.strategy.invariants_ok(sig_short) is True

    # ---- 5. Flat when features missing / insufficient ----

    def test_flat_when_no_features(self):
        """Returns flat when features dict is empty."""
        ctx = _build_market_context(regime="RANGE", features={})
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_bb_lower_missing(self):
        """Returns flat when bb_lower is missing."""
        features = {
            "adx": 15.0, "rsi": 22.0, "atr": 2.0,
            "bb_upper": 105.0, "bb_middle": 100.0,
            # bb_lower missing
        }
        ctx = _build_market_context(regime="RANGE", features=features)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_bb_middle_missing(self):
        """Returns flat when bb_middle is missing."""
        features = {
            "adx": 15.0, "rsi": 22.0, "atr": 2.0,
            "bb_upper": 105.0, "bb_lower": 95.0,
            # bb_middle missing
        }
        ctx = _build_market_context(regime="RANGE", features=features)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_adx_too_high(self):
        """Returns flat when ADX > max_adx (25) -- trending, not ranging."""
        last_close = 95.0
        candles = _make_candles_with_last(last_close=last_close, n=100, base_price=100.0)
        features = {
            "adx": 30.0,   # > 25 (max_adx)
            "rsi": 22.0,
            "atr": 2.0,
            "bb_upper": 105.0, "bb_lower": 95.1, "bb_middle": 100.0,
        }
        ctx = _build_market_context(regime="RANGE", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_rsi_neutral(self):
        """Returns flat when RSI is in the neutral zone (not oversold/overbought)."""
        last_close = 95.0
        candles = _make_candles_with_last(last_close=last_close, n=100, base_price=100.0)
        features = {
            "adx": 15.0,
            "rsi": 50.0,   # neutral
            "atr": 2.0,
            "bb_upper": 105.0, "bb_lower": 95.1, "bb_middle": 100.0,
        }
        ctx = _build_market_context(regime="RANGE", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_price_in_middle(self):
        """Returns flat when close is far from both bands (in the middle)."""
        last_close = 100.0
        candles = _make_candles_with_last(last_close=last_close, n=100, base_price=100.0)
        features = {
            "adx": 15.0,
            "rsi": 22.0,
            "atr": 2.0,
            "bb_upper": 105.0, "bb_lower": 95.0, "bb_middle": 100.0,
        }
        # dist_to_lower = (100 - 95) / 10 = 0.5 -> > 0.15
        ctx = _build_market_context(regime="RANGE", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_atr_zero(self):
        """Returns flat when ATR is zero."""
        features = {
            "adx": 15.0, "rsi": 22.0, "atr": 0.0,
            "bb_upper": 105.0, "bb_lower": 95.0, "bb_middle": 100.0,
        }
        ctx = _build_market_context(regime="RANGE", features=features)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_candles_empty(self):
        """Returns flat when candles is empty."""
        candles = pd.DataFrame(columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
        features = {
            "adx": 15.0, "rsi": 22.0, "atr": 2.0,
            "bb_upper": 105.0, "bb_lower": 95.0, "bb_middle": 100.0,
        }
        ctx = _build_market_context(regime="RANGE", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_bb_width_zero(self):
        """Returns flat when BB upper == BB lower (zero width)."""
        last_close = 100.0
        candles = _make_candles_with_last(last_close=last_close, n=100, base_price=100.0)
        features = {
            "adx": 15.0, "rsi": 22.0, "atr": 2.0,
            "bb_upper": 100.0, "bb_lower": 100.0, "bb_middle": 100.0,
        }
        ctx = _build_market_context(regime="RANGE", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)


# =======================================================================
# VolExpansionStrategy
# =======================================================================


class TestVolExpansionStrategy:
    """Tests for the VolExpansionStrategy."""

    def setup_method(self):
        self.strategy = VolExpansionStrategy()

    # ---- 1. Non-applicable regimes ----

    @pytest.mark.parametrize(
        "regime",
        ["TREND", "RANGE", "SQUEEZE_RISK", "MEAN_REVERT_CRASH", "UNKNOWN"],
    )
    def test_applicable_regimes(self, regime: str):
        """Strategy only applies to VOLATILE_BREAKOUT."""
        assert regime not in self.strategy.applicable_regimes()
        assert self.strategy.applicable_regimes() == ["VOLATILE_BREAKOUT"]

    # ---- 2. LONG signal ----

    def test_long_signal_expansion_above_mid(self):
        """Price above middle BB with positive slope + prior squeeze -> long."""
        last_close = 105.0
        candles = _make_candles_with_last(
            last_close=last_close, last_volume=800.0, n=100, base_price=100.0,
        )
        features = {
            "bb_width_pct": 0.85,           # > 0.7 (expansion) and > 0.85 (heuristic squeeze)
            "atr": 2.0,
            "bb_middle": 100.0,             # close (105) > bb_middle -> long
            "ma_slope": 0.5,                # positive slope
            "volume_sma": 400.0,            # 800 > 1.5 * 400 = 600 -> confirmed
            "prev_bb_width_pct": 0.15,      # < 0.2 -> prior squeeze
        }
        ctx = _build_market_context(regime="VOLATILE_BREAKOUT", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_valid_long(signal, self.strategy)
        assert signal.metadata.get("strategy") == "vol_expansion"

    def test_long_with_history_squeeze(self):
        """Prior squeeze detected via bb_width_pct_history list."""
        last_close = 105.0
        candles = _make_candles_with_last(
            last_close=last_close, last_volume=800.0, n=100, base_price=100.0,
        )
        features = {
            "bb_width_pct": 0.8,
            "atr": 2.0,
            "bb_middle": 100.0,
            "ma_slope": 0.5,
            "volume_sma": 400.0,
            "bb_width_pct_history": [0.5, 0.3, 0.15, 0.3, 0.5],  # 0.15 < 0.2 -> squeeze
        }
        ctx = _build_market_context(regime="VOLATILE_BREAKOUT", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_valid_long(signal, self.strategy)

    # ---- 3. SHORT signal ----

    def test_short_signal_expansion_below_mid(self):
        """Price below middle BB with negative slope + prior squeeze -> short."""
        last_close = 95.0
        candles = _make_candles_with_last(
            last_close=last_close, last_volume=800.0, n=100, base_price=100.0,
        )
        features = {
            "bb_width_pct": 0.9,            # > 0.7 + > 0.85 heuristic
            "atr": 2.0,
            "bb_middle": 100.0,             # close (95) < bb_middle -> short
            "ma_slope": -0.5,               # negative slope
            "volume_sma": 400.0,            # 800 > 600 -> confirmed
            "prev_bb_width_pct": 0.10,      # < 0.2 -> prior squeeze
        }
        ctx = _build_market_context(regime="VOLATILE_BREAKOUT", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_valid_short(signal, self.strategy)
        assert signal.metadata.get("strategy") == "vol_expansion"

    # ---- 4. Invariants pass ----

    def test_invariants_on_produced_signals(self):
        """All produced signals pass invariants."""
        # Long
        candles = _make_candles_with_last(last_close=105.0, last_volume=800.0, n=100)
        feat = {
            "bb_width_pct": 0.9, "atr": 2.0, "bb_middle": 100.0,
            "ma_slope": 0.5, "volume_sma": 400.0, "prev_bb_width_pct": 0.1,
        }
        ctx = _build_market_context(regime="VOLATILE_BREAKOUT", features=feat, candles=candles)
        sig = self.strategy.compute_signal(ctx)
        assert self.strategy.invariants_ok(sig) is True

        # Short
        candles2 = _make_candles_with_last(last_close=95.0, last_volume=800.0, n=100)
        feat2 = {
            "bb_width_pct": 0.9, "atr": 2.0, "bb_middle": 100.0,
            "ma_slope": -0.5, "volume_sma": 400.0, "prev_bb_width_pct": 0.1,
        }
        ctx2 = _build_market_context(regime="VOLATILE_BREAKOUT", features=feat2, candles=candles2)
        sig2 = self.strategy.compute_signal(ctx2)
        assert self.strategy.invariants_ok(sig2) is True

    # ---- 5. Flat when features missing / insufficient ----

    def test_flat_when_no_features(self):
        """Returns flat when features dict is empty."""
        ctx = _build_market_context(regime="VOLATILE_BREAKOUT", features={})
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_bb_width_pct_missing(self):
        """Returns flat when bb_width_pct is missing."""
        features = {"atr": 2.0, "bb_middle": 100.0}
        ctx = _build_market_context(regime="VOLATILE_BREAKOUT", features=features)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_no_prior_squeeze(self):
        """Returns flat when there was no prior squeeze."""
        last_close = 105.0
        candles = _make_candles_with_last(last_close=last_close, last_volume=800.0, n=100)
        features = {
            "bb_width_pct": 0.75,           # > 0.7 expansion but NOT > 0.85 heuristic
            "atr": 2.0,
            "bb_middle": 100.0,
            "ma_slope": 0.5,
            "volume_sma": 400.0,
            "prev_bb_width_pct": 0.5,       # > 0.2 -> no prior squeeze
        }
        ctx = _build_market_context(regime="VOLATILE_BREAKOUT", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_expansion_insufficient(self):
        """Returns flat when bb_width_pct < expansion threshold (0.7)."""
        last_close = 105.0
        candles = _make_candles_with_last(last_close=last_close, last_volume=800.0, n=100)
        features = {
            "bb_width_pct": 0.5,            # < 0.7 -> no expansion
            "atr": 2.0,
            "bb_middle": 100.0,
            "ma_slope": 0.5,
            "volume_sma": 400.0,
            "prev_bb_width_pct": 0.1,       # prior squeeze exists
        }
        ctx = _build_market_context(regime="VOLATILE_BREAKOUT", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_atr_zero(self):
        """Returns flat when ATR is zero."""
        features = {
            "bb_width_pct": 0.9, "atr": 0.0, "bb_middle": 100.0,
            "ma_slope": 0.5, "prev_bb_width_pct": 0.1,
        }
        ctx = _build_market_context(regime="VOLATILE_BREAKOUT", features=features)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_candles_too_few(self):
        """Returns flat when fewer than 2 candles."""
        candles = _make_candles(n=1, base_price=100.0)
        features = {
            "bb_width_pct": 0.9, "atr": 2.0, "bb_middle": 100.0,
            "ma_slope": 0.5, "prev_bb_width_pct": 0.1,
        }
        ctx = _build_market_context(regime="VOLATILE_BREAKOUT", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_low_confidence_from_conflicting_slope_returns_flat(self):
        """Conflicting slope penalty can push confidence below min -> flat."""
        last_close = 105.0
        candles = _make_candles_with_last(
            last_close=last_close, last_volume=100.0, n=100, base_price=100.0,
        )
        features = {
            "bb_width_pct": 0.72,       # just above expansion threshold
            "atr": 2.0,
            "bb_middle": 100.0,
            "ma_slope": -0.5,           # conflicting: long direction but negative slope
            "volume_sma": 400.0,        # 100 < 600 -> volume NOT confirmed
            "prev_bb_width_pct": 0.1,
        }
        ctx = _build_market_context(regime="VOLATILE_BREAKOUT", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        # Confidence: 0.15 (base) + small expansion + 0.0 (vol) + (-0.10) (slope penalty)
        # Likely below 0.3 -> flat
        _assert_flat(signal)


# =======================================================================
# FundingExtremesStrategy
# =======================================================================


class TestFundingExtremesStrategy:
    """Tests for the FundingExtremesStrategy."""

    def setup_method(self):
        self.strategy = FundingExtremesStrategy()

    # ---- 1. Applicable regimes ----

    def test_applicable_regimes(self):
        """Strategy applies to TREND, RANGE, VOLATILE_BREAKOUT, MEAN_REVERT_CRASH."""
        expected = ["TREND", "RANGE", "VOLATILE_BREAKOUT", "MEAN_REVERT_CRASH"]
        assert self.strategy.applicable_regimes() == expected
        assert "UNKNOWN" not in self.strategy.applicable_regimes()
        assert "SQUEEZE_RISK" not in self.strategy.applicable_regimes()

    # ---- 2. LONG signal: fade crowded shorts ----

    def test_long_signal_extreme_negative_funding(self):
        """Extreme negative funding + oversold RSI -> fade shorts -> long."""
        last_close = 100.0
        candles = _make_candles_with_last(last_close=last_close, n=100, base_price=100.0)
        features = {
            "funding_zscore": -3.0,         # < -2.0 threshold
            "funding_percentile": 0.05,     # < 0.1 (1.0 - 0.9)
            "rsi": 18.0,                    # < 25 (rsi_exhaustion_long)
            "atr": 2.0,
        }
        ctx = _build_market_context(regime="TREND", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_valid_long(signal, self.strategy)
        assert signal.metadata.get("strategy") == "funding_extremes"

    def test_long_without_percentile_confirmation(self):
        """Long signal fires with z-score + RSI even without percentile bonus."""
        last_close = 100.0
        candles = _make_candles_with_last(last_close=last_close, n=100, base_price=100.0)
        features = {
            "funding_zscore": -3.5,
            "rsi": 15.0,
            "atr": 2.0,
            # funding_percentile omitted
        }
        ctx = _build_market_context(regime="RANGE", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_valid_long(signal, self.strategy)

    # ---- 3. SHORT signal: fade crowded longs ----

    def test_short_signal_extreme_positive_funding(self):
        """Extreme positive funding + overbought RSI -> fade longs -> short."""
        last_close = 100.0
        candles = _make_candles_with_last(last_close=last_close, n=100, base_price=100.0)
        features = {
            "funding_zscore": 3.0,          # > 2.0 threshold
            "funding_percentile": 0.95,     # > 0.9
            "rsi": 82.0,                    # > 75 (rsi_exhaustion_short)
            "atr": 2.0,
        }
        ctx = _build_market_context(regime="TREND", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_valid_short(signal, self.strategy)
        assert signal.metadata.get("strategy") == "funding_extremes"

    def test_short_without_percentile_confirmation(self):
        """Short fires with z-score + RSI even without percentile."""
        last_close = 100.0
        candles = _make_candles_with_last(last_close=last_close, n=100, base_price=100.0)
        features = {
            "funding_zscore": 4.0,
            "rsi": 85.0,
            "atr": 2.0,
            # funding_percentile omitted
        }
        ctx = _build_market_context(regime="VOLATILE_BREAKOUT", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_valid_short(signal, self.strategy)

    # ---- 4. Invariants pass ----

    def test_invariants_on_produced_signals(self):
        """Invariants pass on both long and short signals."""
        candles = _make_candles_with_last(last_close=100.0, n=100)
        # Long
        feat_long = {
            "funding_zscore": -3.0, "funding_percentile": 0.05,
            "rsi": 18.0, "atr": 2.0,
        }
        ctx_long = _build_market_context(regime="TREND", features=feat_long, candles=candles)
        sig_long = self.strategy.compute_signal(ctx_long)
        assert self.strategy.invariants_ok(sig_long) is True

        # Short
        feat_short = {
            "funding_zscore": 3.0, "funding_percentile": 0.95,
            "rsi": 82.0, "atr": 2.0,
        }
        ctx_short = _build_market_context(regime="TREND", features=feat_short, candles=candles)
        sig_short = self.strategy.compute_signal(ctx_short)
        assert self.strategy.invariants_ok(sig_short) is True

    # ---- 5. Flat when features missing / insufficient ----

    def test_flat_when_no_features(self):
        """Returns flat when features dict is empty."""
        ctx = _build_market_context(regime="TREND", features={})
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_funding_zscore_missing(self):
        """Returns flat when funding_zscore is missing."""
        features = {"rsi": 18.0, "atr": 2.0}
        ctx = _build_market_context(regime="TREND", features=features)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_rsi_missing(self):
        """Returns flat when RSI is missing."""
        features = {"funding_zscore": -3.0, "atr": 2.0}
        ctx = _build_market_context(regime="TREND", features=features)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_atr_missing(self):
        """Returns flat when ATR is missing."""
        features = {"funding_zscore": -3.0, "rsi": 18.0}
        ctx = _build_market_context(regime="TREND", features=features)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_funding_not_extreme(self):
        """Returns flat when funding z-score is within normal range."""
        last_close = 100.0
        candles = _make_candles_with_last(last_close=last_close, n=100)
        features = {
            "funding_zscore": 0.5,   # within [-2, 2]
            "rsi": 82.0,
            "atr": 2.0,
        }
        ctx = _build_market_context(regime="TREND", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_rsi_not_exhausted(self):
        """Returns flat when RSI is neutral despite extreme funding."""
        last_close = 100.0
        candles = _make_candles_with_last(last_close=last_close, n=100)
        features = {
            "funding_zscore": -3.0,
            "rsi": 50.0,            # neutral, not < 25
            "atr": 2.0,
        }
        ctx = _build_market_context(regime="TREND", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_atr_zero(self):
        """Returns flat when ATR is zero."""
        features = {"funding_zscore": -3.0, "rsi": 18.0, "atr": 0.0}
        ctx = _build_market_context(regime="TREND", features=features)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_candles_empty(self):
        """Returns flat when candles is empty."""
        candles = pd.DataFrame(columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
        features = {"funding_zscore": -3.0, "rsi": 18.0, "atr": 2.0}
        ctx = _build_market_context(regime="TREND", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_conflicting_funding_and_rsi(self):
        """Extreme positive funding but RSI oversold -> no match -> flat."""
        last_close = 100.0
        candles = _make_candles_with_last(last_close=last_close, n=100)
        features = {
            "funding_zscore": 3.0,   # positive -> would want short
            "rsi": 18.0,             # oversold -> exhaustion for long, not short
            "atr": 2.0,
        }
        ctx = _build_market_context(regime="TREND", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)


# =======================================================================
# SqueezeBreakoutStrategy
# =======================================================================


class TestSqueezeBreakoutStrategy:
    """Tests for the SqueezeBreakoutStrategy."""

    def setup_method(self):
        self.strategy = SqueezeBreakoutStrategy()

    # ---- 1. Non-applicable regimes ----

    @pytest.mark.parametrize(
        "regime",
        ["TREND", "RANGE", "VOLATILE_BREAKOUT", "MEAN_REVERT_CRASH", "UNKNOWN"],
    )
    def test_applicable_regimes(self, regime: str):
        """Strategy only applies to SQUEEZE_RISK."""
        assert regime not in self.strategy.applicable_regimes()
        assert self.strategy.applicable_regimes() == ["SQUEEZE_RISK"]

    # ---- 2. LONG signal ----

    def test_long_signal_squeeze_above_mid(self):
        """Price above middle BB with positive slope in a squeeze -> long."""
        last_close = 105.0
        candles = _make_candles_with_last(
            last_close=last_close, last_volume=500.0, n=100, base_price=100.0,
        )
        features = {
            "bb_width_pct": 0.05,       # < 0.15 (squeeze)
            "bb_middle": 100.0,         # close (105) > bb_middle -> above mid
            "ma_slope": 0.5,            # positive slope -> aligned with long
            "atr": 2.0,
            "volume_sma": 400.0,        # 500 > 0.8 * 400 = 320 -> ready
        }
        ctx = _build_market_context(regime="SQUEEZE_RISK", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_valid_long(signal, self.strategy)
        assert signal.metadata.get("strategy") == "squeeze_breakout"

    # ---- 3. SHORT signal ----

    def test_short_signal_squeeze_below_mid(self):
        """Price below middle BB with negative slope in a squeeze -> short."""
        last_close = 95.0
        candles = _make_candles_with_last(
            last_close=last_close, last_volume=500.0, n=100, base_price=100.0,
        )
        features = {
            "bb_width_pct": 0.05,
            "bb_middle": 100.0,         # close (95) < bb_middle -> below mid
            "ma_slope": -0.5,           # negative slope -> aligned with short
            "atr": 2.0,
            "volume_sma": 400.0,
        }
        ctx = _build_market_context(regime="SQUEEZE_RISK", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_valid_short(signal, self.strategy)
        assert signal.metadata.get("strategy") == "squeeze_breakout"

    # ---- 4. Invariants pass ----

    def test_invariants_on_produced_signals(self):
        """Invariants pass for all produced signals."""
        # Long
        candles = _make_candles_with_last(last_close=105.0, last_volume=500.0, n=100)
        feat = {
            "bb_width_pct": 0.05, "bb_middle": 100.0, "ma_slope": 0.5,
            "atr": 2.0, "volume_sma": 400.0,
        }
        ctx = _build_market_context(regime="SQUEEZE_RISK", features=feat, candles=candles)
        sig = self.strategy.compute_signal(ctx)
        assert self.strategy.invariants_ok(sig) is True

        # Short
        candles2 = _make_candles_with_last(last_close=95.0, last_volume=500.0, n=100)
        feat2 = {
            "bb_width_pct": 0.05, "bb_middle": 100.0, "ma_slope": -0.5,
            "atr": 2.0, "volume_sma": 400.0,
        }
        ctx2 = _build_market_context(regime="SQUEEZE_RISK", features=feat2, candles=candles2)
        sig2 = self.strategy.compute_signal(ctx2)
        assert self.strategy.invariants_ok(sig2) is True

    # ---- 5. Flat when features missing / insufficient ----

    def test_flat_when_no_features(self):
        """Returns flat when features dict is empty."""
        ctx = _build_market_context(regime="SQUEEZE_RISK", features={})
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_bb_width_pct_missing(self):
        """Returns flat when bb_width_pct is missing."""
        features = {"bb_middle": 100.0, "ma_slope": 0.5, "atr": 2.0}
        ctx = _build_market_context(regime="SQUEEZE_RISK", features=features)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_ma_slope_missing(self):
        """Returns flat when ma_slope is missing."""
        features = {"bb_width_pct": 0.05, "bb_middle": 100.0, "atr": 2.0}
        ctx = _build_market_context(regime="SQUEEZE_RISK", features=features)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_no_squeeze(self):
        """Returns flat when bb_width_pct >= squeeze threshold (0.15)."""
        last_close = 105.0
        candles = _make_candles_with_last(last_close=last_close, last_volume=500.0, n=100)
        features = {
            "bb_width_pct": 0.5,    # NOT in squeeze (>= 0.15)
            "bb_middle": 100.0,
            "ma_slope": 0.5,
            "atr": 2.0,
            "volume_sma": 400.0,
        }
        ctx = _build_market_context(regime="SQUEEZE_RISK", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_volume_not_ready(self):
        """Returns flat when current volume < readiness * volume_sma."""
        last_close = 105.0
        candles = _make_candles_with_last(
            last_close=last_close, last_volume=100.0, n=100, base_price=100.0,
        )
        features = {
            "bb_width_pct": 0.05,
            "bb_middle": 100.0,
            "ma_slope": 0.5,
            "atr": 2.0,
            "volume_sma": 500.0,    # 100 < 0.8 * 500 = 400 -> not ready
        }
        ctx = _build_market_context(regime="SQUEEZE_RISK", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_direction_conflicting(self):
        """Returns flat when price above mid but slope negative (no alignment)."""
        last_close = 105.0
        candles = _make_candles_with_last(
            last_close=last_close, last_volume=500.0, n=100, base_price=100.0,
        )
        features = {
            "bb_width_pct": 0.05,
            "bb_middle": 100.0,         # close > mid -> would be long
            "ma_slope": -0.5,           # negative slope -> conflicts with long
            "atr": 2.0,
            "volume_sma": 400.0,
        }
        ctx = _build_market_context(regime="SQUEEZE_RISK", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        # price_above_mid=True, slope_positive=False -> side stays flat
        _assert_flat(signal)

    def test_flat_when_atr_zero(self):
        """Returns flat when ATR is zero."""
        features = {
            "bb_width_pct": 0.05, "bb_middle": 100.0, "ma_slope": 0.5, "atr": 0.0,
        }
        ctx = _build_market_context(regime="SQUEEZE_RISK", features=features)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_candles_too_few(self):
        """Returns flat when fewer than 2 candles."""
        candles = _make_candles(n=1, base_price=100.0)
        features = {
            "bb_width_pct": 0.05, "bb_middle": 100.0, "ma_slope": 0.5,
            "atr": 2.0, "volume_sma": 400.0,
        }
        ctx = _build_market_context(regime="SQUEEZE_RISK", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_long_without_volume_sma_still_works(self):
        """When volume_sma is absent, volume_ready defaults True."""
        last_close = 105.0
        candles = _make_candles_with_last(
            last_close=last_close, last_volume=500.0, n=100, base_price=100.0,
        )
        features = {
            "bb_width_pct": 0.05,
            "bb_middle": 100.0,
            "ma_slope": 0.5,
            "atr": 2.0,
            # volume_sma omitted -> defaults to volume_ready=True
        }
        ctx = _build_market_context(regime="SQUEEZE_RISK", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_valid_long(signal, self.strategy)


# =======================================================================
# CrashMeanRevStrategy
# =======================================================================


class TestCrashMeanRevStrategy:
    """Tests for the CrashMeanRevStrategy."""

    def setup_method(self):
        self.strategy = CrashMeanRevStrategy()

    # ---- 1. Non-applicable regimes ----

    @pytest.mark.parametrize(
        "regime",
        ["TREND", "RANGE", "VOLATILE_BREAKOUT", "SQUEEZE_RISK", "UNKNOWN"],
    )
    def test_applicable_regimes(self, regime: str):
        """Strategy only applies to MEAN_REVERT_CRASH."""
        assert regime not in self.strategy.applicable_regimes()
        assert self.strategy.applicable_regimes() == ["MEAN_REVERT_CRASH"]

    # ---- 2. LONG signal: catching falling knife ----

    def test_long_signal_crash_exhaustion(self):
        """Close below lower BB + extreme oversold RSI + high wick + high vol -> long."""
        bb_lower = 96.0
        bb_upper = 106.0
        bb_middle = 101.0
        last_close = 94.0   # below bb_lower
        candles = _make_candles_with_last(last_close=last_close, n=100, base_price=100.0)
        features = {
            "rsi": 12.0,               # < 20 (rsi_oversold)
            "atr": 3.0,
            "wick_ratio": 0.75,         # > 0.6 (min_wick_ratio)
            "realized_vol": 0.8,        # > 0.5 (min_realized_vol)
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_middle": bb_middle,
        }
        ctx = _build_market_context(regime="MEAN_REVERT_CRASH", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_valid_long(signal, self.strategy)
        assert signal.metadata.get("strategy") == "crash_meanrev"

    # ---- 3. SHORT signal: fading spike ----

    def test_short_signal_spike_exhaustion(self):
        """Close above upper BB + extreme overbought RSI + high wick + high vol -> short."""
        bb_lower = 94.0
        bb_upper = 104.0
        bb_middle = 99.0
        last_close = 107.0   # above bb_upper
        candles = _make_candles_with_last(last_close=last_close, n=100, base_price=100.0)
        features = {
            "rsi": 88.0,               # > 80 (rsi_overbought)
            "atr": 3.0,
            "wick_ratio": 0.75,
            "realized_vol": 0.8,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_middle": bb_middle,
        }
        ctx = _build_market_context(regime="MEAN_REVERT_CRASH", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_valid_short(signal, self.strategy)
        assert signal.metadata.get("strategy") == "crash_meanrev"

    # ---- 4. Invariants pass ----

    def test_invariants_on_produced_signals(self):
        """Invariants pass for both long and short signals."""
        # Long
        candles = _make_candles_with_last(last_close=93.0, n=100, base_price=100.0)
        feat = {
            "rsi": 10.0, "atr": 3.0, "wick_ratio": 0.8, "realized_vol": 1.0,
            "bb_upper": 106.0, "bb_lower": 96.0, "bb_middle": 101.0,
        }
        ctx = _build_market_context(regime="MEAN_REVERT_CRASH", features=feat, candles=candles)
        sig = self.strategy.compute_signal(ctx)
        assert self.strategy.invariants_ok(sig) is True

        # Short
        candles2 = _make_candles_with_last(last_close=108.0, n=100, base_price=100.0)
        feat2 = {
            "rsi": 90.0, "atr": 3.0, "wick_ratio": 0.8, "realized_vol": 1.0,
            "bb_upper": 106.0, "bb_lower": 96.0, "bb_middle": 101.0,
        }
        ctx2 = _build_market_context(regime="MEAN_REVERT_CRASH", features=feat2, candles=candles2)
        sig2 = self.strategy.compute_signal(ctx2)
        assert self.strategy.invariants_ok(sig2) is True

    # ---- 5. Flat when features missing / insufficient ----

    def test_flat_when_no_features(self):
        """Returns flat when features dict is empty."""
        ctx = _build_market_context(regime="MEAN_REVERT_CRASH", features={})
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_rsi_missing(self):
        """Returns flat when RSI is missing."""
        features = {
            "atr": 3.0, "wick_ratio": 0.75, "realized_vol": 0.8,
            "bb_upper": 106.0, "bb_lower": 96.0, "bb_middle": 101.0,
        }
        ctx = _build_market_context(regime="MEAN_REVERT_CRASH", features=features)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_wick_ratio_missing(self):
        """Returns flat when wick_ratio is missing."""
        features = {
            "rsi": 12.0, "atr": 3.0, "realized_vol": 0.8,
            "bb_upper": 106.0, "bb_lower": 96.0, "bb_middle": 101.0,
        }
        ctx = _build_market_context(regime="MEAN_REVERT_CRASH", features=features)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_realized_vol_missing(self):
        """Returns flat when realized_vol is missing."""
        features = {
            "rsi": 12.0, "atr": 3.0, "wick_ratio": 0.75,
            "bb_upper": 106.0, "bb_lower": 96.0, "bb_middle": 101.0,
        }
        ctx = _build_market_context(regime="MEAN_REVERT_CRASH", features=features)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_bb_features_missing(self):
        """Returns flat when Bollinger Band features are missing."""
        features = {
            "rsi": 12.0, "atr": 3.0, "wick_ratio": 0.75, "realized_vol": 0.8,
            # bb_upper, bb_lower, bb_middle all missing
        }
        ctx = _build_market_context(regime="MEAN_REVERT_CRASH", features=features)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_realized_vol_too_low(self):
        """Returns flat when realized_vol < min_realized_vol (0.5)."""
        last_close = 94.0
        candles = _make_candles_with_last(last_close=last_close, n=100, base_price=100.0)
        features = {
            "rsi": 12.0, "atr": 3.0, "wick_ratio": 0.75,
            "realized_vol": 0.3,    # < 0.5
            "bb_upper": 106.0, "bb_lower": 96.0, "bb_middle": 101.0,
        }
        ctx = _build_market_context(regime="MEAN_REVERT_CRASH", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_wick_ratio_too_low(self):
        """Returns flat when wick_ratio < min_wick_ratio (0.6)."""
        last_close = 94.0
        candles = _make_candles_with_last(last_close=last_close, n=100, base_price=100.0)
        features = {
            "rsi": 12.0, "atr": 3.0,
            "wick_ratio": 0.3,      # < 0.6
            "realized_vol": 0.8,
            "bb_upper": 106.0, "bb_lower": 96.0, "bb_middle": 101.0,
        }
        ctx = _build_market_context(regime="MEAN_REVERT_CRASH", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_close_inside_bands(self):
        """Returns flat when close is between lower and upper BB (no breach)."""
        last_close = 100.0  # inside bands
        candles = _make_candles_with_last(last_close=last_close, n=100, base_price=100.0)
        features = {
            "rsi": 12.0, "atr": 3.0, "wick_ratio": 0.75, "realized_vol": 0.8,
            "bb_upper": 106.0, "bb_lower": 96.0, "bb_middle": 101.0,
        }
        ctx = _build_market_context(regime="MEAN_REVERT_CRASH", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_rsi_not_extreme(self):
        """Returns flat when RSI is not extreme despite BB breach."""
        last_close = 94.0   # below lower BB
        candles = _make_candles_with_last(last_close=last_close, n=100, base_price=100.0)
        features = {
            "rsi": 45.0,           # not extreme (between 20 and 80)
            "atr": 3.0,
            "wick_ratio": 0.75,
            "realized_vol": 0.8,
            "bb_upper": 106.0, "bb_lower": 96.0, "bb_middle": 101.0,
        }
        ctx = _build_market_context(regime="MEAN_REVERT_CRASH", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_atr_zero(self):
        """Returns flat when ATR is zero."""
        features = {
            "rsi": 12.0, "atr": 0.0, "wick_ratio": 0.75, "realized_vol": 0.8,
            "bb_upper": 106.0, "bb_lower": 96.0, "bb_middle": 101.0,
        }
        ctx = _build_market_context(regime="MEAN_REVERT_CRASH", features=features)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_candles_empty(self):
        """Returns flat when candles is empty."""
        candles = pd.DataFrame(columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
        features = {
            "rsi": 12.0, "atr": 3.0, "wick_ratio": 0.75, "realized_vol": 0.8,
            "bb_upper": 106.0, "bb_lower": 96.0, "bb_middle": 101.0,
        }
        ctx = _build_market_context(regime="MEAN_REVERT_CRASH", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_flat_when_bb_width_zero(self):
        """Returns flat when BB upper == BB lower (zero width)."""
        last_close = 94.0
        candles = _make_candles_with_last(last_close=last_close, n=100, base_price=100.0)
        features = {
            "rsi": 12.0, "atr": 3.0, "wick_ratio": 0.75, "realized_vol": 0.8,
            "bb_upper": 100.0, "bb_lower": 100.0, "bb_middle": 100.0,
        }
        ctx = _build_market_context(regime="MEAN_REVERT_CRASH", features=features, candles=candles)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)

    def test_nan_feature_treated_as_missing(self):
        """NaN values in features should be treated as missing -> flat."""
        features = {
            "rsi": float("nan"), "atr": 3.0, "wick_ratio": 0.75, "realized_vol": 0.8,
            "bb_upper": 106.0, "bb_lower": 96.0, "bb_middle": 101.0,
        }
        ctx = _build_market_context(regime="MEAN_REVERT_CRASH", features=features)
        signal = self.strategy.compute_signal(ctx)
        _assert_flat(signal)


# =======================================================================
# Cross-strategy: metadata and signal structure
# =======================================================================


class TestCrossStrategySignalStructure:
    """Cross-cutting tests that apply to all strategies."""

    STRATEGY_CLASSES = [
        TrendBreakoutStrategy,
        RangeMeanRevStrategy,
        VolExpansionStrategy,
        FundingExtremesStrategy,
        SqueezeBreakoutStrategy,
        CrashMeanRevStrategy,
    ]

    def test_all_strategies_have_unique_names(self):
        """Each strategy has a distinct name."""
        names = [cls().name for cls in self.STRATEGY_CLASSES]
        assert len(names) == len(set(names)), f"Duplicate strategy names: {names}"

    def test_all_strategies_return_signal_type(self):
        """compute_signal always returns a Signal instance, even with empty ctx."""
        for cls in self.STRATEGY_CLASSES:
            strategy = cls()
            ctx = _build_market_context(regime="UNKNOWN", features={})
            signal = strategy.compute_signal(ctx)
            assert isinstance(signal, Signal), (
                f"{strategy.name} did not return a Signal instance"
            )

    def test_all_flat_signals_pass_invariants(self):
        """Flat signals from every strategy pass invariants_ok."""
        for cls in self.STRATEGY_CLASSES:
            strategy = cls()
            ctx = _build_market_context(regime="UNKNOWN", features={})
            signal = strategy.compute_signal(ctx)
            assert strategy.invariants_ok(signal) is True, (
                f"{strategy.name} flat signal failed invariants"
            )

    def test_all_strategies_have_applicable_regimes(self):
        """Every strategy declares at least one applicable regime."""
        for cls in self.STRATEGY_CLASSES:
            strategy = cls()
            regimes = strategy.applicable_regimes()
            assert len(regimes) > 0, f"{strategy.name} has no applicable regimes"

    def test_metadata_includes_strategy_name(self):
        """Non-flat signals should include the strategy name in metadata."""
        # Build feature sets that will trigger each strategy
        trigger_configs = [
            # TrendBreakout long
            (
                TrendBreakoutStrategy(),
                "TREND",
                {"adx": 35.0, "rsi": 32.0, "atr": 2.0, "sma_50": 105.0},
                110.0,
            ),
            # RangeMeanRev long
            (
                RangeMeanRevStrategy(),
                "RANGE",
                {
                    "adx": 15.0, "rsi": 22.0, "atr": 2.0,
                    "bb_upper": 105.0, "bb_lower": 95.1, "bb_middle": 100.0,
                },
                95.0,
            ),
            # FundingExtremes long
            (
                FundingExtremesStrategy(),
                "TREND",
                {
                    "funding_zscore": -3.0, "funding_percentile": 0.05,
                    "rsi": 18.0, "atr": 2.0,
                },
                100.0,
            ),
            # CrashMeanRev long
            (
                CrashMeanRevStrategy(),
                "MEAN_REVERT_CRASH",
                {
                    "rsi": 12.0, "atr": 3.0, "wick_ratio": 0.75, "realized_vol": 0.8,
                    "bb_upper": 106.0, "bb_lower": 96.0, "bb_middle": 101.0,
                },
                94.0,
            ),
        ]
        for strategy, regime, features, close_price in trigger_configs:
            candles = _make_candles_with_last(last_close=close_price, n=100, base_price=100.0)
            ctx = _build_market_context(regime=regime, features=features, candles=candles)
            signal = strategy.compute_signal(ctx)
            if signal.side != "flat":
                assert "strategy" in signal.metadata, (
                    f"{strategy.name} non-flat signal missing 'strategy' in metadata"
                )
                assert signal.metadata["strategy"] == strategy.name
