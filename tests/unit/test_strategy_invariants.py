"""Tests for strategy base class, signal invariants, and ensemble strategy."""

import numpy as np
import pandas as pd

from autotrader.hl.types import Signal
from autotrader.strategies.base import BaseStrategy, MarketContext, flat_signal
from autotrader.strategies.ensemble import EnsembleStrategy

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _make_candles(n: int = 100, base_price: float = 100.0) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with a gentle uptrend."""
    rng = np.random.default_rng(42)
    timestamps = list(range(1_000_000, 1_000_000 + n * 60_000, 60_000))
    closes = base_price + np.cumsum(rng.normal(0.05, 0.5, size=n))
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


def _make_market_context(
    regime: str = "TREND",
    features: dict | None = None,
    candles: pd.DataFrame | None = None,
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
        account_equity=10_000.0,
    )


# -----------------------------------------------------------------------
# BaseStrategy
# -----------------------------------------------------------------------


class TestBaseStrategy:
    def test_base_strategy_flat_signal(self):
        """BaseStrategy.compute_signal always returns a flat signal."""
        strategy = BaseStrategy()
        ctx = _make_market_context()
        signal = strategy.compute_signal(ctx)

        assert isinstance(signal, Signal)
        assert signal.side == "flat"
        assert signal.confidence == 0.0
        assert signal.entry is None
        assert signal.stop is None
        assert signal.take_profit is None

    def test_base_strategy_name(self):
        """BaseStrategy has the name 'base'."""
        strategy = BaseStrategy()
        assert strategy.name == "base"

    def test_base_strategy_applicable_regimes_empty(self):
        """BaseStrategy by default applies to no regimes."""
        strategy = BaseStrategy()
        assert strategy.applicable_regimes() == []


# -----------------------------------------------------------------------
# invariants_ok
# -----------------------------------------------------------------------


class TestInvariantsOk:
    def test_invariants_ok_flat(self):
        """A flat signal always passes invariants_ok."""
        strategy = BaseStrategy()
        signal = flat_signal()
        assert strategy.invariants_ok(signal) is True

    def test_invariants_ok_valid_long(self):
        """A valid long signal (stop < entry < TP, confidence in [0,1]) passes."""
        strategy = BaseStrategy()
        signal = Signal(
            side="long",
            entry=100.0,
            stop=95.0,  # below entry
            take_profit=110.0,  # above entry
            confidence=0.75,
        )
        assert strategy.invariants_ok(signal) is True

    def test_invariants_ok_valid_short(self):
        """A valid short signal (stop > entry > TP, confidence in [0,1]) passes."""
        strategy = BaseStrategy()
        signal = Signal(
            side="short",
            entry=100.0,
            stop=105.0,  # above entry
            take_profit=90.0,  # below entry
            confidence=0.6,
        )
        assert strategy.invariants_ok(signal) is True

    def test_invariants_ok_missing_stop(self):
        """A non-flat signal without a stop price fails invariants_ok."""
        strategy = BaseStrategy()
        signal = Signal(
            side="long",
            entry=100.0,
            stop=None,  # missing!
            take_profit=110.0,
            confidence=0.5,
        )
        assert strategy.invariants_ok(signal) is False

    def test_invariants_ok_missing_entry(self):
        """A non-flat signal without an entry price fails invariants_ok."""
        strategy = BaseStrategy()
        signal = Signal(
            side="long",
            entry=None,  # missing!
            stop=95.0,
            take_profit=110.0,
            confidence=0.5,
        )
        assert strategy.invariants_ok(signal) is False

    def test_invariants_ok_missing_take_profit(self):
        """A non-flat signal without a take_profit fails invariants_ok."""
        strategy = BaseStrategy()
        signal = Signal(
            side="long",
            entry=100.0,
            stop=95.0,
            take_profit=None,  # missing!
            confidence=0.5,
        )
        assert strategy.invariants_ok(signal) is False

    def test_invariants_ok_wrong_stop_side_long(self):
        """Long signal with stop above entry fails invariants_ok."""
        strategy = BaseStrategy()
        signal = Signal(
            side="long",
            entry=100.0,
            stop=105.0,  # wrong: above entry for a long
            take_profit=110.0,
            confidence=0.5,
        )
        assert strategy.invariants_ok(signal) is False

    def test_invariants_ok_wrong_stop_side_short(self):
        """Short signal with stop below entry fails invariants_ok."""
        strategy = BaseStrategy()
        signal = Signal(
            side="short",
            entry=100.0,
            stop=95.0,  # wrong: below entry for a short
            take_profit=90.0,
            confidence=0.5,
        )
        assert strategy.invariants_ok(signal) is False

    def test_invariants_ok_wrong_tp_side_long(self):
        """Long signal with take_profit below entry fails invariants_ok."""
        strategy = BaseStrategy()
        signal = Signal(
            side="long",
            entry=100.0,
            stop=95.0,
            take_profit=90.0,  # wrong: below entry for a long
            confidence=0.5,
        )
        assert strategy.invariants_ok(signal) is False

    def test_invariants_ok_wrong_tp_side_short(self):
        """Short signal with take_profit above entry fails invariants_ok."""
        strategy = BaseStrategy()
        signal = Signal(
            side="short",
            entry=100.0,
            stop=105.0,
            take_profit=110.0,  # wrong: above entry for a short
            confidence=0.5,
        )
        assert strategy.invariants_ok(signal) is False

    def test_invariants_ok_confidence_out_of_range(self):
        """Confidence outside [0, 1] fails invariants_ok."""
        strategy = BaseStrategy()
        signal = Signal(
            side="long",
            entry=100.0,
            stop=95.0,
            take_profit=110.0,
            confidence=1.5,  # out of range
        )
        assert strategy.invariants_ok(signal) is False

    def test_invariants_ok_unrecognised_side(self):
        """An unrecognised side string (e.g. 'sideways') fails invariants_ok."""
        strategy = BaseStrategy()
        signal = Signal(
            side="sideways",
            entry=100.0,
            stop=95.0,
            take_profit=110.0,
            confidence=0.5,
        )
        assert strategy.invariants_ok(signal) is False


# -----------------------------------------------------------------------
# EnsembleStrategy
# -----------------------------------------------------------------------


class TestEnsembleStrategy:
    def test_ensemble_returns_signal(self):
        """EnsembleStrategy.compute_signal returns a Signal instance."""
        ensemble = EnsembleStrategy()
        ctx = _make_market_context(regime="UNKNOWN")
        signal = ensemble.compute_signal(ctx)
        assert isinstance(signal, Signal)

    def test_ensemble_flat_for_unknown_regime(self):
        """When the regime is UNKNOWN, no strategies are applicable -> flat."""
        ensemble = EnsembleStrategy()
        ctx = _make_market_context(regime="UNKNOWN")
        signal = ensemble.compute_signal(ctx)
        assert signal.side == "flat"

    def test_ensemble_selects_trend_strategies(self):
        """For TREND regime, the ensemble selects trend-applicable strategies."""
        ensemble = EnsembleStrategy()
        applicable = ensemble.select_strategies("TREND")
        names = [s.name for s in applicable]
        assert "trend_breakout" in names
        assert "funding_extremes" in names  # funding_extremes applies to TREND too

    def test_ensemble_with_context(self):
        """Build a MarketContext with real candles and run the full ensemble."""
        candles = _make_candles(n=100, base_price=3000.0)
        last_close = float(candles["close"].iloc[-1])

        # Provide features that the TrendBreakoutStrategy would need
        features = {
            "adx": 35.0,  # strong trend
            "rsi": 38.0,  # pulled back (below 40)
            "atr": 15.0,
            "sma_50": last_close - 20.0,  # price above MA (uptrend)
            "sma": last_close - 20.0,
            "hurst": 0.7,
            "bb_width_pct": 0.5,
            "realized_vol": 0.5,
            "ma_slope": 0.5,
            # For funding_extremes: provide but don't trigger
            "funding_zscore": 0.5,
            "funding_percentile": 0.5,
        }

        ctx = _make_market_context(
            regime="TREND",
            features=features,
            candles=candles,
        )

        ensemble = EnsembleStrategy()
        signal = ensemble.compute_signal(ctx)

        assert isinstance(signal, Signal)
        # With ADX=35 and RSI pulled back, trend_breakout should fire
        # (unless confidence too low).  Either way, we get a valid Signal.
        if signal.side != "flat":
            # If non-flat, it must pass invariants
            strategy = BaseStrategy()
            assert strategy.invariants_ok(signal) is True
            assert 0.0 <= signal.confidence <= 1.0
            assert signal.entry is not None
            assert signal.stop is not None
            assert signal.take_profit is not None
