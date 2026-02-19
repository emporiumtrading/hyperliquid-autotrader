"""Tests for the regime classifier and hysteresis filter."""

from autotrader.regimes.classifier import REGIME_LABELS, RegimeClassifier, classify
from autotrader.regimes.hysteresis import HysteresisFilter

# -----------------------------------------------------------------------
# RegimeClassifier
# -----------------------------------------------------------------------


class TestRegimeClassifier:
    def test_classify_trend(self):
        """High ADX + high hurst + strong RSI deviation -> TREND."""
        clf = RegimeClassifier()
        features = {
            "adx": 45.0,  # well above trend threshold (25)
            "hurst": 0.75,  # well above trend threshold (0.6)
            "rsi": 72.0,  # strongly away from 50
            "ma_slope": 0.8,  # positive slope (uptrend)
            "bb_width_pct": 0.6,
            "realized_vol": 0.5,
            "vol_ratio": 1.0,
        }
        regime, confidence = clf.classify(features)
        assert regime == "TREND", f"Expected TREND, got {regime}"
        assert confidence > 0.0

    def test_classify_range(self):
        """Low ADX + low hurst + RSI near 50 -> RANGE."""
        clf = RegimeClassifier()
        features = {
            "adx": 12.0,  # well below range threshold (20)
            "hurst": 0.25,  # well below range threshold (0.4)
            "rsi": 50.0,  # perfectly centered
            "bb_width_pct": 0.15,  # narrow bands
            "realized_vol": 0.3,
            "vol_ratio": 0.8,
            "ma_slope": 0.0,
        }
        regime, confidence = clf.classify(features)
        assert regime == "RANGE", f"Expected RANGE, got {regime}"
        assert confidence > 0.0

    def test_classify_volatile_breakout(self):
        """High BB width + high vol ratio + rising ADX -> VOLATILE_BREAKOUT."""
        clf = RegimeClassifier()
        features = {
            "adx": 30.0,  # above range threshold
            "hurst": 0.5,  # neutral
            "rsi": 55.0,  # near center
            "bb_width_pct": 0.95,  # very high -> expansion
            "realized_vol": 1.5,  # high vol
            "vol_ratio": 2.5,  # well above expansion threshold (1.5)
            "ma_slope": 0.3,
        }
        regime, confidence = clf.classify(features)
        assert regime == "VOLATILE_BREAKOUT", f"Expected VOLATILE_BREAKOUT, got {regime}"
        assert confidence > 0.0

    def test_classify_unknown_empty(self):
        """Empty features dict -> UNKNOWN with 0.0 confidence."""
        clf = RegimeClassifier()
        regime, confidence = clf.classify({})
        assert regime == "UNKNOWN"
        assert confidence == 0.0

    def test_classify_unknown_insufficient_features(self):
        """Features missing critical keys (ADX, RSI) -> UNKNOWN."""
        clf = RegimeClassifier()
        regime, confidence = clf.classify({"hurst": 0.5, "ma_slope": 0.1})
        assert regime == "UNKNOWN"
        assert confidence == 0.0

    def test_classify_returns_confidence(self):
        """Confidence for a non-UNKNOWN regime is between 0 and 1."""
        clf = RegimeClassifier()
        features = {
            "adx": 40.0,
            "hurst": 0.7,
            "rsi": 65.0,
            "bb_width_pct": 0.5,
            "realized_vol": 0.5,
            "vol_ratio": 1.0,
            "ma_slope": 0.5,
        }
        regime, confidence = clf.classify(features)
        assert regime != "UNKNOWN"
        assert 0.0 < confidence <= 1.0

    def test_classify_module_level_function(self):
        """The module-level classify() function returns a string label."""
        result = classify({})
        assert isinstance(result, str)
        assert result == "UNKNOWN"

    def test_classify_all_labels_valid(self):
        """All regime labels returned by the classifier are in REGIME_LABELS."""
        clf = RegimeClassifier()
        test_cases = [
            {"adx": 45.0, "hurst": 0.8, "rsi": 70.0, "bb_width_pct": 0.5, "realized_vol": 0.5},
            {"adx": 10.0, "hurst": 0.2, "rsi": 50.0, "bb_width_pct": 0.1, "realized_vol": 0.2},
            {
                "adx": 30.0,
                "hurst": 0.5,
                "rsi": 55.0,
                "bb_width_pct": 0.95,
                "realized_vol": 1.5,
                "vol_ratio": 3.0,
            },
            {},
        ]
        for features in test_cases:
            regime, _ = clf.classify(features)
            assert regime in REGIME_LABELS, f"Regime {regime!r} not in {REGIME_LABELS}"


# -----------------------------------------------------------------------
# HysteresisFilter
# -----------------------------------------------------------------------


class TestHysteresisFilter:
    def test_hysteresis_requires_persistence(self):
        """Regime does not switch immediately; requires min_bars consecutive observations."""
        hf = HysteresisFilter(min_bars=3, min_confidence=0.4)

        # Start at UNKNOWN
        assert hf.current_regime == "UNKNOWN"

        # Feed TREND with sufficient confidence -- should NOT switch after 1 bar
        result1 = hf.update("TREND", 0.7)
        assert result1 == "UNKNOWN", "Should NOT switch after 1 bar"

        # Feed TREND again -- still not enough (need 3)
        result2 = hf.update("TREND", 0.7)
        assert result2 == "UNKNOWN", "Should NOT switch after 2 bars"

        # Third consecutive TREND -> NOW it switches
        result3 = hf.update("TREND", 0.7)
        assert result3 == "TREND", "Should switch after 3 bars"

    def test_hysteresis_stays_if_same_regime(self):
        """If the proposed regime matches the current regime, no transition is needed."""
        hf = HysteresisFilter(min_bars=3, min_confidence=0.4)
        # Force current regime to TREND via 3 consecutive updates
        hf.update("TREND", 0.7)
        hf.update("TREND", 0.7)
        hf.update("TREND", 0.7)
        assert hf.current_regime == "TREND"

        # Continue feeding TREND -- should stay immediately
        result = hf.update("TREND", 0.8)
        assert result == "TREND"

    def test_hysteresis_resets_pending_on_regime_change(self):
        """If the pending regime changes before reaching min_bars, counter resets."""
        hf = HysteresisFilter(min_bars=3, min_confidence=0.4)

        hf.update("TREND", 0.7)  # pending TREND, count=1
        hf.update("TREND", 0.7)  # pending TREND, count=2
        hf.update("RANGE", 0.6)  # switch to pending RANGE, count=1 (TREND count lost)
        hf.update("RANGE", 0.6)  # pending RANGE, count=2
        result = hf.update("RANGE", 0.6)  # pending RANGE, count=3 -> switch!
        assert result == "RANGE"

    def test_hysteresis_low_confidence_ignored(self):
        """Proposals below min_confidence are ignored entirely."""
        hf = HysteresisFilter(min_bars=2, min_confidence=0.5)

        hf.update("TREND", 0.3)  # below min_confidence -> ignored
        hf.update("TREND", 0.3)  # still ignored
        hf.update("TREND", 0.3)  # still ignored
        assert hf.current_regime == "UNKNOWN"

    def test_hysteresis_history_tracking(self):
        """The filter tracks history of (regime, confidence) tuples."""
        hf = HysteresisFilter(min_bars=2, min_confidence=0.4)

        hf.update("TREND", 0.6)
        hf.update("TREND", 0.7)

        assert len(hf.history) == 2
        # After 2 bars, regime switches to TREND
        assert hf.history[-1] == ("TREND", 0.7)

    def test_hysteresis_reset(self):
        """reset() clears all state back to initial values."""
        hf = HysteresisFilter(min_bars=2, min_confidence=0.4)
        hf.update("TREND", 0.7)
        hf.update("TREND", 0.7)
        assert hf.current_regime == "TREND"

        hf.reset()
        assert hf.current_regime == "UNKNOWN"
        assert hf.current_confidence == 0.0
        assert hf.pending_regime is None
        assert hf.pending_bars == 0
        assert hf.history == []
