"""Tests for governance: baseline registry, gate evaluation, drift detection, probation."""

from __future__ import annotations

import time
from unittest.mock import patch

from autotrader.governance.drift import DriftDetector
from autotrader.governance.gates import GateConfig, evaluate_candidate
from autotrader.governance.probation import ProbationEvaluator
from autotrader.governance.registry import BaselineRegistry
from autotrader.utils.time import MS_PER_DAY, now_ms

# ---------------------------------------------------------------------------
# BaselineRegistry tests
# ---------------------------------------------------------------------------


class TestBaselineRegistryCreateSaveLoad:
    def test_baseline_registry_create_save_load(self, tmp_path):
        """Create a baseline, save it, then load it back."""
        registry = BaselineRegistry(baselines_dir=str(tmp_path / "baselines"))
        baseline = registry.create_baseline(
            strategy_name="trend_breakout",
            strategy_config={"lookback": 20},
            metrics={"sharpe": 1.5, "max_drawdown": 0.08},
            dataset_hash="abc123",
            git_commit="deadbeef",
            run_id="run_001",
        )

        assert baseline["version"] == 1
        assert baseline["strategy_name"] == "trend_breakout"
        assert baseline["metrics_summary"]["sharpe"] == 1.5

        # Save
        history_path = registry.save_current(baseline)
        assert history_path  # non-empty

        # Load
        loaded = registry.load_current()
        assert loaded is not None
        assert loaded["version"] == 1
        assert loaded["strategy_name"] == "trend_breakout"
        assert loaded["dataset_hash"] == "abc123"


class TestBaselineRegistryHistory:
    def test_baseline_registry_history(self, tmp_path):
        """Save multiple baselines and verify list_history returns all."""
        registry = BaselineRegistry(baselines_dir=str(tmp_path / "baselines"))

        # Create and save first baseline
        b1 = registry.create_baseline(
            strategy_name="v1",
            strategy_config={},
            metrics={"sharpe": 1.0},
            dataset_hash="hash1",
        )
        registry.save_current(b1)

        # Brief pause to ensure different timestamp in filename
        time.sleep(0.05)

        # Create and save second baseline (version auto-increments)
        b2 = registry.create_baseline(
            strategy_name="v2",
            strategy_config={},
            metrics={"sharpe": 2.0},
            dataset_hash="hash2",
        )
        registry.save_current(b2)

        history = registry.list_history()
        assert len(history) == 2
        assert history[0]["version"] == 1
        assert history[1]["version"] == 2


class TestBaselineRegistryRollback:
    def test_baseline_registry_rollback(self, tmp_path):
        """Rollback restores a previous version as the current baseline."""
        registry = BaselineRegistry(baselines_dir=str(tmp_path / "baselines"))

        b1 = registry.create_baseline(
            strategy_name="v1",
            strategy_config={},
            metrics={"sharpe": 1.0},
            dataset_hash="hash1",
        )
        registry.save_current(b1)

        time.sleep(0.05)

        b2 = registry.create_baseline(
            strategy_name="v2",
            strategy_config={},
            metrics={"sharpe": 2.0},
            dataset_hash="hash2",
        )
        registry.save_current(b2)

        # Current should be v2
        current = registry.load_current()
        assert current["version"] == 2

        # Rollback to v1
        restored = registry.rollback(version=1)
        assert restored["version"] == 1

        # Current is now v1
        current_after = registry.load_current()
        assert current_after["version"] == 1


# ---------------------------------------------------------------------------
# Gate evaluation tests
# ---------------------------------------------------------------------------


class TestEvaluateCandidateNoBaseline:
    def test_evaluate_candidate_no_baseline(self):
        """First candidate with no baseline auto-passes."""
        candidate = {
            "utility": 0.5,
            "sharpe": 1.0,
            "max_drawdown": 0.05,
            "profit_factor": 2.0,
            "win_rate": 0.6,
            "walkforward": {"oos_pnl": 100.0},
            "robustness": {"pass_rate": 0.9},
        }
        result = evaluate_candidate(candidate, baseline_report=None)
        assert result["passed"] is True
        assert result["recommendation"] == "promote"


class TestEvaluateCandidateBeatsBaseline:
    def test_evaluate_candidate_beats_baseline(self):
        """A candidate that beats the baseline on all metrics passes."""
        baseline = {
            "utility": 0.3,
            "sharpe": 0.8,
            "max_drawdown": 0.10,
            "profit_factor": 1.5,
            "win_rate": 0.5,
            "cvar_95": 0.02,
        }
        candidate = {
            "utility": 0.5,
            "sharpe": 1.5,
            "max_drawdown": 0.05,
            "profit_factor": 2.0,
            "win_rate": 0.6,
            "cvar_95": 0.01,
            "walkforward": {"oos_pnl": 500.0},
            "robustness": {"pass_rate": 0.9},
        }
        config = GateConfig(
            utility_improvement_threshold=0.05,
            max_drawdown_pct=0.20,
            max_cvar_95=0.10,
            min_sharpe=0.5,
            min_profit_factor=1.2,
            min_win_rate=0.35,
        )
        result = evaluate_candidate(candidate, baseline, config)
        assert result["passed"] is True
        assert result["recommendation"] == "promote"


class TestEvaluateCandidateFails:
    def test_evaluate_candidate_fails(self):
        """A candidate with worse utility than baseline fails."""
        baseline = {
            "utility": 0.5,
            "sharpe": 1.5,
            "max_drawdown": 0.05,
            "profit_factor": 2.0,
            "win_rate": 0.6,
            "cvar_95": 0.02,
        }
        candidate = {
            "utility": 0.3,  # worse than baseline + threshold
            "sharpe": 0.3,  # below min_sharpe
            "max_drawdown": 0.30,  # over limit
            "profit_factor": 0.8,  # below minimum
            "win_rate": 0.2,  # below minimum
            "cvar_95": 0.01,
            "walkforward": {"oos_pnl": 500.0},
            "robustness": {"pass_rate": 0.9},
        }
        result = evaluate_candidate(candidate, baseline)
        assert result["passed"] is False
        assert len(result["reasons"]) > 0


# ---------------------------------------------------------------------------
# DriftDetector tests
# ---------------------------------------------------------------------------


class TestDriftDetectorNoDrift:
    def test_drift_detector_no_drift(self):
        """Normal observations with matching expected/realized should show no drift."""
        detector = DriftDetector(config={"lookback_days": 7})
        ts = now_ms()

        for i in range(20):
            detector.add_observation(
                timestamp_ms=ts - (20 - i) * 3600_000,
                expected_pnl=100.0,
                realized_pnl=100.0 + (i % 3 - 1) * 5,  # small noise around expected
                expected_slippage=1.0,
                realized_slippage=1.0 + (i % 2) * 0.2,
                predicted_regime="TREND",
                actual_outcome="TREND",
            )

        result = detector.check_drift()
        assert result["drifting"] is False
        assert result["severity"] == "none"


class TestDriftDetectorPnlDrift:
    def test_drift_detector_pnl_drift(self):
        """Consistently bad PnL relative to expected should trigger drift."""
        detector = DriftDetector(
            config={
                "lookback_days": 7,
                "pnl_drift_threshold": -2.0,
            }
        )
        ts = now_ms()

        for i in range(20):
            detector.add_observation(
                timestamp_ms=ts - (20 - i) * 3600_000,
                expected_pnl=100.0,
                realized_pnl=-50.0,  # consistently much worse
                expected_slippage=1.0,
                realized_slippage=1.0,
                predicted_regime="TREND",
                actual_outcome="TREND",
            )

        result = detector.check_drift()
        assert result["drifting"] is True
        # PnL drift should be in the signals
        assert any("PnL drift" in s for s in result["signals"])


# ---------------------------------------------------------------------------
# ProbationEvaluator tests
# ---------------------------------------------------------------------------


class TestProbationEvaluator:
    def test_probation_evaluator(self):
        """Test the full probation lifecycle with enough trades and time."""
        evaluator = ProbationEvaluator(
            config={
                "probation_days": 1,  # very short for testing
                "min_trades": 5,
                "max_drawdown_pct": 0.10,
                "min_profit_factor": 1.0,
            }
        )

        # Patch now_ms so we can control time passage
        start_time = now_ms()

        with patch("autotrader.governance.probation.now_ms", return_value=start_time):
            evaluator.start()

        # Add profitable trades
        for i in range(10):
            evaluator.add_trade({"pnl": 50.0, "side": "long"})

        # Add daily PnLs that don't breach 10% drawdown from peak cumulative.
        # Cumulative: 100, 200, 195, 275 -> peak=200, trough=195, dd=(195-200)/200=-0.025
        evaluator.add_daily_pnl(100.0)
        evaluator.add_daily_pnl(100.0)
        evaluator.add_daily_pnl(-5.0)
        evaluator.add_daily_pnl(80.0)

        # Simulate time passing beyond probation_days (1 day)
        future_time = start_time + 2 * MS_PER_DAY
        with patch("autotrader.governance.probation.now_ms", return_value=future_time):
            result = evaluator.evaluate()

        assert result["passed"] is True
        assert result["can_promote"] is True
        assert result["trades_count"] == 10
        assert result["pnl"] == 500.0  # 10 trades * 50
