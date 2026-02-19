"""Smoke tests: verify all major modules can be imported without error."""

import importlib

# -----------------------------------------------------------------------
# Top-level package
# -----------------------------------------------------------------------


def test_import_autotrader_version():
    """The top-level ``autotrader`` package exposes a __version__ string."""
    import autotrader

    assert hasattr(autotrader, "__version__")
    assert isinstance(autotrader.__version__, str)
    assert autotrader.__version__ == "0.1.0"


# -----------------------------------------------------------------------
# autotrader.hl.*
# -----------------------------------------------------------------------


def test_import_hl_client():
    mod = importlib.import_module("autotrader.hl.client")
    assert hasattr(mod, "HLClient")


def test_import_hl_ws():
    mod = importlib.import_module("autotrader.hl.ws")
    assert hasattr(mod, "HLWebSocket")


def test_import_hl_types():
    mod = importlib.import_module("autotrader.hl.types")
    assert hasattr(mod, "Signal")
    assert hasattr(mod, "CandleData")
    assert hasattr(mod, "OrderSpec")


def test_import_hl_rate_limiter():
    mod = importlib.import_module("autotrader.hl.rate_limiter")
    assert callable(mod.acquire)
    assert callable(mod.try_acquire)
    assert callable(mod.available)


def test_import_hl_nonces():
    mod = importlib.import_module("autotrader.hl.nonces")
    assert hasattr(mod, "NonceManager")
    assert callable(mod.get_next)
    assert callable(mod.init)


# -----------------------------------------------------------------------
# autotrader.data.collectors.*
# -----------------------------------------------------------------------


def test_import_data_collectors_candles():
    importlib.import_module("autotrader.data.collectors.candles")


def test_import_data_collectors_funding_oi():
    importlib.import_module("autotrader.data.collectors.funding_oi")


def test_import_data_collectors_l2book():
    importlib.import_module("autotrader.data.collectors.l2book")


def test_import_data_collectors_user_state():
    importlib.import_module("autotrader.data.collectors.user_state")


# -----------------------------------------------------------------------
# autotrader.data.transforms.*
# -----------------------------------------------------------------------


def test_import_data_transforms_cleaning():
    importlib.import_module("autotrader.data.transforms.cleaning")


def test_import_data_transforms_resample():
    importlib.import_module("autotrader.data.transforms.resample")


# -----------------------------------------------------------------------
# autotrader.store.*
# -----------------------------------------------------------------------


def test_import_store_datastore():
    importlib.import_module("autotrader.store.datastore")


def test_import_store_parquet():
    importlib.import_module("autotrader.store.parquet")


def test_import_store_dataset_hash():
    importlib.import_module("autotrader.store.dataset_hash")


# -----------------------------------------------------------------------
# autotrader.features.*
# -----------------------------------------------------------------------


def test_import_features_technical():
    importlib.import_module("autotrader.features.technical")


def test_import_features_microstructure():
    importlib.import_module("autotrader.features.microstructure")


def test_import_features_positioning():
    importlib.import_module("autotrader.features.positioning")


# -----------------------------------------------------------------------
# autotrader.regimes.*
# -----------------------------------------------------------------------


def test_import_regimes_classifier():
    mod = importlib.import_module("autotrader.regimes.classifier")
    assert hasattr(mod, "RegimeClassifier")
    assert callable(mod.classify)


def test_import_regimes_hysteresis():
    mod = importlib.import_module("autotrader.regimes.hysteresis")
    assert hasattr(mod, "HysteresisFilter")
    assert callable(mod.apply)


# -----------------------------------------------------------------------
# autotrader.strategies.*
# -----------------------------------------------------------------------


def test_import_strategies_base():
    mod = importlib.import_module("autotrader.strategies.base")
    assert hasattr(mod, "BaseStrategy")
    assert hasattr(mod, "MarketContext")
    assert hasattr(mod, "IStrategy")
    assert callable(mod.flat_signal)


def test_import_strategies_trend_breakout():
    mod = importlib.import_module("autotrader.strategies.trend_breakout")
    assert hasattr(mod, "TrendBreakoutStrategy")


def test_import_strategies_range_meanrev():
    mod = importlib.import_module("autotrader.strategies.range_meanrev")
    assert hasattr(mod, "RangeMeanRevStrategy")


def test_import_strategies_vol_expansion():
    mod = importlib.import_module("autotrader.strategies.vol_expansion")
    assert hasattr(mod, "VolExpansionStrategy")


def test_import_strategies_funding_extremes():
    mod = importlib.import_module("autotrader.strategies.funding_extremes")
    assert hasattr(mod, "FundingExtremesStrategy")


def test_import_strategies_ensemble():
    mod = importlib.import_module("autotrader.strategies.ensemble")
    assert hasattr(mod, "EnsembleStrategy")
    assert callable(mod.combine)


# -----------------------------------------------------------------------
# autotrader.risk.*
# -----------------------------------------------------------------------


def test_import_risk_constraints():
    mod = importlib.import_module("autotrader.risk.constraints")
    assert hasattr(mod, "RiskConfig")
    assert hasattr(mod, "RiskState")
    assert callable(mod.check_portfolio)
    assert callable(mod.check_trade)
    assert callable(mod.load_risk_config)


def test_import_risk_exposure():
    importlib.import_module("autotrader.risk.exposure")


def test_import_risk_leverage():
    importlib.import_module("autotrader.risk.leverage")


def test_import_risk_sizing():
    importlib.import_module("autotrader.risk.sizing")


def test_import_risk_approvals():
    importlib.import_module("autotrader.risk.approvals")


# -----------------------------------------------------------------------
# autotrader.backtest.*
# -----------------------------------------------------------------------


def test_import_backtest_engine():
    importlib.import_module("autotrader.backtest.engine")


def test_import_backtest_cost_model():
    importlib.import_module("autotrader.backtest.cost_model")


def test_import_backtest_metrics():
    importlib.import_module("autotrader.backtest.metrics")


def test_import_backtest_reporting():
    importlib.import_module("autotrader.backtest.reporting")


def test_import_backtest_walkforward():
    importlib.import_module("autotrader.backtest.walkforward")


def test_import_backtest_robustness():
    importlib.import_module("autotrader.backtest.robustness")


# -----------------------------------------------------------------------
# autotrader.execution.*
# -----------------------------------------------------------------------


def test_import_execution_slippage():
    importlib.import_module("autotrader.execution.slippage")


def test_import_execution_broker():
    importlib.import_module("autotrader.execution.broker")


def test_import_execution_order_manager():
    importlib.import_module("autotrader.execution.order_manager")


def test_import_execution_reconciliation():
    importlib.import_module("autotrader.execution.reconciliation")


# -----------------------------------------------------------------------
# autotrader.governance.*
# -----------------------------------------------------------------------


def test_import_governance_registry():
    importlib.import_module("autotrader.governance.registry")


def test_import_governance_gates():
    importlib.import_module("autotrader.governance.gates")


def test_import_governance_drift():
    importlib.import_module("autotrader.governance.drift")


def test_import_governance_probation():
    importlib.import_module("autotrader.governance.probation")


def test_import_governance_approvals():
    importlib.import_module("autotrader.governance.approvals")


# -----------------------------------------------------------------------
# autotrader.monitoring.*
# -----------------------------------------------------------------------


def test_import_monitoring_logger():
    mod = importlib.import_module("autotrader.monitoring.logger")
    assert callable(mod.setup_logging)
    assert callable(mod.get_logger)


def test_import_monitoring_metrics():
    mod = importlib.import_module("autotrader.monitoring.metrics")
    assert hasattr(mod, "MetricsCollector")


def test_import_monitoring_alerts():
    importlib.import_module("autotrader.monitoring.alerts")


# -----------------------------------------------------------------------
# autotrader.runtime.*
# -----------------------------------------------------------------------


def test_import_runtime_startup_checks():
    mod = importlib.import_module("autotrader.runtime.startup_checks")
    assert callable(mod.run_startup_checks)


def test_import_runtime_kill_switch():
    importlib.import_module("autotrader.runtime.kill_switch")


def test_import_runtime_scheduler():
    mod = importlib.import_module("autotrader.runtime.scheduler")
    assert hasattr(mod, "TradingScheduler")


# -----------------------------------------------------------------------
# autotrader.main
# -----------------------------------------------------------------------


def test_import_main():
    mod = importlib.import_module("autotrader.main")
    assert callable(mod.main)
