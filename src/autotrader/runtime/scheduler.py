"""Main trading loop scheduler.

Orchestrates data collection, feature computation, regime classification,
strategy signal generation, risk approval, order execution, reconciliation,
drift detection, and kill switch monitoring in a single polling loop.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import structlog

from autotrader.data.collectors.candles import stream_candles_to_store
from autotrader.data.collectors.funding_oi import fetch_asset_contexts
from autotrader.data.collectors.l2book import snapshot_l2
from autotrader.data.collectors.user_state import fetch_user_state
from autotrader.execution.broker import Broker
from autotrader.execution.order_manager import OrderManager
from autotrader.execution.reconciliation import Reconciler
from autotrader.features.positioning import (
    funding_rate_percentile,
    funding_rate_zscore,
)
from autotrader.features.technical import (
    adx as compute_adx,
)
from autotrader.features.technical import (
    atr as compute_atr,
)
from autotrader.features.technical import (
    bb_width_percentile,
    bollinger_bands,
    hurst_exponent,
    sma,
)
from autotrader.features.technical import (
    ma_slope as compute_ma_slope,
)
from autotrader.features.technical import (
    realized_vol as compute_realized_vol,
)
from autotrader.features.technical import (
    rsi as compute_rsi,
)
from autotrader.features.technical import (
    volume_sma as compute_volume_sma,
)
from autotrader.features.technical import (
    wick_ratio as compute_wick_ratio,
)
from autotrader.governance.drift import DriftDetector
from autotrader.governance.probation import ProbationEvaluator
from autotrader.governance.registry import BaselineRegistry
from autotrader.hl.client import HLClient, create_client
from autotrader.hl.types import Signal
from autotrader.monitoring.metrics import metrics
from autotrader.regimes.classifier import RegimeClassifier
from autotrader.regimes.hysteresis import HysteresisFilter
from autotrader.risk.approvals import approve_trade
from autotrader.risk.constraints import (
    RiskConfig,
    RiskState,
    load_risk_config,
)
from autotrader.risk.exposure import ExposureTracker
from autotrader.runtime.kill_switch import KillSwitch
from autotrader.store.parquet import ParquetStore
from autotrader.strategies.base import MarketContext
from autotrader.strategies.ensemble import EnsembleStrategy
from autotrader.utils.ids import generate_trade_id
from autotrader.utils.time import now_ms

logger = structlog.get_logger(__name__)

# Map strategy signal sides to the canonical buy/sell used by the approval
# and execution layers.
_SIDE_TO_APPROVAL = {"long": "buy", "short": "sell", "buy": "buy", "sell": "sell"}


class TradingScheduler:
    """Orchestrates the full live trading loop.

    Parameters
    ----------
    config : dict
        The fully-resolved application configuration dictionary.  Expected
        sections: ``hyperliquid``, ``risk``, ``universe``, ``timeframes``,
        and optionally ``execution``, ``regime``, ``hysteresis``,
        ``drift``, ``probation``, ``baselines_dir``.
    """

    def __init__(self, config: dict) -> None:
        self.config = config

        # ---- Environment ----
        self.env: str = config.get("env", config.get("environment", "paper")).lower()

        # ---- HL client ----
        hl_cfg = config.get("hyperliquid", {})
        self.client: HLClient = create_client(hl_cfg)

        # ---- Data store ----
        store_dir = config.get("store_dir", "data/processed")
        self.store: ParquetStore = ParquetStore(base_dir=store_dir)

        # ---- Strategy ----
        strategy_cfg = config.get("strategy", None)
        self.strategy: EnsembleStrategy = EnsembleStrategy(config=strategy_cfg)

        # ---- Regime ----
        regime_cfg = config.get("regime", None)
        self.regime_classifier: RegimeClassifier = RegimeClassifier(regime_cfg)

        hyst_cfg = config.get("hysteresis", {})
        self.hysteresis: HysteresisFilter = HysteresisFilter(
            min_bars=int(hyst_cfg.get("min_bars", 3)),
            min_confidence=float(hyst_cfg.get("min_confidence", 0.4)),
        )

        # ---- Risk ----
        self.risk_config: RiskConfig = load_risk_config(config)
        self.exposure_tracker: ExposureTracker = ExposureTracker()

        # ---- Execution ----
        exec_cfg = config.get("execution", {})
        mode = "paper" if self.env == "paper" else "live"
        self.broker: Broker = Broker(client=self.client, mode=mode)
        self.order_manager: OrderManager = OrderManager(broker=self.broker)
        self.reconciler: Reconciler = Reconciler(
            exposure_tracker=self.exposure_tracker,
            order_manager=self.order_manager,
        )

        # ---- Governance ----
        drift_cfg = config.get("drift", None)
        self.drift_detector: DriftDetector = DriftDetector(config=drift_cfg)

        ks_path = config.get("kill_switch_path", "data/kill_switch.json")
        self.kill_switch: KillSwitch = KillSwitch(state_path=ks_path)

        baselines_dir = config.get("baselines_dir", "artifacts/baselines")
        self.registry: BaselineRegistry = BaselineRegistry(baselines_dir=baselines_dir)

        # Probation is only active in canary mode
        self.probation: ProbationEvaluator | None = None
        if self.env == "canary":
            probation_cfg = config.get("probation", None)
            self.probation = ProbationEvaluator(config=probation_cfg)
            self.probation.start()

        # ---- Timeframes / universe ----
        timeframes_raw = config.get("timeframes", ["15m"])
        if isinstance(timeframes_raw, str):
            self.timeframes: list[str] = [timeframes_raw]
        else:
            self.timeframes = list(timeframes_raw)
        self.primary_timeframe: str = self.timeframes[0]

        universe_cfg = config.get("universe", {})
        if isinstance(universe_cfg, list):
            self._static_universe: list[str] = list(universe_cfg)
            self._top_n: int = len(universe_cfg)
            self._min_volume: float = 0.0
        elif isinstance(universe_cfg, dict):
            self._static_universe = list(universe_cfg.get("symbols", []))
            self._top_n = int(universe_cfg.get("top_n", 20))
            self._min_volume = float(universe_cfg.get("min_volume", 0.0))
        else:
            self._static_universe = []
            self._top_n = 20
            self._min_volume = 0.0

        self.universe: list[str] = []

        # ---- Loop control ----
        self.poll_interval_sec: float = float(
            exec_cfg.get("poll_interval_sec", config.get("poll_interval_sec", 60.0))
        )
        self.running: bool = False

        # ---- Equity tracking (used for RiskState) ----
        self._equity: float = float(config.get("initial_equity", 10_000.0))
        self._peak_equity: float = self._equity

        logger.info(
            "scheduler.init",
            env=self.env,
            mode=mode,
            timeframes=self.timeframes,
            poll_interval_sec=self.poll_interval_sec,
        )

    # ------------------------------------------------------------------
    # Single iteration
    # ------------------------------------------------------------------

    def run_once(self) -> dict:
        """Execute a single iteration of the trading loop.

        Steps
        -----
        1. Check kill switch -- abort if triggered.
        2. Refresh the trading universe.
        3. For each symbol in the universe:
           a. Fetch / update candles.
           b. Fetch L2 book snapshot.
           c. Compute features from candle data.
           d. Classify regime with hysteresis.
           e. Build ``MarketContext``.
           f. Generate strategy signal.
           g. If signal is actionable, run risk approval.
           h. If approved, submit order via the order manager.
        4. Reconcile fills.
        5. Check kill switch auto-trigger conditions.
        6. Check for performance drift.
        7. Update metrics gauges.
        8. Return a summary dict.

        Returns
        -------
        dict
            Summary of the iteration with keys: ``timestamp_ms``,
            ``universe``, ``signals``, ``orders``, ``kills_triggered``,
            ``drift``.
        """
        ts = now_ms()
        summary: dict[str, Any] = {
            "timestamp_ms": ts,
            "universe": [],
            "signals": [],
            "orders": [],
            "kill_triggered": False,
            "drift": {},
        }

        # 1. Kill switch check
        if self.kill_switch.is_triggered():
            logger.warning(
                "scheduler.run_once.kill_switch_active",
                reason=self.kill_switch.trigger_reason(),
            )
            summary["kill_triggered"] = True
            return summary

        # 2. Refresh universe
        try:
            self.universe = self._refresh_universe()
        except Exception as exc:
            logger.error("scheduler.universe_refresh_failed", error=str(exc))
            # Fall back to previous universe if available
            if not self.universe:
                return summary

        summary["universe"] = list(self.universe)

        # Refresh equity from user state (best-effort)
        self._update_equity()

        # 3. Process each symbol
        for symbol in self.universe:
            try:
                result = self._process_symbol(symbol)
                if result.get("signal_side") and result["signal_side"] != "flat":
                    summary["signals"].append(result)
                if result.get("order_id"):
                    summary["orders"].append(result)
            except Exception as exc:
                logger.error(
                    "scheduler.symbol_failed",
                    symbol=symbol,
                    error=str(exc),
                    exc_info=True,
                )

        # 4. Reconcile fills
        try:
            self._reconcile_fills()
        except Exception as exc:
            logger.error("scheduler.reconcile_failed", error=str(exc))

        # 5. Kill switch auto-check
        risk_state = self._build_risk_state()
        kill_triggered = self.kill_switch.check_conditions(risk_state, self.risk_config)
        summary["kill_triggered"] = kill_triggered

        if kill_triggered:
            logger.critical(
                "scheduler.kill_switch_auto_triggered",
                reason=self.kill_switch.trigger_reason(),
            )
            # Cancel all orders on kill switch
            try:
                self.broker.cancel_all()
            except Exception:
                logger.error("scheduler.cancel_all_on_kill_failed")

        # 6. Probation evaluation (canary mode)
        if self.probation is not None:
            try:
                if not self.probation.is_active():
                    # Probation window has elapsed -- evaluate and decide
                    evaluation = self.probation.evaluate()
                    summary["probation"] = evaluation
                    if evaluation.get("can_promote"):
                        logger.info(
                            "scheduler.probation.graduated",
                            trades=evaluation.get("trades_count"),
                            pnl=evaluation.get("pnl"),
                        )
                        # Promote: switch from canary to live
                        self.env = "live"
                        self.probation = None
                    else:
                        logger.warning(
                            "scheduler.probation.failed",
                            reasons=evaluation.get("reasons"),
                        )
                        # Rollback: trigger kill switch to halt trading
                        self.kill_switch.trigger(
                            reason=f"Probation failed: {evaluation.get('reasons')}"
                        )
                        summary["kill_triggered"] = True
                else:
                    # Feed daily PnL into probation tracker
                    self.probation.add_daily_pnl(self.reconciler.get_daily_pnl())
            except Exception as exc:
                logger.error("scheduler.probation_eval_failed", error=str(exc))

        # 7. Drift detection
        try:
            drift_result = self.drift_detector.check_drift()
            summary["drift"] = drift_result
            if drift_result.get("drifting"):
                logger.warning(
                    "scheduler.drift_detected",
                    severity=drift_result.get("severity"),
                    signals=drift_result.get("signals"),
                )
        except Exception as exc:
            logger.error("scheduler.drift_check_failed", error=str(exc))

        # 8. Update metrics
        metrics.set_gauge("equity_usd", self._equity)
        metrics.set_gauge("peak_equity_usd", self._peak_equity)
        metrics.set_gauge("open_positions_count", float(self.exposure_tracker.position_count()))
        metrics.set_gauge("total_notional_usd", self.exposure_tracker.total_notional())
        metrics.set_gauge("universe_size", float(len(self.universe)))
        metrics.inc_counter("scheduler_iterations_total")

        logger.info(
            "scheduler.run_once.complete",
            universe_size=len(self.universe),
            signals=len(summary["signals"]),
            orders=len(summary["orders"]),
            equity=round(self._equity, 2),
        )

        return summary

    # ------------------------------------------------------------------
    # Continuous loop
    # ------------------------------------------------------------------

    def run_loop(self, max_iterations: int | None = None) -> None:
        """Run the trading loop continuously.

        The loop executes :meth:`run_once` on each iteration, sleeps for
        ``poll_interval_sec``, and repeats until the kill switch is
        triggered, :meth:`shutdown` is called, or *max_iterations* is
        reached.

        Individual iteration failures are caught and logged so that a
        single bad tick does not crash the entire process.

        Parameters
        ----------
        max_iterations : int | None
            If set, exit the loop after this many iterations (useful for
            testing).  ``None`` means run indefinitely.
        """
        self.running = True
        iteration = 0

        logger.info(
            "scheduler.loop.start",
            poll_interval_sec=self.poll_interval_sec,
            max_iterations=max_iterations,
        )

        try:
            while self.running:
                # Check termination conditions before executing
                if self.kill_switch.is_triggered():
                    logger.warning(
                        "scheduler.loop.kill_switch_halt",
                        reason=self.kill_switch.trigger_reason(),
                    )
                    break

                if max_iterations is not None and iteration >= max_iterations:
                    logger.info(
                        "scheduler.loop.max_iterations_reached",
                        iterations=iteration,
                    )
                    break

                # Execute one iteration
                try:
                    summary = self.run_once()

                    # If the kill switch was triggered during this iteration, stop
                    if summary.get("kill_triggered"):
                        logger.warning("scheduler.loop.kill_during_iteration")
                        break

                except Exception as exc:
                    logger.error(
                        "scheduler.loop.iteration_error",
                        iteration=iteration,
                        error=str(exc),
                        exc_info=True,
                    )

                iteration += 1

                # Sleep between iterations (interruptible by setting
                # self.running = False from another thread)
                if self.running:
                    time.sleep(self.poll_interval_sec)

        finally:
            self.shutdown()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Gracefully shut down the scheduler.

        Sets ``running`` to ``False``, cancels all open orders, and logs
        the final state.
        """
        self.running = False

        logger.info("scheduler.shutdown.start")

        # Cancel all open orders
        try:
            cancelled = self.broker.cancel_all()
            logger.info("scheduler.shutdown.orders_cancelled", count=cancelled)
        except Exception as exc:
            logger.error(
                "scheduler.shutdown.cancel_failed",
                error=str(exc),
            )

        logger.info(
            "scheduler.shutdown.complete",
            equity=round(self._equity, 2),
            open_positions=self.exposure_tracker.position_count(),
        )

    # ------------------------------------------------------------------
    # Universe management
    # ------------------------------------------------------------------

    def _refresh_universe(self) -> list[str]:
        """Fetch asset contexts and select the top-N symbols by liquidity.

        If a static universe is configured (explicit symbol list), it is
        used directly.  Otherwise, symbols are ranked by ``day_ntl_vlm``
        and the top *N* that exceed the minimum volume threshold are
        selected.

        Returns
        -------
        list[str]
            Ordered list of symbols to trade this iteration.
        """
        if self._static_universe:
            return list(self._static_universe)

        ctx_df = fetch_asset_contexts(self.client)

        if ctx_df.empty:
            logger.warning("scheduler.universe.no_contexts")
            return list(self.universe) if self.universe else []

        # Filter by minimum volume
        if self._min_volume > 0:
            ctx_df = ctx_df[ctx_df["day_ntl_vlm"] >= self._min_volume]

        # Sort by daily notional volume descending and pick top N
        ctx_df = ctx_df.sort_values("day_ntl_vlm", ascending=False)
        symbols = ctx_df["name"].head(self._top_n).tolist()

        logger.info(
            "scheduler.universe.refreshed",
            count=len(symbols),
            top_symbols=symbols[:5],
        )
        return symbols

    # ------------------------------------------------------------------
    # Per-symbol processing
    # ------------------------------------------------------------------

    def _process_symbol(self, symbol: str) -> dict[str, Any]:
        """Run the full signal-to-order pipeline for a single symbol.

        Returns a dict summarising what happened (signal, approval,
        order result).
        """
        result: dict[str, Any] = {
            "symbol": symbol,
            "signal_side": "flat",
            "confidence": 0.0,
            "approved": False,
            "order_id": "",
        }

        # a. Fetch / update candles
        try:
            stream_candles_to_store(
                client=self.client,
                store=self.store,
                coins=[symbol],
                interval=self.primary_timeframe,
            )
        except Exception as exc:
            logger.warning(
                "scheduler.candle_fetch_failed",
                symbol=symbol,
                error=str(exc),
            )

        # Read candles from store
        candles = self.store.read_candles(
            symbol=symbol,
            timeframe=self.primary_timeframe,
            start_ms=0,
            end_ms=now_ms(),
        )

        if candles.empty or len(candles) < 50:
            logger.debug(
                "scheduler.insufficient_candles",
                symbol=symbol,
                count=len(candles) if not candles.empty else 0,
            )
            return result

        # b. Fetch L2 book snapshot
        book: dict | None = None
        try:
            book = snapshot_l2(self.client, symbol)
        except Exception as exc:
            logger.debug(
                "scheduler.l2_fetch_failed",
                symbol=symbol,
                error=str(exc),
            )

        # c. Compute features
        features = self._compute_features(candles)

        # d. Classify regime + hysteresis
        raw_regime, raw_confidence = self.regime_classifier.classify(features)
        effective_regime = self.hysteresis.update(raw_regime, raw_confidence)
        effective_confidence = self.hysteresis.current_confidence

        # e. Build MarketContext
        # Extract funding rate from the latest asset context if available
        funding_rate: float | None = None
        open_interest: float | None = None
        try:
            ctx_df = fetch_asset_contexts(self.client)
            if not ctx_df.empty:
                symbol_ctx = ctx_df[ctx_df["name"] == symbol]
                if not symbol_ctx.empty:
                    funding_rate = float(symbol_ctx.iloc[0]["funding"])
                    open_interest = float(symbol_ctx.iloc[0]["open_interest"])
        except Exception:
            pass

        ctx = MarketContext(
            symbol=symbol,
            timeframe=self.primary_timeframe,
            candles=candles,
            features=features,
            regime=effective_regime,
            regime_confidence=effective_confidence,
            book=book,
            funding_rate=funding_rate,
            open_interest=open_interest,
            account_equity=self._equity,
        )

        # f. Get signal from strategy
        try:
            signal = self.strategy.compute_signal(ctx)
        except Exception as exc:
            logger.error(
                "scheduler.strategy_error",
                symbol=symbol,
                error=str(exc),
            )
            return result

        result["signal_side"] = signal.side
        result["confidence"] = signal.confidence

        if signal.side == "flat":
            return result

        # g. Risk approval
        approval_side = _SIDE_TO_APPROVAL.get(signal.side, signal.side)
        if approval_side not in ("buy", "sell"):
            return result

        approval_signal = Signal(
            side=approval_side,
            entry=signal.entry,
            stop=signal.stop,
            take_profit=signal.take_profit,
            confidence=signal.confidence,
            metadata=signal.metadata,
        )

        risk_state = self._build_risk_state()
        volatility_pct = features.get("realized_vol")

        approval = approve_trade(
            signal=approval_signal,
            risk_state=risk_state,
            risk_config=self.risk_config,
            regime=effective_regime,
            regime_confidence=effective_confidence,
            volatility_pct=volatility_pct,
        )

        result["approved"] = approval.get("approved", False)

        if not approval.get("approved", False):
            logger.info(
                "scheduler.trade_rejected",
                symbol=symbol,
                side=approval_side,
                reasons=approval.get("reasons", []),
            )
            return result

        # h. Submit order via order manager
        trade_id = generate_trade_id()
        size_coins = approval.get("size_coins", 0.0)
        entry_px = signal.entry if signal.entry is not None else 0.0
        stop_px = signal.stop
        tp_px = signal.take_profit

        try:
            managed_order = self.order_manager.submit_entry(
                symbol=symbol,
                side=approval_side,
                size=size_coins,
                price=entry_px,
                trade_id=trade_id,
                stop_px=stop_px,
                tp_px=tp_px,
            )
            result["order_id"] = managed_order.order_id
            result["trade_id"] = trade_id

            logger.info(
                "scheduler.order_submitted",
                symbol=symbol,
                side=approval_side,
                size=size_coins,
                price=entry_px,
                trade_id=trade_id,
                order_state=managed_order.state.value,
            )

            # Record trade for probation tracking
            if self.probation is not None and self.probation.is_active():
                self.probation.add_trade(
                    {
                        "trade_id": trade_id,
                        "symbol": symbol,
                        "side": approval_side,
                        "size": size_coins,
                        "price": entry_px,
                        "pnl": 0.0,  # PnL unknown at entry time
                    }
                )

        except Exception as exc:
            logger.error(
                "scheduler.order_submit_failed",
                symbol=symbol,
                error=str(exc),
            )

        return result

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def _compute_features(self, candles: pd.DataFrame) -> dict:
        """Compute all features needed for regime classification and strategy.

        This mirrors the feature computation in the backtest engine,
        ensuring consistency between backtested and live behaviour.

        Parameters
        ----------
        candles : pd.DataFrame
            OHLCV data with columns ``open``, ``high``, ``low``, ``close``,
            ``volume``.

        Returns
        -------
        dict
            Feature name to latest scalar value.
        """
        close = candles["close"]
        high = candles["high"]
        low = candles["low"]
        open_ = candles["open"]
        volume = candles["volume"]

        features: dict = {}

        # ADX
        _adx = compute_adx(high, low, close, period=14)
        features["adx"] = self._last_valid(_adx)

        # RSI
        _rsi = compute_rsi(close, period=14)
        features["rsi"] = self._last_valid(_rsi)

        # ATR
        _atr = compute_atr(high, low, close, period=14)
        features["atr"] = self._last_valid(_atr)

        # SMA 50
        _sma50 = sma(close, 50)
        features["sma_50"] = self._last_valid(_sma50)

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = bollinger_bands(close, period=20, num_std=2.0)
        features["bb_upper"] = self._last_valid(bb_upper)
        features["bb_middle"] = self._last_valid(bb_middle)
        features["bb_lower"] = self._last_valid(bb_lower)

        # BB width percentile
        if len(close) >= 100:
            _bbwp = bb_width_percentile(close, period=20, num_std=2.0, lookback=100)
            features["bb_width_pct"] = self._last_valid(_bbwp)
        else:
            features["bb_width_pct"] = 0.5

        # MA slope
        _slope = compute_ma_slope(close, period=20, lookback=5)
        features["ma_slope"] = self._last_valid(_slope)

        # Realized vol
        _rvol = compute_realized_vol(close, period=20)
        features["realized_vol"] = self._last_valid(_rvol)

        # Vol ratio (short-term / long-term realized vol)
        if len(close) >= 60:
            _rvol_long = compute_realized_vol(close, period=60)
            rvol_short = features["realized_vol"]
            rvol_long = self._last_valid(_rvol_long)
            if rvol_long is not None and rvol_long > 1e-12 and rvol_short is not None:
                features["vol_ratio"] = rvol_short / rvol_long
            else:
                features["vol_ratio"] = 1.0
        else:
            features["vol_ratio"] = 1.0

        # Hurst exponent
        _hurst = hurst_exponent(close, max_lag=20)
        features["hurst"] = self._last_valid(_hurst)

        # Volume SMA
        _vsma = compute_volume_sma(volume, period=20)
        features["volume_sma"] = self._last_valid(_vsma)

        # Wick ratio
        _wick = compute_wick_ratio(open_, high, low, close)
        features["wick_ratio"] = self._last_valid(_wick)

        # Funding rate features (from stored funding data if available)
        try:
            funding_df = self.store.read_funding(
                symbol=candles.get("symbol", ""),  # type: ignore[arg-type]
                start_ms=0,
                end_ms=now_ms(),
            )
            if not funding_df.empty and "funding_rate" in funding_df.columns:
                fr_series = funding_df["funding_rate"]
                _fz = funding_rate_zscore(fr_series, period=min(100, len(fr_series)))
                features["funding_zscore"] = self._last_valid(_fz)
                _fp = funding_rate_percentile(fr_series, period=min(100, len(fr_series)))
                features["funding_percentile"] = self._last_valid(_fp)
            else:
                features["funding_zscore"] = 0.0
                features["funding_percentile"] = 0.5
        except Exception:
            features["funding_zscore"] = 0.0
            features["funding_percentile"] = 0.5

        return features

    @staticmethod
    def _last_valid(series: pd.Series) -> float | None:
        """Return the last non-NaN value from *series*, or ``None``."""
        if series.empty:
            return None
        val = series.iloc[-1]
        if pd.isna(val):
            valid = series.dropna()
            if valid.empty:
                return None
            return float(valid.iloc[-1])
        return float(val)

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    def _reconcile_fills(self) -> None:
        """Fetch recent fills from the broker and process them through
        the reconciler."""
        fills = self.broker.get_fills()
        for fill in fills:
            self.reconciler.process_fill(fill)

    # ------------------------------------------------------------------
    # Equity / risk state
    # ------------------------------------------------------------------

    def _update_equity(self) -> None:
        """Fetch user state from the exchange and update the equity tracker.

        Falls back to the last known equity on failure (best-effort).
        """
        try:
            user_state = fetch_user_state(self.client)
            account_value = user_state.get("account_value", 0.0)
            if account_value > 0:
                self._equity = account_value
                self._peak_equity = max(self._peak_equity, self._equity)

            # Sync positions with the exchange
            positions = user_state.get("positions", [])
            if positions:
                self.reconciler.sync_positions(positions)
        except Exception as exc:
            logger.warning(
                "scheduler.equity_update_failed",
                error=str(exc),
            )

    def _build_risk_state(self) -> RiskState:
        """Construct the current :class:`RiskState` from live data."""
        return self.exposure_tracker.to_risk_state(
            equity=self._equity,
            peak_equity=self._peak_equity,
            daily_pnl=self.reconciler.get_daily_pnl(),
            weekly_pnl=self.reconciler.get_weekly_pnl(),
        )
