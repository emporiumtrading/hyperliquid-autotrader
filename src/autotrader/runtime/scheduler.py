"""Main trading loop scheduler.

Orchestrates data collection, feature computation, regime classification,
strategy signal generation, risk approval, order execution, reconciliation,
drift detection, and kill switch monitoring in a single polling loop.

Implements:
- Multi-timeframe analysis (regime on 1h/4h, signals on 15m, execution on 1m)
- Spread/depth-based universe filtering
- No-trade windows (stale data, extreme vol, degraded rate limits)
- Automated drift response (reduce exposure, halt on critical)
- Order chase / re-pricing logic for unfilled limit orders
- Lot size / tick size rounding from exchange metadata
- Latency metrics for the trading loop
"""

from __future__ import annotations

import math
import time
from typing import Any

import pandas as pd
import structlog

from autotrader.data.collectors.candles import stream_candles_to_store
from autotrader.data.collectors.funding_oi import fetch_asset_contexts
from autotrader.data.collectors.l2book import snapshot_l2
from autotrader.data.collectors.user_state import fetch_user_state
from autotrader.data.transforms.resample import resample_ohlcv
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
from autotrader.hl.rate_limiter import available as rate_limit_available
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

# Default multi-TF roles
_DEFAULT_REGIME_TIMEFRAMES = ["1h"]
_DEFAULT_SIGNAL_TIMEFRAME = "15m"


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

        # ---- Order chase config ----
        self._chase_timeout_sec: float = float(exec_cfg.get("chase_timeout_sec", 8.0))
        self._chase_max_retries: int = int(exec_cfg.get("chase_max_retries", 3))
        self._max_slippage_bps: float = float(exec_cfg.get("max_slippage_bps", 5.0))

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

        # ---- Multi-timeframe config ----
        tf_cfg = config.get("timeframes", {})
        if isinstance(tf_cfg, dict):
            self._regime_timeframes: list[str] = list(
                tf_cfg.get("regime", _DEFAULT_REGIME_TIMEFRAMES)
            )
            self._signal_timeframe: str = tf_cfg.get("signal", _DEFAULT_SIGNAL_TIMEFRAME)
            self._execution_timeframe: str = tf_cfg.get("execution", "1m")
        elif isinstance(tf_cfg, list):
            # Legacy: first item is signal TF
            self._regime_timeframes = _DEFAULT_REGIME_TIMEFRAMES
            self._signal_timeframe = tf_cfg[0] if tf_cfg else _DEFAULT_SIGNAL_TIMEFRAME
            self._execution_timeframe = "1m"
        else:
            self._regime_timeframes = _DEFAULT_REGIME_TIMEFRAMES
            self._signal_timeframe = str(tf_cfg) if tf_cfg else _DEFAULT_SIGNAL_TIMEFRAME
            self._execution_timeframe = "1m"

        # Collect all unique timeframes we need to fetch
        self._all_timeframes: list[str] = list(
            dict.fromkeys(
                [self._signal_timeframe] + self._regime_timeframes + [self._execution_timeframe]
            )
        )
        self.primary_timeframe: str = self._signal_timeframe

        # ---- Universe config ----
        universe_cfg = config.get("universe", {})
        if isinstance(universe_cfg, list):
            self._static_universe: list[str] = list(universe_cfg)
            self._top_n: int = len(universe_cfg)
            self._min_volume: float = 0.0
            self._max_spread_bps: float = 999.0
            self._min_depth_usd: float = 0.0
        elif isinstance(universe_cfg, dict):
            self._static_universe = list(universe_cfg.get("symbols", []))
            self._top_n = int(universe_cfg.get("top_n", 20))
            self._min_volume = float(universe_cfg.get("min_volume", 0.0))
            self._max_spread_bps = float(universe_cfg.get("max_spread_bps", 3.0))
            self._min_depth_usd = float(universe_cfg.get("min_depth_usd", 200_000.0))
        else:
            self._static_universe = []
            self._top_n = 20
            self._min_volume = 0.0
            self._max_spread_bps = 3.0
            self._min_depth_usd = 200_000.0

        self.universe: list[str] = []

        # ---- No-trade window config ----
        ntw_cfg = config.get("no_trade_windows", {})
        self._max_data_age_ms: int = int(
            ntw_cfg.get("max_data_age_ms", 5 * 60 * 1000)
        )  # 5 min
        self._extreme_vol_threshold: float = float(
            ntw_cfg.get("extreme_vol_threshold", 2.0)
        )  # 2x normal vol ratio
        self._min_rate_limit_tokens: float = float(
            ntw_cfg.get("min_rate_limit_tokens", 100.0)
        )

        # ---- Asset metadata (sz_decimals, max_leverage) ----
        self._asset_meta: dict[str, dict] = {}  # symbol -> {sz_decimals, max_leverage}
        self._meta_loaded: bool = False

        # ---- Loop control ----
        self.poll_interval_sec: float = float(
            exec_cfg.get("poll_interval_sec", config.get("poll_interval_sec", 60.0))
        )
        self.running: bool = False

        # ---- Equity tracking (used for RiskState) ----
        self._equity: float = float(config.get("initial_equity", 10_000.0))
        self._peak_equity: float = self._equity

        # ---- Drift response state ----
        self._drift_reduced: bool = False

        logger.info(
            "scheduler.init",
            env=self.env,
            mode=mode,
            regime_timeframes=self._regime_timeframes,
            signal_timeframe=self._signal_timeframe,
            execution_timeframe=self._execution_timeframe,
            poll_interval_sec=self.poll_interval_sec,
        )

    # ------------------------------------------------------------------
    # Asset metadata (lot size / tick size)
    # ------------------------------------------------------------------

    def _load_asset_meta(self) -> None:
        """Load exchange metadata for sz_decimals and max_leverage."""
        if self._meta_loaded:
            return
        try:
            meta = self.client.get_meta()
            universe = meta.get("universe", [])
            for asset in universe:
                name = asset.get("name", "")
                if name:
                    self._asset_meta[name] = {
                        "sz_decimals": int(asset.get("szDecimals", 3)),
                        "max_leverage": int(asset.get("maxLeverage", 50)),
                    }
            self._meta_loaded = True
            logger.info("scheduler.asset_meta_loaded", count=len(self._asset_meta))
        except Exception as exc:
            logger.warning("scheduler.asset_meta_load_failed", error=str(exc))

    def _round_size(self, symbol: str, size: float) -> float:
        """Round order size to the exchange's sz_decimals for the asset."""
        meta = self._asset_meta.get(symbol, {})
        sz_decimals = meta.get("sz_decimals", 3)
        factor = 10**sz_decimals
        return math.floor(size * factor) / factor

    def _round_price(self, symbol: str, price: float) -> float:
        """Round price to a reasonable precision (5 significant figures)."""
        if price <= 0:
            return 0.0
        # Hyperliquid uses 5 significant figures for prices
        magnitude = math.floor(math.log10(abs(price)))
        decimals = max(0, 4 - magnitude)
        factor = 10**decimals
        return round(price * factor) / factor

    # ------------------------------------------------------------------
    # No-trade window checks
    # ------------------------------------------------------------------

    def _check_no_trade_window(
        self, symbol: str, candles: pd.DataFrame, features: dict
    ) -> str | None:
        """Check if a no-trade window is active for this symbol.

        Returns a reason string if trading should be skipped, or None.
        """
        # 1. Stale data check
        if not candles.empty and "timestamp_ms" in candles.columns:
            last_ts = int(candles["timestamp_ms"].iloc[-1])
            age_ms = now_ms() - last_ts
            if age_ms > self._max_data_age_ms:
                return f"stale_data: last candle {age_ms / 1000:.0f}s old"

        # 2. Extreme volatility check
        vol_ratio = features.get("vol_ratio")
        if vol_ratio is not None and vol_ratio > self._extreme_vol_threshold:
            return f"extreme_vol: vol_ratio={vol_ratio:.2f} > {self._extreme_vol_threshold}"

        # 3. Rate limit degradation
        try:
            tokens = rate_limit_available()
            if tokens < self._min_rate_limit_tokens:
                return f"rate_limit_low: {tokens:.0f} tokens remaining"
        except Exception:
            pass

        return None

    # ------------------------------------------------------------------
    # Single iteration
    # ------------------------------------------------------------------

    def run_once(self) -> dict:
        """Execute a single iteration of the trading loop."""
        loop_start = time.monotonic()
        ts = now_ms()
        summary: dict[str, Any] = {
            "timestamp_ms": ts,
            "universe": [],
            "signals": [],
            "orders": [],
            "kill_triggered": False,
            "drift": {},
            "no_trade_windows": [],
        }

        # 1. Kill switch check
        if self.kill_switch.is_triggered():
            logger.warning(
                "scheduler.run_once.kill_switch_active",
                reason=self.kill_switch.trigger_reason(),
            )
            summary["kill_triggered"] = True
            return summary

        # Load asset metadata once
        self._load_asset_meta()

        # 2. Refresh universe (with spread/depth filters)
        try:
            self.universe = self._refresh_universe()
        except Exception as exc:
            logger.error("scheduler.universe_refresh_failed", error=str(exc))
            if not self.universe:
                return summary

        summary["universe"] = list(self.universe)

        # Refresh equity from user state (best-effort)
        self._update_equity()

        # Cache asset contexts for the iteration (avoid re-fetching per symbol)
        cached_ctx_df: pd.DataFrame | None = None
        try:
            cached_ctx_df = fetch_asset_contexts(self.client)
        except Exception:
            pass

        # 3. Process each symbol
        for symbol in self.universe:
            try:
                result = self._process_symbol(symbol, cached_ctx_df)
                if result.get("no_trade_reason"):
                    summary["no_trade_windows"].append(
                        {"symbol": symbol, "reason": result["no_trade_reason"]}
                    )
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

        # 5. Chase unfilled orders
        try:
            self._chase_unfilled_orders()
        except Exception as exc:
            logger.error("scheduler.chase_failed", error=str(exc))

        # 6. Kill switch auto-check
        risk_state = self._build_risk_state()
        kill_triggered = self.kill_switch.check_conditions(risk_state, self.risk_config)
        summary["kill_triggered"] = kill_triggered

        if kill_triggered:
            logger.critical(
                "scheduler.kill_switch_auto_triggered",
                reason=self.kill_switch.trigger_reason(),
            )
            try:
                self.broker.cancel_all()
            except Exception:
                logger.error("scheduler.cancel_all_on_kill_failed")

        # 7. Probation evaluation (canary mode)
        if self.probation is not None:
            try:
                if not self.probation.is_active():
                    evaluation = self.probation.evaluate()
                    summary["probation"] = evaluation
                    if evaluation.get("can_promote"):
                        logger.info(
                            "scheduler.probation.graduated",
                            trades=evaluation.get("trades_count"),
                            pnl=evaluation.get("pnl"),
                        )
                        self.env = "live"
                        self.probation = None
                    else:
                        logger.warning(
                            "scheduler.probation.failed",
                            reasons=evaluation.get("reasons"),
                        )
                        self.kill_switch.trigger(
                            reason=f"Probation failed: {evaluation.get('reasons')}"
                        )
                        summary["kill_triggered"] = True
                else:
                    self.probation.add_daily_pnl(self.reconciler.get_daily_pnl())
            except Exception as exc:
                logger.error("scheduler.probation_eval_failed", error=str(exc))

        # 8. Drift detection + automated response
        try:
            drift_result = self.drift_detector.check_drift()
            summary["drift"] = drift_result
            if drift_result.get("drifting"):
                self._handle_drift(drift_result)
        except Exception as exc:
            logger.error("scheduler.drift_check_failed", error=str(exc))

        # 9. Update metrics
        loop_elapsed_ms = (time.monotonic() - loop_start) * 1000
        metrics.set_gauge("equity_usd", self._equity)
        metrics.set_gauge("peak_equity_usd", self._peak_equity)
        metrics.set_gauge("open_positions_count", float(self.exposure_tracker.position_count()))
        metrics.set_gauge("total_notional_usd", self.exposure_tracker.total_notional())
        metrics.set_gauge("universe_size", float(len(self.universe)))
        metrics.observe("loop_latency_ms", loop_elapsed_ms)
        metrics.inc_counter("scheduler_iterations_total")

        logger.info(
            "scheduler.run_once.complete",
            universe_size=len(self.universe),
            signals=len(summary["signals"]),
            orders=len(summary["orders"]),
            equity=round(self._equity, 2),
            loop_ms=round(loop_elapsed_ms, 1),
        )

        return summary

    # ------------------------------------------------------------------
    # Continuous loop
    # ------------------------------------------------------------------

    def run_loop(self, max_iterations: int | None = None) -> None:
        """Run the trading loop continuously."""
        self.running = True
        iteration = 0

        logger.info(
            "scheduler.loop.start",
            poll_interval_sec=self.poll_interval_sec,
            max_iterations=max_iterations,
        )

        try:
            while self.running:
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

                try:
                    summary = self.run_once()
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

                if self.running:
                    time.sleep(self.poll_interval_sec)

        finally:
            self.shutdown()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Gracefully shut down the scheduler."""
        self.running = False
        logger.info("scheduler.shutdown.start")

        try:
            cancelled = self.broker.cancel_all()
            logger.info("scheduler.shutdown.orders_cancelled", count=cancelled)
        except Exception as exc:
            logger.error("scheduler.shutdown.cancel_failed", error=str(exc))

        logger.info(
            "scheduler.shutdown.complete",
            equity=round(self._equity, 2),
            open_positions=self.exposure_tracker.position_count(),
        )

    # ------------------------------------------------------------------
    # Universe management (with spread / depth filters)
    # ------------------------------------------------------------------

    def _refresh_universe(self) -> list[str]:
        """Fetch asset contexts and select top-N symbols by liquidity.

        Applies volume, spread, and depth filters per PRD section 6.1.
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

        # Sort by daily notional volume descending and pick top N candidates
        ctx_df = ctx_df.sort_values("day_ntl_vlm", ascending=False)
        candidates = ctx_df["name"].head(self._top_n * 2).tolist()  # overfetch for filtering

        # Apply spread and depth filters
        filtered: list[str] = []
        for symbol in candidates:
            if len(filtered) >= self._top_n:
                break
            try:
                book = snapshot_l2(self.client, symbol)
                spread = book.get("spread_bps", 999.0)
                depth = book.get("bid_depth_usd", 0.0) + book.get("ask_depth_usd", 0.0)
                if spread > self._max_spread_bps:
                    logger.debug(
                        "scheduler.universe.spread_filter",
                        symbol=symbol,
                        spread_bps=spread,
                    )
                    continue
                if depth < self._min_depth_usd:
                    logger.debug(
                        "scheduler.universe.depth_filter",
                        symbol=symbol,
                        depth_usd=depth,
                    )
                    continue
                filtered.append(symbol)
            except Exception:
                # If we can't check L2, still include based on volume ranking
                filtered.append(symbol)

        logger.info(
            "scheduler.universe.refreshed",
            count=len(filtered),
            top_symbols=filtered[:5],
        )
        return filtered

    # ------------------------------------------------------------------
    # Per-symbol processing
    # ------------------------------------------------------------------

    def _process_symbol(
        self, symbol: str, cached_ctx_df: pd.DataFrame | None
    ) -> dict[str, Any]:
        """Run the full signal-to-order pipeline for a single symbol."""
        result: dict[str, Any] = {
            "symbol": symbol,
            "signal_side": "flat",
            "confidence": 0.0,
            "approved": False,
            "order_id": "",
            "no_trade_reason": "",
        }

        # a. Fetch / update candles for all needed timeframes
        for tf in self._all_timeframes:
            try:
                stream_candles_to_store(
                    client=self.client,
                    store=self.store,
                    coins=[symbol],
                    interval=tf,
                )
            except Exception as exc:
                logger.warning(
                    "scheduler.candle_fetch_failed",
                    symbol=symbol,
                    timeframe=tf,
                    error=str(exc),
                )

        # Read signal-timeframe candles from store
        candles = self.store.read_candles(
            symbol=symbol,
            timeframe=self._signal_timeframe,
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

        # c. Compute features on signal timeframe
        features = self._compute_features(candles)

        # c2. Multi-TF regime features: compute from higher TF candles
        regime_features = self._compute_regime_features(symbol, features)

        # d. No-trade window check
        ntw_reason = self._check_no_trade_window(symbol, candles, features)
        if ntw_reason:
            result["no_trade_reason"] = ntw_reason
            logger.info("scheduler.no_trade_window", symbol=symbol, reason=ntw_reason)
            metrics.inc_counter("no_trade_windows_total")
            return result

        # e. Classify regime with hysteresis (using higher-TF features)
        raw_regime, raw_confidence = self.regime_classifier.classify(regime_features)
        effective_regime = self.hysteresis.update(raw_regime, raw_confidence)
        effective_confidence = self.hysteresis.current_confidence

        # f. Build MarketContext
        funding_rate: float | None = None
        open_interest: float | None = None
        if cached_ctx_df is not None and not cached_ctx_df.empty:
            symbol_ctx = cached_ctx_df[cached_ctx_df["name"] == symbol]
            if not symbol_ctx.empty:
                funding_rate = float(symbol_ctx.iloc[0]["funding"])
                open_interest = float(symbol_ctx.iloc[0]["open_interest"])

        ctx = MarketContext(
            symbol=symbol,
            timeframe=self._signal_timeframe,
            candles=candles,
            features=features,
            regime=effective_regime,
            regime_confidence=effective_confidence,
            book=book,
            funding_rate=funding_rate,
            open_interest=open_interest,
            account_equity=self._equity,
        )

        # g. Get signal from strategy
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

        # h. Risk approval
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

        # i. Apply lot size / tick size rounding
        trade_id = generate_trade_id()
        raw_size = approval.get("size_coins", 0.0)
        size_coins = self._round_size(symbol, raw_size)
        entry_px = self._round_price(
            symbol, signal.entry if signal.entry is not None else 0.0
        )
        stop_px = self._round_price(symbol, signal.stop) if signal.stop else None
        tp_px = self._round_price(symbol, signal.take_profit) if signal.take_profit else None

        if size_coins <= 0:
            logger.info(
                "scheduler.size_rounded_to_zero",
                symbol=symbol,
                raw_size=raw_size,
            )
            return result

        # j. Submit order via order manager
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
                        "pnl": 0.0,
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
    # Multi-timeframe regime features
    # ------------------------------------------------------------------

    def _compute_regime_features(self, symbol: str, signal_features: dict) -> dict:
        """Compute regime features from higher timeframe candles.

        Falls back to signal-timeframe features if higher-TF data is
        unavailable or insufficient.
        """
        regime_features = dict(signal_features)  # start with signal TF features

        for tf in self._regime_timeframes:
            try:
                htf_candles = self.store.read_candles(
                    symbol=symbol,
                    timeframe=tf,
                    start_ms=0,
                    end_ms=now_ms(),
                )

                if htf_candles.empty or len(htf_candles) < 30:
                    continue

                close = htf_candles["close"]
                high = htf_candles["high"]
                low = htf_candles["low"]

                # Override key regime features with higher-TF values
                _adx = compute_adx(high, low, close, period=14)
                htf_adx = self._last_valid(_adx)
                if htf_adx is not None:
                    regime_features[f"adx_{tf}"] = htf_adx
                    regime_features["adx"] = htf_adx  # override signal-TF

                _hurst = hurst_exponent(close, max_lag=20)
                htf_hurst = self._last_valid(_hurst)
                if htf_hurst is not None:
                    regime_features[f"hurst_{tf}"] = htf_hurst
                    regime_features["hurst"] = htf_hurst

                _rvol = compute_realized_vol(close, period=20)
                htf_rvol = self._last_valid(_rvol)
                if htf_rvol is not None:
                    regime_features[f"realized_vol_{tf}"] = htf_rvol

                _slope = compute_ma_slope(close, period=20, lookback=5)
                htf_slope = self._last_valid(_slope)
                if htf_slope is not None:
                    regime_features[f"ma_slope_{tf}"] = htf_slope

            except Exception as exc:
                logger.debug(
                    "scheduler.regime_tf_failed",
                    symbol=symbol,
                    timeframe=tf,
                    error=str(exc),
                )

        return regime_features

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def _compute_features(self, candles: pd.DataFrame) -> dict:
        """Compute all features needed for regime classification and strategy.

        This mirrors the feature computation in the backtest engine,
        ensuring consistency between backtested and live behaviour.
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
    # Order chase / re-pricing
    # ------------------------------------------------------------------

    def _chase_unfilled_orders(self) -> None:
        """Check for stale unfilled limit orders and re-price them.

        If an order has been pending longer than ``chase_timeout_sec`` and
        hasn't exceeded ``chase_max_retries``, cancel and re-submit at
        the current market price (with slippage cap).
        """
        active_orders = self.order_manager.get_active_orders()
        current_ms = now_ms()
        chase_timeout_ms = int(self._chase_timeout_sec * 1000)

        for order in active_orders:
            if order.state.value != "submitted":
                continue

            age_ms = current_ms - order.timestamp_ms
            if age_ms < chase_timeout_ms:
                continue

            retries = order.metadata.get("chase_retries", 0) if hasattr(order, "metadata") else 0
            if retries >= self._chase_max_retries:
                logger.info(
                    "scheduler.chase.max_retries",
                    order_id=order.order_id,
                    retries=retries,
                )
                # Cancel the order instead of chasing further
                try:
                    self.order_manager.cancel_trade_orders(order.trade_id)
                except Exception:
                    pass
                continue

            # Re-price: fetch current book and adjust price
            symbol = order.symbol
            try:
                book = snapshot_l2(self.client, symbol)
                mid_px = book.get("mid_px", 0.0)
                if mid_px <= 0:
                    continue

                # Check slippage from original price
                orig_px = order.price
                if orig_px > 0:
                    slippage_bps = abs(mid_px - orig_px) / orig_px * 10_000
                    if slippage_bps > self._max_slippage_bps:
                        logger.info(
                            "scheduler.chase.slippage_exceeded",
                            order_id=order.order_id,
                            slippage_bps=round(slippage_bps, 2),
                        )
                        self.order_manager.cancel_trade_orders(order.trade_id)
                        continue

                # Cancel and re-submit at new price
                self.broker.cancel_order(symbol, order.order_id)
                new_px = self._round_price(symbol, mid_px)

                new_result = self.broker.place_order(
                    symbol=symbol,
                    side=order.side,
                    size=order.remaining_size,
                    price=new_px,
                    order_type="limit",
                    tif="Ioc",  # IOC for chase orders to avoid stacking
                )

                metrics.inc_counter("orders_chased_total")
                logger.info(
                    "scheduler.chase.repriced",
                    order_id=order.order_id,
                    old_px=orig_px,
                    new_px=new_px,
                    new_status=new_result.status,
                )

            except Exception as exc:
                logger.warning(
                    "scheduler.chase.failed",
                    order_id=order.order_id,
                    error=str(exc),
                )

    # ------------------------------------------------------------------
    # Drift response
    # ------------------------------------------------------------------

    def _handle_drift(self, drift_result: dict) -> None:
        """Take automated action based on drift severity.

        - warning: reduce exposure by cancelling open orders for new entries
        - critical: halt trading via kill switch
        """
        severity = drift_result.get("severity", "none")
        action = drift_result.get("action", "continue")

        logger.warning(
            "scheduler.drift_detected",
            severity=severity,
            action=action,
            signals=drift_result.get("signals"),
        )

        if severity == "critical" or action == "halt_trading_and_review":
            logger.critical("scheduler.drift_critical_halt")
            self.kill_switch.trigger(
                reason=f"Critical drift detected: {drift_result.get('signals')}"
            )
        elif severity == "warning" or action == "reduce_exposure":
            if not self._drift_reduced:
                logger.warning("scheduler.drift_reducing_exposure")
                # Cancel all pending entry orders to stop new positions
                try:
                    self.broker.cancel_all()
                    metrics.inc_counter("drift_exposure_reductions_total")
                except Exception:
                    pass
                self._drift_reduced = True
        else:
            # Drift cleared -- allow trading again
            self._drift_reduced = False

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    def _reconcile_fills(self) -> None:
        """Fetch recent fills from the broker and process them through
        the reconciler."""
        fills = self.broker.get_fills()
        for fill in fills:
            self.reconciler.process_fill(fill)

            # Fill quality metrics
            fill_px = float(fill.get("price", fill.get("px", 0.0)))
            if fill_px > 0:
                metrics.observe("fill_price", fill_px)
            fill_fee = float(fill.get("fee", 0.0))
            if fill_fee > 0:
                metrics.observe("fill_fee_usd", fill_fee)

    # ------------------------------------------------------------------
    # Equity / risk state
    # ------------------------------------------------------------------

    def _update_equity(self) -> None:
        """Fetch user state from the exchange and update the equity tracker."""
        try:
            user_state = fetch_user_state(self.client)
            account_value = user_state.get("account_value", 0.0)
            if account_value > 0:
                self._equity = account_value
                self._peak_equity = max(self._peak_equity, self._equity)

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
