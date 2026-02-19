"""Event-driven backtest engine.

Processes candle data bar-by-bar, computing features, classifying regimes,
generating strategy signals, and simulating trade execution with realistic
costs.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import pandas as pd
import structlog

from autotrader.backtest.cost_model import CostConfig, CostModel
from autotrader.backtest.metrics import compute_metrics
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
    macd as compute_macd,
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
from autotrader.hl.types import Signal
from autotrader.regimes.classifier import RegimeClassifier
from autotrader.regimes.hysteresis import HysteresisFilter
from autotrader.risk.approvals import approve_trade
from autotrader.risk.constraints import RiskConfig, RiskState, load_risk_config
from autotrader.strategies.base import BaseStrategy, MarketContext
from autotrader.strategies.ensemble import EnsembleStrategy
from autotrader.utils.ids import generate_run_id, generate_trade_id

logger = structlog.get_logger(__name__)

# Map strategy signal sides to approve_trade expected sides
_SIDE_TO_APPROVAL = {"long": "buy", "short": "sell"}
_SIDE_TO_LONG = {"long": True, "buy": True, "short": False, "sell": False}


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------


@dataclass
class TradeRecord:
    """A single trade lifecycle record (open -> close)."""

    trade_id: str = ""
    symbol: str = ""
    side: str = ""
    entry_time: int = 0  # ms
    exit_time: int = 0  # ms
    entry_px: float = 0.0
    exit_px: float = 0.0
    size: float = 0.0
    notional: float = 0.0
    leverage: float = 1.0
    pnl: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    funding: float = 0.0
    holding_bars: int = 0
    exit_reason: str = ""
    stop: float = 0.0
    take_profit: float = 0.0


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """Bar-by-bar event-driven backtest engine.

    Parameters
    ----------
    config : dict
        Backtest configuration.  Relevant keys:

        - ``initial_equity`` (float): Starting equity in USD (default 10 000).
        - ``warmup_bars`` (int): Bars to skip for indicator warm-up (default 50).
        - ``funding_rate`` (float): Assumed constant funding rate if not in
          candle data (default 0.0001).
        - ``bar_hours`` (float): Duration of one bar in hours (used for
          funding cost; default 0.25 for 15-min bars).
        - ``risk``: Sub-dict consumed by :func:`load_risk_config`.
        - ``regime``: Sub-dict for :class:`RegimeClassifier`.
        - ``hysteresis``: Sub-dict with ``min_bars`` and ``min_confidence``.
        - ``cost``: Sub-dict for :class:`CostConfig`.

    strategy : BaseStrategy | EnsembleStrategy | None
        Trading strategy.  Defaults to an :class:`EnsembleStrategy`.
    cost_model : CostModel | None
        Cost model.  Built from config if not supplied.
    """

    def __init__(
        self,
        config: dict,
        strategy: BaseStrategy | EnsembleStrategy | None = None,
        cost_model: CostModel | None = None,
    ) -> None:
        self.config = config

        # Strategy
        strategy_cfg = config.get("strategy", None)
        self.strategy: BaseStrategy | EnsembleStrategy = (
            strategy if strategy is not None else EnsembleStrategy(config=strategy_cfg)
        )

        # Cost model
        if cost_model is not None:
            self.cost_model = cost_model
        else:
            cost_cfg_dict = config.get("cost", {})
            valid_cost_fields = {f.name for f in CostConfig.__dataclass_fields__.values()}
            filtered = {k: v for k, v in cost_cfg_dict.items() if k in valid_cost_fields}
            self.cost_model = CostModel(CostConfig(**filtered) if filtered else None)

        # Risk
        self.risk_config: RiskConfig = load_risk_config(config)

        # Regime detection
        regime_cfg = config.get("regime", None)
        self.regime_classifier = RegimeClassifier(regime_cfg)

        hyst_cfg = config.get("hysteresis", {})
        self.hysteresis_filter = HysteresisFilter(
            min_bars=int(hyst_cfg.get("min_bars", 3)),
            min_confidence=float(hyst_cfg.get("min_confidence", 0.4)),
        )

        # Backtest state (reset on each run)
        self._initial_equity: float = float(config.get("initial_equity", 10_000.0))
        self._warmup_bars: int = int(config.get("warmup_bars", 50))
        self._funding_rate: float = float(config.get("funding_rate", 0.0001))
        self._bar_hours: float = float(config.get("bar_hours", 0.25))

        # Per-asset metadata for lot/tick rounding (PRD §11.2.5)
        # Keys: symbol -> {"sz_decimals": int, "max_leverage": int}
        self._asset_meta: dict[str, dict] = config.get("asset_meta", {})

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        candles: pd.DataFrame,
        symbol: str = "ETH",
        timeframe: str = "15m",
    ) -> dict:
        """Execute the backtest over *candles* and return results.

        Parameters
        ----------
        candles : pd.DataFrame
            Must contain columns: ``timestamp_ms``, ``open``, ``high``,
            ``low``, ``close``, ``volume``.
        symbol : str
            Trading pair symbol.
        timeframe : str
            Candle interval string.

        Returns
        -------
        dict
            ``{run_id, symbol, timeframe, trades, equity_curve, metrics,
            regime_history, config}``.
        """
        run_id = generate_run_id()
        logger.info(
            "backtest_start",
            run_id=run_id,
            symbol=symbol,
            timeframe=timeframe,
            bars=len(candles),
        )

        # Reset state
        equity = self._initial_equity
        peak_equity = equity
        open_trades: dict[str, TradeRecord] = {}
        closed_trades: list[TradeRecord] = []
        equity_history: list[tuple[int, float]] = []
        regime_history: list[tuple[int, str, float]] = []

        self.hysteresis_filter.reset()

        # Risk state
        risk_state = RiskState(
            equity=equity,
            peak_equity=peak_equity,
            daily_pnl=0.0,
            weekly_pnl=0.0,
            open_positions=0,
            total_notional=0.0,
        )

        candles = candles.reset_index(drop=True)
        n_bars = len(candles)

        if n_bars < self._warmup_bars + 1:
            logger.warning(
                "insufficient_bars",
                n_bars=n_bars,
                warmup=self._warmup_bars,
            )
            return self._build_result(run_id, symbol, timeframe, [], [], [], {})

        # Bar loop -- start after warmup
        for i in range(self._warmup_bars, n_bars):
            bar = candles.iloc[i]
            ts = int(bar["timestamp_ms"]) if "timestamp_ms" in candles.columns else i

            # ----- 1. Compute features -----
            candles_slice = candles.iloc[: i + 1]
            features = self._compute_features(candles_slice)

            # ----- 2. Classify regime -----
            raw_regime, raw_confidence = self.regime_classifier.classify(features)

            # ----- 3. Apply hysteresis -----
            effective_regime = self.hysteresis_filter.update(raw_regime, raw_confidence)
            effective_confidence = self.hysteresis_filter.current_confidence
            regime_history.append((ts, effective_regime, effective_confidence))

            # ----- 4-5. Check exits -----
            self._check_exits(bar, ts, open_trades, closed_trades, equity)

            # Recalculate equity after exits
            equity = self._initial_equity
            for ct in closed_trades:
                equity += ct.pnl - ct.fees - ct.slippage - ct.funding
            # Add unrealised PnL from open trades
            for ot in open_trades.values():
                close_px = float(bar["close"])
                if ot.side in ("long", "buy"):
                    unrealised = (close_px - ot.entry_px) * ot.size
                else:
                    unrealised = (ot.entry_px - close_px) * ot.size
                equity += unrealised
            peak_equity = max(peak_equity, equity)

            # Update risk state
            risk_state.equity = equity
            risk_state.peak_equity = peak_equity
            risk_state.open_positions = len(open_trades)
            risk_state.total_notional = sum(t.notional for t in open_trades.values())

            # ----- 6. Build MarketContext -----
            ctx = MarketContext(
                symbol=symbol,
                timeframe=timeframe,
                candles=candles_slice,
                features=features,
                regime=effective_regime,
                regime_confidence=effective_confidence,
                book=None,
                funding_rate=self._funding_rate,
                open_interest=None,
                account_equity=equity,
            )

            # ----- 7. Get signal -----
            signal = self.strategy.compute_signal(ctx)

            # ----- 8. Process actionable signal -----
            if signal.side not in ("flat",) and signal.entry is not None:
                # Map strategy side to approval side
                approval_side = _SIDE_TO_APPROVAL.get(signal.side, signal.side)
                approval_signal = Signal(
                    side=approval_side,
                    entry=signal.entry,
                    stop=signal.stop,
                    take_profit=signal.take_profit,
                    confidence=signal.confidence,
                    metadata=signal.metadata,
                )

                # Check we don't already have a trade in this direction
                already_in = any(t.side == signal.side for t in open_trades.values())
                if not already_in:
                    volatility_pct = features.get("realized_vol")
                    approval = approve_trade(
                        signal=approval_signal,
                        risk_state=risk_state,
                        risk_config=self.risk_config,
                        regime=effective_regime,
                        regime_confidence=effective_confidence,
                        volatility_pct=volatility_pct,
                    )

                    if approval.get("approved", False):
                        self._open_trade(signal, approval, ts, symbol, open_trades)

            # ----- 9. Increment holding bars for open trades -----
            for ot in open_trades.values():
                ot.holding_bars += 1

            # ----- 10. Record equity -----
            equity_history.append((ts, equity))

        # Close any remaining open trades at last bar's close
        if open_trades:
            last_bar = candles.iloc[-1]
            last_ts = (
                int(last_bar["timestamp_ms"]) if "timestamp_ms" in candles.columns else n_bars - 1
            )
            last_close = float(last_bar["close"])
            for tid in list(open_trades.keys()):
                trade = open_trades.pop(tid)
                self._close_trade(trade, last_close, last_ts, "end_of_data", closed_trades)

        return self._build_result(
            run_id,
            symbol,
            timeframe,
            closed_trades,
            equity_history,
            regime_history,
            self.config,
        )

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def _compute_features(self, candles_slice: pd.DataFrame) -> dict:
        """Compute all features from candle history up to (and including)
        the current bar.

        Parameters
        ----------
        candles_slice : pd.DataFrame
            Candle data from bar 0 through the current bar (inclusive).

        Returns
        -------
        dict
            Feature name to latest scalar value.
        """
        close = candles_slice["close"]
        high = candles_slice["high"]
        low = candles_slice["low"]
        open_ = candles_slice["open"]
        volume = candles_slice["volume"]

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
        features["sma"] = features["sma_50"]

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

        # Vol ratio (current realized vol / longer-window realized vol)
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

        # MACD
        macd_line, macd_signal, macd_hist = compute_macd(close, fast=12, slow=26, signal=9)
        features["macd_line"] = self._last_valid(macd_line)
        features["macd_signal"] = self._last_valid(macd_signal)
        features["macd_hist"] = self._last_valid(macd_hist)

        return features

    @staticmethod
    def _last_valid(series: pd.Series) -> float | None:
        """Return the last non-NaN value from *series*, or ``None``."""
        if series.empty:
            return None
        val = series.iloc[-1]
        if pd.isna(val):
            # Walk backwards to find last valid
            valid = series.dropna()
            if valid.empty:
                return None
            return float(valid.iloc[-1])
        return float(val)

    # ------------------------------------------------------------------
    # Lot / tick rounding (PRD §11.2.5: "HL tick/lot from metadata")
    # ------------------------------------------------------------------

    def _round_size(self, symbol: str, size: float) -> float:
        """Round order size to the exchange's sz_decimals for the asset.

        Mirrors :meth:`TradingScheduler._round_size` for live/backtest parity.
        """
        meta = self._asset_meta.get(symbol, {})
        sz_decimals = meta.get("sz_decimals", 3)
        factor = 10**sz_decimals
        return math.floor(size * factor) / factor

    def _round_price(self, symbol: str, price: float) -> float:
        """Round price to 5 significant figures.

        Mirrors :meth:`TradingScheduler._round_price` for live/backtest parity.
        """
        if price <= 0:
            return 0.0
        magnitude = math.floor(math.log10(abs(price)))
        decimals = max(0, 4 - magnitude)
        factor = 10**decimals
        return round(price * factor) / factor

    # ------------------------------------------------------------------
    # Exit checking
    # ------------------------------------------------------------------

    def _check_exits(
        self,
        bar: pd.Series,
        timestamp_ms: int,
        open_trades: dict[str, TradeRecord],
        closed_trades: list[TradeRecord],
        current_equity: float,
    ) -> None:
        """Check all open trades for stop-loss or take-profit hits.

        For longs:
            - If ``bar.low <= stop`` -> exit at stop price.
            - If ``bar.high >= take_profit`` -> exit at take_profit.
        For shorts:
            - If ``bar.high >= stop`` -> exit at stop price.
            - If ``bar.low <= take_profit`` -> exit at take_profit.

        If both stop and TP would be hit in the same bar, stop takes
        precedence (conservative assumption).
        """
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])

        to_close: list[tuple[str, float, str]] = []  # (trade_id, exit_px, reason)

        for tid, trade in open_trades.items():
            is_long = trade.side in ("long", "buy")

            if is_long:
                stop_hit = trade.stop > 0 and bar_low <= trade.stop
                tp_hit = trade.take_profit > 0 and bar_high >= trade.take_profit
            else:
                stop_hit = trade.stop > 0 and bar_high >= trade.stop
                tp_hit = trade.take_profit > 0 and bar_low <= trade.take_profit

            if stop_hit and tp_hit:
                # Conservative: assume stop hit first
                to_close.append((tid, trade.stop, "stop_loss"))
            elif stop_hit:
                to_close.append((tid, trade.stop, "stop_loss"))
            elif tp_hit:
                to_close.append((tid, trade.take_profit, "take_profit"))

        for tid, exit_px, reason in to_close:
            trade = open_trades.pop(tid)
            self._close_trade(trade, exit_px, timestamp_ms, reason, closed_trades)

    # ------------------------------------------------------------------
    # Trade lifecycle
    # ------------------------------------------------------------------

    def _open_trade(
        self,
        signal: Signal,
        approval: dict,
        timestamp_ms: int,
        symbol: str,
        open_trades: dict[str, TradeRecord],
    ) -> None:
        """Record a new trade opening from an approved signal.

        Parameters
        ----------
        signal : Signal
            The strategy signal (with side, entry, stop, take_profit).
        approval : dict
            Output of :func:`approve_trade` with sizing information.
        timestamp_ms : int
            Timestamp of the current bar.
        symbol : str
            Asset symbol.
        open_trades : dict
            Mutable dict of currently open trades.
        """
        trade_id = generate_trade_id()
        raw_entry_px = signal.entry if signal.entry is not None else 0.0
        raw_size_coins = approval.get("size_coins", 0.0)
        leverage = approval.get("leverage", 1.0)

        # Apply lot/tick rounding (PRD §11.2.5: parity with live)
        size_coins = self._round_size(symbol, raw_size_coins)
        entry_px = self._round_price(symbol, raw_entry_px)

        if size_coins <= 0:
            return  # rounded to zero — skip

        size_usd = size_coins * entry_px if entry_px > 0 else approval.get("size_usd", 0.0)

        # Entry slippage -- deducted from equity via cost accounting
        entry_slip = self.cost_model.compute_slippage(size_usd)
        entry_fee = self.cost_model.compute_entry_cost(
            size_usd, is_maker=self.cost_model.config.prefer_maker
        )

        trade = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            side=signal.side,
            entry_time=timestamp_ms,
            entry_px=entry_px,
            size=size_coins,
            notional=size_usd,
            leverage=leverage,
            stop=self._round_price(symbol, signal.stop) if signal.stop else 0.0,
            take_profit=self._round_price(symbol, signal.take_profit) if signal.take_profit else 0.0,
            fees=entry_fee,
            slippage=entry_slip,
        )

        open_trades[trade_id] = trade

        logger.debug(
            "trade_opened",
            trade_id=trade_id,
            side=signal.side,
            entry_px=entry_px,
            size=size_coins,
            notional=size_usd,
            leverage=leverage,
        )

    def _close_trade(
        self,
        trade: TradeRecord,
        exit_px: float,
        exit_time: int,
        reason: str,
        closed_trades: list[TradeRecord],
    ) -> None:
        """Close a trade, computing final PnL with all costs.

        Parameters
        ----------
        trade : TradeRecord
            The open trade to close.
        exit_px : float
            Price at which the trade is exited.
        exit_time : int
            Timestamp of the exit bar (ms).
        reason : str
            Reason for closing (``"stop_loss"``, ``"take_profit"``,
            ``"end_of_data"``).
        closed_trades : list[TradeRecord]
            Mutable list where the closed trade will be appended.
        """
        trade.exit_time = exit_time
        trade.exit_px = exit_px
        trade.exit_reason = reason

        # Raw PnL (before costs)
        is_long = trade.side in ("long", "buy")
        if is_long:
            raw_pnl = (exit_px - trade.entry_px) * trade.size
        else:
            raw_pnl = (trade.entry_px - exit_px) * trade.size

        # Exit costs
        exit_notional = exit_px * trade.size
        exit_fee = self.cost_model.compute_exit_cost(exit_notional, is_maker=False)

        # Funding cost for the holding period
        # Longs pay positive funding, shorts receive (negative notional).
        # CostModel preserves sign: positive = cost, negative = rebate.
        hours_held = trade.holding_bars * self._bar_hours
        funding_sign = 1.0 if is_long else -1.0
        funding = self.cost_model.compute_funding_cost(
            trade.notional * funding_sign,
            self._funding_rate,
            hours_held,
        )

        trade.fees += exit_fee
        # Preserve sign: positive = cost, negative = rebate (shorts earn
        # funding when rate is positive).
        trade.funding = funding
        trade.pnl = raw_pnl
        # Net PnL is raw_pnl minus all costs -- but we store raw in .pnl
        # and attribute costs separately so metrics can distinguish.

        closed_trades.append(trade)

        logger.debug(
            "trade_closed",
            trade_id=trade.trade_id,
            side=trade.side,
            entry_px=trade.entry_px,
            exit_px=exit_px,
            pnl=raw_pnl,
            fees=trade.fees,
            slippage=trade.slippage,
            funding=trade.funding,
            reason=reason,
            holding_bars=trade.holding_bars,
        )

    # ------------------------------------------------------------------
    # Result builder
    # ------------------------------------------------------------------

    def _build_result(
        self,
        run_id: str,
        symbol: str,
        timeframe: str,
        closed_trades: list[TradeRecord],
        equity_history: list[tuple[int, float]],
        regime_history: list[tuple[int, str, float]],
        config: dict,
    ) -> dict:
        """Package backtest outputs into the standard result dict."""
        # Build trades DataFrame
        if closed_trades:
            trades_df = pd.DataFrame([asdict(t) for t in closed_trades])
        else:
            trades_df = pd.DataFrame(
                columns=[
                    "trade_id",
                    "symbol",
                    "side",
                    "entry_time",
                    "exit_time",
                    "entry_px",
                    "exit_px",
                    "size",
                    "notional",
                    "leverage",
                    "pnl",
                    "fees",
                    "slippage",
                    "funding",
                    "holding_bars",
                    "exit_reason",
                    "stop",
                    "take_profit",
                ]
            )

        # Build equity curve
        if equity_history:
            timestamps, equities = zip(*equity_history)
            equity_curve = pd.Series(data=list(equities), index=list(timestamps), name="equity")
        else:
            equity_curve = pd.Series(dtype=float, name="equity")

        # Compute metrics
        metrics = compute_metrics(equity_curve, trades_df)

        logger.info(
            "backtest_complete",
            run_id=run_id,
            total_trades=metrics.total_trades,
            total_return=f"{metrics.total_return:.4f}",
            sharpe=f"{metrics.sharpe_ratio:.4f}",
            max_dd=f"{metrics.max_drawdown:.4f}",
            utility=f"{metrics.utility:.4f}",
        )

        return {
            "run_id": run_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "trades": trades_df,
            "equity_curve": equity_curve,
            "metrics": metrics,
            "regime_history": regime_history,
            "config": config,
        }
