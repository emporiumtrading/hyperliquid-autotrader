"""Walk-forward testing.

Splits candle data into rolling train/test windows and evaluates strategy
performance on each out-of-sample (OOS) fold to detect overfitting and
estimate realistic forward performance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
import structlog

from autotrader.backtest.engine import BacktestEngine
from autotrader.backtest.metrics import BacktestMetrics, compute_metrics
from autotrader.strategies.base import BaseStrategy
from autotrader.strategies.ensemble import EnsembleStrategy

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardConfig:
    """Parameters for walk-forward analysis.

    Attributes
    ----------
    train_bars : int
        Number of bars in each training window (used for context but the
        strategy is not re-optimised -- just provides warm-up data).
    test_bars : int
        Number of bars in each out-of-sample test window.
    step_bars : int
        Number of bars to advance between successive folds.
    min_train_bars : int
        Minimum required training bars; folds that would have fewer are
        skipped.
    """

    train_bars: int = 2000
    test_bars: int = 500
    step_bars: int = 500
    min_train_bars: int = 1000


# ---------------------------------------------------------------------------
# Walk-forward test
# ---------------------------------------------------------------------------


class WalkForwardTest:
    """Rolling walk-forward out-of-sample evaluator.

    Parameters
    ----------
    config : WalkForwardConfig | None
        Walk-forward configuration.  Uses defaults if ``None``.
    """

    def __init__(self, config: WalkForwardConfig | None = None) -> None:
        self.config = config or WalkForwardConfig()

    # ------------------------------------------------------------------
    # Fold generation
    # ------------------------------------------------------------------

    def generate_folds(self, total_bars: int) -> list[tuple[int, int, int, int]]:
        """Generate (train_start, train_end, test_start, test_end) tuples.

        Parameters
        ----------
        total_bars : int
            Total number of candle bars available.

        Returns
        -------
        list[tuple[int, int, int, int]]
            List of fold boundaries as integer bar indices.
        """
        folds: list[tuple[int, int, int, int]] = []

        train_start = 0
        while True:
            train_end = train_start + self.config.train_bars
            test_start = train_end
            test_end = test_start + self.config.test_bars

            # Ensure we have enough data
            if train_end > total_bars:
                break
            if test_end > total_bars:
                # Use whatever remaining bars are available for the last fold
                test_end = total_bars
                if test_end - test_start < 1:
                    break

            # Check minimum training requirement
            if train_end - train_start < self.config.min_train_bars:
                train_start += self.config.step_bars
                continue

            folds.append((train_start, train_end, test_start, test_end))

            # Step forward
            train_start += self.config.step_bars

            # Stop if the test window would start past available data
            if test_start >= total_bars:
                break

        return folds

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------

    def run(
        self,
        candles: pd.DataFrame,
        strategy: BaseStrategy | EnsembleStrategy,
        backtest_config: dict,
        symbol: str = "ETH",
        timeframe: str = "15m",
    ) -> dict:
        """Execute walk-forward testing.

        For each fold, the strategy is evaluated on the out-of-sample
        window using the training window as warm-up context (the strategy
        is **not** re-optimised between folds -- this evaluates how a
        fixed strategy generalises over time).

        Parameters
        ----------
        candles : pd.DataFrame
            Full candle dataset with OHLCV columns.
        strategy : BaseStrategy | EnsembleStrategy
            The strategy instance to evaluate.
        backtest_config : dict
            Configuration dict passed to :class:`BacktestEngine`.
        symbol : str
            Asset symbol.
        timeframe : str
            Candle timeframe string.

        Returns
        -------
        dict
            ``{folds, oos_equity_curve, oos_metrics, fold_metrics, config}``.
        """
        candles = candles.reset_index(drop=True)
        total_bars = len(candles)
        folds = self.generate_folds(total_bars)

        if not folds:
            logger.warning(
                "no_folds_generated",
                total_bars=total_bars,
                train_bars=self.config.train_bars,
                test_bars=self.config.test_bars,
            )
            return {
                "folds": [],
                "oos_equity_curve": pd.Series(dtype=float),
                "oos_metrics": BacktestMetrics(),
                "fold_metrics": [],
                "config": asdict(self.config),
            }

        logger.info(
            "walkforward_start",
            n_folds=len(folds),
            total_bars=total_bars,
        )

        fold_results: list[dict] = []
        fold_metrics_list: list[BacktestMetrics] = []
        oos_equity_pieces: list[pd.Series] = []
        oos_trades_pieces: list[pd.DataFrame] = []

        for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds):
            logger.info(
                "walkforward_fold",
                fold=fold_idx,
                train=f"{train_start}:{train_end}",
                test=f"{test_start}:{test_end}",
            )

            # The OOS test includes the tail of the training window for
            # indicator warm-up.  We provide enough history so that the
            # engine's warmup_bars requirement is met.
            warmup = int(backtest_config.get("warmup_bars", 50))
            context_start = max(0, test_start - warmup)

            fold_candles = candles.iloc[context_start:test_end].reset_index(drop=True)

            engine = BacktestEngine(
                config=backtest_config,
                strategy=strategy,
            )
            result = engine.run(fold_candles, symbol=symbol, timeframe=timeframe)

            fold_results.append(result)
            fold_metrics_list.append(result["metrics"])

            # Collect OOS equity and trades
            if not result["equity_curve"].empty:
                oos_equity_pieces.append(result["equity_curve"])
            if not result["trades"].empty:
                oos_trades_pieces.append(result["trades"])

        # Combine OOS equity curves into one continuous series
        if oos_equity_pieces:
            oos_equity_curve = _chain_equity_curves(
                oos_equity_pieces,
                float(backtest_config.get("initial_equity", 10_000.0)),
            )
        else:
            oos_equity_curve = pd.Series(dtype=float)

        # Combine OOS trades
        if oos_trades_pieces:
            oos_trades = pd.concat(oos_trades_pieces, ignore_index=True)
        else:
            oos_trades = pd.DataFrame()

        # Aggregate OOS metrics
        oos_metrics = compute_metrics(oos_equity_curve, oos_trades)

        logger.info(
            "walkforward_complete",
            n_folds=len(folds),
            oos_return=f"{oos_metrics.total_return:.4f}",
            oos_sharpe=f"{oos_metrics.sharpe_ratio:.4f}",
            oos_utility=f"{oos_metrics.utility:.4f}",
        )

        return {
            "folds": fold_results,
            "oos_equity_curve": oos_equity_curve,
            "oos_metrics": oos_metrics,
            "fold_metrics": fold_metrics_list,
            "config": asdict(self.config),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chain_equity_curves(pieces: list[pd.Series], initial_equity: float) -> pd.Series:
    """Chain multiple equity curve segments into one continuous curve.

    Each fold starts at its own initial equity.  We scale subsequent folds
    so that the combined curve is continuous (fold N+1 starts where fold N
    ended).

    Parameters
    ----------
    pieces : list[pd.Series]
        Individual equity curves from each fold.
    initial_equity : float
        The configured starting equity.

    Returns
    -------
    pd.Series
        Combined continuous equity curve.
    """
    if not pieces:
        return pd.Series(dtype=float)

    combined_index: list = []
    combined_values: list[float] = []

    current_equity = initial_equity

    for piece in pieces:
        if piece.empty:
            continue

        fold_start = float(piece.iloc[0])
        if fold_start <= 0:
            fold_start = initial_equity

        # Scale factor so this fold begins at current_equity
        scale = current_equity / fold_start

        for idx_val, eq_val in zip(piece.index, piece.values):
            combined_index.append(idx_val)
            combined_values.append(float(eq_val) * scale)

        if combined_values:
            current_equity = combined_values[-1]

    return pd.Series(data=combined_values, index=combined_index, name="equity")
