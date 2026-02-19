"""Comprehensive backtest metrics.

Computes performance, risk, and cost-attribution metrics from an equity
curve and a trade ledger produced by the backtest engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Metrics dataclass
# ---------------------------------------------------------------------------


@dataclass
class BacktestMetrics:
    """Complete set of backtest performance metrics."""

    total_return: float = 0.0
    cagr: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration_bars: int = 0
    calmar_ratio: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_rr: float = 0.0
    total_trades: int = 0
    long_trades: int = 0
    short_trades: int = 0
    avg_holding_bars: int = 0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    worst_day: float = 0.0
    best_day: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    total_funding: float = 0.0
    net_profit: float = 0.0
    gross_profit: float = 0.0
    utility: float = 0.0


# ---------------------------------------------------------------------------
# Drawdown helpers
# ---------------------------------------------------------------------------


def compute_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Return a drawdown percentage series.

    Values are 0 at equity peaks and negative during drawdowns (e.g. -0.10
    means a 10 % drawdown from peak).

    Parameters
    ----------
    equity_curve : pd.Series
        Equity values indexed by timestamp or bar number.

    Returns
    -------
    pd.Series
        Drawdown fraction series (same index as *equity_curve*).
    """
    if equity_curve.empty:
        return pd.Series(dtype=float)

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    # Replace inf/-inf that could arise from a zero peak.
    drawdown = drawdown.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return drawdown


# ---------------------------------------------------------------------------
# Rolling Sharpe
# ---------------------------------------------------------------------------


def compute_rolling_sharpe(equity_curve: pd.Series, window: int = 60) -> pd.Series:
    """Compute a rolling annualised Sharpe ratio from the equity curve.

    Parameters
    ----------
    equity_curve : pd.Series
        Equity values indexed by timestamp or bar number.
    window : int
        Number of *daily* return observations in the rolling window.

    Returns
    -------
    pd.Series
        Rolling Sharpe ratio (same index as *equity_curve*).
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return pd.Series(dtype=float)

    daily_returns = equity_curve.pct_change().dropna()
    if len(daily_returns) < window:
        return pd.Series(np.nan, index=daily_returns.index)

    rolling_mean = daily_returns.rolling(window=window, min_periods=window).mean()
    rolling_std = daily_returns.rolling(window=window, min_periods=window).std(ddof=1)
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(365)
    rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan)
    return rolling_sharpe


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_daily_returns(equity_curve: pd.Series) -> pd.Series:
    """Convert an equity curve to daily percentage returns.

    If the equity curve's index is a DatetimeIndex, resample to daily.
    Otherwise treat every observation as one period.
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return pd.Series(dtype=float)

    if isinstance(equity_curve.index, pd.DatetimeIndex):
        daily_equity = equity_curve.resample("1D").last().dropna()
    else:
        daily_equity = equity_curve

    returns = daily_equity.pct_change().dropna()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    return returns


def _compute_cvar(returns: pd.Series, confidence: float) -> float:
    """Conditional Value-at-Risk (expected shortfall) at *confidence* level.

    Returns a **positive** number representing the expected loss in the
    worst (1 - confidence) tail.
    """
    if returns.empty:
        return 0.0
    cutoff = np.percentile(returns.values, (1.0 - confidence) * 100.0)
    tail = returns[returns <= cutoff]
    if tail.empty:
        return 0.0
    return float(-tail.mean())


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------


def compute_metrics(
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    config: dict | None = None,
) -> BacktestMetrics:
    """Compute all backtest metrics from an equity curve and trade ledger.

    Parameters
    ----------
    equity_curve : pd.Series
        Equity values indexed by timestamp (ms or DatetimeIndex).
    trades : pd.DataFrame
        Trade ledger with columns: ``entry_time``, ``exit_time``, ``side``,
        ``entry_px``, ``exit_px``, ``size``, ``pnl``, ``fees``, ``slippage``,
        ``funding``, ``holding_bars``.
    config : dict | None
        Optional overrides:

        - ``risk_free_rate`` (float): annualised risk-free rate (default 0.0).
        - ``lambda_dd`` (float): drawdown penalty for utility (default 1.0).
        - ``mu_cvar`` (float): CVaR penalty for utility (default 0.5).
        - ``nu_turnover`` (float): turnover cost penalty for utility
          (default 0.1).

    Returns
    -------
    BacktestMetrics
        Fully populated metrics dataclass.
    """
    cfg = config or {}
    risk_free_rate: float = float(cfg.get("risk_free_rate", 0.0))
    lambda_dd: float = float(cfg.get("lambda_dd", 1.0))
    mu_cvar: float = float(cfg.get("mu_cvar", 0.5))
    nu_turnover: float = float(cfg.get("nu_turnover", 0.1))

    m = BacktestMetrics()

    # ------------------------------------------------------------------
    # Equity-curve metrics
    # ------------------------------------------------------------------
    if equity_curve.empty or len(equity_curve) < 2:
        return m

    initial_equity = float(equity_curve.iloc[0])
    final_equity = float(equity_curve.iloc[-1])

    if initial_equity <= 0:
        return m

    # Total return
    m.total_return = (final_equity - initial_equity) / initial_equity

    # CAGR
    n_bars = len(equity_curve)
    # Estimate duration in years: if DatetimeIndex use actual time, else
    # assume each bar is 15 minutes (sensible default for crypto).
    if isinstance(equity_curve.index, pd.DatetimeIndex) and n_bars >= 2:
        total_seconds = (equity_curve.index[-1] - equity_curve.index[0]).total_seconds()
        years = max(total_seconds / (365.25 * 86400), 1e-9)
    else:
        # Assume 15-min bars, 96 bars/day, 365 days/year
        years = max(n_bars / (96.0 * 365.0), 1e-9)

    if final_equity > 0 and initial_equity > 0:
        m.cagr = (final_equity / initial_equity) ** (1.0 / years) - 1.0
    else:
        m.cagr = -1.0

    # Daily returns
    daily_returns = _compute_daily_returns(equity_curve)

    if len(daily_returns) > 1:
        std_ret = float(daily_returns.std(ddof=1))

        # Sharpe ratio (annualised, crypto: 365 days)
        daily_rf = risk_free_rate / 365.0
        excess = daily_returns - daily_rf
        excess_mean = float(excess.mean())
        if std_ret > 1e-12:
            m.sharpe_ratio = (excess_mean / std_ret) * np.sqrt(365)
        else:
            m.sharpe_ratio = 0.0

        # Sortino ratio (annualised)
        downside = excess[excess < 0.0]
        if len(downside) > 0:
            downside_std = float(np.sqrt((downside**2).mean()))
            if downside_std > 1e-12:
                m.sortino_ratio = (excess_mean / downside_std) * np.sqrt(365)
            else:
                m.sortino_ratio = 0.0
        else:
            # No downside returns observed
            m.sortino_ratio = float(m.sharpe_ratio) if m.sharpe_ratio > 0 else 0.0

        # Best / worst day
        m.best_day = float(daily_returns.max())
        m.worst_day = float(daily_returns.min())

        # CVaR
        m.cvar_95 = _compute_cvar(daily_returns, 0.95)
        m.cvar_99 = _compute_cvar(daily_returns, 0.99)

    # Drawdown
    dd_series = compute_drawdown_series(equity_curve)
    m.max_drawdown = float(abs(dd_series.min())) if not dd_series.empty else 0.0

    # Max drawdown duration (in bars)
    if not dd_series.empty:
        running_max = equity_curve.cummax()
        in_drawdown = equity_curve < running_max
        # Compute consecutive groups in drawdown
        groups = (~in_drawdown).cumsum()
        if in_drawdown.any():
            dd_durations = in_drawdown.groupby(groups).sum()
            m.max_drawdown_duration_bars = int(dd_durations.max())
        else:
            m.max_drawdown_duration_bars = 0
    else:
        m.max_drawdown_duration_bars = 0

    # Calmar ratio
    if m.max_drawdown > 1e-12:
        m.calmar_ratio = m.cagr / m.max_drawdown
    else:
        m.calmar_ratio = 0.0

    # ------------------------------------------------------------------
    # Trade-level metrics
    # ------------------------------------------------------------------
    if trades is not None and not trades.empty:
        m.total_trades = len(trades)

        if "side" in trades.columns:
            m.long_trades = int((trades["side"].isin(["long", "buy"])).sum())
            m.short_trades = int((trades["side"].isin(["short", "sell"])).sum())

        if "pnl" in trades.columns:
            pnl = trades["pnl"].astype(float)
            wins = pnl[pnl > 0]
            losses = pnl[pnl < 0]

            m.win_rate = float(len(wins)) / max(m.total_trades, 1)
            m.avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
            m.avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

            # Avg reward:risk ratio
            if m.avg_loss != 0:
                m.avg_rr = abs(m.avg_win / m.avg_loss)
            else:
                m.avg_rr = 0.0 if m.avg_win == 0 else float("inf")

            # Profit factor = gross_wins / gross_losses
            gross_wins = float(wins.sum()) if len(wins) > 0 else 0.0
            gross_losses = float(abs(losses.sum())) if len(losses) > 0 else 0.0
            if gross_losses > 1e-12:
                m.profit_factor = gross_wins / gross_losses
            else:
                m.profit_factor = float("inf") if gross_wins > 0 else 0.0

            m.gross_profit = float(pnl.sum())

        if "holding_bars" in trades.columns:
            m.avg_holding_bars = int(trades["holding_bars"].mean())

        # Cost attribution
        if "fees" in trades.columns:
            m.total_fees = float(trades["fees"].astype(float).sum())
        if "slippage" in trades.columns:
            m.total_slippage = float(trades["slippage"].astype(float).sum())
        if "funding" in trades.columns:
            m.total_funding = float(trades["funding"].astype(float).sum())

        m.net_profit = m.gross_profit - m.total_fees - m.total_slippage - m.total_funding
    else:
        m.net_profit = final_equity - initial_equity

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    # Turnover cost = total fees + slippage as fraction of initial equity
    turnover_cost = (m.total_fees + m.total_slippage) / max(initial_equity, 1e-12)
    m.utility = (
        m.cagr - lambda_dd * m.max_drawdown - mu_cvar * m.cvar_95 - nu_turnover * turnover_cost
    )

    return m
