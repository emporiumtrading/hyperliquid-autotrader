"""Probation period logic (canary).

After a new strategy version is promoted, it enters a probation / canary
period with tighter risk limits.  The :class:`ProbationEvaluator` collects
trades and daily PnL during this window and decides whether the strategy
can graduate to full production or must be rolled back.
"""

from __future__ import annotations

import structlog

from autotrader.utils.time import MS_PER_DAY, now_ms

logger = structlog.get_logger(__name__)


class ProbationEvaluator:
    """Evaluates a strategy during its canary / probation period.

    Parameters
    ----------
    config:
        Optional configuration dictionary.  Recognised keys:

        * ``probation_days`` (int) -- length of probation in days (default 7).
        * ``min_trades`` (int) -- minimum trades required before graduation
          (default 10).
        * ``max_drawdown_pct`` (float) -- maximum drawdown allowed during
          probation, tighter than the live limit (default 0.10 = 10%).
        * ``min_profit_factor`` (float) -- minimum acceptable profit factor
          during probation (default 1.0).
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.probation_days: int = int(cfg.get("probation_days", 7))
        self.min_trades: int = int(cfg.get("min_trades", 10))
        self.max_drawdown_pct: float = float(cfg.get("max_drawdown_pct", 0.10))
        self.min_profit_factor: float = float(cfg.get("min_profit_factor", 1.0))
        self.started_at: int | None = None
        self.trades: list[dict] = []
        self.daily_pnls: list[float] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Mark the beginning of the probation period.

        Resets all accumulated state so the evaluator can be reused.
        """
        self.started_at = now_ms()
        self.trades.clear()
        self.daily_pnls.clear()
        logger.info("probation.start", started_at=self.started_at)

    def is_active(self) -> bool:
        """Return whether probation is currently in progress.

        Probation is active from the moment :meth:`start` is called until
        the probation window elapses.
        """
        if self.started_at is None:
            return False
        elapsed_ms = now_ms() - self.started_at
        return elapsed_ms < self.probation_days * MS_PER_DAY

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def add_trade(self, trade: dict) -> None:
        """Record a trade executed during probation.

        Parameters
        ----------
        trade:
            Trade dictionary.  Expected to contain at least a ``pnl``
            key (float).  Additional keys (``side``, ``size``, ``symbol``,
            etc.) are stored but not required.
        """
        self.trades.append(trade)
        logger.debug(
            "probation.add_trade",
            trade_count=len(self.trades),
            pnl=trade.get("pnl"),
        )

    def add_daily_pnl(self, pnl: float) -> None:
        """Record a daily PnL observation.

        Parameters
        ----------
        pnl:
            The day's profit or loss.
        """
        self.daily_pnls.append(pnl)
        logger.debug("probation.add_daily_pnl", day=len(self.daily_pnls), pnl=pnl)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> dict:
        """Evaluate whether the strategy can graduate from probation.

        Returns
        -------
        dict
            Evaluation result with keys:

            * ``passed`` (bool) -- ``True`` if all criteria are met.
            * ``reasons`` (list[str]) -- descriptions of any failures.
            * ``days_elapsed`` (int) -- number of full days since start.
            * ``trades_count`` (int) -- total trades recorded.
            * ``pnl`` (float) -- cumulative PnL over the period.
            * ``max_dd`` (float) -- maximum drawdown observed.
            * ``can_promote`` (bool) -- shorthand: passed AND probation
              period is complete.
        """
        reasons: list[str] = []

        # Days elapsed
        if self.started_at is not None:
            elapsed_ms = now_ms() - self.started_at
            days_elapsed = int(elapsed_ms / MS_PER_DAY)
        else:
            days_elapsed = 0

        # Period check
        period_complete = days_elapsed >= self.probation_days
        if not period_complete:
            reasons.append(
                f"Probation period incomplete: {days_elapsed}/{self.probation_days} days"
            )

        # Trade count
        trades_count = len(self.trades)
        enough_trades = trades_count >= self.min_trades
        if not enough_trades:
            reasons.append(f"Insufficient trades: {trades_count}/{self.min_trades}")

        # Cumulative PnL from trades
        trade_pnls = [float(t.get("pnl", 0.0)) for t in self.trades]
        total_pnl = sum(trade_pnls)

        # Maximum drawdown from daily PnL series
        max_dd = _compute_max_drawdown(self.daily_pnls)
        dd_ok = abs(max_dd) <= self.max_drawdown_pct
        if not dd_ok:
            reasons.append(
                f"Max drawdown {abs(max_dd):.4f} exceeds probation limit "
                f"{self.max_drawdown_pct:.4f}"
            )

        # Profit factor
        gross_profit = sum(p for p in trade_pnls if p > 0)
        gross_loss = abs(sum(p for p in trade_pnls if p < 0))
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        pf_ok = profit_factor >= self.min_profit_factor
        if not pf_ok and enough_trades:
            reasons.append(
                f"Profit factor {profit_factor:.4f} below minimum " f"{self.min_profit_factor:.4f}"
            )

        passed = period_complete and enough_trades and dd_ok and pf_ok
        can_promote = passed and period_complete

        logger.info(
            "probation.evaluate",
            passed=passed,
            days_elapsed=days_elapsed,
            trades_count=trades_count,
            pnl=total_pnl,
            max_dd=max_dd,
            profit_factor=profit_factor,
            reasons=reasons,
        )

        return {
            "passed": passed,
            "reasons": reasons,
            "days_elapsed": days_elapsed,
            "trades_count": trades_count,
            "pnl": total_pnl,
            "max_dd": max_dd,
            "profit_factor": profit_factor,
            "can_promote": can_promote,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_max_drawdown(daily_pnls: list[float]) -> float:
    """Compute maximum drawdown from a sequence of daily PnL values.

    The drawdown is expressed as a fraction of peak cumulative PnL.
    Returns 0.0 if there are no observations or no drawdown.

    A negative return value indicates the magnitude of the worst drawdown
    (e.g. -0.12 means a 12% drawdown from peak).
    """
    if not daily_pnls:
        return 0.0

    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0

    for pnl in daily_pnls:
        cumulative += pnl
        if cumulative > peak:
            peak = cumulative
        if peak > 0:
            dd = (cumulative - peak) / peak
            if dd < max_dd:
                max_dd = dd

    return max_dd
