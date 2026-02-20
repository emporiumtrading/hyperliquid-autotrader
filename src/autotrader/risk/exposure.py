"""Exposure tracking for open positions.

Maintains a live registry of :class:`Position` objects, recalculates
unrealised PnL on price updates, and exposes aggregate portfolio metrics
such as total notional, net/gross exposure, and margin usage.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import structlog

from autotrader.risk.constraints import RiskState

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Position dataclass
# ---------------------------------------------------------------------------


@dataclass
class Position:
    """A single open position tracked by the :class:`ExposureTracker`."""

    symbol: str
    side: str  # "long" or "short"
    size: float  # absolute coin quantity
    entry_px: float
    current_px: float
    leverage: float
    unrealized_pnl: float = 0.0
    margin_used: float = 0.0


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class ReturnTracker:
    """Collects per-symbol signed returns for rolling correlation computation.

    Stores the most recent *lookback* return observations per symbol,
    then builds a pairwise correlation matrix on demand.
    """

    def __init__(self, lookback: int = 60, min_obs: int = 20) -> None:
        self.lookback = lookback
        self.min_obs = min_obs
        self._returns: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def record(self, symbol: str, ret: float) -> None:
        """Append a return observation for *symbol*."""
        with self._lock:
            if symbol not in self._returns:
                self._returns[symbol] = deque(maxlen=self.lookback)
            self._returns[symbol].append(ret)

    def correlation_matrix(
        self, symbols: list[str]
    ) -> np.ndarray | None:
        """Return a pairwise correlation matrix for *symbols*.

        Returns ``None`` when fewer than 2 symbols have at least
        ``min_obs`` overlapping observations.
        """
        with self._lock:
            valid = [
                s for s in symbols
                if s in self._returns and len(self._returns[s]) >= self.min_obs
            ]
            if len(valid) < 2:
                return None

            # Align to the shortest overlapping length
            n = min(len(self._returns[s]) for s in valid)
            data = np.column_stack(
                [list(self._returns[s])[-n:] for s in valid]
            )

        # np.corrcoef returns (k, k) for k columns when rowvar=False
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.corrcoef(data, rowvar=False)
        # Replace NaN (e.g. zero-variance series) with 0
        corr = np.nan_to_num(corr, nan=0.0)
        return corr


def _correlation_clusters(
    symbols: list[str],
    corr: np.ndarray,
    threshold: float = 0.5,
) -> list[list[str]]:
    """Find connected components where |correlation| > *threshold*.

    Simple union-find over the correlation matrix -- no external deps.
    """
    n = len(symbols)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr[i, j]) > threshold:
                union(i, j)

    groups: dict[int, list[str]] = {}
    for i, sym in enumerate(symbols):
        root = find(i)
        groups.setdefault(root, []).append(sym)

    return list(groups.values())


class ExposureTracker:
    """Maintains a set of open positions and computes aggregate exposures."""

    def __init__(
        self,
        corr_lookback: int = 60,
        corr_min_obs: int = 20,
        corr_threshold: float = 0.5,
    ) -> None:
        self.positions: dict[str, Position] = {}
        self._lock = threading.Lock()
        self._return_tracker = ReturnTracker(
            lookback=corr_lookback, min_obs=corr_min_obs
        )
        self._corr_threshold = corr_threshold

    # -- mutators -----------------------------------------------------------

    def add_position(self, pos: Position) -> None:
        """Register a new position (or overwrite if same symbol exists)."""
        with self._lock:
            pos.unrealized_pnl = self._calc_unrealized_pnl(pos)
            pos.margin_used = self._calc_margin(pos)
            self.positions[pos.symbol] = pos
        logger.info(
            "position_added",
            symbol=pos.symbol,
            side=pos.side,
            size=pos.size,
            entry_px=pos.entry_px,
            leverage=pos.leverage,
        )

    def remove_position(self, symbol: str) -> None:
        """Remove a position by symbol.  No-op if it does not exist."""
        with self._lock:
            removed = self.positions.pop(symbol, None)
        if removed is not None:
            logger.info("position_removed", symbol=symbol)
        else:
            logger.debug("position_remove_noop", symbol=symbol)

    def update_price(self, symbol: str, price: float) -> None:
        """Update the mark price for *symbol* and recalculate PnL/margin."""
        with self._lock:
            pos = self.positions.get(symbol)
            if pos is None:
                return
            old_px = pos.current_px
            pos.current_px = price
            pos.unrealized_pnl = self._calc_unrealized_pnl(pos)
            pos.margin_used = self._calc_margin(pos)
        # Record signed return for correlation tracking
        if old_px > 0:
            sign = 1.0 if pos.side == "long" else -1.0
            ret = sign * (price - old_px) / old_px
            self._return_tracker.record(symbol, ret)

    def record_return(self, symbol: str, ret: float) -> None:
        """Manually record a signed return for correlation tracking."""
        self._return_tracker.record(symbol, ret)

    # -- aggregation --------------------------------------------------------

    def total_notional(self) -> float:
        """Sum of ``abs(size * current_px)`` for every open position."""
        with self._lock:
            return sum(abs(p.size * p.current_px) for p in self.positions.values())

    def total_unrealized_pnl(self) -> float:
        """Sum of unrealised PnL across all open positions."""
        with self._lock:
            return sum(p.unrealized_pnl for p in self.positions.values())

    def total_margin_used(self) -> float:
        """Sum of margin committed across all open positions."""
        with self._lock:
            return sum(p.margin_used for p in self.positions.values())

    def net_exposure(self) -> float:
        """Signed exposure: long notional minus short notional."""
        with self._lock:
            total = 0.0
            for p in self.positions.values():
                notional = abs(p.size * p.current_px)
                if p.side == "long":
                    total += notional
                else:
                    total -= notional
            return total

    def gross_exposure(self) -> float:
        """Unsigned total exposure (sum of absolute notional values)."""
        return self.total_notional()

    def position_count(self) -> int:
        """Number of open positions."""
        with self._lock:
            return len(self.positions)

    def get_position(self, symbol: str) -> Position | None:
        """Return the :class:`Position` for *symbol*, or ``None``."""
        with self._lock:
            return self.positions.get(symbol)

    # -- correlation-based exposure limits ------------------------------------

    def same_direction_count(self, side: str) -> int:
        """Count positions with the same direction."""
        with self._lock:
            return sum(1 for p in self.positions.values() if p.side == side)

    def same_direction_notional(self, side: str) -> float:
        """Total notional of positions in the same direction."""
        with self._lock:
            return sum(
                abs(p.size * p.current_px)
                for p in self.positions.values()
                if p.side == side
            )

    def max_correlated_cluster_exposure(
        self, max_cluster_pct: float = 0.6
    ) -> tuple[bool, float]:
        """Check if any correlated cluster exceeds *max_cluster_pct* of gross.

        Uses a rolling correlation matrix when sufficient return history
        is available (at least ``min_obs`` observations for two or more
        held symbols).  Falls back to a directional heuristic (long vs
        short dominance) when return data is sparse.

        Correlation-based clustering:
        * Build a pairwise correlation matrix from recent signed returns.
        * Two symbols are "correlated" when |rho| > ``corr_threshold``
          (default 0.5).
        * Connected components of the correlation graph form clusters.
        * Cluster notional = sum of notional of the cluster's members.
        * If any cluster's share of gross notional exceeds
          *max_cluster_pct*, the check fails.

        Returns
        -------
        tuple[bool, float]
            ``(is_exceeded, worst_cluster_pct)``
        """
        with self._lock:
            symbols = list(self.positions.keys())
            notional_map = {
                s: abs(p.size * p.current_px)
                for s, p in self.positions.items()
            }

        gross = sum(notional_map.values())
        if gross <= 0 or len(symbols) < 2:
            return False, 0.0

        # Try correlation-based clustering first
        corr = self._return_tracker.correlation_matrix(symbols)
        if corr is not None:
            clusters = _correlation_clusters(
                symbols, corr, self._corr_threshold
            )
            worst_pct = 0.0
            for cluster in clusters:
                cluster_notional = sum(notional_map.get(s, 0.0) for s in cluster)
                pct = cluster_notional / gross
                worst_pct = max(worst_pct, pct)
            exceeded = worst_pct > max_cluster_pct
            if exceeded:
                logger.info(
                    "correlated_cluster_breach",
                    worst_pct=round(worst_pct, 3),
                    cluster_count=len(clusters),
                )
            return exceeded, worst_pct

        # Fallback: directional heuristic when return data is sparse
        with self._lock:
            long_n = sum(
                abs(p.size * p.current_px)
                for p in self.positions.values()
                if p.side == "long"
            )
            short_n = sum(
                abs(p.size * p.current_px)
                for p in self.positions.values()
                if p.side == "short"
            )

        dominant_pct = max(long_n, short_n) / gross
        return dominant_pct > max_cluster_pct, dominant_pct

    # -- bridge to RiskState ------------------------------------------------

    def to_risk_state(
        self,
        equity: float,
        peak_equity: float,
        daily_pnl: float,
        weekly_pnl: float,
    ) -> RiskState:
        """Create a :class:`RiskState` from the tracker's live data.

        The caller supplies equity-level parameters that are not tracked
        inside this class (account equity, peak, and PnL windows).
        """
        return RiskState(
            equity=equity,
            peak_equity=peak_equity,
            daily_pnl=daily_pnl,
            weekly_pnl=weekly_pnl,
            open_positions=self.position_count(),
            total_notional=self.total_notional(),
        )

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _calc_unrealized_pnl(pos: Position) -> float:
        """Compute unrealised PnL for a single position."""
        price_delta = pos.current_px - pos.entry_px
        if pos.side == "short":
            price_delta = -price_delta
        return price_delta * pos.size

    @staticmethod
    def _calc_margin(pos: Position) -> float:
        """Compute margin required for a single position."""
        notional = abs(pos.size * pos.entry_px)
        if pos.leverage > 0:
            return notional / pos.leverage
        return notional
