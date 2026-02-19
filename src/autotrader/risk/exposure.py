"""Exposure tracking for open positions.

Maintains a live registry of :class:`Position` objects, recalculates
unrealised PnL on price updates, and exposes aggregate portfolio metrics
such as total notional, net/gross exposure, and margin usage.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

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


class ExposureTracker:
    """Maintains a set of open positions and computes aggregate exposures."""

    def __init__(self) -> None:
        self.positions: dict[str, Position] = {}
        self._lock = threading.Lock()

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
            pos.current_px = price
            pos.unrealized_pnl = self._calc_unrealized_pnl(pos)
            pos.margin_used = self._calc_margin(pos)

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
        """Check if the dominant direction exceeds *max_cluster_pct* of gross.

        Correlated positions (all longs or all shorts) shouldn't dominate
        the book.  This is a heuristic: if > 60% of gross exposure is in
        one direction, we consider the portfolio excessively correlated.

        Returns
        -------
        tuple[bool, float]
            ``(is_exceeded, dominant_pct)``
        """
        with self._lock:
            long_notional = sum(
                abs(p.size * p.current_px)
                for p in self.positions.values()
                if p.side == "long"
            )
            short_notional = sum(
                abs(p.size * p.current_px)
                for p in self.positions.values()
                if p.side == "short"
            )

        gross = long_notional + short_notional
        if gross <= 0:
            return False, 0.0

        dominant_pct = max(long_notional, short_notional) / gross
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
