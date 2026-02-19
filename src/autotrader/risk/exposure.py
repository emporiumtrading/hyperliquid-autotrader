"""Exposure tracking for open positions.

Maintains a live registry of :class:`Position` objects, recalculates
unrealised PnL on price updates, and exposes aggregate portfolio metrics
such as total notional, net/gross exposure, and margin usage.
"""

from __future__ import annotations

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

    # -- mutators -----------------------------------------------------------

    def add_position(self, pos: Position) -> None:
        """Register a new position (or overwrite if same symbol exists)."""
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
        removed = self.positions.pop(symbol, None)
        if removed is not None:
            logger.info("position_removed", symbol=symbol)
        else:
            logger.debug("position_remove_noop", symbol=symbol)

    def update_price(self, symbol: str, price: float) -> None:
        """Update the mark price for *symbol* and recalculate PnL/margin."""
        pos = self.positions.get(symbol)
        if pos is None:
            return
        pos.current_px = price
        pos.unrealized_pnl = self._calc_unrealized_pnl(pos)
        pos.margin_used = self._calc_margin(pos)

    # -- aggregation --------------------------------------------------------

    def total_notional(self) -> float:
        """Sum of ``abs(size * current_px)`` for every open position."""
        return sum(abs(p.size * p.current_px) for p in self.positions.values())

    def total_unrealized_pnl(self) -> float:
        """Sum of unrealised PnL across all open positions."""
        return sum(p.unrealized_pnl for p in self.positions.values())

    def total_margin_used(self) -> float:
        """Sum of margin committed across all open positions."""
        return sum(p.margin_used for p in self.positions.values())

    def net_exposure(self) -> float:
        """Signed exposure: long notional minus short notional."""
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
        return len(self.positions)

    def get_position(self, symbol: str) -> Position | None:
        """Return the :class:`Position` for *symbol*, or ``None``."""
        return self.positions.get(symbol)

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
