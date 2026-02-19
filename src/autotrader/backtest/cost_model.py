"""Realistic cost model for backtesting.

Models maker/taker fees, size-dependent slippage, spread costs, and
hourly funding payments to produce accurate PnL attribution during
backtests.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CostConfig:
    """Parameters controlling trade-cost estimation.

    Attributes
    ----------
    maker_fee_bps : float
        Maker fee in basis points (0.2 bps = 0.02 %).
    taker_fee_bps : float
        Taker fee in basis points (0.5 bps = 0.05 %).
    slippage_bps : float
        Base slippage in basis points (1.0 bps = 0.01 % of notional).
    slippage_size_factor : float
        Additional slippage per $100 000 of notional (in bps).
    funding_interval_hours : float
        Hyperliquid pays funding every hour by default.
    prefer_maker : bool
        Whether the system should try to use maker orders when possible.
    """

    maker_fee_bps: float = 0.2
    taker_fee_bps: float = 0.5
    slippage_bps: float = 1.0
    slippage_size_factor: float = 0.1
    funding_interval_hours: float = 1.0
    prefer_maker: bool = True


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


class CostModel:
    """Compute realistic trading costs for backtesting.

    Parameters
    ----------
    config : CostConfig | None
        Cost configuration.  Uses sensible Hyperliquid defaults when
        ``None``.
    """

    def __init__(self, config: CostConfig | None = None) -> None:
        self.config = config or CostConfig()

    # ------------------------------------------------------------------
    # Fee helpers
    # ------------------------------------------------------------------

    def _fee_bps_to_fraction(self, bps: float) -> float:
        """Convert basis-point fee to a decimal fraction (1 bps = 0.0001)."""
        return bps / 10_000.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_entry_cost(self, notional: float, is_maker: bool = True) -> float:
        """Compute the fee (in USD) for opening a position.

        Parameters
        ----------
        notional : float
            Position notional value in USD.
        is_maker : bool
            ``True`` if the entry uses a maker (limit) order.

        Returns
        -------
        float
            Fee amount in USD (always >= 0).
        """
        bps = self.config.maker_fee_bps if is_maker else self.config.taker_fee_bps
        return abs(notional) * self._fee_bps_to_fraction(bps)

    def compute_exit_cost(self, notional: float, is_maker: bool = False) -> float:
        """Compute the fee (in USD) for closing a position.

        Exits default to taker (market / stop / TP orders).

        Parameters
        ----------
        notional : float
            Position notional value in USD.
        is_maker : bool
            ``True`` if the exit uses a maker (limit) order.

        Returns
        -------
        float
            Fee amount in USD (always >= 0).
        """
        bps = self.config.maker_fee_bps if is_maker else self.config.taker_fee_bps
        return abs(notional) * self._fee_bps_to_fraction(bps)

    def compute_slippage(self, notional: float, spread_bps: float = 1.0) -> float:
        """Estimate execution slippage in USD.

        Combines three components:

        1. **Base slippage** -- a fixed cost in bps of notional.
        2. **Size-dependent slippage** -- additional cost that grows
           linearly with notional (per $100 000 chunk).
        3. **Spread cost** -- half-spread impact in bps.

        Parameters
        ----------
        notional : float
            Trade notional in USD.
        spread_bps : float
            Observed bid-ask spread in basis points.

        Returns
        -------
        float
            Estimated slippage in USD (always >= 0).
        """
        abs_notional = abs(notional)

        # 1. Base slippage
        base = abs_notional * self._fee_bps_to_fraction(self.config.slippage_bps)

        # 2. Size-dependent slippage (linear in number of $100k chunks)
        size_chunks = abs_notional / 100_000.0
        size_slip = abs_notional * self._fee_bps_to_fraction(
            self.config.slippage_size_factor * size_chunks
        )

        # 3. Spread component -- assume crossing half the spread
        spread_cost = abs_notional * self._fee_bps_to_fraction(spread_bps / 2.0)

        return base + size_slip + spread_cost

    def compute_funding_cost(
        self,
        notional: float,
        funding_rate: float,
        hours_held: float,
    ) -> float:
        """Compute cumulative funding cost over the holding period.

        Hyperliquid settles funding every ``funding_interval_hours`` (default
        1 hour).  Each settlement pays ``notional * funding_rate``.

        Parameters
        ----------
        notional : float
            Position notional in USD (positive for long, negative for short
            is fine -- the sign is preserved).
        funding_rate : float
            Per-interval funding rate (e.g. 0.0001 = 0.01 %).  Positive
            means longs pay shorts.
        hours_held : float
            Number of hours the position was held.

        Returns
        -------
        float
            Total funding cost in USD.  Positive = cost to the trader
            (longs when rate > 0), negative = received.
        """
        if self.config.funding_interval_hours <= 0:
            return 0.0

        num_payments = hours_held / self.config.funding_interval_hours
        return num_payments * notional * funding_rate

    def total_trade_cost(
        self,
        notional: float,
        funding_rate: float,
        hours_held: float,
        spread_bps: float = 1.0,
    ) -> dict:
        """Compute a full cost breakdown for a round-trip trade.

        Parameters
        ----------
        notional : float
            Position notional in USD.
        funding_rate : float
            Per-interval funding rate.
        hours_held : float
            Hours the position is held.
        spread_bps : float
            Observed bid-ask spread in bps.

        Returns
        -------
        dict
            ``{entry_fee, exit_fee, slippage, funding, total}`` -- all in
            USD.
        """
        is_maker_entry = self.config.prefer_maker
        entry_fee = self.compute_entry_cost(notional, is_maker=is_maker_entry)
        exit_fee = self.compute_exit_cost(notional, is_maker=False)
        slippage = self.compute_slippage(notional, spread_bps)
        funding = self.compute_funding_cost(notional, funding_rate, hours_held)

        return {
            "entry_fee": entry_fee,
            "exit_fee": exit_fee,
            "slippage": slippage,
            "funding": funding,
            "total": entry_fee + exit_fee + slippage + funding,
        }
