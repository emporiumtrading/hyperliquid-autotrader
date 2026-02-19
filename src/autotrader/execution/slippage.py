"""Slippage estimation and prediction.

Provides :class:`SlippageModel` which estimates execution slippage based on
order notional size, current spread, and order-book depth.  Used by the paper
broker to simulate realistic fills and by the risk layer to compute
worst-case execution costs.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)

# Maximum slippage cap in basis points.
_MAX_SLIPPAGE_BPS: float = 50.0


class SlippageModel:
    """Estimate expected slippage for an order.

    The model combines four additive components:

    1. **Base slippage** -- a constant floor representing minimum market-impact
       even for the smallest orders.
    2. **Size impact** -- linear cost that grows with notional value,
       reflecting the price impact of moving through the book.
    3. **Spread component** -- half the bid/ask spread, since a market order
       crosses at least half the spread.
    4. **Depth penalty** -- an extra charge when the order is large relative
       to visible book depth, indicating potential hidden liquidity gaps.

    Parameters
    ----------
    base_bps:
        Minimum expected slippage in basis points.
    size_impact_factor:
        Additional bps per $100k of notional size.
    """

    def __init__(
        self,
        base_bps: float = 1.0,
        size_impact_factor: float = 0.1,
    ) -> None:
        self.base_bps = base_bps
        self.size_impact_factor = size_impact_factor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(
        self,
        notional: float,
        spread_bps: float = 1.0,
        depth_usd: float = 1_000_000.0,
    ) -> float:
        """Estimate slippage in **USD** for a given notional order size.

        Parameters
        ----------
        notional:
            Order notional value in USD.
        spread_bps:
            Current bid-ask spread in basis points.
        depth_usd:
            Visible order-book depth in USD (both sides combined).

        Returns
        -------
        float
            Estimated slippage cost in USD.
        """
        total_bps = self._compute_bps(notional, spread_bps, depth_usd)
        slippage_usd = notional * total_bps / 10_000.0
        return slippage_usd

    def estimate_bps(
        self,
        notional: float,
        spread_bps: float = 1.0,
        depth_usd: float = 1_000_000.0,
    ) -> float:
        """Estimate slippage in **basis points** for a given notional order size.

        Parameters are the same as :meth:`estimate`.

        Returns
        -------
        float
            Estimated slippage in basis points, clamped to a maximum of 50 bps.
        """
        return self._compute_bps(notional, spread_bps, depth_usd)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_bps(
        self,
        notional: float,
        spread_bps: float,
        depth_usd: float,
    ) -> float:
        """Compute total slippage in bps from the four components."""
        # 1. Base slippage
        total = self.base_bps

        # 2. Size impact: linear in $100k units
        size_impact = (notional / 100_000.0) * self.size_impact_factor
        total += size_impact

        # 3. Spread component: half the spread
        spread_component = spread_bps / 2.0
        total += spread_component

        # 4. Depth penalty: triggered when order exceeds 10% of visible depth
        if depth_usd > 0 and notional > depth_usd * 0.1:
            # Quadratic penalty scaled by how much the order exceeds the
            # depth threshold.  At notional == depth_usd, penalty is ~5 bps.
            excess_ratio = (notional - depth_usd * 0.1) / depth_usd
            depth_penalty = excess_ratio * 5.0
            total += depth_penalty

        # Clamp to maximum
        total = min(total, _MAX_SLIPPAGE_BPS)

        logger.debug(
            "slippage_estimate",
            notional=notional,
            spread_bps=spread_bps,
            depth_usd=depth_usd,
            estimated_bps=round(total, 4),
        )

        return total
