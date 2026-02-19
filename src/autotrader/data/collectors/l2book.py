"""L2 order-book snapshot collector.

Fetches the current L2 book from Hyperliquid and computes derived
microstructure metrics (spread, mid-price, depth).
"""

from __future__ import annotations

import structlog

from autotrader.hl.client import HLClient

logger = structlog.get_logger(__name__)


def _parse_levels(raw_levels: list) -> list[dict]:
    """Convert raw L2 levels from the API into dicts with ``px`` and ``sz``.

    Each raw level is either a dict ``{"px": ..., "sz": ..., "n": ...}``
    or a list/tuple ``[px, sz, n_orders]``.
    """
    parsed: list[dict] = []
    for lvl in raw_levels:
        if isinstance(lvl, dict):
            parsed.append(
                {
                    "px": float(lvl.get("px", 0.0)),
                    "sz": float(lvl.get("sz", 0.0)),
                    "n": int(lvl.get("n", 0)),
                }
            )
        elif isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
            parsed.append(
                {
                    "px": float(lvl[0]),
                    "sz": float(lvl[1]),
                    "n": int(lvl[2]) if len(lvl) > 2 else 0,
                }
            )
    return parsed


def snapshot_l2(client: HLClient, coin: str) -> dict:
    """Fetch and parse the current L2 order-book snapshot.

    Parameters
    ----------
    client:
        Hyperliquid REST client.
    coin:
        Asset symbol, e.g. ``"ETH"``.

    Returns
    -------
    dict
        Parsed book with keys:

        * ``bids`` -- list of ``{"px", "sz", "n"}`` dicts (best bid first)
        * ``asks`` -- list of ``{"px", "sz", "n"}`` dicts (best ask first)
        * ``mid_px`` -- mid-price (average of best bid and best ask)
        * ``spread_bps`` -- spread in basis points
        * ``bid_depth_usd`` -- USD depth of the top 5 bid levels
        * ``ask_depth_usd`` -- USD depth of the top 5 ask levels
    """
    raw = client.get_l2_book(coin)

    # The response wraps levels under a "levels" key: [[bids], [asks]]
    levels = raw.get("levels", [[], []])
    raw_bids = levels[0] if len(levels) > 0 else []
    raw_asks = levels[1] if len(levels) > 1 else []

    bids = _parse_levels(raw_bids)
    asks = _parse_levels(raw_asks)

    # Compute derived metrics
    best_bid = bids[0]["px"] if bids else 0.0
    best_ask = asks[0]["px"] if asks else 0.0

    if best_bid > 0 and best_ask > 0:
        mid_px = (best_bid + best_ask) / 2.0
        spread_bps = ((best_ask - best_bid) / mid_px) * 10_000
    else:
        mid_px = best_bid or best_ask
        spread_bps = 0.0

    # Top-5 depth in USD (price * size)
    top_n = 5
    bid_depth_usd = sum(b["px"] * b["sz"] for b in bids[:top_n])
    ask_depth_usd = sum(a["px"] * a["sz"] for a in asks[:top_n])

    result = {
        "bids": bids,
        "asks": asks,
        "mid_px": mid_px,
        "spread_bps": round(spread_bps, 4),
        "bid_depth_usd": round(bid_depth_usd, 2),
        "ask_depth_usd": round(ask_depth_usd, 2),
    }

    logger.debug(
        "snapshot_l2",
        coin=coin,
        mid_px=result["mid_px"],
        spread_bps=result["spread_bps"],
        bid_depth_usd=result["bid_depth_usd"],
        ask_depth_usd=result["ask_depth_usd"],
        n_bids=len(bids),
        n_asks=len(asks),
    )

    return result
