"""User state collector -- positions, margin, and account value.

Fetches the clearinghouse state from Hyperliquid and normalises it into
a flat dictionary suitable for monitoring dashboards and risk checks.
"""

from __future__ import annotations

import structlog

from autotrader.hl.client import HLClient

logger = structlog.get_logger(__name__)


def fetch_user_state(client: HLClient, address: str | None = None) -> dict:
    """Fetch and parse the user's clearinghouse state.

    Parameters
    ----------
    client:
        Hyperliquid REST client.
    address:
        Ethereum address to query.  Falls back to the client's default
        ``account_address`` if not provided.

    Returns
    -------
    dict
        Parsed state containing:

        * ``account_value`` -- total account value (float)
        * ``total_margin_used`` -- margin currently in use (float)
        * ``total_ntl_pos`` -- total notional position value (float)
        * ``withdrawable`` -- withdrawable balance (float)
        * ``positions`` -- list of position dicts, each with:
          ``coin``, ``size``, ``entry_px``, ``mark_px``,
          ``unrealized_pnl``, ``leverage``, ``margin_used``,
          ``liquidation_px``
    """
    raw = client.get_user_state(address=address)

    # Parse margin summary
    margin = raw.get("marginSummary", {})
    account_value = float(margin.get("accountValue", 0.0))
    total_margin_used = float(margin.get("totalMarginUsed", 0.0))
    total_ntl_pos = float(margin.get("totalNtlPos", 0.0))

    # withdrawable may be at the top level or inside marginSummary
    withdrawable = float(raw.get("withdrawable", margin.get("withdrawable", 0.0)))

    # Parse positions
    asset_positions = raw.get("assetPositions", [])
    positions: list[dict] = []

    for ap in asset_positions:
        # Each element is either a position dict directly or wrapped in
        # {"type": "oneWay", "position": {...}}
        pos = ap.get("position", ap) if isinstance(ap, dict) else ap

        coin = pos.get("coin", "")
        szi = float(pos.get("szi", pos.get("size", 0.0)))
        entry_px = float(pos.get("entryPx", pos.get("entry_px", 0.0)))
        mark_px_val = float(pos.get("markPx", pos.get("mark_px", 0.0)))

        # Compute unrealized PnL: (mark - entry) * size
        if entry_px > 0 and mark_px_val > 0:
            unrealized_pnl = (mark_px_val - entry_px) * szi
        else:
            unrealized_pnl = float(pos.get("unrealizedPnl", 0.0))

        leverage_info = pos.get("leverage", {})
        if isinstance(leverage_info, dict):
            leverage_val = float(leverage_info.get("value", 0.0))
        else:
            leverage_val = float(leverage_info) if leverage_info else 0.0

        margin_used_val = float(pos.get("marginUsed", 0.0))
        liquidation_px = float(pos.get("liquidationPx", 0.0)) if pos.get("liquidationPx") else 0.0

        # Skip empty positions (zero size)
        if szi == 0.0:
            continue

        positions.append(
            {
                "coin": coin,
                "size": szi,
                "entry_px": entry_px,
                "mark_px": mark_px_val,
                "unrealized_pnl": round(unrealized_pnl, 6),
                "leverage": leverage_val,
                "margin_used": margin_used_val,
                "liquidation_px": liquidation_px,
            }
        )

    result = {
        "account_value": account_value,
        "total_margin_used": total_margin_used,
        "total_ntl_pos": total_ntl_pos,
        "withdrawable": withdrawable,
        "positions": positions,
    }

    logger.info(
        "fetch_user_state",
        account_value=account_value,
        total_margin_used=total_margin_used,
        num_positions=len(positions),
    )

    return result
