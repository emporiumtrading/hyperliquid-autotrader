"""Funding rate and open interest collectors.

Fetches historical funding via ``fundingHistory`` and live asset context
snapshots via ``metaAndAssetCtxs``.
"""

from __future__ import annotations

import pandas as pd
import structlog

from autotrader.hl.client import HLClient
from autotrader.store.datastore import DataStore
from autotrader.utils.time import now_ms

logger = structlog.get_logger(__name__)


def fetch_funding_rates(
    client: HLClient,
    store: DataStore,
    coin: str,
    start_ms: int,
    end_ms: int | None = None,
) -> int:
    """Fetch historical funding rates and persist to *store*.

    Parameters
    ----------
    client:
        Hyperliquid REST client.
    store:
        Storage backend.
    coin:
        Asset symbol, e.g. ``"ETH"``.
    start_ms:
        Start timestamp in milliseconds.
    end_ms:
        End timestamp in milliseconds.  Defaults to the current time.

    Returns
    -------
    int
        Number of rows written to the store.
    """
    if end_ms is None:
        end_ms = now_ms()

    logger.info(
        "fetch_funding_rates",
        coin=coin,
        start_ms=start_ms,
        end_ms=end_ms,
    )

    raw = client.get_funding_history(coin=coin, start_time=start_ms, end_time=end_ms)

    if not raw:
        logger.warning("fetch_funding_rates_empty", coin=coin)
        return 0

    rows: list[dict] = []
    for entry in raw:
        rows.append(
            {
                "timestamp_ms": int(entry.get("time", entry.get("timestamp_ms", 0))),
                "funding_rate": float(entry.get("fundingRate", entry.get("funding_rate", 0.0))),
                "premium": float(entry.get("premium", 0.0)),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp_ms").reset_index(drop=True)

    written = store.write_funding(coin, df)
    logger.info(
        "fetch_funding_rates_done",
        coin=coin,
        rows=written,
    )
    return written


def fetch_asset_contexts(client: HLClient) -> pd.DataFrame:
    """Fetch live asset contexts (funding, OI, mark price, volume).

    Calls ``metaAndAssetCtxs`` and parses the response into a DataFrame.

    Parameters
    ----------
    client:
        Hyperliquid REST client.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``name``, ``funding``, ``open_interest``,
        ``mark_px``, ``day_ntl_vlm``.
    """
    raw = client.get_meta_and_asset_ctxs()

    # The response is a list of two elements:
    #   [0] = meta (contains "universe" list of asset metadata)
    #   [1] = list of asset context dicts
    if not raw or len(raw) < 2:
        logger.warning("fetch_asset_contexts_empty")
        return pd.DataFrame(columns=["name", "funding", "open_interest", "mark_px", "day_ntl_vlm"])

    meta_universe = raw[0].get("universe", [])
    asset_ctxs = raw[1]

    rows: list[dict] = []
    for i, ctx in enumerate(asset_ctxs):
        # Match the context to its metadata entry by index
        name = meta_universe[i]["name"] if i < len(meta_universe) else f"UNKNOWN_{i}"
        rows.append(
            {
                "name": name,
                "funding": float(ctx.get("funding", 0.0)),
                "open_interest": float(ctx.get("openInterest", ctx.get("open_interest", 0.0))),
                "mark_px": float(ctx.get("markPx", ctx.get("mark_px", 0.0))),
                "day_ntl_vlm": float(ctx.get("dayNtlVlm", ctx.get("day_ntl_vlm", 0.0))),
            }
        )

    df = pd.DataFrame(rows)
    logger.info("fetch_asset_contexts_done", num_assets=len(df))
    return df
