"""Candle collector -- bootstrap historical data and poll for new bars.

Uses the Hyperliquid ``candleSnapshot`` endpoint via :class:`HLClient` and
persists results through any :class:`DataStore` backend.
"""

from __future__ import annotations

import pandas as pd
import structlog

from autotrader.hl.client import HLClient
from autotrader.store.datastore import DataStore
from autotrader.utils.time import interval_ms, now_ms

logger = structlog.get_logger(__name__)

# The HL API returns at most 5000 candles per request.
_HL_CANDLE_LIMIT = 5000


def _raw_candles_to_df(raw: list[dict]) -> pd.DataFrame:
    """Convert raw HL API candle dicts to a typed DataFrame.

    Each raw dict has keys: ``t`` (open timestamp ms), ``T`` (close timestamp ms),
    ``o``, ``h``, ``l``, ``c``, ``v``.  Some responses use full names
    (``open``, ``high``, ...).  This function handles both formats.
    """
    if not raw:
        return pd.DataFrame(columns=["timestamp_ms", "open", "high", "low", "close", "volume"])

    rows: list[dict] = []
    for c in raw:
        rows.append(
            {
                "timestamp_ms": int(c.get("t", c.get("timestamp_ms", c.get("T", 0)))),
                "open": float(c.get("o", c.get("open", 0.0))),
                "high": float(c.get("h", c.get("high", 0.0))),
                "low": float(c.get("l", c.get("low", 0.0))),
                "close": float(c.get("c", c.get("close", 0.0))),
                "volume": float(c.get("v", c.get("volume", 0.0))),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    return df


def bootstrap_candles(
    client: HLClient,
    store: DataStore,
    coin: str,
    interval: str,
    num_candles: int = 5000,
) -> int:
    """Fetch historical candles and write them to *store*.

    If *num_candles* exceeds the per-request limit (5000) the function
    paginates backwards from the current time, making as many requests as
    necessary.

    Parameters
    ----------
    client:
        Hyperliquid REST client.
    store:
        Storage backend.
    coin:
        Asset symbol, e.g. ``"ETH"``.
    interval:
        Bar interval string, e.g. ``"15m"``.
    num_candles:
        Total number of candles to fetch (approximate -- the API may return
        slightly fewer on the final page).

    Returns
    -------
    int
        Number of rows written to the store.
    """
    bar_ms = interval_ms(interval)
    end_ms = now_ms()
    total_written = 0
    remaining = num_candles

    while remaining > 0:
        # Each request fetches up to _HL_CANDLE_LIMIT candles.
        batch_size = min(remaining, _HL_CANDLE_LIMIT)
        start_ms = end_ms - (batch_size * bar_ms)

        logger.info(
            "bootstrap_candles_fetch",
            coin=coin,
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
            batch_size=batch_size,
        )

        raw = client.get_candle_snapshot(
            coin=coin,
            interval=interval,
            start_time=start_ms,
            end_time=end_ms,
        )
        df = _raw_candles_to_df(raw)

        if df.empty:
            logger.warning(
                "bootstrap_candles_empty",
                coin=coin,
                interval=interval,
                start_ms=start_ms,
                end_ms=end_ms,
            )
            break

        written = store.write_candles(coin, interval, df)
        total_written += written

        # Move the window backwards for the next page
        earliest = int(df["timestamp_ms"].iloc[0])
        end_ms = earliest - 1
        remaining -= len(df)

        # If we received fewer candles than expected the history is exhausted
        if len(df) < batch_size:
            break

    logger.info(
        "bootstrap_candles_done",
        coin=coin,
        interval=interval,
        total_written=total_written,
    )
    return total_written


def stream_candles_to_store(
    client: HLClient,
    store: DataStore,
    coins: list[str],
    interval: str,
) -> None:
    """Poll for new candles and append them to the store.

    This function is designed to be called periodically (e.g. by a scheduler
    or cron job).  For each coin it determines the latest stored timestamp,
    fetches any new candles since then, and writes them.

    Parameters
    ----------
    client:
        Hyperliquid REST client.
    store:
        Storage backend.
    coins:
        List of asset symbols to update.
    interval:
        Bar interval string.
    """
    bar_ms = interval_ms(interval)
    current_ms = now_ms()

    for coin in coins:
        # Determine the latest stored candle for this coin/interval.
        # Read a small window backwards from now to find the max timestamp.
        existing = store.read_candles(coin, interval, start_ms=0, end_ms=current_ms)

        if existing.empty:
            # No data yet -- fetch a small initial batch
            start_ms = current_ms - (_HL_CANDLE_LIMIT * bar_ms)
        else:
            # Start from the bar after the last stored candle
            start_ms = int(existing["timestamp_ms"].max()) + 1

        if start_ms >= current_ms:
            logger.debug("stream_candles_up_to_date", coin=coin, interval=interval)
            continue

        logger.info(
            "stream_candles_fetch",
            coin=coin,
            interval=interval,
            start_ms=start_ms,
            end_ms=current_ms,
        )

        raw = client.get_candle_snapshot(
            coin=coin,
            interval=interval,
            start_time=start_ms,
            end_time=current_ms,
        )
        df = _raw_candles_to_df(raw)

        if df.empty:
            logger.debug("stream_candles_no_new", coin=coin, interval=interval)
            continue

        written = store.write_candles(coin, interval, df)
        logger.info(
            "stream_candles_written",
            coin=coin,
            interval=interval,
            rows=written,
        )
