"""Postgres-backed implementation of :class:`~autotrader.store.datastore.DataStore`.

Requires ``psycopg2`` (or ``psycopg2-binary``).  If the driver is not
installed the module can still be imported, but instantiating
:class:`PostgresStore` will raise :class:`ImportError`.

Tables are created automatically on first use with ``ON CONFLICT DO NOTHING``
upserts so duplicate rows are silently ignored.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import structlog

from autotrader.store.datastore import DataStore

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# SQL DDL
# ---------------------------------------------------------------------------

_CREATE_CANDLES = """
CREATE TABLE IF NOT EXISTS candles (
    symbol      TEXT     NOT NULL,
    timeframe   TEXT     NOT NULL,
    timestamp_ms BIGINT  NOT NULL,
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (symbol, timeframe, timestamp_ms)
);
"""

_CREATE_FILLS = """
CREATE TABLE IF NOT EXISTS fills (
    timestamp_ms BIGINT  NOT NULL,
    oid          BIGINT  NOT NULL,
    px           DOUBLE PRECISION NOT NULL,
    sz           DOUBLE PRECISION NOT NULL,
    side         TEXT     NOT NULL,
    fee          DOUBLE PRECISION NOT NULL,
    symbol       TEXT     NOT NULL,
    PRIMARY KEY (symbol, oid)
);
"""

_CREATE_FUNDING = """
CREATE TABLE IF NOT EXISTS funding (
    symbol        TEXT     NOT NULL,
    timestamp_ms  BIGINT  NOT NULL,
    funding_rate  DOUBLE PRECISION NOT NULL,
    premium       DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (symbol, timestamp_ms)
);
"""


class PostgresStore(DataStore):
    """Postgres-backed storage backend.

    Parameters
    ----------
    dsn:
        A ``psycopg2``-compatible DSN string, e.g.
        ``"host=localhost dbname=autotrader user=postgres"``.

    Raises
    ------
    ImportError
        If ``psycopg2`` is not installed.
    """

    def __init__(self, dsn: str) -> None:
        try:
            import psycopg2  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "psycopg2 is required for PostgresStore. "
                "Install it with: pip install psycopg2-binary"
            ) from exc

        self._dsn = dsn
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = True
        self._ensure_tables()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_tables(self) -> None:
        """Create tables and performance indexes if they do not already exist."""
        with self._conn.cursor() as cur:
            cur.execute(_CREATE_CANDLES)
            cur.execute(_CREATE_FILLS)
            cur.execute(_CREATE_FUNDING)
            # Performance indexes for common query patterns
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_candles_symbol_ts "
                "ON candles (symbol, timestamp_ms)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_candles_timeframe "
                "ON candles (timeframe)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_fills_ts "
                "ON fills (timestamp_ms)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_fills_symbol "
                "ON fills (symbol)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_funding_symbol_ts "
                "ON funding (symbol, timestamp_ms)"
            )
        logger.info("postgres_tables_ensured")

    def _execute_many(self, sql: str, rows: list[tuple[Any, ...]]) -> int:
        """Execute *sql* for each row in *rows* and return the count of rows affected."""
        if not rows:
            return 0
        with self._conn.cursor() as cur:
            cur.executemany(sql, rows)
            # executemany does not reliably set rowcount for ON CONFLICT DO NOTHING,
            # so we return the number of input rows as an upper-bound estimate.
            return len(rows)

    def _query_df(self, sql: str, params: tuple[Any, ...], columns: list[str]) -> pd.DataFrame:
        """Execute a SELECT query and return results as a DataFrame."""
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            data = cur.fetchall()
        if not data:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(data, columns=columns)

    # ------------------------------------------------------------------
    # Candles
    # ------------------------------------------------------------------

    def write_candles(self, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        sql = """
            INSERT INTO candles (symbol, timeframe, timestamp_ms, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """
        rows = [
            (
                symbol,
                timeframe,
                int(r.timestamp_ms),
                float(r.open),
                float(r.high),
                float(r.low),
                float(r.close),
                float(r.volume),
            )
            for r in df.itertuples(index=False)
        ]
        return self._execute_many(sql, rows)

    def read_candles(self, symbol: str, timeframe: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        sql = """
            SELECT timestamp_ms, open, high, low, close, volume
            FROM candles
            WHERE symbol = %s AND timeframe = %s
              AND timestamp_ms >= %s AND timestamp_ms <= %s
            ORDER BY timestamp_ms
        """
        cols = ["timestamp_ms", "open", "high", "low", "close", "volume"]
        return self._query_df(sql, (symbol, timeframe, start_ms, end_ms), cols)

    # ------------------------------------------------------------------
    # Fills
    # ------------------------------------------------------------------

    def write_fills(self, fills: pd.DataFrame) -> int:
        if fills.empty:
            return 0
        sql = """
            INSERT INTO fills (timestamp_ms, oid, px, sz, side, fee, symbol)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """
        rows = [
            (
                int(r.timestamp_ms),
                int(r.oid),
                float(r.px),
                float(r.sz),
                str(r.side),
                float(r.fee),
                str(r.symbol),
            )
            for r in fills.itertuples(index=False)
        ]
        return self._execute_many(sql, rows)

    def read_fills(self, symbol: str | None, start_ms: int, end_ms: int) -> pd.DataFrame:
        cols = ["timestamp_ms", "oid", "px", "sz", "side", "fee", "symbol"]
        if symbol is not None:
            sql = """
                SELECT timestamp_ms, oid, px, sz, side, fee, symbol
                FROM fills
                WHERE symbol = %s AND timestamp_ms >= %s AND timestamp_ms <= %s
                ORDER BY timestamp_ms
            """
            return self._query_df(sql, (symbol, start_ms, end_ms), cols)
        else:
            sql = """
                SELECT timestamp_ms, oid, px, sz, side, fee, symbol
                FROM fills
                WHERE timestamp_ms >= %s AND timestamp_ms <= %s
                ORDER BY timestamp_ms
            """
            return self._query_df(sql, (start_ms, end_ms), cols)

    # ------------------------------------------------------------------
    # Funding
    # ------------------------------------------------------------------

    def write_funding(self, symbol: str, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        sql = """
            INSERT INTO funding (symbol, timestamp_ms, funding_rate, premium)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """
        rows = [
            (
                symbol,
                int(r.timestamp_ms),
                float(r.funding_rate),
                float(r.premium),
            )
            for r in df.itertuples(index=False)
        ]
        return self._execute_many(sql, rows)

    def read_funding(self, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        sql = """
            SELECT timestamp_ms, funding_rate, premium
            FROM funding
            WHERE symbol = %s AND timestamp_ms >= %s AND timestamp_ms <= %s
            ORDER BY timestamp_ms
        """
        cols = ["timestamp_ms", "funding_rate", "premium"]
        return self._query_df(sql, (symbol, start_ms, end_ms), cols)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_symbols(self, timeframe: str) -> list[str]:
        sql = """
            SELECT DISTINCT symbol FROM candles
            WHERE timeframe = %s
            ORDER BY symbol
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (timeframe,))
            return [row[0] for row in cur.fetchall()]
