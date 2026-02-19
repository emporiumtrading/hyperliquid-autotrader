"""Parquet-based implementation of :class:`~autotrader.store.datastore.DataStore`.

Data is organised on disk as::

    <base_dir>/
        candles/<SYMBOL>/<timeframe>.parquet
        fills/<SYMBOL>.parquet
        funding/<SYMBOL>.parquet

All files use ``pyarrow`` for I/O.  Writes are append-friendly: existing
data is read, concatenated with new rows, deduplicated by ``timestamp_ms``,
sorted, and rewritten atomically.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from autotrader.store.datastore import DataStore

logger = structlog.get_logger(__name__)

# Canonical column lists for empty-DataFrame construction
_CANDLE_COLS: dict[str, str] = {
    "timestamp_ms": "int64",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
}

_FILL_COLS: dict[str, str] = {
    "timestamp_ms": "int64",
    "oid": "int64",
    "px": "float64",
    "sz": "float64",
    "side": "object",
    "fee": "float64",
    "symbol": "object",
}

_FUNDING_COLS: dict[str, str] = {
    "timestamp_ms": "int64",
    "funding_rate": "float64",
    "premium": "float64",
}


def _empty_df(cols: dict[str, str]) -> pd.DataFrame:
    """Return an empty DataFrame with the given column dtypes."""
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in cols.items()})


class ParquetStore(DataStore):
    """Parquet file-based storage backend.

    Parameters
    ----------
    base_dir:
        Root directory for all stored data.  Defaults to ``"data/processed"``.
    """

    def __init__(self, base_dir: str = "data/processed") -> None:
        self._base = Path(base_dir)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _path(self, kind: str, symbol: str, timeframe: str = "") -> Path:
        """Build the on-disk path for a given data kind, symbol, and timeframe.

        Examples
        --------
        >>> store = ParquetStore("/tmp/data")
        >>> store._path("candles", "ETH", "15m")
        PosixPath('/tmp/data/candles/ETH/15m.parquet')
        >>> store._path("fills", "ETH")
        PosixPath('/tmp/data/fills/ETH.parquet')
        """
        if timeframe:
            return self._base / kind / symbol / f"{timeframe}.parquet"
        return self._base / kind / f"{symbol}.parquet"

    # ------------------------------------------------------------------
    # Internal read/merge helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_parquet_safe(path: Path, cols: dict[str, str]) -> pd.DataFrame:
        """Read a parquet file, returning an empty typed DataFrame if missing."""
        if not path.exists():
            return _empty_df(cols)
        try:
            return pq.read_table(path).to_pandas()
        except Exception:
            logger.warning("parquet_read_error", path=str(path), exc_info=True)
            return _empty_df(cols)

    @staticmethod
    def _write_parquet(df: pd.DataFrame, path: Path) -> None:
        """Write *df* to *path*, creating parent directories as needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, path, compression="snappy")

    def _merge_write(
        self,
        path: Path,
        new_df: pd.DataFrame,
        dedup_col: str,
        cols: dict[str, str],
    ) -> int:
        """Read existing parquet, concat *new_df*, dedup, sort, and rewrite.

        Returns the number of rows written.
        """
        if new_df.empty:
            return 0

        existing = self._read_parquet_safe(path, cols)
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=[dedup_col], keep="last")
        merged = merged.sort_values(dedup_col).reset_index(drop=True)

        self._write_parquet(merged, path)
        rows_written = len(merged) - len(existing)
        logger.debug(
            "parquet_merge_write",
            path=str(path),
            existing=len(existing),
            new=len(new_df),
            written=rows_written,
        )
        return max(rows_written, 0)

    # ------------------------------------------------------------------
    # Candles
    # ------------------------------------------------------------------

    def write_candles(self, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
        path = self._path("candles", symbol, timeframe)
        return self._merge_write(path, df, "timestamp_ms", _CANDLE_COLS)

    def read_candles(self, symbol: str, timeframe: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        path = self._path("candles", symbol, timeframe)
        df = self._read_parquet_safe(path, _CANDLE_COLS)
        if df.empty:
            return df
        mask = (df["timestamp_ms"] >= start_ms) & (df["timestamp_ms"] <= end_ms)
        return df.loc[mask].sort_values("timestamp_ms").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Fills
    # ------------------------------------------------------------------

    def write_fills(self, fills: pd.DataFrame) -> int:
        if fills.empty:
            return 0

        total = 0
        for symbol, group in fills.groupby("symbol"):
            path = self._path("fills", str(symbol))
            total += self._merge_write(path, group, "oid", _FILL_COLS)
        return total

    def read_fills(self, symbol: str | None, start_ms: int, end_ms: int) -> pd.DataFrame:
        if symbol is not None:
            path = self._path("fills", symbol)
            df = self._read_parquet_safe(path, _FILL_COLS)
        else:
            # Read all fill files and concatenate
            fills_dir = self._base / "fills"
            if not fills_dir.exists():
                return _empty_df(_FILL_COLS)
            frames: list[pd.DataFrame] = []
            for p in fills_dir.glob("*.parquet"):
                frames.append(self._read_parquet_safe(p, _FILL_COLS))
            if not frames:
                return _empty_df(_FILL_COLS)
            df = pd.concat(frames, ignore_index=True)

        if df.empty:
            return df
        mask = (df["timestamp_ms"] >= start_ms) & (df["timestamp_ms"] <= end_ms)
        return df.loc[mask].sort_values("timestamp_ms").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Funding
    # ------------------------------------------------------------------

    def write_funding(self, symbol: str, df: pd.DataFrame) -> int:
        path = self._path("funding", symbol)
        return self._merge_write(path, df, "timestamp_ms", _FUNDING_COLS)

    def read_funding(self, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        path = self._path("funding", symbol)
        df = self._read_parquet_safe(path, _FUNDING_COLS)
        if df.empty:
            return df
        mask = (df["timestamp_ms"] >= start_ms) & (df["timestamp_ms"] <= end_ms)
        return df.loc[mask].sort_values("timestamp_ms").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_symbols(self, timeframe: str) -> list[str]:
        candles_dir = self._base / "candles"
        if not candles_dir.exists():
            return []
        symbols: list[str] = []
        for d in sorted(candles_dir.iterdir()):
            if d.is_dir() and (d / f"{timeframe}.parquet").exists():
                symbols.append(d.name)
        return symbols
