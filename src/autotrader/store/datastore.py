"""Abstract datastore interface for candles, fills, and funding data.

All concrete backends (Parquet, Postgres, etc.) implement :class:`DataStore`
so that collectors, strategies, and analysis code can remain storage-agnostic.

DataFrame schemas
-----------------
candles : timestamp_ms (int64), open (float64), high (float64),
          low (float64), close (float64), volume (float64)
fills   : timestamp_ms (int64), oid (int64), px (float64), sz (float64),
          side (str), fee (float64), symbol (str)
funding : timestamp_ms (int64), funding_rate (float64), premium (float64)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class DataStore(ABC):
    """Abstract base class that every storage backend must implement."""

    # ------------------------------------------------------------------
    # Candles
    # ------------------------------------------------------------------

    @abstractmethod
    def write_candles(self, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
        """Persist candle rows and return the number of rows written.

        Parameters
        ----------
        symbol:
            Asset symbol, e.g. ``"ETH"``.
        timeframe:
            Bar interval string, e.g. ``"15m"``.
        df:
            DataFrame following the *candles* schema.

        Returns
        -------
        int
            Number of rows actually written (after deduplication).
        """

    @abstractmethod
    def read_candles(self, symbol: str, timeframe: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        """Read candle rows within a timestamp range (inclusive on both ends).

        Returns an empty DataFrame with the correct columns when no data is
        available.
        """

    # ------------------------------------------------------------------
    # Fills
    # ------------------------------------------------------------------

    @abstractmethod
    def write_fills(self, fills: pd.DataFrame) -> int:
        """Persist fill rows and return the number of rows written.

        The DataFrame must contain a ``symbol`` column so fills can be stored
        per-asset.
        """

    @abstractmethod
    def read_fills(self, symbol: str | None, start_ms: int, end_ms: int) -> pd.DataFrame:
        """Read fill rows within a timestamp range.

        If *symbol* is ``None``, return fills for all symbols.
        """

    # ------------------------------------------------------------------
    # Funding
    # ------------------------------------------------------------------

    @abstractmethod
    def write_funding(self, symbol: str, df: pd.DataFrame) -> int:
        """Persist funding rows and return the number of rows written."""

    @abstractmethod
    def read_funding(self, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        """Read funding rows within a timestamp range."""

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @abstractmethod
    def list_symbols(self, timeframe: str) -> list[str]:
        """Return the list of symbols that have stored candle data for
        the given *timeframe*.
        """
