"""Base strategy interface and MarketContext.

Provides the :class:`MarketContext` dataclass (everything a strategy needs),
the :class:`IStrategy` protocol, and the :class:`BaseStrategy` default
implementation.

The :class:`Signal` type is imported from :mod:`autotrader.hl.types` so that
the rest of the system uses a single canonical definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from autotrader.hl.types import Signal

# ---------------------------------------------------------------------------
# Market context
# ---------------------------------------------------------------------------


@dataclass
class MarketContext:
    """Everything a strategy needs to produce a signal.

    Attributes
    ----------
    symbol : str
        The trading pair / asset symbol (e.g. ``"ETH"``).
    timeframe : str
        Candle timeframe string (e.g. ``"5m"``, ``"1h"``).
    candles : pd.DataFrame
        OHLCV DataFrame with at least columns
        ``open, high, low, close, volume`` and optionally ``timestamp_ms``.
    features : dict
        Pre-computed feature values keyed by name (e.g. ``"adx"``, ``"rsi"``,
        ``"atr"``, ``"sma_50"``, ``"bb_upper"``, ``"bb_lower"``, etc.).
    regime : str
        Current effective regime label.
    regime_confidence : float
        Confidence of the regime classification.
    book : dict | None
        Optional L2 order book snapshot.
    funding_rate : float | None
        Current funding rate for the perpetual.
    open_interest : float | None
        Current open interest value.
    account_equity : float
        Account equity in USD, used by risk-sizing helpers.
    """

    symbol: str
    timeframe: str
    candles: pd.DataFrame
    features: dict
    regime: str
    regime_confidence: float
    book: dict | None = None
    funding_rate: float | None = None
    open_interest: float | None = None
    account_equity: float = 10_000.0


# ---------------------------------------------------------------------------
# Flat signal helper
# ---------------------------------------------------------------------------

_FLAT_SIGNAL = Signal(
    side="flat",
    entry=None,
    stop=None,
    take_profit=None,
    confidence=0.0,
    metadata={},
)


def flat_signal() -> Signal:
    """Return a frozen flat (no-op) signal."""
    return _FLAT_SIGNAL


# ---------------------------------------------------------------------------
# Strategy protocol
# ---------------------------------------------------------------------------


class IStrategy(Protocol):
    """Protocol that all strategies must satisfy."""

    name: str

    def applicable_regimes(self) -> list[str]: ...

    def compute_signal(self, ctx: MarketContext) -> Signal: ...

    def invariants_ok(self, signal: Signal) -> bool: ...


# ---------------------------------------------------------------------------
# Base implementation
# ---------------------------------------------------------------------------


class BaseStrategy:
    """Concrete base implementing :class:`IStrategy`.

    Override in concrete strategy modules.  Subclasses **must** set the
    ``name`` class attribute and implement :meth:`compute_signal` and
    :meth:`applicable_regimes`.
    """

    name: str = "base"

    def applicable_regimes(self) -> list[str]:
        """Return the list of regime labels this strategy is designed for."""
        return []

    def compute_signal(self, ctx: MarketContext) -> Signal:
        """Produce a trading signal given the market context.

        The default implementation returns a flat (no-trade) signal.
        """
        return flat_signal()

    def invariants_ok(self, signal: Signal) -> bool:
        """Validate that a non-flat signal satisfies basic sanity checks.

        Checks
        ------
        1. If side is not ``"flat"``, entry / stop / take_profit must all be
           present (not ``None``).
        2. Stop must be on the correct side of entry (below for long, above
           for short).
        3. Confidence must be in [0, 1].

        Returns ``True`` when all checks pass.
        """
        if signal.side == "flat":
            return True

        # 1. Non-flat signals require all price levels.
        if signal.entry is None or signal.stop is None or signal.take_profit is None:
            return False

        # 2. Stop on correct side.
        if signal.side == "long":
            if signal.stop >= signal.entry:
                return False
            if signal.take_profit <= signal.entry:
                return False
        elif signal.side == "short":
            if signal.stop <= signal.entry:
                return False
            if signal.take_profit >= signal.entry:
                return False
        else:
            # Unrecognised side string.
            return False

        # 3. Confidence range.
        if not (0.0 <= signal.confidence <= 1.0):
            return False

        return True
