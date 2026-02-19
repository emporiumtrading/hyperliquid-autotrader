"""Request/response types for the Hyperliquid API.

All data structures are plain dataclasses so they work with
:func:`dataclasses.asdict` and the project's JSON serialisation helpers.
Frozen dataclasses are used for value-object types that should be immutable
after creation (candle data, L2 levels, fills, signals).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CandleData:
    """A single OHLCV candle bar."""

    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


# ---------------------------------------------------------------------------
# Asset metadata & context
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AssetMeta:
    """Static metadata for a perpetual asset."""

    name: str
    sz_decimals: int
    max_leverage: int


@dataclass(slots=True)
class AssetCtx:
    """Live context snapshot for an asset (funding, OI, mark price, etc.)."""

    funding: float
    open_interest: float
    mark_px: float
    day_ntl_vlm: float
    prev_day_px: float


# ---------------------------------------------------------------------------
# Order book
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class L2Level:
    """A single price level in an L2 order book."""

    px: float
    sz: float
    n_orders: int


@dataclass(slots=True)
class L2Book:
    """L2 order book snapshot."""

    bids: list[L2Level]
    asks: list[L2Level]
    timestamp_ms: int


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OrderSpec:
    """Specification for placing a new order on Hyperliquid."""

    asset: int
    is_buy: bool
    limit_px: float
    sz: float
    order_type: str
    reduce_only: bool
    tif: str = "Gtc"


@dataclass(slots=True)
class OrderResponse:
    """Response received after submitting an order."""

    status: str
    data: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fills
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Fill:
    """A single fill (execution) record."""

    oid: int
    px: float
    sz: float
    side: str
    timestamp_ms: int
    fee: float
    crossed: bool


# ---------------------------------------------------------------------------
# User state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class UserState:
    """Snapshot of a user's margin summary and positions."""

    margin_summary: dict = field(default_factory=dict)
    asset_positions: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Signal:
    """A trading signal produced by a strategy.

    Attributes
    ----------
    side:
        ``"buy"`` or ``"sell"``.
    entry:
        Suggested entry price, or ``None`` for market orders.
    stop:
        Suggested stop-loss price, or ``None`` if not applicable.
    take_profit:
        Suggested take-profit price, or ``None`` if not applicable.
    confidence:
        Confidence score in the range [0, 1].
    metadata:
        Arbitrary extra data attached by the strategy.
    """

    side: str
    entry: float | None
    stop: float | None
    take_profit: float | None
    confidence: float
    metadata: dict = field(default_factory=dict)
