"""Order lifecycle management.

The :class:`OrderManager` tracks every order from submission through fill or
cancellation, manages the relationship between trade IDs and their associated
orders (entry, stop-loss, take-profit), and provides query methods for active
and historical order state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import structlog

from autotrader.execution.broker import Broker, OrderResult
from autotrader.monitoring.metrics import metrics
from autotrader.utils.time import now_ms

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Order states
# ---------------------------------------------------------------------------


class OrderState(Enum):
    """Lifecycle states for a managed order."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


# Active states are those that may still result in a fill.
_ACTIVE_STATES = frozenset(
    {
        OrderState.PENDING,
        OrderState.SUBMITTED,
        OrderState.PARTIAL,
    }
)


# ---------------------------------------------------------------------------
# ManagedOrder
# ---------------------------------------------------------------------------


@dataclass
class ManagedOrder:
    """A tracked order with full lifecycle metadata.

    Attributes
    ----------
    order_id:
        Unique order identifier.
    symbol:
        Trading pair symbol.
    side:
        ``"buy"`` or ``"sell"``.
    size:
        Requested size in base asset units.
    price:
        Limit price in USD.
    order_type:
        ``"limit"``, ``"market"``, ``"stop_loss"``, ``"take_profit"``.
    state:
        Current lifecycle state.
    filled_size:
        Cumulative filled quantity.
    filled_price:
        Volume-weighted average fill price.
    fee:
        Total fees paid for fills on this order.
    created_at:
        Epoch milliseconds when the order was created.
    updated_at:
        Epoch milliseconds of the most recent state change.
    stop_px:
        Stop-loss trigger price (``None`` if not applicable).
    tp_px:
        Take-profit trigger price (``None`` if not applicable).
    parent_trade_id:
        The trade ID this order belongs to.
    error:
        Human-readable error description (empty string when no error).
    """

    order_id: str
    symbol: str
    side: str
    size: float
    price: float
    order_type: str
    state: OrderState
    filled_size: float = 0.0
    filled_price: float = 0.0
    fee: float = 0.0
    created_at: int = 0
    updated_at: int = 0
    stop_px: float | None = None
    tp_px: float | None = None
    parent_trade_id: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# OrderManager
# ---------------------------------------------------------------------------


class OrderManager:
    """Manages the full lifecycle of orders and their association with trades.

    Parameters
    ----------
    broker:
        The :class:`Broker` instance used to place and cancel orders.
    """

    def __init__(self, broker: Broker) -> None:
        self.broker = broker
        self.orders: dict[str, ManagedOrder] = {}
        self.trade_orders: dict[str, list[str]] = {}  # trade_id -> [order_ids]

    # ------------------------------------------------------------------
    # Entry orders
    # ------------------------------------------------------------------

    def submit_entry(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        trade_id: str,
        stop_px: float | None = None,
        tp_px: float | None = None,
    ) -> ManagedOrder:
        """Submit an entry order and optionally attach stop/TP trigger orders.

        Parameters
        ----------
        symbol:
            Trading pair symbol.
        side:
            ``"buy"`` or ``"sell"``.
        size:
            Order size in base units.
        price:
            Limit price in USD.
        trade_id:
            Unique trade identifier that groups related orders.
        stop_px:
            Optional stop-loss price.  A trigger order is placed automatically
            when the entry fills.
        tp_px:
            Optional take-profit price.  A trigger order is placed
            automatically when the entry fills.

        Returns
        -------
        ManagedOrder
            The managed entry order with its current state.
        """
        ts = now_ms()

        # Place the entry order through the broker
        result: OrderResult = self.broker.place_order(
            symbol=symbol,
            side=side,
            size=size,
            price=price,
            order_type="limit",
            reduce_only=False,
            stop_px=stop_px,
            tp_px=tp_px,
        )

        # Map broker status to OrderState
        state = self._map_status(result.status)

        managed = ManagedOrder(
            order_id=result.order_id,
            symbol=symbol,
            side=side,
            size=size,
            price=price,
            order_type="limit",
            state=state,
            filled_size=result.filled_sz,
            filled_price=result.filled_px,
            fee=result.fee,
            created_at=ts,
            updated_at=ts,
            stop_px=stop_px,
            tp_px=tp_px,
            parent_trade_id=trade_id,
            error=result.error,
        )

        # Track the order
        self.orders[result.order_id] = managed
        self.trade_orders.setdefault(trade_id, []).append(result.order_id)

        metrics.inc_counter("managed_orders_total")

        logger.info(
            "entry_submitted",
            order_id=result.order_id,
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            size=size,
            price=price,
            state=state.value,
        )

        # If the entry was filled immediately, place stop and TP trigger orders
        if state == OrderState.FILLED:
            self._place_protective_orders(
                symbol=symbol,
                side=side,
                size=result.filled_sz,
                trade_id=trade_id,
                stop_px=stop_px,
                tp_px=tp_px,
            )

        return managed

    # ------------------------------------------------------------------
    # Exit orders
    # ------------------------------------------------------------------

    def submit_exit(
        self,
        symbol: str,
        trade_id: str,
        size: float,
        price: float,
        reason: str = "signal",
    ) -> ManagedOrder:
        """Submit an exit (close) order for an existing trade.

        Parameters
        ----------
        symbol:
            Trading pair symbol.
        trade_id:
            The trade to close.
        size:
            Size to close.
        price:
            Limit price.
        reason:
            Human-readable reason for the exit (e.g. ``"signal"``,
            ``"stop_loss"``, ``"take_profit"``, ``"manual"``).

        Returns
        -------
        ManagedOrder
        """
        ts = now_ms()

        # Determine exit side: opposite of the entry side
        entry_side = self._infer_trade_side(trade_id)
        exit_side = "sell" if entry_side == "buy" else "buy"

        result: OrderResult = self.broker.place_order(
            symbol=symbol,
            side=exit_side,
            size=size,
            price=price,
            order_type="limit",
            reduce_only=True,
        )

        state = self._map_status(result.status)

        managed = ManagedOrder(
            order_id=result.order_id,
            symbol=symbol,
            side=exit_side,
            size=size,
            price=price,
            order_type="exit",
            state=state,
            filled_size=result.filled_sz,
            filled_price=result.filled_px,
            fee=result.fee,
            created_at=ts,
            updated_at=ts,
            parent_trade_id=trade_id,
            error=result.error,
        )

        self.orders[result.order_id] = managed
        self.trade_orders.setdefault(trade_id, []).append(result.order_id)

        metrics.inc_counter("managed_orders_total")

        logger.info(
            "exit_submitted",
            order_id=result.order_id,
            trade_id=trade_id,
            symbol=symbol,
            side=exit_side,
            size=size,
            price=price,
            reason=reason,
            state=state.value,
        )

        return managed

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def cancel_trade_orders(self, trade_id: str) -> int:
        """Cancel all orders associated with a trade.

        Parameters
        ----------
        trade_id:
            The trade whose orders should be cancelled.

        Returns
        -------
        int
            Number of orders successfully cancelled.
        """
        order_ids = self.trade_orders.get(trade_id, [])
        cancelled = 0

        for oid in order_ids:
            managed = self.orders.get(oid)
            if managed is None:
                continue
            if managed.state not in _ACTIVE_STATES:
                continue

            success = self.broker.cancel_order(managed.symbol, oid)
            if success:
                managed.state = OrderState.CANCELLED
                managed.updated_at = now_ms()
                cancelled += 1
                logger.info(
                    "trade_order_cancelled",
                    order_id=oid,
                    trade_id=trade_id,
                )

        return cancelled

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def update_order(
        self,
        order_id: str,
        status: str,
        filled_sz: float = 0,
        filled_px: float = 0,
        fee: float = 0,
    ) -> None:
        """Update the state of a managed order.

        Called by the reconciler when new fill data is available or when
        an order status changes on the exchange.

        Parameters
        ----------
        order_id:
            The order to update.
        status:
            New status string (mapped to :class:`OrderState`).
        filled_sz:
            Additional filled quantity (added to existing).
        filled_px:
            Average price of the new fill.
        fee:
            Additional fee (added to existing).
        """
        managed = self.orders.get(order_id)
        if managed is None:
            logger.warning("update_order_not_found", order_id=order_id)
            return

        new_state = self._map_status(status)
        old_state = managed.state
        ts = now_ms()

        # Update fill data with volume-weighted average price
        if filled_sz > 0:
            prev_notional = managed.filled_size * managed.filled_price
            new_notional = filled_sz * filled_px
            total_size = managed.filled_size + filled_sz
            if total_size > 0:
                managed.filled_price = (prev_notional + new_notional) / total_size
            managed.filled_size = total_size
            managed.fee += fee

        managed.state = new_state
        managed.updated_at = ts

        if old_state != new_state:
            logger.info(
                "order_state_changed",
                order_id=order_id,
                old_state=old_state.value,
                new_state=new_state.value,
                filled_size=managed.filled_size,
                filled_price=managed.filled_price,
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_active_orders(self, symbol: str | None = None) -> list[ManagedOrder]:
        """Return all orders in an active state.

        Parameters
        ----------
        symbol:
            Optional filter by symbol.

        Returns
        -------
        list[ManagedOrder]
        """
        result: list[ManagedOrder] = []
        for order in self.orders.values():
            if order.state not in _ACTIVE_STATES:
                continue
            if symbol is not None and order.symbol != symbol:
                continue
            result.append(order)
        return result

    def get_orders_for_trade(self, trade_id: str) -> list[ManagedOrder]:
        """Return all orders associated with a trade.

        Parameters
        ----------
        trade_id:
            The trade identifier.

        Returns
        -------
        list[ManagedOrder]
        """
        order_ids = self.trade_orders.get(trade_id, [])
        result: list[ManagedOrder] = []
        for oid in order_ids:
            managed = self.orders.get(oid)
            if managed is not None:
                result.append(managed)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _map_status(self, status: str) -> OrderState:
        """Map a broker status string to an :class:`OrderState`."""
        mapping = {
            "filled": OrderState.FILLED,
            "partial": OrderState.PARTIAL,
            "rejected": OrderState.REJECTED,
            "pending": OrderState.SUBMITTED,
            "submitted": OrderState.SUBMITTED,
            "cancelled": OrderState.CANCELLED,
            "expired": OrderState.EXPIRED,
        }
        return mapping.get(status, OrderState.PENDING)

    def _infer_trade_side(self, trade_id: str) -> str:
        """Infer the entry side of a trade from its first order.

        Defaults to ``"buy"`` if no entry order can be found.
        """
        order_ids = self.trade_orders.get(trade_id, [])
        for oid in order_ids:
            managed = self.orders.get(oid)
            if managed is not None and managed.order_type in ("limit", "market"):
                return managed.side
        return "buy"

    def _place_protective_orders(
        self,
        symbol: str,
        side: str,
        size: float,
        trade_id: str,
        stop_px: float | None,
        tp_px: float | None,
    ) -> None:
        """Place stop-loss and/or take-profit trigger orders after an entry fill.

        The trigger orders are on the opposite side of the entry.
        """
        # Stop and TP are exit orders, so they are on the opposite side.
        exit_side = "sell" if side == "buy" else "buy"

        if stop_px is not None:
            sl_result = self.broker.place_trigger_order(
                symbol=symbol,
                side=exit_side,
                size=size,
                trigger_px=stop_px,
                order_type="stop_loss",
            )
            sl_state = self._map_status(sl_result.status)
            sl_managed = ManagedOrder(
                order_id=sl_result.order_id,
                symbol=symbol,
                side=exit_side,
                size=size,
                price=stop_px,
                order_type="stop_loss",
                state=sl_state,
                created_at=now_ms(),
                updated_at=now_ms(),
                stop_px=stop_px,
                parent_trade_id=trade_id,
                error=sl_result.error,
            )
            self.orders[sl_result.order_id] = sl_managed
            self.trade_orders.setdefault(trade_id, []).append(sl_result.order_id)

            logger.info(
                "stop_loss_placed",
                order_id=sl_result.order_id,
                trade_id=trade_id,
                symbol=symbol,
                trigger_px=stop_px,
                status=sl_state.value,
            )

        if tp_px is not None:
            tp_result = self.broker.place_trigger_order(
                symbol=symbol,
                side=exit_side,
                size=size,
                trigger_px=tp_px,
                order_type="take_profit",
            )
            tp_state = self._map_status(tp_result.status)
            tp_managed = ManagedOrder(
                order_id=tp_result.order_id,
                symbol=symbol,
                side=exit_side,
                size=size,
                price=tp_px,
                order_type="take_profit",
                state=tp_state,
                created_at=now_ms(),
                updated_at=now_ms(),
                tp_px=tp_px,
                parent_trade_id=trade_id,
                error=tp_result.error,
            )
            self.orders[tp_result.order_id] = tp_managed
            self.trade_orders.setdefault(trade_id, []).append(tp_result.order_id)

            logger.info(
                "take_profit_placed",
                order_id=tp_result.order_id,
                trade_id=trade_id,
                symbol=symbol,
                trigger_px=tp_px,
                status=tp_state.value,
            )
