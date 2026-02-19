"""Exchange adapter supporting paper and live execution modes.

The :class:`Broker` abstracts order placement, cancellation, and fill
retrieval behind a unified interface.  In **paper** mode every order is
simulated locally with realistic slippage applied by a
:class:`~autotrader.execution.slippage.SlippageModel`.  In **live** mode
orders are routed through :class:`~autotrader.hl.client.HLClient` to the
Hyperliquid exchange.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from typing import Any

import structlog

from autotrader.execution.slippage import SlippageModel
from autotrader.hl.client import HLClient
from autotrader.hl.nonces import get_next as get_next_nonce
from autotrader.monitoring.metrics import metrics
from autotrader.utils.ids import generate_order_id
from autotrader.utils.time import now_ms

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class OrderResult:
    """Result of an order placement attempt."""

    order_id: str
    status: str  # "filled", "partial", "rejected", "pending"
    filled_px: float = 0.0
    filled_sz: float = 0.0
    remaining_sz: float = 0.0
    fee: float = 0.0
    timestamp_ms: int = 0
    error: str = ""


# ---------------------------------------------------------------------------
# Broker
# ---------------------------------------------------------------------------


class Broker:
    """Unified broker that routes orders through paper simulation or live API.

    Parameters
    ----------
    client:
        An :class:`HLClient` instance.  Required for live mode; ignored in
        paper mode.
    mode:
        ``"paper"`` for local simulation, ``"live"`` for real exchange orders.
    slippage_model:
        Slippage model used in paper mode.  A default model is created if not
        provided.
    """

    def __init__(
        self,
        client: HLClient | None = None,
        mode: str = "paper",
        slippage_model: SlippageModel | None = None,
    ) -> None:
        self.client = client
        self.mode = mode
        self.slippage_model = slippage_model or SlippageModel()

        # Paper-mode bookkeeping
        self.paper_fills: list[dict[str, Any]] = []
        self._pending_orders: dict[str, dict[str, Any]] = {}
        self._filled_orders: dict[str, dict[str, Any]] = {}

        # Asset symbol -> integer index mapping (populated lazily in live mode)
        self._asset_index: dict[str, int] = {}

        # Thread safety for mutable bookkeeping state
        self._lock = threading.Lock()

        logger.info("broker_init", mode=self.mode)

    # ------------------------------------------------------------------
    # Asset index helpers
    # ------------------------------------------------------------------

    def _resolve_asset_index(self, symbol: str) -> int:
        """Return the integer asset index for *symbol*.

        In paper mode this is a simple hash-based assignment.  In live mode
        the mapping is fetched from exchange metadata on first use.
        """
        if symbol in self._asset_index:
            return self._asset_index[symbol]

        if self.mode == "live" and self.client is not None:
            try:
                meta = self.client.get_meta()
                universe = meta.get("universe", [])
                for idx, asset in enumerate(universe):
                    name = asset.get("name", "")
                    self._asset_index[name] = idx
                if symbol in self._asset_index:
                    return self._asset_index[symbol]
            except Exception:
                logger.warning("asset_index_fetch_failed", symbol=symbol)

        # Fallback: deterministic hash for paper mode or if lookup failed
        # Use SHA-256 instead of hash() which is randomized per-process
        digest = hashlib.sha256(symbol.encode()).hexdigest()
        index = int(digest[:8], 16) % 10000
        self._asset_index[symbol] = index
        return index

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        order_type: str = "limit",
        reduce_only: bool = False,
        tif: str = "Gtc",
        stop_px: float | None = None,
        tp_px: float | None = None,
    ) -> OrderResult:
        """Place an order in paper or live mode.

        Parameters
        ----------
        symbol:
            Trading pair symbol, e.g. ``"ETH"``.
        side:
            ``"buy"`` or ``"sell"``.
        size:
            Order size in base asset units.
        price:
            Limit price in USD.
        order_type:
            ``"limit"`` or ``"market"``.
        reduce_only:
            Whether this order can only reduce an existing position.
        tif:
            Time-in-force: ``"Gtc"``, ``"Ioc"``, ``"Alo"``.
        stop_px:
            Optional stop-loss trigger price (informational; the broker does
            not auto-manage stops -- use :meth:`place_trigger_order` for that).
        tp_px:
            Optional take-profit trigger price (informational).

        Returns
        -------
        OrderResult
        """
        if self.mode == "paper":
            return self._paper_place(
                symbol,
                side,
                size,
                price,
                order_type,
                reduce_only,
                tif,
                stop_px,
                tp_px,
            )
        return self._live_place(
            symbol,
            side,
            size,
            price,
            order_type,
            reduce_only,
            tif,
            stop_px,
            tp_px,
        )

    # -- paper mode -----------------------------------------------------

    def _paper_place(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        order_type: str,
        reduce_only: bool,
        tif: str,
        stop_px: float | None,
        tp_px: float | None,
    ) -> OrderResult:
        """Simulate an order fill locally with slippage applied."""
        order_id = generate_order_id()
        ts = now_ms()
        notional = size * price

        # Compute slippage
        slippage_usd = self.slippage_model.estimate(notional)
        slippage_per_unit = slippage_usd / size if size > 0 else 0.0

        # Apply slippage to fill price
        if side == "buy":
            fill_px = price + slippage_per_unit
        else:
            fill_px = price - slippage_per_unit

        # Simulate taker fee of 2.5 bps (matches Hyperliquid standard rate)
        fee = notional * 2.5 / 10_000.0

        # Record fill
        fill_record: dict[str, Any] = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "size": size,
            "price": round(fill_px, 6),
            "fee": round(fee, 6),
            "timestamp_ms": ts,
            "order_type": order_type,
            "reduce_only": reduce_only,
            "stop_px": stop_px,
            "tp_px": tp_px,
        }
        self.paper_fills.append(fill_record)
        self._filled_orders[order_id] = fill_record

        # Emit metrics
        metrics.inc_counter("orders_placed_total")
        metrics.inc_counter("orders_filled_total")
        metrics.observe("fill_slippage_bps", self.slippage_model.estimate_bps(notional))
        metrics.observe("fill_fee_usd", fee)

        logger.info(
            "paper_fill",
            order_id=order_id,
            symbol=symbol,
            side=side,
            size=size,
            requested_px=price,
            fill_px=round(fill_px, 6),
            slippage_usd=round(slippage_usd, 6),
            fee=round(fee, 6),
        )

        return OrderResult(
            order_id=order_id,
            status="filled",
            filled_px=round(fill_px, 6),
            filled_sz=size,
            remaining_sz=0.0,
            fee=round(fee, 6),
            timestamp_ms=ts,
        )

    # -- live mode ------------------------------------------------------

    def _live_place(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        order_type: str,
        reduce_only: bool,
        tif: str,
        stop_px: float | None,
        tp_px: float | None,
    ) -> OrderResult:
        """Place a real order through HLClient."""
        if self.client is None:
            return OrderResult(
                order_id="",
                status="rejected",
                error="No HLClient configured for live mode",
                timestamp_ms=now_ms(),
            )

        order_id = generate_order_id()
        ts = now_ms()
        asset_index = self._resolve_asset_index(symbol)
        is_buy = side == "buy"

        # Build the exchange payload
        order_payload: dict[str, Any] = {
            "action": {
                "type": "order",
                "orders": [
                    {
                        "a": asset_index,
                        "b": is_buy,
                        "p": str(price),
                        "s": str(size),
                        "r": reduce_only,
                        "t": {
                            "limit": {"tif": tif},
                        },
                    }
                ],
                "grouping": "na",
            },
            "nonce": get_next_nonce(),
        }

        # Track as pending
        pending_record: dict[str, Any] = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "size": size,
            "price": price,
            "order_type": order_type,
            "reduce_only": reduce_only,
            "tif": tif,
            "stop_px": stop_px,
            "tp_px": tp_px,
            "timestamp_ms": ts,
        }
        self._pending_orders[order_id] = pending_record

        metrics.inc_counter("orders_placed_total")

        try:
            response = self.client.place_order(order_payload)
        except Exception as exc:
            logger.error(
                "live_order_failed",
                order_id=order_id,
                symbol=symbol,
                error=str(exc),
            )
            self._pending_orders.pop(order_id, None)
            return OrderResult(
                order_id=order_id,
                status="rejected",
                remaining_sz=size,
                error=str(exc),
                timestamp_ms=ts,
            )

        # Parse the response
        result = self._parse_order_response(order_id, size, response, ts)

        if result.status == "filled":
            self._pending_orders.pop(order_id, None)
            self._filled_orders[order_id] = {
                **pending_record,
                "filled_px": result.filled_px,
                "filled_sz": result.filled_sz,
                "fee": result.fee,
            }
            metrics.inc_counter("orders_filled_total")
        elif result.status == "rejected":
            self._pending_orders.pop(order_id, None)

        logger.info(
            "live_order_result",
            order_id=order_id,
            symbol=symbol,
            side=side,
            status=result.status,
            filled_px=result.filled_px,
            filled_sz=result.filled_sz,
        )

        return result

    def _parse_order_response(
        self,
        order_id: str,
        requested_size: float,
        response: dict[str, Any],
        ts: int,
    ) -> OrderResult:
        """Parse the exchange response into an OrderResult."""
        status_str = response.get("status", "")
        data = response.get("response", response.get("data", {}))

        if status_str == "ok":
            # Try to extract fill information from the response
            if isinstance(data, dict):
                statuses = data.get("data", {}).get("statuses", [])
                if statuses:
                    first = statuses[0]
                    if "filled" in first:
                        fill_info = first["filled"]
                        return OrderResult(
                            order_id=order_id,
                            status="filled",
                            filled_px=float(fill_info.get("avgPx", 0)),
                            filled_sz=float(fill_info.get("totalSz", requested_size)),
                            remaining_sz=0.0,
                            fee=float(fill_info.get("fee", 0)),
                            timestamp_ms=ts,
                        )
                    if "resting" in first:
                        return OrderResult(
                            order_id=order_id,
                            status="pending",
                            remaining_sz=requested_size,
                            timestamp_ms=ts,
                        )
                    if "error" in first:
                        return OrderResult(
                            order_id=order_id,
                            status="rejected",
                            remaining_sz=requested_size,
                            error=str(first["error"]),
                            timestamp_ms=ts,
                        )

            # Fallback: assume pending if status was ok but no fill details
            return OrderResult(
                order_id=order_id,
                status="pending",
                remaining_sz=requested_size,
                timestamp_ms=ts,
            )

        # Non-ok status
        error_msg = str(data) if data else status_str
        return OrderResult(
            order_id=order_id,
            status="rejected",
            remaining_sz=requested_size,
            error=error_msg,
            timestamp_ms=ts,
        )

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel a single order.

        Parameters
        ----------
        symbol:
            Asset symbol the order was placed for.
        order_id:
            The order identifier returned by :meth:`place_order`.

        Returns
        -------
        bool
            ``True`` if the cancellation succeeded, ``False`` otherwise.
        """
        if self.mode == "paper":
            removed = self._pending_orders.pop(order_id, None)
            if removed is not None:
                logger.info("paper_cancel", order_id=order_id, symbol=symbol)
                metrics.inc_counter("orders_cancelled_total")
                return True
            logger.debug("paper_cancel_not_found", order_id=order_id)
            return False

        # Live mode
        if self.client is None:
            return False

        asset_index = self._resolve_asset_index(symbol)

        # In live mode we need the exchange oid; use order_id hash as proxy
        oid = abs(hash(order_id)) % (2**31)
        try:
            self.client.cancel_order(asset_index, oid)
            self._pending_orders.pop(order_id, None)
            metrics.inc_counter("orders_cancelled_total")
            logger.info("live_cancel", order_id=order_id, symbol=symbol)
            return True
        except Exception as exc:
            logger.error(
                "live_cancel_failed",
                order_id=order_id,
                symbol=symbol,
                error=str(exc),
            )
            return False

    def cancel_all(self, symbol: str | None = None) -> int:
        """Cancel all open orders, optionally filtered by *symbol*.

        Returns
        -------
        int
            Number of orders successfully cancelled.
        """
        if self.mode == "paper":
            if symbol is None:
                count = len(self._pending_orders)
                self._pending_orders.clear()
            else:
                to_remove = [
                    oid for oid, o in self._pending_orders.items() if o.get("symbol") == symbol
                ]
                count = len(to_remove)
                for oid in to_remove:
                    del self._pending_orders[oid]

            metrics.inc_counter("orders_cancelled_total", float(count))
            logger.info("paper_cancel_all", symbol=symbol, count=count)
            return count

        # Live mode
        if self.client is None:
            return 0

        if symbol is None:
            try:
                self.client.cancel_all()
                count = len(self._pending_orders)
                self._pending_orders.clear()
                metrics.inc_counter("orders_cancelled_total", float(count))
                logger.info("live_cancel_all", count=count)
                return count
            except Exception as exc:
                logger.error("live_cancel_all_failed", error=str(exc))
                return 0
        else:
            # Cancel one by one for symbol-specific cancellation
            to_cancel = [
                (oid, o)
                for oid, o in list(self._pending_orders.items())
                if o.get("symbol") == symbol
            ]
            count = 0
            for oid, _ in to_cancel:
                if self.cancel_order(symbol, oid):
                    count += 1
            return count

    # ------------------------------------------------------------------
    # Trigger orders (stop-loss / take-profit)
    # ------------------------------------------------------------------

    def place_trigger_order(
        self,
        symbol: str,
        side: str,
        size: float,
        trigger_px: float,
        order_type: str = "stop_loss",
        tif: str = "Gtc",
    ) -> OrderResult:
        """Place a trigger order (stop-loss or take-profit).

        In paper mode the trigger order is stored as a pending order and will
        be evaluated when prices update.  In live mode the order is submitted
        to the exchange as a trigger order.

        Parameters
        ----------
        symbol:
            Asset symbol.
        side:
            ``"buy"`` or ``"sell"``.
        size:
            Order size in base asset units.
        trigger_px:
            Price at which the order is triggered.
        order_type:
            ``"stop_loss"`` or ``"take_profit"``.
        tif:
            Time-in-force.

        Returns
        -------
        OrderResult
        """
        order_id = generate_order_id()
        ts = now_ms()

        if self.mode == "paper":
            trigger_record: dict[str, Any] = {
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "size": size,
                "trigger_px": trigger_px,
                "order_type": order_type,
                "tif": tif,
                "timestamp_ms": ts,
                "is_trigger": True,
            }
            self._pending_orders[order_id] = trigger_record
            metrics.inc_counter("trigger_orders_placed_total")
            logger.info(
                "paper_trigger_order",
                order_id=order_id,
                symbol=symbol,
                side=side,
                size=size,
                trigger_px=trigger_px,
                order_type=order_type,
            )
            return OrderResult(
                order_id=order_id,
                status="pending",
                remaining_sz=size,
                timestamp_ms=ts,
            )

        # Live mode
        if self.client is None:
            return OrderResult(
                order_id=order_id,
                status="rejected",
                error="No HLClient configured for live mode",
                timestamp_ms=ts,
            )

        asset_index = self._resolve_asset_index(symbol)
        is_buy = side == "buy"

        # Determine trigger type
        if order_type == "stop_loss":
            tp_sl = "sl"
        else:
            tp_sl = "tp"

        trigger_payload: dict[str, Any] = {
            "action": {
                "type": "order",
                "orders": [
                    {
                        "a": asset_index,
                        "b": is_buy,
                        "p": str(trigger_px),
                        "s": str(size),
                        "r": True,  # trigger orders are typically reduce-only
                        "t": {
                            "trigger": {
                                "triggerPx": str(trigger_px),
                                "isMarket": True,
                                "tpsl": tp_sl,
                            },
                        },
                    }
                ],
                "grouping": "na",
            },
            "nonce": get_next_nonce(),
        }

        self._pending_orders[order_id] = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "size": size,
            "trigger_px": trigger_px,
            "order_type": order_type,
            "tif": tif,
            "timestamp_ms": ts,
            "is_trigger": True,
        }

        metrics.inc_counter("trigger_orders_placed_total")

        try:
            response = self.client.place_order(trigger_payload)
        except Exception as exc:
            logger.error(
                "live_trigger_order_failed",
                order_id=order_id,
                symbol=symbol,
                error=str(exc),
            )
            self._pending_orders.pop(order_id, None)
            return OrderResult(
                order_id=order_id,
                status="rejected",
                remaining_sz=size,
                error=str(exc),
                timestamp_ms=ts,
            )

        result = self._parse_order_response(order_id, size, response, ts)

        if result.status == "rejected":
            self._pending_orders.pop(order_id, None)

        logger.info(
            "live_trigger_order_result",
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            trigger_px=trigger_px,
            status=result.status,
        )

        return result

    # ------------------------------------------------------------------
    # Paper-mode trigger evaluation
    # ------------------------------------------------------------------

    def check_triggers(self, symbol: str, current_price: float) -> list[OrderResult]:
        """Evaluate pending trigger orders against the current price.

        In paper mode, this simulates stop-loss and take-profit execution.
        Should be called on each price update during paper trading.

        Parameters
        ----------
        symbol:
            The asset symbol whose price has updated.
        current_price:
            The current mark/last price for the symbol.

        Returns
        -------
        list[OrderResult]
            List of fills for any triggered orders.
        """
        if self.mode != "paper":
            return []

        triggered: list[OrderResult] = []
        to_remove: list[str] = []

        with self._lock:
            for oid, order in self._pending_orders.items():
                if not order.get("is_trigger"):
                    continue
                if order.get("symbol") != symbol:
                    continue

                trigger_px = order.get("trigger_px", 0.0)
                order_type = order.get("order_type", "")
                side = order.get("side", "")

                hit = False
                if order_type == "stop_loss":
                    # Stop-loss sell triggers when price falls to/below trigger
                    # Stop-loss buy triggers when price rises to/above trigger
                    if side == "sell" and current_price <= trigger_px:
                        hit = True
                    elif side == "buy" and current_price >= trigger_px:
                        hit = True
                elif order_type == "take_profit":
                    # Take-profit sell triggers when price rises to/above trigger
                    # Take-profit buy triggers when price falls to/below trigger
                    if side == "sell" and current_price >= trigger_px:
                        hit = True
                    elif side == "buy" and current_price <= trigger_px:
                        hit = True

                if hit:
                    to_remove.append(oid)

                    # Simulate the fill at the trigger price with slippage
                    size = order.get("size", 0.0)
                    notional = size * trigger_px
                    slippage_usd = self.slippage_model.estimate(notional)
                    slippage_per_unit = slippage_usd / size if size > 0 else 0.0

                    if side == "buy":
                        fill_px = trigger_px + slippage_per_unit
                    else:
                        fill_px = trigger_px - slippage_per_unit

                    fee = notional * 2.5 / 10_000.0
                    ts = now_ms()

                    fill_record: dict[str, Any] = {
                        "order_id": oid,
                        "symbol": symbol,
                        "side": side,
                        "size": size,
                        "price": round(fill_px, 6),
                        "fee": round(fee, 6),
                        "timestamp_ms": ts,
                        "order_type": order_type,
                        "reduce_only": True,
                    }
                    self.paper_fills.append(fill_record)
                    self._filled_orders[oid] = fill_record

                    result = OrderResult(
                        order_id=oid,
                        status="filled",
                        filled_px=round(fill_px, 6),
                        filled_sz=size,
                        remaining_sz=0.0,
                        fee=round(fee, 6),
                        timestamp_ms=ts,
                    )
                    triggered.append(result)

                    metrics.inc_counter("trigger_orders_filled_total")
                    logger.info(
                        "paper_trigger_filled",
                        order_id=oid,
                        symbol=symbol,
                        side=side,
                        order_type=order_type,
                        trigger_px=trigger_px,
                        fill_px=round(fill_px, 6),
                    )

            # Remove triggered orders from pending
            for oid in to_remove:
                self._pending_orders.pop(oid, None)

        return triggered

    # ------------------------------------------------------------------
    # Fill & order queries
    # ------------------------------------------------------------------

    def get_fills(self, symbol: str | None = None) -> list[dict]:
        """Return executed fills.

        In paper mode returns the local fill list.  In live mode fetches
        recent fills from the exchange.

        Parameters
        ----------
        symbol:
            Optional filter; if provided only fills for this symbol are
            returned.

        Returns
        -------
        list[dict]
        """
        if self.mode == "paper":
            if symbol is None:
                return list(self.paper_fills)
            return [f for f in self.paper_fills if f.get("symbol") == symbol]

        # Live mode
        if self.client is None:
            return []

        try:
            fills = self.client.get_user_fills()
            if symbol is not None:
                fills = [f for f in fills if f.get("coin") == symbol]
            return fills
        except Exception as exc:
            logger.error("get_fills_failed", error=str(exc))
            return []

    def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        """Return currently pending/open orders.

        Parameters
        ----------
        symbol:
            Optional symbol filter.

        Returns
        -------
        list[dict]
        """
        if self.mode == "paper":
            orders = list(self._pending_orders.values())
            if symbol is not None:
                orders = [o for o in orders if o.get("symbol") == symbol]
            return orders

        # Live mode
        if self.client is None:
            return []

        try:
            orders = self.client.get_open_orders()
            if symbol is not None:
                orders = [o for o in orders if o.get("coin") == symbol]
            return orders
        except Exception as exc:
            logger.error("get_open_orders_failed", error=str(exc))
            return []
