"""WebSocket-based order update reconciliation.

Bridges the async :class:`~autotrader.hl.ws.HLWebSocket` with the
synchronous :class:`~autotrader.runtime.scheduler.TradingScheduler` by
running the WebSocket listener in a background thread and collecting
order-update / fill events into thread-safe queues.

PRD §10 requires: "Reconciliation: orderUpdates WS + orderStatus REST."
This module handles the WS side; the scheduler's existing REST-based
``_reconcile_fills`` handles the REST fallback.

Usage
-----
    reconciler = WSOrderReconciler(ws_url="wss://...", user_address="0x...")
    reconciler.start()

    # Each scheduler iteration:
    for update in reconciler.drain_order_updates():
        process(update)

    # On shutdown:
    reconciler.stop()
"""

from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any

import structlog

from autotrader.hl.ws import HLWebSocket
from autotrader.monitoring.metrics import metrics

logger = structlog.get_logger(__name__)

# Maximum number of buffered events before oldest are dropped.
_MAX_QUEUE_SIZE = 10_000


class WSOrderReconciler:
    """Collect ``orderUpdates`` and ``userFills`` from the Hyperliquid WS.

    Runs the async WebSocket client in a dedicated daemon thread so the
    synchronous scheduler can call :meth:`drain_order_updates` and
    :meth:`drain_user_fills` on every iteration without blocking.

    Parameters
    ----------
    ws_url : str
        WebSocket endpoint URL.
    user_address : str
        The user's Ethereum-style address for subscriptions.
    """

    def __init__(
        self,
        ws_url: str = "wss://api.hyperliquid.xyz/ws",
        user_address: str = "",
    ) -> None:
        self._ws_url = ws_url
        self._user_address = user_address

        # Thread-safe queues for cross-thread communication
        self._order_updates: queue.Queue[dict[str, Any]] = queue.Queue(
            maxsize=_MAX_QUEUE_SIZE
        )
        self._user_fills: queue.Queue[dict[str, Any]] = queue.Queue(
            maxsize=_MAX_QUEUE_SIZE
        )

        self._ws: HLWebSocket | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background WebSocket listener thread."""
        if self._running:
            logger.debug("ws_reconciler.already_running")
            return

        if not self._user_address:
            logger.warning("ws_reconciler.no_user_address, skipping WS reconciliation")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_event_loop,
            name="ws-reconciler",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "ws_reconciler.started",
            ws_url=self._ws_url,
            user=self._user_address[:10] + "...",
        )

    def stop(self) -> None:
        """Shut down the background WebSocket listener."""
        self._running = False

        if self._loop is not None and not self._loop.is_closed():
            # Schedule the disconnect coroutine on the event loop
            asyncio.run_coroutine_threadsafe(self._shutdown_ws(), self._loop)

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        logger.info("ws_reconciler.stopped")

    @property
    def is_running(self) -> bool:
        """Return whether the background listener is active."""
        return self._running and self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Drain queues (called by the synchronous scheduler)
    # ------------------------------------------------------------------

    def drain_order_updates(self) -> list[dict[str, Any]]:
        """Return and remove all buffered ``orderUpdates`` messages.

        Each message is the raw dict from the WS ``data`` payload.
        Typical fields: ``order`` (with ``oid``, ``coin``, ``side``,
        ``sz``, ``limitPx``, ``orderType``), ``status``, ``statusTimestamp``.
        """
        updates: list[dict[str, Any]] = []
        while True:
            try:
                updates.append(self._order_updates.get_nowait())
            except queue.Empty:
                break
        return updates

    def drain_user_fills(self) -> list[dict[str, Any]]:
        """Return and remove all buffered ``userFills`` messages.

        Each message is the raw dict from the WS ``data`` payload.
        """
        fills: list[dict[str, Any]] = []
        while True:
            try:
                fills.append(self._user_fills.get_nowait())
            except queue.Empty:
                break
        return fills

    # ------------------------------------------------------------------
    # Background thread internals
    # ------------------------------------------------------------------

    def _run_event_loop(self) -> None:
        """Entry point for the background thread.  Creates an event loop
        and runs the WS connection + subscriptions."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_main())
        except Exception as exc:
            logger.error("ws_reconciler.loop_error", error=str(exc))
        finally:
            self._loop.close()
            self._loop = None

    async def _async_main(self) -> None:
        """Async main: connect, subscribe, and keep alive until stopped."""
        self._ws = HLWebSocket(ws_url=self._ws_url)

        try:
            await self._ws.connect()
        except Exception as exc:
            logger.error("ws_reconciler.connect_failed", error=str(exc))
            self._running = False
            return

        # Subscribe to orderUpdates
        await self._ws.subscribe(
            "orderUpdates",
            {"user": self._user_address},
            self._on_order_update,
        )

        # Subscribe to userFills
        await self._ws.subscribe(
            "userFills",
            {"user": self._user_address},
            self._on_user_fill,
        )

        logger.info(
            "ws_reconciler.subscribed",
            channels=["orderUpdates", "userFills"],
        )

        # Keep alive until stopped
        while self._running:
            await asyncio.sleep(1.0)

        await self._ws.disconnect()

    async def _shutdown_ws(self) -> None:
        """Disconnect the WS from within the event loop."""
        if self._ws is not None:
            try:
                await self._ws.disconnect()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # WS callbacks
    # ------------------------------------------------------------------

    def _on_order_update(self, msg: dict[str, Any]) -> None:
        """Callback for ``orderUpdates`` channel messages."""
        data = msg.get("data", msg)

        # orderUpdates can contain a list of updates or a single update
        updates = data if isinstance(data, list) else [data]

        for update in updates:
            try:
                self._order_updates.put_nowait(update)
                metrics.inc_counter("ws_order_updates_received_total")
            except queue.Full:
                # Drop oldest to make room
                try:
                    self._order_updates.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._order_updates.put_nowait(update)
                except queue.Full:
                    pass
                metrics.inc_counter("ws_order_updates_dropped_total")

        logger.debug(
            "ws_reconciler.order_update",
            count=len(updates),
        )

    def _on_user_fill(self, msg: dict[str, Any]) -> None:
        """Callback for ``userFills`` channel messages."""
        data = msg.get("data", msg)

        fills = data if isinstance(data, list) else [data]

        for fill in fills:
            try:
                self._user_fills.put_nowait(fill)
                metrics.inc_counter("ws_user_fills_received_total")
            except queue.Full:
                try:
                    self._user_fills.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._user_fills.put_nowait(fill)
                except queue.Full:
                    pass
                metrics.inc_counter("ws_user_fills_dropped_total")

        logger.debug(
            "ws_reconciler.user_fill",
            count=len(fills),
        )
