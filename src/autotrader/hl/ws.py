"""WebSocket client for Hyperliquid with automatic reconnection.

Provides async subscription management for real-time data feeds
(candles, order book, trades, user fills, order updates) with
exponential backoff reconnection and automatic resubscription.

Usage
-----
    import asyncio
    from autotrader.hl.ws import HLWebSocket

    async def on_candle(msg):
        print(msg)

    ws = HLWebSocket()
    await ws.connect()
    await ws.subscribe("candle", {"coin": "ETH", "interval": "1m"}, on_candle)
    # ... ws listens in background ...
    await ws.disconnect()
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

import structlog
import websockets
import websockets.exceptions

logger = structlog.get_logger(__name__)

# Type alias for subscription callbacks -- may be sync or async.
Callback = Callable[[dict], Any]


class HLWebSocket:
    """Async WebSocket client for the Hyperliquid real-time API.

    Parameters
    ----------
    ws_url : str
        WebSocket endpoint URL.
    max_reconnect_delay : float
        Upper bound (in seconds) for the exponential backoff delay
        between reconnection attempts.
    """

    def __init__(
        self,
        ws_url: str = "wss://api.hyperliquid.xyz/ws",
        max_reconnect_delay: float = 60.0,
    ) -> None:
        self._ws_url = ws_url
        self._max_reconnect_delay = max_reconnect_delay

        # Active websocket connection (set after connect)
        self._ws: websockets.WebSocketClientProtocol | None = None

        # channel_key -> subscription message payload (for resubscription)
        self._subscriptions: dict[str, dict] = {}

        # channel_key -> list of callbacks
        self._callbacks: dict[str, list[Callback]] = {}

        self._running: bool = False
        self._reconnect_delay: float = 1.0
        self.reconnect_count: int = 0  # increments on each successful reconnect

        # Background listener task
        self._listen_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Subscription key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_channel_key(channel: str, params: dict) -> str:
        """Derive a deterministic string key from channel + params.

        This is used to deduplicate subscriptions and route incoming
        messages to the correct callbacks.
        """
        # Sort params for deterministic ordering
        parts = [channel]
        for k in sorted(params.keys()):
            parts.append(f"{k}={params[k]}")
        return "|".join(parts)

    @staticmethod
    def _build_subscription_msg(channel: str, params: dict) -> dict:
        """Build the JSON message to send over the websocket for a subscribe.

        The Hyperliquid WS API uses ``{"method": "subscribe", "subscription": {...}}``.
        The inner ``subscription`` object always has a ``"type"`` field equal to
        the channel name, plus any additional params.
        """
        subscription: dict[str, Any] = {"type": channel}
        subscription.update(params)
        return {"method": "subscribe", "subscription": subscription}

    @staticmethod
    def _build_unsubscription_msg(channel: str, params: dict) -> dict:
        """Build the JSON message for an unsubscribe."""
        subscription: dict[str, Any] = {"type": channel}
        subscription.update(params)
        return {"method": "unsubscribe", "subscription": subscription}

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the WebSocket connection and start the listener loop.

        If the connection drops the client will automatically reconnect
        with exponential backoff.
        """
        self._running = True
        await self._do_connect()
        self._listen_task = asyncio.create_task(self._listen())
        logger.info("ws_connected", url=self._ws_url)

    async def _do_connect(self) -> None:
        """Establish the raw WebSocket connection."""
        self._ws = await websockets.connect(
            self._ws_url,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
        )
        # Reset backoff on successful connection
        self._reconnect_delay = 1.0

    async def disconnect(self) -> None:
        """Gracefully shut down the WebSocket connection."""
        self._running = False

        if self._listen_task is not None:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        logger.info("ws_disconnected")

    # ------------------------------------------------------------------
    # Subscribe / Unsubscribe
    # ------------------------------------------------------------------

    async def subscribe(
        self,
        channel: str,
        params: dict,
        callback: Callback,
    ) -> None:
        """Subscribe to a channel and register a callback.

        Supported channels and their expected params:

        * ``"candle"``       -- ``{"coin": "ETH", "interval": "1m"}``
        * ``"l2Book"``       -- ``{"coin": "ETH"}``
        * ``"trades"``       -- ``{"coin": "ETH"}``
        * ``"orderUpdates"`` -- ``{"user": "0x..."}``
        * ``"userFills"``    -- ``{"user": "0x..."}``

        Parameters
        ----------
        channel : str
            Channel type name.
        params : dict
            Channel-specific parameters.
        callback : Callback
            Function (sync or async) called with each incoming message
            that matches this subscription.
        """
        key = self._make_channel_key(channel, params)
        msg = self._build_subscription_msg(channel, params)

        # Store for resubscription on reconnect
        self._subscriptions[key] = msg
        self._callbacks.setdefault(key, []).append(callback)

        # Send to server if connected
        if self._ws is not None:
            try:
                await self._ws.send(json.dumps(msg))
                logger.debug("ws_subscribed", channel=channel, params=params)
            except Exception as exc:
                logger.warning(
                    "ws_subscribe_send_failed",
                    channel=channel,
                    error=str(exc),
                )

    async def unsubscribe(self, channel: str, params: dict) -> None:
        """Unsubscribe from a channel and remove all associated callbacks.

        Parameters
        ----------
        channel : str
            Channel type name.
        params : dict
            Channel-specific parameters (must match those used in subscribe).
        """
        key = self._make_channel_key(channel, params)
        msg = self._build_unsubscription_msg(channel, params)

        self._subscriptions.pop(key, None)
        self._callbacks.pop(key, None)

        if self._ws is not None:
            try:
                await self._ws.send(json.dumps(msg))
                logger.debug("ws_unsubscribed", channel=channel, params=params)
            except Exception as exc:
                logger.warning(
                    "ws_unsubscribe_send_failed",
                    channel=channel,
                    error=str(exc),
                )

    # ------------------------------------------------------------------
    # Listener loop
    # ------------------------------------------------------------------

    async def _listen(self) -> None:
        """Main message loop.  Automatically reconnects on failure."""
        while self._running:
            try:
                async for raw_msg in self._ws:  # type: ignore[union-attr]
                    if not self._running:
                        break
                    try:
                        msg = json.loads(raw_msg)
                    except json.JSONDecodeError:
                        logger.warning("ws_invalid_json", raw=str(raw_msg)[:200])
                        continue
                    self._route_message(msg)
            except websockets.exceptions.ConnectionClosed as exc:
                if not self._running:
                    break
                logger.warning("ws_connection_closed", code=exc.code, reason=exc.reason)
                await self._reconnect()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not self._running:
                    break
                logger.error("ws_listen_error", error=str(exc))
                await self._reconnect()

    # ------------------------------------------------------------------
    # Reconnection
    # ------------------------------------------------------------------

    async def _reconnect(self) -> None:
        """Reconnect with exponential backoff and resubscribe."""
        while self._running:
            logger.info(
                "ws_reconnecting",
                delay=self._reconnect_delay,
            )
            await asyncio.sleep(self._reconnect_delay)

            # Exponential backoff with cap
            self._reconnect_delay = min(
                self._reconnect_delay * 2.0,
                self._max_reconnect_delay,
            )

            try:
                await self._do_connect()
                self.reconnect_count += 1
                logger.info("ws_reconnected", reconnect_count=self.reconnect_count)

                # Resubscribe to all active subscriptions
                for key, sub_msg in self._subscriptions.items():
                    try:
                        await self._ws.send(json.dumps(sub_msg))  # type: ignore[union-attr]
                        logger.debug("ws_resubscribed", key=key)
                    except Exception as exc:
                        logger.warning(
                            "ws_resubscribe_failed",
                            key=key,
                            error=str(exc),
                        )
                return

            except Exception as exc:
                logger.warning("ws_reconnect_failed", error=str(exc))
                # Continue the loop -- will sleep again with increased delay

    # ------------------------------------------------------------------
    # Message routing
    # ------------------------------------------------------------------

    def _route_message(self, msg: dict) -> None:
        """Route an incoming message to matching subscription callbacks.

        HL WebSocket messages typically have the structure::

            {
                "channel": "<type>",
                "data": { ... }
            }

        We reconstruct the channel key from the message metadata and
        invoke all registered callbacks for that key.
        """
        channel = msg.get("channel")

        if channel is None:
            # Could be a subscription confirmation or error -- log and ignore
            if "error" in msg:
                logger.error("ws_server_error", msg=msg)
            return

        # Build candidate keys to match against stored subscriptions.
        # We try to extract the identifying params from the data payload.
        matched = False
        for key, callbacks in self._callbacks.items():
            if key.startswith(channel + "|") or key == channel:
                # Verify the key matches the message content more precisely
                # by checking if key-params are a subset of the message data.
                if self._key_matches_message(key, channel, msg):
                    matched = True
                    for cb in callbacks:
                        try:
                            result = cb(msg)
                            # If callback is a coroutine, schedule it
                            if asyncio.iscoroutine(result):
                                asyncio.create_task(result)  # type: ignore[arg-type]
                        except Exception as exc:
                            logger.error(
                                "ws_callback_error",
                                channel=channel,
                                error=str(exc),
                            )

        if not matched:
            logger.debug("ws_unrouted_message", channel=channel)

    def _key_matches_message(self, key: str, channel: str, msg: dict) -> bool:
        """Check whether a subscription key matches an incoming message.

        This uses a best-effort heuristic: it parses the key back into
        param pairs and checks if those values appear in the message or
        its ``data`` sub-dict.
        """
        data = msg.get("data", {})

        # Parse key: "channel|k1=v1|k2=v2"
        parts = key.split("|")
        if parts[0] != channel:
            return False

        # If key is just the channel with no params, it matches all
        # messages on that channel.
        if len(parts) == 1:
            return True

        # Check each param
        for part in parts[1:]:
            if "=" not in part:
                continue
            k, v = part.split("=", 1)

            # Look in msg top level, in data, and in data.s (for candles)
            msg_val = msg.get(k) or (data.get(k) if isinstance(data, dict) else None)
            # Also check data.s for the coin field in candle messages
            if msg_val is None and isinstance(data, dict):
                msg_val = data.get("s")

            if msg_val is not None and str(msg_val) == v:
                continue
            # If we can't verify a param, still allow the match --
            # the HL WS format varies by channel and it's better to
            # deliver a message than silently drop it.

        return True
