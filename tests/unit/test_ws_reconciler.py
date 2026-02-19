"""Tests for the WS-based order reconciliation bridge."""

from __future__ import annotations

import queue
import time
from unittest.mock import MagicMock, patch

import pytest

from autotrader.execution.ws_reconciler import WSOrderReconciler


# ---------------------------------------------------------------------------
# Queue / drain tests (no network)
# ---------------------------------------------------------------------------


class TestWSOrderReconcilerQueues:
    """Test the queue-based drain interface without starting the WS."""

    def test_drain_order_updates_empty(self) -> None:
        rec = WSOrderReconciler(user_address="0xabc")
        assert rec.drain_order_updates() == []

    def test_drain_user_fills_empty(self) -> None:
        rec = WSOrderReconciler(user_address="0xabc")
        assert rec.drain_user_fills() == []

    def test_drain_order_updates_returns_all_and_clears(self) -> None:
        rec = WSOrderReconciler(user_address="0xabc")
        # Manually push items into the queue
        rec._order_updates.put({"oid": 1, "status": "filled"})
        rec._order_updates.put({"oid": 2, "status": "canceled"})

        results = rec.drain_order_updates()
        assert len(results) == 2
        assert results[0]["oid"] == 1
        assert results[1]["oid"] == 2
        # Queue should be empty now
        assert rec.drain_order_updates() == []

    def test_drain_user_fills_returns_all_and_clears(self) -> None:
        rec = WSOrderReconciler(user_address="0xabc")
        rec._user_fills.put({"coin": "ETH", "sz": "1.0", "px": "3000"})
        rec._user_fills.put({"coin": "BTC", "sz": "0.01", "px": "60000"})

        results = rec.drain_user_fills()
        assert len(results) == 2
        assert results[0]["coin"] == "ETH"
        assert results[1]["coin"] == "BTC"
        assert rec.drain_user_fills() == []


class TestWSOrderReconcilerCallbacks:
    """Test the WS callback methods directly (no network)."""

    def test_on_order_update_single(self) -> None:
        rec = WSOrderReconciler(user_address="0xabc")
        msg = {"data": {"order": {"oid": 123}, "status": "filled"}}
        rec._on_order_update(msg)

        updates = rec.drain_order_updates()
        assert len(updates) == 1
        assert updates[0]["status"] == "filled"

    def test_on_order_update_list(self) -> None:
        rec = WSOrderReconciler(user_address="0xabc")
        msg = {
            "data": [
                {"order": {"oid": 1}, "status": "filled"},
                {"order": {"oid": 2}, "status": "canceled"},
            ]
        }
        rec._on_order_update(msg)

        updates = rec.drain_order_updates()
        assert len(updates) == 2

    def test_on_user_fill_single(self) -> None:
        rec = WSOrderReconciler(user_address="0xabc")
        msg = {
            "data": {
                "coin": "ETH",
                "oid": 456,
                "side": "Buy",
                "sz": "1.5",
                "px": "3100.0",
                "fee": "0.12",
                "time": 1700000000000,
            }
        }
        rec._on_user_fill(msg)

        fills = rec.drain_user_fills()
        assert len(fills) == 1
        assert fills[0]["coin"] == "ETH"

    def test_on_user_fill_list(self) -> None:
        rec = WSOrderReconciler(user_address="0xabc")
        msg = {
            "data": [
                {"coin": "ETH", "sz": "1.0", "px": "3000"},
                {"coin": "BTC", "sz": "0.1", "px": "60000"},
            ]
        }
        rec._on_user_fill(msg)

        fills = rec.drain_user_fills()
        assert len(fills) == 2

    def test_queue_overflow_drops_oldest(self) -> None:
        """When the queue is full, oldest items should be dropped."""
        rec = WSOrderReconciler(user_address="0xabc")
        # Replace with a tiny queue to test overflow
        rec._order_updates = queue.Queue(maxsize=2)
        rec._order_updates.put({"oid": 1})
        rec._order_updates.put({"oid": 2})

        # This should drop oid=1 and insert oid=3
        msg = {"data": {"oid": 3, "status": "filled"}}
        rec._on_order_update(msg)

        updates = rec.drain_order_updates()
        oids = [u.get("oid") for u in updates]
        assert 3 in oids
        assert len(updates) == 2


class TestWSOrderReconcilerLifecycle:
    """Test start/stop without real network."""

    def test_start_without_user_address_is_noop(self) -> None:
        rec = WSOrderReconciler(user_address="")
        rec.start()
        assert not rec.is_running

    @patch("autotrader.execution.ws_reconciler.HLWebSocket")
    def test_start_and_stop(self, mock_ws_cls: MagicMock) -> None:
        """Start should spawn a thread; stop should join it."""
        # Make the connect/subscribe/disconnect return fresh coroutines each call
        mock_ws = MagicMock()
        mock_ws.connect = MagicMock(side_effect=lambda *a, **kw: _make_awaitable(None))
        mock_ws.subscribe = MagicMock(side_effect=lambda *a, **kw: _make_awaitable(None))
        mock_ws.disconnect = MagicMock(side_effect=lambda *a, **kw: _make_awaitable(None))
        mock_ws_cls.return_value = mock_ws

        rec = WSOrderReconciler(
            ws_url="wss://test",
            user_address="0xabc123",
        )
        rec.start()

        # Give the thread time to start
        time.sleep(0.5)
        assert rec.is_running

        rec.stop()
        time.sleep(0.5)
        assert not rec.is_running

    def test_double_start_is_safe(self) -> None:
        rec = WSOrderReconciler(user_address="0xabc")
        # Manually mark as running to simulate
        rec._running = True
        rec.start()  # should be a no-op
        rec._running = False  # cleanup


# ---------------------------------------------------------------------------
# Scheduler integration: _normalise_ws_fill
# ---------------------------------------------------------------------------


class TestNormaliseWSFill:
    """Test the static normalisation helper used by the scheduler."""

    def test_valid_fill(self) -> None:
        from autotrader.runtime.scheduler import TradingScheduler

        ws_fill = {
            "oid": 12345,
            "coin": "ETH",
            "side": "Buy",
            "sz": "2.5",
            "px": "3100.50",
            "fee": "0.15",
            "time": 1700000000000,
        }
        result = TradingScheduler._normalise_ws_fill(ws_fill)
        assert result is not None
        assert result["order_id"] == "12345"
        assert result["symbol"] == "ETH"
        assert result["side"] == "buy"
        assert result["size"] == 2.5
        assert result["price"] == 3100.50
        assert result["fee"] == 0.15
        assert result["timestamp_ms"] == 1700000000000

    def test_empty_fill_returns_zeroes(self) -> None:
        from autotrader.runtime.scheduler import TradingScheduler

        result = TradingScheduler._normalise_ws_fill({})
        assert result is not None
        assert result["size"] == 0
        assert result["price"] == 0

    def test_invalid_fill_returns_none(self) -> None:
        from autotrader.runtime.scheduler import TradingScheduler

        result = TradingScheduler._normalise_ws_fill({"sz": "not_a_number"})
        assert result is None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import asyncio


async def _async_noop(*args, **kwargs):
    """Coroutine that returns None immediately."""
    return None


def _make_awaitable(value):
    """Create a fresh coroutine that immediately returns *value*."""
    async def _coro():
        return value
    return _coro()
