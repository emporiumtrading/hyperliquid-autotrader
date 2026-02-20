"""Extended execution tests: partial fills, order state transitions, trailing stops,
reconciliation, close_all_positions, slippage edge cases.

These tests complement the existing test_execution.py with coverage for the gaps
identified in dimension D11 of the PRD variance analysis.
"""

from __future__ import annotations

import pytest

from autotrader.execution.broker import Broker, OrderResult
from autotrader.execution.order_manager import ManagedOrder, OrderManager, OrderState
from autotrader.execution.reconciliation import Reconciler
from autotrader.execution.slippage import SlippageModel
from autotrader.risk.exposure import ExposureTracker, Position


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _paper_broker() -> Broker:
    return Broker(client=None, mode="paper")


def _make_reconciler() -> tuple[Reconciler, ExposureTracker, OrderManager]:
    broker = _paper_broker()
    exposure = ExposureTracker()
    manager = OrderManager(broker)
    reconciler = Reconciler(exposure, manager)
    return reconciler, exposure, manager


# ===========================================================================
# SlippageModel edge cases
# ===========================================================================


class TestSlippageEdgeCases:
    def test_zero_notional(self):
        """Zero notional produces zero slippage."""
        model = SlippageModel(base_bps=1.0, size_impact_factor=0.1)
        assert model.estimate(notional=0.0) == 0.0

    def test_capped_at_max_bps(self):
        """Extremely large orders get capped at 50 bps."""
        model = SlippageModel(base_bps=1.0, size_impact_factor=0.1)
        bps = model.estimate_bps(notional=100_000_000.0, depth_usd=1_000.0)
        assert bps <= 50.0

    def test_depth_penalty_activates(self):
        """Orders exceeding 10% of depth incur extra cost."""
        model = SlippageModel(base_bps=1.0, size_impact_factor=0.0)
        # Small order: no depth penalty
        small = model.estimate_bps(notional=5_000.0, spread_bps=0.0, depth_usd=100_000.0)
        # Large order exceeding 10% of depth
        large = model.estimate_bps(notional=50_000.0, spread_bps=0.0, depth_usd=100_000.0)
        assert large > small

    def test_spread_component(self):
        """Wider spread increases slippage."""
        model = SlippageModel(base_bps=0.0, size_impact_factor=0.0)
        narrow = model.estimate_bps(notional=10_000.0, spread_bps=1.0, depth_usd=1_000_000.0)
        wide = model.estimate_bps(notional=10_000.0, spread_bps=10.0, depth_usd=1_000_000.0)
        assert wide > narrow


# ===========================================================================
# Paper Broker extended tests
# ===========================================================================


class TestPaperBrokerSell:
    def test_sell_order_slippage_direction(self):
        """Sell orders should have fill price below the requested price (slippage)."""
        broker = _paper_broker()
        result = broker.place_order(
            symbol="ETH", side="sell", size=1.0, price=3000.0, order_type="limit"
        )
        assert result.status == "filled"
        assert result.filled_px < 3000.0  # slippage pushes sell price down


class TestPaperBrokerMarketOrder:
    def test_market_order_fills(self):
        """Market orders fill in paper mode."""
        broker = _paper_broker()
        result = broker.place_order(
            symbol="BTC", side="buy", size=0.5, price=50000.0, order_type="market"
        )
        assert result.status == "filled"
        assert result.filled_sz == 0.5


class TestPaperBrokerReduceOnly:
    def test_reduce_only_flag(self):
        """Paper broker handles reduce_only orders."""
        broker = _paper_broker()
        result = broker.place_order(
            symbol="SOL", side="sell", size=10.0, price=100.0,
            order_type="market", reduce_only=True, tif="Ioc",
        )
        assert result.status == "filled"


class TestCloseAllPositions:
    def test_close_all_positions_basic(self):
        """close_all_positions generates reverse orders for each position."""
        broker = _paper_broker()
        positions = [
            {"symbol": "ETH", "side": "long", "size": 5.0, "current_px": 3000.0},
            {"symbol": "BTC", "side": "short", "size": 0.1, "current_px": 50000.0},
        ]
        results = broker.close_all_positions(positions)
        assert len(results) == 2
        assert all(r.status == "filled" for r in results)

    def test_close_all_positions_empty(self):
        """Empty position list returns empty results."""
        broker = _paper_broker()
        results = broker.close_all_positions([])
        assert results == []

    def test_close_all_positions_none(self):
        """None position list returns empty results."""
        broker = _paper_broker()
        results = broker.close_all_positions(None)
        assert results == []

    def test_close_all_positions_skips_zero_size(self):
        """Positions with zero size are skipped."""
        broker = _paper_broker()
        positions = [
            {"symbol": "ETH", "side": "long", "size": 0.0, "current_px": 3000.0},
        ]
        results = broker.close_all_positions(positions)
        assert results == []


# ===========================================================================
# OrderManager: update_order state transitions
# ===========================================================================


class TestOrderManagerUpdateOrder:
    def test_update_to_partial_fill(self):
        """update_order correctly handles partial fills."""
        broker = _paper_broker()
        manager = OrderManager(broker)

        # Submit entry
        managed = manager.submit_entry(
            symbol="ETH", side="buy", size=10.0, price=3000.0,
            trade_id="t1", stop_px=2900.0, tp_px=3200.0,
        )
        oid = managed.order_id

        # Simulate a partial fill update externally (overriding paper broker behavior)
        manager.orders[oid].state = OrderState.SUBMITTED
        manager.orders[oid].filled_size = 0.0
        manager.orders[oid].filled_price = 0.0
        manager.orders[oid].protective_placed = False

        manager.update_order(oid, status="partial", filled_sz=3.0, filled_px=3001.0, fee=0.5)
        assert manager.orders[oid].state == OrderState.PARTIAL
        assert abs(manager.orders[oid].filled_size - 3.0) < 1e-6
        assert abs(manager.orders[oid].filled_price - 3001.0) < 1e-6

    def test_update_cumulative_fills(self):
        """Multiple partial fills accumulate correctly with VWAP."""
        broker = _paper_broker()
        manager = OrderManager(broker)

        managed = manager.submit_entry(
            symbol="ETH", side="buy", size=10.0, price=3000.0,
            trade_id="t2",
        )
        oid = managed.order_id

        # Reset to simulate live (non-paper) flow
        manager.orders[oid].state = OrderState.SUBMITTED
        manager.orders[oid].filled_size = 0.0
        manager.orders[oid].filled_price = 0.0
        manager.orders[oid].fee = 0.0

        # Fill 1: 4 units @ 3000
        manager.update_order(oid, status="partial", filled_sz=4.0, filled_px=3000.0, fee=1.0)
        assert abs(manager.orders[oid].filled_size - 4.0) < 1e-6

        # Fill 2: 6 units @ 3010
        manager.update_order(oid, status="filled", filled_sz=6.0, filled_px=3010.0, fee=1.5)
        assert abs(manager.orders[oid].filled_size - 10.0) < 1e-6
        # VWAP: (4*3000 + 6*3010) / 10 = 30060/10 = 3006
        assert abs(manager.orders[oid].filled_price - 3006.0) < 1e-6
        assert abs(manager.orders[oid].fee - 2.5) < 1e-6
        assert manager.orders[oid].state == OrderState.FILLED

    def test_update_nonexistent_order(self):
        """Updating a nonexistent order is a no-op."""
        broker = _paper_broker()
        manager = OrderManager(broker)
        # Should not raise
        manager.update_order("nonexistent_id", status="filled", filled_sz=1.0, filled_px=100.0)

    def test_protective_orders_on_partial_fill(self):
        """Protective orders are placed on first partial fill."""
        broker = _paper_broker()
        manager = OrderManager(broker)

        managed = manager.submit_entry(
            symbol="ETH", side="buy", size=10.0, price=3000.0,
            trade_id="t3", stop_px=2900.0, tp_px=3200.0,
        )
        oid = managed.order_id

        # Reset to simulate pending state
        manager.orders[oid].state = OrderState.SUBMITTED
        manager.orders[oid].filled_size = 0.0
        manager.orders[oid].protective_placed = False

        initial_count = len(manager.orders)

        # First partial fill should trigger protective orders
        manager.update_order(oid, status="partial", filled_sz=3.0, filled_px=3001.0)
        assert manager.orders[oid].protective_placed is True
        # Should have added SL + TP orders
        assert len(manager.orders) > initial_count


# ===========================================================================
# OrderManager: cancel_trade_orders
# ===========================================================================


class TestCancelTradeOrders:
    def test_cancel_pending_orders(self):
        """cancel_trade_orders cancels active orders for a trade."""
        broker = _paper_broker()
        manager = OrderManager(broker)

        managed = manager.submit_entry(
            symbol="ETH", side="buy", size=5.0, price=3000.0,
            trade_id="cancel_test", stop_px=2900.0, tp_px=3200.0,
        )
        # Entry is filled (paper), but SL and TP are pending
        trade_orders = manager.get_orders_for_trade("cancel_test")
        pending = [o for o in trade_orders if o.state in (OrderState.PENDING, OrderState.SUBMITTED)]
        assert len(pending) >= 2  # SL + TP

        cancelled = manager.cancel_trade_orders("cancel_test")
        assert cancelled >= 2

    def test_cancel_nonexistent_trade(self):
        """Cancelling a nonexistent trade returns 0."""
        broker = _paper_broker()
        manager = OrderManager(broker)
        assert manager.cancel_trade_orders("no_such_trade") == 0


# ===========================================================================
# OrderManager: trailing stops
# ===========================================================================


class TestTrailingStop:
    def test_trailing_stop_moves_up_for_long(self):
        """Trailing stop moves up when price rises for a long position."""
        broker = _paper_broker()
        manager = OrderManager(broker)

        manager.submit_entry(
            symbol="ETH", side="buy", size=5.0, price=3000.0,
            trade_id="trail_long", stop_px=2800.0, tp_px=3500.0,
        )

        # Price rises significantly
        moved = manager.update_trailing_stop(
            trade_id="trail_long",
            current_price=3300.0,
            trail_atr=50.0,
            trail_multiplier=2.0,
        )
        # New stop = 3300 - 100 = 3200 > old stop 2800 → should move
        assert moved is True

    def test_trailing_stop_doesnt_move_down_for_long(self):
        """Trailing stop should not move down for a long position."""
        broker = _paper_broker()
        manager = OrderManager(broker)

        manager.submit_entry(
            symbol="ETH", side="buy", size=5.0, price=3000.0,
            trade_id="trail_no_down", stop_px=2900.0, tp_px=3500.0,
        )

        # Price drops below entry — new stop would be below old stop
        moved = manager.update_trailing_stop(
            trade_id="trail_no_down",
            current_price=2950.0,
            trail_atr=50.0,
            trail_multiplier=2.0,
        )
        # New stop = 2950 - 100 = 2850 < old stop 2900 → should NOT move
        assert moved is False

    def test_trailing_stop_short(self):
        """Trailing stop moves down when price drops for a short position."""
        broker = _paper_broker()
        manager = OrderManager(broker)

        manager.submit_entry(
            symbol="ETH", side="sell", size=5.0, price=3000.0,
            trade_id="trail_short", stop_px=3200.0, tp_px=2700.0,
        )

        # Price drops
        moved = manager.update_trailing_stop(
            trade_id="trail_short",
            current_price=2800.0,
            trail_atr=50.0,
            trail_multiplier=2.0,
        )
        # New stop = 2800 + 100 = 2900 < old stop 3200 → should move
        assert moved is True

    def test_trailing_stop_no_atr(self):
        """Zero ATR returns False."""
        broker = _paper_broker()
        manager = OrderManager(broker)

        manager.submit_entry(
            symbol="ETH", side="buy", size=5.0, price=3000.0,
            trade_id="trail_zero", stop_px=2800.0, tp_px=3500.0,
        )
        moved = manager.update_trailing_stop(
            trade_id="trail_zero", current_price=3300.0, trail_atr=0.0
        )
        assert moved is False

    def test_active_trade_symbols(self):
        """active_trade_symbols returns trades with filled entries and active stops."""
        broker = _paper_broker()
        manager = OrderManager(broker)

        manager.submit_entry(
            symbol="ETH", side="buy", size=5.0, price=3000.0,
            trade_id="active1", stop_px=2800.0, tp_px=3500.0,
        )
        manager.submit_entry(
            symbol="BTC", side="sell", size=0.1, price=50000.0,
            trade_id="active2", stop_px=52000.0, tp_px=45000.0,
        )

        active = manager.active_trade_symbols()
        symbols = {sym for _, sym in active}
        assert "ETH" in symbols
        assert "BTC" in symbols


# ===========================================================================
# OrderManager: queries
# ===========================================================================


class TestOrderManagerQueries:
    def test_get_active_orders_filter(self):
        """get_active_orders filters by symbol."""
        broker = _paper_broker()
        manager = OrderManager(broker)

        manager.submit_entry(
            symbol="ETH", side="buy", size=5.0, price=3000.0,
            trade_id="q1", stop_px=2800.0, tp_px=3500.0,
        )
        manager.submit_entry(
            symbol="BTC", side="buy", size=0.1, price=50000.0,
            trade_id="q2", stop_px=48000.0, tp_px=55000.0,
        )

        eth_active = manager.get_active_orders(symbol="ETH")
        btc_active = manager.get_active_orders(symbol="BTC")

        # All ETH active orders should be for ETH
        for o in eth_active:
            assert o.symbol == "ETH"
        for o in btc_active:
            assert o.symbol == "BTC"

    def test_get_orders_for_unknown_trade(self):
        """get_orders_for_trade returns empty for unknown trade."""
        broker = _paper_broker()
        manager = OrderManager(broker)
        assert manager.get_orders_for_trade("unknown") == []


# ===========================================================================
# Reconciler: process_fill
# ===========================================================================


class TestReconcilerProcessFill:
    def test_open_new_position(self):
        """A fill with no existing position opens a new one."""
        rec, exposure, _ = _make_reconciler()
        result = rec.process_fill({
            "order_id": "o1",
            "symbol": "ETH",
            "side": "buy",
            "size": 5.0,
            "price": 3000.0,
            "fee": 1.5,
            "timestamp_ms": 1000,
        })
        assert result["is_close"] is False
        assert exposure.position_count() == 1
        pos = exposure.get_position("ETH")
        assert pos.side == "long"
        assert pos.size == 5.0

    def test_close_position_pnl(self):
        """Closing a position computes correct realized PnL."""
        rec, exposure, _ = _make_reconciler()

        # Open long
        rec.process_fill({
            "order_id": "o1", "symbol": "ETH", "side": "buy",
            "size": 5.0, "price": 3000.0, "fee": 1.0, "timestamp_ms": 1000,
        })

        # Close long at higher price
        result = rec.process_fill({
            "order_id": "o2", "symbol": "ETH", "side": "sell",
            "size": 5.0, "price": 3100.0, "fee": 1.0, "timestamp_ms": 2000,
        })
        assert result["is_close"] is True
        # PnL = (3100 - 3000) * 5 = 500
        assert abs(result["pnl"] - 500.0) < 1e-6
        assert exposure.position_count() == 0

    def test_close_short_position_pnl(self):
        """Closing a short position computes correct PnL."""
        rec, exposure, _ = _make_reconciler()

        # Open short
        rec.process_fill({
            "order_id": "o1", "symbol": "BTC", "side": "sell",
            "size": 1.0, "price": 50000.0, "fee": 0.5, "timestamp_ms": 1000,
        })

        # Close short at lower price (profit)
        result = rec.process_fill({
            "order_id": "o2", "symbol": "BTC", "side": "buy",
            "size": 1.0, "price": 49000.0, "fee": 0.5, "timestamp_ms": 2000,
        })
        assert result["is_close"] is True
        # PnL = (50000 - 49000) * 1 = 1000
        assert abs(result["pnl"] - 1000.0) < 1e-6

    def test_partial_close(self):
        """Partial close reduces position but keeps remainder."""
        rec, exposure, _ = _make_reconciler()

        rec.process_fill({
            "order_id": "o1", "symbol": "ETH", "side": "buy",
            "size": 10.0, "price": 3000.0, "fee": 1.0, "timestamp_ms": 1000,
        })

        result = rec.process_fill({
            "order_id": "o2", "symbol": "ETH", "side": "sell",
            "size": 4.0, "price": 3050.0, "fee": 0.5, "timestamp_ms": 2000,
        })
        assert result["is_close"] is True
        # PnL = (3050 - 3000) * 4 = 200
        assert abs(result["pnl"] - 200.0) < 1e-6
        # Remaining position: 6 units
        pos = exposure.get_position("ETH")
        assert abs(pos.size - 6.0) < 1e-6

    def test_add_to_existing_position(self):
        """Buying more of an existing long adds to position with VWAP entry."""
        rec, exposure, _ = _make_reconciler()

        rec.process_fill({
            "order_id": "o1", "symbol": "ETH", "side": "buy",
            "size": 5.0, "price": 3000.0, "fee": 1.0, "timestamp_ms": 1000,
        })
        rec.process_fill({
            "order_id": "o2", "symbol": "ETH", "side": "buy",
            "size": 5.0, "price": 3100.0, "fee": 1.0, "timestamp_ms": 2000,
        })

        pos = exposure.get_position("ETH")
        assert abs(pos.size - 10.0) < 1e-6
        # VWAP entry: (5*3000 + 5*3100) / 10 = 3050
        assert abs(pos.entry_px - 3050.0) < 1e-6

    def test_fill_deduplication(self):
        """Duplicate fills are not double-counted."""
        rec, exposure, _ = _make_reconciler()

        fill = {
            "order_id": "o1", "symbol": "ETH", "side": "buy",
            "size": 5.0, "price": 3000.0, "fee": 1.0, "timestamp_ms": 1000,
        }
        rec.process_fill(fill)
        rec.process_fill(fill)  # duplicate

        pos = exposure.get_position("ETH")
        assert abs(pos.size - 5.0) < 1e-6  # should not be 10

    def test_pnl_accumulation(self):
        """Daily and weekly PnL accumulators track correctly."""
        rec, _, _ = _make_reconciler()

        # Open and close with profit
        rec.process_fill({
            "order_id": "o1", "symbol": "ETH", "side": "buy",
            "size": 10.0, "price": 3000.0, "fee": 2.0, "timestamp_ms": 1000,
        })
        rec.process_fill({
            "order_id": "o2", "symbol": "ETH", "side": "sell",
            "size": 10.0, "price": 3050.0, "fee": 2.0, "timestamp_ms": 2000,
        })

        # PnL = 500, fees = 4 total for close + open
        # daily_pnl = net of all fills:
        #   fill1: realized=0, fee=2, net=-2
        #   fill2: realized=500, fee=2, net=498
        assert rec.daily_pnl > 0
        assert rec.weekly_pnl > 0

    def test_pnl_history(self):
        """PnL history is populated with each fill."""
        rec, _, _ = _make_reconciler()
        rec.process_fill({
            "order_id": "o1", "symbol": "ETH", "side": "buy",
            "size": 1.0, "price": 100.0, "fee": 0.1, "timestamp_ms": 1000,
        })
        history = rec.get_pnl_history()
        assert len(history) == 1
        assert history[0]["symbol"] == "ETH"


# ===========================================================================
# Reconciler: sync_positions
# ===========================================================================


class TestReconcilerSyncPositions:
    def test_sync_adds_missing_position(self):
        """Positions on exchange but not in tracker are added."""
        rec, exposure, _ = _make_reconciler()
        exchange = [
            {"coin": "ETH", "szi": "5.0", "entryPx": "3000", "leverage": {"value": "3"}},
        ]
        discrepancies = rec.sync_positions(exchange)
        assert len(discrepancies) == 1
        assert "adding" in discrepancies[0].lower()
        assert exposure.position_count() == 1

    def test_sync_removes_phantom_position(self):
        """Positions in tracker but not on exchange are removed."""
        rec, exposure, _ = _make_reconciler()
        exposure.add_position(Position("ETH", "long", 5.0, 3000.0, 3000.0, 3.0))
        discrepancies = rec.sync_positions([])  # empty exchange
        assert len(discrepancies) == 1
        assert "removing" in discrepancies[0].lower()
        assert exposure.position_count() == 0

    def test_sync_corrects_size_mismatch(self):
        """Size mismatch between tracker and exchange is corrected."""
        rec, exposure, _ = _make_reconciler()
        exposure.add_position(Position("ETH", "long", 5.0, 3000.0, 3000.0, 3.0))
        exchange = [
            {"coin": "ETH", "szi": "8.0", "entryPx": "3000", "leverage": {"value": "3"}},
        ]
        discrepancies = rec.sync_positions(exchange)
        assert len(discrepancies) == 1
        assert "size" in discrepancies[0].lower()
        pos = exposure.get_position("ETH")
        assert abs(pos.size - 8.0) < 1e-6

    def test_sync_clean(self):
        """No discrepancies when tracker matches exchange."""
        rec, exposure, _ = _make_reconciler()
        exposure.add_position(Position("ETH", "long", 5.0, 3000.0, 3000.0, 3.0))
        exchange = [
            {"coin": "ETH", "szi": "5.0", "entryPx": "3000", "leverage": {"value": "3"}},
        ]
        discrepancies = rec.sync_positions(exchange)
        assert discrepancies == []

    def test_sync_side_mismatch(self):
        """Side flip between tracker and exchange is corrected."""
        rec, exposure, _ = _make_reconciler()
        exposure.add_position(Position("ETH", "long", 5.0, 3000.0, 3000.0, 3.0))
        exchange = [
            {"coin": "ETH", "szi": "-5.0", "entryPx": "3100", "leverage": {"value": "3"}},
        ]
        discrepancies = rec.sync_positions(exchange)
        assert len(discrepancies) == 1
        assert "side" in discrepancies[0].lower()
        pos = exposure.get_position("ETH")
        assert pos.side == "short"

    def test_sync_skips_zero_size(self):
        """Exchange positions with zero size are ignored."""
        rec, exposure, _ = _make_reconciler()
        exchange = [
            {"coin": "ETH", "szi": "0.0", "entryPx": "3000", "leverage": {"value": "1"}},
        ]
        discrepancies = rec.sync_positions(exchange)
        assert discrepancies == []
        assert exposure.position_count() == 0


# ===========================================================================
# Reconciler: PnL window resets
# ===========================================================================


class TestPnLWindows:
    def test_manual_daily_reset(self):
        rec, _, _ = _make_reconciler()
        rec.daily_pnl = 500.0
        rec.reset_daily()
        assert rec.daily_pnl == 0.0

    def test_manual_weekly_reset(self):
        rec, _, _ = _make_reconciler()
        rec.weekly_pnl = 1000.0
        rec.reset_weekly()
        assert rec.weekly_pnl == 0.0


# ===========================================================================
# ManagedOrder properties
# ===========================================================================


class TestManagedOrderProperties:
    def test_remaining_size(self):
        order = ManagedOrder(
            order_id="x", symbol="ETH", side="buy", size=10.0,
            price=3000.0, order_type="limit", state=OrderState.PARTIAL,
            filled_size=4.0,
        )
        assert abs(order.remaining_size - 6.0) < 1e-6

    def test_remaining_size_fully_filled(self):
        order = ManagedOrder(
            order_id="x", symbol="ETH", side="buy", size=10.0,
            price=3000.0, order_type="limit", state=OrderState.FILLED,
            filled_size=10.0,
        )
        assert order.remaining_size == 0.0

    def test_timestamp_alias(self):
        order = ManagedOrder(
            order_id="x", symbol="ETH", side="buy", size=10.0,
            price=3000.0, order_type="limit", state=OrderState.PENDING,
            created_at=12345,
        )
        assert order.timestamp_ms == 12345

    def test_trade_id_alias(self):
        order = ManagedOrder(
            order_id="x", symbol="ETH", side="buy", size=10.0,
            price=3000.0, order_type="limit", state=OrderState.PENDING,
            parent_trade_id="abc",
        )
        assert order.trade_id == "abc"

    def test_default_metadata(self):
        order = ManagedOrder(
            order_id="x", symbol="ETH", side="buy", size=10.0,
            price=3000.0, order_type="limit", state=OrderState.PENDING,
        )
        assert order.metadata == {}
