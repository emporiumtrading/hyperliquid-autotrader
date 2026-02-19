"""Tests for execution layer: slippage model, paper broker, order manager."""

from __future__ import annotations

from autotrader.execution.broker import Broker, OrderResult
from autotrader.execution.order_manager import OrderManager, OrderState
from autotrader.execution.slippage import SlippageModel

# ---------------------------------------------------------------------------
# SlippageModel tests
# ---------------------------------------------------------------------------


class TestSlippageModelBasic:
    def test_slippage_model_basic(self):
        """Basic slippage estimate returns a positive USD value."""
        model = SlippageModel(base_bps=1.0, size_impact_factor=0.1)
        slip = model.estimate(notional=50_000.0, spread_bps=1.0, depth_usd=1_000_000.0)
        assert slip > 0.0
        assert isinstance(slip, float)

    def test_slippage_increases_with_size(self):
        """Larger orders should have more slippage in both absolute and per-dollar terms."""
        model = SlippageModel(base_bps=1.0, size_impact_factor=0.1)

        slip_small = model.estimate(notional=10_000.0, spread_bps=1.0, depth_usd=1_000_000.0)
        slip_large = model.estimate(notional=500_000.0, spread_bps=1.0, depth_usd=1_000_000.0)

        assert slip_large > slip_small

        # Per-dollar slippage also increases because of size-dependent component
        per_dollar_small = slip_small / 10_000.0
        per_dollar_large = slip_large / 500_000.0
        assert per_dollar_large > per_dollar_small


# ---------------------------------------------------------------------------
# Paper Broker tests
# ---------------------------------------------------------------------------


class TestPaperBrokerFill:
    def test_paper_broker_fill(self):
        """Paper broker fills orders immediately with slippage applied."""
        broker = Broker(client=None, mode="paper")
        result = broker.place_order(
            symbol="ETH",
            side="buy",
            size=1.0,
            price=3000.0,
            order_type="limit",
        )

        assert isinstance(result, OrderResult)
        assert result.status == "filled"
        assert result.filled_sz == 1.0
        assert result.remaining_sz == 0.0
        assert result.order_id != ""
        # Fill price should be above the requested price for a buy (slippage)
        assert result.filled_px > 3000.0
        assert result.fee > 0.0


class TestPaperBrokerCancel:
    def test_paper_broker_cancel(self):
        """Cancelling a pending trigger order works in paper mode."""
        broker = Broker(client=None, mode="paper")

        # Place a trigger order (pending, not immediately filled)
        trigger_result = broker.place_trigger_order(
            symbol="ETH",
            side="sell",
            size=1.0,
            trigger_px=2900.0,
            order_type="stop_loss",
        )
        assert trigger_result.status == "pending"
        order_id = trigger_result.order_id

        # Verify it's in pending orders
        open_orders = broker.get_open_orders(symbol="ETH")
        assert any(o["order_id"] == order_id for o in open_orders)

        # Cancel it
        cancelled = broker.cancel_order("ETH", order_id)
        assert cancelled is True

        # No longer in open orders
        open_orders_after = broker.get_open_orders(symbol="ETH")
        assert not any(o["order_id"] == order_id for o in open_orders_after)


# ---------------------------------------------------------------------------
# OrderManager tests
# ---------------------------------------------------------------------------


class TestOrderManagerSubmit:
    def test_order_manager_submit(self):
        """submit_entry creates a managed order and tracks it."""
        broker = Broker(client=None, mode="paper")
        manager = OrderManager(broker)

        managed = manager.submit_entry(
            symbol="ETH",
            side="buy",
            size=2.0,
            price=3000.0,
            trade_id="trade_001",
            stop_px=2900.0,
            tp_px=3200.0,
        )

        assert managed.state == OrderState.FILLED  # paper mode fills immediately
        assert managed.symbol == "ETH"
        assert managed.side == "buy"
        assert managed.parent_trade_id == "trade_001"
        assert managed.filled_size == 2.0
        assert managed.filled_price > 0

        # The order should be tracked in the manager
        assert managed.order_id in manager.orders
        assert "trade_001" in manager.trade_orders
        assert managed.order_id in manager.trade_orders["trade_001"]


class TestOrderManagerLifecycle:
    def test_order_manager_lifecycle(self):
        """Full order lifecycle: submit entry -> verify fill -> submit exit."""
        broker = Broker(client=None, mode="paper")
        manager = OrderManager(broker)

        # 1. Submit entry
        entry = manager.submit_entry(
            symbol="BTC",
            side="buy",
            size=0.1,
            price=50_000.0,
            trade_id="trade_lifecycle",
            stop_px=49_000.0,
            tp_px=52_000.0,
        )
        assert entry.state == OrderState.FILLED
        assert entry.filled_size == 0.1

        # 2. Verify protective orders were placed (stop + TP)
        trade_orders = manager.get_orders_for_trade("trade_lifecycle")
        # Should have: entry + stop_loss + take_profit = 3 orders
        assert len(trade_orders) == 3

        # Check stop loss order exists
        sl_orders = [o for o in trade_orders if o.order_type == "stop_loss"]
        assert len(sl_orders) == 1
        assert sl_orders[0].stop_px == 49_000.0

        # Check take profit order exists
        tp_orders = [o for o in trade_orders if o.order_type == "take_profit"]
        assert len(tp_orders) == 1
        assert tp_orders[0].tp_px == 52_000.0

        # 3. Submit exit
        exit_order = manager.submit_exit(
            symbol="BTC",
            trade_id="trade_lifecycle",
            size=0.1,
            price=51_000.0,
            reason="signal",
        )
        assert exit_order.state == OrderState.FILLED
        assert exit_order.side == "sell"  # opposite of entry buy
        assert exit_order.filled_size == 0.1

        # 4. All orders for the trade should now exist
        all_orders = manager.get_orders_for_trade("trade_lifecycle")
        assert len(all_orders) == 4  # entry + SL + TP + exit
