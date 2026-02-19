"""Fill and position reconciliation.

The :class:`Reconciler` bridges the :class:`~autotrader.execution.order_manager.OrderManager`
and the :class:`~autotrader.risk.exposure.ExposureTracker` by processing fills,
computing realised PnL, synchronising with exchange-reported positions, and
maintaining rolling daily/weekly PnL accumulators.
"""

from __future__ import annotations

import math
from typing import Any

import structlog

from autotrader.execution.order_manager import ManagedOrder, OrderManager
from autotrader.monitoring.metrics import metrics
from autotrader.risk.exposure import ExposureTracker, Position
from autotrader.utils.time import MS_PER_DAY, now_ms

logger = structlog.get_logger(__name__)

# One week in milliseconds.
_MS_PER_WEEK = MS_PER_DAY * 7


class Reconciler:
    """Reconciles fills against managed orders and the exposure tracker.

    Parameters
    ----------
    exposure_tracker:
        The :class:`ExposureTracker` that maintains the live position set.
    order_manager:
        The :class:`OrderManager` that owns the lifecycle of every order.
    """

    def __init__(
        self,
        exposure_tracker: ExposureTracker,
        order_manager: OrderManager,
    ) -> None:
        self.exposure = exposure_tracker
        self.orders = order_manager

        # PnL tracking
        self.pnl_history: list[dict[str, Any]] = []
        self.daily_pnl: float = 0.0
        self.weekly_pnl: float = 0.0
        self.last_daily_reset: int = now_ms()
        self.last_weekly_reset: int = now_ms()

        # Set of fill keys already processed to avoid double-counting.
        self._processed_fills: set[str] = set()

    # ------------------------------------------------------------------
    # Fill processing
    # ------------------------------------------------------------------

    def process_fill(self, fill: dict[str, Any]) -> dict[str, Any]:
        """Process a single fill and update all downstream state.

        Parameters
        ----------
        fill:
            A fill dictionary as returned by the broker.  Expected keys:
            ``order_id``, ``symbol``, ``side``, ``size``, ``price``, ``fee``,
            ``timestamp_ms``.

        Returns
        -------
        dict
            Summary with keys: ``symbol``, ``side``, ``size``, ``price``,
            ``pnl``, ``fee``, ``is_close``.
        """
        ts = now_ms()
        self._maybe_reset_pnl_windows(ts)

        order_id: str = fill.get("order_id", "")
        symbol: str = fill.get("symbol", "")
        side: str = fill.get("side", "")
        size: float = float(fill.get("size", 0))
        price: float = float(fill.get("price", 0))
        fee: float = float(fill.get("fee", 0))
        fill_ts: int = int(fill.get("timestamp_ms", ts))

        # Deduplicate: build a key from order_id + fill details
        fill_key = f"{order_id}:{symbol}:{side}:{size}:{price}:{fill_ts}"
        if fill_key in self._processed_fills:
            logger.debug("fill_already_processed", fill_key=fill_key)
            return {
                "symbol": symbol,
                "side": side,
                "size": size,
                "price": price,
                "pnl": 0.0,
                "fee": fee,
                "is_close": False,
            }
        self._processed_fills.add(fill_key)

        # 1. Match to managed order and update it
        managed = self.orders.orders.get(order_id)
        if managed is not None:
            fill_status = "filled" if size >= managed.size else "partial"
            self.orders.update_order(
                order_id=order_id,
                status=fill_status,
                filled_sz=size,
                filled_px=price,
                fee=fee,
            )

        # 2. Determine if this fill opens or closes a position
        existing = self.exposure.get_position(symbol)
        realized_pnl = 0.0
        is_close = False

        if existing is not None:
            # There is an existing position
            existing_is_long = existing.side == "long"
            fill_is_long = side == "buy"

            if existing_is_long == fill_is_long:
                # Same direction: adding to position
                self._add_to_position(existing, size, price)
            else:
                # Opposite direction: reducing or closing
                is_close = True
                if size >= existing.size:
                    # Full close (or flip -- we only close here; flipping
                    # would require a second step the caller handles)
                    realized_pnl = self._compute_realized_pnl(
                        existing,
                        existing.size,
                        price,
                    )
                    self.exposure.remove_position(symbol)

                    # If size exceeds existing, open a new position for the remainder
                    remainder = size - existing.size
                    if remainder > 1e-12:
                        new_side = "long" if fill_is_long else "short"
                        new_pos = Position(
                            symbol=symbol,
                            side=new_side,
                            size=remainder,
                            entry_px=price,
                            current_px=price,
                            leverage=existing.leverage,
                        )
                        self.exposure.add_position(new_pos)
                        is_close = False  # partially a new open
                else:
                    # Partial close
                    realized_pnl = self._compute_realized_pnl(
                        existing,
                        size,
                        price,
                    )
                    existing.size -= size
                    existing.current_px = price
                    existing.unrealized_pnl = ExposureTracker._calc_unrealized_pnl(existing)
                    existing.margin_used = ExposureTracker._calc_margin(existing)
        else:
            # No existing position: open a new one
            pos_side = "long" if side == "buy" else "short"
            leverage = 1.0
            # Try to infer leverage from managed order's trade context
            if managed is not None and managed.parent_trade_id:
                leverage = self._infer_leverage(managed)

            new_pos = Position(
                symbol=symbol,
                side=pos_side,
                size=size,
                entry_px=price,
                current_px=price,
                leverage=leverage,
            )
            self.exposure.add_position(new_pos)

        # 3. Accumulate PnL
        net_pnl = realized_pnl - fee
        self.daily_pnl += net_pnl
        self.weekly_pnl += net_pnl

        # 4. Record PnL history entry
        pnl_entry: dict[str, Any] = {
            "timestamp_ms": fill_ts,
            "symbol": symbol,
            "realized_pnl": round(realized_pnl, 6),
            "fees": round(fee, 6),
        }
        self.pnl_history.append(pnl_entry)

        # 5. Emit metrics
        metrics.inc_counter("fills_processed_total")
        metrics.observe("realized_pnl_usd", realized_pnl)
        metrics.observe("fill_fee_usd", fee)
        metrics.set_gauge("daily_pnl_usd", self.daily_pnl)
        metrics.set_gauge("weekly_pnl_usd", self.weekly_pnl)
        metrics.set_gauge("open_positions_count", float(self.exposure.position_count()))
        metrics.set_gauge("total_notional_usd", self.exposure.total_notional())

        logger.info(
            "fill_processed",
            order_id=order_id,
            symbol=symbol,
            side=side,
            size=size,
            price=price,
            fee=fee,
            realized_pnl=round(realized_pnl, 6),
            is_close=is_close,
            daily_pnl=round(self.daily_pnl, 2),
        )

        return {
            "symbol": symbol,
            "side": side,
            "size": size,
            "price": price,
            "pnl": round(realized_pnl, 6),
            "fee": round(fee, 6),
            "is_close": is_close,
        }

    # ------------------------------------------------------------------
    # Position synchronisation
    # ------------------------------------------------------------------

    def sync_positions(self, exchange_positions: list[dict[str, Any]]) -> list[str]:
        """Compare the exposure tracker with exchange-reported positions.

        Any discrepancies are logged, and the exposure tracker is updated to
        match the exchange truth.

        Parameters
        ----------
        exchange_positions:
            List of position dicts as reported by the exchange.  Expected keys
            per entry: ``coin`` (symbol), ``szi`` (signed size, positive for
            long, negative for short), ``entryPx`` (entry price),
            ``positionValue`` (notional), ``leverage`` (dict with ``value``).

        Returns
        -------
        list[str]
            Human-readable descriptions of any discrepancies found.
        """
        discrepancies: list[str] = []

        # Build a set of exchange symbols for quick lookup
        exchange_map: dict[str, dict[str, Any]] = {}
        for pos in exchange_positions:
            sym = pos.get("coin", "")
            if not sym:
                continue
            szi = float(pos.get("szi", 0))
            if abs(szi) < 1e-12:
                continue
            exchange_map[sym] = pos

        # Check for positions in the tracker that are not on the exchange
        tracked_symbols = set(self.exposure.positions.keys())
        exchange_symbols = set(exchange_map.keys())

        for sym in tracked_symbols - exchange_symbols:
            discrepancies.append(
                f"Position {sym} exists in tracker but not on exchange -- removing"
            )
            self.exposure.remove_position(sym)
            logger.warning("sync_phantom_position_removed", symbol=sym)

        # Check for positions on the exchange that are not in the tracker
        for sym in exchange_symbols - tracked_symbols:
            ex_pos = exchange_map[sym]
            szi = float(ex_pos.get("szi", 0))
            entry_px = float(ex_pos.get("entryPx", 0))
            leverage_data = ex_pos.get("leverage", {})
            leverage = (
                float(leverage_data.get("value", 1))
                if isinstance(leverage_data, dict)
                else float(leverage_data or 1)
            )

            pos_side = "long" if szi > 0 else "short"
            discrepancies.append(
                f"Position {sym} exists on exchange but not in tracker "
                f"({pos_side} {abs(szi)} @ {entry_px}) -- adding"
            )
            new_pos = Position(
                symbol=sym,
                side=pos_side,
                size=abs(szi),
                entry_px=entry_px,
                current_px=entry_px,
                leverage=leverage,
            )
            self.exposure.add_position(new_pos)
            logger.warning("sync_missing_position_added", symbol=sym, side=pos_side, size=abs(szi))

        # Check for size/side mismatches in positions that exist in both
        for sym in tracked_symbols & exchange_symbols:
            tracked_pos = self.exposure.positions[sym]
            ex_pos = exchange_map[sym]
            szi = float(ex_pos.get("szi", 0))
            entry_px = float(ex_pos.get("entryPx", 0))
            leverage_data = ex_pos.get("leverage", {})
            leverage = (
                float(leverage_data.get("value", 1))
                if isinstance(leverage_data, dict)
                else float(leverage_data or 1)
            )

            ex_side = "long" if szi > 0 else "short"
            ex_size = abs(szi)

            size_mismatch = not math.isclose(tracked_pos.size, ex_size, rel_tol=1e-6)
            side_mismatch = tracked_pos.side != ex_side

            if size_mismatch or side_mismatch:
                desc_parts: list[str] = [f"Position {sym} mismatch:"]
                if side_mismatch:
                    desc_parts.append(f"side tracker={tracked_pos.side} exchange={ex_side}")
                if size_mismatch:
                    desc_parts.append(f"size tracker={tracked_pos.size:.6f} exchange={ex_size:.6f}")
                desc = " ".join(desc_parts) + " -- updating"
                discrepancies.append(desc)

                # Update the tracked position to match exchange
                updated = Position(
                    symbol=sym,
                    side=ex_side,
                    size=ex_size,
                    entry_px=entry_px,
                    current_px=tracked_pos.current_px,
                    leverage=leverage,
                )
                self.exposure.add_position(updated)
                logger.warning(
                    "sync_position_corrected",
                    symbol=sym,
                    old_side=tracked_pos.side,
                    new_side=ex_side,
                    old_size=tracked_pos.size,
                    new_size=ex_size,
                )

        if discrepancies:
            logger.warning(
                "position_sync_discrepancies",
                count=len(discrepancies),
                details=discrepancies,
            )
        else:
            logger.debug("position_sync_clean")

        return discrepancies

    # ------------------------------------------------------------------
    # PnL accessors
    # ------------------------------------------------------------------

    def get_daily_pnl(self) -> float:
        """Return accumulated daily PnL in USD."""
        self._maybe_reset_pnl_windows(now_ms())
        return self.daily_pnl

    def get_weekly_pnl(self) -> float:
        """Return accumulated weekly PnL in USD."""
        self._maybe_reset_pnl_windows(now_ms())
        return self.weekly_pnl

    def reset_daily(self) -> None:
        """Manually reset the daily PnL counter."""
        self.daily_pnl = 0.0
        self.last_daily_reset = now_ms()
        metrics.set_gauge("daily_pnl_usd", 0.0)
        logger.info("daily_pnl_reset")

    def reset_weekly(self) -> None:
        """Manually reset the weekly PnL counter."""
        self.weekly_pnl = 0.0
        self.last_weekly_reset = now_ms()
        metrics.set_gauge("weekly_pnl_usd", 0.0)
        logger.info("weekly_pnl_reset")

    def get_pnl_history(self) -> list[dict[str, Any]]:
        """Return the full PnL history as a list of event dicts.

        Each entry has keys: ``timestamp_ms``, ``symbol``, ``realized_pnl``,
        ``fees``.
        """
        return list(self.pnl_history)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_reset_pnl_windows(self, ts: int) -> None:
        """Auto-reset daily/weekly PnL if the window has elapsed."""
        if ts - self.last_daily_reset >= MS_PER_DAY:
            logger.info(
                "auto_daily_pnl_reset",
                previous_pnl=round(self.daily_pnl, 2),
            )
            self.daily_pnl = 0.0
            self.last_daily_reset = ts
            metrics.set_gauge("daily_pnl_usd", 0.0)

        if ts - self.last_weekly_reset >= _MS_PER_WEEK:
            logger.info(
                "auto_weekly_pnl_reset",
                previous_pnl=round(self.weekly_pnl, 2),
            )
            self.weekly_pnl = 0.0
            self.last_weekly_reset = ts
            metrics.set_gauge("weekly_pnl_usd", 0.0)

    @staticmethod
    def _compute_realized_pnl(
        position: Position,
        close_size: float,
        close_price: float,
    ) -> float:
        """Compute realized PnL for closing *close_size* of *position*.

        For a long position: ``(close_price - entry_px) * close_size``
        For a short position: ``(entry_px - close_price) * close_size``
        """
        if position.side == "long":
            return (close_price - position.entry_px) * close_size
        else:
            return (position.entry_px - close_price) * close_size

    @staticmethod
    def _add_to_position(
        position: Position,
        add_size: float,
        add_price: float,
    ) -> None:
        """Increase a position size using volume-weighted average entry price."""
        prev_notional = position.size * position.entry_px
        new_notional = add_size * add_price
        total_size = position.size + add_size
        if total_size > 0:
            position.entry_px = (prev_notional + new_notional) / total_size
        position.size = total_size
        position.current_px = add_price
        position.unrealized_pnl = ExposureTracker._calc_unrealized_pnl(position)
        position.margin_used = ExposureTracker._calc_margin(position)

    @staticmethod
    def _infer_leverage(managed: ManagedOrder) -> float:
        """Try to infer leverage from a managed order's context.

        Defaults to ``1.0`` if no leverage information is available.
        """
        # Leverage is not directly stored on ManagedOrder; default to 1x.
        return 1.0
