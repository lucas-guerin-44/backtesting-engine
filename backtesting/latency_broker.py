"""Latency-aware execution layer wrapping the core Broker.

Models order lifecycle: submitted → acknowledged → filled, with
configurable network and queue delays. Supports MARKET, LIMIT, and STOP
order types, using tick.bid/ask when available for realistic fill prices.

Two execution modes
-------------------
Without order_book (default)
    Simple price-comparison fills. A LIMIT order fills the moment the
    market-side price crosses the limit. No FIFO, no partial fills.

With order_book
    Orders are routed through a MatchingEngine. Resting limit orders
    queue in the book and drain FIFO per price level. Partial fills are
    supported via ``max_qty_per_level`` on the MatchingEngine.

Usage
-----
>>> from backtesting.latency_broker import LatencyAwareBroker
>>> from backtesting.order_book import OrderBook, MatchingEngine
>>> from backtesting.order import Order, OrderType
>>>
>>> book = OrderBook()
>>> engine = MatchingEngine(book, max_qty_per_level=50.0)
>>> lb = LatencyAwareBroker(
...     broker=portfolio.broker,
...     ack_latency_ns=10_000_000,
...     order_book=engine,
... )
>>> bt = TickBacktester(ticks, strategy, latency_broker=lb)
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import pandas as pd

from backtesting.broker import Broker
from backtesting.latency_models import LatencyModel
from backtesting.order import Order, OrderType, PendingOrder
from backtesting.order_book import Fill, MatchingEngine
from backtesting.tick import Tick
from backtesting.types import Bar

if TYPE_CHECKING:
    from backtesting.latency_metrics import FillRecord


class LatencyAwareBroker:
    """Wraps a Broker to simulate order lifecycle latency.

    Orders submitted via ``submit()`` enter a pending queue tagged with
    a nanosecond fill-eligibility timestamp. On each tick, ``process_tick()``
    scans the queue and fills any matured orders.

    Without ``order_book``: simple price-comparison fills.
    With ``order_book``: fills via MatchingEngine with FIFO and partial fills.

    Parameters
    ----------
    broker : Broker
        Underlying broker. All fills delegate to broker.open_trade().
    ack_latency_ns : int
        Nanoseconds from submission to fill (used when ``latency_model`` is None).
    fill_latency_ns : int
        Additional nanoseconds added to ``ack_latency_ns`` (used when
        ``latency_model`` is None).
    order_book : MatchingEngine, optional
        If provided, routes fills through the matching engine instead of
        simple price comparison.
    latency_model : LatencyModel, optional
        If provided, each order samples its delay from this model instead
        of using the fixed ``ack_latency_ns + fill_latency_ns`` values.
        See ``backtesting.latency_models`` for available models.
    """

    def __init__(
        self,
        broker: Broker,
        ack_latency_ns: int = 0,
        fill_latency_ns: int = 0,
        order_book: Optional[MatchingEngine] = None,
        latency_model: Optional[LatencyModel] = None,
    ) -> None:
        self._broker = broker
        self.ack_latency_ns = ack_latency_ns
        self.fill_latency_ns = fill_latency_ns
        self._engine = order_book
        self._latency_model = latency_model
        self._pending: List[PendingOrder] = []
        # Resting orders inside the engine book: order_id -> (Order, qty_remaining)
        self._resting: Dict[str, Tuple[Order, float]] = {}
        # Fill records for latency metrics collection
        # Populated by _execute() / _execute_fill() when market_price is captured.
        self._fill_records: List = []
        # Submission-time price index: order_id -> market_price at submit
        self._submit_prices: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Pass-throughs so TickBacktester treats this as a drop-in for Broker
    # ------------------------------------------------------------------

    @property
    def positions(self):
        return self._broker.positions

    @property
    def closed_trades(self):
        return self._broker.closed_trades

    def has_open_position(self, symbol: str) -> bool:
        return self._broker.has_open_position(symbol)

    def position_side(self, symbol: str) -> int:
        return self._broker.position_side(symbol)

    def update_stop(self, symbol: str, new_stop: float) -> None:
        self._broker.update_stop(symbol, new_stop)

    # ------------------------------------------------------------------
    # Latency queue
    # ------------------------------------------------------------------

    @property
    def pending_count(self) -> int:
        """Orders in the time-delay queue + orders resting in the book."""
        return len(self._pending) + len(self._resting)

    @property
    def fill_records(self) -> List:
        """FillRecord objects collected during this run (latency metrics)."""
        return self._fill_records

    def submit(self, order: Order, ts: pd.Timestamp, market_price: float = 0.0) -> None:
        """Enqueue an order with a nanosecond fill-eligibility timestamp.

        Parameters
        ----------
        order : Order
        ts : pd.Timestamp
            Submission timestamp (used to compute ``fill_after_ns``).
        market_price : float, optional
            Best market price at submission time (ask for buys, bid for sells).
            Captured for slippage calculation in latency metrics. 0.0 = unknown.
        """
        if self._latency_model is not None:
            delay_ns = self._latency_model.sample_ns()
        else:
            delay_ns = self.ack_latency_ns + self.fill_latency_ns
        fill_after_ns = ts.value + delay_ns
        self._pending.append(PendingOrder(order=order, fill_after_ns=fill_after_ns))
        if market_price > 0:
            self._submit_prices[order.order_id] = market_price

    def process_tick(self, tick: Tick) -> None:
        """Advance the execution layer by one tick.

        Called by TickBacktester on every tick before strategy signals.
        Drains the latency queue and, when an order_book is wired in,
        also drains resting limit orders from the book.
        """
        if self._engine is not None:
            self._process_tick_with_engine(tick)
        else:
            self._process_tick_simple(tick)

    # ------------------------------------------------------------------
    # Engine path (order_book wired in)
    # ------------------------------------------------------------------

    def _process_tick_with_engine(self, tick: Tick) -> None:
        """Tick processing when an order book / matching engine is wired in."""
        # Step 1: advance the book and drain resting orders
        resting_fills = self._engine.process_tick(tick)
        for fill in resting_fills:
            self._apply_resting_fill(fill, tick)

        # Step 2: process time-matured pending orders
        tick_ns = tick.ts.value
        tick_price = tick.price
        still_pending: List[PendingOrder] = []

        for po in self._pending:
            order = po.order

            if order.type == OrderType.STOP:
                if not po.activated:
                    if order.side > 0 and tick_price >= order.stop_trigger:
                        po.activated = True
                    elif order.side < 0 and tick_price <= order.stop_trigger:
                        po.activated = True
                if not po.activated:
                    still_pending.append(po)
                    continue
                if tick_ns >= po.fill_after_ns:
                    self._submit_to_engine(order, tick)
                else:
                    still_pending.append(po)
                continue

            if tick_ns < po.fill_after_ns:
                still_pending.append(po)
                continue

            self._submit_to_engine(order, tick)

        self._pending = still_pending

    def _submit_to_engine(self, order: Order, tick: Tick) -> None:
        """Route a time-matured order to the matching engine."""
        fills = self._engine.submit(order, tick.ts)
        filled_qty = 0.0
        for fill in fills:
            self._execute_fill(order, fill, tick)
            filled_qty += fill.qty
        remaining = order.qty - filled_qty
        if remaining > 1e-9:
            # Partially or fully resting in the book
            self._resting[order.order_id] = (order, remaining)

    def _apply_resting_fill(self, fill: Fill, tick: Tick) -> None:
        """Apply a fill that arrived from a resting order in the engine book."""
        entry = self._resting.get(fill.order_id)
        if entry is None:
            return
        order, remaining = entry
        remaining -= fill.qty
        self._execute_fill(order, fill, tick)
        if remaining <= 1e-9:
            del self._resting[fill.order_id]
        else:
            self._resting[fill.order_id] = (order, remaining)

    # ------------------------------------------------------------------
    # Simple price-comparison path (no order book)
    # ------------------------------------------------------------------

    def _process_tick_simple(self, tick: Tick) -> None:
        """Tick processing without an order book."""
        if not self._pending:
            return

        tick_ns = tick.ts.value
        tick_price = tick.price
        still_pending: List[PendingOrder] = []

        for po in self._pending:
            order = po.order

            if order.type == OrderType.STOP:
                if not po.activated:
                    if order.side > 0 and tick_price >= order.stop_trigger:
                        po.activated = True
                    elif order.side < 0 and tick_price <= order.stop_trigger:
                        po.activated = True
                if not po.activated:
                    still_pending.append(po)
                    continue
                if tick_ns >= po.fill_after_ns:
                    fill_px = self._market_fill_price(order.side, tick)
                    self._execute(order, tick, fill_px)
                else:
                    still_pending.append(po)
                continue

            if tick_ns < po.fill_after_ns:
                still_pending.append(po)
                continue

            if order.type == OrderType.MARKET:
                fill_px = self._market_fill_price(order.side, tick)
                self._execute(order, tick, fill_px)

            elif order.type == OrderType.LIMIT:
                ref_px = (
                    (tick.ask if tick.ask is not None else tick_price)
                    if order.side > 0
                    else (tick.bid if tick.bid is not None else tick_price)
                )
                if order.side > 0 and ref_px <= order.limit_price:
                    self._execute(order, tick, order.limit_price)
                elif order.side < 0 and ref_px >= order.limit_price:
                    self._execute(order, tick, order.limit_price)
                else:
                    still_pending.append(po)

        self._pending = still_pending

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_fill(
        self,
        order: Order,
        fill_qty: float,
        fill_price: float,
        submitted_at: pd.Timestamp,
        filled_at: pd.Timestamp,
    ) -> None:
        """Append a FillRecord for latency metrics collection."""
        # Lazy import to avoid circular import at module level.
        from backtesting.latency_metrics import FillRecord
        latency_us = (filled_at.value - submitted_at.value) / 1_000.0
        price_at_submit = self._submit_prices.get(order.order_id, 0.0)
        self._fill_records.append(
            FillRecord(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                submitted_at=submitted_at,
                filled_at=filled_at,
                fill_latency_us=latency_us,
                price_at_submit=price_at_submit,
                fill_price=fill_price,
                fill_qty=fill_qty,
                order_qty=order.qty,
            )
        )

    def _market_fill_price(self, side: int, tick: Tick) -> float:
        if side > 0:
            return tick.ask if tick.ask is not None else tick.price
        return tick.bid if tick.bid is not None else tick.price

    def _execute(self, order: Order, tick: Tick, fill_price: float) -> None:
        """Simple fill: full order qty at a single price (no order book)."""
        synthetic_bar = Bar(
            ts=tick.ts,
            open=fill_price, high=fill_price,
            low=fill_price, close=fill_price,
        )
        self._broker.open_trade(
            symbol=order.symbol,
            bar=synthetic_bar,
            side=order.side,
            size=order.qty,
            stop=order.protective_stop,
            take_profit=order.take_profit,
            entry_price=fill_price,
            current_prices={order.symbol: fill_price},
        )
        self._record_fill(order, order.qty, fill_price, order.submitted_at, tick.ts)

    def _execute_fill(self, order: Order, fill: Fill, tick: Tick) -> None:
        """Engine fill: partial or full qty from a matching engine Fill."""
        synthetic_bar = Bar(
            ts=fill.ts,
            open=fill.price, high=fill.price,
            low=fill.price, close=fill.price,
        )
        self._broker.open_trade(
            symbol=order.symbol,
            bar=synthetic_bar,
            side=order.side,
            size=fill.qty,
            stop=order.protective_stop,
            take_profit=order.take_profit,
            entry_price=fill.price,
            current_prices={order.symbol: fill.price},
        )
        self._record_fill(order, fill.qty, fill.price, order.submitted_at, fill.ts)
