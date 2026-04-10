"""Order book and matching engine for realistic limit order execution.

Provides a Level 1 order book (best bid/ask updated from ticks) with
a synthetic resting-order layer for limit orders submitted by strategies.
The matching engine handles FIFO queue priority and partial fills.

Key concepts
------------
PriceLevel
    A FIFO queue of (order_id, qty_remaining) pairs at a single price.
    When a level is hit, orders drain front-to-back. The first order
    submitted gets the first fill — no equal-priority tie-breaking.

OrderBook
    Tracks best bid/ask from the live tick stream. Also maintains two
    synthetic resting books: bid levels (limit buy orders waiting for
    price to fall) and ask levels (limit sell orders waiting for price
    to rise).

MatchingEngine
    Wraps an OrderBook. ``submit()`` either fills immediately (if the
    book crosses the limit price) or rests the order in the book.
    ``process_tick()`` advances the book by one tick and drains any
    resting levels that are now crossable. Returns ``Fill`` objects.
    Partial fills are native: ``max_qty_per_level`` caps how many units
    can fill per price level per tick.

Usage
-----
>>> book = OrderBook()
>>> engine = MatchingEngine(book, max_qty_per_level=50.0)
>>>
>>> # Feed ticks to update best bid/ask
>>> for tick in ticks:
...     fills = engine.process_tick(tick)
...     for fill in fills:
...         print(fill)
>>>
>>> # Submit an order that may rest or fill immediately
>>> from backtesting.order import Order, OrderType
>>> import pandas as pd
>>> order = Order(type=OrderType.LIMIT, symbol="X", side=1, qty=100.0,
...               protective_stop=98.0, take_profit=None,
...               submitted_at=pd.Timestamp("2024-01-01"),
...               limit_price=100.0)
>>> fills = engine.submit(order, pd.Timestamp("2024-01-01"))
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from backtesting.order import Order, OrderType
from backtesting.tick import Tick


@dataclass
class Fill:
    """Result of a single order match.

    Parameters
    ----------
    order_id : str
        Matches ``Order.order_id`` of the originating order.
    price : float
        Actual fill price.
    qty : float
        Units filled (may be less than the order's full qty for partial fills).
    ts : pd.Timestamp
        Timestamp of the tick that triggered this fill.
    """
    order_id: str
    price: float
    qty: float
    ts: pd.Timestamp


class PriceLevel:
    """FIFO queue of resting orders at a single price level.

    Parameters
    ----------
    price : float
        The price this level represents.
    """

    __slots__ = ("price", "_orders")

    def __init__(self, price: float) -> None:
        self.price = price
        # Each entry is (order_id, qty_remaining)
        self._orders: deque[Tuple[str, float]] = deque()

    def add(self, order_id: str, qty: float) -> None:
        """Enqueue an order at the back of the FIFO queue."""
        self._orders.append((order_id, qty))

    def consume(self, available_qty: float) -> List[Tuple[str, float]]:
        """Drain up to ``available_qty`` units from the front of the queue.

        Returns a list of ``(order_id, filled_qty)`` pairs. Orders fill
        in submission order (FIFO). Partial fills are supported: if the
        front order is larger than available, only ``available_qty`` units
        fill and the remainder stays at the front.

        Parameters
        ----------
        available_qty : float
            Maximum units to consume this call.

        Returns
        -------
        list of (order_id, filled_qty)
        """
        fills: List[Tuple[str, float]] = []
        remaining = available_qty
        while self._orders and remaining > 0:
            order_id, qty = self._orders[0]
            fill_qty = min(qty, remaining)
            fills.append((order_id, fill_qty))
            remaining -= fill_qty
            if fill_qty >= qty:
                self._orders.popleft()          # fully consumed
            else:
                self._orders[0] = (order_id, qty - fill_qty)  # partial
                break
        return fills

    def is_empty(self) -> bool:
        return len(self._orders) == 0

    def __len__(self) -> int:
        return len(self._orders)


class OrderBook:
    """Level 1 order book with a synthetic resting-order layer.

    Updated tick by tick via ``update()``. Resting limit orders are
    stored in two dicts keyed by price: bid levels (limit buys waiting
    for ask to fall) and ask levels (limit sells waiting for bid to rise).

    This is not a real exchange book — it models our own orders only.
    The best bid/ask comes from the live tick stream; the resting layer
    is synthetic, built from orders submitted by strategies.
    """

    def __init__(self) -> None:
        self._best_bid: Optional[float] = None
        self._best_ask: Optional[float] = None
        # Resting limit buy orders: price -> PriceLevel
        self._bid_levels: Dict[float, PriceLevel] = {}
        # Resting limit sell orders: price -> PriceLevel
        self._ask_levels: Dict[float, PriceLevel] = {}

    @property
    def best_bid(self) -> Optional[float]:
        """Current best bid from the tick stream."""
        return self._best_bid

    @property
    def best_ask(self) -> Optional[float]:
        """Current best ask from the tick stream."""
        return self._best_ask

    def update(self, tick: Tick) -> None:
        """Advance the book to this tick's bid/ask."""
        if tick.bid is not None:
            self._best_bid = tick.bid
        if tick.ask is not None:
            self._best_ask = tick.ask
        # Fall back to mid price when no spread data
        if tick.bid is None and tick.ask is None:
            self._best_bid = tick.price
            self._best_ask = tick.price

    def add_resting_bid(self, order_id: str, price: float, qty: float) -> None:
        """Add a limit buy order to the resting bid book."""
        if price not in self._bid_levels:
            self._bid_levels[price] = PriceLevel(price)
        self._bid_levels[price].add(order_id, qty)

    def add_resting_ask(self, order_id: str, price: float, qty: float) -> None:
        """Add a limit sell order to the resting ask book."""
        if price not in self._ask_levels:
            self._ask_levels[price] = PriceLevel(price)
        self._ask_levels[price].add(order_id, qty)

    def resting_bid_count(self) -> int:
        """Total number of resting limit buy orders across all levels."""
        return sum(len(lvl) for lvl in self._bid_levels.values())

    def resting_ask_count(self) -> int:
        """Total number of resting limit sell orders across all levels."""
        return sum(len(lvl) for lvl in self._ask_levels.values())


class MatchingEngine:
    """Matches incoming orders against an OrderBook.

    Handles MARKET and LIMIT orders with FIFO priority at each price level.
    Partial fills are supported via ``max_qty_per_level``.

    Parameters
    ----------
    book : OrderBook
        The order book to match against. Updated by ``process_tick()``.
    max_qty_per_level : float
        Maximum units that can fill per price level per tick. Use
        ``float('inf')`` (the default) for unlimited liquidity. Set a
        finite value to simulate thin markets with partial fills.
    """

    def __init__(
        self,
        book: OrderBook,
        max_qty_per_level: float = float("inf"),
    ) -> None:
        self.book = book
        self.max_qty_per_level = max_qty_per_level

    def process_tick(self, tick: Tick) -> List[Fill]:
        """Advance the book by one tick and drain any newly crossable levels.

        Call this on every tick before submitting new orders. It updates
        the book's best bid/ask then checks all resting orders for fills.

        Parameters
        ----------
        tick : Tick
            The incoming tick.

        Returns
        -------
        list of Fill
            Fills generated from resting orders that became crossable.
        """
        self.book.update(tick)
        fills: List[Fill] = []

        ask = self.book.best_ask
        bid = self.book.best_bid

        # Drain resting limit buys: fill when ask <= limit_price
        if ask is not None and self.book._bid_levels:
            # Best (highest) bid levels fill first
            crossable = sorted(
                (p for p in self.book._bid_levels if ask <= p),
                reverse=True,
            )
            for price in crossable:
                level = self.book._bid_levels[price]
                for order_id, qty in level.consume(self.max_qty_per_level):
                    fills.append(Fill(order_id=order_id, price=ask, qty=qty, ts=tick.ts))
                if level.is_empty():
                    del self.book._bid_levels[price]

        # Drain resting limit sells: fill when bid >= limit_price
        if bid is not None and self.book._ask_levels:
            # Best (lowest) ask levels fill first
            crossable = sorted(
                (p for p in self.book._ask_levels if bid >= p),
            )
            for price in crossable:
                level = self.book._ask_levels[price]
                for order_id, qty in level.consume(self.max_qty_per_level):
                    fills.append(Fill(order_id=order_id, price=bid, qty=qty, ts=tick.ts))
                if level.is_empty():
                    del self.book._ask_levels[price]

        return fills

    def submit(self, order: Order, ts: pd.Timestamp) -> List[Fill]:
        """Submit an order for immediate matching or book placement.

        MARKET orders fill at the current best ask (buy) or best bid (sell).
        LIMIT orders fill immediately if the book is already crossable;
        otherwise they rest in the book until ``process_tick()`` drains them.
        STOP orders are treated as MARKET (activation is handled by
        ``LatencyAwareBroker`` before calling this method).

        Returns fills generated immediately. A resting limit order returns
        an empty list; its fills will arrive via future ``process_tick()`` calls.

        Parameters
        ----------
        order : Order
            The order to match.
        ts : pd.Timestamp
            Timestamp to stamp on immediate fills.

        Returns
        -------
        list of Fill
            Immediate fills. May be empty (order resting) or partial.
        """
        if order.type in (OrderType.MARKET, OrderType.STOP):
            return self._fill_market(order, ts)
        elif order.type == OrderType.LIMIT:
            return self._fill_or_rest_limit(order, ts)
        return []

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fill_market(self, order: Order, ts: pd.Timestamp) -> List[Fill]:
        price = self.book.best_ask if order.side > 0 else self.book.best_bid
        if price is None:
            return []
        fill_qty = min(order.qty, self.max_qty_per_level)
        return [Fill(order_id=order.order_id, price=price, qty=fill_qty, ts=ts)]

    def _fill_or_rest_limit(self, order: Order, ts: pd.Timestamp) -> List[Fill]:
        if order.side > 0:
            ask = self.book.best_ask
            if ask is not None and ask <= order.limit_price:
                fill_qty = min(order.qty, self.max_qty_per_level)
                remainder = order.qty - fill_qty
                if remainder > 1e-9:
                    # Partial fill — rest the unfilled portion in the book
                    self.book.add_resting_bid(order.order_id, order.limit_price, remainder)
                return [Fill(order_id=order.order_id, price=ask, qty=fill_qty, ts=ts)]
            else:
                self.book.add_resting_bid(order.order_id, order.limit_price, order.qty)
                return []
        else:
            bid = self.book.best_bid
            if bid is not None and bid >= order.limit_price:
                fill_qty = min(order.qty, self.max_qty_per_level)
                remainder = order.qty - fill_qty
                if remainder > 1e-9:
                    self.book.add_resting_ask(order.order_id, order.limit_price, remainder)
                return [Fill(order_id=order.order_id, price=bid, qty=fill_qty, ts=ts)]
            else:
                self.book.add_resting_ask(order.order_id, order.limit_price, order.qty)
                return []
