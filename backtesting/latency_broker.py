"""Latency-aware execution layer wrapping the core Broker.

Models order lifecycle: submitted → acknowledged → filled, with
configurable network and queue delays. Supports MARKET, LIMIT, and STOP
order types, using tick.bid/ask when available for realistic fill prices.

When the TickBacktester is given a LatencyAwareBroker, strategies can
return ``Order`` objects (instead of ``Trade``) from ``on_bar``/``on_tick``.
The order enters the latency queue and fills once the clock catches up.

Usage
-----
>>> from backtesting.latency_broker import LatencyAwareBroker
>>> from backtesting.order import Order, OrderType
>>> import pandas as pd
>>>
>>> # 10 ms acknowledgment delay, 5 ms fill delay
>>> latency_broker = LatencyAwareBroker(
...     broker=portfolio.broker,
...     ack_latency_ns=10_000_000,
...     fill_latency_ns=5_000_000,
... )
>>> bt = TickBacktester(ticks, strategy, latency_broker=latency_broker)
"""

from typing import List, Optional

import pandas as pd

from backtesting.broker import Broker
from backtesting.order import Order, OrderType, PendingOrder
from backtesting.tick import Tick
from backtesting.types import Bar


class LatencyAwareBroker:
    """Wraps a Broker to simulate order lifecycle latency.

    Orders submitted via ``submit()`` enter a pending queue tagged with
    a nanosecond fill-eligibility timestamp. On each tick, ``process_tick()``
    scans the queue and fills any matured orders against the current price.

    MARKET: fills at tick.ask (buy) / tick.bid (sell), or tick.price when
            no spread data is available.
    LIMIT:  fills only if the market-side price has crossed the limit level.
            Stays in the queue otherwise — does not expire.
    STOP:   waits for tick.price to cross stop_trigger, then fills as MARKET
            on the next eligible tick.

    Parameters
    ----------
    broker : Broker
        The underlying broker. All fills are delegated to broker.open_trade().
    ack_latency_ns : int
        Nanoseconds from order submission to acknowledgment (network RTT).
    fill_latency_ns : int
        Additional nanoseconds from ack to fill (exchange queue time).
    """

    def __init__(
        self,
        broker: Broker,
        ack_latency_ns: int = 0,
        fill_latency_ns: int = 0,
    ):
        self._broker = broker
        self.ack_latency_ns = ack_latency_ns
        self.fill_latency_ns = fill_latency_ns
        self._pending: List[PendingOrder] = []

    # ------------------------------------------------------------------
    # Pass-throughs so TickBacktester can treat this as a drop-in for Broker
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
        """Number of orders still waiting to be filled."""
        return len(self._pending)

    def submit(self, order: Order, ts: pd.Timestamp) -> None:
        """Enqueue an order. It becomes eligible to fill after the configured delay.

        Parameters
        ----------
        order : Order
            The order to queue.
        ts : pd.Timestamp
            The timestamp at which the strategy submitted the order.
            Fill eligibility = ts.value + ack_latency_ns + fill_latency_ns.
        """
        fill_after_ns = ts.value + self.ack_latency_ns + self.fill_latency_ns
        self._pending.append(PendingOrder(order=order, fill_after_ns=fill_after_ns))

    def process_tick(self, tick: Tick) -> None:
        """Attempt to fill matured pending orders against this tick's price.

        Called by TickBacktester on every tick, before strategy signals fire.
        Orders that are not yet eligible (by timestamp) or whose price
        condition is not met (LIMIT) stay in the queue.

        Parameters
        ----------
        tick : Tick
            The current tick. Uses tick.ask/bid when available for fills.
        """
        if not self._pending:
            return

        tick_ns = tick.ts.value
        tick_price = tick.price
        still_pending: List[PendingOrder] = []

        for po in self._pending:
            order = po.order

            # ----------------------------------------------------------
            # STOP: wait for price to cross the activation trigger first
            # ----------------------------------------------------------
            if order.type == OrderType.STOP:
                if not po.activated:
                    if order.side > 0 and tick_price >= order.stop_trigger:
                        po.activated = True
                    elif order.side < 0 and tick_price <= order.stop_trigger:
                        po.activated = True

                if not po.activated:
                    still_pending.append(po)
                    continue

                # Activated — treat as MARKET once delay has elapsed
                if tick_ns >= po.fill_after_ns:
                    fill_px = self._market_fill_price(order.side, tick)
                    self._execute(order, tick, fill_px)
                else:
                    still_pending.append(po)
                continue

            # ----------------------------------------------------------
            # Not yet eligible by time
            # ----------------------------------------------------------
            if tick_ns < po.fill_after_ns:
                still_pending.append(po)
                continue

            # ----------------------------------------------------------
            # MARKET
            # ----------------------------------------------------------
            if order.type == OrderType.MARKET:
                fill_px = self._market_fill_price(order.side, tick)
                self._execute(order, tick, fill_px)

            # ----------------------------------------------------------
            # LIMIT: fill only when market-side price crosses the limit
            # ----------------------------------------------------------
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

    def _market_fill_price(self, side: int, tick: Tick) -> float:
        """Return the realistic fill price for a market order on this tick."""
        if side > 0:
            return tick.ask if tick.ask is not None else tick.price
        return tick.bid if tick.bid is not None else tick.price

    def _execute(self, order: Order, tick: Tick, fill_price: float) -> None:
        """Delegate a fill to the underlying Broker via a synthetic bar."""
        synthetic_bar = Bar(
            ts=tick.ts,
            open=fill_price,
            high=fill_price,
            low=fill_price,
            close=fill_price,
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
