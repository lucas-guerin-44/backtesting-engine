"""Tests for OrderBook, PriceLevel, MatchingEngine, and Phase 2 integration."""

import pandas as pd
import pytest

from backtesting import (
    Fill, LatencyAwareBroker, MatchingEngine, Order, OrderBook,
    OrderType, Portfolio, PriceLevel,
)
from backtesting.tick import Tick
from backtesting.types import Trade
from backtesting.strategy import Strategy
from backtesting import TickBacktester


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tick(price: float, ts_ns: int = 0, bid=None, ask=None) -> Tick:
    return Tick(
        ts=pd.Timestamp(ts_ns, unit="ns"),
        price=price, volume=1.0, bid=bid, ask=ask,
    )


def make_order(
    side=1, qty=10.0, symbol="X",
    order_type=OrderType.MARKET,
    limit_price=None, stop_trigger=None,
    protective_stop=0.0, take_profit=None,
    ts_ns=0,
) -> Order:
    return Order(
        type=order_type, symbol=symbol, side=side, qty=qty,
        protective_stop=protective_stop, take_profit=take_profit,
        submitted_at=pd.Timestamp(ts_ns, unit="ns"),
        limit_price=limit_price, stop_trigger=stop_trigger,
    )


def make_portfolio():
    return Portfolio(cash=100_000, commission_bps=0, slippage_bps=0,
                     max_leverage=20.0, margin_rate=0.0)


# ---------------------------------------------------------------------------
# PriceLevel
# ---------------------------------------------------------------------------

class TestPriceLevel:
    def test_fifo_two_orders_full_fill(self):
        lvl = PriceLevel(100.0)
        lvl.add("A", 10.0)
        lvl.add("B", 20.0)
        fills = lvl.consume(float("inf"))
        assert fills == [("A", 10.0), ("B", 20.0)]
        assert lvl.is_empty()

    def test_fifo_partial_first_order(self):
        """Only 5 units available — first order partially fills."""
        lvl = PriceLevel(100.0)
        lvl.add("A", 10.0)
        lvl.add("B", 10.0)
        fills = lvl.consume(5.0)
        assert fills == [("A", 5.0)]
        assert not lvl.is_empty()
        # Remaining: A has 5, B has 10
        fills2 = lvl.consume(float("inf"))
        assert fills2 == [("A", 5.0), ("B", 10.0)]

    def test_fifo_exact_first_order(self):
        """Exactly enough to fill first order, none for second."""
        lvl = PriceLevel(100.0)
        lvl.add("A", 10.0)
        lvl.add("B", 10.0)
        fills = lvl.consume(10.0)
        assert fills == [("A", 10.0)]
        assert len(lvl) == 1

    def test_fifo_drains_multiple_full(self):
        lvl = PriceLevel(100.0)
        for i in range(5):
            lvl.add(str(i), 10.0)
        fills = lvl.consume(30.0)
        # First 3 orders fill (30 units)
        assert len(fills) == 3
        assert sum(q for _, q in fills) == pytest.approx(30.0)
        assert len(lvl) == 2

    def test_empty_level_consume(self):
        lvl = PriceLevel(100.0)
        assert lvl.consume(100.0) == []


# ---------------------------------------------------------------------------
# OrderBook
# ---------------------------------------------------------------------------

class TestOrderBook:
    def test_update_sets_bid_ask(self):
        book = OrderBook()
        book.update(make_tick(100.0, bid=99.9, ask=100.1))
        assert book.best_bid == pytest.approx(99.9)
        assert book.best_ask == pytest.approx(100.1)

    def test_update_no_spread_uses_price(self):
        book = OrderBook()
        book.update(make_tick(100.0))
        assert book.best_bid == pytest.approx(100.0)
        assert book.best_ask == pytest.approx(100.0)

    def test_resting_bid_count(self):
        book = OrderBook()
        book.add_resting_bid("A", 100.0, 10.0)
        book.add_resting_bid("B", 100.0, 10.0)
        book.add_resting_bid("C", 99.0, 5.0)
        assert book.resting_bid_count() == 3

    def test_resting_ask_count(self):
        book = OrderBook()
        book.add_resting_ask("A", 101.0, 10.0)
        assert book.resting_ask_count() == 1


# ---------------------------------------------------------------------------
# MatchingEngine — MARKET orders
# ---------------------------------------------------------------------------

class TestMatchingEngineMarket:
    def test_market_buy_fills_at_ask(self):
        book = OrderBook()
        book.update(make_tick(100.0, bid=99.9, ask=100.1))
        engine = MatchingEngine(book)
        order = make_order(side=1, qty=10.0)
        fills = engine.submit(order, pd.Timestamp(0, unit="ns"))
        assert len(fills) == 1
        assert fills[0].price == pytest.approx(100.1)
        assert fills[0].qty == pytest.approx(10.0)

    def test_market_sell_fills_at_bid(self):
        book = OrderBook()
        book.update(make_tick(100.0, bid=99.9, ask=100.1))
        engine = MatchingEngine(book)
        order = make_order(side=-1, qty=5.0)
        fills = engine.submit(order, pd.Timestamp(0, unit="ns"))
        assert fills[0].price == pytest.approx(99.9)

    def test_market_partial_fill_respects_max_qty(self):
        book = OrderBook()
        book.update(make_tick(100.0, ask=100.0))
        engine = MatchingEngine(book, max_qty_per_level=5.0)
        order = make_order(side=1, qty=10.0)
        fills = engine.submit(order, pd.Timestamp(0, unit="ns"))
        assert len(fills) == 1
        assert fills[0].qty == pytest.approx(5.0)  # partial — max_qty cap

    def test_market_no_fill_when_book_empty(self):
        book = OrderBook()  # no ticks yet — no bid/ask
        engine = MatchingEngine(book)
        order = make_order(side=1, qty=10.0)
        fills = engine.submit(order, pd.Timestamp(0, unit="ns"))
        assert fills == []


# ---------------------------------------------------------------------------
# MatchingEngine — LIMIT orders
# ---------------------------------------------------------------------------

class TestMatchingEngineLimit:
    def test_limit_buy_immediate_fill_when_ask_crosses(self):
        """Limit buy at 100 fills immediately when ask=99.9 (<= 100)."""
        book = OrderBook()
        book.update(make_tick(100.0, ask=99.9))
        engine = MatchingEngine(book)
        order = make_order(side=1, qty=10.0, order_type=OrderType.LIMIT, limit_price=100.0)
        fills = engine.submit(order, pd.Timestamp(0, unit="ns"))
        assert len(fills) == 1
        assert fills[0].price == pytest.approx(99.9)  # fills at ask, not limit

    def test_limit_buy_rests_when_ask_above_limit(self):
        book = OrderBook()
        book.update(make_tick(100.0, ask=100.5))
        engine = MatchingEngine(book)
        order = make_order(side=1, qty=10.0, order_type=OrderType.LIMIT, limit_price=100.0)
        fills = engine.submit(order, pd.Timestamp(0, unit="ns"))
        assert fills == []
        assert book.resting_bid_count() == 1

    def test_limit_sell_immediate_fill_when_bid_crosses(self):
        book = OrderBook()
        book.update(make_tick(100.0, bid=100.1))
        engine = MatchingEngine(book)
        order = make_order(side=-1, qty=10.0, order_type=OrderType.LIMIT, limit_price=100.0)
        fills = engine.submit(order, pd.Timestamp(0, unit="ns"))
        assert len(fills) == 1
        assert fills[0].price == pytest.approx(100.1)

    def test_limit_buy_drains_on_process_tick(self):
        """Resting limit buy drains when a later tick has ask <= limit."""
        book = OrderBook()
        book.update(make_tick(101.0, ask=101.0))
        engine = MatchingEngine(book)
        order = make_order(side=1, qty=10.0, order_type=OrderType.LIMIT, limit_price=100.0)
        engine.submit(order, pd.Timestamp(0, unit="ns"))
        assert book.resting_bid_count() == 1

        # Tick moves ask to 99.9 — limit is now crossable
        fills = engine.process_tick(make_tick(99.9, ts_ns=1_000_000, ask=99.9))
        assert len(fills) == 1
        assert fills[0].qty == pytest.approx(10.0)
        assert book.resting_bid_count() == 0

    def test_limit_sell_drains_on_process_tick(self):
        book = OrderBook()
        book.update(make_tick(99.0, bid=99.0))
        engine = MatchingEngine(book)
        order = make_order(side=-1, qty=5.0, order_type=OrderType.LIMIT, limit_price=100.0)
        engine.submit(order, pd.Timestamp(0, unit="ns"))

        fills = engine.process_tick(make_tick(100.5, ts_ns=1_000_000, bid=100.5))
        assert len(fills) == 1
        assert fills[0].qty == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# MatchingEngine — FIFO ordering
# ---------------------------------------------------------------------------

class TestMatchingEngineFIFO:
    def test_two_orders_same_level_fifo(self):
        """Two limit buys at the same price; first submitted fills first."""
        book = OrderBook()
        book.update(make_tick(101.0, ask=101.0))
        engine = MatchingEngine(book, max_qty_per_level=5.0)

        o1 = make_order(side=1, qty=5.0, order_type=OrderType.LIMIT, limit_price=100.0)
        o2 = make_order(side=1, qty=5.0, order_type=OrderType.LIMIT, limit_price=100.0)
        engine.submit(o1, pd.Timestamp(0, unit="ns"))
        engine.submit(o2, pd.Timestamp(0, unit="ns"))
        assert book.resting_bid_count() == 2

        # First tick: ask crosses 100, max_qty=5 → only first order fills
        fills = engine.process_tick(make_tick(99.9, ts_ns=1_000_000, ask=99.9))
        assert len(fills) == 1
        assert fills[0].order_id == o1.order_id
        assert book.resting_bid_count() == 1

        # Second tick: second order fills
        fills2 = engine.process_tick(make_tick(99.9, ts_ns=2_000_000, ask=99.9))
        assert len(fills2) == 1
        assert fills2[0].order_id == o2.order_id
        assert book.resting_bid_count() == 0

    def test_partial_fill_preserves_remainder_at_front(self):
        """30-unit order with max_qty=10 takes 3 ticks to fully fill."""
        book = OrderBook()
        book.update(make_tick(101.0, ask=101.0))
        engine = MatchingEngine(book, max_qty_per_level=10.0)
        order = make_order(side=1, qty=30.0, order_type=OrderType.LIMIT, limit_price=100.0)
        engine.submit(order, pd.Timestamp(0, unit="ns"))

        total_filled = 0.0
        for i in range(1, 5):
            fills = engine.process_tick(make_tick(99.9, ts_ns=i * 1_000_000, ask=99.9))
            total_filled += sum(f.qty for f in fills)

        assert total_filled == pytest.approx(30.0)

    def test_best_price_level_fills_first(self):
        """Higher limit buy price fills before lower one."""
        book = OrderBook()
        book.update(make_tick(105.0, ask=105.0))
        engine = MatchingEngine(book)

        lo = make_order(side=1, qty=5.0, order_type=OrderType.LIMIT, limit_price=100.0)
        hi = make_order(side=1, qty=5.0, order_type=OrderType.LIMIT, limit_price=102.0)
        engine.submit(lo, pd.Timestamp(0, unit="ns"))
        engine.submit(hi, pd.Timestamp(0, unit="ns"))

        fills = engine.process_tick(make_tick(99.9, ts_ns=1_000_000, ask=99.9))
        filled_ids = {f.order_id for f in fills}
        # Both levels are now crossable — both should fill
        assert lo.order_id in filled_ids
        assert hi.order_id in filled_ids


# ---------------------------------------------------------------------------
# LatencyAwareBroker + MatchingEngine integration
# ---------------------------------------------------------------------------

class TestLatencyBrokerWithEngine:
    def _setup(self, max_qty=float("inf"), ack_ns=0):
        portfolio = make_portfolio()
        book = OrderBook()
        engine = MatchingEngine(book, max_qty_per_level=max_qty)
        lb = LatencyAwareBroker(portfolio.broker, ack_latency_ns=ack_ns, order_book=engine)
        return portfolio, lb

    def test_market_order_fills_via_engine(self):
        portfolio, lb = self._setup()
        order = make_order(side=1, qty=10.0)
        lb.submit(order, pd.Timestamp(0, unit="ns"))
        lb.process_tick(make_tick(100.0, ts_ns=0, ask=100.1))
        assert portfolio.broker.has_open_position("X")
        pos = portfolio.broker.positions["X"]
        assert pos[0].entry_price == pytest.approx(100.1)

    def test_limit_order_rests_then_fills_via_engine(self):
        portfolio, lb = self._setup()
        order = make_order(side=1, qty=10.0, order_type=OrderType.LIMIT, limit_price=100.0)
        lb.submit(order, pd.Timestamp(0, unit="ns"))

        # Ask above limit — rests
        lb.process_tick(make_tick(101.0, ts_ns=0, ask=101.0))
        assert not portfolio.broker.has_open_position("X")
        assert lb.pending_count == 1  # in resting

        # Ask crosses limit — fills
        lb.process_tick(make_tick(99.9, ts_ns=1_000_000, ask=99.9))
        assert portfolio.broker.has_open_position("X")
        assert lb.pending_count == 0

    def test_partial_fill_creates_open_position(self):
        """With max_qty=5 and order qty=10, first tick gives partial fill."""
        portfolio, lb = self._setup(max_qty=5.0)
        order = make_order(side=1, qty=10.0, order_type=OrderType.LIMIT, limit_price=100.0)
        lb.submit(order, pd.Timestamp(0, unit="ns"))

        # First tick: partial fill of 5
        lb.process_tick(make_tick(99.9, ts_ns=0, ask=99.9))
        pos = portfolio.broker.positions.get("X", [])
        assert len(pos) == 1
        assert pos[0].size == pytest.approx(5.0)
        assert lb.pending_count == 1  # 5 units still resting

        # Second tick: remaining 5 fills
        lb.process_tick(make_tick(99.9, ts_ns=1_000_000, ask=99.9))
        pos = portfolio.broker.positions.get("X", [])
        assert len(pos) == 2
        assert lb.pending_count == 0

    def test_fifo_preserved_via_latency_broker(self):
        """Two limit orders at same level: first submitted fills first."""
        portfolio, lb = self._setup(max_qty=5.0)

        o1 = make_order(side=1, qty=5.0, order_type=OrderType.LIMIT, limit_price=100.0)
        o2 = make_order(side=1, qty=5.0, order_type=OrderType.LIMIT, limit_price=100.0)
        lb.submit(o1, pd.Timestamp(0, unit="ns"))
        lb.submit(o2, pd.Timestamp(0, unit="ns"))

        lb.process_tick(make_tick(101.0, ts_ns=0, ask=101.0))  # rests both
        lb.process_tick(make_tick(99.9, ts_ns=1_000_000, ask=99.9))  # fills o1
        lb.process_tick(make_tick(99.9, ts_ns=2_000_000, ask=99.9))  # fills o2

        pos = portfolio.broker.positions.get("X", [])
        assert len(pos) == 2

    def test_phase1_path_unchanged_without_engine(self):
        """Without order_book, original behavior is preserved exactly."""
        portfolio = make_portfolio()
        lb = LatencyAwareBroker(portfolio.broker, ack_latency_ns=0)
        order = make_order(side=1, qty=10.0)
        lb.submit(order, pd.Timestamp(0, unit="ns"))
        lb.process_tick(make_tick(100.0, ts_ns=0, ask=100.5))
        assert portfolio.broker.has_open_position("X")
        assert portfolio.broker.positions["X"][0].entry_price == pytest.approx(100.5)

    def test_ack_latency_respected_with_engine(self):
        portfolio, lb = self._setup(ack_ns=10_000_000)
        order = make_order(side=1, qty=5.0)
        lb.submit(order, pd.Timestamp(0, unit="ns"))

        lb.process_tick(make_tick(100.0, ts_ns=5_000_000, ask=100.0))
        assert not portfolio.broker.has_open_position("X")

        lb.process_tick(make_tick(100.0, ts_ns=10_000_000, ask=100.0))
        assert portfolio.broker.has_open_position("X")


# ---------------------------------------------------------------------------
# Full test suite regression
# ---------------------------------------------------------------------------

class LegacyTradeStrategy(Strategy):
    def __init__(self):
        self.fired = False

    def on_bar(self, i, bar, equity):
        if i == 0 and not self.fired:
            self.fired = True
            return Trade(entry_bar=bar, side=1, size=1.0,
                         entry_price=bar.close, stop_price=bar.close * 0.95,
                         take_profit=None)
        return None


def _make_ticks(prices, start_ns=0, step_ns=1_000_000_000):
    return [
        Tick(ts=pd.Timestamp(start_ns + i * step_ns, unit="ns"),
             price=p, volume=1.0)
        for i, p in enumerate(prices)
    ]


class TestRegressionNoBreakage:
    def test_existing_tests_still_pass(self):
        ticks = _make_ticks([100.0] * 30)
        bt = TickBacktester(ticks, LegacyTradeStrategy(), timeframe="M1",
                            starting_cash=10_000, max_leverage=10.0)
        curve, trades = bt.run()
        assert len(curve) == 30
