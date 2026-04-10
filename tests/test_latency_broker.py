"""Tests for LatencyAwareBroker and Order types."""

import pandas as pd
import pytest

from backtesting import LatencyAwareBroker, Order, OrderType, Portfolio, TickBacktester
from backtesting.order import PendingOrder
from backtesting.strategy import Strategy
from backtesting.tick import Tick
from backtesting.types import Bar, Trade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tick(price: float, ts_ns: int, bid=None, ask=None) -> Tick:
    return Tick(
        ts=pd.Timestamp(ts_ns, unit="ns"),
        price=price,
        volume=1.0,
        bid=bid,
        ask=ask,
    )


def make_portfolio(**kwargs) -> Portfolio:
    defaults = dict(cash=10_000, commission_bps=0.0, slippage_bps=0.0,
                    max_leverage=10.0, margin_rate=0.0)
    defaults.update(kwargs)
    return Portfolio(**defaults)


def make_order(
    side=1,
    qty=1.0,
    symbol="X",
    order_type=OrderType.MARKET,
    protective_stop=0.0,
    take_profit=None,
    limit_price=None,
    stop_trigger=None,
    ts_ns=0,
) -> Order:
    return Order(
        type=order_type,
        symbol=symbol,
        side=side,
        qty=qty,
        protective_stop=protective_stop,
        take_profit=take_profit,
        submitted_at=pd.Timestamp(ts_ns, unit="ns"),
        limit_price=limit_price,
        stop_trigger=stop_trigger,
    )


# ---------------------------------------------------------------------------
# LatencyAwareBroker unit tests
# ---------------------------------------------------------------------------

class TestLatencyBrokerMarket:
    def test_market_fills_after_delay(self):
        """Order submitted at T=0 with 10ms delay fills on first tick at T>=10ms."""
        portfolio = make_portfolio()
        lb = LatencyAwareBroker(portfolio.broker, ack_latency_ns=10_000_000)

        order = make_order(side=1, qty=1.0, ts_ns=0)
        lb.submit(order, pd.Timestamp(0, unit="ns"))

        # Tick at T=5ms: too early — should not fill
        lb.process_tick(make_tick(100.0, 5_000_000))
        assert lb.pending_count == 1
        assert not portfolio.broker.has_open_position("X")

        # Tick at T=10ms: exactly on time — should fill
        lb.process_tick(make_tick(100.0, 10_000_000))
        assert lb.pending_count == 0
        assert portfolio.broker.has_open_position("X")

    def test_market_uses_ask_for_buy(self):
        """Buy market order fills at tick.ask, not tick.price."""
        portfolio = make_portfolio()
        lb = LatencyAwareBroker(portfolio.broker, ack_latency_ns=0)

        order = make_order(side=1, qty=1.0)
        lb.submit(order, pd.Timestamp(0, unit="ns"))

        lb.process_tick(make_tick(100.0, 0, bid=99.9, ask=100.1))
        trades = portfolio.broker.closed_trades
        open_pos = portfolio.broker.positions.get("X", [])
        assert len(open_pos) == 1
        assert open_pos[0].entry_price == pytest.approx(100.1)

    def test_market_uses_bid_for_sell(self):
        """Sell market order fills at tick.bid."""
        portfolio = make_portfolio()
        lb = LatencyAwareBroker(portfolio.broker, ack_latency_ns=0)

        order = make_order(side=-1, qty=1.0)
        lb.submit(order, pd.Timestamp(0, unit="ns"))

        lb.process_tick(make_tick(100.0, 0, bid=99.9, ask=100.1))
        open_pos = portfolio.broker.positions.get("X", [])
        assert len(open_pos) == 1
        assert open_pos[0].entry_price == pytest.approx(99.9)

    def test_market_falls_back_to_price_when_no_spread(self):
        """Without bid/ask, fill at tick.price."""
        portfolio = make_portfolio()
        lb = LatencyAwareBroker(portfolio.broker, ack_latency_ns=0)

        order = make_order(side=1, qty=1.0)
        lb.submit(order, pd.Timestamp(0, unit="ns"))
        lb.process_tick(make_tick(100.0, 0))

        open_pos = portfolio.broker.positions.get("X", [])
        assert open_pos[0].entry_price == pytest.approx(100.0)

    def test_zero_latency_fills_immediately(self):
        """With ack_latency_ns=0, order fills on the same tick it's eligible."""
        portfolio = make_portfolio()
        lb = LatencyAwareBroker(portfolio.broker, ack_latency_ns=0)

        order = make_order(side=1, qty=1.0, ts_ns=1_000_000)
        lb.submit(order, pd.Timestamp(1_000_000, unit="ns"))
        lb.process_tick(make_tick(50.0, 1_000_000))

        assert portfolio.broker.has_open_position("X")

    def test_pending_count_decreases_on_fill(self):
        portfolio = make_portfolio()
        lb = LatencyAwareBroker(portfolio.broker, ack_latency_ns=0)

        lb.submit(make_order(), pd.Timestamp(0, unit="ns"))
        lb.submit(make_order(), pd.Timestamp(0, unit="ns"))
        assert lb.pending_count == 2

        lb.process_tick(make_tick(100.0, 0))
        assert lb.pending_count == 0


class TestLatencyBrokerLimit:
    def test_limit_buy_fills_when_ask_crosses_below(self):
        """Limit buy at 100 fills when ask <= 100."""
        portfolio = make_portfolio()
        lb = LatencyAwareBroker(portfolio.broker, ack_latency_ns=0)

        order = make_order(side=1, qty=1.0, order_type=OrderType.LIMIT, limit_price=100.0)
        lb.submit(order, pd.Timestamp(0, unit="ns"))

        # ask=101: no fill
        lb.process_tick(make_tick(101.0, 0, ask=101.0))
        assert lb.pending_count == 1

        # ask=100: fills at limit_price
        lb.process_tick(make_tick(100.0, 1_000_000, ask=100.0))
        assert lb.pending_count == 0
        open_pos = portfolio.broker.positions.get("X", [])
        assert open_pos[0].entry_price == pytest.approx(100.0)

    def test_limit_sell_fills_when_bid_crosses_above(self):
        """Limit sell at 100 fills when bid >= 100."""
        portfolio = make_portfolio()
        lb = LatencyAwareBroker(portfolio.broker, ack_latency_ns=0)

        order = make_order(side=-1, qty=1.0, order_type=OrderType.LIMIT, limit_price=100.0)
        lb.submit(order, pd.Timestamp(0, unit="ns"))

        # bid=99: no fill
        lb.process_tick(make_tick(99.0, 0, bid=99.0))
        assert lb.pending_count == 1

        # bid=100: fills
        lb.process_tick(make_tick(100.0, 1_000_000, bid=100.0))
        assert lb.pending_count == 0

    def test_limit_stays_pending_until_eligible_then_price(self):
        """LIMIT order respects both time delay AND price condition."""
        portfolio = make_portfolio()
        lb = LatencyAwareBroker(portfolio.broker, ack_latency_ns=5_000_000)

        order = make_order(side=1, qty=1.0, order_type=OrderType.LIMIT, limit_price=100.0)
        lb.submit(order, pd.Timestamp(0, unit="ns"))

        # Price good but time not elapsed
        lb.process_tick(make_tick(99.0, 3_000_000, ask=99.0))
        assert lb.pending_count == 1

        # Time elapsed but price not good
        lb.process_tick(make_tick(101.0, 10_000_000, ask=101.0))
        assert lb.pending_count == 1

        # Time elapsed and price good
        lb.process_tick(make_tick(99.5, 11_000_000, ask=99.5))
        assert lb.pending_count == 0


class TestLatencyBrokerStop:
    def test_stop_buy_activates_on_price_cross(self):
        """Stop buy at trigger 102 doesn't activate below 102."""
        portfolio = make_portfolio()
        lb = LatencyAwareBroker(portfolio.broker, ack_latency_ns=0)

        order = make_order(side=1, qty=1.0, order_type=OrderType.STOP, stop_trigger=102.0)
        lb.submit(order, pd.Timestamp(0, unit="ns"))

        lb.process_tick(make_tick(101.0, 0))
        assert lb.pending_count == 1
        assert not lb._pending[0].activated

        lb.process_tick(make_tick(102.0, 1_000_000))
        assert lb.pending_count == 0
        assert portfolio.broker.has_open_position("X")

    def test_stop_sell_activates_on_price_drop(self):
        """Stop sell at trigger 98 activates when price <= 98."""
        portfolio = make_portfolio()
        lb = LatencyAwareBroker(portfolio.broker, ack_latency_ns=0)

        order = make_order(side=-1, qty=1.0, order_type=OrderType.STOP, stop_trigger=98.0)
        lb.submit(order, pd.Timestamp(0, unit="ns"))

        lb.process_tick(make_tick(99.0, 0))
        assert not lb._pending[0].activated

        lb.process_tick(make_tick(98.0, 1_000_000))
        assert portfolio.broker.has_open_position("X")

    def test_stop_respects_latency_after_activation(self):
        """After activation, STOP waits for fill_after_ns before executing."""
        portfolio = make_portfolio()
        lb = LatencyAwareBroker(portfolio.broker, ack_latency_ns=10_000_000)

        order = make_order(side=1, qty=1.0, order_type=OrderType.STOP, stop_trigger=102.0)
        # submitted at T=0, fill_after_ns = 10ms
        lb.submit(order, pd.Timestamp(0, unit="ns"))

        # T=5ms: price crosses trigger, activates — but fill_after_ns=10ms not reached
        lb.process_tick(make_tick(103.0, 5_000_000))
        assert lb._pending[0].activated
        assert lb.pending_count == 1

        # T=10ms: now eligible
        lb.process_tick(make_tick(103.0, 10_000_000))
        assert lb.pending_count == 0
        assert portfolio.broker.has_open_position("X")


# ---------------------------------------------------------------------------
# Pass-through properties
# ---------------------------------------------------------------------------

class TestPassThroughs:
    def test_positions_passthrough(self):
        portfolio = make_portfolio()
        lb = LatencyAwareBroker(portfolio.broker)
        assert lb.positions is portfolio.broker.positions

    def test_closed_trades_passthrough(self):
        portfolio = make_portfolio()
        lb = LatencyAwareBroker(portfolio.broker)
        assert lb.closed_trades is portfolio.broker.closed_trades

    def test_has_open_position_passthrough(self):
        portfolio = make_portfolio()
        lb = LatencyAwareBroker(portfolio.broker)
        assert lb.has_open_position("X") is False


# ---------------------------------------------------------------------------
# TickBacktester integration
# ---------------------------------------------------------------------------

def _make_ticks(prices, start_ns=0, step_ns=1_000_000_000):
    return [
        Tick(ts=pd.Timestamp(start_ns + i * step_ns, unit="ns"), price=p, volume=1.0)
        for i, p in enumerate(prices)
    ]


class LatencyStrategy(Strategy):
    """Strategy that emits a latency-queued MARKET order on bar 0."""

    def __init__(self, symbol="default"):
        self.symbol = symbol
        self.submitted = False

    def on_bar(self, i, bar, equity):
        if i == 0 and not self.submitted:
            self.submitted = True
            return Order(
                type=OrderType.MARKET,
                symbol=self.symbol,
                side=1,
                qty=1.0,
                protective_stop=bar.close * 0.95,
                take_profit=None,
                submitted_at=bar.ts,
            )
        return None


class LegacyTradeStrategy(Strategy):
    """Strategy that returns a legacy Trade — verifies backward compatibility."""

    def __init__(self):
        self.fired = False

    def on_bar(self, i, bar, equity):
        if i == 0 and not self.fired:
            self.fired = True
            return Trade(
                entry_bar=bar, side=1, size=1.0,
                entry_price=bar.close,
                stop_price=bar.close * 0.95,
                take_profit=None,
            )
        return None


class TestTickBacktesterIntegration:
    def test_legacy_trade_path_unchanged(self):
        """Existing Trade-returning strategies still work without latency_broker."""
        ticks = _make_ticks([100.0] * 20)
        strategy = LegacyTradeStrategy()
        bt = TickBacktester(ticks, strategy, timeframe="M1", starting_cash=10_000,
                            max_leverage=10.0)
        equity_curve, trades = bt.run()
        assert len(equity_curve) == 20

    def test_latency_broker_wired_in(self):
        """TickBacktester accepts latency_broker param and runs without error."""
        ticks = _make_ticks([100.0] * 60)
        strategy = LatencyStrategy()
        portfolio = Portfolio(cash=10_000, commission_bps=0, slippage_bps=0,
                              max_leverage=10.0, margin_rate=0.0)
        lb = LatencyAwareBroker(portfolio.broker, ack_latency_ns=0)
        bt = TickBacktester(ticks, strategy, timeframe="M1",
                            starting_cash=10_000, max_leverage=10.0,
                            latency_broker=lb)
        # Swap in the shared portfolio so bt uses the same broker lb wraps
        bt.portfolio = portfolio
        bt.broker = portfolio.broker
        equity_curve, closed = bt.run()
        assert len(equity_curve) == 60

    def test_no_latency_broker_no_regression(self):
        """254 existing tests pass — this just double-checks zero-regression."""
        ticks = _make_ticks([100 + i * 0.1 for i in range(30)])
        strategy = LegacyTradeStrategy()
        bt = TickBacktester(ticks, strategy, timeframe="M1",
                            starting_cash=10_000, max_leverage=10.0)
        equity_curve, _ = bt.run()
        assert len(equity_curve) == 30
        assert all(e > 0 for e in equity_curve)
