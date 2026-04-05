"""Tests for the Broker execution layer."""

import pandas as pd
import pytest

from backtesting.broker import Broker
from backtesting.portfolio import Portfolio
from backtesting.types import Bar, Trade


@pytest.fixture
def portfolio():
    return Portfolio(cash=10_000, commission_bps=0.0, slippage_bps=0.0,
                     max_leverage=1.0, margin_rate=0.0)


@pytest.fixture
def broker(portfolio):
    return portfolio.broker


def make_bar(ts="2024-01-01", o=100, h=105, lo=95, c=102):
    return Bar(pd.Timestamp(ts), float(o), float(h), float(lo), float(c))


class TestOpenTrade:
    def test_basic_open(self, broker):
        bar = make_bar()
        broker.open_trade("TEST", bar, side=1, size=10.0, stop=90.0, take_profit=115.0)
        assert broker.has_open_position("TEST")
        assert len(broker.positions["TEST"]) == 1

    def test_zero_size_rejected(self, broker):
        bar = make_bar()
        broker.open_trade("TEST", bar, side=1, size=0.0, stop=90.0, take_profit=115.0)
        assert not broker.has_open_position("TEST")

    def test_custom_entry_price(self, broker):
        bar = make_bar(o=100, c=102)
        broker.open_trade("TEST", bar, side=1, size=5.0, stop=90.0,
                          take_profit=115.0, entry_price=101.5)
        trade = broker.positions["TEST"][0]
        assert abs(trade.entry_price - 101.5) < 0.01

    def test_buying_power_limit(self):
        """With max_leverage=1.0, can't open a position larger than equity."""
        port = Portfolio(cash=1_000, max_leverage=1.0)
        bar = make_bar(o=100, c=100)
        # Try to buy 20 units at ~100 = $2000 notional > $1000 equity
        port.broker.open_trade("TEST", bar, side=1, size=20.0, stop=90.0, take_profit=110.0)
        if port.broker.has_open_position("TEST"):
            trade = port.broker.positions["TEST"][0]
            # Size should be clamped to what buying power allows
            assert trade.size * trade.entry_price <= 1_000 + 1.0  # small tolerance


class TestStopExecution:
    def test_long_stop_normal(self, broker):
        """Stop hit within bar range -> fill at stop price."""
        bar = make_bar()
        broker.open_trade("TEST", bar, side=1, size=5.0, stop=96.0, take_profit=120.0)

        # Next bar: low touches stop
        exit_bar = make_bar(ts="2024-01-02", o=101, h=103, lo=95, c=98)
        broker.close_due_to_stop("TEST", exit_bar)

        assert len(broker.closed_trades) == 1
        assert abs(broker.closed_trades[0].exit_price - 96.0) < 0.01

    def test_long_stop_gap(self, broker):
        """Open below stop (gap) -> fill at open, not stop."""
        bar = make_bar()
        broker.open_trade("TEST", bar, side=1, size=5.0, stop=96.0, take_profit=120.0)

        # Next bar: opens below stop (gap down)
        exit_bar = make_bar(ts="2024-01-02", o=90, h=92, lo=88, c=91)
        broker.close_due_to_stop("TEST", exit_bar)

        assert len(broker.closed_trades) == 1
        assert abs(broker.closed_trades[0].exit_price - 90.0) < 0.01

    def test_short_stop_normal(self, broker):
        """Short stop hit within bar -> fill at stop price."""
        bar = make_bar()
        broker.open_trade("TEST", bar, side=-1, size=5.0, stop=108.0, take_profit=80.0)

        exit_bar = make_bar(ts="2024-01-02", o=104, h=109, lo=103, c=107)
        broker.close_due_to_stop("TEST", exit_bar)

        assert len(broker.closed_trades) == 1
        assert abs(broker.closed_trades[0].exit_price - 108.0) < 0.01

    def test_short_stop_gap(self, broker):
        """Short: open above stop (gap up) -> fill at open."""
        bar = make_bar()
        broker.open_trade("TEST", bar, side=-1, size=5.0, stop=108.0, take_profit=80.0)

        exit_bar = make_bar(ts="2024-01-02", o=115, h=118, lo=114, c=116)
        broker.close_due_to_stop("TEST", exit_bar)

        assert len(broker.closed_trades) == 1
        assert abs(broker.closed_trades[0].exit_price - 115.0) < 0.01


class TestTPExecution:
    def test_long_tp_hit(self, broker):
        bar = make_bar()
        broker.open_trade("TEST", bar, side=1, size=5.0, stop=90.0, take_profit=110.0)

        exit_bar = make_bar(ts="2024-01-02", o=108, h=112, lo=107, c=111)
        broker.close_due_to_tp("TEST", exit_bar)

        assert len(broker.closed_trades) == 1
        assert abs(broker.closed_trades[0].exit_price - 110.0) < 0.01

    def test_short_tp_hit(self, broker):
        bar = make_bar()
        broker.open_trade("TEST", bar, side=-1, size=5.0, stop=110.0, take_profit=90.0)

        exit_bar = make_bar(ts="2024-01-02", o=93, h=94, lo=89, c=91)
        broker.close_due_to_tp("TEST", exit_bar)

        assert len(broker.closed_trades) == 1
        assert abs(broker.closed_trades[0].exit_price - 90.0) < 0.01

    def test_tp_not_hit(self, broker):
        bar = make_bar()
        broker.open_trade("TEST", bar, side=1, size=5.0, stop=90.0, take_profit=120.0)

        exit_bar = make_bar(ts="2024-01-02", o=102, h=110, lo=101, c=108)
        broker.close_due_to_tp("TEST", exit_bar)

        assert len(broker.closed_trades) == 0
        assert broker.has_open_position("TEST")


class TestSlippageAndCommission:
    def test_slippage_widens_spread(self):
        port = Portfolio(cash=10_000, slippage_bps=100.0)  # 1% slippage
        bar = make_bar(o=100)
        port.broker.open_trade("TEST", bar, side=1, size=5.0, stop=90.0, take_profit=120.0)

        trade = port.broker.positions["TEST"][0]
        # Long entry: slippage makes price worse (higher)
        assert trade.entry_price > 100.0

    def test_commission_reduces_cash(self):
        port = Portfolio(cash=10_000, commission_bps=100.0)  # 1% commission
        bar = make_bar(o=100)
        port.broker.open_trade("TEST", bar, side=1, size=10.0, stop=90.0, take_profit=120.0)

        # Commission on ~$1000 notional at 1% = ~$10
        assert port.cash < 10_000
