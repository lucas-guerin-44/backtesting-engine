"""Tests for the Portfolio equity and drawdown tracking."""

import pandas as pd
import pytest

from backtesting.portfolio import Portfolio
from backtesting.types import Bar


def make_bar(ts, price):
    return Bar(pd.Timestamp(ts), price, price, price, price)


class TestEquityTracking:
    def test_initial_equity(self):
        port = Portfolio(cash=50_000)
        eq = port.compute_equity({})
        assert eq == 50_000

    def test_equity_with_open_long(self):
        port = Portfolio(cash=10_000, commission_bps=0.0, slippage_bps=0.0)
        bar = make_bar("2024-01-01", 100.0)
        port.broker.open_trade("TEST", bar, side=1, size=10.0, stop=90.0, take_profit=120.0)

        # Price moved to 110 -> unrealized PnL = (110-100) * 1 * 10 = 100
        eq = port.compute_equity({"TEST": 110.0})
        assert eq > 10_000

    def test_equity_curve_grows_with_updates(self):
        port = Portfolio(cash=10_000)
        for i in range(5):
            port.update(pd.Timestamp(f"2024-01-0{i+1}"), {"TEST": 100.0})
        assert len(port.equity_curve) == 5


class TestDrawdown:
    def test_no_drawdown_on_flat(self):
        port = Portfolio(cash=10_000)
        for i in range(5):
            port.update(pd.Timestamp(f"2024-01-0{i+1}"), {})
        assert port.max_drawdown == 0.0

    def test_drawdown_after_loss(self):
        port = Portfolio(cash=10_000, commission_bps=0.0, slippage_bps=0.0,
                         max_leverage=10.0)
        bar = make_bar("2024-01-01", 100.0)
        port.broker.open_trade("TEST", bar, side=1, size=50.0, stop=80.0, take_profit=150.0)

        # Price at 100 -> equity ~10000
        port.update(pd.Timestamp("2024-01-01"), {"TEST": 100.0})
        # Price drops to 95 -> unrealized loss = (95-100)*50 = -250
        port.update(pd.Timestamp("2024-01-02"), {"TEST": 95.0})

        assert port.max_drawdown > 0.0


class TestMarginCall:
    def test_margin_call_liquidates_positions(self):
        """When equity drops below 50% of margin requirement, positions are force-closed."""
        # margin_rate=0.1 so projected_margin = notional * 0.1
        # With cash=10_000, max_leverage=10, we can open up to 100k notional
        port = Portfolio(cash=10_000, commission_bps=0.0, slippage_bps=0.0,
                         max_leverage=10.0, margin_rate=0.1)

        bar = make_bar("2024-01-01", 100.0)
        # Open position: 50 units * 100 = $5000 notional
        # margin check: projected_margin = 5000 * 0.1 = 500, equity=10000 > 500 -> OK
        port.broker.open_trade("TEST", bar, side=1, size=50.0, stop=50.0, take_profit=200.0)
        assert port.broker.has_open_position("TEST"), "Trade should have been opened"

        # Initial update at entry price
        port.update(pd.Timestamp("2024-01-01"), {"TEST": 100.0})

        # Price crashes to 10 -> unrealized PnL = (10-100)*50 = -4500
        # Equity = 10000 - 4500 = 5500, gross = 50*10 = 500
        # margin_req = 500 * 0.1 = 50, 50% threshold = 25
        # 5500 > 25 so no margin call yet. Need a bigger position.
        # Instead, use a much larger position to trigger margin call
        port2 = Portfolio(cash=10_000, commission_bps=0.0, slippage_bps=0.0,
                          max_leverage=10.0, margin_rate=0.1)
        bar2 = make_bar("2024-01-01", 100.0)
        # 500 units * 100 = $50k notional (within 10x leverage of 10k = 100k)
        port2.broker.open_trade("TEST", bar2, side=1, size=500.0, stop=50.0, take_profit=200.0)
        assert port2.broker.has_open_position("TEST"), "Large trade should have been opened"

        port2.update(pd.Timestamp("2024-01-01"), {"TEST": 100.0})

        # Price crashes to 80 -> PnL = (80-100)*500 = -10000, equity = 0
        # gross = 500*80 = 40000, margin_req = 40000 * 0.1 = 4000, 50% = 2000
        # equity (0) < 2000 -> margin call!
        port2.update(pd.Timestamp("2024-01-02"), {"TEST": 80.0})

        assert not port2.broker.has_open_position("TEST")
        assert len(port2.broker.closed_trades) >= 1
