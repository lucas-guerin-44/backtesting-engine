"""Tests for core data types."""

import pandas as pd

from backtesting.types import Bar, Trade


class TestBar:
    def test_creation(self):
        bar = Bar(pd.Timestamp("2024-01-01"), 100.0, 105.0, 95.0, 102.0)
        assert bar.open == 100.0
        assert bar.high == 105.0
        assert bar.low == 95.0
        assert bar.close == 102.0
        assert bar.volume is None

    def test_with_volume(self):
        bar = Bar(pd.Timestamp("2024-01-01"), 100.0, 105.0, 95.0, 102.0, volume=1000.0)
        assert bar.volume == 1000.0


class TestTrade:
    def test_creation_defaults(self):
        bar = Bar(pd.Timestamp("2024-01-01"), 100.0, 105.0, 95.0, 102.0)
        trade = Trade(
            entry_bar=bar, side=1, size=10.0, entry_price=100.0,
            stop_price=95.0, take_profit=110.0,
        )
        assert trade.side == 1
        assert trade.size == 10.0
        assert trade.exit_price is None
        assert trade.pnl is None
        assert trade.bars_held == 0

    def test_long_trade(self):
        bar = Bar(pd.Timestamp("2024-01-01"), 100.0, 105.0, 95.0, 102.0)
        trade = Trade(
            entry_bar=bar, side=1, size=10.0, entry_price=100.0,
            stop_price=95.0, take_profit=110.0,
        )
        # Simulate exit
        trade.exit_price = 110.0
        trade.pnl = (110.0 - 100.0) * 1 * 10.0
        assert trade.pnl == 100.0

    def test_short_trade(self):
        bar = Bar(pd.Timestamp("2024-01-01"), 100.0, 105.0, 95.0, 102.0)
        trade = Trade(
            entry_bar=bar, side=-1, size=10.0, entry_price=100.0,
            stop_price=105.0, take_profit=90.0,
        )
        trade.exit_price = 90.0
        trade.pnl = (90.0 - 100.0) * (-1) * 10.0
        assert trade.pnl == 100.0
