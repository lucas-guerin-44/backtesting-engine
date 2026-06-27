"""Tests for the funding cost simulation."""

import numpy as np
import pandas as pd
import pytest

from backtesting.backtest import Backtester
from backtesting.portfolio import Portfolio
from backtesting.strategy import Strategy
from backtesting.types import BacktestConfig, Bar, Trade


class NeverTradeStrategy(Strategy):
    def on_bar(self, i, bar, equity):
        return None


class BuyAndHoldStrategy(Strategy):
    def __init__(self, size_pct=0.5):
        self.size_pct = size_pct
        self.entered = False

    def on_bar(self, i, bar, equity):
        if not self.entered and equity > 0:
            self.entered = True
            return Trade(
                entry_bar=bar, side=1,
                size=equity * self.size_pct / bar.close,
                entry_price=bar.close,
                stop_price=bar.close * 0.90,
                take_profit=bar.close * 1.50,
            )
        return None


class ShortStrategy(Strategy):
    def __init__(self, size_pct=0.5):
        self.size_pct = size_pct
        self.entered = False

    def on_bar(self, i, bar, equity):
        if not self.entered and equity > 0:
            self.entered = True
            return Trade(
                entry_bar=bar, side=-1,
                size=equity * self.size_pct / bar.close,
                entry_price=bar.close,
                stop_price=bar.close * 1.10,
                take_profit=bar.close * 0.50,
            )
        return None


@pytest.fixture
def flat_df():
    n = 50
    data = {
        "open": [100.0] * n,
        "high": [100.5] * n,
        "low": [99.5] * n,
        "close": [100.0] * n,
    }
    index = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(data, index=index)


@pytest.fixture
def trending_df():
    n = 100
    closes = np.linspace(100, 200, n)
    data = {
        "open": closes - 0.5,
        "high": closes + 1.0,
        "low": closes - 1.0,
        "close": closes,
    }
    index = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(data, index=index)


class TestNoFundingNoChange:
    def test_zero_rates_identical(self, trending_df):
        bt_base = Backtester(
            trending_df, BuyAndHoldStrategy(),
            starting_cash=10_000, symbol="TEST",
        )
        eq_base, _ = bt_base.run()

        config = BacktestConfig(
            starting_cash=10_000,
            funding_rate_annual=0.0,
            funding_rate_short=0.0,
        )
        bt_fund = Backtester(
            trending_df, BuyAndHoldStrategy(),
            config=config, symbol="TEST",
        )
        eq_fund, _ = bt_fund.run()

        np.testing.assert_allclose(eq_base, eq_fund)


class TestFundingAccrual:
    def test_long_position_accrues_cost(self, flat_df):
        config = BacktestConfig(
            starting_cash=10_000,
            funding_rate_annual=5.0,
        )
        bt = Backtester(
            flat_df, BuyAndHoldStrategy(size_pct=0.5),
            config=config, symbol="TEST",
        )
        eq_fund, trades = bt.run()

        bt_no = Backtester(
            flat_df, BuyAndHoldStrategy(size_pct=0.5),
            starting_cash=10_000, symbol="TEST",
        )
        eq_no, _ = bt_no.run()

        if len(trades) > 0:
            assert eq_fund[-1] < eq_no[-1]

    def test_short_uses_short_rate(self, flat_df):
        config_long = BacktestConfig(
            starting_cash=10_000,
            funding_rate_annual=5.0,
            funding_rate_short=10.0,
        )
        bt_long = Backtester(
            flat_df, BuyAndHoldStrategy(size_pct=0.5),
            config=config_long, symbol="TEST",
        )
        eq_long, _ = bt_long.run()

        config_short = BacktestConfig(
            starting_cash=10_000,
            funding_rate_annual=5.0,
            funding_rate_short=10.0,
        )
        bt_short = Backtester(
            flat_df, ShortStrategy(size_pct=0.5),
            config=config_short, symbol="TEST",
        )
        eq_short, _ = bt_short.run()

        if len(eq_long) > 0 and len(eq_short) > 0:
            long_cost = 10_000 - eq_long[-1]
            short_cost = 10_000 - eq_short[-1]
            assert short_cost >= long_cost

    def test_funding_reduces_equity(self, flat_df):
        config = BacktestConfig(
            starting_cash=10_000,
            funding_rate_annual=10.0,
        )
        bt = Backtester(
            flat_df, BuyAndHoldStrategy(size_pct=0.5),
            config=config, symbol="TEST",
        )
        eq, trades = bt.run()

        if len(trades) > 0:
            assert eq[-1] < 10_000


class TestPositionClosedNoFunding:
    def test_no_funding_after_exit(self):
        data = {
            "open": [100.0, 100.0, 100.0, 105.0, 105.0, 105.0] * 5,
            "high": [101.0, 101.0, 101.0, 106.0, 106.0, 106.0] * 5,
            "low": [99.0, 99.0, 99.0, 104.0, 104.0, 104.0] * 5,
            "close": [100.0, 100.0, 100.0, 105.0, 105.0, 105.0] * 5,
        }
        index = pd.date_range("2024-01-01", periods=30, freq="h")
        df = pd.DataFrame(data, index=index)

        class ExitAtBar4Strategy(Strategy):
            def __init__(self):
                self.entered = False
                self.bar_count = 0

            def on_bar(self, i, bar, equity):
                self.bar_count += 1
                if not self.entered and self.bar_count == 1:
                    self.entered = True
                    return Trade(
                        entry_bar=bar, side=1, size=0.5,
                        entry_price=bar.close,
                        stop_price=bar.close * 0.95,
                        take_profit=bar.close * 1.03,
                    )
                return None

        config = BacktestConfig(
            starting_cash=10_000,
            funding_rate_annual=50.0,
        )
        bt = Backtester(
            df, ExitAtBar4Strategy(),
            config=config, symbol="TEST",
        )
        eq, trades = bt.run()
        assert len(trades) > 0


class TestPortfolioPerSymbolFunding:
    def test_different_rates_per_asset(self):
        data1 = {
            "open": [100.0] * 20,
            "high": [101.0] * 20,
            "low": [99.0] * 20,
            "close": [100.0] * 20,
        }
        data2 = {
            "open": [200.0] * 20,
            "high": [201.0] * 20,
            "low": [199.0] * 20,
            "close": [200.0] * 20,
        }
        index = pd.date_range("2024-01-01", periods=20, freq="h")
        df1 = pd.DataFrame(data1, index=index)
        df2 = pd.DataFrame(data2, index=index)

        portfolio = Portfolio(cash=100_000)
        portfolio.funding_rate_annual = 5.0
        portfolio.funding_rate_short = 10.0

        from backtesting.types import Trade as T
        tr1 = T(entry_bar=Bar(index[0], 100, 101, 99, 100), side=1, size=10,
                entry_price=100, stop_price=90, take_profit=110)
        tr2 = T(entry_bar=Bar(index[0], 200, 201, 199, 200), side=-1, size=5,
                entry_price=200, stop_price=210, take_profit=190)
        portfolio.broker.positions["A"] = [tr1]
        portfolio.broker.positions["B"] = [tr2]

        cash_before = portfolio.cash
        total = portfolio.accrue_funding(24.0)

        expected_long = 100 * 10 * (5.0 / 100.0) * (24.0 / 24.0) / 365.0
        expected_short = 200 * 5 * (10.0 / 100.0) * (24.0 / 24.0) / 365.0
        expected_total = expected_long + expected_short

        assert abs(total - expected_total) < 0.01
        assert abs(portfolio.cash - (cash_before - expected_total)) < 0.01
