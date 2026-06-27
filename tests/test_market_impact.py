"""Tests for the market impact model."""

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


class BuyOnceStrategy(Strategy):
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
                take_profit=bar.close * 1.10,
            )
        return None


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


class TestNoAdvFlatSlippage:
    def test_no_adv_identical_to_base(self, trending_df):
        bt_base = Backtester(
            trending_df, BuyOnceStrategy(),
            starting_cash=10_000, slippage_bps=2.0, symbol="TEST",
        )
        _, trades_base = bt_base.run()

        config = BacktestConfig(
            starting_cash=10_000, slippage_bps=2.0,
            typical_daily_volume=None,
        )
        btImpact = Backtester(
            trending_df, BuyOnceStrategy(),
            config=config, symbol="TEST",
        )
        _, trades_impact = btImpact.run()

        assert len(trades_base) == len(trades_impact)
        if len(trades_base) > 0:
            assert abs(trades_base[0].entry_price - trades_impact[0].entry_price) < 0.01


class TestImpactScaling:
    def test_small_order_minimal_impact(self):
        portfolio = Portfolio(
            cash=100_000, slippage_bps=1.0,
            typical_daily_volume=500_000, daily_volatility=120.0,
        )
        impact = portfolio.impact_slippage_bps(0.05)
        assert impact < 1.5

    def test_large_order_significant_impact(self):
        portfolio = Portfolio(
            cash=100_000, slippage_bps=1.0,
            typical_daily_volume=500_000, daily_volatility=120.0,
        )
        impact = portfolio.impact_slippage_bps(100)
        assert impact > 1.5

    def test_impact_scales_with_sqrt(self):
        portfolio = Portfolio(
            cash=100_000, slippage_bps=0.0,
            typical_daily_volume=500_000, daily_volatility=120.0,
        )
        impact_1x = portfolio.impact_slippage_bps(10)
        impact_4x = portfolio.impact_slippage_bps(40)
        ratio = impact_4x / impact_1x if impact_1x > 0 else 0
        assert 1.8 < ratio < 2.2

    def test_zero_order_size_returns_base(self):
        portfolio = Portfolio(
            cash=100_000, slippage_bps=3.0,
            typical_daily_volume=500_000, daily_volatility=120.0,
        )
        assert portfolio.impact_slippage_bps(0) == 3.0


class TestAutoComputeVolatility:
    def test_computes_from_prices(self):
        portfolio = Portfolio(
            cash=100_000, typical_daily_volume=500_000,
        )
        prices = np.random.lognormal(mean=7.0, sigma=0.02, size=100)
        portfolio.compute_daily_volatility(prices)
        assert portfolio.daily_volatility is not None
        assert portfolio.daily_volatility > 0

    def test_too_few_prices_skips(self):
        portfolio = Portfolio(
            cash=100_000, typical_daily_volume=500_000,
        )
        portfolio.compute_daily_volatility(np.array([100.0, 101.0]))
        assert portfolio.daily_volatility is None


class TestBacktestWithImpact:
    def test_full_backtest_with_impact(self, trending_df):
        config = BacktestConfig(
            starting_cash=10_000,
            slippage_bps=1.0,
            typical_daily_volume=500_000,
            impact_scaling=0.5,
        )
        bt = Backtester(
            trending_df, BuyOnceStrategy(size_pct=0.5),
            config=config, symbol="TEST",
        )
        equity, trades = bt.run()
        assert len(trades) > 0
        assert equity[-1] > 0

    def test_impact_increases_cost_vs_no_impact(self, trending_df):
        bt_no_impact = Backtester(
            trending_df, BuyOnceStrategy(size_pct=0.5),
            starting_cash=10_000, slippage_bps=1.0, symbol="TEST",
        )
        _, trades_no = bt_no_impact.run()

        config = BacktestConfig(
            starting_cash=10_000, slippage_bps=1.0,
            typical_daily_volume=100,
            daily_volatility=200.0,
        )
        bt_impact = Backtester(
            trending_df, BuyOnceStrategy(size_pct=0.5),
            config=config, symbol="TEST",
        )
        _, trades_impact = bt_impact.run()

        if len(trades_no) > 0 and len(trades_impact) > 0:
            assert trades_impact[0].entry_price >= trades_no[0].entry_price
