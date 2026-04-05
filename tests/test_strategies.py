"""Tests for trading strategies and indicator helpers."""

import numpy as np
import pandas as pd
import pytest

from backtesting.backtest import Backtester
from backtesting.indicators import EMA, ATR, RSI, BollingerBands, ema_array, atr_array, rsi_array
from backtesting.strategy import Strategy
from backtesting.types import Bar, Trade
from strategies import (
    DonchianBreakoutStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    TrendFollowingStrategy,
    risk_adjusted_size,
)

ALL_STRATEGIES = [
    TrendFollowingStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    DonchianBreakoutStrategy,
]


class TestStrategyBaseClass:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            Strategy()

    def test_all_strategies_are_subclasses(self):
        for cls in ALL_STRATEGIES:
            assert issubclass(cls, Strategy), f"{cls.__name__} is not a Strategy subclass"


class TestIncrementalIndicators:
    def test_ema_warmup(self):
        ind = EMA(period=5)
        for i in range(4):
            assert ind.update(100.0 + i) is None
        assert ind.update(104.0) is not None

    def test_ema_tracks_price(self):
        ind = EMA(period=3)
        vals = []
        for p in [100, 102, 104, 106, 108]:
            v = ind.update(p)
            if v is not None:
                vals.append(v)
        # EMA should be increasing in an uptrend
        assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))

    def test_atr_warmup(self):
        ind = ATR(period=5)
        for i in range(4):
            assert ind.update(101.0, 99.0, 100.0) is None
        assert ind.update(101.0, 99.0, 100.0) is not None

    def test_atr_positive(self):
        ind = ATR(period=3)
        for _ in range(5):
            ind.update(105.0, 95.0, 100.0)
        assert ind.value > 0

    def test_rsi_warmup(self):
        ind = RSI(period=5)
        for i in range(5):
            assert ind.update(100.0 + i) is None
        assert ind.update(105.0) is not None

    def test_rsi_high_on_uptrend(self):
        ind = RSI(period=5)
        val = None
        for p in range(100, 120):
            val = ind.update(float(p))
        assert val is not None
        assert val > 70

    def test_bb_warmup(self):
        bb = BollingerBands(period=5)
        for i in range(4):
            lo, mid, hi = bb.update(100.0)
            assert lo is None
        lo, mid, hi = bb.update(100.0)
        assert lo is not None

    def test_bb_bands_symmetric(self):
        bb = BollingerBands(period=5, num_std=2.0)
        for p in [100, 101, 102, 101, 100]:
            lo, mid, hi = bb.update(p)
        assert lo is not None
        # Bands should be symmetric around mid
        assert abs((hi - mid) - (mid - lo)) < 0.001


class TestVectorizedIndicators:
    def test_ema_array_length(self):
        prices = np.linspace(100, 200, 50)
        result = ema_array(prices, period=10)
        assert len(result) == 50
        assert np.isnan(result[0])
        assert not np.isnan(result[-1])

    def test_atr_array_length(self):
        n = 50
        h = np.random.RandomState(42).uniform(101, 105, n)
        l = h - np.random.RandomState(43).uniform(2, 4, n)
        c = (h + l) / 2
        result = atr_array(h, l, c, period=14)
        assert len(result) == n
        assert not np.isnan(result[-1])

    def test_rsi_array_range(self):
        prices = np.linspace(100, 200, 50)
        result = rsi_array(prices, period=14)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0)
        assert np.all(valid <= 100)


class TestRiskAdjustedSize:
    def test_basic_sizing(self):
        size = risk_adjusted_size(10_000, 100.0, 95.0, 0.02, 10_000, 0.15)
        assert 35 < size < 45  # Risk $200, $5 per unit -> ~40

    def test_zero_cash(self):
        assert risk_adjusted_size(0, 100, 95, 0.02, 10_000, 0.15) == 0.0

    def test_circuit_breaker(self):
        assert risk_adjusted_size(8500, 100, 95, 0.02, 10_000, 0.15) == 0.0

    def test_scales_with_drawdown(self):
        full = risk_adjusted_size(10_000, 100, 95, 0.02, 10_000, 0.15)
        partial = risk_adjusted_size(9_500, 100, 95, 0.02, 10_000, 0.15)
        assert partial < full

    def test_capped_to_cash(self):
        size = risk_adjusted_size(1000, 100, 99.99, 0.5, 1000, 0.15)
        assert size <= 1000 / 100.0


class TestStrategySignals:
    def _run(self, cls, df, **kwargs):
        return Backtester(df, cls(**kwargs), starting_cash=10_000).run()

    def test_trend_following(self, trending_df):
        eq, _ = self._run(TrendFollowingStrategy, trending_df)
        assert len(eq) == len(trending_df)

    def test_mean_reversion(self, flat_df):
        eq, _ = self._run(MeanReversionStrategy, flat_df)
        assert len(eq) == len(flat_df)

    def test_momentum(self, trending_df):
        eq, _ = self._run(MomentumStrategy, trending_df)
        assert len(eq) == len(trending_df)

    def test_donchian(self, trending_df):
        eq, _ = self._run(DonchianBreakoutStrategy, trending_df)
        assert len(eq) == len(trending_df)

    def test_all_handle_short_data(self):
        df = pd.DataFrame({
            "open": [100.0, 101.0], "high": [102.0, 103.0],
            "low": [98.0, 99.0], "close": [101.0, 102.0],
        }, index=pd.date_range("2024-01-01", periods=2, freq="h"))

        for cls in ALL_STRATEGIES:
            eq, _ = Backtester(df, cls(), starting_cash=10_000).run()
            assert len(eq) == 2, f"{cls.__name__} failed"

    def test_drawdown_guard(self):
        n = 100
        prices = np.concatenate([np.linspace(100, 80, 40), np.linspace(80, 100, 60)])
        df = pd.DataFrame({
            "open": prices - 0.5, "high": prices + 1.0,
            "low": prices - 1.0, "close": prices,
        }, index=pd.date_range("2024-01-01", periods=n, freq="h"))

        bt = Backtester(df, MomentumStrategy(
            risk_per_trade=0.10, max_dd_halt=0.05, cooldown_bars=1, lookback=5
        ), starting_cash=10_000)
        eq, _ = bt.run()
        assert eq[-1] > 0

    def test_drawdown_guard_uses_equity_not_cash(self):
        """The drawdown guard must track total equity (cash + open P&L),
        not just available cash. This test verifies the guard triggers
        at the correct equity level."""
        # Scenario: enter long, price drops enough to push equity drawdown
        # past the halt threshold, then recovers. If the guard incorrectly
        # uses cash (which stays constant during an open position), it won't
        # trigger at the right time.

        # Use a strict 5% halt so the test is unambiguous
        halt = 0.05

        # Test risk_adjusted_size directly with equity vs cash semantics
        # Peak equity = 10000, current equity = 9400 (6% drawdown > 5% halt)
        assert risk_adjusted_size(
            equity=9400, entry_price=100, stop_price=95,
            risk_per_trade=0.02, peak_equity=10_000, max_dd_halt=halt,
        ) == 0.0, "Guard should halt at 6% drawdown with 5% threshold"

        # Peak equity = 10000, current equity = 9600 (4% drawdown < 5% halt)
        size = risk_adjusted_size(
            equity=9600, entry_price=100, stop_price=95,
            risk_per_trade=0.02, peak_equity=10_000, max_dd_halt=halt,
        )
        assert size > 0, "Guard should allow trading at 4% drawdown with 5% threshold"

        # Verify the strategy receives equity (not available_cash) from Backtester
        # by running a backtest where the distinction matters
        n = 60
        # Uptrend then sharp drop
        prices = np.concatenate([
            np.linspace(100, 120, 30),  # Uptrend (triggers entries)
            np.linspace(120, 100, 30),  # Drop (should trigger halt)
        ])
        df = pd.DataFrame({
            "open": prices - 0.3, "high": prices + 0.5,
            "low": prices - 0.5, "close": prices,
        }, index=pd.date_range("2024-01-01", periods=n, freq="h"))

        # Very aggressive risk with tight halt — maximizes the difference
        # between cash-based and equity-based drawdown tracking
        bt = Backtester(df, MomentumStrategy(
            risk_per_trade=0.10, max_dd_halt=halt, cooldown_bars=1,
            lookback=5, entry_threshold=0.02,
        ), starting_cash=10_000)
        eq, trades = bt.run()

        # The equity curve should never go below (1 - halt) * peak
        # (within a small tolerance for execution costs)
        peak = np.maximum.accumulate(eq)
        min_allowed = peak * (1 - halt - 0.05)  # 5% tolerance for slippage
        assert np.all(eq >= min_allowed), (
            f"Equity dropped below halt threshold: "
            f"min equity {eq.min():.0f}, peak at that point {peak[eq.argmin()]:.0f}"
        )
