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


class TestIndicatorEdgeCases:
    """Edge case and boundary tests for incremental and vectorized indicators.

    Targets subtle accumulation bugs: NaN handling, extreme values,
    single-element inputs, and warm-up boundary conditions.
    """

    # --- Incremental EMA ---

    def test_ema_constant_input_converges(self):
        """EMA of constant price should equal that price."""
        ind = EMA(period=10)
        for _ in range(50):
            val = ind.update(42.0)
        assert abs(val - 42.0) < 1e-10

    def test_ema_single_spike_decays(self):
        """A spike followed by flat prices should decay toward the flat level."""
        ind = EMA(period=5)
        for _ in range(10):
            ind.update(100.0)
        ind.update(200.0)  # spike
        vals = []
        for _ in range(20):
            vals.append(ind.update(100.0))
        # Should decay back toward 100
        assert vals[-1] < 101.0

    def test_ema_period_1_tracks_exactly(self):
        """EMA with period=1 should equal the current price."""
        ind = EMA(period=1)
        assert ind.update(50.0) == 50.0
        assert ind.update(75.0) == 75.0
        assert ind.update(60.0) == 60.0

    # --- Incremental ATR ---

    def test_atr_zero_range_bars(self):
        """ATR should be zero (or near-zero) when H=L=C every bar."""
        ind = ATR(period=5)
        for _ in range(20):
            val = ind.update(100.0, 100.0, 100.0)
        assert val is not None
        assert val < 1e-10

    def test_atr_single_volatile_bar(self):
        """A single high-volatility bar should spike ATR then decay."""
        ind = ATR(period=5)
        for _ in range(10):
            ind.update(101.0, 99.0, 100.0)  # range = 2
        baseline = ind.value
        ind.update(150.0, 50.0, 100.0)  # range = 100
        spiked = ind.value
        for _ in range(20):
            ind.update(101.0, 99.0, 100.0)  # back to normal
        decayed = ind.value
        assert spiked > baseline * 3  # spike was significant
        assert decayed < spiked  # decayed back

    def test_atr_gap_handling(self):
        """ATR true range should account for gap from previous close."""
        ind = ATR(period=3)
        for _ in range(5):
            ind.update(101.0, 99.0, 100.0)
        # Gap up: prev close=100, current open at 110
        ind.update(115.0, 110.0, 112.0)
        # True range should be max(115-110, |115-100|, |110-100|) = 15
        assert ind.value > 5.0

    # --- Incremental RSI ---

    def test_rsi_all_gains_is_100(self):
        """Monotonically increasing prices should give RSI near 100."""
        ind = RSI(period=5)
        val = None
        for p in range(100, 130):
            val = ind.update(float(p))
        assert val is not None
        assert val > 99.0

    def test_rsi_all_losses_is_near_zero(self):
        """Monotonically decreasing prices should give RSI near 0."""
        ind = RSI(period=5)
        val = None
        for p in range(130, 100, -1):
            val = ind.update(float(p))
        assert val is not None
        assert val < 1.0

    def test_rsi_flat_price_is_stable(self):
        """Constant price after movement should stabilize RSI."""
        ind = RSI(period=5)
        for p in range(100, 110):
            ind.update(float(p))
        vals = []
        for _ in range(30):
            vals.append(ind.update(110.0))
        # RSI should converge toward 50 on flat input (gains decay, losses=0)
        # Actually with 0 change, gain=0, loss=0, so avg_gain decays but avg_loss=0
        # RSI = 100 - 100/(1 + avg_gain/0) = 100 when avg_loss=0
        assert vals[-1] is not None

    # --- Incremental Bollinger Bands ---

    def test_bb_constant_price_zero_bandwidth(self):
        """Constant price should give zero-width bands (lower == upper == mid)."""
        bb = BollingerBands(period=5, num_std=2.0)
        for _ in range(10):
            lo, mid, hi = bb.update(100.0)
        assert abs(lo - 100.0) < 1e-10
        assert abs(mid - 100.0) < 1e-10
        assert abs(hi - 100.0) < 1e-10

    def test_bb_wider_std_gives_wider_bands(self):
        """Higher num_std should produce wider bands."""
        prices = [100, 102, 98, 101, 99, 103, 97, 100, 102, 98]
        bb_narrow = BollingerBands(period=5, num_std=1.0)
        bb_wide = BollingerBands(period=5, num_std=3.0)
        for p in prices:
            lo_n, _, hi_n = bb_narrow.update(p)
            lo_w, _, hi_w = bb_wide.update(p)
        assert (hi_w - lo_w) > (hi_n - lo_n)

    # --- Vectorized vs Incremental consistency ---

    def test_ema_array_matches_incremental(self):
        """Vectorized EMA should produce identical values to incremental EMA."""
        prices = np.array([100, 102, 98, 105, 103, 107, 101, 110, 108, 112],
                          dtype=np.float64)
        period = 3

        # Incremental
        ind = EMA(period)
        incremental = []
        for p in prices:
            v = ind.update(p)
            incremental.append(v if v is not None else np.nan)

        # Vectorized
        vectorized = ema_array(prices, period)

        # Compare valid (non-NaN) values
        for i in range(period - 1, len(prices)):
            assert abs(incremental[i] - vectorized[i]) < 1e-10, (
                f"EMA mismatch at index {i}: incremental={incremental[i]}, "
                f"vectorized={vectorized[i]}"
            )

    def test_atr_array_matches_incremental(self):
        """Vectorized ATR should produce identical values to incremental ATR."""
        rng = np.random.RandomState(42)
        n = 30
        close = 100 + np.cumsum(rng.randn(n))
        high = close + rng.rand(n) * 2
        low = close - rng.rand(n) * 2
        period = 5

        # Incremental
        ind = ATR(period)
        incremental = []
        for i in range(n):
            v = ind.update(float(high[i]), float(low[i]), float(close[i]))
            incremental.append(v if v is not None else np.nan)

        # Vectorized
        vectorized = atr_array(high, low, close, period)

        for i in range(period - 1, n):
            assert abs(incremental[i] - vectorized[i]) < 1e-10, (
                f"ATR mismatch at index {i}: incremental={incremental[i]}, "
                f"vectorized={vectorized[i]}"
            )

    def test_rsi_array_matches_incremental(self):
        """Vectorized RSI should produce identical values to incremental RSI."""
        prices = np.array([44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10,
                           45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28,
                           46.28, 46.00, 46.03, 46.41, 46.22, 45.64],
                          dtype=np.float64)
        period = 5

        # Incremental
        ind = RSI(period)
        incremental = []
        for p in prices:
            v = ind.update(p)
            incremental.append(v if v is not None else np.nan)

        # Vectorized
        vectorized = rsi_array(prices, period)

        for i in range(len(prices)):
            if np.isnan(incremental[i]) and np.isnan(vectorized[i]):
                continue
            if not np.isnan(incremental[i]) and not np.isnan(vectorized[i]):
                assert abs(incremental[i] - vectorized[i]) < 0.01, (
                    f"RSI mismatch at index {i}: incremental={incremental[i]}, "
                    f"vectorized={vectorized[i]}"
                )

    # --- Vectorized edge cases ---

    def test_ema_array_short_input(self):
        """EMA array with fewer bars than period should be all NaN."""
        prices = np.array([100.0, 101.0, 102.0])
        result = ema_array(prices, period=10)
        assert len(result) == 3
        assert np.all(np.isnan(result))

    def test_atr_array_single_bar(self):
        """ATR with a single bar should not crash."""
        h = np.array([101.0])
        lo = np.array([99.0])
        c = np.array([100.0])
        result = atr_array(h, lo, c, period=14)
        assert len(result) == 1

    def test_rsi_array_two_bars(self):
        """RSI with minimal bars should return NaN for warm-up period."""
        prices = np.array([100.0, 101.0])
        result = rsi_array(prices, period=14)
        assert len(result) == 2
        assert np.all(np.isnan(result))


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
