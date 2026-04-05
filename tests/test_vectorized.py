"""Tests for the vectorized backtester."""

import numpy as np
import pandas as pd
import pytest

from backtesting.vectorized import VectorizedBacktester, VectorizedTrade, shift
from backtesting.vectorized_signals import (
    trend_following_signals, mean_reversion_signals,
    momentum_signals, donchian_signals,
)


@pytest.fixture
def trending_arrays():
    """100 bars of clean uptrend as raw numpy arrays."""
    n = 100
    c = np.linspace(100, 200, n)
    return c - 0.5, c + 1.0, c - 1.0, c  # open, high, low, close


@pytest.fixture
def flat_arrays():
    n = 100
    c = np.full(n, 100.0)
    return c - 0.5, c + 0.5, c - 0.5, c


class TestShift:
    def test_shift_forward(self):
        arr = np.array([1, 2, 3, 4, 5], dtype=float)
        result = shift(arr, 1)
        assert np.isnan(result[0])
        np.testing.assert_array_equal(result[1:], [1, 2, 3, 4])

    def test_shift_zero(self):
        arr = np.array([1, 2, 3], dtype=float)
        np.testing.assert_array_equal(shift(arr, 0), arr)


class TestVectorizedBacktester:
    def test_no_entries_flat_equity(self, flat_arrays):
        o, h, lo, c = flat_arrays
        bt = VectorizedBacktester(o, h, lo, c, starting_cash=10_000)
        entries = np.zeros(len(c), dtype=bool)
        sides = np.zeros(len(c), dtype=int)
        stops = np.full(len(c), np.nan)
        tps = np.full(len(c), np.nan)

        eq, trades = bt.run(entries, sides, stops, tps)
        assert len(trades) == 0
        np.testing.assert_allclose(eq, 10_000.0)

    def test_equity_length_matches_data(self, trending_arrays):
        o, h, lo, c = trending_arrays
        bt = VectorizedBacktester(o, h, lo, c)
        entries = np.zeros(len(c), dtype=bool)
        eq, _ = bt.run(entries, np.zeros(len(c), dtype=int),
                        np.full(len(c), np.nan), np.full(len(c), np.nan))
        assert len(eq) == len(c)

    def test_single_winning_trade(self):
        """Long entry at 100, TP at 110 — should hit TP and profit."""
        n = 20
        c = np.linspace(100, 120, n)
        o, h, lo = c - 0.5, c + 1.0, c - 1.0

        entries = np.zeros(n, dtype=bool)
        entries[0] = True
        sides = np.ones(n, dtype=int)
        stops = np.full(n, 80.0)
        tps = np.full(n, 110.0)

        bt = VectorizedBacktester(o, h, lo, c, starting_cash=10_000)
        eq, trades = bt.run(entries, sides, stops, tps)

        assert len(trades) == 1
        assert trades[0].pnl > 0
        assert eq[-1] > 10_000

    def test_gap_aware_stop(self):
        """Price gaps past stop — should fill at open, not stop."""
        o = np.array([100.0, 85.0])
        h = np.array([101.0, 86.0])
        lo = np.array([99.0, 84.0])
        c = np.array([100.0, 85.0])

        entries = np.array([True, False])
        sides = np.array([1, 0])
        stops = np.array([90.0, np.nan])
        tps = np.array([120.0, np.nan])

        bt = VectorizedBacktester(o, h, lo, c, starting_cash=10_000)
        eq, trades = bt.run(entries, sides, stops, tps)

        assert len(trades) == 1
        # Should fill at open (85), not stop (90)
        assert trades[0].exit_price < 90.0

    def test_commission_reduces_equity(self, trending_arrays):
        o, h, lo, c = trending_arrays
        entries = np.zeros(len(c), dtype=bool)
        entries[0] = True
        sides = np.ones(len(c), dtype=int)
        stops = np.full(len(c), 50.0)
        tps = np.full(len(c), 150.0)

        bt_clean = VectorizedBacktester(o, h, lo, c, starting_cash=10_000, commission_bps=0.0)
        eq_clean, _ = bt_clean.run(entries, sides, stops, tps)

        bt_comm = VectorizedBacktester(o, h, lo, c, starting_cash=10_000, commission_bps=50.0)
        eq_comm, _ = bt_comm.run(entries, sides, stops, tps)

        assert eq_comm[-1] < eq_clean[-1]

    def test_drawdown_guard_halts_trading(self):
        """After losses push drawdown past threshold, no new trades should fire."""
        n = 50
        c = np.concatenate([np.linspace(100, 80, 20), np.linspace(80, 100, 30)])
        o, h, lo = c - 0.5, c + 1.0, c - 1.0

        entries = np.ones(n, dtype=bool)
        sides = np.ones(n, dtype=int)
        stops = c - 5.0
        tps = c + 20.0

        bt = VectorizedBacktester(o, h, lo, c, starting_cash=10_000)
        eq, trades = bt.run(entries, sides, stops, tps, max_dd_halt=0.05)

        assert eq[-1] > 0  # Didn't blow up


class TestVectorizedSignals:
    def test_trend_following_produces_valid_arrays(self, trending_arrays):
        o, h, lo, c = trending_arrays
        entries, sides, stops, tps = trend_following_signals(o, h, lo, c)
        assert entries.dtype == bool
        assert len(entries) == len(c)
        assert set(np.unique(sides[entries])).issubset({-1, 1})

    def test_mean_reversion_produces_valid_arrays(self, flat_arrays):
        o, h, lo, c = flat_arrays
        entries, sides, stops, tps = mean_reversion_signals(o, h, lo, c)
        assert entries.dtype == bool
        assert len(entries) == len(c)

    def test_momentum_produces_valid_arrays(self, trending_arrays):
        o, h, lo, c = trending_arrays
        entries, sides, stops, tps = momentum_signals(o, h, lo, c)
        assert entries.dtype == bool
        assert len(entries) == len(c)

    def test_donchian_produces_valid_arrays(self, trending_arrays):
        o, h, lo, c = trending_arrays
        entries, sides, stops, tps = donchian_signals(o, h, lo, c)
        assert entries.dtype == bool
        assert len(entries) == len(c)

    def test_full_roundtrip(self, trending_arrays):
        """Signal generator -> vectorized backtester -> results."""
        o, h, lo, c = trending_arrays
        entries, sides, stops, tps = trend_following_signals(o, h, lo, c)

        bt = VectorizedBacktester(o, h, lo, c, starting_cash=10_000)
        eq, trades = bt.run(entries, sides, stops, tps)

        assert len(eq) == len(c)
        assert eq[-1] > 0
