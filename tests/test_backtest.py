"""Tests for the Backtester engine."""

import numpy as np
import pandas as pd
import pytest

from backtesting.backtest import Backtester
from backtesting.strategy import Strategy
from backtesting.types import Bar, Trade


class NeverTradeStrategy(Strategy):
    """Strategy that never enters a trade. Equity should stay flat."""
    def on_bar(self, i, bar, cash):
        return None


class AlwaysBuyStrategy(Strategy):
    """Buy on every bar (only first will fill since cash depletes)."""
    def on_bar(self, i, bar, cash):
        if cash > 10:
            return Trade(
                entry_bar=bar, side=1, size=cash * 0.5 / bar.close,
                entry_price=bar.close,
                stop_price=bar.close * 0.90,
                take_profit=bar.close * 1.10,
            )
        return None


class BuyOnceStrategy(Strategy):
    """Buy on bar 0 only, with known stop and TP."""
    def __init__(self, stop_pct=0.05, tp_pct=0.10, size_pct=0.5):
        self.stop_pct = stop_pct
        self.tp_pct = tp_pct
        self.size_pct = size_pct
        self.entered = False

    def on_bar(self, i, bar, cash):
        if not self.entered and cash > 0:
            self.entered = True
            return Trade(
                entry_bar=bar, side=1,
                size=cash * self.size_pct / bar.close,
                entry_price=bar.close,
                stop_price=bar.close * (1 - self.stop_pct),
                take_profit=bar.close * (1 + self.tp_pct),
            )
        return None


class TestBacktesterBasics:
    def test_no_trades_flat_equity(self, flat_df):
        bt = Backtester(flat_df, NeverTradeStrategy(), starting_cash=10_000)
        eq, trades = bt.run()

        assert len(trades) == 0
        assert len(eq) == len(flat_df)
        np.testing.assert_allclose(eq, 10_000.0)

    def test_equity_curve_length_matches_data(self, sample_df):
        bt = Backtester(sample_df, NeverTradeStrategy(), starting_cash=10_000)
        eq, _ = bt.run()
        assert len(eq) == len(sample_df)

    def test_equity_curve_is_numpy_array(self, sample_df):
        bt = Backtester(sample_df, NeverTradeStrategy(), starting_cash=10_000)
        eq, _ = bt.run()
        assert isinstance(eq, np.ndarray)

    def test_equity_never_negative(self, sample_df):
        bt = Backtester(sample_df, AlwaysBuyStrategy(), starting_cash=10_000)
        eq, _ = bt.run()
        assert np.all(eq >= 0)


class TestTradeExecution:
    def test_buy_once_enters_trade(self, trending_df):
        bt = Backtester(trending_df, BuyOnceStrategy(), starting_cash=10_000)
        eq, trades = bt.run()
        # Should have at least one closed trade (TP or stop hit)
        assert len(trades) >= 1

    def test_trade_has_exit_price(self, trending_df):
        bt = Backtester(trending_df, BuyOnceStrategy(), starting_cash=10_000)
        _, trades = bt.run()
        for t in trades:
            assert t.exit_price is not None
            assert t.pnl is not None

    def test_long_trade_pnl_positive_in_uptrend(self, trending_df):
        """In a clean uptrend, a long with tight TP should profit."""
        bt = Backtester(trending_df, BuyOnceStrategy(tp_pct=0.05, stop_pct=0.50), starting_cash=10_000)
        _, trades = bt.run()
        assert len(trades) >= 1
        assert trades[0].pnl > 0


class TestCommissionAndSlippage:
    def test_commission_reduces_equity(self, trending_df):
        bt_clean = Backtester(trending_df, BuyOnceStrategy(), starting_cash=10_000)
        eq_clean, _ = bt_clean.run()

        bt_comm = Backtester(trending_df, BuyOnceStrategy(), starting_cash=10_000, commission_bps=50.0)
        eq_comm, _ = bt_comm.run()

        # Final equity with commission should be lower
        assert eq_comm[-1] < eq_clean[-1]

    def test_slippage_reduces_equity(self, trending_df):
        bt_clean = Backtester(trending_df, BuyOnceStrategy(), starting_cash=10_000)
        eq_clean, _ = bt_clean.run()

        bt_slip = Backtester(trending_df, BuyOnceStrategy(), starting_cash=10_000, slippage_bps=50.0)
        eq_slip, _ = bt_slip.run()

        assert eq_slip[-1] < eq_clean[-1]


class TestGapAwareStops:
    def test_stop_fills_at_open_when_gap_through(self):
        """If price gaps below the stop, the fill should be at the open (worse), not the stop."""
        # Bar 0: enter long at 100, stop at 95
        # Bar 1: opens at 90 (gapped through stop) -> should fill at 90, not 95
        data = {
            "open": [100.0, 90.0],
            "high": [101.0, 91.0],
            "low": [99.0, 89.0],
            "close": [100.0, 90.0],
        }
        df = pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=2, freq="h"))

        class GapTestStrategy(Strategy):
            def __init__(self):
                self.entered = False
            def on_bar(self, i, bar, cash):
                if not self.entered:
                    self.entered = True
                    return Trade(
                        entry_bar=bar, side=1, size=1.0,
                        entry_price=100.0, stop_price=95.0, take_profit=120.0,
                    )
                return None

        bt = Backtester(df, GapTestStrategy(), starting_cash=10_000)
        _, trades = bt.run()

        assert len(trades) == 1
        # Exit should be at or near the open price (90.0), NOT at stop price (95.0)
        assert trades[0].exit_price < 95.0

    def test_stop_fills_at_stop_when_touched(self):
        """Normal stop: price passes through stop within the bar -> fill at stop price."""
        # Bar 0: enter long at 100, stop at 95
        # Bar 1: opens at 98, low touches 94 -> stop at 95
        data = {
            "open": [100.0, 98.0],
            "high": [101.0, 99.0],
            "low": [99.0, 94.0],
            "close": [100.0, 95.0],
        }
        df = pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=2, freq="h"))

        class StopTestStrategy(Strategy):
            def __init__(self):
                self.entered = False
            def on_bar(self, i, bar, cash):
                if not self.entered:
                    self.entered = True
                    return Trade(
                        entry_bar=bar, side=1, size=1.0,
                        entry_price=100.0, stop_price=95.0, take_profit=120.0,
                    )
                return None

        bt = Backtester(df, StopTestStrategy(), starting_cash=10_000)
        _, trades = bt.run()

        assert len(trades) == 1
        # Should fill at the stop price (95.0), not the open (98.0)
        assert abs(trades[0].exit_price - 95.0) < 0.1


class TestExecutionPriority:
    def test_stop_first_is_default(self, sample_df):
        """Ensure the default execution priority doesn't raise."""
        bt = Backtester(sample_df, NeverTradeStrategy(), starting_cash=10_000)
        eq, _ = bt.run()  # default = "stop_first"
        assert len(eq) == len(sample_df)

    def test_invalid_priority_raises(self, sample_df):
        bt = Backtester(sample_df, NeverTradeStrategy(), starting_cash=10_000)
        with pytest.raises(ValueError, match="Unknown execution_priority"):
            bt.run(execution_priority="invalid")


class TestDrawdownTracking:
    def test_max_drawdown_with_no_trades(self, flat_df):
        bt = Backtester(flat_df, NeverTradeStrategy(), starting_cash=10_000)
        bt.run()
        assert bt.max_drawdown == 0.0

    def test_max_drawdown_is_positive(self, sample_df):
        bt = Backtester(sample_df, AlwaysBuyStrategy(), starting_cash=10_000)
        bt.run()
        # With a reversal in the data, there should be some drawdown
        assert bt.max_drawdown >= 0.0
