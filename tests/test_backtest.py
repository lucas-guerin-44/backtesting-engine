"""Tests for the Backtester engine."""

import numpy as np
import pandas as pd
import pytest

from backtesting.backtest import Backtester
from backtesting.strategy import Strategy
from backtesting.types import BacktestConfig, Bar, Trade


class NeverTradeStrategy(Strategy):
    """Strategy that never enters a trade. Equity should stay flat."""
    def on_bar(self, i, bar, equity):
        return None


class AlwaysBuyStrategy(Strategy):
    """Buy on every bar (only first will fill since equity depletes)."""
    def on_bar(self, i, bar, equity):
        if equity > 10:
            return Trade(
                entry_bar=bar, side=1, size=equity * 0.5 / bar.close,
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

    def on_bar(self, i, bar, equity):
        if not self.entered and equity > 0:
            self.entered = True
            return Trade(
                entry_bar=bar, side=1,
                size=equity * self.size_pct / bar.close,
                entry_price=bar.close,
                stop_price=bar.close * (1 - self.stop_pct),
                take_profit=bar.close * (1 + self.tp_pct),
            )
        return None


class TestBacktesterBasics:
    def test_explicit_annualization_override(self, sample_df):
        bt = Backtester(
            sample_df, NeverTradeStrategy(), freq_per_year=365,
        )
        assert bt.freq_per_year == 365

    def test_nonpositive_annualization_override_rejected(self, sample_df):
        with pytest.raises(ValueError, match="freq_per_year must be > 0"):
            Backtester(sample_df, NeverTradeStrategy(), freq_per_year=0)

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

    def test_optional_terminal_liquidation_closes_open_position(self):
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.0, 101.0, 102.0],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )
        bt = Backtester(df, BuyOnceStrategy(stop_pct=0.5, tp_pct=0.5), starting_cash=10_000)

        equity, trades = bt.run(liquidate_at_end=True)

        assert len(trades) == 1
        assert trades[0].exit_ts == df.index[-1]
        assert len(bt.positions) == 0
        assert equity[-1] == pytest.approx(bt.cash)


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

    def test_closed_trade_pnl_includes_entry_and_exit_commission(self):
        data = {
            "open": [100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 111.0],
            "low": [99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 110.0],
        }
        df = pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=3, freq="h"))

        class CommissionStrategy(Strategy):
            def __init__(self):
                super().__init__()
                self.entered = False

            def on_bar(self, i, bar, equity):
                if self.entered:
                    return None
                self.entered = True
                return Trade(bar, 1, 1.0, 100.0, 90.0, 110.0)

        bt = Backtester(
            df, CommissionStrategy(), commission_bps=100.0, slippage_bps=10.0,
        )
        _, trades = bt.run()

        assert len(trades) == 1
        trade = trades[0]
        entry_commission = abs(trade.entry_price * trade.size) * 0.01
        exit_commission = abs(trade.exit_price * trade.size) * 0.01
        gross_pnl = (trade.exit_price - trade.entry_price) * trade.side * trade.size
        assert trade.pnl == pytest.approx(
            gross_pnl - entry_commission - exit_commission,
        )
        assert trades[0].pnl == pytest.approx(bt.cash - bt.starting_cash)


class TestExecutionTimeMarks:
    def test_buying_power_uses_open_not_the_unavailable_close(self):
        data = {
            "open": [100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 200.0],
            "low": [99.0, 99.0, 50.0],
            "close": [100.0, 100.0, 50.0],
        }
        df = pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=3, freq="D"))

        class TwoEntries(Strategy):
            def on_bar(self, i, bar, equity):
                if i == 0:
                    return Trade(bar, 1, 1.0, bar.close, 1.0, 1_000.0)
                if i == 1:
                    return Trade(bar, 1, 20.0, bar.close, 1.0, 1_000.0)
                return None

        bt = Backtester(
            df, TwoEntries(), starting_cash=1_000, max_leverage=1.5,
        )
        bt.run()

        # At the second fill, the open is 100. With the first lot marked at
        # that executable price, available notional is 1,400. The bar's later
        # close of 50 must not influence the fill size.
        assert bt.broker.positions["default"][1].size == pytest.approx(14.0)


class TestFundingTiming:
    def test_new_open_position_is_not_charged_for_prior_interval(self):
        df = pd.DataFrame(
            {
                "open": [100.0, 100.0, 100.0],
                "high": [101.0, 101.0, 101.0],
                "low": [99.0, 99.0, 99.0],
                "close": [100.0, 100.0, 100.0],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )
        config = BacktestConfig(
            starting_cash=10_000,
            funding_rate_annual=36_500.0,
        )

        bt = Backtester(
            df, BuyOnceStrategy(size_pct=0.01), config=config,
        )
        equity, _ = bt.run()

        # One unit at 100 is held only from day 1's open to day 2's open:
        # exactly one day of funding at 36,500% annualized is 100.
        assert equity[-1] == pytest.approx(9_900.0)


class TestGapAwareStops:
    def test_stop_fills_at_open_when_gap_through(self):
        """If price gaps below the stop, the fill should be at the open (worse), not the stop."""
        # Bar 0: signal long at close=100, stop at 95
        # Bar 1: entry fills at open=100; price holds
        # Bar 2: opens at 90 (gapped through stop) -> should fill at 90, not 95
        data = {
            "open": [100.0, 100.0, 90.0],
            "high": [101.0, 101.0, 91.0],
            "low": [99.0, 99.0, 89.0],
            "close": [100.0, 100.0, 90.0],
        }
        df = pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=3, freq="h"))

        class GapTestStrategy(Strategy):
            def __init__(self):
                self.entered = False
            def on_bar(self, i, bar, equity):
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
        # Bar 0: signal long at close=100, stop at 95
        # Bar 1: entry fills at open=100; price holds
        # Bar 2: opens at 98, low touches 94 -> stop at 95
        data = {
            "open": [100.0, 100.0, 98.0],
            "high": [101.0, 101.0, 99.0],
            "low": [99.0, 99.0, 94.0],
            "close": [100.0, 100.0, 95.0],
        }
        df = pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=3, freq="h"))

        class StopTestStrategy(Strategy):
            def __init__(self):
                self.entered = False
            def on_bar(self, i, bar, equity):
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

    @pytest.mark.parametrize(
        ("priority", "expected_exit", "limit_price"),
        [
            ("stop_first", 95.0, None),
            ("tp_first", 105.0, None),
            ("stop_first", 95.0, 100.0),
        ],
    )
    def test_entry_bar_stop_and_target_are_checked(
        self, priority, expected_exit, limit_price,
    ):
        """A position filled at the open is exposed to the rest of that bar."""
        data = {
            "open": [100.0, 100.0, 100.0],
            "high": [101.0, 106.0, 101.0],
            "low": [99.0, 94.0, 99.0],
            "close": [100.0, 100.0, 100.0],
        }
        df = pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=3, freq="h"))

        class EntryBarExitStrategy(Strategy):
            def __init__(self):
                super().__init__()
                self.entered = False

            def on_bar(self, i, bar, equity):
                if self.entered:
                    return None
                self.entered = True
                return Trade(
                    bar, 1, 1.0, 100.0, 95.0, 105.0,
                    limit_price=limit_price,
                )

        _, trades = Backtester(df, EntryBarExitStrategy()).run(
            execution_priority=priority,
        )

        assert len(trades) == 1
        assert trades[0].exit_ts == df.index[1]
        assert trades[0].exit_price == pytest.approx(expected_exit)


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


class TestTrailingStopTiming:
    def test_close_based_stop_update_is_not_checked_against_same_bar(self):
        data = {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 110.0, 103.0, 102.0],
            "low": [99.0, 99.0, 100.0, 99.0],
            "close": [100.0, 105.0, 101.0, 100.0],
        }
        df = pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=4, freq="h"))

        class TrailStrategy(Strategy):
            def __init__(self):
                super().__init__()
                self.entered = False

            def on_bar(self, i, bar, equity):
                if not self.entered:
                    self.entered = True
                    return Trade(bar, 1, 1.0, 100.0, 90.0, 120.0)
                return None

            def manage_position(self, bar, trade):
                trade.stop_price = 104.0

        _, trades = Backtester(df, TrailStrategy(), starting_cash=10_000).run()

        # The new 104 stop is set after bar 2 has been checked, so its low of
        # 100 must not retrospectively stop out the position. It is active on
        # bar 3, where the open gaps through it.
        assert len(trades) == 1
        assert trades[0].exit_ts == df.index[3]
        assert trades[0].exit_price == 100.0


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


class TestLimitPriceImprovement:
    def test_buy_limit_fills_at_open_when_gapped_through(self):
        data = {
            "open": [100.0, 98.0, 98.0],
            "high": [101.0, 99.0, 99.0],
            "low": [99.0, 97.0, 97.0],
            "close": [100.0, 98.0, 98.0],
        }
        df = pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=3, freq="D"))

        class LimitBuy(Strategy):
            def __init__(self):
                super().__init__()
                self.entered = False

            def on_bar(self, i, bar, equity):
                if self.entered:
                    return None
                self.entered = True
                return Trade(bar, 1, 1.0, 100.0, 10.0, 1_000.0, limit_price=100.0)

        _, trades = Backtester(df, LimitBuy()).run(liquidate_at_end=True)

        # Bar 1 opens at 98, through the 100 buy limit, so the fill receives
        # price improvement at the open rather than the limit price.
        assert len(trades) == 1
        assert trades[0].entry_price == pytest.approx(98.0)


class TestSingleUse:
    def test_backtester_rejects_rerun(self, flat_df):
        bt = Backtester(flat_df, NeverTradeStrategy(), starting_cash=10_000)
        bt.run()
        with pytest.raises(RuntimeError, match="single-use"):
            bt.run()


class TestContractMultiplier:
    def test_multiplier_scales_mark_to_market_equity(self):
        """Inline equity must scale P&L by the contract multiplier, matching
        broker/portfolio accounting rather than silently assuming 1.0."""
        data = {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 110.0, 101.0],
            "low": [99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 105.0, 100.0],
        }
        df = pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=4, freq="h"))

        class EnterOnce(Strategy):
            def __init__(self):
                super().__init__()
                self.entered = False

            def on_bar(self, i, bar, equity):
                if self.entered:
                    return None
                self.entered = True
                return Trade(bar, 1, 2.0, bar.close, 1.0, 1_000.0)

        bt = Backtester(
            df, EnterOnce(), starting_cash=10_000,
            contract_multipliers={"default": 10},
        )
        equity, _ = bt.run()

        # Filled at bar 1 open (100); marked at bar 2 close (105),
        # size 2 * multiplier 10 => +100 of unrealized P&L.
        assert equity[1] == pytest.approx(10_000)
        assert equity[2] == pytest.approx(10_000 + (105.0 - 100.0) * 2 * 10)


class TestTimestampColumn:
    def test_range_index_with_timestamp_column_is_used(self):
        """A 'timestamp' column must drive event timing even when the frame
        has a plain RangeIndex (previously coerced into epoch nanoseconds)."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
            "open": [100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0],
        })

        bt = Backtester(df, NeverTradeStrategy(), starting_cash=10_000)
        equity, _ = bt.run()

        assert len(equity) == 3
        assert bt._ts[0] == pd.Timestamp("2024-01-01 00:00:00")
        assert bt.freq_per_year > 1
