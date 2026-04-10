"""Tests for tick data types, aggregation, validation, and tick-level backtesting."""

import numpy as np
import pandas as pd
import pytest

from backtesting.data import validate_ticks
from backtesting.strategy import Strategy
from backtesting.tick import Tick, TickAggregator
from backtesting.tick_backtest import TickBacktester
from backtesting.types import Bar, BacktestConfig, Trade


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ticks(prices, start="2024-01-01 09:00:00", freq_ms=100):
    """Generate ticks at regular millisecond intervals."""
    base = pd.Timestamp(start)
    return [
        Tick(ts=base + pd.Timedelta(milliseconds=i * freq_ms), price=p)
        for i, p in enumerate(prices)
    ]


def _make_minute_ticks(minute_prices, start="2024-01-01 09:00:00"):
    """Generate ticks spread across minute boundaries.

    minute_prices: list of lists — each inner list is ticks within one minute.
    """
    ticks = []
    base = pd.Timestamp(start)
    for m_idx, prices in enumerate(minute_prices):
        minute_start = base + pd.Timedelta(minutes=m_idx)
        for t_idx, p in enumerate(prices):
            # Space ticks evenly within the minute
            offset_ms = int((t_idx / max(len(prices), 1)) * 59_000)
            ticks.append(Tick(
                ts=minute_start + pd.Timedelta(milliseconds=offset_ms),
                price=p,
            ))
    return ticks


@pytest.fixture
def uptrend_ticks():
    """Ticks forming 5 one-minute bars with a clear uptrend."""
    return _make_minute_ticks([
        [100.0, 100.5, 101.0, 100.8],  # min 0: O=100, H=101, L=100, C=100.8
        [101.0, 101.5, 102.0, 101.8],  # min 1
        [102.0, 102.5, 103.0, 102.8],  # min 2
        [103.0, 103.5, 104.0, 103.8],  # min 3
        [104.0, 104.5, 105.0, 104.8],  # min 4
    ])


# ---------------------------------------------------------------------------
# TickAggregator tests
# ---------------------------------------------------------------------------

class TestTickAggregator:
    def test_first_tick_no_bar(self):
        agg = TickAggregator("1min")
        tick = Tick(ts=pd.Timestamp("2024-01-01 09:00:30"), price=100.0)
        bar = agg.update(tick)
        assert bar is None
        assert agg.current_bar is not None
        assert agg.current_bar.open == 100.0

    def test_same_minute_updates_ohlc(self):
        agg = TickAggregator("1min")
        ticks = [
            Tick(ts=pd.Timestamp("2024-01-01 09:00:00"), price=100.0),
            Tick(ts=pd.Timestamp("2024-01-01 09:00:10"), price=102.0),
            Tick(ts=pd.Timestamp("2024-01-01 09:00:20"), price=99.0),
            Tick(ts=pd.Timestamp("2024-01-01 09:00:30"), price=101.0),
        ]
        for t in ticks:
            agg.update(t)

        bar = agg.current_bar
        assert bar.open == 100.0
        assert bar.high == 102.0
        assert bar.low == 99.0
        assert bar.close == 101.0

    def test_boundary_crossing_emits_bar(self):
        agg = TickAggregator("1min")
        # First minute
        agg.update(Tick(ts=pd.Timestamp("2024-01-01 09:00:00"), price=100.0))
        agg.update(Tick(ts=pd.Timestamp("2024-01-01 09:00:30"), price=101.0))

        # Cross into second minute — should emit the first bar
        bar = agg.update(Tick(ts=pd.Timestamp("2024-01-01 09:01:00"), price=102.0))

        assert bar is not None
        assert bar.ts == pd.Timestamp("2024-01-01 09:00:00")
        assert bar.open == 100.0
        assert bar.close == 101.0

        # New bar started
        assert agg.current_bar.open == 102.0

    def test_flush_emits_partial_bar(self):
        agg = TickAggregator("1min")
        agg.update(Tick(ts=pd.Timestamp("2024-01-01 09:00:15"), price=100.0))
        agg.update(Tick(ts=pd.Timestamp("2024-01-01 09:00:45"), price=101.0))

        bar = agg.flush()
        assert bar is not None
        assert bar.open == 100.0
        assert bar.close == 101.0
        assert agg.current_bar is None

    def test_flush_empty_returns_none(self):
        agg = TickAggregator("1min")
        assert agg.flush() is None

    def test_5min_aggregation(self):
        agg = TickAggregator("M5")
        bars = []
        # 10 minutes of ticks → should produce 2 five-minute bars
        for minute in range(10):
            ts = pd.Timestamp("2024-01-01 09:00:00") + pd.Timedelta(minutes=minute)
            bar = agg.update(Tick(ts=ts, price=100.0 + minute))
            if bar is not None:
                bars.append(bar)

        assert len(bars) == 1  # 09:00-09:04 emitted when 09:05 tick arrives
        assert bars[0].ts == pd.Timestamp("2024-01-01 09:00:00")
        assert bars[0].open == 100.0
        assert bars[0].close == 104.0  # Last tick in first 5-min window

        # Flush the second partial bar
        final = agg.flush()
        assert final is not None
        assert final.open == 105.0

    def test_volume_accumulated(self):
        agg = TickAggregator("1min")
        agg.update(Tick(ts=pd.Timestamp("2024-01-01 09:00:00"), price=100.0, volume=1.5))
        agg.update(Tick(ts=pd.Timestamp("2024-01-01 09:00:30"), price=101.0, volume=2.5))

        assert agg.current_bar.volume == 4.0

    def test_tick_count(self):
        agg = TickAggregator("1min")
        agg.update(Tick(ts=pd.Timestamp("2024-01-01 09:00:00"), price=100.0))
        agg.update(Tick(ts=pd.Timestamp("2024-01-01 09:00:10"), price=101.0))
        agg.update(Tick(ts=pd.Timestamp("2024-01-01 09:00:20"), price=102.0))

        assert agg.tick_count == 3


# ---------------------------------------------------------------------------
# Tick validation tests
# ---------------------------------------------------------------------------

class TestValidateTicks:
    def test_valid_ticks(self):
        ticks = _make_ticks([100.0, 101.0, 102.0])
        report = validate_ticks(ticks)
        assert report.is_valid
        assert report.n_bars == 3

    def test_empty_list(self):
        report = validate_ticks([])
        assert not report.is_valid

    def test_nan_price(self):
        ticks = _make_ticks([100.0, float("nan"), 102.0])
        report = validate_ticks(ticks)
        assert not report.is_valid
        assert any("NaN" in iss.message for iss in report.issues)

    def test_inf_price(self):
        ticks = _make_ticks([100.0, float("inf"), 102.0])
        report = validate_ticks(ticks)
        assert not report.is_valid

    def test_negative_price_warning(self):
        ticks = _make_ticks([100.0, -1.0, 102.0])
        report = validate_ticks(ticks)
        # Negative price is a warning, not error
        assert report.is_valid
        assert report.n_warnings > 0

    def test_non_monotonic_timestamps(self):
        ticks = [
            Tick(ts=pd.Timestamp("2024-01-01 09:00:00"), price=100.0),
            Tick(ts=pd.Timestamp("2024-01-01 09:00:02"), price=101.0),
            Tick(ts=pd.Timestamp("2024-01-01 09:00:01"), price=102.0),  # Out of order
        ]
        report = validate_ticks(ticks)
        assert not report.is_valid

    def test_duplicate_timestamps_warning(self):
        ts = pd.Timestamp("2024-01-01 09:00:00")
        ticks = [
            Tick(ts=ts, price=100.0),
            Tick(ts=ts, price=100.1),
        ]
        report = validate_ticks(ticks)
        assert report.is_valid  # Warning, not error
        assert report.n_warnings > 0


# ---------------------------------------------------------------------------
# TickBacktester tests
# ---------------------------------------------------------------------------

class _AlwaysBuyOnBar(Strategy):
    """Test strategy: buys on every bar completion if no position."""

    def __init__(self):
        self._has_position = False

    def on_bar(self, i, bar, equity):
        if self._has_position:
            return None
        self._has_position = True
        size = equity * 0.1 / bar.close
        return Trade(
            entry_bar=bar, side=1, size=size,
            entry_price=bar.close,
            stop_price=bar.close * 0.95,
            take_profit=bar.close * 1.10,
        )

    def manage_position(self, bar, trade):
        self._has_position = True


class _TickLevelStrategy(Strategy):
    """Test strategy: uses on_tick() to enter when price crosses a threshold."""

    def __init__(self, threshold=102.0):
        self.threshold = threshold
        self._triggered = False

    def on_bar(self, i, bar, equity):
        return None  # Does nothing on bar — all logic is in on_tick

    def on_tick(self, tick, current_bar, equity):
        if self._triggered:
            return None
        if tick.price >= self.threshold:
            self._triggered = True
            size = equity * 0.1 / tick.price
            # Build a synthetic bar for Trade.entry_bar
            bar = Bar(tick.ts, tick.price, tick.price, tick.price, tick.price)
            return Trade(
                entry_bar=bar, side=1, size=size,
                entry_price=tick.price,
                stop_price=tick.price * 0.95,
                take_profit=tick.price * 1.10,
            )
        return None


class TestTickBacktester:
    def test_basic_run(self, uptrend_ticks):
        strategy = _AlwaysBuyOnBar()
        bt = TickBacktester(
            uptrend_ticks, strategy, timeframe="1min", starting_cash=10_000,
        )
        equity_curve, trades = bt.run()

        assert len(equity_curve) == len(uptrend_ticks)
        assert equity_curve[0] > 0
        # Should have completed bars
        assert len(bt.bars) > 0

    def test_empty_ticks(self):
        strategy = _AlwaysBuyOnBar()
        bt = TickBacktester([], strategy, timeframe="1min")
        equity_curve, trades = bt.run()
        assert len(equity_curve) == 0
        assert len(trades) == 0

    def test_equity_starts_at_cash(self, uptrend_ticks):
        strategy = _AlwaysBuyOnBar()
        bt = TickBacktester(
            uptrend_ticks, strategy, timeframe="1min", starting_cash=50_000,
        )
        equity_curve, _ = bt.run()
        assert equity_curve[0] == pytest.approx(50_000, rel=0.01)

    def test_tick_level_entry(self, uptrend_ticks):
        """Strategy enters via on_tick(), not on_bar()."""
        strategy = _TickLevelStrategy(threshold=101.5)
        bt = TickBacktester(
            uptrend_ticks, strategy, timeframe="1min", starting_cash=10_000,
        )
        equity_curve, trades = bt.run()

        # Should have opened a position (via on_tick) that eventually
        # either hits TP or remains open
        assert len(bt.broker.closed_trades) + len(bt.positions) >= 1

    def test_stop_loss_at_tick_price(self):
        """Stop triggers at exact tick price, not bar boundary."""
        ticks = [
            Tick(ts=pd.Timestamp("2024-01-01 09:00:00"), price=100.0),
            Tick(ts=pd.Timestamp("2024-01-01 09:00:01"), price=101.0),
            # Bar completes, strategy enters long at ~101
            Tick(ts=pd.Timestamp("2024-01-01 09:01:00"), price=101.0),
            # Fill happens here at 101.0
            Tick(ts=pd.Timestamp("2024-01-01 09:01:01"), price=100.0),
            Tick(ts=pd.Timestamp("2024-01-01 09:01:02"), price=99.0),
            # Stop at 95% of 101 = 95.95 — not triggered yet
            Tick(ts=pd.Timestamp("2024-01-01 09:01:03"), price=95.0),
            # This tick should trigger the stop
            Tick(ts=pd.Timestamp("2024-01-01 09:01:04"), price=94.0),
        ]

        strategy = _AlwaysBuyOnBar()
        bt = TickBacktester(ticks, strategy, timeframe="1min", starting_cash=10_000)
        _, trades = bt.run()

        assert len(trades) == 1
        # Stop was at entry * 0.95 = ~95.95, tick that breached was 95.0
        assert trades[0].pnl < 0  # It's a loss

    def test_take_profit_at_tick_price(self):
        """TP triggers at exact tick price."""
        ticks = [
            Tick(ts=pd.Timestamp("2024-01-01 09:00:00"), price=100.0),
            Tick(ts=pd.Timestamp("2024-01-01 09:00:30"), price=100.5),
            # Bar completes at minute boundary
            Tick(ts=pd.Timestamp("2024-01-01 09:01:00"), price=100.0),
            # Fill at 100.0, TP at 110.0
            Tick(ts=pd.Timestamp("2024-01-01 09:01:10"), price=105.0),
            Tick(ts=pd.Timestamp("2024-01-01 09:01:20"), price=110.0),
            # TP hit
            Tick(ts=pd.Timestamp("2024-01-01 09:01:30"), price=112.0),
        ]

        strategy = _AlwaysBuyOnBar()
        bt = TickBacktester(ticks, strategy, timeframe="1min", starting_cash=10_000)
        _, trades = bt.run()

        assert len(trades) == 1
        assert trades[0].pnl > 0

    def test_bars_match_aggregator(self, uptrend_ticks):
        """Completed bars from backtester should match standalone aggregation."""
        strategy = _AlwaysBuyOnBar()
        bt = TickBacktester(
            uptrend_ticks, strategy, timeframe="1min", starting_cash=10_000,
        )
        bt.run()

        # Standalone aggregation
        agg = TickAggregator("1min")
        standalone_bars = []
        for tick in uptrend_ticks:
            bar = agg.update(tick)
            if bar is not None:
                standalone_bars.append(bar)
        last = agg.flush()
        if last:
            standalone_bars.append(last)

        assert len(bt.bars) == len(standalone_bars)
        for a, b in zip(bt.bars, standalone_bars):
            assert a.open == b.open
            assert a.high == b.high
            assert a.low == b.low
            assert a.close == b.close

    def test_config_object(self, uptrend_ticks):
        config = BacktestConfig(
            starting_cash=25_000, commission_bps=5.0, slippage_bps=2.0,
        )
        bt = TickBacktester(
            uptrend_ticks, _AlwaysBuyOnBar(), timeframe="1min", config=config,
        )
        equity_curve, _ = bt.run()
        assert equity_curve[0] == pytest.approx(25_000, rel=0.01)

    def test_no_lookahead_on_bar(self):
        """on_bar signal fills at the NEXT tick, not at the bar's close."""
        ticks = [
            Tick(ts=pd.Timestamp("2024-01-01 09:00:00"), price=100.0),
            Tick(ts=pd.Timestamp("2024-01-01 09:00:30"), price=100.0),
            # Bar 0 completes, signal generated
            Tick(ts=pd.Timestamp("2024-01-01 09:01:00"), price=105.0),
            # Fill should happen HERE at 105, not back at 100
            Tick(ts=pd.Timestamp("2024-01-01 09:01:30"), price=106.0),
        ]

        strategy = _AlwaysBuyOnBar()
        bt = TickBacktester(ticks, strategy, timeframe="1min", starting_cash=10_000)
        bt.run()

        # Check that the position was entered at the tick price after the bar
        open_trades = bt.positions
        closed_trades = bt.trades
        all_trades = open_trades + closed_trades
        assert len(all_trades) >= 1
        # Entry should be around 105 (the next tick after bar completion),
        # not 100 (the bar's close). Slippage may shift it slightly.
        assert all_trades[0].entry_price == pytest.approx(105.0, rel=0.01)
