"""Benchmark tick backtester performance.

Generates synthetic tick data and measures throughput (ticks/sec).
Run: python benchmarks/bench_tick.py
"""

import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from backtesting.indicators import EMA, ATR
from backtesting.strategy import Strategy
from backtesting.tick import Tick, TickAggregator
from backtesting.tick_backtest import TickBacktester, _HAS_CYTHON_TICK
from backtesting.types import Bar, Trade


class BenchStrategy(Strategy):
    """Realistic strategy for benchmarking — dual EMA crossover on bars."""

    def __init__(self):
        self._fast = EMA(10)
        self._slow = EMA(30)
        self._atr = ATR(14)
        self._prev_fast = None
        self._prev_slow = None
        self._has_position = False

    def on_bar(self, i, bar, equity):
        fast = self._fast.update(bar.close)
        slow = self._slow.update(bar.close)
        atr = self._atr.update(bar.high, bar.low, bar.close)

        if fast is None or slow is None or atr is None or atr <= 0:
            self._prev_fast, self._prev_slow = fast, slow
            return None

        if self._prev_fast is None or self._prev_slow is None:
            self._prev_fast, self._prev_slow = fast, slow
            return None

        if self._has_position:
            self._prev_fast, self._prev_slow = fast, slow
            return None

        trade = None
        if self._prev_fast <= self._prev_slow and fast > slow:
            stop = bar.close - 2 * atr
            tp = bar.close + 4 * atr
            size = equity * 0.02 / (2 * atr)
            if size > 0:
                trade = Trade(entry_bar=bar, side=1, size=size,
                              entry_price=bar.close, stop_price=stop, take_profit=tp)
                self._has_position = True

        self._prev_fast, self._prev_slow = fast, slow
        return trade

    def manage_position(self, bar, trade):
        self._has_position = True


class TickLevelBenchStrategy(BenchStrategy):
    """Same as BenchStrategy but also has on_tick() override."""

    def on_tick(self, tick, current_bar, equity):
        # Simulate some work per tick (e.g., micro-level checks)
        _ = tick.price * 1.001
        return None

    def manage_position_tick(self, tick, trade):
        # Simulate trailing stop check per tick
        if trade.side > 0 and tick.price > trade.entry_price * 1.02:
            new_stop = tick.price * 0.99
            if new_stop > trade.stop_price:
                trade.stop_price = new_stop


def generate_ticks(n_ticks: int, ticks_per_bar: int = 100) -> list:
    """Generate synthetic tick data with realistic-ish price movement."""
    rng = np.random.default_rng(42)
    base = pd.Timestamp("2024-01-01 09:00:00")

    # Random walk with drift
    returns = rng.normal(0.0001, 0.001, n_ticks)
    prices = 2000.0 * np.exp(np.cumsum(returns))

    # Timestamps at ~100ms intervals
    timestamps = [base + pd.Timedelta(milliseconds=i * 100) for i in range(n_ticks)]

    return [Tick(ts=timestamps[i], price=prices[i], volume=rng.uniform(0.1, 5.0))
            for i in range(n_ticks)]


def run_benchmark(n_ticks: int, label: str, strategy_cls, timeframe: str = "M1"):
    """Run a single benchmark and report results."""
    ticks = generate_ticks(n_ticks)

    strategy = strategy_cls()
    bt = TickBacktester(ticks, strategy, timeframe=timeframe, starting_cash=100_000)

    start = time.perf_counter()
    equity_curve, trades = bt.run()
    elapsed = time.perf_counter() - start

    rate = n_ticks / elapsed
    print(f"  {label:40s}  {n_ticks:>10,} ticks  {elapsed:6.3f}s  "
          f"{rate:>12,.0f} ticks/sec  {len(trades):>4} trades  {len(bt.bars):>6} bars")
    return elapsed, rate


def main():
    print("=" * 100)
    print("Tick Backtester Benchmark")
    print(f"Cython tick extensions: {'YES' if _HAS_CYTHON_TICK else 'NO (pure Python)'}")
    print("=" * 100)

    sizes = [10_000, 100_000, 500_000, 1_000_000]

    print("\n--- Bar-only strategy (on_tick not overridden -> skipped in loop) ---")
    for n in sizes:
        run_benchmark(n, f"bar_only_{n // 1000}k", BenchStrategy)

    print("\n--- Tick-level strategy (on_tick + manage_position_tick active) ---")
    for n in sizes:
        run_benchmark(n, f"tick_level_{n // 1000}k", TickLevelBenchStrategy)

    print("\n--- Timeframe comparison (bar-only, 500k ticks) ---")
    for tf in ["M1", "M5", "M15", "H1"]:
        run_benchmark(500_000, f"tf={tf}", BenchStrategy, timeframe=tf)

    # Aggregator-only benchmark
    print("\n--- TickAggregator standalone (no strategy, pure aggregation) ---")
    for n in sizes:
        ticks = generate_ticks(n)
        agg = TickAggregator("M1")
        start = time.perf_counter()
        n_bars = 0
        for tick in ticks:
            if agg.update(tick) is not None:
                n_bars += 1
        elapsed = time.perf_counter() - start
        rate = n / elapsed
        print(f"  aggregator_{n // 1000}k                           "
              f"  {n:>10,} ticks  {elapsed:6.3f}s  {rate:>12,.0f} ticks/sec  {n_bars:>6} bars")


if __name__ == "__main__":
    main()
