#!/usr/bin/env python3
"""Reproducible benchmark: event-driven vs vectorized vs Cython engines.

Generates synthetic OHLC data and times each engine on identical workloads.
Reports bars/sec for direct comparison.

Usage:
    python benchmarks/bench_engines.py
    python benchmarks/bench_engines.py --bars 50000 --trials 100
"""

import argparse
import time
import sys
import os

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.backtest import Backtester
from backtesting.vectorized import VectorizedBacktester, _HAS_CYTHON
from backtesting.vectorized_signals import momentum_signals
from strategies import MomentumStrategy


def make_benchmark_data(n_bars: int, seed: int = 42):
    """Generate synthetic trending OHLC data."""
    rng = np.random.RandomState(seed)
    returns = 0.0005 + rng.randn(n_bars) * 0.01
    prices = 100.0 * np.cumprod(1 + returns)

    close = prices
    open_ = close * (1 + rng.randn(n_bars) * 0.001)
    high = np.maximum(open_, close) * (1 + rng.rand(n_bars) * 0.003)
    low = np.minimum(open_, close) * (1 - rng.rand(n_bars) * 0.003)

    index = pd.date_range("2010-01-01", periods=n_bars, freq="D")
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
    }, index=index)
    return df, open_, high, low, close


def bench_event_driven(df, n_runs=3):
    """Benchmark the event-driven engine."""
    params = {
        "lookback": 20, "entry_threshold": 0.03, "atr_period": 14,
        "atr_stop_mult": 2.0, "atr_target_mult": 3.0, "risk_per_trade": 0.02,
        "max_dd_halt": 0.15, "cooldown_bars": 10,
    }

    times = []
    n_trades = 0
    for _ in range(n_runs):
        strategy = MomentumStrategy(**params)
        bt = Backtester(df, strategy, starting_cash=10_000)

        t0 = time.perf_counter()
        _, trades = bt.run()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        n_trades = len(trades)

    best = min(times)
    return best, n_trades


def bench_vectorized(o, h, lo, c, n_runs=3):
    """Benchmark the vectorized engine (uses Cython if compiled)."""
    signal_params = {
        "lookback": 20, "entry_threshold": 0.03, "atr_period": 14,
        "atr_stop_mult": 2.0, "atr_target_mult": 3.0,
    }

    # Pre-compute signals (not included in timing — same for both engines)
    entries, sides, stops, tps = momentum_signals(o, h, lo, c, **signal_params)

    times = []
    n_trades = 0
    for _ in range(n_runs):
        bt = VectorizedBacktester(o, h, lo, c, starting_cash=10_000)

        t0 = time.perf_counter()
        _, trades = bt.run(entries, sides, stops, tps,
                           risk_per_trade=0.02, max_dd_halt=0.15, cooldown_bars=10)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        n_trades = len(trades)

    best = min(times)
    return best, n_trades


def bench_optimizer_trial(df, o, h, lo, c, n_trials=50):
    """Benchmark a realistic optimizer workload: N trials with vectorized engine."""
    from optimizer import optimize

    t0 = time.perf_counter()
    result = optimize(
        MomentumStrategy,
        param_space={"lookback": (5, 40), "entry_threshold": (0.01, 0.06)},
        df=df, n_trials=n_trials, objective="sharpe",
        engine="vectorized",
        fixed_params={
            "atr_period": 14, "atr_stop_mult": 2.0, "atr_target_mult": 3.0,
            "risk_per_trade": 0.02, "max_dd_halt": 0.15, "cooldown_bars": 10,
        },
    )
    elapsed = time.perf_counter() - t0
    return elapsed, result.best_score


def main():
    parser = argparse.ArgumentParser(description="Benchmark backtesting engines")
    parser.add_argument("--bars", type=int, default=10_000, help="Number of bars (default: 10000)")
    parser.add_argument("--runs", type=int, default=5, help="Runs per engine (default: 5)")
    parser.add_argument("--trials", type=int, default=100, help="Optimizer trials (default: 100)")
    args = parser.parse_args()

    n_bars = args.bars
    n_runs = args.runs
    n_trials = args.trials

    print(f"Benchmark: {n_bars:,} bars, {n_runs} runs per engine, {n_trials} optimizer trials")
    print(f"Cython compiled: {'yes' if _HAS_CYTHON else 'no'}")
    print("-" * 60)

    df, o, h, lo, c = make_benchmark_data(n_bars)

    # --- Event-driven ---
    ev_time, ev_trades = bench_event_driven(df, n_runs)
    ev_bars_sec = n_bars / ev_time
    print(f"Event-driven:  {ev_time:.4f}s  ({ev_bars_sec:,.0f} bars/sec)  {ev_trades} trades")

    # --- Vectorized ---
    vec_time, vec_trades = bench_vectorized(o, h, lo, c, n_runs)
    vec_bars_sec = n_bars / vec_time
    cython_label = " (Cython)" if _HAS_CYTHON else " (Python)"
    print(f"Vectorized{cython_label}: {vec_time:.4f}s  ({vec_bars_sec:,.0f} bars/sec)  {vec_trades} trades")

    # --- Speedup ---
    speedup = ev_time / vec_time if vec_time > 0 else float("inf")
    print(f"Speedup:       {speedup:.1f}x")

    # --- Optimizer ---
    print(f"\nOptimizer ({n_trials} trials, vectorized engine):")
    opt_time, opt_score = bench_optimizer_trial(df, o, h, lo, c, n_trials)
    total_bars = n_bars * n_trials
    print(f"  Total time:  {opt_time:.2f}s")
    print(f"  Total bars:  {total_bars:,} ({total_bars / opt_time:,.0f} bars/sec effective)")
    print(f"  Best score:  {opt_score:.4f}")

    print("\n" + "=" * 60)
    print("Hardware: run `python -c \"import platform; print(platform.processor())\"` for CPU info")
    print("Note: times are best-of-N to reduce variance from OS scheduling.")


if __name__ == "__main__":
    main()
