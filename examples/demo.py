#!/usr/bin/env python3
"""
Backtesting Engine — End-to-End Demo

Run this to see the full workflow:
  1. Load OHLC data
  2. Backtest all four strategies
  3. Optimize one strategy's parameters
  4. Walk-forward validate to check for overfitting
  5. Compare event-driven vs vectorized performance

Usage:
    python examples/demo.py

Requires OHLC data in ohlc_data/XAUUSD_D1.csv (or any CSV with
timestamp, open, high, low, close columns).
"""

import os
import sys
import time

import numpy as np
import pandas as pd

# Add project root to path so imports work from examples/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.backtest import Backtester
from backtesting.vectorized import VectorizedBacktester
from backtesting.vectorized_signals import trend_following_signals
from optimizer import optimize, walk_forward
from strategies import (
    DonchianBreakoutStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    TrendFollowingStrategy,
)
from ai_analyst import analyze_backtest
from utils import compute_sharpe


def load_data(path="ohlc_data/XAUUSD_D1.csv"):
    """Load OHLC data from CSV."""
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.")
        print("Place an OHLC CSV with columns: timestamp, open, high, low, close")
        sys.exit(1)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df


def section(title):
    print()
    print(f"{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print()


def main():
    df = load_data()
    print(f"Loaded {len(df)} bars: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Instrument: XAUUSD D1")
    print(f"Price range: {df['close'].min():.0f} - {df['close'].max():.0f}")

    # ------------------------------------------------------------------
    section("1. Backtest All Strategies")
    # ------------------------------------------------------------------

    strategies = {
        "Trend Following": TrendFollowingStrategy(),
        "Mean Reversion": MeanReversionStrategy(),
        "Momentum": MomentumStrategy(),
        "Donchian Breakout": DonchianBreakoutStrategy(),
    }

    print(f"{'Strategy':<20s} {'Return':>8s} {'Max DD':>8s} {'Sharpe':>8s} {'Trades':>7s} {'Win %':>7s}")
    print("-" * 60)

    analyst_metrics = {}

    for name, strat in strategies.items():
        bt = Backtester(df, strat, starting_cash=10_000,
                        commission_bps=5.0, slippage_bps=2.0, symbol="XAUUSD")
        eq, trades = bt.run()

        wins = len([t for t in trades if t.pnl and t.pnl > 0])
        win_rate = wins / len(trades) * 100 if trades else 0
        ret = (eq[-1] - 10_000) / 10_000 * 100
        sharpe = compute_sharpe(eq)

        analyst_metrics[name] = {
            "pct_return": ret, "sharpe": sharpe,
            "max_drawdown": bt.max_drawdown * 100,
            "total_trades": len(trades), "win_rate": win_rate,
        }

        print(f"{name:<20s} {ret:>+7.2f}% {bt.max_drawdown*100:>7.2f}% {sharpe:>8.4f} {len(trades):>7d} {win_rate:>6.1f}%")

    # ------------------------------------------------------------------
    section("2. Parameter Optimization (Trend Following)")
    # ------------------------------------------------------------------

    param_space = {
        "fast_period": (5, 40),
        "slow_period": (20, 100),
        "atr_stop_mult": (1.0, 5.0),
        "risk_per_trade": (0.01, 0.05),
    }
    fixed_params = {"use_trailing_stop": True, "allow_reentry": True}

    print(f"Running 50 Optuna trials (Bayesian search)...")
    t0 = time.perf_counter()
    result = optimize(
        TrendFollowingStrategy, param_space, df,
        n_trials=50, objective="sharpe",
        commission_bps=5.0, slippage_bps=2.0,
        fixed_params=fixed_params,
        min_trades=10, top_k_avg=5,
    )
    elapsed = time.perf_counter() - t0

    print(f"Completed in {elapsed:.1f}s")
    print(f"Best Sharpe: {result.best_score:.4f}")
    print(f"Best params:")
    for k, v in result.best_params.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    analyst_metrics["Trend Following"]["optimization"] = {
        "best_sharpe": result.best_score, "n_trials": 50,
    }

    print(f"\nTop 5 trials:")
    print(result.all_trials.head(5).to_string(index=False))

    # ------------------------------------------------------------------
    section("3. Walk-Forward Validation (the overfitting test)")
    # ------------------------------------------------------------------

    print("Splitting data into 3 train/test windows...")
    print("For each window: optimize on train, evaluate on test (unseen data)")
    print()

    wf = walk_forward(
        TrendFollowingStrategy, param_space, df,
        n_splits=3, train_ratio=0.7, n_trials=30,
        objective="sharpe", commission_bps=5.0, slippage_bps=2.0,
        fixed_params=fixed_params,
        min_trades=10, top_k_avg=5, anchored=True,
    )

    print(wf.summary.to_string(index=False))
    print()
    print(f"In-sample mean Sharpe:      {wf.in_sample_mean:>8.4f}  (optimized — always looks good)")
    print(f"Out-of-sample mean Sharpe:  {wf.out_of_sample_mean:>8.4f}  (the honest number)")
    print(f"Degradation:                {wf.degradation:>8.4f}  (IS - OOS, measures overfitting)")
    print()
    analyst_metrics["Trend Following"]["walk_forward"] = {
        "is_mean": wf.in_sample_mean, "oos_mean": wf.out_of_sample_mean,
        "degradation": wf.degradation,
    }

    if wf.degradation > 0.5:
        print("Verdict: Large degradation — the optimizer is curve-fitting, not finding real signal.")
    elif wf.out_of_sample_mean > 0:
        print("Verdict: Positive OOS Sharpe — there may be a real edge here.")
    else:
        print("Verdict: Negative OOS Sharpe — no reliable edge detected.")

    # ------------------------------------------------------------------
    section("4. Statistical Significance")
    # ------------------------------------------------------------------

    from backtesting.statistics import compute_statistical_report

    # Run the best strategy and check if the edge is real
    bt = Backtester(df, TrendFollowingStrategy(
        use_trailing_stop=True, allow_reentry=True), starting_cash=10_000,
                    commission_bps=5.0, slippage_bps=2.0, symbol="XAUUSD")
    eq, trades = bt.run()

    report = compute_statistical_report(eq, trades, n_trials_tested=50, seed=42)
    print(report)

    # ------------------------------------------------------------------
    section("5. Results Database")
    # ------------------------------------------------------------------

    from results_db import ResultsDB

    db_path = "exports/demo_results.db"
    os.makedirs("exports", exist_ok=True)
    with ResultsDB(db_path) as db:
        for name, strat_cls, kwargs in [
            ("Trend Following", TrendFollowingStrategy,
             {"use_trailing_stop": True, "allow_reentry": True}),
            ("Momentum", MomentumStrategy, {}),
        ]:
            strat = strat_cls(**kwargs)
            bt = Backtester(df, strat, starting_cash=10_000,
                            commission_bps=5.0, slippage_bps=2.0)
            eq, trades = bt.run()
            db.save_run(name, {}, eq, trades, starting_cash=10_000,
                        commission_bps=5.0, slippage_bps=2.0)

        print(f"Saved runs to {db_path}")
        print()
        print("Query: all runs ordered by Sharpe:")
        print(db.query_runs()[["strategy_name", "sharpe", "pct_return", "max_drawdown", "total_trades"]].to_string(index=False))

    # ------------------------------------------------------------------
    section("6. Performance: Event-Driven vs Vectorized")
    # ------------------------------------------------------------------

    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    lo = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)

    # Event-driven
    t0 = time.perf_counter()
    for _ in range(200):
        Backtester(df, TrendFollowingStrategy(
            use_trailing_stop=True, allow_reentry=True), starting_cash=10_000).run()
    evt_time = (time.perf_counter() - t0) / 200
    evt_bps = len(df) / evt_time

    # Vectorized
    t0 = time.perf_counter()
    for _ in range(1000):
        entries, sides, stops, tps = trend_following_signals(o, h, lo, c)
        VectorizedBacktester(o, h, lo, c, starting_cash=10_000).run(
            entries, sides, stops, tps, risk_per_trade=0.02, cooldown_bars=5)
    vec_time = (time.perf_counter() - t0) / 1000
    vec_bps = len(df) / vec_time

    print(f"Event-driven:  {evt_bps:>10,.0f} bars/sec  ({evt_time*1000:.1f} ms/run)")
    print(f"Vectorized:    {vec_bps:>10,.0f} bars/sec  ({vec_time*1000:.1f} ms/run)")
    print(f"Speedup:       {vec_bps/evt_bps:.1f}x")
    print()
    print(f"At vectorized speed, 1000 optimizer trials = ~{1000*vec_time:.0f}s")

    # ------------------------------------------------------------------
    section("7. AI Analyst (if enabled)")
    # ------------------------------------------------------------------

    analyze_backtest(analyst_metrics)

    print("Done. See README.md for full documentation.")


if __name__ == "__main__":
    main()
