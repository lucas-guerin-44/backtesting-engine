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
from backtesting.plot import plot_strategy_comparison
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
from utils import compute_sharpe, fetch_ohlc

INSTRUMENT = "XAUUSD"
TIMEFRAME = "D1"
START_DATE = "2012-01-01"
END_DATE = "2025-12-31"


def load_data():
    """Fetch OHLC data from the datalake."""
    raw = fetch_ohlc(INSTRUMENT, TIMEFRAME, START_DATE, END_DATE)
    if raw.empty:
        print(f"ERROR: No data returned for {INSTRUMENT}")
        sys.exit(1)
    df = raw[["timestamp", "open", "high", "low", "close"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
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
    print(f"Instrument: {INSTRUMENT} {TIMEFRAME}")
    print(f"Price range: {df['close'].min():.0f} - {df['close'].max():.0f}")

    # ------------------------------------------------------------------
    section("1. Backtest All Strategies")
    # ------------------------------------------------------------------

    _tf_kw = {"use_trailing_stop": True, "allow_reentry": True,
              "atr_stop_mult": 3.0, "trend_filter_period": 200}
    _f200 = {"trend_filter_period": 200}
    strategies = {
        "Trend Following": TrendFollowingStrategy(**_tf_kw),
        "Mean Reversion": MeanReversionStrategy(**_f200),
        "Momentum": MomentumStrategy(**_f200),
        "Donchian Breakout": DonchianBreakoutStrategy(**_f200),
    }

    print(f"{'Strategy':<20s} {'Return':>8s} {'Max DD':>8s} {'Sharpe':>8s} {'Trades':>7s} {'Win %':>7s}")
    print("-" * 60)

    analyst_metrics = {}
    equity_curves = {}

    for name, strat in strategies.items():
        bt = Backtester(df, strat, starting_cash=10_000,
                        commission_bps=5.0, slippage_bps=2.0, symbol=INSTRUMENT)
        eq, trades = bt.run()
        equity_curves[name] = eq

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

    # Buy & hold benchmark
    closes = df["close"]
    bnh_equity = (closes / closes.iloc[0] * 10_000).to_numpy()
    equity_curves["Buy & Hold"] = bnh_equity
    bnh_ret = (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0] * 100
    bnh_peak = closes.cummax()
    bnh_dd = ((bnh_peak - closes) / bnh_peak).max() * 100
    print(f"{'Buy & Hold':<20s} {bnh_ret:>+7.2f}%   {bnh_dd:>6.2f}%        -       -       -")

    os.makedirs("docs", exist_ok=True)
    plot_strategy_comparison(equity_curves, save_path="docs/strategy_comparison.png", show=False)

    # ------------------------------------------------------------------
    section("2. Parameter Optimization (Momentum)")
    # ------------------------------------------------------------------

    # Full search: find the optimal parameter region
    full_param_space = {
        "lookback": (5, 40),
        "entry_threshold": (0.01, 0.08),
        "atr_stop_mult": (1.0, 5.0),
        "atr_target_mult": (2.0, 8.0),
        "risk_per_trade": (0.01, 0.05),
    }
    fixed_params = {"trend_filter_period": 200}

    print(f"Running 1000 Optuna trials (Bayesian search)...")
    t0 = time.perf_counter()
    result = optimize(
        MomentumStrategy, full_param_space, df,
        n_trials=1000, objective="sharpe",
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

    analyst_metrics["Momentum"]["optimization"] = {
        "best_sharpe": result.best_score, "n_trials": 1000,
    }

    print(f"\nTop 5 trials:")
    print(result.all_trials.head(5).to_string(index=False))

    # ------------------------------------------------------------------
    section("3. Walk-Forward Validation (the overfitting test)")
    # ------------------------------------------------------------------

    # Lock down the converged params, only optimize risk_per_trade
    locked_param_space = {"risk_per_trade": (0.01, 0.05)}
    locked_fixed = {
        "lookback": result.best_params.get("lookback", 6),
        "entry_threshold": result.best_params.get("entry_threshold", 0.018),
        "atr_stop_mult": result.best_params.get("atr_stop_mult", 1.35),
        "atr_target_mult": result.best_params.get("atr_target_mult", 2.1),
        "trend_filter_period": 200,
    }

    print("Locked params from optimization convergence:")
    for k, v in locked_fixed.items():
        if k != "trend_filter_period":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("Optimizing only: risk_per_trade")
    print()
    print("Splitting data into 3 train/test windows (200 trials each)...")
    print("For each window: optimize on train, evaluate on test (unseen data)")
    print()

    wf = walk_forward(
        MomentumStrategy, locked_param_space, df,
        n_splits=3, train_ratio=0.7, n_trials=200,
        objective="sharpe", commission_bps=5.0, slippage_bps=2.0,
        fixed_params=locked_fixed,
        min_trades=10, top_k_avg=5, anchored=True,
    )

    print(wf.summary.to_string(index=False))
    print()
    print(f"In-sample mean Sharpe:      {wf.in_sample_mean:>8.4f}  (optimized — always looks good)")
    print(f"Out-of-sample mean Sharpe:  {wf.out_of_sample_mean:>8.4f}  (the honest number)")
    print(f"Degradation:                {wf.degradation:>8.4f}  (IS - OOS, measures overfitting)")
    print()
    analyst_metrics["Momentum"]["walk_forward"] = {
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
    section("4. True Holdout Test (2012-2022 train, 2023-2025 test)")
    # ------------------------------------------------------------------

    holdout_split = "2023-01-01"
    train_df = df[df.index < holdout_split]
    test_df = df[df.index >= holdout_split]
    print(f"Train: {train_df.index[0].date()} to {train_df.index[-1].date()} ({len(train_df)} bars)")
    print(f"Test:  {test_df.index[0].date()} to {test_df.index[-1].date()} ({len(test_df)} bars)")
    print()

    # Optimize on train with full param space
    print("Optimizing on training data (1000 trials)...")
    t0 = time.perf_counter()
    holdout_opt = optimize(
        MomentumStrategy, full_param_space, train_df,
        n_trials=1000, objective="sharpe",
        commission_bps=5.0, slippage_bps=2.0,
        fixed_params=fixed_params,
        min_trades=10, top_k_avg=5,
    )
    elapsed = time.perf_counter() - t0
    print(f"Completed in {elapsed:.1f}s")
    print(f"In-sample Sharpe: {holdout_opt.best_score:.4f}")
    print(f"Best params:")
    for k, v in holdout_opt.best_params.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Evaluate on holdout
    holdout_params = {**fixed_params, **holdout_opt.best_params}
    bt_holdout = Backtester(test_df, MomentumStrategy(**holdout_params),
                            starting_cash=10_000, commission_bps=5.0,
                            slippage_bps=2.0, symbol=INSTRUMENT)
    eq_h, trades_h = bt_holdout.run()
    holdout_ret = (eq_h[-1] - 10_000) / 10_000 * 100
    holdout_sharpe = compute_sharpe(eq_h)
    holdout_dd = bt_holdout.max_drawdown * 100
    wins_h = len([t for t in trades_h if t.pnl and t.pnl > 0])
    win_rate_h = wins_h / len(trades_h) * 100 if trades_h else 0

    # Holdout B&H
    h_closes = test_df["close"]
    h_bnh = (h_closes.iloc[-1] - h_closes.iloc[0]) / h_closes.iloc[0] * 100

    print(f"\nHoldout results (2023-2025, completely unseen):")
    print(f"  Strategy:   {holdout_ret:>+8.2f}% return, {holdout_dd:.2f}% max DD, "
          f"Sharpe {holdout_sharpe:.4f}, {len(trades_h)} trades, {win_rate_h:.1f}% win")
    print(f"  Buy & Hold: {h_bnh:>+8.2f}% return")
    print(f"  Degradation: {holdout_opt.best_score - holdout_sharpe:.4f} (IS - holdout)")

    # ------------------------------------------------------------------
    section("5. Statistical Significance")
    # ------------------------------------------------------------------

    from backtesting.statistics import compute_statistical_report

    # Run the best strategy and check if the edge is real
    bt = Backtester(df, MomentumStrategy(trend_filter_period=200),
                    starting_cash=10_000, commission_bps=5.0, slippage_bps=2.0,
                    symbol=INSTRUMENT)
    eq, trades = bt.run()

    report = compute_statistical_report(eq, trades, n_trials_tested=1000, seed=42)
    print(report)

    # ------------------------------------------------------------------
    section("6. Results Database")
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
    section("7. Performance: Event-Driven vs Vectorized")
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
    section("8. AI Analyst (if enabled)")
    # ------------------------------------------------------------------

    analyze_backtest(analyst_metrics)

    print("Done. See README.md for full documentation.")


if __name__ == "__main__":
    main()
