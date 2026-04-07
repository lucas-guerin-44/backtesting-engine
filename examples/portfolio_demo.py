#!/usr/bin/env python3
"""
Multi-Asset Portfolio Backtest Demo

Fetches 8 instruments from the datalake and runs:
  1. Single-asset baselines with mixed strategies per asset class
  2. Multi-asset portfolio under all three allocation schemes
  3. Portfolio parameter optimization (Bayesian search)
  4. Walk-forward validation (the overfitting test)

Usage:
    python examples/portfolio_demo.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.allocation import (
    CorrelationAwareAllocator,
    EqualWeightAllocator,
    RegimeAllocator,
    RiskParityAllocator,
)
from backtesting.backtest import Backtester
from backtesting.plot import plot_portfolio
from backtesting.portfolio_backtest import PortfolioBacktester, RiskLimits
from portfolio_optimizer import StrategyConfig, portfolio_optimize
from strategies import (
    DonchianBreakoutStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    TrendFollowingStrategy,
)
from backtesting.statistics import compute_sharpe
from utils import fetch_ohlc


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INSTRUMENTS = [
    "XAUUSD", "EURUSD", "SPX500",
    "NDX100", "GER40", "USOUSD",
]
TIMEFRAME = "D1"
START_DATE = "2012-01-01"
END_DATE = "2025-12-31"
STARTING_CASH = 100_000
COMMISSION_BPS = 5.0
SLIPPAGE_BPS = 2.0

# Per-asset transaction costs (reflects real market microstructure)
COSTS_BY_SYMBOL = {
    "BTCUSD": {"commission_bps": 10.0, "slippage_bps": 5.0},   # Crypto: wider spreads
    "EURUSD": {"commission_bps": 2.0, "slippage_bps": 1.0},    # Major FX: very tight
    "GBPUSD": {"commission_bps": 2.0, "slippage_bps": 1.0},    # Major FX: very tight
    "XAUUSD": {"commission_bps": 5.0, "slippage_bps": 3.0},    # Gold: moderate
    "USOUSD": {"commission_bps": 5.0, "slippage_bps": 3.0},    # Oil: moderate
    "SPX500": {"commission_bps": 3.0, "slippage_bps": 1.0},    # Index CFD: tight
    "NDX100": {"commission_bps": 3.0, "slippage_bps": 1.0},    # Index CFD: tight
    "GER40":  {"commission_bps": 3.0, "slippage_bps": 2.0},    # European index
}

# Map each instrument to a strategy that fits its asset class
# Trend-following assets get trailing stop + re-entry enabled
_TF_KWARGS = {"use_trailing_stop": True, "allow_reentry": True, "atr_stop_mult": 3.0,
              "trend_filter_period": 200}
_FILTER_200 = {"trend_filter_period": 200}

STRATEGY_MAP = {
    "XAUUSD": ("Trend Following", TrendFollowingStrategy, _TF_KWARGS),
    "BTCUSD": ("Momentum",        MomentumStrategy,       _FILTER_200),
    "USOUSD": ("Donchian",        DonchianBreakoutStrategy, _FILTER_200),
    "SPX500": ("Trend Following", TrendFollowingStrategy, _TF_KWARGS),
    "NDX100": ("Momentum",        MomentumStrategy,       _FILTER_200),
    "GER40":  ("Trend Following", TrendFollowingStrategy, _TF_KWARGS),
    "EURUSD": ("Donchian",        DonchianBreakoutStrategy, _FILTER_200),
    "GBPUSD": ("Mean Reversion",  MeanReversionStrategy,  _FILTER_200),
}


def fetch_and_prepare(instrument: str) -> pd.DataFrame:
    """Fetch from datalake and return a DatetimeIndex OHLC DataFrame."""
    raw = fetch_ohlc(instrument, TIMEFRAME, START_DATE, END_DATE)
    if raw.empty:
        raise RuntimeError(f"No data returned for {instrument}")
    df = raw[["timestamp", "open", "high", "low", "close"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def section(title):
    print()
    print(f"{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print()


def main():
    # ------------------------------------------------------------------
    section("Fetching data from datalake")
    # ------------------------------------------------------------------

    dataframes = {}
    for inst in INSTRUMENTS:
        try:
            df = fetch_and_prepare(inst)
            dataframes[inst] = df
            print(f"  {inst:<8s}  {len(df):>6,} bars  "
                  f"{df.index[0].date()} -> {df.index[-1].date()}")
        except Exception as e:
            print(f"  {inst:<8s}  SKIPPED ({e})")

    if len(dataframes) < 2:
        print("Need at least 2 instruments. Check datalake connectivity.")
        sys.exit(1)

    symbols = sorted(dataframes.keys())
    print(f"\n  {len(symbols)} instruments loaded")

    # ------------------------------------------------------------------
    section("1. Single-Asset Baselines (mixed strategies)")
    # ------------------------------------------------------------------

    print(f"{'Instrument':<10s} {'Strategy':<18s} {'Bars':>7s} {'Return':>9s} "
          f"{'B&H':>9s} {'Max DD':>8s} {'Sharpe':>8s} {'Trades':>7s} {'Win %':>7s}")
    print("-" * 88)

    for sym in symbols:
        strat_name, strat_cls, strat_kw = STRATEGY_MAP.get(sym, ("Trend Following", TrendFollowingStrategy, _TF_KWARGS))
        strat = strat_cls(**strat_kw)
        bt = Backtester(
            dataframes[sym], strat, starting_cash=STARTING_CASH,
            commission_bps=COMMISSION_BPS, slippage_bps=SLIPPAGE_BPS,
            symbol=sym,
        )
        eq, trades = bt.run()
        wins = len([t for t in trades if t.pnl and t.pnl > 0])
        win_rate = wins / len(trades) * 100 if trades else 0
        ret = (eq[-1] - STARTING_CASH) / STARTING_CASH * 100
        sharpe = compute_sharpe(eq)

        # Buy-and-hold benchmark
        closes = dataframes[sym]["close"]
        bnh = (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0] * 100

        print(f"{sym:<10s} {strat_name:<18s} {len(eq):>7,} {ret:>+8.2f}% "
              f"{bnh:>+8.2f}% {bt.max_drawdown*100:>7.2f}% {sharpe:>8.4f} "
              f"{len(trades):>7d} {win_rate:>6.1f}%")

    # ------------------------------------------------------------------
    section("2. Multi-Asset Portfolio — Allocation Comparison")
    # ------------------------------------------------------------------

    # Build trend/reversion symbol sets from the strategy map
    trend_syms = {s for s in symbols
                  if STRATEGY_MAP.get(s, (None,))[0] in ("Trend Following", "Momentum", "Donchian")}
    revert_syms = {s for s in symbols
                   if STRATEGY_MAP.get(s, (None,))[0] == "Mean Reversion"}

    allocators = {
        "Equal Weight": EqualWeightAllocator(),
        "Risk Parity": RiskParityAllocator(min_lookback=60, max_weight=0.30),
        "Corr-Aware": CorrelationAwareAllocator(
            min_lookback=60, corr_threshold=0.5, max_weight=0.30,
        ),
        "Regime-Aware": RegimeAllocator(
            trend_symbols=trend_syms,
            reversion_symbols=revert_syms,
            vol_lookback=200,
            vol_history=5000,
            vol_threshold_pct=50.0,
            regime_boost=2.0,
            min_lookback=300,
            max_weight=0.30,
        ),
    }

    # Portfolio-level risk constraints applied to all runs
    limits = RiskLimits(
        max_gross_exposure=0.9,    # Max 90% of equity deployed
        max_net_exposure=0.80,     # Max 80% net directional exposure
        max_single_asset=0.30,     # No single asset > 30% of equity
        max_open_positions=6,      # Max 6 assets with open positions
    )

    print(f"Risk limits: gross<={limits.max_gross_exposure:.0%}, "
          f"net<={limits.max_net_exposure:.0%}, "
          f"per-asset<={limits.max_single_asset:.0%}, "
          f"max positions={limits.max_open_positions}")
    print()
    # Equal-weight buy-and-hold benchmark
    bnh_returns = []
    for sym in symbols:
        closes = dataframes[sym]["close"]
        bnh_returns.append((closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0])
    bnh_portfolio = np.mean(bnh_returns) * 100

    print(f"{'Allocator':<16s} {'Return':>9s} {'Max DD':>8s} {'Sharpe':>8s} "
          f"{'Trades':>7s}")
    print("-" * 52)
    print(f"{'Buy & Hold':<16s} {bnh_portfolio:>+8.2f}%      -        -        -")

    portfolio_results = {}  # For charting

    for alloc_name, allocator in allocators.items():
        strats = {}
        for sym in symbols:
            _, strat_cls, strat_kw = STRATEGY_MAP.get(sym, ("Trend Following", TrendFollowingStrategy, _TF_KWARGS))
            strats[sym] = strat_cls(**strat_kw)

        pbt = PortfolioBacktester(
            dataframes=dataframes,
            strategies=strats,
            allocator=allocator,
            starting_cash=STARTING_CASH,
            commission_bps=COMMISSION_BPS,
            slippage_bps=SLIPPAGE_BPS,
            rebalance_frequency=21,
            vol_lookback=500,
            risk_limits=limits,
            costs_by_symbol=COSTS_BY_SYMBOL,
        )
        result = pbt.run()
        portfolio_results[alloc_name] = result

        ret = (result.equity_curve[-1] - STARTING_CASH) / STARTING_CASH * 100
        sharpe = compute_sharpe(result.equity_curve)

        print(f"{alloc_name:<16s} {ret:>+8.2f}% {pbt.max_drawdown*100:>7.2f}% "
              f"{sharpe:>8.4f} {len(result.trades):>7d}")

        if result.allocation_history:
            last_w = result.allocation_history[-1]
            top = sorted(last_w.items(), key=lambda x: x[1], reverse=True)
            weights_str = "  ".join(f"{s}: {w:.1%}" for s, w in top)
            print(f"  Weights: {weights_str}")
        skips = [e for e in result.audit_log if e.event == "skip"]
        if skips:
            from collections import Counter
            reasons = Counter(s.reason for s in skips)
            reasons_str = ", ".join(f"{r}: {n}" for r, n in reasons.most_common())
            print(f"  Skipped: {len(skips)} trades ({reasons_str})")
            
        print(" ")

    # Build equal-weight buy & hold equity curve aligned to portfolio timestamps
    # Use union of all timestamps with forward-fill (same as PortfolioBacktester)
    master_idx = dataframes[symbols[0]].index
    for sym in symbols[1:]:
        master_idx = master_idx.union(dataframes[sym].index)
    master_idx = master_idx.sort_values()

    normalized = []
    for sym in symbols:
        closes = dataframes[sym]["close"].reindex(master_idx, method="ffill")
        first_valid = closes.first_valid_index()
        normalized.append(closes / closes.loc[first_valid])
    bnh_df = pd.concat(normalized, axis=1).ffill().bfill()
    bnh_equity = (bnh_df.mean(axis=1) * STARTING_CASH).values
    portfolio_results["Buy & Hold"] = bnh_equity

    # Save allocation comparison chart
    os.makedirs("docs", exist_ok=True)
    plot_portfolio(
        portfolio_results,
        starting_cash=STARTING_CASH,
        save_path="docs/portfolio_allocation.png",
        show=False,
    )

    opt_symbols = symbols
    opt_dfs = dataframes

    # ------------------------------------------------------------------
    section(f"3. Portfolio Optimization ({len(opt_symbols)} assets, 200 trials)")
    # ------------------------------------------------------------------

    # Full search first: find the optimal parameter region per asset.
    # This uses the full param space to discover where the signal lives.
    full_configs = {}
    for sym in opt_symbols:
        _, strat_cls, strat_kw = STRATEGY_MAP.get(sym, ("Trend Following", TrendFollowingStrategy, _TF_KWARGS))
        if strat_cls == TrendFollowingStrategy:
            space = {
                "fast_period": (10, 40),
                "slow_period": (30, 100),
                "atr_stop_mult": (1.0, 4.0),
                "risk_per_trade": (0.01, 0.04),
            }
        elif strat_cls == MeanReversionStrategy:
            space = {
                "bb_period": (10, 30),
                "bb_std": (1.5, 3.0),
                "rsi_period": (7, 21),
                "atr_stop_mult": (1.5, 4.0),
                "risk_per_trade": (0.01, 0.04),
            }
        elif strat_cls == MomentumStrategy:
            space = {
                "lookback": (10, 40),
                "entry_threshold": (0.01, 0.06),
                "atr_stop_mult": (1.0, 4.0),
                "atr_target_mult": (2.0, 6.0),
                "risk_per_trade": (0.01, 0.04),
            }
        else:  # Donchian
            space = {
                "channel_period": (10, 40),
                "atr_stop_mult": (1.0, 4.0),
                "risk_reward": (1.5, 4.0),
                "risk_per_trade": (0.01, 0.04),
            }
        full_configs[sym] = StrategyConfig(strategy_cls=strat_cls, param_space=space,
                                           fixed_params=strat_kw)

    strat_names = {s: STRATEGY_MAP.get(s, ("?",))[0] for s in opt_symbols}
    print(f"Optimizing {len(opt_symbols)} assets:")
    for s in opt_symbols:
        print(f"  {s:<10s} {strat_names[s]}")
    print(f"\nRunning 200 Optuna trials (Bayesian search)...\n")

    t0 = time.perf_counter()
    opt_result = portfolio_optimize(
        strategy_configs=full_configs,
        dataframes=opt_dfs,
        allocator=RiskParityAllocator(min_lookback=60, max_weight=0.30),
        n_trials=200,
        objective="sharpe",
        starting_cash=STARTING_CASH,
        commission_bps=COMMISSION_BPS,
        slippage_bps=SLIPPAGE_BPS,
        rebalance_frequency=21,
        vol_lookback=500,
        min_trades=10,
        top_k_avg=5,
    )
    elapsed = time.perf_counter() - t0

    print(f"Completed in {elapsed:.1f}s")
    print(f"Best portfolio Sharpe: {opt_result.best_score:.4f}\n")
    print("Best params per asset:")
    for sym in opt_symbols:
        params = opt_result.best_strategy_params.get(sym, {})
        params_str = ", ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in params.items()
        )
        print(f"  {sym} ({strat_names[sym]}): {params_str}")

    # ------------------------------------------------------------------
    section("4. Walk-Forward Validation (the overfitting test)")
    # ------------------------------------------------------------------

    # Two-stage walk-forward: for each split, BOTH optimization stages run
    # on training data only. The OOS window is never seen during parameter
    # selection. This avoids the contamination of locking params from a
    # full-dataset optimization.
    #
    #   Stage 1: optimize ALL params on training data (find signal region)
    #   Stage 2: lock signal params, optimize only risk_per_trade on training data
    #   Evaluate: run locked+sized params on test data (completely unseen)

    n_splits = 3
    train_ratio = 0.7
    wf_allocator = RiskParityAllocator(min_lookback=60, max_weight=0.30)

    # Build master timestamp index for splitting
    master_idx = opt_dfs[opt_symbols[0]].index
    for sym in opt_symbols[1:]:
        master_idx = master_idx.union(opt_dfs[sym].index)
    master_idx = master_idx.sort_values()
    total_bars = len(master_idx)
    window_size = total_bars // n_splits

    from optimizer import OBJECTIVES as _OBJ
    from utils import infer_freq_per_year
    obj_fn = _OBJ["sharpe"]
    freq = infer_freq_per_year(opt_dfs[opt_symbols[0]].index)

    print(f"Two-stage walk-forward: {n_splits} splits, both stages train-only.")
    print(f"  Stage 1: full optimization per asset on training data (200 trials)")
    print(f"  Stage 2: lock signal params, optimize risk_per_trade (200 trials)")
    print(f"  Evaluate: test on unseen data\n")

    wf_splits = []
    for split_idx in range(n_splits):
        window_start = split_idx * window_size
        window_end = min(window_start + window_size, total_bars)
        split_point = window_start + int((window_end - window_start) * train_ratio)

        train_start_ts = master_idx[0]  # anchored
        train_end_ts = master_idx[split_point - 1]
        test_start_ts = master_idx[split_point]
        test_end_ts = master_idx[window_end - 1]

        # Split each asset by date range
        train_dfs = {}
        test_dfs = {}
        valid = True
        for sym in opt_symbols:
            d = opt_dfs[sym]
            tr = d[(d.index >= train_start_ts) & (d.index <= train_end_ts)]
            te = d[(d.index >= test_start_ts) & (d.index <= test_end_ts)]
            if len(tr) < 20 or len(te) < 5:
                valid = False
                break
            train_dfs[sym] = tr
            test_dfs[sym] = te
        if not valid:
            continue

        # Stage 1: full optimization on training data only
        stage1 = portfolio_optimize(
            strategy_configs=full_configs,
            dataframes=train_dfs,
            allocator=wf_allocator,
            n_trials=200,
            objective="sharpe",
            starting_cash=STARTING_CASH,
            commission_bps=COMMISSION_BPS,
            slippage_bps=SLIPPAGE_BPS,
            rebalance_frequency=21,
            vol_lookback=500,
            min_trades=10,
            top_k_avg=5,
        )

        # Stage 2: lock signal params from stage 1, optimize risk_per_trade only
        locked_configs = {}
        for sym in opt_symbols:
            _, strat_cls, strat_kw = STRATEGY_MAP.get(sym, ("Trend Following", TrendFollowingStrategy, _TF_KWARGS))
            optimized = stage1.best_strategy_params.get(sym, {})
            locked_fixed = dict(strat_kw)
            for k, v in optimized.items():
                if k != "risk_per_trade":
                    locked_fixed[k] = v
            locked_configs[sym] = StrategyConfig(
                strategy_cls=strat_cls,
                param_space={"risk_per_trade": (0.01, 0.04)},
                fixed_params=locked_fixed,
            )

        stage2 = portfolio_optimize(
            strategy_configs=locked_configs,
            dataframes=train_dfs,
            allocator=wf_allocator,
            n_trials=200,
            objective="sharpe",
            starting_cash=STARTING_CASH,
            commission_bps=COMMISSION_BPS,
            slippage_bps=SLIPPAGE_BPS,
            rebalance_frequency=21,
            vol_lookback=500,
            min_trades=10,
            top_k_avg=5,
        )

        # Evaluate on test data (never seen by either stage)
        test_strategies = {}
        for sym in opt_symbols:
            merged = {**locked_configs[sym].fixed_params,
                      **stage2.best_strategy_params.get(sym, {})}
            test_strategies[sym] = locked_configs[sym].strategy_cls(**merged)

        pbt_test = PortfolioBacktester(
            dataframes=test_dfs,
            strategies=test_strategies,
            allocator=wf_allocator,
            starting_cash=STARTING_CASH,
            commission_bps=COMMISSION_BPS,
            slippage_bps=SLIPPAGE_BPS,
            rebalance_frequency=21,
            vol_lookback=500,
        )
        test_result = pbt_test.run()
        oos_score = obj_fn(test_result.equity_curve, test_result.trades,
                           freq_per_year=freq)
        oos_ret = (test_result.equity_curve[-1] - STARTING_CASH) / STARTING_CASH * 100

        wf_splits.append({
            "split": split_idx,
            "train": f"{train_start_ts.date()} to {train_end_ts.date()}",
            "test": f"{test_start_ts.date()} to {test_end_ts.date()}",
            "IS_score": round(stage2.best_score, 4),
            "OOS_score": round(oos_score, 4),
            "OOS_return_pct": round(oos_ret, 2),
            "OOS_max_dd_pct": round(pbt_test.max_drawdown * 100, 2),
            "OOS_trades": len(test_result.trades),
        })

    wf_summary = pd.DataFrame(wf_splits)
    is_scores = [s["IS_score"] for s in wf_splits]
    oos_scores = [s["OOS_score"] for s in wf_splits]
    is_mean = round(np.mean(is_scores), 4)
    oos_mean = round(np.mean(oos_scores), 4)
    degradation = round(is_mean - oos_mean, 4)

    print(wf_summary.to_string(index=False))
    print()
    print(f"In-sample mean:      {is_mean:>8.4f}  (optimized — always looks good)")
    print(f"Out-of-sample mean:  {oos_mean:>8.4f}  (the honest number)")
    print(f"Degradation:         {degradation:>8.4f}  (IS - OOS)")
    print()
    if degradation > 0.5:
        print(">> Large degradation — optimizer is curve-fitting.")
    elif oos_mean > 0:
        print(">> Positive OOS — there may be a real portfolio-level edge.")
    else:
        print(">> Negative OOS — no reliable portfolio edge detected.")

    print("Done.")


if __name__ == "__main__":
    main()
