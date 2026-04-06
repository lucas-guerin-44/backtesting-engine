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
from portfolio_optimizer import StrategyConfig, portfolio_optimize, portfolio_walk_forward
from strategies import (
    DonchianBreakoutStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    TrendFollowingStrategy,
)
from ai_analyst import analyze_portfolio
from utils import compute_sharpe, fetch_ohlc


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INSTRUMENTS = [
    "XAUUSD", "EURUSD", "SPX500",
    "NDX100", "GER40", "GBPUSD", "USOUSD",
]
TIMEFRAME = "D1"
START_DATE = "2019-01-01"
END_DATE = "2024-12-31"
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
_TF_KWARGS = {"use_trailing_stop": True, "allow_reentry": True, "atr_stop_mult": 3.0}

STRATEGY_MAP = {
    "XAUUSD": ("Trend Following", TrendFollowingStrategy, _TF_KWARGS),
    "BTCUSD": ("Momentum",        MomentumStrategy,       {}),
    "USOUSD": ("Donchian",        DonchianBreakoutStrategy, {}),
    "SPX500": ("Trend Following", TrendFollowingStrategy, _TF_KWARGS),
    "NDX100": ("Momentum",        MomentumStrategy,       {}),
    "GER40":  ("Trend Following", TrendFollowingStrategy, _TF_KWARGS),
    "EURUSD": ("Mean Reversion",  MeanReversionStrategy,  {}),
    "GBPUSD": ("Mean Reversion",  MeanReversionStrategy,  {}),
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
        "Risk Parity": RiskParityAllocator(min_lookback=60),
        "Corr-Aware": CorrelationAwareAllocator(min_lookback=60),
        "Regime-Aware": RegimeAllocator(
            trend_symbols=trend_syms,
            reversion_symbols=revert_syms,
            vol_lookback=200,
            vol_history=5000,
            vol_threshold_pct=50.0,
            regime_boost=2.0,
            min_lookback=300,
        ),
    }

    # Portfolio-level risk constraints applied to all runs
    limits = RiskLimits(
        max_gross_exposure=0.9,    # Max 90% of equity deployed
        max_net_exposure=0.80,     # Max 80% net directional exposure
        max_single_asset=0.25,     # No single asset > 25% of equity
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
    analyst_alloc_metrics = {}

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
            rebalance_frequency=500,
            vol_lookback=500,
            risk_limits=limits,
            costs_by_symbol=COSTS_BY_SYMBOL,
        )
        result = pbt.run()
        portfolio_results[alloc_name] = result

        ret = (result.equity_curve[-1] - STARTING_CASH) / STARTING_CASH * 100
        sharpe = compute_sharpe(result.equity_curve)

        alloc_m = {
            "pct_return": ret, "sharpe": sharpe,
            "max_drawdown": pbt.max_drawdown * 100,
            "total_trades": len(result.trades),
        }
        if result.allocation_history:
            alloc_m["weights"] = result.allocation_history[-1]
        analyst_alloc_metrics[alloc_name] = alloc_m

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
    section(f"3. Portfolio Optimization ({len(opt_symbols)} assets, 30 trials)")
    # ------------------------------------------------------------------

    configs = {}
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
        configs[sym] = StrategyConfig(strategy_cls=strat_cls, param_space=space,
                                      fixed_params=strat_kw)

    strat_names = {s: STRATEGY_MAP.get(s, ("?",))[0] for s in opt_symbols}
    print(f"Optimizing {len(opt_symbols)} assets:")
    for s in opt_symbols:
        print(f"  {s:<10s} {strat_names[s]}")
    print(f"\nRunning 30 Optuna trials (Bayesian search)...\n")

    t0 = time.perf_counter()
    opt_result = portfolio_optimize(
        strategy_configs=configs,
        dataframes=opt_dfs,
        allocator=RiskParityAllocator(min_lookback=60),
        n_trials=30,
        objective="sharpe",
        starting_cash=STARTING_CASH,
        commission_bps=COMMISSION_BPS,
        slippage_bps=SLIPPAGE_BPS,
        rebalance_frequency=500,
        vol_lookback=500,
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

    print(f"3 train/test splits across all {len(opt_symbols)} assets...")
    print("For each window: optimize on train, evaluate on unseen test data\n")

    wf = portfolio_walk_forward(
        strategy_configs=configs,
        dataframes=opt_dfs,
        allocator=RiskParityAllocator(min_lookback=60),
        n_splits=3,
        train_ratio=0.7,
        n_trials=20,
        objective="sharpe",
        starting_cash=STARTING_CASH,
        commission_bps=COMMISSION_BPS,
        slippage_bps=SLIPPAGE_BPS,
        rebalance_frequency=500,
        vol_lookback=500,
    )

    print(wf.summary.to_string(index=False))
    print()
    print(f"In-sample mean:      {wf.in_sample_mean:>8.4f}  (optimized — always looks good)")
    print(f"Out-of-sample mean:  {wf.out_of_sample_mean:>8.4f}  (the honest number)")
    print(f"Degradation:         {wf.degradation:>8.4f}  (IS - OOS)")
    print()
    if wf.degradation > 0.5:
        print(">> Large degradation — optimizer is curve-fitting.")
    elif wf.out_of_sample_mean > 0:
        print(">> Positive OOS — there may be a real portfolio-level edge.")
    else:
        print(">> Negative OOS — no reliable portfolio edge detected.")

    # ------------------------------------------------------------------
    section("5. AI Analyst (if enabled)")
    # ------------------------------------------------------------------

    analyst_wf = {
        "is_mean": wf.in_sample_mean, "oos_mean": wf.out_of_sample_mean,
        "degradation": wf.degradation,
    }
    analyze_portfolio(analyst_alloc_metrics, walk_forward=analyst_wf)

    print("Done.")


if __name__ == "__main__":
    main()
