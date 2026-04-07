"""Portfolio-level parameter optimizer with walk-forward validation.

Extends the single-asset optimizer to handle multiple assets simultaneously:
- Optimizes per-asset strategy parameters (prefixed with symbol name in Optuna)
- Optionally optimizes allocation weights alongside strategy params
- Walk-forward validation applies the same train/test date splits to all assets

Uses Optuna's Bayesian (TPE) sampler, same as the single-asset optimizer.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import numpy as np
import optuna
import pandas as pd

from backtesting.allocation import (
    Allocator,
    AllocationWeights,
    EqualWeightAllocator,
)
from backtesting.portfolio_backtest import PortfolioBacktester
from optimizer import OBJECTIVES, _suggest_param, _average_top_k_params, _check_constraints
from utils import compute_sharpe, infer_freq_per_year

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Configuration for one asset's strategy in portfolio optimization."""
    strategy_cls: Type
    param_space: Dict[str, Any]
    fixed_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioOptimizationResult:
    """Result from portfolio optimization."""
    best_strategy_params: Dict[str, Dict[str, Any]]
    best_allocation_weights: Optional[Dict[str, float]]
    best_score: float
    objective_name: str
    n_trials: int
    all_trials: pd.DataFrame


@dataclass
class PortfolioWalkForwardResult:
    """Walk-forward result at portfolio level."""
    splits: List[Dict[str, Any]]
    in_sample_mean: float
    out_of_sample_mean: float
    degradation: float
    summary: pd.DataFrame


class _FixedWeightAllocator(Allocator):
    """Allocator that always returns fixed weights (used during optimization)."""

    def __init__(self, weights: Dict[str, float]):
        self._weights = weights

    def compute_weights(self, symbols, close_arrays, lookback, current_idx):
        return AllocationWeights(weights=dict(self._weights), method="fixed")


def _build_strategies_from_trial(
    trial: optuna.Trial,
    strategy_configs: Dict[str, StrategyConfig],
) -> tuple:
    """Sample per-asset parameters from Optuna and instantiate strategies.

    Returns (strategies_dict, per_asset_params_dict).
    """
    strategies = {}
    all_params = {}
    for sym, config in sorted(strategy_configs.items()):
        params = {}
        for name, bounds in config.param_space.items():
            prefixed = f"{sym}__{name}"
            params[name] = _suggest_param(trial, prefixed, bounds)
        if not _check_constraints(params):
            return None, None
        params.update(config.fixed_params)
        strategies[sym] = config.strategy_cls(**params)
        all_params[sym] = params
    return strategies, all_params


def portfolio_optimize(
    strategy_configs: Dict[str, StrategyConfig],
    dataframes: Dict[str, pd.DataFrame],
    allocator: Optional[Allocator] = None,
    n_trials: int = 100,
    objective: str = "sharpe",
    starting_cash: float = 10_000,
    commission_bps: float = 5.0,
    slippage_bps: float = 2.0,
    max_leverage: float = 1.0,
    rebalance_frequency: int = 0,
    vol_lookback: int = 60,
    optimize_weights: bool = False,
    n_jobs: int = 1,
    min_trades: int = 0,
    top_k_avg: int = 1,
) -> PortfolioOptimizationResult:
    """Optimize portfolio strategy parameters using Bayesian search.

    Parameters
    ----------
    strategy_configs : dict
        Symbol -> StrategyConfig with cls, param_space, and fixed_params.
    dataframes : dict
        Symbol -> OHLC DataFrame.
    allocator : Allocator, optional
        Base allocator (ignored if ``optimize_weights=True``).
    n_trials : int
        Number of Optuna trials.
    objective : str
        What to maximize: "sharpe", "return", "calmar", or "sortino".
    optimize_weights : bool
        If True, also optimize per-asset allocation weights.

    Returns
    -------
    PortfolioOptimizationResult
    """
    obj_fn = OBJECTIVES.get(objective)
    if obj_fn is None:
        raise ValueError(f"Unknown objective '{objective}'. Choose from: {list(OBJECTIVES)}")

    symbols = sorted(strategy_configs.keys())
    base_allocator = allocator or EqualWeightAllocator()
    # Infer frequency from the first asset's timestamps
    freq = infer_freq_per_year(dataframes[symbols[0]].index)

    def _objective(trial: optuna.Trial) -> float:
        try:
            strategies, _ = _build_strategies_from_trial(trial, strategy_configs)
            if strategies is None:
                return -999.0

            if optimize_weights:
                raw_w = {s: trial.suggest_float(f"weight__{s}", 0.05, 1.0)
                         for s in symbols}
                total = sum(raw_w.values())
                weights = {s: w / total for s, w in raw_w.items()}
                alloc = _FixedWeightAllocator(weights)
            else:
                alloc = base_allocator

            pbt = PortfolioBacktester(
                dataframes=dataframes,
                strategies=strategies,
                allocator=alloc,
                starting_cash=starting_cash,
                commission_bps=commission_bps,
                slippage_bps=slippage_bps,
                max_leverage=max_leverage,
                rebalance_frequency=rebalance_frequency,
                vol_lookback=vol_lookback,
            )
            result = pbt.run()
            score = obj_fn(result.equity_curve, result.trades, freq_per_year=freq)
            if not np.isfinite(score):
                return -999.0
            if min_trades > 0 and len(result.trades) < min_trades:
                score = score * (len(result.trades) / min_trades)
            return score
        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            return -999.0

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler())
    study.optimize(_objective, n_trials=n_trials, n_jobs=n_jobs)

    # Parse best params back into per-asset dicts
    # Build a flat param_space with prefixed names for top-k averaging
    if top_k_avg > 1:
        flat_space = {}
        for sym, config in sorted(strategy_configs.items()):
            for name, bounds in config.param_space.items():
                flat_space[f"{sym}__{name}"] = bounds
        if optimize_weights:
            for s in symbols:
                flat_space[f"weight__{s}"] = (0.05, 1.0)
        flat_params = _average_top_k_params(study, flat_space, top_k_avg)
    else:
        flat_params = study.best_params

    best_strategy_params: Dict[str, Dict[str, Any]] = {s: {} for s in symbols}
    best_weights = None
    for key, val in flat_params.items():
        if key.startswith("weight__"):
            continue
        sym, param_name = key.split("__", 1)
        best_strategy_params[sym][param_name] = val

    if optimize_weights:
        raw_w = {s: flat_params.get(f"weight__{s}", 1.0) for s in symbols}
        total = sum(raw_w.values())
        best_weights = {s: w / total for s, w in raw_w.items()}

    # Build trials DataFrame
    rows = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            rows.append({**t.params, "score": t.value, "trial": t.number})
    trials_df = pd.DataFrame(rows).sort_values(
        "score", ascending=False).reset_index(drop=True)

    return PortfolioOptimizationResult(
        best_strategy_params=best_strategy_params,
        best_allocation_weights=best_weights,
        best_score=study.best_value,
        objective_name=objective,
        n_trials=n_trials,
        all_trials=trials_df,
    )


def portfolio_walk_forward(
    strategy_configs: Dict[str, StrategyConfig],
    dataframes: Dict[str, pd.DataFrame],
    allocator: Optional[Allocator] = None,
    n_splits: int = 4,
    train_ratio: float = 0.7,
    n_trials: int = 50,
    objective: str = "sharpe",
    starting_cash: float = 10_000,
    commission_bps: float = 5.0,
    slippage_bps: float = 2.0,
    max_leverage: float = 1.0,
    rebalance_frequency: int = 0,
    vol_lookback: int = 60,
    optimize_weights: bool = False,
    n_jobs: int = 1,
    min_trades: int = 0,
    top_k_avg: int = 1,
    anchored: bool = False,
) -> PortfolioWalkForwardResult:
    """Walk-forward validation at the portfolio level.

    Splits ALL asset DataFrames into the same rolling train/test windows,
    optimizes on training data, evaluates on test data.

    Returns
    -------
    PortfolioWalkForwardResult
    """
    obj_fn = OBJECTIVES.get(objective)
    if obj_fn is None:
        raise ValueError(f"Unknown objective '{objective}'. Choose from: {list(OBJECTIVES)}")

    symbols = sorted(strategy_configs.keys())
    freq = infer_freq_per_year(dataframes[symbols[0]].index)

    # Use union of all timestamps for splitting
    master_idx = dataframes[symbols[0]].index
    for sym in symbols[1:]:
        master_idx = master_idx.union(dataframes[sym].index)
    master_idx = master_idx.sort_values()
    total_bars = len(master_idx)

    window_size = total_bars // n_splits
    if window_size < 50:
        raise ValueError(f"Not enough data: {total_bars} bars / {n_splits} splits = "
                         f"{window_size} bars per window (need at least 50)")

    splits = []

    for split_idx in range(n_splits):
        window_start = split_idx * window_size
        window_end = min(window_start + window_size, total_bars)
        split_point = window_start + int((window_end - window_start) * train_ratio)

        train_start_ts = master_idx[0] if anchored else master_idx[window_start]
        train_end_ts = master_idx[split_point - 1]
        test_start_ts = master_idx[split_point]
        test_end_ts = master_idx[window_end - 1]

        # Split each asset's DataFrame by the date range
        train_dfs = {}
        test_dfs = {}
        valid = True
        for sym in symbols:
            df = dataframes[sym]
            train = df[(df.index >= train_start_ts) & (df.index <= train_end_ts)]
            test = df[(df.index >= test_start_ts) & (df.index <= test_end_ts)]
            if len(train) < 20 or len(test) < 5:
                valid = False
                break
            train_dfs[sym] = train
            test_dfs[sym] = test

        if not valid:
            continue

        # 1. Optimize on training data
        opt_result = portfolio_optimize(
            strategy_configs=strategy_configs,
            dataframes=train_dfs,
            allocator=allocator,
            n_trials=n_trials,
            objective=objective,
            starting_cash=starting_cash,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            max_leverage=max_leverage,
            rebalance_frequency=rebalance_frequency,
            vol_lookback=vol_lookback,
            optimize_weights=optimize_weights,
            n_jobs=n_jobs,
            min_trades=min_trades,
            top_k_avg=top_k_avg,
        )

        # 2. Evaluate best params on test data (out-of-sample)
        test_strategies = {}
        for sym in symbols:
            merged_params = {**strategy_configs[sym].fixed_params,
                            **opt_result.best_strategy_params.get(sym, {})}
            test_strategies[sym] = strategy_configs[sym].strategy_cls(**merged_params)

        if optimize_weights and opt_result.best_allocation_weights:
            test_allocator = _FixedWeightAllocator(opt_result.best_allocation_weights)
        else:
            test_allocator = allocator or EqualWeightAllocator()

        pbt = PortfolioBacktester(
            dataframes=test_dfs,
            strategies=test_strategies,
            allocator=test_allocator,
            starting_cash=starting_cash,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            max_leverage=max_leverage,
            rebalance_frequency=rebalance_frequency,
            vol_lookback=vol_lookback,
        )
        test_result = pbt.run()
        oos_score = obj_fn(test_result.equity_curve, test_result.trades,
                           freq_per_year=freq)

        split_result = {
            "split": split_idx,
            "train_start": str(train_start_ts.date()) if hasattr(train_start_ts, 'date') else str(train_start_ts),
            "train_end": str(train_end_ts.date()) if hasattr(train_end_ts, 'date') else str(train_end_ts),
            "test_start": str(test_start_ts.date()) if hasattr(test_start_ts, 'date') else str(test_start_ts),
            "test_end": str(test_end_ts.date()) if hasattr(test_end_ts, 'date') else str(test_end_ts),
            "best_params": opt_result.best_strategy_params,
            "in_sample_score": round(opt_result.best_score, 4),
            "out_of_sample_score": round(oos_score, 4),
            "oos_return_pct": round(
                (test_result.equity_curve[-1] - starting_cash) / starting_cash * 100, 2),
            "oos_max_dd_pct": round(pbt.max_drawdown * 100, 2),
            "oos_trades": len(test_result.trades),
        }
        splits.append(split_result)

        logger.info(
            f"Split {split_idx}: IS={opt_result.best_score:.4f} -> "
            f"OOS={oos_score:.4f} ({split_result['oos_return_pct']:+.2f}%)"
        )

    if not splits:
        raise ValueError("No valid splits produced. Check data length and n_splits.")

    is_scores = [s["in_sample_score"] for s in splits]
    oos_scores = [s["out_of_sample_score"] for s in splits]

    summary_df = pd.DataFrame([{
        "split": s["split"],
        "train": f"{s['train_start']} to {s['train_end']}",
        "test": f"{s['test_start']} to {s['test_end']}",
        "IS_score": s["in_sample_score"],
        "OOS_score": s["out_of_sample_score"],
        "OOS_return_pct": s["oos_return_pct"],
        "OOS_max_dd_pct": s["oos_max_dd_pct"],
        "OOS_trades": s["oos_trades"],
    } for s in splits])

    return PortfolioWalkForwardResult(
        splits=splits,
        in_sample_mean=round(np.mean(is_scores), 4),
        out_of_sample_mean=round(np.mean(oos_scores), 4),
        degradation=round(np.mean(is_scores) - np.mean(oos_scores), 4),
        summary=summary_df,
    )
