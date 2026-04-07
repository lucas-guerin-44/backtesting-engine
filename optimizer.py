"""Strategy parameter optimizer with walk-forward validation.

Supports two modes:
- **Single-period**: optimize parameters on one date range (fast, good for exploration).
- **Walk-forward**: split data into rolling train/test windows, optimize on each training
  window, evaluate on the out-of-sample test window. This is the only way to get a
  realistic estimate of live performance — single-period results are always overfit.

Uses Optuna's Bayesian (TPE) sampler for efficient search over the parameter space.

Example usage::

    from optimizer import optimize, walk_forward
    from strategies import TrendFollowingStrategy

    # Define what to search over
    param_space = {
        "fast_period": (5, 30),
        "slow_period": (20, 100),
        "atr_stop_mult": (1.0, 4.0),
        "risk_per_trade": (0.01, 0.05),
    }

    # Single-period optimization
    results = optimize(
        strategy_cls=TrendFollowingStrategy,
        param_space=param_space,
        df=df,
        n_trials=100,
        objective="sharpe",
    )

    # Walk-forward validation (the real test)
    wf_results = walk_forward(
        strategy_cls=TrendFollowingStrategy,
        param_space=param_space,
        df=df,
        n_splits=4,
        train_ratio=0.7,
        n_trials=50,
        objective="sharpe",
    )
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import optuna
import pandas as pd

from backtesting.backtest import Backtester
from utils import compute_sharpe, infer_freq_per_year

# Suppress Optuna's verbose trial logging (we log summaries ourselves)
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------

def _sharpe_objective(equity_curve: np.ndarray, trades: list,
                      freq_per_year: int = 252) -> float:
    """Maximize Sharpe ratio."""
    return compute_sharpe(equity_curve, freq_per_year=freq_per_year)


def _return_objective(equity_curve: np.ndarray, trades: list,
                      freq_per_year: int = 252) -> float:
    """Maximize percentage return."""
    if len(equity_curve) < 2 or equity_curve[0] == 0:
        return 0.0
    return (equity_curve[-1] - equity_curve[0]) / equity_curve[0]


def _calmar_objective(equity_curve: np.ndarray, trades: list,
                      freq_per_year: int = 252) -> float:
    """Maximize Calmar ratio (return / max drawdown). Penalizes large drawdowns."""
    if len(equity_curve) < 2 or equity_curve[0] == 0:
        return 0.0
    ret = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
    peak = np.maximum.accumulate(equity_curve)
    dd = (peak - equity_curve) / np.where(peak > 0, peak, 1.0)
    max_dd = np.max(dd)
    if max_dd <= 0:
        return ret * 100  # No drawdown — return is the score
    return ret / max_dd


def _sortino_objective(equity_curve: np.ndarray, trades: list,
                       freq_per_year: int = 252) -> float:
    """Maximize Sortino ratio (return / downside deviation)."""
    if len(equity_curve) < 2:
        return 0.0
    returns = np.diff(equity_curve) / equity_curve[:-1]
    returns = returns[np.isfinite(returns)]
    if len(returns) == 0:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0:
        return np.mean(returns) * np.sqrt(freq_per_year)
    downside_std = np.std(downside)
    if downside_std == 0:
        return 0.0
    return (np.mean(returns) / downside_std) * np.sqrt(freq_per_year)


OBJECTIVES: Dict[str, Callable] = {
    "sharpe": _sharpe_objective,
    "return": _return_objective,
    "calmar": _calmar_objective,
    "sortino": _sortino_objective,
}


# ---------------------------------------------------------------------------
# Parameter space sampling
# ---------------------------------------------------------------------------

def _suggest_param(trial: optuna.Trial, name: str, bounds) -> Any:
    """Suggest a parameter value from Optuna based on the bounds type.

    Bounds formats:
    - (low, high) with int values  -> int range
    - (low, high) with float values -> float range
    - [val1, val2, ...] -> categorical
    """
    if isinstance(bounds, list):
        return trial.suggest_categorical(name, bounds)
    low, high = bounds
    if isinstance(low, int) and isinstance(high, int):
        return trial.suggest_int(name, low, high)
    return trial.suggest_float(name, float(low), float(high))


# Constraints: list of (param_a, param_b, min_gap) tuples.
# If both params are present in a trial, require param_b - param_a >= min_gap.
# Used to prevent degenerate parameter combinations (e.g. fast_period ≈ slow_period).
PARAM_CONSTRAINTS = [
    ("fast_period", "slow_period", 15),
]


def _check_constraints(params: Dict[str, Any], prefix: str = "") -> bool:
    """Return True if all parameter constraints are satisfied."""
    for param_a, param_b, min_gap in PARAM_CONSTRAINTS:
        key_a = f"{prefix}{param_a}" if prefix else param_a
        key_b = f"{prefix}{param_b}" if prefix else param_b
        if key_a in params and key_b in params:
            if params[key_b] - params[key_a] < min_gap:
                return False
    return True


def _average_top_k_params(
    study: optuna.Study,
    param_space: Dict[str, Any],
    k: int,
) -> Dict[str, Any]:
    """Average the top K trials' parameters.

    Integer params are rounded, float params are averaged,
    categorical params use the mode (most frequent value).
    """
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value, reverse=True)
    top_k = completed[:min(k, len(completed))]

    if len(top_k) <= 1:
        return study.best_params

    averaged = {}
    for name, bounds in param_space.items():
        values = [t.params[name] for t in top_k if name in t.params]
        if not values:
            averaged[name] = study.best_params.get(name)
            continue

        if isinstance(bounds, list):
            # Categorical: mode
            from collections import Counter
            averaged[name] = Counter(values).most_common(1)[0][0]
        else:
            low, high = bounds
            avg = sum(values) / len(values)
            if isinstance(low, int) and isinstance(high, int):
                averaged[name] = round(avg)
            else:
                averaged[name] = avg

    return averaged


# ---------------------------------------------------------------------------
# Core optimization
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Result from a single optimization run."""
    best_params: Dict[str, Any]
    best_score: float
    objective_name: str
    n_trials: int
    all_trials: pd.DataFrame  # Every trial's params + score


@dataclass
class WalkForwardResult:
    """Result from walk-forward validation."""
    splits: List[Dict[str, Any]]  # Per-split results
    in_sample_mean: float         # Average IS objective
    out_of_sample_mean: float     # Average OOS objective (the real metric)
    degradation: float            # IS mean - OOS mean (overfitting signal)
    summary: pd.DataFrame         # DataFrame of all splits


def optimize(
    strategy_cls: Type,
    param_space: Dict[str, Any],
    df: pd.DataFrame,
    n_trials: int = 100,
    objective: str = "sharpe",
    starting_cash: float = 10_000,
    commission_bps: float = 5.0,
    slippage_bps: float = 2.0,
    symbol: str = "default",
    fixed_params: Optional[Dict[str, Any]] = None,
    n_jobs: int = 1,
    min_trades: int = 0,
    top_k_avg: int = 1,
) -> OptimizationResult:
    """Optimize strategy parameters using Bayesian search (Optuna TPE).

    Parameters
    ----------
    strategy_cls : type
        Strategy class to optimize.
    param_space : dict
        Parameter name -> bounds. Supports:
        - ``(low, high)`` for int/float ranges
        - ``[val1, val2, ...]`` for categorical choices
    df : pd.DataFrame
        OHLC data indexed by timestamp.
    n_trials : int
        Number of parameter combinations to try.
    objective : str
        What to maximize: "sharpe", "return", "calmar", or "sortino".
    starting_cash : float
        Initial portfolio cash.
    commission_bps : float
        Commission in basis points.
    slippage_bps : float
        Slippage in basis points.
    symbol : str
        Instrument symbol.
    fixed_params : dict, optional
        Parameters to pass to the strategy that are NOT optimized.

    Returns
    -------
    OptimizationResult
    """
    obj_fn = OBJECTIVES.get(objective)
    if obj_fn is None:
        raise ValueError(f"Unknown objective '{objective}'. Choose from: {list(OBJECTIVES)}")

    fixed = fixed_params or {}
    freq = infer_freq_per_year(df.index)

    def _objective(trial: optuna.Trial) -> float:
        params = {name: _suggest_param(trial, name, bounds)
                  for name, bounds in param_space.items()}
        if not _check_constraints(params):
            return -999.0
        params.update(fixed)

        try:
            strategy = strategy_cls(**params)
            bt = Backtester(df, strategy, starting_cash=starting_cash,
                            commission_bps=commission_bps, slippage_bps=slippage_bps,
                            symbol=symbol)
            equity_curve, trades = bt.run()
            score = obj_fn(equity_curve, trades, freq_per_year=freq)
            if not np.isfinite(score):
                return -999.0
            if min_trades > 0 and len(trades) < min_trades:
                score = score * (len(trades) / min_trades)
            return score
        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            return -999.0

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(_objective, n_trials=n_trials, n_jobs=n_jobs)

    # Build trials DataFrame
    rows = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            row = {**t.params, "score": t.value, "trial": t.number}
            rows.append(row)
    trials_df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

    best_params = (_average_top_k_params(study, param_space, top_k_avg)
                   if top_k_avg > 1 else study.best_params)

    return OptimizationResult(
        best_params=best_params,
        best_score=study.best_value,
        objective_name=objective,
        n_trials=n_trials,
        all_trials=trials_df,
    )


def walk_forward(
    strategy_cls: Type,
    param_space: Dict[str, Any],
    df: pd.DataFrame,
    n_splits: int = 4,
    train_ratio: float = 0.7,
    n_trials: int = 50,
    objective: str = "sharpe",
    starting_cash: float = 10_000,
    commission_bps: float = 5.0,
    slippage_bps: float = 2.0,
    symbol: str = "default",
    fixed_params: Optional[Dict[str, Any]] = None,
    n_jobs: int = 1,
    min_trades: int = 0,
    top_k_avg: int = 1,
    anchored: bool = False,
) -> WalkForwardResult:
    """Walk-forward optimization: the only honest way to evaluate parameter tuning.

    Splits the data into ``n_splits`` rolling windows. For each window:
    1. Optimize parameters on the training portion (in-sample).
    2. Evaluate the best parameters on the test portion (out-of-sample).

    The gap between in-sample and out-of-sample performance is the
    **degradation** — a direct measure of overfitting.

    Parameters
    ----------
    strategy_cls : type
        Strategy class to optimize.
    param_space : dict
        Parameter name -> bounds.
    df : pd.DataFrame
        Full OHLC dataset indexed by timestamp.
    n_splits : int
        Number of walk-forward windows.
    train_ratio : float
        Fraction of each window used for training (rest is test).
    n_trials : int
        Optuna trials per training window.
    objective : str
        Objective function name.
    starting_cash : float
        Initial cash for each split.
    commission_bps : float
        Commission in basis points.
    slippage_bps : float
        Slippage in basis points.
    symbol : str
        Instrument symbol.
    fixed_params : dict, optional
        Non-optimized parameters.

    Returns
    -------
    WalkForwardResult
    """
    obj_fn = OBJECTIVES.get(objective)
    if obj_fn is None:
        raise ValueError(f"Unknown objective '{objective}'. Choose from: {list(OBJECTIVES)}")

    freq = infer_freq_per_year(df.index)
    total_bars = len(df)
    window_size = total_bars // n_splits
    if window_size < 50:
        raise ValueError(f"Not enough data: {total_bars} bars / {n_splits} splits = "
                         f"{window_size} bars per window (need at least 50)")

    fixed = fixed_params or {}
    splits = []

    for split_idx in range(n_splits):
        window_start = split_idx * window_size
        window_end = min(window_start + window_size, total_bars)
        split_point = window_start + int((window_end - window_start) * train_ratio)

        train_start_idx = 0 if anchored else window_start
        train_df = df.iloc[train_start_idx:split_point]
        test_df = df.iloc[split_point:window_end]

        if len(train_df) < 30 or len(test_df) < 10:
            continue

        train_start = train_df.index[0]
        train_end = train_df.index[-1]
        test_start = test_df.index[0]
        test_end = test_df.index[-1]

        # 1. Optimize on training data
        opt_result = optimize(
            strategy_cls=strategy_cls, param_space=param_space, df=train_df,
            n_trials=n_trials, objective=objective, starting_cash=starting_cash,
            commission_bps=commission_bps, slippage_bps=slippage_bps,
            symbol=symbol, fixed_params=fixed_params, n_jobs=n_jobs,
            min_trades=min_trades, top_k_avg=top_k_avg,
        )

        # 2. Evaluate best params on test data (out-of-sample)
        best_params = {**opt_result.best_params, **fixed}
        strategy = strategy_cls(**best_params)
        bt = Backtester(test_df, strategy, starting_cash=starting_cash,
                        commission_bps=commission_bps, slippage_bps=slippage_bps,
                        symbol=symbol)
        eq, trades = bt.run()
        oos_score = obj_fn(eq, trades, freq_per_year=freq)

        split_result = {
            "split": split_idx,
            "train_start": str(train_start.date()),
            "train_end": str(train_end.date()),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "train_bars": len(train_df),
            "test_bars": len(test_df),
            "best_params": opt_result.best_params,
            "in_sample_score": round(opt_result.best_score, 4),
            "out_of_sample_score": round(oos_score, 4),
            "oos_return_pct": round((eq[-1] - starting_cash) / starting_cash * 100, 2),
            "oos_max_dd_pct": round(bt.max_drawdown * 100, 2),
            "oos_trades": len(trades),
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

    return WalkForwardResult(
        splits=splits,
        in_sample_mean=round(np.mean(is_scores), 4),
        out_of_sample_mean=round(np.mean(oos_scores), 4),
        degradation=round(np.mean(is_scores) - np.mean(oos_scores), 4),
        summary=summary_df,
    )
