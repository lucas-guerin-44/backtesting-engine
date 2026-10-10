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

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import optuna
import pandas as pd

from backtesting.backtest import Backtester
from backtesting.indicators import ema_array
from backtesting.statistics import (
    StatisticalReport,
    compute_sharpe,
    compute_statistical_report,
    passes_statistical_gate,
)
from backtesting.strategy import Strategy
from backtesting.types import Bar
from backtesting.vectorized import VectorizedBacktester
from backtesting.vectorized_signals import (
    trend_following_signals, mean_reversion_signals,
    momentum_signals, donchian_signals,
)
from utils import infer_freq_per_year

# Suppress Optuna's verbose trial logging (we log summaries ourselves)
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vectorized signal dispatch
# ---------------------------------------------------------------------------

_VECTORIZED_SIGNALS = {
    "TrendFollowingStrategy": trend_following_signals,
    "MeanReversionStrategy": mean_reversion_signals,
    "MomentumStrategy": momentum_signals,
    "DonchianBreakoutStrategy": donchian_signals,
}

# Parameters that belong to the backtester / risk-management layer, not the
# signal generator.  These are stripped before calling signal functions.
_NON_SIGNAL_PARAMS = {
    "risk_per_trade", "max_dd_halt", "cooldown_bars",
    "trend_filter_period", "use_trailing_stop", "allow_reentry",
}


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

def _statistical_report(
    run_backtest: Callable[[Dict[str, Any]], Tuple[np.ndarray, list]],
    params: Dict[str, Any],
    n_trials_tested: int,
    freq_per_year: int,
    seed: Optional[int],
) -> Optional[StatisticalReport]:
    """Run the Phase-3 statistical battery on a single configuration.

    Returns ``None`` (rather than raising) if the backtest or battery fails, so
    optimization results are never lost to a diagnostic.
    """
    try:
        equity_curve, trades = run_backtest(params)
    except Exception as e:
        logger.warning(f"Statistical report skipped (backtest failed): {e}")
        return None
    try:
        return compute_statistical_report(
            equity_curve=equity_curve,
            trades=trades,
            n_trials_tested=n_trials_tested,
            seed=seed,
            freq_per_year=freq_per_year,
        )
    except Exception as e:
        logger.warning(f"Statistical report failed: {e}")
        return None


@dataclass
class OptimizationResult:
    """Result from a single optimization run."""
    best_params: Dict[str, Any]
    best_score: float
    objective_name: str
    n_trials: int
    all_trials: pd.DataFrame  # Every trial's params + score
    seed: Optional[int]
    n_trials_tested: int = 0
    statistical_report: Optional[StatisticalReport] = None

    @property
    def bootstrap_ci(self):
        """Bootstrap Sharpe CI for the best configuration, if computed."""
        return self.statistical_report.bootstrap_ci if self.statistical_report else None

    @property
    def deflated_sharpe_p_value(self) -> Optional[float]:
        """Deflated Sharpe p-value for the best configuration, if computed."""
        dsr = self.statistical_report.deflated_sharpe if self.statistical_report else None
        return dsr.p_value if dsr is not None else None

    def passes_statistical_gate(self, alpha: float = 0.05) -> bool:
        """Whether the result clears the Phase-3 statistical gate."""
        return passes_statistical_gate(self.statistical_report, alpha=alpha)


@dataclass
class WalkForwardResult:
    """Result from walk-forward validation."""
    splits: List[Dict[str, Any]]  # Per-split results
    in_sample_mean: float         # Average IS objective
    out_of_sample_mean: float     # Average OOS objective (the real metric)
    degradation: float            # IS mean - OOS mean (overfitting signal)
    summary: pd.DataFrame         # DataFrame of all splits
    oos_index: pd.Index           # Chronological timestamps for stitched OOS curve
    oos_equity_curve: np.ndarray  # Compounded marked-to-market OOS equity
    oos_sharpe: float             # Sharpe of the complete stitched OOS curve
    oos_total_return_pct: float   # Return over the stitched OOS periods
    seed: Optional[int]
    n_trials_tested: int = 0
    statistical_report: Optional[StatisticalReport] = None

    @property
    def bootstrap_ci(self):
        """Bootstrap Sharpe CI for the stitched OOS curve, if computed."""
        return self.statistical_report.bootstrap_ci if self.statistical_report else None

    @property
    def deflated_sharpe_p_value(self) -> Optional[float]:
        """Deflated Sharpe p-value for the stitched OOS curve, if computed."""
        dsr = self.statistical_report.deflated_sharpe if self.statistical_report else None
        return dsr.p_value if dsr is not None else None

    def passes_statistical_gate(self, alpha: float = 0.05) -> bool:
        """Whether the OOS result clears the Phase-3 statistical gate."""
        return passes_statistical_gate(self.statistical_report, alpha=alpha)


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
    engine: str = "event",
    seed: Optional[int] = None,
    freq_per_year: Optional[int] = None,
    compute_statistics: bool = True,
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
    seed : int, optional
        Random seed for Optuna's sampler. Use ``n_jobs=1`` for deterministic
        trial ordering and fully reproducible results.
    freq_per_year : int, optional
        Annualization periods. If omitted, estimated from observed timestamps;
        set explicitly for short samples or a selected market-calendar convention.
    compute_statistics : bool
        Attach a Phase-3 ``StatisticalReport`` for the best configuration using
        ``n_trials`` as the multiple-testing count. Set False for internal
        nested calls where the caller computes the report itself.

    Returns
    -------
    OptimizationResult
    """
    obj_fn = OBJECTIVES.get(objective)
    if obj_fn is None:
        raise ValueError(f"Unknown objective '{objective}'. Choose from: {list(OBJECTIVES)}")

    if engine not in ("event", "vectorized"):
        raise ValueError(f"Unknown engine '{engine}'. Choose 'event' or 'vectorized'.")
    if seed is not None and n_jobs != 1:
        raise ValueError("Seeded optimization requires n_jobs=1 for reproducibility")
    if freq_per_year is not None and freq_per_year <= 0:
        raise ValueError("freq_per_year must be > 0")

    fixed = fixed_params or {}
    freq = int(freq_per_year) if freq_per_year is not None else infer_freq_per_year(df.index)

    # Pre-extract OHLC arrays once for vectorized engine
    if engine == "vectorized":
        strat_name = strategy_cls.__name__
        sig_fn = _VECTORIZED_SIGNALS.get(strat_name)
        if sig_fn is None:
            raise ValueError(
                f"No vectorized signal function for '{strat_name}'. "
                f"Supported: {list(_VECTORIZED_SIGNALS)}"
            )
        _col = {c.lower(): c for c in df.columns}
        _ohlc_open = df[_col["open"]].to_numpy(dtype=np.float64)
        _ohlc_high = df[_col["high"]].to_numpy(dtype=np.float64)
        _ohlc_low = df[_col["low"]].to_numpy(dtype=np.float64)
        _ohlc_close = df[_col["close"]].to_numpy(dtype=np.float64)
        # Discover which kwargs the signal function actually accepts
        _sig_params = set(inspect.signature(sig_fn).parameters.keys()) - {
            "open", "high", "low", "close",
        }

    def _run_backtest(params: Dict[str, Any]):
        """Run one backtest for ``params``; return (equity_curve, trades)."""
        if engine == "vectorized":
            # Filter params to only those the signal function accepts
            sig_kwargs = {k: v for k, v in params.items()
                          if k in _sig_params}
            entries, sides, stops, tps = sig_fn(
                _ohlc_open, _ohlc_high, _ohlc_low, _ohlc_close,
                **sig_kwargs,
            )

            # Post-processing: apply trend filter if requested
            tfp = params.get("trend_filter_period", 0)
            if tfp and tfp > 0:
                trend_ema = ema_array(_ohlc_close, int(tfp))
                # Mask out longs where close < ema, shorts where close > ema
                long_mask = (_ohlc_close < trend_ema) & (sides == 1)
                short_mask = (_ohlc_close > trend_ema) & (sides == -1)
                block = long_mask | short_mask
                entries = entries & ~block
                sides = np.where(block, 0, sides)

            bt = VectorizedBacktester(
                _ohlc_open, _ohlc_high, _ohlc_low, _ohlc_close,
                starting_cash=starting_cash,
                commission_bps=commission_bps,
                slippage_bps=slippage_bps,
            )
            return bt.run(
                entries, sides, stops, tps,
                risk_per_trade=params.get("risk_per_trade", 0.02),
                max_dd_halt=params.get("max_dd_halt", 0.15),
                cooldown_bars=params.get("cooldown_bars", 10),
            )

        strategy = strategy_cls(**params)
        bt = Backtester(df, strategy, starting_cash=starting_cash,
                        commission_bps=commission_bps, slippage_bps=slippage_bps,
                        symbol=symbol, freq_per_year=freq)
        return bt.run()

    def _evaluate_params(params: Dict[str, Any], trial_number=None) -> float:
        try:
            equity_curve, trades = _run_backtest(params)
            score = obj_fn(equity_curve, trades, freq_per_year=freq)
            if not np.isfinite(score):
                return -999.0
            if min_trades > 0 and len(trades) < min_trades:
                score = score * (len(trades) / min_trades)
            return score
        except Exception as e:
            label = f"Trial {trial_number}" if trial_number is not None else "Candidate params"
            logger.warning(f"{label} failed: {e}")
            return -999.0

    def _objective(trial: optuna.Trial) -> float:
        params = {name: _suggest_param(trial, name, bounds)
                  for name, bounds in param_space.items()}
        if not _check_constraints(params):
            return -999.0
        params.update(fixed)
        return _evaluate_params(params, trial.number)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
    )
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
    best_score = study.best_value
    if top_k_avg > 1:
        # The averaged parameter set is a new candidate, not the best trial.
        # Re-score it so structured results always pair matching params/score.
        best_score = _evaluate_params({**best_params, **fixed})

    report = None
    if compute_statistics:
        report = _statistical_report(
            _run_backtest, {**best_params, **fixed},
            n_trials_tested=n_trials, freq_per_year=freq, seed=seed,
        )

    return OptimizationResult(
        best_params=best_params,
        best_score=best_score,
        objective_name=objective,
        n_trials=n_trials,
        all_trials=trials_df,
        seed=seed,
        n_trials_tested=n_trials,
        statistical_report=report,
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
    engine: str = "event",
    seed: Optional[int] = None,
    freq_per_year: Optional[int] = None,
    warmup_bars: Optional[int] = None,
    compute_statistics: bool = True,
) -> WalkForwardResult:
    """Walk-forward optimization: the only honest way to evaluate parameter tuning.

    Reserves the first ``train_ratio`` of the data for initial training, then
    divides the remaining history into contiguous OOS test periods. Each fold
    optimizes on past data only; OOS equity is compounded across test periods.

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
        Fraction of the full dataset reserved for initial training. Remaining
        bars are divided into contiguous OOS test periods.
    n_trials : int
        Optuna trials per training window.
    objective : str
        Objective function name.
    starting_cash : float
        Initial cash for each split.
    seed : int, optional
        Base sampler seed. Each fold uses ``seed + fold_index``. Seeded runs
        require ``n_jobs=1`` for deterministic trial ordering.
    commission_bps : float
        Commission in basis points.
    slippage_bps : float
        Slippage in basis points.
    symbol : str
        Instrument symbol.
    fixed_params : dict, optional
        Non-optimized parameters.
    freq_per_year : int, optional
        Annualization periods shared by all training and OOS folds. If omitted,
        estimated once from the full dataset's observed timestamps.
    warmup_bars : int, optional
        Historical bars used to prime each OOS fold without scoring them or
        allowing entries. If omitted, uses the strategy's ``warmup_bars`` value.
        Custom event-driven strategies should override ``warmup_bar(bar)`` to
        update indicator state without placing trades.
    compute_statistics : bool
        Attach a Phase-3 ``StatisticalReport`` for the stitched OOS equity curve.
        The multiple-testing count is ``n_trials * n_splits`` (every parameter
        combination evaluated across every fold).

    Returns
    -------
    WalkForwardResult
    """
    obj_fn = OBJECTIVES.get(objective)
    if obj_fn is None:
        raise ValueError(f"Unknown objective '{objective}'. Choose from: {list(OBJECTIVES)}")

    freq = infer_freq_per_year(df.index)
    if freq_per_year is not None:
        if freq_per_year <= 0:
            raise ValueError("freq_per_year must be > 0")
        freq = int(freq_per_year)
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    if warmup_bars is not None and warmup_bars < 0:
        raise ValueError("warmup_bars must be >= 0")

    total_bars = len(df)
    initial_train_bars = int(total_bars * train_ratio)
    oos_bars = total_bars - initial_train_bars
    test_size = oos_bars // n_splits if n_splits > 0 else 0
    if initial_train_bars < 50 or test_size < 10:
        raise ValueError(
            f"Not enough data: {initial_train_bars} initial training bars and "
            f"{test_size} bars per OOS split (need at least 50 and 10)"
        )

    fixed = fixed_params or {}
    splits = []
    oos_curves = [np.array([starting_cash], dtype=np.float64)]
    oos_timestamps = [df.index[initial_train_bars - 1]]
    oos_cash = starting_cash

    for split_idx in range(n_splits):
        test_start_idx = initial_train_bars + split_idx * test_size
        test_end_idx = total_bars if split_idx == n_splits - 1 else test_start_idx + test_size
        train_end_idx = test_start_idx
        if anchored:
            train_start_idx = 0
        else:
            train_start_idx = max(0, train_end_idx - initial_train_bars)

        train_df = df.iloc[train_start_idx:train_end_idx]
        test_df = df.iloc[test_start_idx:test_end_idx]

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
            seed=(seed + split_idx if seed is not None else None),
            min_trades=min_trades, top_k_avg=top_k_avg, engine=engine,
            freq_per_year=freq, compute_statistics=False,
        )

        # 2. Evaluate best params on test data (out-of-sample)
        best_params = {**opt_result.best_params, **fixed}
        oos_starting_cash = oos_cash
        warmup_strategy = strategy_cls(**best_params)
        requested_warmup = (
            int(warmup_bars) if warmup_bars is not None
            else max(0, int(getattr(warmup_strategy, "warmup_bars", 0)))
        )
        if (
            engine == "event"
            and requested_warmup > 0
            and type(warmup_strategy).warmup_bar is Strategy.warmup_bar
        ):
            raise ValueError(
                f"{strategy_cls.__name__} requests warm-up bars but does not "
                "override Strategy.warmup_bar()"
            )
        warmup_start_idx = max(train_start_idx, test_start_idx - requested_warmup)
        actual_warmup = test_start_idx - warmup_start_idx
        context_test_df = df.iloc[warmup_start_idx:test_end_idx]

        if engine == "vectorized":
            sig_fn = _VECTORIZED_SIGNALS[strategy_cls.__name__]
            _sig_params = set(inspect.signature(sig_fn).parameters.keys()) - {
                "open", "high", "low", "close",
            }
            t_col = {c.lower(): c for c in context_test_df.columns}
            t_open = context_test_df[t_col["open"]].to_numpy(dtype=np.float64)
            t_high = context_test_df[t_col["high"]].to_numpy(dtype=np.float64)
            t_low = context_test_df[t_col["low"]].to_numpy(dtype=np.float64)
            t_close = context_test_df[t_col["close"]].to_numpy(dtype=np.float64)

            sig_kwargs = {k: v for k, v in best_params.items()
                          if k in _sig_params}
            entries, sides, stops, tps = sig_fn(
                t_open, t_high, t_low, t_close, **sig_kwargs,
            )
            # Context contributes indicator history only; it must not open
            # positions or affect OOS equity.
            entries[:actual_warmup] = False
            sides[:actual_warmup] = 0

            tfp = best_params.get("trend_filter_period", 0)
            if tfp and tfp > 0:
                trend_ema = ema_array(t_close, int(tfp))
                long_mask = (t_close < trend_ema) & (sides == 1)
                short_mask = (t_close > trend_ema) & (sides == -1)
                block = long_mask | short_mask
                entries = entries & ~block
                sides = np.where(block, 0, sides)

            bt = VectorizedBacktester(
                t_open, t_high, t_low, t_close,
                starting_cash=oos_starting_cash,
                commission_bps=commission_bps,
                slippage_bps=slippage_bps,
            )
            combined_eq, trades = bt.run(
                entries, sides, stops, tps,
                risk_per_trade=best_params.get("risk_per_trade", 0.02),
                max_dd_halt=best_params.get("max_dd_halt", 0.15),
                cooldown_bars=best_params.get("cooldown_bars", 10),
            )
            eq = combined_eq[actual_warmup:]
        else:
            strategy = warmup_strategy
            if actual_warmup:
                context_open = context_test_df["open"].to_numpy(dtype=np.float64)[:actual_warmup]
                context_high = context_test_df["high"].to_numpy(dtype=np.float64)[:actual_warmup]
                context_low = context_test_df["low"].to_numpy(dtype=np.float64)[:actual_warmup]
                context_close = context_test_df["close"].to_numpy(dtype=np.float64)[:actual_warmup]
                warmup_method = strategy.warmup_bar
                for i in range(actual_warmup):
                    warmup_method(Bar(
                        df.index[warmup_start_idx + i], context_open[i],
                        context_high[i], context_low[i], context_close[i],
                    ))
            bt = Backtester(test_df, strategy, starting_cash=oos_starting_cash,
                            commission_bps=commission_bps, slippage_bps=slippage_bps,
                            symbol=symbol, freq_per_year=freq)
            eq, trades = bt.run(liquidate_at_end=True)

        oos_score = obj_fn(eq, trades, freq_per_year=freq)
        oos_cash = float(eq[-1]) if len(eq) else oos_cash
        oos_curves.append(np.asarray(eq, dtype=np.float64))
        oos_timestamps.extend(test_df.index.tolist())

        split_result = {
            "split": split_idx,
            "train_start": str(train_start.date()),
            "train_end": str(train_end.date()),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "train_bars": len(train_df),
            "test_bars": len(test_df),
            "warmup_bars": actual_warmup,
            "best_params": opt_result.best_params,
            "in_sample_score": round(opt_result.best_score, 4),
            "out_of_sample_score": round(oos_score, 4),
            "oos_return_pct": round((eq[-1] - oos_starting_cash) / oos_starting_cash * 100, 2),
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
        "warmup_bars": s["warmup_bars"],
        "IS_score": s["in_sample_score"],
        "OOS_score": s["out_of_sample_score"],
        "OOS_return_pct": s["oos_return_pct"],
        "OOS_max_dd_pct": s["oos_max_dd_pct"],
        "OOS_trades": s["oos_trades"],
    } for s in splits])

    stitched_equity = np.concatenate(oos_curves)
    stitched_index = pd.Index(oos_timestamps)
    stitched_return = (
        (oos_cash - starting_cash) / starting_cash * 100.0
        if starting_cash > 0 else 0.0
    )

    n_trials_tested = n_trials * len(splits)
    report = None
    if compute_statistics:
        try:
            report = compute_statistical_report(
                equity_curve=stitched_equity,
                trades=[],
                n_trials_tested=n_trials_tested,
                seed=seed,
                freq_per_year=freq,
            )
        except Exception as e:
            logger.warning(f"Statistical report failed: {e}")

    return WalkForwardResult(
        splits=splits,
        in_sample_mean=round(np.mean(is_scores), 4),
        out_of_sample_mean=round(np.mean(oos_scores), 4),
        degradation=round(np.mean(is_scores) - np.mean(oos_scores), 4),
        summary=summary_df,
        oos_index=stitched_index,
        oos_equity_curve=stitched_equity,
        oos_sharpe=round(compute_sharpe(stitched_equity, freq_per_year=freq), 4),
        oos_total_return_pct=round(stitched_return, 2),
        seed=seed,
        n_trials_tested=n_trials_tested,
        statistical_report=report,
    )
