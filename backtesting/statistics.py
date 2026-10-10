"""Statistical significance testing for backtest results.

Three tests, each answering a different question:

- ``bootstrap_sharpe_ci``:  "Is the Sharpe statistically different from zero?"
- ``permutation_test``:     "Is return drift unusual under a blocked sign-flip null?"
- ``deflated_sharpe_ratio``:"Is this Sharpe still significant after testing N param combos?"

The Deflated Sharpe Ratio is based on Bailey & Lopez de Prado (2014),
"The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
Overfitting, and Non-Normality."
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Sharpe ratio
# ---------------------------------------------------------------------------

def compute_sharpe(equity_curve, risk_free: float = 0.0, annualize: bool = True, freq_per_year: int = 252) -> float:
    """Compute Sharpe ratio from an equity curve.

    Parameters
    ----------
    equity_curve : array-like
        Sequence of portfolio equity values over time.
    risk_free : float
        Annual risk-free rate (default 0).
    annualize : bool
        Whether to annualize the ratio.
    freq_per_year : int
        Number of observation periods per year (252 for daily bars).

    Returns
    -------
    float
        The Sharpe ratio, or 0.0 if it cannot be computed.
    """
    equity_curve = np.array(equity_curve, dtype=float)
    if len(equity_curve) < 3 or freq_per_year <= 0:
        return 0.0

    valid_idx = equity_curve[:-1] > 1e-8
    if not np.any(valid_idx):
        return 0.0

    returns = np.diff(equity_curve)[valid_idx] / equity_curve[:-1][valid_idx]
    excess_returns = returns - risk_free / freq_per_year

    std = np.std(excess_returns, ddof=1)
    if not np.isfinite(std) or std == 0:
        return 0.0

    sharpe = np.mean(excess_returns) / std
    if annualize:
        sharpe *= np.sqrt(freq_per_year)

    return sharpe


def period_returns(equity_curve) -> np.ndarray:
    """Per-period simple returns from an equity curve.

    Uses the same valid-step filter as the bootstrap/permutation tests so the
    DSR moments and the resampled statistics describe the same return series.
    """
    equity_curve = np.asarray(equity_curve, dtype=np.float64)
    if len(equity_curve) < 2:
        return np.array([], dtype=np.float64)
    valid = equity_curve[:-1] > 1e-8
    if not np.any(valid):
        return np.array([], dtype=np.float64)
    return np.diff(equity_curve)[valid] / equity_curve[:-1][valid]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SharpeCI:
    """Bootstrap confidence interval on Sharpe ratio."""
    observed_sharpe: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_bootstrap: int
    significant: bool  # True if CI excludes zero

    def __str__(self) -> str:
        sig = "SIGNIFICANT" if self.significant else "not significant"
        return (f"Sharpe: {self.observed_sharpe:.4f}  "
                f"CI [{self.ci_lower:.4f}, {self.ci_upper:.4f}] "
                f"({self.confidence_level:.0%}) — {sig}")


@dataclass
class PermutationTestResult:
    """Monte Carlo permutation test result."""
    observed_sharpe: float
    p_value: float
    percentile: float  # What percentile the real Sharpe falls at
    n_permutations: int
    significant: bool  # True if p_value < alpha

    def __str__(self) -> str:
        sig = "SIGNIFICANT" if self.significant else "not significant"
        return (f"Sharpe: {self.observed_sharpe:.4f}  "
                f"p-value: {self.p_value:.4f}  "
                f"percentile: {self.percentile:.1f}th — {sig}")


@dataclass
class DeflatedSharpeResult:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)."""
    observed_sharpe: float
    deflated_sharpe: float
    p_value: float
    n_trials_tested: int
    significant: bool

    def __str__(self) -> str:
        sig = "SURVIVES deflation" if self.significant else "DOES NOT survive deflation"
        return (f"Observed SR: {self.observed_sharpe:.4f}  "
                f"Deflated SR: {self.deflated_sharpe:.4f}  "
                f"(p={self.p_value:.4f}, {self.n_trials_tested} trials) — {sig}")


@dataclass
class StatisticalReport:
    """Combined statistical analysis of a backtest."""
    bootstrap_ci: SharpeCI
    permutation_test: PermutationTestResult
    deflated_sharpe: Optional[DeflatedSharpeResult]
    skewness: float = 0.0
    kurtosis: float = 3.0

    def __str__(self) -> str:
        lines = [
            "Statistical Significance Report",
            "=" * 50,
            f"  Bootstrap CI:     {self.bootstrap_ci}",
            f"  Permutation test: {self.permutation_test}",
        ]
        if self.deflated_sharpe:
            lines.append(f"  Deflated Sharpe:  {self.deflated_sharpe}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

def bootstrap_sharpe_ci(
    equity_curve: np.ndarray,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    risk_free: float = 0.0,
    freq_per_year: int = 252,
    seed: Optional[int] = None,
) -> SharpeCI:
    """Bootstrap confidence interval on the annualized Sharpe ratio.

    Resamples returns with replacement ``n_bootstrap`` times (vectorized,
    no Python loop), computes Sharpe on each sample, and reports the CI.

    Parameters
    ----------
    equity_curve : np.ndarray
        Array of portfolio equity values.
    n_bootstrap : int
        Number of bootstrap resamples.
    confidence : float
        Confidence level (e.g., 0.95 for 95% CI).
    risk_free : float
        Annual risk-free rate.
    freq_per_year : int
        Observations per year (252 for daily).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    SharpeCI
    """
    equity_curve = np.asarray(equity_curve, dtype=np.float64)
    observed = compute_sharpe(equity_curve, risk_free=risk_free,
                              freq_per_year=freq_per_year)

    if freq_per_year <= 0:
        return SharpeCI(observed, 0.0, 0.0, confidence, n_bootstrap, False)

    if len(equity_curve) < 3:
        return SharpeCI(observed, observed, observed, confidence, n_bootstrap, False)

    # Compute returns
    valid = equity_curve[:-1] > 1e-8
    if not np.any(valid):
        return SharpeCI(observed, 0.0, 0.0, confidence, n_bootstrap, False)

    returns = np.diff(equity_curve)[valid] / equity_curve[:-1][valid]
    n_returns = len(returns)

    if n_returns < 2:
        return SharpeCI(observed, observed, observed, confidence, n_bootstrap, False)

    # Vectorized bootstrap: sample all at once as a (n_bootstrap, n_returns) matrix
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n_returns, size=(n_bootstrap, n_returns))
    samples = returns[indices]

    # Compute Sharpe for each bootstrap sample
    rf_per_period = risk_free / freq_per_year
    excess = samples - rf_per_period
    means = excess.mean(axis=1)
    stds = excess.std(axis=1, ddof=1)

    # Avoid division by zero
    valid_mask = stds > 0
    sharpes = np.full(n_bootstrap, 0.0)
    sharpes[valid_mask] = (means[valid_mask] / stds[valid_mask]) * np.sqrt(freq_per_year)

    # Percentile CI
    alpha = 1 - confidence
    ci_lower = float(np.percentile(sharpes, 100 * alpha / 2))
    ci_upper = float(np.percentile(sharpes, 100 * (1 - alpha / 2)))

    # Significant if CI excludes zero
    significant = ci_lower > 0 or ci_upper < 0

    return SharpeCI(observed, ci_lower, ci_upper, confidence, n_bootstrap, significant)


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def permutation_test(
    equity_curve: np.ndarray,
    trades: list,
    n_permutations: int = 10_000,
    alpha: float = 0.05,
    risk_free: float = 0.0,
    freq_per_year: int = 252,
    seed: Optional[int] = None,
    block_size: int = 5,
) -> PermutationTestResult:
    """Sign-flip test for whether bar returns have positive/negative drift.

    The test operates on equity-curve returns so null and observed Sharpe
    ratios have the same units. Excess returns are centered under the null,
    then flipped in short contiguous blocks to retain some local dependence.
    ``trades`` remains accepted for API compatibility but is not used because
    trade PnLs cannot be compared directly with a bar-level Sharpe ratio.

    Parameters
    ----------
    equity_curve : np.ndarray
        Array of portfolio equity values.
    trades : list
        List of Trade objects (each must have a ``.pnl`` attribute).
    n_permutations : int
        Number of random permutations.
    alpha : float
        Significance level (default 0.05).
    seed : int, optional
        Random seed for reproducibility.
    block_size : int
        Number of adjacent returns sharing a sign flip.

    Returns
    -------
    PermutationTestResult
    """
    equity_curve = np.asarray(equity_curve, dtype=np.float64)
    observed = compute_sharpe(equity_curve, risk_free=risk_free,
                              freq_per_year=freq_per_year)

    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")

    valid = equity_curve[:-1] > 1e-8
    returns = np.diff(equity_curve)[valid] / equity_curve[:-1][valid]
    if len(returns) < 2 or freq_per_year <= 0:
        return PermutationTestResult(observed, 1.0, 0.0, n_permutations, False)

    excess = returns - risk_free / freq_per_year
    centered = excess - excess.mean()
    rng = np.random.default_rng(seed)
    n_blocks = (len(excess) + block_size - 1) // block_size
    block_signs = rng.choice([-1.0, 1.0], size=(n_permutations, n_blocks))
    signs = np.repeat(block_signs, block_size, axis=1)[:, :len(excess)]
    null_returns = centered[None, :] * signs
    means = null_returns.mean(axis=1)
    stds = null_returns.std(axis=1, ddof=1)
    null_sharpes = np.zeros(n_permutations)
    valid_std = stds > 0
    null_sharpes[valid_std] = (
        means[valid_std] / stds[valid_std] * np.sqrt(freq_per_year)
    )

    # p-value: fraction of null Sharpes >= observed
    if observed >= 0:
        extreme = np.count_nonzero(null_sharpes >= observed)
    else:
        extreme = np.count_nonzero(null_sharpes <= observed)
    p_value = float((extreme + 1) / (n_permutations + 1))

    percentile = float(np.mean(null_sharpes <= observed) * 100)

    return PermutationTestResult(
        observed, p_value, percentile, n_permutations,
        significant=p_value < alpha,
    )


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------

_EULER_MASCHERONI = 0.5772156649015329

def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    all_trial_sharpes: Optional[np.ndarray] = None,
    freq_per_year: float = 1.0,
) -> DeflatedSharpeResult:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    Adjusts the observed Sharpe ratio for multiple hypothesis testing.
    When you test N parameter combinations, the best one's Sharpe is
    inflated by selection bias. The DSR corrects for this.

    Parameters
    ----------
    observed_sharpe : float
        The best Sharpe ratio found.
    n_trials : int
        Number of parameter combinations tested.
    n_observations : int
        Number of return observations in the backtest.
    skewness : float
        Skewness of returns (0.0 for normal).
    kurtosis : float
        Kurtosis of returns (3.0 for normal).
    all_trial_sharpes : np.ndarray, optional
        Annualized Sharpe ratios from all trials. If provided, their variance
        is converted to per-period units before applying the DSR formula.
    freq_per_year : float
        Observations per year used to annualize the input Sharpe ratios.

    Returns
    -------
    DeflatedSharpeResult
    """
    if n_trials <= 0 or n_observations <= 1 or freq_per_year <= 0:
        return DeflatedSharpeResult(observed_sharpe, observed_sharpe, 0.5, max(n_trials, 1), False)

    annualization = np.sqrt(freq_per_year)
    observed_period_sharpe = observed_sharpe / annualization

    # Variance of the Sharpe ratio estimator
    if all_trial_sharpes is not None and len(all_trial_sharpes) > 1:
        period_sharpes = np.asarray(all_trial_sharpes, dtype=np.float64) / annualization
        sr_var = float(np.var(period_sharpes, ddof=1))
    else:
        # Asymptotic formula: Var(SR) ≈ (1 - skew*SR + (kurt-1)/4 * SR²) / T
        sr_var = (1.0 - skewness * observed_period_sharpe +
                  (kurtosis - 1) / 4.0 * observed_period_sharpe ** 2) / n_observations

    sr_std = np.sqrt(max(sr_var, 1e-12))

    # Expected maximum Sharpe under the null (i.i.d. trials)
    # E[max] ≈ sr_std * ((1-γ) * Φ⁻¹(1 - 1/N) + γ * Φ⁻¹(1 - 1/(N*e)))
    if n_trials == 1:
        expected_max_sr = 0.0
    else:
        z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        expected_max_sr = sr_std * (
            (1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2
        )

    # Compute in per-period units, then return the deflated ratio in the same
    # annualized units as observed_sharpe.
    deflated_period = observed_period_sharpe - expected_max_sr
    deflated = deflated_period * annualization

    # p-value: probability of observing this SR by chance
    # P(SR* > observed | null) using the SR distribution
    if sr_std > 0:
        z_score = deflated_period / sr_std
        p_value = 1.0 - float(stats.norm.cdf(z_score))
    else:
        p_value = 0.5

    return DeflatedSharpeResult(
        observed_sharpe=observed_sharpe,
        deflated_sharpe=deflated,
        p_value=p_value,
        n_trials_tested=n_trials,
        significant=p_value < 0.05,
    )


# ---------------------------------------------------------------------------
# Combined report
# ---------------------------------------------------------------------------

def compute_statistical_report(
    equity_curve: np.ndarray,
    trades: list,
    n_trials_tested: int,
    returns: Optional[np.ndarray] = None,
    n_bootstrap: int = 10_000,
    n_permutations: int = 5_000,
    seed: Optional[int] = None,
    freq_per_year: int = 252,
    risk_free: float = 0.0,
) -> StatisticalReport:
    """Run all three statistical tests and return a combined report.

    Parameters
    ----------
    equity_curve : np.ndarray
        Array of portfolio equity values.
    trades : list
        List of Trade objects.
    n_trials_tested : int
        Number of parameter combinations actually evaluated during development
        (for the DSR). Required: callers must state how many trials they ran
        rather than silently defaulting to a single trial.
    returns : np.ndarray, optional
        Per-period return series aligned with ``equity_curve``. When omitted,
        it is derived from the equity curve. Its skewness and kurtosis feed the
        DSR variance so the adjustment does not assume normality.
    n_bootstrap : int
        Bootstrap resamples for CI.
    n_permutations : int
        Permutations for the permutation test.
    seed : int, optional
        Random seed.
    freq_per_year : int
        Observations per year used to annualize the Sharpe statistics.
    risk_free : float
        Annual risk-free rate applied to each return series.

    Returns
    -------
    StatisticalReport
    """
    if n_trials_tested is None or n_trials_tested < 1:
        raise ValueError("n_trials_tested must be a positive integer")

    equity_curve = np.asarray(equity_curve, dtype=np.float64)
    if freq_per_year <= 0:
        raise ValueError("freq_per_year must be > 0")

    ci = bootstrap_sharpe_ci(
        equity_curve, n_bootstrap=n_bootstrap, seed=seed,
        freq_per_year=freq_per_year, risk_free=risk_free,
    )

    perm = permutation_test(
        equity_curve, trades, n_permutations=n_permutations, seed=seed,
        freq_per_year=freq_per_year, risk_free=risk_free,
    )

    ret = period_returns(equity_curve) if returns is None else np.asarray(returns, dtype=np.float64)
    if len(ret) > 3 and np.std(ret) > 0:
        skewness = float(stats.skew(ret))
        kurtosis = float(stats.kurtosis(ret, fisher=False))
    else:
        skewness, kurtosis = 0.0, 3.0

    dsr = None
    if n_trials_tested > 1 and len(equity_curve) > 1:
        observed = compute_sharpe(
            equity_curve, risk_free=risk_free, freq_per_year=freq_per_year,
        )
        dsr = deflated_sharpe_ratio(
            observed_sharpe=observed,
            n_trials=n_trials_tested,
            n_observations=len(equity_curve) - 1,
            skewness=skewness,
            kurtosis=kurtosis,
            freq_per_year=freq_per_year,
        )

    return StatisticalReport(
        bootstrap_ci=ci, permutation_test=perm, deflated_sharpe=dsr,
        skewness=skewness, kurtosis=kurtosis,
    )


def passes_statistical_gate(report: Optional[StatisticalReport], alpha: float = 0.05) -> bool:
    """Phase-3 statistical gate from ``docs/WORKFLOW.md``.

    Passes when the bootstrap 95% CI excludes zero, the permutation p-value is
    ``<= alpha``, and the deflated Sharpe p-value is ``<= alpha`` (Bailey &
    Lopez de Prado 2014). A missing DSR (e.g. ``n_trials_tested == 1``) fails.
    """
    if report is None:
        return False
    if not report.bootstrap_ci.significant:
        return False
    if report.permutation_test.p_value > alpha:
        return False
    dsr = report.deflated_sharpe
    if dsr is None or dsr.p_value > alpha:
        return False
    return True


# ---------------------------------------------------------------------------
# Convenience: compute_stats (replaces per-experiment metric functions)
# ---------------------------------------------------------------------------

def compute_stats(
    equity_curve,
    trades: list,
    freq_per_year: int = 252,
    risk_free: float = 0.0,
) -> dict:
    """Compute a standard set of backtest statistics from an equity curve.

    Returns a dict with keys:
        sharpe, sortino, calmar, max_drawdown, cagr, total_return_pct,
        win_rate, profit_factor, avg_win, avg_loss, total_trades,
        bars_in_market, avg_bars_held

    Parameters
    ----------
    equity_curve : array-like
        Sequence of portfolio equity values over time.
    trades : list
        List of Trade objects (each must have .pnl, .bars_held attributes).
    freq_per_year : int
        Number of observation periods per year (252 for daily).
    risk_free : float
        Annual risk-free rate (default 0).
    """
    eq = np.asarray(equity_curve, dtype=np.float64)

    if len(eq) == 0 or not np.any(eq > 1e-8):
        return {k: 0.0 for k in [
            "sharpe", "sortino", "calmar", "max_drawdown", "cagr",
            "total_return_pct", "win_rate", "profit_factor",
            "avg_win", "avg_loss", "total_trades", "bars_in_market",
            "avg_bars_held",
        ]}

    # Sharpe (reuse compute_sharpe)
    sharpe = compute_sharpe(eq, risk_free=risk_free, freq_per_year=freq_per_year)

    # Returns (for Sortino, CAGR, etc.)
    prior_equity = eq[:-1]
    valid_prior = prior_equity > 1e-8
    returns = np.diff(eq)[valid_prior] / prior_equity[valid_prior]
    if risk_free > 0:
        returns = returns - risk_free / freq_per_year

    # Sortino
    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 0.0
    sortino = (np.mean(returns) / downside_std * np.sqrt(freq_per_year)) if downside_std > 0 else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.where(peak > 0, peak, 1.0)
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

    # CAGR
    n_years = max(len(eq) - 1, 0) / freq_per_year
    if n_years > 0 and eq[0] > 0:
        cagr = -1.0 if eq[-1] <= 0 else (eq[-1] / eq[0]) ** (1.0 / n_years) - 1.0
    else:
        cagr = 0.0

    # Calmar
    calmar = cagr / max_dd if max_dd > 0 else 0.0

    # Total return
    total_return_pct = (eq[-1] - eq[0]) / eq[0] * 100.0 if eq[0] > 0 else 0.0

    # Trade-level stats
    pnls = np.array([t.pnl for t in trades if t.pnl is not None], dtype=np.float64)
    n_trades = len(pnls)

    if n_trades > 0:
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        win_rate = len(wins) / n_trades * 100.0
        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_win = 0.0
        avg_loss = 0.0

    bars_held = [t.bars_held for t in trades if t.bars_held is not None]
    avg_bars_held = float(np.mean(bars_held)) if bars_held else 0.0

    return {
        "sharpe": round(float(sharpe), 4),
        "sortino": round(float(sortino), 4),
        "calmar": round(float(calmar), 4),
        "max_drawdown": round(max_dd * 100, 2),
        "cagr": round(cagr * 100, 2),
        "total_return_pct": round(total_return_pct, 2),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "total_trades": n_trades,
        "bars_in_market": sum(bars_held) if bars_held else 0,
        "avg_bars_held": round(avg_bars_held, 1),
    }


# ---------------------------------------------------------------------------
# Convenience: print_report (standard formatted output)
# ---------------------------------------------------------------------------

def print_report(
    label: str,
    equity_curve,
    trades: list,
    freq_per_year: int = 252,
    years: float = None,
) -> dict:
    """Print a standard formatted report and return the stats dict.

    Replaces per-experiment report_run() functions with a single standard.

    Parameters
    ----------
    label : str
        Strategy/experiment name for the header.
    equity_curve : array-like
        Portfolio equity values.
    trades : list
        List of Trade objects.
    freq_per_year : int
        Bars per year (252 for daily, 6292 for H1, etc.).
    years : float, optional
        Override for annualization period. If None, inferred from data length.
    """
    eq = np.asarray(equity_curve, dtype=np.float64)
    stats = compute_stats(eq, trades, freq_per_year=freq_per_year)

    if years is None:
        years = max(len(eq) - 1, 0) / freq_per_year if freq_per_year > 0 else 1.0

    print()
    print("=" * 50)
    print(f"  {label}")
    print("=" * 50)
    print(f"  Total Return:     {stats['total_return_pct']:>+8.2f}%")
    print(f"  CAGR:             {stats['cagr']:>+8.2f}%")
    print(f"  Max Drawdown:     {stats['max_drawdown']:>8.2f}%")
    print(f"  Sharpe:           {stats['sharpe']:>8.4f}")
    print(f"  Sortino:          {stats['sortino']:>8.4f}")
    print(f"  Calmar:           {stats['calmar']:>8.4f}")
    print(f"  Win Rate:         {stats['win_rate']:>8.1f}%")
    print(f"  Profit Factor:    {stats['profit_factor']:>8.2f}")
    print(f"  Total Trades:     {stats['total_trades']:>8d}")
    print(f"  Avg Bars Held:    {stats['avg_bars_held']:>8.1f}")
    print(f"  Period:           {years:>8.1f} years")
    print()

    return stats
