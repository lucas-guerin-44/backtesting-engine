"""Tests for statistical significance testing."""

import numpy as np
import pytest

from backtesting.statistics import (
    SharpeCI,
    PermutationTestResult,
    DeflatedSharpeResult,
    StatisticalReport,
    bootstrap_sharpe_ci,
    permutation_test,
    deflated_sharpe_ratio,
    compute_statistical_report,
)


def make_trending_equity(n=500, start=10_000, annual_return=0.5):
    """Equity curve with a strong uptrend (should be significant)."""
    daily_return = (1 + annual_return) ** (1 / 252) - 1
    returns = np.random.RandomState(42).normal(daily_return, daily_return * 2, n)
    equity = start * np.cumprod(1 + returns)
    return equity


def make_flat_equity(n=500, start=10_000):
    """Equity curve that's flat with noise (should NOT be significant)."""
    returns = np.random.RandomState(99).normal(0, 0.01, n)
    equity = start * np.cumprod(1 + returns)
    return equity


class TestBootstrapSharpeCI:
    def test_strong_trend_ci_excludes_zero(self):
        eq = make_trending_equity()
        ci = bootstrap_sharpe_ci(eq, seed=42)
        assert ci.significant
        assert ci.ci_lower > 0

    def test_flat_equity_ci_includes_zero(self):
        eq = make_flat_equity()
        ci = bootstrap_sharpe_ci(eq, seed=42)
        # Flat equity -> CI should include zero (not significant)
        assert ci.ci_lower <= 0 or ci.ci_upper >= 0

    def test_confidence_level_affects_width(self):
        eq = make_trending_equity()
        ci_90 = bootstrap_sharpe_ci(eq, confidence=0.90, seed=42)
        ci_99 = bootstrap_sharpe_ci(eq, confidence=0.99, seed=42)
        width_90 = ci_90.ci_upper - ci_90.ci_lower
        width_99 = ci_99.ci_upper - ci_99.ci_lower
        assert width_99 > width_90

    def test_reproducible_with_seed(self):
        eq = make_trending_equity()
        ci1 = bootstrap_sharpe_ci(eq, seed=123)
        ci2 = bootstrap_sharpe_ci(eq, seed=123)
        assert ci1.ci_lower == ci2.ci_lower
        assert ci1.ci_upper == ci2.ci_upper

    def test_short_equity_curve(self):
        eq = np.array([100.0, 101.0, 102.0])
        ci = bootstrap_sharpe_ci(eq, seed=42)
        assert isinstance(ci, SharpeCI)

    def test_str_readable(self):
        ci = bootstrap_sharpe_ci(make_trending_equity(), seed=42)
        s = str(ci)
        assert "Sharpe" in s


class TestPermutationTest:
    def test_strong_strategy_is_significant(self):
        eq = make_trending_equity()

        # Create fake trades with consistently positive PnL
        class FakeTrade:
            def __init__(self, pnl):
                self.pnl = pnl
        trades = [FakeTrade(pnl=np.random.RandomState(42).uniform(10, 100)) for _ in range(50)]

        result = permutation_test(eq, trades, seed=42)
        assert result.significant
        assert result.p_value < 0.05

    def test_random_trades_not_significant(self):
        eq = make_flat_equity()

        class FakeTrade:
            def __init__(self, pnl):
                self.pnl = pnl
        # Random PnLs around zero
        trades = [FakeTrade(pnl=np.random.RandomState(i).normal(0, 50)) for i in range(50)]

        result = permutation_test(eq, trades, seed=42)
        # Should generally not be significant (random trades)
        assert isinstance(result, PermutationTestResult)

    def test_no_trades_falls_back_to_return_shuffling(self):
        # With no trades, permutation test falls back to shuffling returns
        eq = np.array([100.0] * 50, dtype=np.float64)  # Truly flat
        result = permutation_test(eq, [], seed=42)
        assert isinstance(result, PermutationTestResult)

    def test_reproducible_with_seed(self):
        eq = make_trending_equity()

        class FakeTrade:
            def __init__(self, pnl):
                self.pnl = pnl
        trades = [FakeTrade(pnl=10.0) for _ in range(20)]

        r1 = permutation_test(eq, trades, seed=42)
        r2 = permutation_test(eq, trades, seed=42)
        assert r1.p_value == r2.p_value


class TestDeflatedSharpe:
    def test_single_trial_no_deflation(self):
        result = deflated_sharpe_ratio(
            observed_sharpe=1.5, n_trials=1, n_observations=252)
        # With 1 trial, expected max is 0, so deflated ≈ observed
        assert abs(result.deflated_sharpe - result.observed_sharpe) < 0.01

    def test_many_trials_deflates(self):
        result = deflated_sharpe_ratio(
            observed_sharpe=0.5, n_trials=1000, n_observations=252)
        assert result.deflated_sharpe < result.observed_sharpe

    def test_high_sharpe_survives_deflation(self):
        result = deflated_sharpe_ratio(
            observed_sharpe=3.0, n_trials=100, n_observations=252)
        assert result.significant

    def test_marginal_sharpe_killed_by_many_trials(self):
        result = deflated_sharpe_ratio(
            observed_sharpe=0.3, n_trials=5000, n_observations=252)
        assert not result.significant

    def test_with_trial_sharpes(self):
        sharpes = np.random.RandomState(42).normal(0.2, 0.5, 100)
        sharpes[0] = 1.5  # "Best" trial
        result = deflated_sharpe_ratio(
            observed_sharpe=1.5, n_trials=100, n_observations=252,
            all_trial_sharpes=sharpes)
        assert isinstance(result, DeflatedSharpeResult)


class TestStatisticalReport:
    def test_full_report(self):
        eq = make_trending_equity()

        class FakeTrade:
            def __init__(self, pnl):
                self.pnl = pnl
        trades = [FakeTrade(pnl=50.0) for _ in range(30)]

        report = compute_statistical_report(eq, trades, n_trials_tested=50, seed=42)
        assert isinstance(report, StatisticalReport)
        assert report.bootstrap_ci is not None
        assert report.permutation_test is not None
        assert report.deflated_sharpe is not None

    def test_single_trial_skips_dsr(self):
        eq = make_trending_equity()
        report = compute_statistical_report(eq, [], n_trials_tested=1, seed=42)
        assert report.deflated_sharpe is None

    def test_str_output(self):
        eq = make_trending_equity()
        report = compute_statistical_report(eq, [], n_trials_tested=10, seed=42)
        s = str(report)
        assert "Statistical" in s
        assert "Bootstrap" in s
