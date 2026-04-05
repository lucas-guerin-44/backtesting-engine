"""Tests for the strategy optimizer."""

import numpy as np
import pandas as pd
import pytest

from optimizer import optimize, walk_forward, OptimizationResult, WalkForwardResult
from strategies import TrendFollowingStrategy, MomentumStrategy


@pytest.fixture
def long_trending_df():
    """200 bars of uptrend — enough for walk-forward splits."""
    n = 200
    closes = np.linspace(100, 200, n) + np.random.RandomState(42).normal(0, 2, n)
    data = {
        "open": closes - 0.5,
        "high": closes + 2.0,
        "low": closes - 2.0,
        "close": closes,
    }
    return pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=n, freq="h"))


PARAM_SPACE = {
    "fast_period": (5, 30),
    "slow_period": (20, 60),
    "atr_stop_mult": (1.0, 4.0),
}


class TestOptimize:
    def test_returns_result_object(self, long_trending_df):
        result = optimize(
            TrendFollowingStrategy, PARAM_SPACE, long_trending_df,
            n_trials=5, objective="sharpe",
        )
        assert isinstance(result, OptimizationResult)
        assert result.best_params is not None
        assert isinstance(result.best_score, float)
        assert len(result.all_trials) == 5

    def test_best_params_within_bounds(self, long_trending_df):
        result = optimize(
            TrendFollowingStrategy, PARAM_SPACE, long_trending_df,
            n_trials=5, objective="sharpe",
        )
        assert 5 <= result.best_params["fast_period"] <= 30
        assert 20 <= result.best_params["slow_period"] <= 60
        assert 1.0 <= result.best_params["atr_stop_mult"] <= 4.0

    def test_all_objectives_work(self, long_trending_df):
        for obj in ["sharpe", "return", "calmar", "sortino"]:
            result = optimize(
                TrendFollowingStrategy, PARAM_SPACE, long_trending_df,
                n_trials=3, objective=obj,
            )
            assert isinstance(result.best_score, float)

    def test_invalid_objective_raises(self, long_trending_df):
        with pytest.raises(ValueError, match="Unknown objective"):
            optimize(TrendFollowingStrategy, PARAM_SPACE, long_trending_df,
                     n_trials=1, objective="nonsense")

    def test_fixed_params_passed_through(self, long_trending_df):
        result = optimize(
            TrendFollowingStrategy, {"fast_period": (5, 15)}, long_trending_df,
            n_trials=3, objective="sharpe",
            fixed_params={"slow_period": 50, "risk_per_trade": 0.01},
        )
        # fixed params should not appear in best_params (they weren't searched)
        assert "slow_period" not in result.best_params


class TestWalkForward:
    def test_returns_result_object(self, long_trending_df):
        result = walk_forward(
            TrendFollowingStrategy, PARAM_SPACE, long_trending_df,
            n_splits=2, n_trials=3, objective="sharpe",
        )
        assert isinstance(result, WalkForwardResult)
        assert len(result.splits) == 2
        assert isinstance(result.summary, pd.DataFrame)

    def test_degradation_is_computed(self, long_trending_df):
        result = walk_forward(
            TrendFollowingStrategy, PARAM_SPACE, long_trending_df,
            n_splits=2, n_trials=3, objective="sharpe",
        )
        # Degradation = IS mean - OOS mean (can be positive or negative)
        expected = result.in_sample_mean - result.out_of_sample_mean
        assert abs(result.degradation - expected) < 0.01

    def test_too_few_bars_raises(self):
        tiny_df = pd.DataFrame({
            "open": [100.0] * 10, "high": [101.0] * 10,
            "low": [99.0] * 10, "close": [100.0] * 10,
        }, index=pd.date_range("2024-01-01", periods=10, freq="h"))

        with pytest.raises(ValueError, match="Not enough data"):
            walk_forward(TrendFollowingStrategy, PARAM_SPACE, tiny_df,
                         n_splits=5, n_trials=1)


class TestParallelOptimize:
    def test_parallel_produces_valid_result(self, long_trending_df):
        result = optimize(
            TrendFollowingStrategy, PARAM_SPACE, long_trending_df,
            n_trials=6, objective="sharpe", n_jobs=2,
        )
        assert isinstance(result, OptimizationResult)
        assert result.best_params is not None
        assert len(result.all_trials) == 6

    def test_n_jobs_minus_one(self, long_trending_df):
        result = optimize(
            TrendFollowingStrategy, PARAM_SPACE, long_trending_df,
            n_trials=4, objective="sharpe", n_jobs=-1,
        )
        assert isinstance(result, OptimizationResult)

    def test_parallel_walk_forward(self, long_trending_df):
        result = walk_forward(
            TrendFollowingStrategy, PARAM_SPACE, long_trending_df,
            n_splits=2, n_trials=3, objective="sharpe", n_jobs=2,
        )
        assert isinstance(result, WalkForwardResult)
        assert len(result.splits) == 2

    def test_parallel_params_within_bounds(self, long_trending_df):
        result = optimize(
            TrendFollowingStrategy, PARAM_SPACE, long_trending_df,
            n_trials=6, objective="sharpe", n_jobs=2,
        )
        assert 5 <= result.best_params["fast_period"] <= 30
        assert 20 <= result.best_params["slow_period"] <= 60
