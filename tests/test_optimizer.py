"""Tests for the strategy optimizer."""

import numpy as np
import pandas as pd
import pytest

import optimizer as optimizer_module
from optimizer import optimize, walk_forward, OptimizationResult, WalkForwardResult
from backtesting.strategy import Strategy
from backtesting.statistics import compute_sharpe
from backtesting.types import Trade
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

    def test_seed_reproduces_trials(self, long_trending_df):
        first = optimize(
            TrendFollowingStrategy, PARAM_SPACE, long_trending_df,
            n_trials=5, seed=123,
        )
        second = optimize(
            TrendFollowingStrategy, PARAM_SPACE, long_trending_df,
            n_trials=5, seed=123,
        )
        assert first.seed == second.seed == 123
        pd.testing.assert_frame_equal(first.all_trials, second.all_trials)

    def test_seeded_parallel_optimization_fails_clearly(self, long_trending_df):
        with pytest.raises(ValueError, match="requires n_jobs=1"):
            optimize(
                TrendFollowingStrategy, PARAM_SPACE, long_trending_df,
                n_trials=2, seed=123, n_jobs=2,
            )

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

    def test_carries_statistical_report_and_gate(self, long_trending_df):
        result = optimize(
            TrendFollowingStrategy, PARAM_SPACE, long_trending_df,
            n_trials=5, objective="sharpe", seed=7,
        )
        assert result.n_trials_tested == 5
        assert result.statistical_report is not None
        assert result.bootstrap_ci is not None
        assert result.deflated_sharpe_p_value is not None
        assert isinstance(result.passes_statistical_gate(), bool)
        assert isinstance(
            optimizer_module.passes_statistical_gate(result.statistical_report), bool,
        )

    def test_compute_statistics_false_skips_report(self, long_trending_df):
        result = optimize(
            TrendFollowingStrategy, PARAM_SPACE, long_trending_df,
            n_trials=3, objective="sharpe", compute_statistics=False,
        )
        assert result.statistical_report is None
        assert result.passes_statistical_gate() is False


class TestWalkForward:
    def test_explicit_annualization_is_used_for_oos_curve(self, long_trending_df):
        result = walk_forward(
            TrendFollowingStrategy, PARAM_SPACE, long_trending_df,
            n_splits=2, n_trials=2, objective="sharpe", seed=12,
            freq_per_year=365,
        )
        assert result.oos_sharpe == pytest.approx(
            round(compute_sharpe(result.oos_equity_curve, freq_per_year=365), 4),
        )

    def test_event_engine_primes_strategy_from_train_context(self, long_trending_df):
        class WarmupProbeStrategy(Strategy):
            warmup_timestamps = []

            def __init__(self, marker=0):
                super().__init__()
                self.warmed = False
                self.warmup_bars = 7

            def warmup_bar(self, bar):
                type(self).warmup_timestamps.append(bar.ts)
                self.warmed = True

            def on_bar(self, i, bar, equity):
                if i == 0 and self.warmed:
                    return Trade(
                        entry_bar=bar, side=1, size=1.0,
                        entry_price=bar.close, stop_price=bar.close - 1_000,
                        take_profit=None,
                    )
                return None

        WarmupProbeStrategy.warmup_timestamps = []
        result = walk_forward(
            WarmupProbeStrategy, {"marker": [0]}, long_trending_df,
            n_splits=1, n_trials=1, objective="return", seed=10,
        )

        first_test = int(len(long_trending_df) * 0.7)
        assert result.splits[0]["warmup_bars"] == 7
        assert WarmupProbeStrategy.warmup_timestamps == list(
            long_trending_df.index[first_test - 7:first_test],
        )
        assert result.splits[0]["oos_trades"] == 1

    def test_vectorized_engine_uses_context_but_masks_warmup_entries(
        self, long_trending_df, monkeypatch,
    ):
        seen_closes = []

        def signals(open, high, low, close):
            seen_closes.append(close.copy())
            entries = np.zeros(len(close), dtype=bool)
            if len(close) > 5:
                entries[5] = True
            sides = np.where(entries, 1, 0)
            stops = close - 1_000
            tps = close + 1_000
            return entries, sides, stops, tps

        monkeypatch.setitem(
            optimizer_module._VECTORIZED_SIGNALS,
            "MomentumStrategy", signals,
        )
        result = walk_forward(
            MomentumStrategy, {"lookback": [1]}, long_trending_df,
            n_splits=1, n_trials=1, objective="return", seed=10,
            engine="vectorized", warmup_bars=5,
        )

        first_test = int(len(long_trending_df) * 0.7)
        expected_context = long_trending_df["close"].to_numpy()[first_test - 5:first_test]
        np.testing.assert_array_equal(seen_closes[-1][:5], expected_context)
        assert result.splits[0]["warmup_bars"] == 5
        assert result.splits[0]["oos_trades"] == 1

    def test_returns_result_object(self, long_trending_df):
        result = walk_forward(
            TrendFollowingStrategy, PARAM_SPACE, long_trending_df,
            n_splits=2, n_trials=3, objective="sharpe",
        )
        assert isinstance(result, WalkForwardResult)
        assert len(result.splits) == 2
        assert isinstance(result.summary, pd.DataFrame)
        assert len(result.oos_equity_curve) == len(result.oos_index)
        assert len(result.oos_equity_curve) == len(long_trending_df) - 140 + 1
        pd.testing.assert_index_equal(result.oos_index, long_trending_df.index[139:])
        assert result.oos_total_return_pct == pytest.approx(
            (result.oos_equity_curve[-1] / 10_000 - 1) * 100,
        )
        assert isinstance(result.oos_sharpe, float)

    def test_degradation_is_computed(self, long_trending_df):
        result = walk_forward(
            TrendFollowingStrategy, PARAM_SPACE, long_trending_df,
            n_splits=2, n_trials=3, objective="sharpe",
        )
        # Degradation = IS mean - OOS mean (can be positive or negative)
        expected = result.in_sample_mean - result.out_of_sample_mean
        assert abs(result.degradation - expected) < 0.01

    def test_carries_statistical_report_with_program_trials(self, long_trending_df):
        result = walk_forward(
            TrendFollowingStrategy, PARAM_SPACE, long_trending_df,
            n_splits=2, n_trials=3, objective="sharpe", seed=5,
        )
        # Every parameter combination in every fold counts as a trial.
        assert result.n_trials_tested == 3 * 2
        assert result.statistical_report is not None
        assert result.bootstrap_ci is not None
        assert isinstance(result.passes_statistical_gate(), bool)

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


class TestMinTrades:
    def test_min_trades_penalizes_low_count(self, long_trending_df):
        """With an unreachably high min_trades, all scores should be penalized."""
        baseline = optimize(
            TrendFollowingStrategy, PARAM_SPACE, long_trending_df,
            n_trials=5, objective="sharpe", min_trades=0,
        )
        penalized = optimize(
            TrendFollowingStrategy, PARAM_SPACE, long_trending_df,
            n_trials=5, objective="sharpe", min_trades=9999,
        )
        # With extreme min_trades, best scores should be much lower
        assert penalized.all_trials["score"].max() <= baseline.all_trials["score"].max()

    def test_min_trades_zero_is_noop(self, long_trending_df):
        result = optimize(
            TrendFollowingStrategy, PARAM_SPACE, long_trending_df,
            n_trials=3, objective="sharpe", min_trades=0,
        )
        assert isinstance(result, OptimizationResult)


class TestTopKAvg:
    def test_top_k_1_returns_best(self, long_trending_df):
        result = optimize(
            TrendFollowingStrategy, PARAM_SPACE, long_trending_df,
            n_trials=5, objective="sharpe", top_k_avg=1,
        )
        assert isinstance(result, OptimizationResult)

    def test_top_k_averages_params(self, long_trending_df):
        result = optimize(
            TrendFollowingStrategy, PARAM_SPACE, long_trending_df,
            n_trials=10, objective="sharpe", top_k_avg=5,
        )
        # Averaged params should still be within bounds
        assert 5 <= result.best_params["fast_period"] <= 30
        assert 20 <= result.best_params["slow_period"] <= 60
        assert 1.0 <= result.best_params["atr_stop_mult"] <= 4.0

    def test_top_k_int_params_are_int(self, long_trending_df):
        result = optimize(
            TrendFollowingStrategy, PARAM_SPACE, long_trending_df,
            n_trials=10, objective="sharpe", top_k_avg=5,
        )
        assert isinstance(result.best_params["fast_period"], int)
        assert isinstance(result.best_params["slow_period"], int)

    def test_top_k_larger_than_trials(self, long_trending_df):
        """top_k_avg > n_trials should still work (uses all trials)."""
        result = optimize(
            TrendFollowingStrategy, PARAM_SPACE, long_trending_df,
            n_trials=3, objective="sharpe", top_k_avg=100,
        )
        assert isinstance(result, OptimizationResult)


class TestAnchoredWalkForward:
    def test_anchored_training_starts_at_zero(self, long_trending_df):
        result = walk_forward(
            TrendFollowingStrategy, PARAM_SPACE, long_trending_df,
            n_splits=2, n_trials=3, objective="sharpe", anchored=True,
        )
        first_date = str(long_trending_df.index[0].date())
        for split in result.splits:
            assert split["train_start"] == first_date

    def test_anchored_false_has_rolling_windows(self, long_trending_df):
        result = walk_forward(
            TrendFollowingStrategy, PARAM_SPACE, long_trending_df,
            n_splits=2, n_trials=3, objective="sharpe", anchored=False,
        )
        train_starts = [s["train_start"] for s in result.splits]
        # Rolling windows should have different training start dates
        assert len(set(train_starts)) > 1

    def test_anchored_produces_valid_result(self, long_trending_df):
        result = walk_forward(
            TrendFollowingStrategy, PARAM_SPACE, long_trending_df,
            n_splits=2, n_trials=3, objective="sharpe", anchored=True,
        )
        assert isinstance(result, WalkForwardResult)
        assert len(result.splits) == 2
        expected_deg = result.in_sample_mean - result.out_of_sample_mean
        assert abs(result.degradation - expected_deg) < 0.01
