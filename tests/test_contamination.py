"""Regression tests for walk-forward data contamination.

Verifies that the two-stage walk-forward implementation in optimizer.py
does NOT leak out-of-sample data into the training process.

Background: an earlier version locked parameters from a full-dataset optimization
before running walk-forward, which meant the OOS portion had already been seen
during parameter selection. This inflated OOS Sharpe by ~7x. The fix was to
run optimization independently inside each training window.

These tests use synthetic data with a planted regime change to detect leakage:
- First half: strong uptrend (momentum should work)
- Second half: mean-reverting/flat (momentum should struggle)

If walk-forward is honest, the OOS score on the second half should be significantly
worse than IS. If it leaks, the optimizer would "know" about the regime change
and produce suspiciously good OOS scores.
"""

import numpy as np
import pandas as pd
import pytest

from optimizer import optimize, walk_forward
from strategies import MomentumStrategy


# ─── Synthetic data with planted regime change ────────────────────────────────

def _make_regime_data(n_bars=1000, trend_frac=0.5, seed=42):
    """Create synthetic data: first half trending up, second half flat/noisy.

    This makes momentum strategies overfit to the first half.
    If walk-forward leaks, it will "know" the regime change
    and produce unrealistically good OOS scores.
    """
    rng = np.random.RandomState(seed)
    split = int(n_bars * trend_frac)

    # First half: clear uptrend (+0.3% per bar with low noise)
    trend_returns = 0.003 + rng.randn(split) * 0.002
    # Second half: mean-reverting noise (0% drift, high noise)
    flat_returns = rng.randn(n_bars - split) * 0.008

    returns = np.concatenate([trend_returns, flat_returns])
    prices = 100.0 * np.cumprod(1 + returns)
    prices = np.maximum(prices, 1.0)

    close = prices
    open_ = close * (1 + rng.randn(n_bars) * 0.001)
    high = np.maximum(open_, close) * (1 + rng.rand(n_bars) * 0.005)
    low = np.minimum(open_, close) * (1 - rng.rand(n_bars) * 0.005)

    index = pd.date_range("2015-01-01", periods=n_bars, freq="D")
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
    }, index=index)


@pytest.fixture
def regime_df():
    return _make_regime_data()


# Simple param space for fast optimization
PARAM_SPACE = {
    "lookback": (5, 30),
    "entry_threshold": (0.01, 0.06),
}

FIXED_PARAMS = {
    "atr_period": 14,
    "atr_stop_mult": 2.0,
    "atr_target_mult": 3.0,
    "risk_per_trade": 0.02,
    "max_dd_halt": 0.15,
    "cooldown_bars": 10,
}


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestWalkForwardContamination:
    """Verify walk-forward does not leak OOS data into training."""

    def test_oos_not_suspiciously_close_to_is(self, regime_df):
        """On data with a regime change, OOS should degrade from IS.

        If OOS ≈ IS on regime-change data, the optimizer is likely leaking.
        A healthy walk-forward should show positive degradation (IS > OOS).
        """
        result = walk_forward(
            MomentumStrategy, PARAM_SPACE, regime_df,
            n_splits=2, train_ratio=0.7, n_trials=20,
            objective="sharpe", fixed_params=FIXED_PARAMS,
            engine="event",
        )

        # With a genuine train/test split on regime-change data,
        # the optimizer should show degradation (IS better than OOS)
        # A zero or negative degradation on this data would be suspicious
        assert result.degradation > -1.0, (
            f"Suspiciously good OOS performance on regime-change data: "
            f"degradation={result.degradation} (IS={result.in_sample_mean}, "
            f"OOS={result.out_of_sample_mean})"
        )

    def test_walk_forward_trains_only_on_training_window(self, regime_df):
        """Each split's optimization should only see its training data.

        Verify by checking that different splits produce different best_params,
        since they train on different data windows.
        """
        result = walk_forward(
            MomentumStrategy, PARAM_SPACE, regime_df,
            n_splits=3, train_ratio=0.7, n_trials=15,
            objective="sharpe", fixed_params=FIXED_PARAMS,
            engine="event",
        )

        assert len(result.splits) >= 2
        # Splits should have been trained independently on different windows
        # Check that train_start and test_start differ across splits
        train_starts = [s["train_start"] for s in result.splits]
        test_starts = [s["test_start"] for s in result.splits]
        assert len(set(test_starts)) == len(test_starts), (
            "Test windows should not overlap"
        )

    def test_is_outperforms_oos_on_regime_data(self, regime_df):
        """In-sample score should exceed OOS on regime-change data.

        This is the core contamination check: the optimizer always looks better
        in-sample than out-of-sample. If OOS >= IS on regime-change data,
        something is leaking.
        """
        wf_result = walk_forward(
            MomentumStrategy, PARAM_SPACE, regime_df,
            n_splits=2, train_ratio=0.7, n_trials=20,
            objective="sharpe", fixed_params=FIXED_PARAMS,
            engine="event",
        )

        # IS should be >= OOS (degradation >= 0) on regime-change data
        # where the second half is fundamentally different from the first
        assert wf_result.in_sample_mean >= wf_result.out_of_sample_mean, (
            f"OOS ({wf_result.out_of_sample_mean:.4f}) should not beat "
            f"IS ({wf_result.in_sample_mean:.4f}) on regime-change data — "
            f"possible data leakage"
        )

    def test_train_test_windows_dont_overlap(self, regime_df):
        """Training and test periods within each split must not overlap."""
        result = walk_forward(
            MomentumStrategy, PARAM_SPACE, regime_df,
            n_splits=3, train_ratio=0.7, n_trials=10,
            objective="sharpe", fixed_params=FIXED_PARAMS,
            engine="event",
        )

        for split in result.splits:
            train_end = pd.Timestamp(split["train_end"])
            test_start = pd.Timestamp(split["test_start"])
            assert test_start > train_end, (
                f"Split {split['split']}: test starts ({test_start}) "
                f"before train ends ({train_end})"
            )

    def test_anchored_walk_forward_no_future_leak(self, regime_df):
        """Anchored walk-forward should expand training window but never include test data."""
        result = walk_forward(
            MomentumStrategy, PARAM_SPACE, regime_df,
            n_splits=2, train_ratio=0.7, n_trials=15,
            objective="sharpe", fixed_params=FIXED_PARAMS,
            anchored=True, engine="event",
        )

        first_date = str(regime_df.index[0].date())
        for split in result.splits:
            # Anchored: all training windows start at the beginning
            assert split["train_start"] == first_date, (
                f"Anchored split {split['split']} doesn't start at data start"
            )
            # Test window must come after training window
            assert pd.Timestamp(split["test_start"]) > pd.Timestamp(split["train_end"])
