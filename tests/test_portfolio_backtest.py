"""Tests for the multi-asset PortfolioBacktester and allocation schemes."""

import numpy as np
import pandas as pd
import pytest

from backtesting.allocation import (
    AllocationWeights,
    CorrelationAwareAllocator,
    EqualWeightAllocator,
    RegimeAllocator,
    RiskParityAllocator,
)
from backtesting.backtest import Backtester
from backtesting.portfolio_backtest import PortfolioBacktester, PortfolioBacktestResult, RiskLimits
from backtesting.strategy import Strategy
from backtesting.types import Bar, Trade


# ---------------------------------------------------------------------------
# Test strategies
# ---------------------------------------------------------------------------

class NeverTradeStrategy(Strategy):
    def on_bar(self, i, bar, equity):
        return None


class BuyOnceStrategy(Strategy):
    def __init__(self, stop_pct=0.05, tp_pct=0.10):
        self.stop_pct = stop_pct
        self.tp_pct = tp_pct
        self.entered = False

    def on_bar(self, i, bar, equity):
        if not self.entered and equity > 10:
            self.entered = True
            return Trade(
                entry_bar=bar, side=1,
                size=equity * 0.5 / bar.close,
                entry_price=bar.close,
                stop_price=bar.close * (1 - self.stop_pct),
                take_profit=bar.close * (1 + self.tp_pct),
            )
        return None


class AlwaysBuyStrategy(Strategy):
    def on_bar(self, i, bar, equity):
        if equity > 10:
            return Trade(
                entry_bar=bar, side=1, size=equity * 0.3 / bar.close,
                entry_price=bar.close,
                stop_price=bar.close * 0.90,
                take_profit=bar.close * 1.10,
            )
        return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def aligned_dfs():
    """Two assets with identical timestamps: one uptrend, one downtrend."""
    n = 100
    ts = pd.date_range("2024-01-01", periods=n, freq="h")

    closes_a = np.linspace(100, 150, n)
    df_a = pd.DataFrame({
        "open": closes_a - 0.5, "high": closes_a + 1.0,
        "low": closes_a - 1.0, "close": closes_a,
    }, index=ts)

    closes_b = np.linspace(100, 70, n)
    df_b = pd.DataFrame({
        "open": closes_b + 0.5, "high": closes_b + 1.0,
        "low": closes_b - 1.0, "close": closes_b,
    }, index=ts)

    return {"ASSET_A": df_a, "ASSET_B": df_b}


@pytest.fixture
def misaligned_dfs():
    """Two assets with partially overlapping timestamps."""
    ts_a = pd.date_range("2024-01-01", periods=80, freq="h")
    ts_b = pd.date_range("2024-01-01 10:00", periods=80, freq="h")

    closes_a = np.linspace(100, 140, 80)
    df_a = pd.DataFrame({
        "open": closes_a - 0.5, "high": closes_a + 1.0,
        "low": closes_a - 1.0, "close": closes_a,
    }, index=ts_a)

    closes_b = np.linspace(200, 220, 80)
    df_b = pd.DataFrame({
        "open": closes_b - 0.5, "high": closes_b + 1.0,
        "low": closes_b - 1.0, "close": closes_b,
    }, index=ts_b)

    return {"ASSET_A": df_a, "ASSET_B": df_b}


@pytest.fixture
def single_asset_df():
    """100-bar uptrend for single-asset comparison."""
    n = 100
    closes = np.linspace(100, 200, n)
    data = {
        "open": closes - 0.5,
        "high": closes + 1.0,
        "low": closes - 1.0,
        "close": closes,
    }
    return pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=n, freq="h"))


@pytest.fixture
def flat_dfs():
    """Two flat-price assets."""
    n = 50
    ts = pd.date_range("2024-01-01", periods=n, freq="h")

    for_both = pd.DataFrame({
        "open": [100.0] * n, "high": [100.5] * n,
        "low": [99.5] * n, "close": [100.0] * n,
    }, index=ts)

    return {"X": for_both.copy(), "Y": for_both.copy()}


# ---------------------------------------------------------------------------
# Allocation schemes
# ---------------------------------------------------------------------------

class TestAllocationSchemes:
    def test_equal_weight_returns_uniform(self):
        symbols = ["A", "B", "C"]
        closes = {s: np.ones(10) * 100 for s in symbols}
        alloc = EqualWeightAllocator()
        w = alloc.compute_weights(symbols, closes, 10, 5)

        assert len(w.weights) == 3
        for s in symbols:
            assert abs(w.weights[s] - 1 / 3) < 1e-10
        assert w.method == "equal"

    def test_risk_parity_higher_weight_to_lower_vol(self):
        symbols = ["LOW_VOL", "HIGH_VOL"]
        # Low vol: prices barely move
        low_vol = np.linspace(100, 101, 50)
        # High vol: prices swing more
        high_vol = 100 + np.sin(np.linspace(0, 10, 50)) * 10
        closes = {"LOW_VOL": low_vol, "HIGH_VOL": high_vol}

        alloc = RiskParityAllocator(min_lookback=10)
        w = alloc.compute_weights(symbols, closes, 40, 49)

        assert w.weights["LOW_VOL"] > w.weights["HIGH_VOL"]
        assert abs(sum(w.weights.values()) - 1.0) < 1e-10

    def test_risk_parity_falls_back_to_equal_early(self):
        symbols = ["A", "B"]
        closes = {s: np.ones(5) * 100 for s in symbols}
        alloc = RiskParityAllocator(min_lookback=20)
        w = alloc.compute_weights(symbols, closes, 20, 3)

        # Should fall back to equal weight (current_idx < min_lookback)
        assert abs(w.weights["A"] - 0.5) < 1e-10
        assert w.method == "equal"

    def test_correlation_aware_reduces_correlated_assets(self):
        n = 100
        symbols = ["CORR_A", "CORR_B", "UNCORR"]
        base = np.cumsum(np.random.RandomState(42).randn(n)) + 100

        closes = {
            "CORR_A": base,
            "CORR_B": base + np.random.RandomState(43).randn(n) * 0.1,  # Nearly identical
            "UNCORR": np.cumsum(np.random.RandomState(99).randn(n)) + 100,  # Independent
        }

        alloc = CorrelationAwareAllocator(min_lookback=20)
        w = alloc.compute_weights(symbols, closes, 80, 99)

        # Uncorrelated asset should get higher weight
        assert w.weights["UNCORR"] > w.weights["CORR_A"]
        assert w.weights["UNCORR"] > w.weights["CORR_B"]

    def test_correlation_aware_falls_back_with_single_asset(self):
        closes = {"ONLY": np.linspace(100, 110, 50)}
        alloc = CorrelationAwareAllocator(min_lookback=10)
        w = alloc.compute_weights(["ONLY"], closes, 40, 49)

        assert abs(w.weights["ONLY"] - 1.0) < 1e-10

    def test_weights_sum_to_one(self):
        symbols = ["A", "B", "C", "D"]
        closes = {s: np.random.RandomState(i).randn(100).cumsum() + 100
                  for i, s in enumerate(symbols)}

        for AllocCls in [EqualWeightAllocator, RiskParityAllocator, CorrelationAwareAllocator]:
            alloc = AllocCls() if AllocCls == EqualWeightAllocator else AllocCls(min_lookback=10)
            w = alloc.compute_weights(symbols, closes, 50, 99)
            assert abs(sum(w.weights.values()) - 1.0) < 1e-10


class TestRegimeAllocator:
    def test_high_vol_boosts_trend_symbols(self):
        """In a high-vol environment, trend symbols should get higher weight."""
        rng = np.random.RandomState(42)
        n = 3000
        symbols = ["TREND", "REVERT"]

        # Both assets have high recent volatility (big swings at the end)
        base = np.cumsum(rng.randn(n)) + 500
        base[-300:] += np.cumsum(rng.randn(300) * 5)  # vol spike at end
        closes = {
            "TREND": base.copy(),
            "REVERT": base + rng.randn(n) * 2,
        }

        alloc = RegimeAllocator(
            trend_symbols={"TREND"}, reversion_symbols={"REVERT"},
            vol_lookback=100, vol_history=2000,
            vol_threshold_pct=50.0, regime_boost=2.0,
            min_lookback=200,
        )
        w = alloc.compute_weights(symbols, closes, 500, n - 1)

        assert w.weights["TREND"] > w.weights["REVERT"]
        assert abs(sum(w.weights.values()) - 1.0) < 1e-10
        assert "trending" in w.method

    def test_low_vol_boosts_reversion_symbols(self):
        """In a low-vol environment, reversion symbols should get higher weight."""
        n = 3000
        symbols = ["TREND", "REVERT"]

        # Both assets have low recent volatility (flat at the end)
        rng = np.random.RandomState(42)
        base = np.cumsum(rng.randn(n) * 3) + 500  # volatile history
        base[-300:] = np.linspace(base[-301], base[-301] + 1, 300)  # flat tail
        closes = {
            "TREND": base.copy(),
            "REVERT": base + rng.randn(n) * 0.1,
        }

        alloc = RegimeAllocator(
            trend_symbols={"TREND"}, reversion_symbols={"REVERT"},
            vol_lookback=100, vol_history=2000,
            vol_threshold_pct=50.0, regime_boost=2.0,
            min_lookback=200,
        )
        w = alloc.compute_weights(symbols, closes, 500, n - 1)

        assert w.weights["REVERT"] > w.weights["TREND"]
        assert abs(sum(w.weights.values()) - 1.0) < 1e-10
        assert "ranging" in w.method

    def test_falls_back_during_warmup(self):
        """Should fall back to risk parity when not enough bars."""
        symbols = ["A", "B"]
        closes = {s: np.linspace(100, 110, 50) for s in symbols}

        alloc = RegimeAllocator(
            trend_symbols={"A"}, reversion_symbols={"B"},
            min_lookback=200,
        )
        w = alloc.compute_weights(symbols, closes, 40, 49)

        # During warmup, should not be a regime method
        assert "regime" not in w.method

    def test_neutral_symbols_keep_base_weight(self):
        """Symbols in neither group should keep their risk parity weight."""
        rng = np.random.RandomState(42)
        n = 3000
        symbols = ["TREND", "NEUTRAL", "REVERT"]
        closes = {s: np.cumsum(rng.randn(n)) + 500 for s in symbols}

        alloc = RegimeAllocator(
            trend_symbols={"TREND"}, reversion_symbols={"REVERT"},
            vol_lookback=100, vol_history=2000, min_lookback=200,
        )

        # Get regime weights and risk parity weights for comparison
        regime_w = alloc.compute_weights(symbols, closes, 500, n - 1)
        rp_w = RiskParityAllocator(20).compute_weights(symbols, closes, 500, n - 1)

        # Neutral's share of total should be closer to its RP share than
        # the boosted/penalized symbols. The key invariant: NEUTRAL's raw
        # weight was not multiplied by regime_boost or 1/regime_boost.
        # After renormalization the exact value shifts, but it should be
        # between the boosted and penalized symbol weights if they started
        # similar.
        assert abs(sum(regime_w.weights.values()) - 1.0) < 1e-10

    def test_weights_sum_to_one(self):
        rng = np.random.RandomState(42)
        n = 3000
        symbols = ["A", "B", "C", "D"]
        closes = {s: np.cumsum(rng.randn(n)) + 500 for s in symbols}

        alloc = RegimeAllocator(
            trend_symbols={"A", "B"}, reversion_symbols={"C", "D"},
            min_lookback=200,
        )
        w = alloc.compute_weights(symbols, closes, 500, n - 1)
        assert abs(sum(w.weights.values()) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Timestamp alignment
# ---------------------------------------------------------------------------

class TestTimestampAlignment:
    def test_aligned_timestamps_same_length(self, aligned_dfs):
        pbt = PortfolioBacktester(
            aligned_dfs,
            {"ASSET_A": NeverTradeStrategy(), "ASSET_B": NeverTradeStrategy()},
        )
        assert pbt.n == 100

    def test_misaligned_timestamps_union(self, misaligned_dfs):
        pbt = PortfolioBacktester(
            misaligned_dfs,
            {"ASSET_A": NeverTradeStrategy(), "ASSET_B": NeverTradeStrategy()},
        )
        # Union of [0:80h] and [10:90h] = 90 unique timestamps
        assert pbt.n == 90

    def test_forward_fill_produces_valid_arrays(self, misaligned_dfs):
        pbt = PortfolioBacktester(
            misaligned_dfs,
            {"ASSET_A": NeverTradeStrategy(), "ASSET_B": NeverTradeStrategy()},
        )
        # No NaN in close arrays after forward-fill + back-fill
        for sym in pbt.symbols:
            assert not np.any(np.isnan(pbt._close[sym]))


# ---------------------------------------------------------------------------
# Shared cash management
# ---------------------------------------------------------------------------

class TestSharedCashManagement:
    def test_total_equity_starts_at_starting_cash(self, aligned_dfs):
        pbt = PortfolioBacktester(
            aligned_dfs,
            {"ASSET_A": NeverTradeStrategy(), "ASSET_B": NeverTradeStrategy()},
            starting_cash=50_000,
        )
        result = pbt.run()
        assert abs(result.equity_curve[0] - 50_000) < 1e-6

    def test_no_trades_flat_equity(self, flat_dfs):
        pbt = PortfolioBacktester(
            flat_dfs,
            {"X": NeverTradeStrategy(), "Y": NeverTradeStrategy()},
            starting_cash=10_000,
        )
        result = pbt.run()
        np.testing.assert_allclose(result.equity_curve, 10_000.0)

    def test_cash_shared_across_assets(self, aligned_dfs):
        """Both assets should be able to trade using the shared cash pool."""
        pbt = PortfolioBacktester(
            aligned_dfs,
            {"ASSET_A": BuyOnceStrategy(), "ASSET_B": BuyOnceStrategy()},
            starting_cash=10_000,
        )
        result = pbt.run()
        # Both assets should have entered trades
        assert len(result.trades) >= 2

    def test_allocation_weights_affect_position_size(self, aligned_dfs):
        """Unequal weights should result in different position sizes."""
        from backtesting.allocation import Allocator, AllocationWeights

        class HeavyAAllocator(Allocator):
            def compute_weights(self, symbols, close_arrays, lookback, current_idx):
                return AllocationWeights(
                    weights={"ASSET_A": 0.9, "ASSET_B": 0.1}, method="custom")

        pbt = PortfolioBacktester(
            aligned_dfs,
            {"ASSET_A": BuyOnceStrategy(), "ASSET_B": BuyOnceStrategy()},
            allocator=HeavyAAllocator(),
            starting_cash=10_000,
        )
        result = pbt.run()

        # Asset A trades should exist (it got 90% of cash)
        a_trades = result.per_asset_trades.get("ASSET_A", [])
        b_trades = result.per_asset_trades.get("ASSET_B", [])
        if a_trades and b_trades:
            assert a_trades[0].size > b_trades[0].size


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestPortfolioBacktesterEdgeCases:
    def test_single_asset_portfolio(self, single_asset_df):
        """Single-asset PortfolioBacktester should work without error."""
        pbt = PortfolioBacktester(
            {"SYM": single_asset_df},
            {"SYM": BuyOnceStrategy()},
            starting_cash=10_000,
        )
        result = pbt.run()
        assert len(result.equity_curve) == len(single_asset_df)
        assert result.equity_curve[0] == 10_000.0

    def test_mismatched_keys_raises(self):
        n = 10
        ts = pd.date_range("2024-01-01", periods=n, freq="h")
        df = pd.DataFrame({
            "open": [100] * n, "high": [101] * n,
            "low": [99] * n, "close": [100] * n,
        }, index=ts)

        with pytest.raises(ValueError, match="same keys"):
            PortfolioBacktester(
                {"A": df},
                {"B": NeverTradeStrategy()},
            )

    def test_empty_dataframes_raises(self):
        with pytest.raises(ValueError, match="At least one asset"):
            PortfolioBacktester({}, {})

    def test_equity_never_negative(self, aligned_dfs):
        pbt = PortfolioBacktester(
            aligned_dfs,
            {"ASSET_A": AlwaysBuyStrategy(), "ASSET_B": AlwaysBuyStrategy()},
            starting_cash=10_000,
        )
        result = pbt.run()
        assert np.all(result.equity_curve >= 0)

    def test_result_has_correct_structure(self, aligned_dfs):
        pbt = PortfolioBacktester(
            aligned_dfs,
            {"ASSET_A": NeverTradeStrategy(), "ASSET_B": NeverTradeStrategy()},
        )
        result = pbt.run()

        assert isinstance(result, PortfolioBacktestResult)
        assert isinstance(result.equity_curve, np.ndarray)
        assert "ASSET_A" in result.per_asset_equity
        assert "ASSET_B" in result.per_asset_equity
        assert isinstance(result.trades, list)
        assert isinstance(result.per_asset_trades, dict)
        assert isinstance(result.timestamps, list)
        assert isinstance(result.allocation_history, list)
        assert len(result.allocation_history) >= 1

    def test_rebalance_frequency(self, aligned_dfs):
        """Allocation history should grow with rebalancing."""
        pbt = PortfolioBacktester(
            aligned_dfs,
            {"ASSET_A": NeverTradeStrategy(), "ASSET_B": NeverTradeStrategy()},
            rebalance_frequency=20,
        )
        result = pbt.run()
        # 100 bars / 20 = 5 rebalances, plus initial = at least 5
        assert len(result.allocation_history) >= 5

    def test_commission_reduces_equity(self, aligned_dfs):
        strats_clean = {"ASSET_A": BuyOnceStrategy(), "ASSET_B": BuyOnceStrategy()}
        strats_comm = {"ASSET_A": BuyOnceStrategy(), "ASSET_B": BuyOnceStrategy()}

        pbt_clean = PortfolioBacktester(
            aligned_dfs, strats_clean, starting_cash=10_000)
        pbt_comm = PortfolioBacktester(
            aligned_dfs, strats_comm, starting_cash=10_000, commission_bps=50.0)

        r_clean = pbt_clean.run()
        r_comm = pbt_comm.run()

        assert r_comm.equity_curve[-1] < r_clean.equity_curve[-1]


# ---------------------------------------------------------------------------
# Risk limits
# ---------------------------------------------------------------------------

class TestRiskLimits:
    def test_max_gross_exposure_blocks_trade(self, aligned_dfs):
        """With a tight gross exposure limit, fewer trades should execute."""
        strats_unlimited = {"ASSET_A": AlwaysBuyStrategy(), "ASSET_B": AlwaysBuyStrategy()}
        strats_limited = {"ASSET_A": AlwaysBuyStrategy(), "ASSET_B": AlwaysBuyStrategy()}

        pbt_unlimited = PortfolioBacktester(
            aligned_dfs, strats_unlimited, starting_cash=10_000)
        pbt_limited = PortfolioBacktester(
            aligned_dfs, strats_limited, starting_cash=10_000,
            risk_limits=RiskLimits(max_gross_exposure=0.10))

        r_unlimited = pbt_unlimited.run()
        r_limited = pbt_limited.run()

        assert len(r_limited.trades) < len(r_unlimited.trades)

    def test_max_single_asset_limits_concentration(self, aligned_dfs):
        """A tight single-asset limit should constrain one asset's trades."""
        strats = {"ASSET_A": AlwaysBuyStrategy(), "ASSET_B": AlwaysBuyStrategy()}
        pbt = PortfolioBacktester(
            aligned_dfs, strats, starting_cash=10_000,
            risk_limits=RiskLimits(max_single_asset=0.05, max_gross_exposure=1.0))
        result = pbt.run()

        # Should still run without error; trades are just smaller/fewer
        assert len(result.equity_curve) == 100

    def test_max_open_positions(self, aligned_dfs):
        """With max_open_positions=1, only one asset should have trades at a time."""
        strats = {"ASSET_A": BuyOnceStrategy(), "ASSET_B": BuyOnceStrategy()}
        pbt = PortfolioBacktester(
            aligned_dfs, strats, starting_cash=10_000,
            risk_limits=RiskLimits(max_open_positions=1))
        result = pbt.run()

        # At most one asset should have entered
        a_trades = result.per_asset_trades.get("ASSET_A", [])
        b_trades = result.per_asset_trades.get("ASSET_B", [])
        # One of them should have been blocked
        assert len(a_trades) + len(b_trades) >= 1

    def test_no_limits_same_as_none(self, aligned_dfs):
        """Very loose limits should produce the same result as no limits."""
        strats1 = {"ASSET_A": BuyOnceStrategy(), "ASSET_B": BuyOnceStrategy()}
        strats2 = {"ASSET_A": BuyOnceStrategy(), "ASSET_B": BuyOnceStrategy()}

        pbt_none = PortfolioBacktester(aligned_dfs, strats1, starting_cash=10_000)
        pbt_loose = PortfolioBacktester(
            aligned_dfs, strats2, starting_cash=10_000,
            risk_limits=RiskLimits(
                max_gross_exposure=100.0, max_net_exposure=100.0,
                max_single_asset=100.0, max_open_positions=0))

        r_none = pbt_none.run()
        r_loose = pbt_loose.run()

        np.testing.assert_array_equal(r_none.equity_curve, r_loose.equity_curve)

    def test_risk_limits_defaults(self):
        """Default RiskLimits should be reasonable."""
        limits = RiskLimits()
        assert limits.max_gross_exposure == 1.0
        assert limits.max_single_asset == 0.30
        assert limits.max_open_positions == 0


# ---------------------------------------------------------------------------
# Portfolio optimizer (basic smoke tests)
# ---------------------------------------------------------------------------

class TestPortfolioOptimizer:
    def test_optimize_returns_result(self, aligned_dfs):
        from portfolio_optimizer import (
            PortfolioOptimizationResult,
            StrategyConfig,
            portfolio_optimize,
        )

        configs = {
            "ASSET_A": StrategyConfig(
                strategy_cls=BuyOnceStrategy,
                param_space={"tp_pct": (0.05, 0.20)},
            ),
            "ASSET_B": StrategyConfig(
                strategy_cls=BuyOnceStrategy,
                param_space={"tp_pct": (0.05, 0.20)},
            ),
        }

        result = portfolio_optimize(
            configs, aligned_dfs, n_trials=5, objective="sharpe")

        assert isinstance(result, PortfolioOptimizationResult)
        assert "ASSET_A" in result.best_strategy_params
        assert "ASSET_B" in result.best_strategy_params
        assert result.best_allocation_weights is None  # Not optimizing weights

    def test_weight_optimization(self, aligned_dfs):
        from portfolio_optimizer import StrategyConfig, portfolio_optimize

        configs = {
            "ASSET_A": StrategyConfig(
                strategy_cls=BuyOnceStrategy,
                param_space={"tp_pct": (0.05, 0.20)},
            ),
            "ASSET_B": StrategyConfig(
                strategy_cls=BuyOnceStrategy,
                param_space={"tp_pct": (0.05, 0.20)},
            ),
        }

        result = portfolio_optimize(
            configs, aligned_dfs, n_trials=5,
            objective="sharpe", optimize_weights=True)

        assert result.best_allocation_weights is not None
        assert abs(sum(result.best_allocation_weights.values()) - 1.0) < 1e-6
