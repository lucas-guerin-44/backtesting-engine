"""Tests for the results database."""

import numpy as np
import pytest

from results_db import ResultsDB, RunRecord


@pytest.fixture
def db(tmp_path):
    """Fresh database for each test."""
    return ResultsDB(path=str(tmp_path / "test.db"))


def make_equity(n=100, start=10_000, trend=0.001):
    return start * np.cumprod(1 + np.random.RandomState(42).normal(trend, 0.01, n))


class FakeTrade:
    def __init__(self, pnl):
        self.pnl = pnl


class TestSaveAndGet:
    def test_save_run_returns_id(self, db):
        run_id = db.save_run("TestStrategy", {"param1": 10}, make_equity(), [])
        assert isinstance(run_id, int)
        assert run_id > 0

    def test_get_run_returns_record(self, db):
        run_id = db.save_run("TestStrategy", {"param1": 10}, make_equity(),
                             [FakeTrade(50), FakeTrade(-20)])
        record = db.get_run(run_id)
        assert isinstance(record, RunRecord)
        assert record.strategy_name == "TestStrategy"
        assert record.params == {"param1": 10}
        assert record.win_count == 1
        assert record.loss_count == 1

    def test_get_nonexistent_run(self, db):
        assert db.get_run(9999) is None

    def test_metrics_computed_correctly(self, db):
        eq = np.array([10_000, 10_500, 10_200, 10_800], dtype=np.float64)
        trades = [FakeTrade(500), FakeTrade(-300), FakeTrade(600)]
        run_id = db.save_run("Test", {}, eq, trades, starting_cash=10_000)
        record = db.get_run(run_id)
        assert record.final_equity == 10_800
        assert abs(record.pct_return - 8.0) < 0.1
        assert record.total_trades == 3
        assert record.win_count == 2
        assert record.loss_count == 1


class TestQuery:
    def test_query_all(self, db):
        for i in range(3):
            db.save_run(f"Strategy{i}", {}, make_equity(), [])
        df = db.query_runs()
        assert len(df) == 3

    def test_filter_by_strategy(self, db):
        db.save_run("Alpha", {}, make_equity(), [])
        db.save_run("Beta", {}, make_equity(), [])
        df = db.query_runs(strategy="Alpha")
        assert len(df) == 1
        assert df.iloc[0]["strategy_name"] == "Alpha"

    def test_filter_by_min_sharpe(self, db):
        # Strong trend -> high Sharpe
        eq_good = 10_000 * np.cumprod(1 + np.full(100, 0.005))
        # Flat -> low Sharpe
        eq_bad = np.full(100, 10_000.0)
        db.save_run("Good", {}, eq_good, [])
        db.save_run("Bad", {}, eq_bad, [])
        df = db.query_runs(min_sharpe=1.0)
        assert len(df) >= 1
        assert all(df["sharpe"] >= 1.0)

    def test_query_limit(self, db):
        for i in range(10):
            db.save_run(f"S{i}", {}, make_equity(), [])
        df = db.query_runs(limit=3)
        assert len(df) == 3


class TestWalkForward:
    def test_save_and_retrieve_splits(self, db):
        class FakeWF:
            splits = [
                {"split": 0, "train_start": "2024-01-01", "train_end": "2024-03-01",
                 "test_start": "2024-03-01", "test_end": "2024-04-01",
                 "in_sample_score": 1.2, "out_of_sample_score": 0.5,
                 "oos_return_pct": 3.5, "oos_max_dd_pct": 2.1, "best_params": {"a": 1}},
                {"split": 1, "train_start": "2024-03-01", "train_end": "2024-05-01",
                 "test_start": "2024-05-01", "test_end": "2024-06-01",
                 "in_sample_score": 0.8, "out_of_sample_score": -0.3,
                 "oos_return_pct": -1.2, "oos_max_dd_pct": 3.5, "best_params": {"a": 2}},
            ]
            in_sample_mean = 1.0
            out_of_sample_mean = 0.1
            degradation = 0.9

        wf_id = db.save_walk_forward(FakeWF(), "TrendFollowing")
        splits = db.get_walk_forward_splits(wf_id)
        assert len(splits) == 2
        assert splits.iloc[0]["split_idx"] == 0


class TestDelete:
    def test_delete_run(self, db):
        run_id = db.save_run("Test", {}, make_equity(), [])
        assert db.delete_run(run_id)
        assert db.get_run(run_id) is None

    def test_delete_nonexistent(self, db):
        assert not db.delete_run(9999)


class TestUtility:
    def test_data_hash_deterministic(self):
        import pandas as pd
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        h1 = ResultsDB.compute_data_hash(df)
        h2 = ResultsDB.compute_data_hash(df)
        assert h1 == h2

    def test_data_hash_changes_with_data(self):
        import pandas as pd
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})
        assert ResultsDB.compute_data_hash(df1) != ResultsDB.compute_data_hash(df2)

    def test_context_manager(self, tmp_path):
        with ResultsDB(str(tmp_path / "cm.db")) as db:
            db.save_run("Test", {}, make_equity(), [])
        # Should not raise after close

    def test_numpy_params_serialized(self, db):
        """Numpy types in params should be serialized to JSON without error."""
        params = {"period": np.int64(20), "mult": np.float64(2.5)}
        run_id = db.save_run("Test", params, make_equity(), [])
        record = db.get_run(run_id)
        assert record.params["period"] == 20
        assert record.params["mult"] == 2.5
