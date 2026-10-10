"""Tests for utility functions."""

import math

import numpy as np
import pytest

import pandas as pd

from backtesting.statistics import compute_sharpe
import utils as data_utils
from utils import fetch_ohlc, infer_freq_per_year, normalize_tf, sanitize


class TestComputeSharpe:
    def test_flat_equity_returns_zero(self):
        eq = [100.0] * 50
        assert compute_sharpe(eq) == 0.0

    def test_positive_returns_positive_sharpe(self):
        eq = list(range(100, 200))
        sharpe = compute_sharpe(eq)
        assert sharpe > 0

    def test_negative_returns_negative_sharpe(self):
        eq = list(range(200, 100, -1))
        sharpe = compute_sharpe(eq)
        assert sharpe < 0

    def test_single_value_returns_zero(self):
        assert compute_sharpe([100.0]) == 0.0

    def test_empty_returns_zero(self):
        assert compute_sharpe([]) == 0.0

    def test_annualization(self):
        eq = list(range(100, 200))
        annualized = compute_sharpe(eq, annualize=True)
        raw = compute_sharpe(eq, annualize=False)
        assert abs(annualized) > abs(raw)


class TestInferFreqPerYear:
    def test_daily_bars(self):
        ts = pd.date_range("2022-01-01", "2024-01-01", freq="B")  # Business days
        assert 250 <= infer_freq_per_year(ts) <= 263

    def test_intraday_session_bars_reflect_market_closures(self):
        days = pd.date_range("2022-01-01", "2024-01-01", freq="B")
        session_hours = pd.to_timedelta([f"{hour}:00:00" for hour in range(10, 16)])
        ts = pd.DatetimeIndex([
            day + offset for day in days for offset in session_hours
        ])
        freq = infer_freq_per_year(ts)
        assert 1_500 <= freq <= 1_600

    def test_5min_bars(self):
        ts = pd.date_range("2024-01-01", periods=100, freq="5min")
        assert infer_freq_per_year(ts) == 105_192

    def test_4h_bars(self):
        ts = pd.date_range("2024-01-01", periods=100, freq="4h")
        assert infer_freq_per_year(ts) == 2_192

    def test_short_sample_uses_observed_calendarized_rate(self):
        ts = pd.date_range("2024-01-01", periods=100, freq="B")
        assert infer_freq_per_year(ts) == 264

    def test_single_timestamp_returns_default(self):
        ts = pd.DatetimeIndex(["2024-01-01"])
        assert infer_freq_per_year(ts) == 252

    def test_empty_returns_default(self):
        assert infer_freq_per_year([]) == 252


class TestNormalizeTf:
    def test_known_mappings(self):
        assert normalize_tf("M1") == "1min"
        assert normalize_tf("M5") == "5min"
        assert normalize_tf("M15") == "15min"
        assert normalize_tf("H1") == "1h"
        assert normalize_tf("H4") == "4h"
        assert normalize_tf("D1") == "1D"

    def test_unknown_passthrough(self):
        assert normalize_tf("W1") == "W1"


def test_cached_and_fresh_data_apply_identical_session_filters(tmp_path, monkeypatch):
    rows = [
        {"instrument": "TEST", "timeframe": "M1", "timestamp": "2024-01-02T14:00:00Z",
         "open": 100, "high": 101, "low": 99, "close": 100},  # Before RTH
        {"instrument": "TEST", "timeframe": "M1", "timestamp": "2024-01-02T14:30:00Z",
         "open": 101, "high": 102, "low": 100, "close": 101},  # RTH open
        {"instrument": "TEST", "timeframe": "M1", "timestamp": "2024-01-02T20:59:00Z",
         "open": 102, "high": 103, "low": 101, "close": 102},  # Before RTH close
        {"instrument": "TEST", "timeframe": "M1", "timestamp": "2024-01-02T21:00:00Z",
         "open": 103, "high": 104, "low": 102, "close": 103},  # At RTH close
        {"instrument": "TEST", "timeframe": "M1", "timestamp": "2024-01-03T15:00:00Z",
         "open": 104, "high": 105, "low": 103, "close": 104},  # Inclusive end date
    ]
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    pd.DataFrame(rows).to_csv(cache_dir / "TEST_M1.csv", index=False)

    monkeypatch.setattr(data_utils, "LOCAL_DATA_DIR", str(cache_dir))
    cached = fetch_ohlc(
        "TEST", "M1", "2024-01-02", "2024-01-03",
        session_tz="America/New_York", rth_open=930, rth_close=1600,
        weekday_filter=[1, 2],
    )

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return rows

    monkeypatch.setattr(data_utils.requests, "get", lambda *args, **kwargs: Response())
    fresh_dir = tmp_path / "fresh"
    fresh_dir.mkdir()
    monkeypatch.setattr(data_utils, "LOCAL_DATA_DIR", str(fresh_dir))
    fresh = fetch_ohlc(
        "TEST", "M1", "2024-01-02", "2024-01-03",
        session_tz="America/New_York", rth_open=930, rth_close=1600,
        weekday_filter=[1, 2],
    )

    pd.testing.assert_index_equal(cached.index, fresh.index)
    pd.testing.assert_frame_equal(cached, fresh, check_dtype=False)
    assert len(cached) == 3


class TestSanitize:
    def test_nan_becomes_none(self):
        assert sanitize(float("nan")) is None

    def test_inf_becomes_none(self):
        assert sanitize(float("inf")) is None
        assert sanitize(float("-inf")) is None

    def test_normal_float_unchanged(self):
        assert sanitize(3.14) == 3.14

    def test_nested_dict(self):
        data = {"a": float("nan"), "b": 1.0, "c": {"d": float("inf")}}
        result = sanitize(data)
        assert result == {"a": None, "b": 1.0, "c": {"d": None}}

    def test_list_with_nans(self):
        data = [1.0, float("nan"), 3.0]
        result = sanitize(data)
        assert result == [1.0, None, 3.0]

    def test_non_float_unchanged(self):
        assert sanitize("hello") == "hello"
        assert sanitize(42) == 42
        assert sanitize(None) is None
