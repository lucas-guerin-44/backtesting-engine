"""Tests for utility functions."""

import math

import numpy as np
import pytest

import pandas as pd

from backtesting.statistics import compute_sharpe
from utils import infer_freq_per_year, normalize_tf, sanitize


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
        ts = pd.date_range("2024-01-01", periods=100, freq="B")  # Business days
        assert infer_freq_per_year(ts) == 252

    def test_hourly_bars(self):
        ts = pd.date_range("2024-01-01", periods=100, freq="h")
        assert infer_freq_per_year(ts) == 6_570

    def test_5min_bars(self):
        ts = pd.date_range("2024-01-01", periods=100, freq="5min")
        assert infer_freq_per_year(ts) == 105_120

    def test_4h_bars(self):
        ts = pd.date_range("2024-01-01", periods=100, freq="4h")
        assert infer_freq_per_year(ts) == 1_643

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
