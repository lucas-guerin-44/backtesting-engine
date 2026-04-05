"""Tests for OHLC data validation."""

import numpy as np
import pandas as pd
import pytest

from backtesting.data import Severity, validate_ohlc


def make_valid_df(n=50):
    """Create a valid OHLC DataFrame."""
    c = np.linspace(100, 150, n)
    return pd.DataFrame({
        "open": c - 0.5,
        "high": c + 1.0,
        "low": c - 1.0,
        "close": c,
    }, index=pd.date_range("2024-01-01", periods=n, freq="h"))


class TestValidData:
    def test_valid_df_passes(self):
        report = validate_ohlc(make_valid_df())
        assert report.is_valid
        assert report.n_errors == 0

    def test_valid_df_with_timestamp_column(self):
        df = make_valid_df()
        df = df.reset_index().rename(columns={"index": "timestamp"})
        report = validate_ohlc(df)
        assert report.is_valid

    def test_report_str_readable(self):
        report = validate_ohlc(make_valid_df())
        s = str(report)
        assert "passed" in s.lower() or "no issues" in s.lower()


class TestErrors:
    def test_missing_column(self):
        df = make_valid_df().drop(columns=["high"])
        report = validate_ohlc(df)
        assert not report.is_valid
        assert any(iss.field == "columns" for iss in report.issues)

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close"])
        report = validate_ohlc(df)
        assert not report.is_valid

    def test_non_numeric_column(self):
        df = make_valid_df()
        df["close"] = "not_a_number"
        report = validate_ohlc(df)
        assert not report.is_valid
        assert any(iss.field == "dtype" for iss in report.issues)

    def test_nan_in_ohlc(self):
        df = make_valid_df()
        df.iloc[5, df.columns.get_loc("close")] = np.nan
        report = validate_ohlc(df)
        assert not report.is_valid
        assert any(iss.field == "nan" for iss in report.issues)

    def test_inf_in_ohlc(self):
        df = make_valid_df()
        df.iloc[3, df.columns.get_loc("open")] = np.inf
        report = validate_ohlc(df)
        assert not report.is_valid
        assert any(iss.field == "inf" for iss in report.issues)

    def test_high_less_than_low(self):
        df = make_valid_df()
        # Swap high and low on one bar
        df.iloc[10, df.columns.get_loc("high")] = 90.0
        df.iloc[10, df.columns.get_loc("low")] = 110.0
        report = validate_ohlc(df)
        assert not report.is_valid
        assert any(iss.field == "high_low" for iss in report.issues)

    def test_high_less_than_close(self):
        df = make_valid_df()
        df.iloc[10, df.columns.get_loc("high")] = df.iloc[10]["close"] - 5.0
        report = validate_ohlc(df)
        assert not report.is_valid

    def test_duplicate_timestamps(self):
        df = make_valid_df()
        df.index = df.index[:49].append(df.index[48:49])  # Duplicate last timestamp
        report = validate_ohlc(df)
        assert not report.is_valid
        assert any("duplicate" in iss.message.lower() for iss in report.issues)

    def test_non_monotonic_timestamps(self):
        df = make_valid_df()
        idx = list(df.index)
        idx[10], idx[11] = idx[11], idx[10]  # Swap two timestamps
        df.index = pd.DatetimeIndex(idx)
        report = validate_ohlc(df)
        assert not report.is_valid
        assert any("monotonic" in iss.message.lower() for iss in report.issues)

    def test_raise_if_invalid_raises(self):
        df = make_valid_df().drop(columns=["high"])
        report = validate_ohlc(df)
        with pytest.raises(ValueError, match="validation failed"):
            report.raise_if_invalid()

    def test_raise_if_invalid_passes_when_valid(self):
        report = validate_ohlc(make_valid_df())
        report.raise_if_invalid()  # Should not raise


class TestWarnings:
    def test_gap_detection(self):
        df = make_valid_df(n=100)
        # Remove bars 40-60 to create a gap
        df = pd.concat([df.iloc[:40], df.iloc[60:]])
        report = validate_ohlc(df)
        assert report.is_valid  # Gaps are warnings, not errors
        assert report.n_warnings > 0
        assert any(iss.field == "gaps" for iss in report.issues)

    def test_gap_detection_with_expected_freq(self):
        df = make_valid_df(n=100)
        df = pd.concat([df.iloc[:40], df.iloc[60:]])
        report = validate_ohlc(df, expected_freq="1h")
        assert any(iss.field == "gaps" for iss in report.issues)

    def test_zero_price_is_warning(self):
        df = make_valid_df()
        df.iloc[5, df.columns.get_loc("low")] = 0.0
        report = validate_ohlc(df)
        # May or may not be valid depending on other checks, but should have a warning
        assert any(iss.field == "price" for iss in report.issues)

    def test_extreme_move_is_warning(self):
        df = make_valid_df()
        # 200% move in one bar
        df.iloc[20, df.columns.get_loc("close")] = df.iloc[19]["close"] * 3.0
        df.iloc[20, df.columns.get_loc("high")] = df.iloc[20]["close"] + 1.0
        report = validate_ohlc(df)
        assert any(iss.field == "extreme_move" for iss in report.issues)
