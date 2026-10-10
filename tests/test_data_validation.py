"""Tests for OHLC data validation."""

import numpy as np
import pandas as pd
import pytest

from backtesting.data import Severity, ValidationConfig, validate_ohlc


def make_valid_df(n=50):
    """Create a valid OHLC DataFrame."""
    c = np.linspace(100, 150, n)
    return pd.DataFrame({
        "open": c - 0.5,
        "high": c + 1.0,
        "low": c - 1.0,
        "close": c,
    }, index=pd.date_range("2024-01-01", periods=n, freq="h"))


def make_valid_daily(n=30, start="2024-01-01"):
    """Create a valid daily OHLC DataFrame."""
    c = np.linspace(100, 130, n)
    return pd.DataFrame({
        "open": c - 0.5,
        "high": c + 1.0,
        "low": c - 1.0,
        "close": c,
    }, index=pd.date_range(start, periods=n, freq="D"))


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


class TestDuplicateCalendarDates:
    def test_duplicate_daily_date_is_error(self):
        df = make_valid_daily()
        # Two distinct timestamps sharing one calendar date (corrupted daily feed)
        extra = pd.DatetimeIndex([df.index[4] + pd.Timedelta(hours=12)])
        df = pd.concat([df, df.iloc[[4]].set_axis(extra)])
        df = df.sort_index()
        assert df.index.duplicated().sum() == 0  # timestamps are unique
        report = validate_ohlc(df)
        assert not report.is_valid
        assert any(iss.field == "duplicate_dates" for iss in report.issues)

    def test_intraday_dates_not_checked_by_default(self):
        # Hourly bars repeat dates legitimately -- must not be flagged.
        report = validate_ohlc(make_valid_df(n=100))
        assert not any(iss.field == "duplicate_dates" for iss in report.issues)

    def test_duplicate_dates_can_be_downgraded(self):
        df = make_valid_daily()
        extra = pd.DatetimeIndex([df.index[4] + pd.Timedelta(hours=12)])
        df = pd.concat([df, df.iloc[[4]].set_axis(extra)]).sort_index()
        config = ValidationConfig(severity={"duplicate_dates": Severity.WARNING})
        report = validate_ohlc(df, config=config)
        assert report.is_valid
        assert any(iss.field == "duplicate_dates" for iss in report.issues)


class TestTradingCalendar:
    def test_weekend_not_checked_by_default(self):
        # 2024-01-06 is a Saturday; default config must not flag it.
        df = make_valid_daily(n=5, start="2024-01-06")
        report = validate_ohlc(df)
        assert not any(iss.field == "non_trading_day" for iss in report.issues)
        assert report.is_valid

    def test_weekend_is_error_when_enabled(self):
        df = make_valid_daily(n=5, start="2024-01-06")
        report = validate_ohlc(df, config=ValidationConfig(check_trading_calendar=True))
        assert not report.is_valid
        assert any(iss.field == "non_trading_day" for iss in report.issues)

    def test_custom_trading_days_allows_weekend(self):
        df = make_valid_daily(n=5, start="2024-01-06")  # Sat..Wed
        report = validate_ohlc(df, config=ValidationConfig(trading_days={5, 6, 0, 1, 2}))
        assert not any(iss.field == "non_trading_day" for iss in report.issues)

    def test_explicit_holiday_flagged(self):
        df = make_valid_daily(n=5, start="2024-01-08")  # Mon..Fri
        report = validate_ohlc(
            df, config=ValidationConfig(holidays={"2024-01-10"})
        )
        assert any(iss.field == "non_trading_day" for iss in report.issues)


class TestSplitJumps:
    def _with_jump(self, multiple=3.0):
        df = make_valid_df()
        new = df.iloc[19]["close"] * multiple
        for col, val in (("open", new), ("high", new + 1.0), ("low", new - 1.0), ("close", new)):
            df.iloc[20, df.columns.get_loc(col)] = val
        return df

    def test_split_jump_is_error_by_default(self):
        report = validate_ohlc(self._with_jump())
        assert not report.is_valid
        assert any(iss.field == "split_jump" for iss in report.issues)

    def test_split_jump_downgradeable_to_warning(self):
        config = ValidationConfig(severity={"split_jump": Severity.WARNING})
        report = validate_ohlc(self._with_jump(), config=config)
        assert report.is_valid
        assert any(iss.field == "split_jump" for iss in report.issues)

    def test_split_jump_threshold_configurable(self):
        df = make_valid_df()
        df.iloc[20, df.columns.get_loc("close")] = df.iloc[19]["close"] * 1.10
        df.iloc[20, df.columns.get_loc("high")] = df.iloc[20]["close"] + 1.0
        assert not any(
            iss.field == "split_jump"
            for iss in validate_ohlc(df).issues
        )
        report = validate_ohlc(df, config=ValidationConfig(split_jump_threshold=0.05))
        assert any(iss.field == "split_jump" for iss in report.issues)


class TestSourceProvenance:
    def test_mixed_source_is_error(self):
        df = make_valid_df()
        df["source"] = "mt5"
        df.iloc[5, df.columns.get_loc("source")] = "yahoo"
        report = validate_ohlc(df)
        assert not report.is_valid
        assert any(iss.field == "source" for iss in report.issues)

    def test_missing_source_values_is_error(self):
        df = make_valid_df()
        df["source"] = "mt5"
        df.iloc[0, df.columns.get_loc("source")] = None
        report = validate_ohlc(df)
        assert not report.is_valid
        assert any(iss.field == "source" for iss in report.issues)

    def test_uniform_source_passes(self):
        df = make_valid_df()
        df["source"] = "mt5"
        report = validate_ohlc(df)
        assert not any(iss.field == "source" for iss in report.issues)

    def test_required_provenance_missing_column_is_error(self):
        report = validate_ohlc(make_valid_df(), config=ValidationConfig(require_provenance=True))
        assert not report.is_valid
        assert any(iss.field == "source" for iss in report.issues)

    def test_mixed_source_downgradeable(self):
        df = make_valid_df()
        df["source"] = ["mt5" if i % 2 else "yahoo" for i in range(len(df))]
        config = ValidationConfig(severity={"source": Severity.WARNING})
        report = validate_ohlc(df, config=config)
        assert report.is_valid
        assert any(iss.field == "source" for iss in report.issues)


class TestSeverityOverrides:
    def test_existing_check_can_be_upgraded(self):
        df = make_valid_df(n=100)
        df = pd.concat([df.iloc[:40], df.iloc[60:]])  # create a gap
        assert validate_ohlc(df).is_valid  # gaps are warnings by default
        config = ValidationConfig(severity={"gaps": Severity.ERROR})
        assert not validate_ohlc(df, config=config).is_valid

    def test_default_severities_exported(self):
        from backtesting.data import DEFAULT_SEVERITIES
        for name in ("duplicate_dates", "non_trading_day", "split_jump", "source"):
            assert DEFAULT_SEVERITIES[name] is Severity.ERROR
