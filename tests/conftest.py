"""Shared fixtures for the test suite."""

import numpy as np
import pandas as pd
import pytest

from backtesting.types import Bar, Trade


@pytest.fixture
def sample_bars():
    """10 bars of synthetic OHLC data with a known uptrend then reversal."""
    return [
        Bar(pd.Timestamp("2024-01-01 09:00"), 100.0, 102.0, 99.0, 101.0),
        Bar(pd.Timestamp("2024-01-01 10:00"), 101.0, 103.0, 100.0, 102.0),
        Bar(pd.Timestamp("2024-01-01 11:00"), 102.0, 105.0, 101.0, 104.0),
        Bar(pd.Timestamp("2024-01-01 12:00"), 104.0, 106.0, 103.0, 105.0),
        Bar(pd.Timestamp("2024-01-01 13:00"), 105.0, 107.0, 104.0, 106.0),
        Bar(pd.Timestamp("2024-01-01 14:00"), 106.0, 106.5, 103.0, 104.0),  # reversal
        Bar(pd.Timestamp("2024-01-01 15:00"), 104.0, 104.5, 101.0, 102.0),
        Bar(pd.Timestamp("2024-01-01 16:00"), 102.0, 103.0, 100.0, 101.0),
        Bar(pd.Timestamp("2024-01-01 17:00"), 101.0, 102.0, 99.0, 100.0),
        Bar(pd.Timestamp("2024-01-01 18:00"), 100.0, 101.0, 98.0, 99.0),
    ]


@pytest.fixture
def sample_df(sample_bars):
    """DataFrame version of sample_bars, indexed by timestamp."""
    data = {
        "open": [b.open for b in sample_bars],
        "high": [b.high for b in sample_bars],
        "low": [b.low for b in sample_bars],
        "close": [b.close for b in sample_bars],
    }
    index = pd.DatetimeIndex([b.ts for b in sample_bars], name="timestamp")
    return pd.DataFrame(data, index=index)


@pytest.fixture
def flat_df():
    """50 bars of flat price data (close=100.0). No strategy should trade profitably."""
    n = 50
    data = {
        "open": [100.0] * n,
        "high": [100.5] * n,
        "low": [99.5] * n,
        "close": [100.0] * n,
    }
    index = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(data, index=index)


@pytest.fixture
def trending_df():
    """100 bars of a clean uptrend (close goes from 100 to 200). Good for trend strategies."""
    n = 100
    closes = np.linspace(100, 200, n)
    data = {
        "open": closes - 0.5,
        "high": closes + 1.0,
        "low": closes - 1.0,
        "close": closes,
    }
    index = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(data, index=index)
