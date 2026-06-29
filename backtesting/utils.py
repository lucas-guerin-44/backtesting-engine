"""Internal utility functions for the backtesting package."""

import pandas as pd


def infer_freq_per_year(timestamps) -> int:
    """Infer the number of observation periods per year from timestamps.

    Uses the median time delta between bars to estimate the bar frequency,
    then converts to annual count. Falls back to 252 (daily) if inference fails.

    Parameters
    ----------
    timestamps : array-like
        Sequence of timestamps (pd.Timestamp, datetime, or DatetimeIndex).

    Returns
    -------
    int
        Estimated number of bars per year.
    """
    if len(timestamps) < 2:
        return 252

    ts = pd.DatetimeIndex(timestamps)
    deltas = ts[1:] - ts[:-1]
    median_delta = deltas.median()

    if median_delta.total_seconds() <= 0:
        return 252

    seconds_per_year = 365.25 * 24 * 3600
    bars_per_year = seconds_per_year / median_delta.total_seconds()

    # Clamp to reasonable range and round to nearest known frequency
    freq_map = [
        (500_000, 525_600),   # M1: ~525k bars/year
        (100_000, 105_120),   # M5: ~105k bars/year
        (30_000, 35_040),     # M15: ~35k bars/year
        (4_000, 6_570),       # H1: ~6.5k bars/year
        (1_000, 1_643),       # H4: ~1.6k bars/year
        (200, 252),           # D1: 252 bars/year
    ]

    for threshold, freq in freq_map:
        if bars_per_year >= threshold:
            return freq

    return 252
