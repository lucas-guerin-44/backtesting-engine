"""Internal utility functions for the backtesting package."""

import pandas as pd


def infer_freq_per_year(timestamps) -> int:
    """Estimate observed periods per calendar year from timestamps.

    The estimate uses the number of intervals divided by the elapsed calendar
    time. Unlike a timeframe lookup, this accounts for sessions and closures
    represented in the supplied history. Short samples, irregular/event bars,
    and partial datasets can still produce a poor estimate; callers should pass
    an explicit ``freq_per_year`` when they need a particular convention.

    Parameters
    ----------
    timestamps : array-like
        Sequence of timestamps (pd.Timestamp, datetime, or DatetimeIndex).

    Returns
    -------
    int
        Estimated number of observations per calendar year, or 252 if it cannot
        be estimated.
    """
    if len(timestamps) < 2:
        return 252

    ts = pd.DatetimeIndex(timestamps)
    elapsed_seconds = (ts[-1] - ts[0]).total_seconds()
    if elapsed_seconds <= 0:
        return 252

    seconds_per_year = 365.25 * 24 * 3600
    intervals_per_year = (len(ts) - 1) * seconds_per_year / elapsed_seconds
    return max(1, int(round(intervals_per_year)))
