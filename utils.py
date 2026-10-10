"""Utility functions for data fetching, frequency inference, and JSON sanitization."""

import math
import os

import numpy as np
import pandas as pd
import requests

from config import DATALAKE_API_KEY, DATALAKE_URL
from backtesting.utils import infer_freq_per_year

LOCAL_DATA_DIR = "./ohlc_data"

# MT4/MT5 timeframe codes -> pandas frequency strings.
_FREQ_MAP = {
    "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
    "H1": "1h", "H4": "4h", "D1": "1D",
}


def fetch_ohlc(
    instrument: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    limit: int = 0,
    session_tz: str = None,
    rth_open: int = None,
    rth_close: int = None,
    weekday_filter: list = None,
) -> pd.DataFrame:
    """Fetch OHLC data, using a local CSV cache with API fallback.

    Handles both paginated responses (``{data, pagination}``) from the
    datalake API and plain JSON arrays from simpler endpoints.

    Parameters
    ----------
    instrument : str
        The instrument symbol, e.g., "XAUUSD".
    timeframe : str
        The timeframe, e.g., "M5", "H1", "D1".
    start_date : str
        Start date in "YYYY-MM-DD" format.
    end_date : str
        End date in "YYYY-MM-DD" format.
    limit : int
        Maximum total rows to fetch (0 = all available data).
    session_tz : str, optional
        Timezone for session filtering, e.g. "America/New_York".
        When set, timestamps are converted to this timezone before
        filtering. Combined with rth_open/rth_close to restrict to
        regular trading hours.
    rth_open : int, optional
        Regular trading hours open time as HHMM integer (e.g. 930 for 9:30).
        Requires session_tz. Bars before this time are dropped.
    rth_close : int, optional
        Regular trading hours close time as HHMM integer (e.g. 1600 for 16:00).
        Requires session_tz. Bars at or after this time are dropped.
    weekday_filter : list, optional
        List of weekday integers to keep (0=Mon .. 4=Fri).
        If None, all weekdays are kept.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: instrument, timeframe, timestamp, open, high, low, close.
    """
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

    filepath = os.path.join(LOCAL_DATA_DIR, f"{instrument}_{timeframe}.csv")

    df = pd.DataFrame()
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, parse_dates=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        start_ts = start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts.tz_convert("UTC")
        end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
        # Date-only end values are inclusive, matching the API query bounds.
        if end_ts == end_ts.normalize():
            end_ts += pd.Timedelta(hours=23, minutes=59, seconds=59)
        df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
        if not df.empty:
            return _apply_session_filter(
                df, session_tz, rth_open, rth_close, weekday_filter,
            )

    url = f"{DATALAKE_URL}/query"
    page_size = 10_000
    params = {
        "instrument": instrument,
        "timeframe": timeframe,
        "start": f"{start_date}T00:00:00",
        "end": f"{end_date}T23:59:59",
        "limit": page_size,
    }

    headers = {"X-API-Key": DATALAKE_API_KEY} if DATALAKE_API_KEY else {}
    all_rows = []
    while True:
        resp = requests.get(url, params=params, headers=headers)
        resp.raise_for_status()
        body = resp.json()

        # Handle both paginated {data, pagination} and plain array responses
        if isinstance(body, dict) and "data" in body:
            rows = body["data"]
            pagination = body.get("pagination", {})
        elif isinstance(body, list):
            rows = body
            pagination = {}
        else:
            break

        if not rows:
            break
        all_rows.extend(rows)

        # Stop if caller requested a specific limit and we have enough
        if limit > 0 and len(all_rows) >= limit:
            all_rows = all_rows[:limit]
            break

        # Follow cursor for next page, or stop if no more data
        if pagination.get("has_more") and pagination.get("next_cursor"):
            params["cursor"] = pagination["next_cursor"]
        else:
            break

    if not all_rows:
        return pd.DataFrame(columns=["instrument", "timeframe", "timestamp", "open", "high", "low", "close"])

    df_new = pd.DataFrame(all_rows)
    df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], utc=True)

    if not df.empty:
        df_combined = pd.concat([df, df_new]).drop_duplicates(subset="timestamp").sort_values("timestamp")
    else:
        df_combined = df_new

    df_combined.to_csv(filepath, index=False)

    # Apply session filtering if requested
    df_combined = _apply_session_filter(
        df_combined, session_tz, rth_open, rth_close, weekday_filter,
    )

    return df_combined


def _apply_session_filter(
    df: pd.DataFrame,
    session_tz: str = None,
    rth_open: int = None,
    rth_close: int = None,
    weekday_filter: list = None,
) -> pd.DataFrame:
    """Apply session timezone conversion and regular trading hours filtering.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC data with 'timestamp' column (UTC).
    session_tz : str, optional
        Target timezone, e.g. "America/New_York".
    rth_open : int, optional
        Open time as HHMM (e.g. 930).
    rth_close : int, optional
        Close time as HHMM (e.g. 1600).
    weekday_filter : list, optional
        Weekday integers to keep (0=Mon .. 4=Fri).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame. If no filters are applied, returns unchanged.
    """
    if df.empty:
        return df

    needs_index = session_tz is not None or rth_open is not None or rth_close is not None or weekday_filter is not None
    if not needs_index:
        return df

    # Ensure timestamp is the index for timezone conversion
    ts_col = "timestamp"
    if ts_col not in df.columns:
        return df

    result = df.copy()
    result[ts_col] = pd.to_datetime(result[ts_col], utc=True)

    if session_tz:
        result[ts_col] = result[ts_col].dt.tz_convert(session_tz)
        result = result.set_index(ts_col).sort_index()
    else:
        result = result.set_index(ts_col).sort_index()

    # Weekday filter (0=Monday .. 6=Sunday)
    if weekday_filter is not None:
        result = result[result.index.weekday.isin(weekday_filter)]

    # RTH filter
    if rth_open is not None and rth_close is not None:
        open_h, open_m = divmod(rth_open, 100)
        close_h, close_m = divmod(rth_close, 100)
        time_open = open_h * 60 + open_m
        time_close = close_h * 60 + close_m
        minutes = result.index.hour * 60 + result.index.minute
        result = result[(minutes >= time_open) & (minutes < time_close)]
    elif rth_open is not None:
        open_h, open_m = divmod(rth_open, 100)
        time_open = open_h * 60 + open_m
        minutes = result.index.hour * 60 + result.index.minute
        result = result[minutes >= time_open]
    elif rth_close is not None:
        close_h, close_m = divmod(rth_close, 100)
        time_close = close_h * 60 + close_m
        minutes = result.index.hour * 60 + result.index.minute
        result = result[minutes < time_close]

    return result


def normalize_tf(tf: str) -> str:
    """Convert MT4/MT5 timeframe codes to pandas-compatible frequency strings."""
    return _FREQ_MAP.get(tf, tf)


def sanitize(obj):
    """Recursively convert NaN/inf floats to None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj
