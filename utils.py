"""Utility functions for data fetching, metric computation, and JSON sanitization."""

import math
import os

import numpy as np
import pandas as pd
import requests

from config import DATALAKE_URL

LOCAL_DATA_DIR = "./ohlc_data"


def fetch_ohlc(instrument: str, timeframe: str, start_date: str, end_date: str, limit: int = 0) -> pd.DataFrame:
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
        start_ts = pd.to_datetime(start_date).tz_localize("UTC")
        end_ts = pd.to_datetime(end_date).tz_localize("UTC")
        df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
        if not df.empty:
            return df

    url = f"{DATALAKE_URL}/query"
    page_size = 10_000
    params = {
        "instrument": instrument,
        "timeframe": timeframe,
        "start": f"{start_date}T00:00:00",
        "end": f"{end_date}T23:59:59",
        "limit": page_size,
    }

    all_rows = []
    while True:
        resp = requests.get(url, params=params)
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
    return df_combined


def compute_sharpe(equity_curve, risk_free: float = 0.0, annualize: bool = True, freq_per_year: int = 252) -> float:
    """Compute Sharpe ratio from an equity curve.

    Parameters
    ----------
    equity_curve : array-like
        Sequence of portfolio equity values over time.
    risk_free : float
        Annual risk-free rate (default 0).
    annualize : bool
        Whether to annualize the ratio.
    freq_per_year : int
        Number of observation periods per year (252 for daily bars).

    Returns
    -------
    float
        The Sharpe ratio, or 0.0 if it cannot be computed.
    """
    equity_curve = np.array(equity_curve, dtype=float)

    valid_idx = equity_curve[:-1] > 1e-8
    if not np.any(valid_idx):
        return 0.0

    returns = np.diff(equity_curve)[valid_idx] / equity_curve[:-1][valid_idx]
    excess_returns = returns - risk_free / freq_per_year

    if np.std(excess_returns) == 0:
        return 0.0

    sharpe = np.mean(excess_returns) / np.std(excess_returns)
    if annualize:
        sharpe *= np.sqrt(freq_per_year)

    return sharpe


def normalize_tf(tf: str) -> str:
    """Convert MT4/MT5 timeframe codes to pandas-compatible frequency strings."""
    mapping = {
        "M1": "1min",
        "M5": "5min",
        "M15": "15min",
        "H1": "1h",
        "H4": "4h",
        "D1": "1D",
    }
    return mapping.get(tf, tf)


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
