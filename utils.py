"""Utility functions for data fetching, frequency inference, and JSON sanitization."""

import math
import os
from typing import List

import numpy as np
import pandas as pd
import requests

from config import DATALAKE_URL

LOCAL_DATA_DIR = "./ohlc_data"
LOCAL_TICK_DIR = "./tick_data"


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


def load_ticks(
    path: str,
    start: str = None,
    end: str = None,
    max_ticks: int = 0,
) -> List["Tick"]:
    """Load tick data from a CSV file.

    Supports MetaTrader 5 tick export format (tab-separated with angle-bracket
    headers) and generic CSV (timestamp, bid, ask or timestamp, price).

    Parameters
    ----------
    path : str
        Path to the CSV file.
    start : str, optional
        Start date filter (inclusive), e.g. ``"2025-01-01"``.
    end : str, optional
        End date filter (inclusive), e.g. ``"2025-12-31"``.
    max_ticks : int
        Maximum ticks to load (0 = all). Useful for quick tests.

    Returns
    -------
    list of Tick
    """
    from backtesting.tick import Tick

    # Sniff the format from the first line
    with open(path, "r", encoding="utf-8-sig") as f:
        header_line = f.readline().strip()

    is_mt5 = "<DATE>" in header_line.upper() or "<BID>" in header_line.upper()

    if is_mt5:
        # MT5 tab-separated: <DATE> <TIME> <BID> <ASK> <LAST> <VOLUME> <FLAGS>
        df = pd.read_csv(
            path,
            sep="\t",
            dtype={"<LAST>": str, "<VOLUME>": str, "<FLAGS>": str},
            low_memory=False,
            nrows=max_ticks if max_ticks > 0 else None,
        )
        # Normalize column names
        df.columns = [c.strip("<>").lower() for c in df.columns]

        # Build timestamp from date + time
        df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"], format="mixed")

        # Mid price from bid/ask.
        # MT5 flag=4 means only ask updated (bid is NaN), flag=2 means only
        # bid updated. Forward-fill to carry the last known value.
        bid = pd.to_numeric(df["bid"], errors="coerce").ffill()
        ask = pd.to_numeric(df["ask"], errors="coerce").ffill()
        df["price"] = (bid + ask) / 2
        df["bid_clean"] = bid
        df["ask_clean"] = ask

        # Volume (often empty for CFDs)
        if "volume" in df.columns:
            df["vol"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
        else:
            df["vol"] = 0.0
    else:
        # Generic CSV: timestamp, price, volume OR timestamp, bid, ask, volume
        df = pd.read_csv(
            path,
            nrows=max_ticks if max_ticks > 0 else None,
        )
        df.columns = [c.strip().lower() for c in df.columns]
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")

        if "bid" in df.columns and "ask" in df.columns:
            bid = pd.to_numeric(df["bid"], errors="coerce")
            ask = pd.to_numeric(df["ask"], errors="coerce")
            df["price"] = (bid + ask) / 2
            df["bid_clean"] = bid
            df["ask_clean"] = ask
        elif "price" in df.columns:
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df["bid_clean"] = np.nan
            df["ask_clean"] = np.nan
        else:
            raise ValueError(f"Cannot find price columns in {path}. "
                             f"Expected 'price' or 'bid'+'ask'. Got: {list(df.columns)}")

        df["vol"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0.0)

    # Date range filter
    if start is not None:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["timestamp"] <= pd.Timestamp(end)]

    # Drop NaN prices
    df = df.dropna(subset=["price"])

    # Build Tick objects
    timestamps = df["timestamp"].values.astype("datetime64[ns]")
    prices = df["price"].values.astype(np.float64)
    volumes = df["vol"].values.astype(np.float64)
    bids = df["bid_clean"].values if "bid_clean" in df.columns else None
    asks = df["ask_clean"].values if "ask_clean" in df.columns else None

    ticks = []
    for i in range(len(prices)):
        ticks.append(Tick(
            ts=pd.Timestamp(timestamps[i]),
            price=prices[i],
            volume=volumes[i],
            bid=float(bids[i]) if bids is not None and not np.isnan(bids[i]) else None,
            ask=float(asks[i]) if asks is not None and not np.isnan(asks[i]) else None,
        ))

    return ticks


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
