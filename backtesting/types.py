"""Core data types for the backtesting engine."""

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class Bar:
    """A single OHLC price bar."""
    ts: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None


@dataclass
class Trade:
    """Represents a trade from entry through (optional) exit."""
    entry_bar: Bar
    side: int               # +1 for long, -1 for short
    size: float
    entry_price: float
    stop_price: float
    take_profit: float
    bars_held: int = 0
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
