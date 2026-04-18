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
class BacktestConfig:
    """Shared configuration for single-asset and portfolio backtesters.

    Consolidates parameters that are common to both ``Backtester`` and
    ``PortfolioBacktester`` into a single reusable object.
    """
    starting_cash: float = 10_000
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    max_leverage: float = 1.0
    margin_rate: float = 0.0


@dataclass
class Trade:
    """Represents a trade from entry through (optional) exit."""
    entry_bar: Bar
    side: int               # +1 for long, -1 for short
    size: float
    entry_price: float
    stop_price: float
    take_profit: float
    limit_price: Optional[float] = None  # Limit fill price (None = market at next open)
    bars_held: int = 0
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
