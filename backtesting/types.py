"""Core data types for the backtesting engine."""

from dataclasses import dataclass, field
from typing import List, Optional, Union

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

    @classmethod
    def at_price(cls, ts, price: float) -> "Bar":
        """Create a bar where all OHLC values are the same (synthetic bar)."""
        return cls(ts=ts, open=price, high=price, low=price, close=price)


@dataclass
class BacktestConfig:
    """Shared configuration for single-asset and portfolio backtesters.

    Consolidates parameters that are common to both ``Backtester`` and
    ``PortfolioBacktester`` into a single reusable object.
    """
    starting_cash: float = 10_000
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    spread_bps: float = 0.0
    max_leverage: float = 1.0
    margin_rate: float = 0.0
    typical_daily_volume: Optional[float] = None
    impact_scaling: float = 0.5
    daily_volatility: Optional[float] = None
    funding_rate_annual: float = 0.0
    funding_rate_short: float = 0.0
    margin_call_threshold: float = 0.5  # Liquidate when equity < threshold * margin requirement

    def to_kwargs(self) -> dict:
        """Return all fields as a dict for passing to Portfolio()."""
        from dataclasses import asdict
        d = asdict(self)
        d["cash"] = d.pop("starting_cash")
        return d


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
    symbol: Optional[str] = None
    exit_ts: Optional[pd.Timestamp] = None
    metadata: Optional[dict] = None


@dataclass
class Lot:
    """A single entry lot within a position (for netting)."""
    entry_bar: Bar
    side: int
    size: float
    entry_price: float
    stop_price: float
    take_profit: float
    symbol: Optional[str] = None
    bars_held: int = 0
    metadata: Optional[dict] = None


class Position:
    """Manages lots for a single symbol with FIFO/LIFO netting.

    Duck-types ``List[Trade]`` so existing code that accesses
    ``broker.positions[sym]`` as a list continues to work.

    Parameters
    ----------
    method : str
        Netting method: ``"fifo"`` (first-in-first-out) or ``"lifo"``
        (last-in-first-out). Default ``"fifo"``.
    """

    def __init__(self, method: str = "fifo"):
        self.lots: List[Lot] = []
        self.method = method

    # --- List duck-typing (backward compatibility) ---

    def __getitem__(self, idx):
        return self.lots[idx]

    def __len__(self):
        return len(self.lots)

    def __iter__(self):
        return iter(self.lots)

    def append(self, lot: Lot):
        self.lots.append(lot)

    def remove(self, lot: Lot):
        self.lots.remove(lot)

    def __bool__(self):
        return len(self.lots) > 0

    # --- Query methods ---

    @property
    def net_size(self) -> float:
        """Total signed size (+long, -short)."""
        return sum(lot.side * lot.size for lot in self.lots)

    @property
    def net_side(self) -> int:
        """Net direction: +1 long, -1 short, 0 flat."""
        ns = self.net_size
        if ns > 0:
            return 1
        elif ns < 0:
            return -1
        return 0

    @property
    def average_entry(self) -> float:
        """Volume-weighted average entry price."""
        if not self.lots:
            return 0.0
        total_size = sum(lot.size for lot in self.lots)
        if total_size == 0:
            return 0.0
        return sum(lot.entry_price * lot.size for lot in self.lots) / total_size

    def unrealized_pnl(self, current_price: float) -> float:
        """Unrealized P&L at the given current price."""
        return sum(
            (current_price - lot.entry_price) * lot.side * lot.size
            for lot in self.lots
        )

    def close(self, method: Optional[str] = None, count: Optional[int] = None) -> List[Lot]:
        """Remove and return lots according to the netting method.

        Parameters
        ----------
        method : str, optional
            Override the default method for this close.
        count : int, optional
            Number of lots to close. None = close all matching direction.

        Returns
        -------
        list of Lot
            The lots that were removed from the position.
        """
        if not self.lots:
            return []

        method = method or self.method

        if method == "lifo":
            ordered = list(reversed(self.lots))
        else:
            ordered = list(self.lots)

        to_close = ordered if count is None else ordered[:count]

        closed = []
        for lot in to_close:
            self.lots.remove(lot)
            closed.append(lot)
        return closed

    def close_all(self) -> List[Lot]:
        """Close all lots and return them."""
        closed = list(self.lots)
        self.lots.clear()
        return closed
