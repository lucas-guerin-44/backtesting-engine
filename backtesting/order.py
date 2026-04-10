"""Order types for latency-aware execution.

Provides the Order abstraction used by LatencyAwareBroker. Unlike Trade
(which represents a position already filled), an Order represents an
*intent* to trade that hasn't been matched yet.

Three order types:
- MARKET : fill at best available price once the latency delay elapses.
- LIMIT  : fill only if price crosses the limit level (after delay).
- STOP   : activate when price crosses stop_trigger, then fill as MARKET.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


@dataclass
class Order:
    """An order submitted to the LatencyAwareBroker.

    Parameters
    ----------
    type : OrderType
        MARKET, LIMIT, or STOP.
    symbol : str
        Instrument identifier.
    side : int
        +1 for long, -1 for short.
    qty : float
        Number of units to trade.
    protective_stop : float
        Stop-loss price once the order fills (the SL on the resulting Trade).
    take_profit : float or None
        Take-profit price once filled.
    submitted_at : pd.Timestamp
        Timestamp when the strategy submitted the order.
    limit_price : float or None
        Fill price for LIMIT orders. The order only fills when the market
        price crosses this level.
    stop_trigger : float or None
        Activation price for STOP orders. The order converts to a MARKET
        order once tick price crosses this level.
    order_id : str
        Auto-generated 8-char hex identifier.
    """

    type: OrderType
    symbol: str
    side: int
    qty: float
    protective_stop: float
    take_profit: Optional[float]
    submitted_at: pd.Timestamp
    limit_price: Optional[float] = None
    stop_trigger: Optional[float] = None
    order_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


@dataclass
class PendingOrder:
    """An order queued inside LatencyAwareBroker, waiting for its fill time.

    Parameters
    ----------
    order : Order
        The original submitted order.
    fill_after_ns : int
        Absolute nanosecond timestamp after which this order is eligible
        to be matched against tick prices.
    activated : bool
        For STOP orders: True once price has crossed ``stop_trigger``.
        Unused for MARKET and LIMIT orders.
    """

    order: Order
    fill_after_ns: int
    activated: bool = False
