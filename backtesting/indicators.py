"""Incremental and vectorized indicator computations.

Two APIs:
- **Incremental classes** (``EMA``, ``ATR``, ``RSI``, ``BollingerBands``):
  call ``.update(price)`` or ``.update_bar(bar)`` per bar. O(1) per update,
  no allocations. Use these inside strategy ``on_bar()`` loops.

- **Vectorized functions** (``ema_array``, ``atr_array``, ``rsi_array``):
  take full numpy arrays, return full indicator arrays. Use these for
  pre-computation in the optimizer or analysis scripts.
"""

import numpy as np
from typing import Optional

from backtesting.types import Bar


# ---------------------------------------------------------------------------
# Incremental indicators (O(1) per bar, zero allocations)
# ---------------------------------------------------------------------------

class EMA:
    """Incremental Exponential Moving Average.

    >>> ind = EMA(period=20)
    >>> for price in prices:
    ...     val = ind.update(price)  # None until warmed up
    """
    __slots__ = ("period", "alpha", "value", "count")

    def __init__(self, period: int):
        self.period = period
        self.alpha = 2.0 / (period + 1)
        self.value: Optional[float] = None
        self.count = 0

    def update(self, price: float) -> Optional[float]:
        self.count += 1
        if self.value is None:
            self.value = price
        else:
            self.value = self.alpha * price + (1.0 - self.alpha) * self.value
        return self.value if self.count >= self.period else None


class ATR:
    """Incremental Average True Range.

    Uses exponential smoothing (Wilder's method) for O(1) updates.

    >>> ind = ATR(period=14)
    >>> for bar in bars:
    ...     val = ind.update(bar.high, bar.low, bar.close)
    """
    __slots__ = ("period", "alpha", "value", "prev_close", "count")

    def __init__(self, period: int):
        self.period = period
        self.alpha = 1.0 / period  # Wilder smoothing
        self.value: Optional[float] = None
        self.prev_close: Optional[float] = None
        self.count = 0

    def update(self, high: float, low: float, close: float) -> Optional[float]:
        if self.prev_close is not None:
            tr = max(high - low, abs(high - self.prev_close), abs(low - self.prev_close))
        else:
            tr = high - low

        self.prev_close = close
        self.count += 1

        if self.value is None:
            self.value = tr
        else:
            self.value = self.alpha * tr + (1.0 - self.alpha) * self.value

        return self.value if self.count >= self.period else None


class RSI:
    """Incremental Relative Strength Index (Wilder smoothing).

    >>> ind = RSI(period=14)
    >>> for price in prices:
    ...     val = ind.update(price)  # None until warmed up
    """
    __slots__ = ("period", "alpha", "avg_gain", "avg_loss", "prev_price", "count")

    def __init__(self, period: int = 14):
        self.period = period
        self.alpha = 1.0 / period
        self.avg_gain = 0.0
        self.avg_loss = 0.0
        self.prev_price: Optional[float] = None
        self.count = 0

    def update(self, price: float) -> Optional[float]:
        if self.prev_price is not None:
            delta = price - self.prev_price
            gain = max(delta, 0.0)
            loss = max(-delta, 0.0)

            self.avg_gain = self.alpha * gain + (1.0 - self.alpha) * self.avg_gain
            self.avg_loss = self.alpha * loss + (1.0 - self.alpha) * self.avg_loss
            self.count += 1

        self.prev_price = price

        if self.count < self.period:
            return None
        if self.avg_loss == 0:
            return 100.0
        rs = self.avg_gain / self.avg_loss
        return 100.0 - (100.0 / (1.0 + rs))


class BollingerBands:
    """Incremental Bollinger Bands using a rolling window.

    Maintains a fixed-size circular buffer for O(1) mean/std updates.

    >>> bb = BollingerBands(period=20, num_std=2.0)
    >>> for price in prices:
    ...     lower, mid, upper = bb.update(price)  # (None,None,None) until warmed up
    """
    __slots__ = ("period", "num_std", "buffer", "idx", "count", "sum_", "sum_sq")

    def __init__(self, period: int = 20, num_std: float = 2.0):
        self.period = period
        self.num_std = num_std
        self.buffer = np.zeros(period, dtype=np.float64)
        self.idx = 0
        self.count = 0
        self.sum_ = 0.0
        self.sum_sq = 0.0

    def update(self, price: float):
        if self.count >= self.period:
            old = self.buffer[self.idx]
            self.sum_ -= old
            self.sum_sq -= old * old

        self.buffer[self.idx] = price
        self.sum_ += price
        self.sum_sq += price * price
        self.idx = (self.idx + 1) % self.period
        self.count += 1

        if self.count < self.period:
            return None, None, None

        mean = self.sum_ / self.period
        variance = self.sum_sq / self.period - mean * mean
        std = np.sqrt(max(variance, 0.0))
        return mean - std * self.num_std, mean, mean + std * self.num_std


# ---------------------------------------------------------------------------
# Vectorized indicators (for pre-computation / optimizer)
# ---------------------------------------------------------------------------

def ema_array(prices: np.ndarray, period: int) -> np.ndarray:
    """Compute EMA over a full price array. Returns NaN for warmup bars."""
    alpha = 2.0 / (period + 1)
    out = np.empty_like(prices, dtype=np.float64)
    out[0] = prices[0]
    for i in range(1, len(prices)):
        out[i] = alpha * prices[i] + (1.0 - alpha) * out[i - 1]
    out[:period - 1] = np.nan
    return out


def atr_array(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Compute ATR over full OHLC arrays. Returns NaN for warmup bars."""
    n = len(high)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    alpha = 1.0 / period
    out = np.empty(n, dtype=np.float64)
    out[0] = tr[0]
    for i in range(1, n):
        out[i] = alpha * tr[i] + (1.0 - alpha) * out[i - 1]
    out[:period - 1] = np.nan
    return out


def rsi_array(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute RSI over a full price array. Returns NaN for warmup bars."""
    n = len(prices)
    out = np.full(n, np.nan, dtype=np.float64)
    deltas = np.diff(prices)
    alpha = 1.0 / period
    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(len(deltas)):
        gain = max(deltas[i], 0.0)
        loss = max(-deltas[i], 0.0)
        avg_gain = alpha * gain + (1.0 - alpha) * avg_gain
        avg_loss = alpha * loss + (1.0 - alpha) * avg_loss
        if i >= period - 1:
            if avg_loss == 0:
                out[i + 1] = 100.0
            else:
                out[i + 1] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
    return out
