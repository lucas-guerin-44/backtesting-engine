"""Tick data types and bar aggregation.

Provides:
- ``Tick``: a single price update (timestamp, price, volume, optional bid/ask).
- ``TickAggregator``: accumulates ticks into OHLC bars by timeframe boundary.

The aggregator floors each tick's timestamp to the bar boundary (e.g., 09:31:14
on a 5-min timeframe belongs to the 09:30:00 bar). When a tick crosses into a
new bar boundary, the previous bar is emitted as complete.

Usage
-----
>>> agg = TickAggregator("5min")
>>> for tick in ticks:
...     bar = agg.update(tick)
...     if bar is not None:
...         print("Completed bar:", bar)
...     # agg.current_bar always has the in-progress partial bar
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from backtesting.types import Bar


@dataclass(slots=True)
class Tick:
    """A single price update."""
    ts: pd.Timestamp
    price: float
    volume: float = 0.0
    bid: Optional[float] = None
    ask: Optional[float] = None


# Map common shorthand timeframes to pandas offset aliases.
_FREQ_MAP = {
    "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
    "H1": "1h", "H4": "4h", "D1": "1D",
}

# Timeframe durations in nanoseconds for fast boundary computation.
_FREQ_NS = {
    "M1": 60_000_000_000,
    "M5": 300_000_000_000,
    "M15": 900_000_000_000,
    "M30": 1_800_000_000_000,
    "H1": 3_600_000_000_000,
    "H4": 14_400_000_000_000,
    "D1": 86_400_000_000_000,
    "1min": 60_000_000_000,
    "5min": 300_000_000_000,
    "15min": 900_000_000_000,
    "30min": 1_800_000_000_000,
    "1h": 3_600_000_000_000,
    "4h": 14_400_000_000_000,
    "1D": 86_400_000_000_000,
}


def _to_pd_freq(timeframe: str) -> str:
    """Convert a shorthand timeframe (e.g. 'M5') to a pandas freq string."""
    return _FREQ_MAP.get(timeframe, timeframe)


class TickAggregator:
    """Accumulates ticks into OHLC bars aligned to timeframe boundaries.

    Parameters
    ----------
    timeframe : str
        Bar timeframe, e.g. ``"M1"``, ``"M5"``, ``"H1"``, ``"1min"``, ``"5min"``.

    Performance notes
    -----------------
    - For known timeframes (M1-D1), bar boundaries are computed via integer
      division on nanosecond timestamps — no ``pd.Timestamp.floor()`` call.
      This is ~50x faster per tick.
    - The ``update()`` method caches the current bar's boundary end timestamp
      to skip even the integer division when consecutive ticks are in the same
      bar (the common case).
    - ``aggregate_batch()`` processes a list of ticks using vectorized numpy
      operations — ~10x faster than calling ``update()`` in a Python loop.
    """

    def __init__(self, timeframe: str):
        self._freq = _to_pd_freq(timeframe)
        self._bar_open_ts: Optional[pd.Timestamp] = None
        self._open: float = 0.0
        self._high: float = 0.0
        self._low: float = 0.0
        self._close: float = 0.0
        self._volume: float = 0.0
        self._tick_count: int = 0

        # Fast-path: integer nanosecond arithmetic for known timeframes
        self._freq_ns: Optional[int] = _FREQ_NS.get(timeframe) or _FREQ_NS.get(self._freq)

        # Cached boundary: [_bar_start_ns, _bar_end_ns) defines the current bar.
        # If the next tick's ns is within this range, skip floor computation entirely.
        self._bar_start_ns: int = 0
        self._bar_end_ns: int = 0

    @property
    def current_bar(self) -> Optional[Bar]:
        """The in-progress (incomplete) bar, or None if no ticks received yet."""
        if self._bar_open_ts is None:
            return None
        return Bar(
            ts=self._bar_open_ts,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=self._volume,
        )

    @property
    def tick_count(self) -> int:
        """Number of ticks in the current (incomplete) bar."""
        return self._tick_count

    def _floor_ts(self, ts: pd.Timestamp) -> pd.Timestamp:
        """Floor a timestamp to the bar boundary."""
        if self._freq_ns is not None:
            ns = ts.value
            floored_ns = (ns // self._freq_ns) * self._freq_ns
            return pd.Timestamp(floored_ns, unit="ns")
        return ts.floor(self._freq)

    def _floor_ns(self, ns: int) -> int:
        """Floor a nanosecond timestamp to bar boundary (returns nanoseconds)."""
        if self._freq_ns is not None:
            return (ns // self._freq_ns) * self._freq_ns
        # Fallback: use pandas (slow)
        return pd.Timestamp(ns, unit="ns").floor(self._freq).value

    def _emit_bar(self) -> Bar:
        """Package the current accumulator state as a completed Bar."""
        return Bar(
            ts=self._bar_open_ts,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=self._volume,
        )

    def update(self, tick: Tick) -> Optional[Bar]:
        """Feed a tick. Returns a completed Bar if this tick crosses a boundary.

        The returned bar is the *previous* completed bar. The tick that triggered
        the boundary crossing becomes the first tick of the new bar.

        Parameters
        ----------
        tick : Tick
            The incoming tick.

        Returns
        -------
        Bar or None
            The completed bar if a boundary was crossed, otherwise None.
        """
        tick_ns = tick.ts.value

        if self._bar_open_ts is None:
            # First tick ever
            bar_ns = self._floor_ns(tick_ns)
            self._start_bar_fast(bar_ns, tick.price, tick.volume)
            return None

        # Fast path: check if tick is within cached [start, end) boundary
        if self._bar_start_ns <= tick_ns < self._bar_end_ns:
            # Same bar — inline OHLC update (no method call)
            price = tick.price
            if price > self._high:
                self._high = price
            if price < self._low:
                self._low = price
            self._close = price
            self._volume += tick.volume
            self._tick_count += 1
            return None

        # Crossed boundary — emit old bar, start new one
        completed = self._emit_bar()
        bar_ns = self._floor_ns(tick_ns)
        self._start_bar_fast(bar_ns, tick.price, tick.volume)
        return completed

    def flush(self) -> Optional[Bar]:
        """Force-emit the current in-progress bar (e.g. at end of data).

        Returns None if no ticks have been received.
        """
        if self._bar_open_ts is None:
            return None
        bar = self._emit_bar()
        self._bar_open_ts = None
        self._tick_count = 0
        return bar

    def _start_bar(self, bar_ts: pd.Timestamp, tick: Tick) -> None:
        self._bar_open_ts = bar_ts
        self._open = tick.price
        self._high = tick.price
        self._low = tick.price
        self._close = tick.price
        self._volume = tick.volume
        self._tick_count = 1
        # Update cached boundary
        ns = bar_ts.value
        self._bar_start_ns = ns
        if self._freq_ns is not None:
            self._bar_end_ns = ns + self._freq_ns
        else:
            self._bar_end_ns = ns  # Will always miss cache -> falls through to floor

    def _start_bar_fast(self, bar_ns: int, price: float, volume: float) -> None:
        """Start a new bar from nanosecond timestamp and raw price/volume."""
        self._bar_open_ts = pd.Timestamp(bar_ns, unit="ns")
        self._open = price
        self._high = price
        self._low = price
        self._close = price
        self._volume = volume
        self._tick_count = 1
        self._bar_start_ns = bar_ns
        if self._freq_ns is not None:
            self._bar_end_ns = bar_ns + self._freq_ns
        else:
            self._bar_end_ns = bar_ns

    def _update_bar(self, tick: Tick) -> None:
        if tick.price > self._high:
            self._high = tick.price
        if tick.price < self._low:
            self._low = tick.price
        self._close = tick.price
        self._volume += tick.volume
        self._tick_count += 1

    def aggregate_batch(self, ticks: List[Tick]) -> List[Bar]:
        """Aggregate a list of ticks into completed bars (vectorized fast path).

        More efficient than calling ``update()`` in a loop. Processes all
        ticks at once using numpy operations for boundary detection, then
        builds bars from the segments.

        The aggregator's internal state is updated to reflect the last
        incomplete bar (accessible via ``current_bar``).

        Parameters
        ----------
        ticks : list of Tick
            Must be sorted by timestamp.

        Returns
        -------
        list of Bar
            Completed bars (does NOT include the final incomplete bar).
        """
        n = len(ticks)
        if n == 0:
            return []

        # Extract to numpy arrays
        prices = np.array([t.price for t in ticks], dtype=np.float64)
        volumes = np.array([t.volume for t in ticks], dtype=np.float64)
        ts_ns = np.array([t.ts.value for t in ticks], dtype=np.int64)

        # Compute bar boundaries via vectorized integer division
        if self._freq_ns is not None:
            freq_ns = self._freq_ns
            boundaries = (ts_ns // freq_ns) * freq_ns
        else:
            # Fallback: use pandas (slower)
            idx = pd.DatetimeIndex([t.ts for t in ticks])
            boundaries = idx.floor(self._freq).asi8

        # Find boundary change points
        changes = np.where(np.diff(boundaries) != 0)[0]  # indices where boundary changes

        completed_bars: List[Bar] = []

        # If we have a partial bar from before, handle it
        if self._bar_open_ts is not None:
            prev_boundary = self._bar_start_ns
            if boundaries[0] == prev_boundary:
                # First ticks extend the existing bar
                if len(changes) > 0:
                    end = changes[0] + 1  # Exclusive end of first segment
                else:
                    end = n
                # Update existing bar with first segment
                seg_prices = prices[:end]
                self._high = max(self._high, float(seg_prices.max()))
                self._low = min(self._low, float(seg_prices.min()))
                self._close = float(seg_prices[-1])
                self._volume += float(volumes[:end].sum())
                self._tick_count += end

                if len(changes) > 0:
                    # Emit the completed bar
                    completed_bars.append(self._emit_bar())
                    # Process remaining segments
                    changes_offset = changes
                else:
                    # All ticks in same bar as existing — done
                    return completed_bars
            else:
                # New boundary — emit existing bar first
                completed_bars.append(self._emit_bar())
                changes_offset = changes
        else:
            changes_offset = changes

        # Process complete segments between boundary changes
        # Segment starts: [0, changes[0]+1, changes[1]+1, ...]
        seg_starts = np.concatenate([[0], changes_offset + 1]) if self._bar_open_ts is None or boundaries[0] != self._bar_start_ns else changes_offset + 1
        if len(seg_starts) == 0:
            return completed_bars

        # Handle remaining segments
        for seg_idx in range(len(seg_starts)):
            start = int(seg_starts[seg_idx])
            if seg_idx + 1 < len(seg_starts):
                end = int(seg_starts[seg_idx + 1])
            else:
                end = n

            seg_prices = prices[start:end]
            seg_volumes = volumes[start:end]
            bar_ns = int(boundaries[start])

            if seg_idx < len(seg_starts) - 1:
                # Complete bar
                completed_bars.append(Bar(
                    ts=pd.Timestamp(bar_ns, unit="ns"),
                    open=float(seg_prices[0]),
                    high=float(seg_prices.max()),
                    low=float(seg_prices.min()),
                    close=float(seg_prices[-1]),
                    volume=float(seg_volumes.sum()),
                ))
            else:
                # Last segment is the new incomplete bar
                self._bar_open_ts = pd.Timestamp(bar_ns, unit="ns")
                self._open = float(seg_prices[0])
                self._high = float(seg_prices.max())
                self._low = float(seg_prices.min())
                self._close = float(seg_prices[-1])
                self._volume = float(seg_volumes.sum())
                self._tick_count = end - start
                self._bar_start_ns = bar_ns
                if self._freq_ns is not None:
                    self._bar_end_ns = bar_ns + self._freq_ns
                else:
                    self._bar_end_ns = bar_ns

        return completed_bars
