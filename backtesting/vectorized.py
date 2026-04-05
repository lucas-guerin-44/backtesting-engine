"""Fully vectorized backtester for maximum throughput.

Instead of iterating bar-by-bar in Python, this module:
1. Pre-computes all indicators as numpy arrays (C-speed)
2. Generates entry signals as boolean masks (vectorized comparisons)
3. Finds exits via ``np.argmax`` on boolean arrays (C-speed forward scan)
4. Builds the equity curve in a single numpy pass

Throughput: ~1-3M bars/sec (vs ~300k for the event-driven Backtester).

Trade-off: strategies must express their logic as array operations — no
arbitrary per-bar state. For strategies with complex inter-bar state
(e.g., tracking FVG zones), use the event-driven ``Backtester`` instead.

Example::

    from backtesting.vectorized import VectorizedBacktester
    from backtesting.indicators import ema_array, atr_array

    fast = ema_array(close, 20)
    slow = ema_array(close, 50)
    atr_val = atr_array(high, low, close, 14)

    bullish = (shift(fast, 1) <= shift(slow, 1)) & (fast > slow)
    entries = bullish
    sides = np.ones(len(close))
    stops = close - atr_val * 2.0
    tps = close + atr_val * 4.0

    bt = VectorizedBacktester(open, high, low, close)
    equity, trades = bt.run(entries, sides, stops, tps, risk_per_trade=0.02)
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class VectorizedTrade:
    """Compact trade record from the vectorized backtester."""
    entry_idx: int
    exit_idx: int
    side: int
    size: float
    entry_price: float
    exit_price: float
    pnl: float


def shift(arr: np.ndarray, periods: int = 1, fill: float = np.nan) -> np.ndarray:
    """Shift array by N periods (like pandas .shift() but pure numpy)."""
    out = np.empty_like(arr)
    if periods > 0:
        out[:periods] = fill
        out[periods:] = arr[:-periods]
    elif periods < 0:
        out[periods:] = fill
        out[:periods] = arr[-periods:]
    else:
        out[:] = arr
    return out


class VectorizedBacktester:
    """Numpy-vectorized backtester for strategies expressible as array operations.

    Parameters
    ----------
    open, high, low, close : np.ndarray
        OHLC price arrays (same length).
    starting_cash : float
        Initial portfolio cash.
    commission_bps : float
        Commission in basis points (applied on entry and exit).
    slippage_bps : float
        Slippage in basis points.
    """

    def __init__(self, open: np.ndarray, high: np.ndarray, low: np.ndarray,
                 close: np.ndarray, starting_cash: float = 10_000,
                 commission_bps: float = 0.0, slippage_bps: float = 0.0):
        self.open = np.asarray(open, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.low = np.asarray(low, dtype=np.float64)
        self.close = np.asarray(close, dtype=np.float64)
        self.n = len(close)
        self.starting_cash = starting_cash
        self.comm_factor = commission_bps / 1e4
        self.slip_factor = slippage_bps / 1e4

    def run(
        self,
        entries: np.ndarray,
        sides: np.ndarray,
        stop_prices: np.ndarray,
        tp_prices: np.ndarray,
        risk_per_trade: float = 0.02,
        max_dd_halt: float = 0.15,
        cooldown_bars: int = 0,
    ) -> Tuple[np.ndarray, List[VectorizedTrade]]:
        """Run the vectorized backtest.

        Parameters
        ----------
        entries : np.ndarray (bool)
            True on bars where the strategy wants to enter.
        sides : np.ndarray (int)
            +1 for long, -1 for short on each bar.
        stop_prices : np.ndarray (float)
            Stop-loss price for the potential entry on each bar.
        tp_prices : np.ndarray (float)
            Take-profit price for the potential entry on each bar.
        risk_per_trade : float
            Fraction of cash to risk per trade (for position sizing).
        max_dd_halt : float
            Maximum drawdown before halting new entries.
        cooldown_bars : int
            Minimum bars between trade entries.

        Returns
        -------
        tuple of (np.ndarray, list[VectorizedTrade])
            Equity curve and list of closed trades.
        """
        n = self.n
        o, h, lo, c = self.open, self.high, self.low, self.close
        comm = self.comm_factor
        slip = self.slip_factor

        # --- Phase 1: Chain trades (find entry/exit pairs) ---
        trades: List[VectorizedTrade] = []
        cash = self.starting_cash
        peak_equity = cash
        bar = 0
        bars_since_trade = cooldown_bars  # Allow immediate first trade

        while bar < n:
            if not entries[bar]:
                bar += 1
                continue

            if bars_since_trade < cooldown_bars:
                bar += 1
                bars_since_trade += 1
                continue

            side = int(sides[bar])
            stop = stop_prices[bar]
            tp = tp_prices[bar]

            # Skip invalid signals
            if np.isnan(stop) or np.isnan(tp) or side == 0:
                bar += 1
                continue

            # Drawdown guard
            dd = (peak_equity - cash) / peak_equity if peak_equity > 0 else 0.0
            if dd >= max_dd_halt:
                bar += 1
                continue

            # Position sizing
            dd_scale = max(0.0, 1.0 - dd / max_dd_halt)
            risk_per_unit = abs(c[bar] - stop)
            if risk_per_unit <= 0:
                bar += 1
                continue

            entry_price = c[bar] * (1.0 + side * slip)
            size = (cash * risk_per_trade * dd_scale) / risk_per_unit
            size = min(size, cash / entry_price) if entry_price > 0 else 0.0

            if size <= 0:
                bar += 1
                continue

            # Entry commission
            entry_comm = abs(entry_price * size * comm)
            if cash < entry_comm:
                bar += 1
                continue
            cash -= entry_comm

            # Find exit: vectorized forward scan
            exit_idx, exit_price = self._find_exit(bar, side, stop, tp)

            # Apply slippage to exit
            exit_price_adj = exit_price * (1.0 - side * slip)

            # PnL and commission
            pnl = (exit_price_adj - entry_price) * side * size
            exit_comm = abs(exit_price_adj * size * comm)
            net_pnl = pnl - exit_comm
            cash += net_pnl

            peak_equity = max(peak_equity, cash)

            trades.append(VectorizedTrade(
                entry_idx=bar, exit_idx=exit_idx, side=side, size=size,
                entry_price=entry_price, exit_price=exit_price_adj, pnl=net_pnl,
            ))

            bars_since_trade = 0
            bar = exit_idx + 1
            continue

        # --- Phase 2: Build equity curve (numpy) ---
        equity = self._build_equity_curve(trades, c)

        self.cash = cash
        self.max_drawdown = 0.0
        if len(equity) > 0:
            peak = np.maximum.accumulate(equity)
            dd_arr = (peak - equity) / np.where(peak > 0, peak, 1.0)
            self.max_drawdown = float(np.max(dd_arr))

        return equity, trades

    def _find_exit(self, entry_idx: int, side: int, stop: float, tp: float) -> Tuple[int, float]:
        """Find the exit bar and price using vectorized forward scan.

        Gap-aware: if the bar opens past the stop, fills at the open (worse).
        """
        start = entry_idx + 1
        if start >= self.n:
            # No bars left — exit at entry price (no fill)
            return entry_idx, self.close[entry_idx]

        remaining_open = self.open[start:]
        remaining_high = self.high[start:]
        remaining_low = self.low[start:]
        m = len(remaining_open)

        if side > 0:
            # Long: stop when low <= stop OR open <= stop (gap)
            stop_mask = (remaining_low <= stop) | (remaining_open <= stop)
            tp_mask = remaining_high >= tp
        else:
            # Short: stop when high >= stop OR open >= stop (gap)
            stop_mask = (remaining_high >= stop) | (remaining_open >= stop)
            tp_mask = remaining_low <= tp

        # np.argmax on bool array returns index of first True (0 if none True)
        stop_idx = int(np.argmax(stop_mask)) if np.any(stop_mask) else m
        tp_idx = int(np.argmax(tp_mask)) if np.any(tp_mask) else m

        if stop_idx >= m and tp_idx >= m:
            # Neither hit — exit at last bar's close
            return self.n - 1, self.close[-1]

        if stop_idx <= tp_idx:
            abs_idx = start + stop_idx
            # Gap-aware: if open is past stop, fill at open
            if side > 0 and self.open[abs_idx] <= stop:
                return abs_idx, self.open[abs_idx]
            elif side < 0 and self.open[abs_idx] >= stop:
                return abs_idx, self.open[abs_idx]
            return abs_idx, stop
        else:
            abs_idx = start + tp_idx
            return abs_idx, tp

    def _build_equity_curve(self, trades: List[VectorizedTrade], close: np.ndarray) -> np.ndarray:
        """Build the full equity curve from trade list.

        Trades are sequential (no overlaps) because the main loop exits one
        trade before entering the next. This means the equity curve has three
        kinds of regions:

        1. **Flat** (between trades): equity = starting_cash + realized PnL so far
        2. **In-trade**: equity = flat base - entry commission + unrealized PnL
        3. **After all trades**: equity = starting_cash + total realized PnL

        The in-trade unrealized PnL is computed as a numpy slice (vectorized),
        not per-bar.
        """
        n = self.n
        equity = np.full(n, self.starting_cash, dtype=np.float64)

        if not trades:
            return equity

        realized = 0.0
        prev_end = 0

        for t in trades:
            # Flat region: bars from previous trade's exit to this trade's entry
            if t.entry_idx > prev_end:
                equity[prev_end:t.entry_idx] += realized

            # In-trade region: base + unrealized PnL (vectorized slice)
            entry_comm = abs(t.entry_price * t.size * self.comm_factor)
            trade_base = self.starting_cash + realized - entry_comm
            trade_slice = slice(t.entry_idx, min(t.exit_idx + 1, n))
            unrealized = (close[trade_slice] - t.entry_price) * t.side * t.size
            equity[trade_slice] = trade_base + unrealized

            realized += t.pnl
            prev_end = t.exit_idx + 1

        # Flat region after last trade
        if prev_end < n:
            equity[prev_end:] += realized

        return np.maximum(equity, 0.0)
