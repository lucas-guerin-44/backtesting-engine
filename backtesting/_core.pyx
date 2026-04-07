# cython: boundscheck=False, wraparound=False, cdivision=True
"""Cython-accelerated backtesting hot paths.

Replaces the Python while-loop in VectorizedBacktester.run() with C-speed
typed code. The trade-chaining loop and exit-finding scan are the two
bottlenecks — everything else (signal generation, equity curve building)
is already numpy-vectorized and fast.

Typical speedup: 10-30x over pure Python on the same logic.
"""

import numpy as np
cimport numpy as np
from libc.math cimport fabs, isnan, INFINITY


# Trade result struct returned to Python
cdef struct TradeResult:
    int entry_idx
    int exit_idx
    int side
    double size
    double entry_price
    double exit_price
    double pnl


cdef (int, double) _find_exit(
    int entry_idx, int side, double stop, double tp,
    double[:] open_arr, double[:] high_arr, double[:] low_arr, double[:] close_arr,
    int n,
) noexcept nogil:
    """Find exit bar and price via forward scan. Gap-aware."""
    cdef int start = entry_idx + 1
    cdef int i
    cdef double o, h, lo

    if start >= n:
        return entry_idx, close_arr[entry_idx]

    for i in range(start, n):
        o = open_arr[i]
        h = high_arr[i]
        lo = low_arr[i]

        if side > 0:
            # Long: check stop first
            if o <= stop:
                return i, o  # gap-aware fill at open
            if lo <= stop:
                return i, stop
            if h >= tp:
                return i, tp
        else:
            # Short: check stop first
            if o >= stop:
                return i, o  # gap-aware fill at open
            if h >= stop:
                return i, stop
            if lo <= tp:
                return i, tp

    # Neither hit — exit at last bar's close
    return n - 1, close_arr[n - 1]


def chain_trades(
    np.ndarray[np.uint8_t, ndim=1] entries,
    np.ndarray[np.float64_t, ndim=1] sides,
    np.ndarray[np.float64_t, ndim=1] stop_prices,
    np.ndarray[np.float64_t, ndim=1] tp_prices,
    np.ndarray[np.float64_t, ndim=1] open_arr,
    np.ndarray[np.float64_t, ndim=1] high_arr,
    np.ndarray[np.float64_t, ndim=1] low_arr,
    np.ndarray[np.float64_t, ndim=1] close_arr,
    double starting_cash,
    double comm_factor,
    double slip_factor,
    double risk_per_trade,
    double max_dd_halt,
    int cooldown_bars,
):
    """Chain trades through the price series. Returns list of trade tuples and final cash.

    This is the C-accelerated version of VectorizedBacktester.run()'s Phase 1.
    """
    cdef int n = len(close_arr)
    cdef int bar = 0
    cdef int bars_since_trade = cooldown_bars
    cdef int side_val, exit_idx
    cdef double cash = starting_cash
    cdef double peak_equity = starting_cash
    cdef double stop, tp, dd, dd_scale, risk_per_unit
    cdef double entry_price, size, entry_comm, exit_price, exit_price_adj
    cdef double pnl, exit_comm, net_pnl

    # Typed memoryviews for C-speed array access
    cdef double[:] o_v = open_arr
    cdef double[:] h_v = high_arr
    cdef double[:] lo_v = low_arr
    cdef double[:] c_v = close_arr
    cdef double[:] stops_v = stop_prices
    cdef double[:] tps_v = tp_prices
    cdef double[:] sides_v = sides
    cdef unsigned char[:] entries_v = entries

    trades = []

    while bar < n:
        if not entries_v[bar]:
            bar += 1
            bars_since_trade += 1
            continue

        if bars_since_trade < cooldown_bars:
            bar += 1
            bars_since_trade += 1
            continue

        side_val = <int>sides_v[bar]
        stop = stops_v[bar]
        tp = tps_v[bar]

        if isnan(stop) or isnan(tp) or side_val == 0:
            bar += 1
            continue

        # Drawdown guard
        dd = (peak_equity - cash) / peak_equity if peak_equity > 0 else 0.0
        if dd >= max_dd_halt:
            bar += 1
            continue

        # Position sizing
        dd_scale = 1.0 - dd / max_dd_halt
        if dd_scale < 0.0:
            dd_scale = 0.0
        risk_per_unit = fabs(c_v[bar] - stop)
        if risk_per_unit <= 0:
            bar += 1
            continue

        entry_price = c_v[bar] * (1.0 + side_val * slip_factor)
        size = (cash * risk_per_trade * dd_scale) / risk_per_unit
        if entry_price > 0 and size > cash / entry_price:
            size = cash / entry_price

        if size <= 0:
            bar += 1
            continue

        # Entry commission
        entry_comm = fabs(entry_price * size * comm_factor)
        if cash < entry_comm:
            bar += 1
            continue
        cash -= entry_comm

        # Find exit
        exit_idx, exit_price = _find_exit(bar, side_val, stop, tp, o_v, h_v, lo_v, c_v, n)

        # Slippage on exit
        exit_price_adj = exit_price * (1.0 - side_val * slip_factor)

        # PnL
        pnl = (exit_price_adj - entry_price) * side_val * size
        exit_comm = fabs(exit_price_adj * size * comm_factor)
        net_pnl = pnl - exit_comm
        cash += net_pnl

        if cash > peak_equity:
            peak_equity = cash

        trades.append((bar, exit_idx, side_val, size, entry_price, exit_price_adj, net_pnl))

        bars_since_trade = 0
        bar = exit_idx + 1

    return trades, cash


def cy_ema_array(np.ndarray[np.float64_t, ndim=1] prices, int period):
    """Cython-accelerated EMA computation."""
    cdef int n = len(prices)
    cdef np.ndarray[np.float64_t, ndim=1] out = np.empty(n, dtype=np.float64)
    cdef double alpha = 2.0 / (period + 1)
    cdef double one_minus_alpha = 1.0 - alpha
    cdef int i

    out[0] = prices[0]
    for i in range(1, n):
        out[i] = alpha * prices[i] + one_minus_alpha * out[i - 1]

    cdef int nan_limit = min(period - 1, n)
    for i in range(nan_limit):
        out[i] = np.nan

    return out


def cy_atr_array(
    np.ndarray[np.float64_t, ndim=1] high,
    np.ndarray[np.float64_t, ndim=1] low,
    np.ndarray[np.float64_t, ndim=1] close,
    int period,
):
    """Cython-accelerated ATR computation."""
    cdef int n = len(high)
    cdef np.ndarray[np.float64_t, ndim=1] out = np.empty(n, dtype=np.float64)
    cdef double alpha = 1.0 / period
    cdef double one_minus_alpha = 1.0 - alpha
    cdef double tr, hl, hc, lc
    cdef int i

    out[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = fabs(high[i] - close[i - 1])
        lc = fabs(low[i] - close[i - 1])
        tr = hl
        if hc > tr:
            tr = hc
        if lc > tr:
            tr = lc
        out[i] = alpha * tr + one_minus_alpha * out[i - 1]

    cdef int nan_limit = min(period - 1, n)
    for i in range(nan_limit):
        out[i] = np.nan

    return out


def cy_rsi_array(np.ndarray[np.float64_t, ndim=1] prices, int period = 14):
    """Cython-accelerated RSI computation."""
    cdef int n = len(prices)
    cdef np.ndarray[np.float64_t, ndim=1] out = np.full(n, np.nan, dtype=np.float64)
    cdef double alpha = 1.0 / period
    cdef double one_minus_alpha = 1.0 - alpha
    cdef double avg_gain = 0.0
    cdef double avg_loss = 0.0
    cdef double delta, gain, loss, rs
    cdef int i

    for i in range(1, n):
        delta = prices[i] - prices[i - 1]
        gain = delta if delta > 0 else 0.0
        loss = -delta if delta < 0 else 0.0
        avg_gain = alpha * gain + one_minus_alpha * avg_gain
        avg_loss = alpha * loss + one_minus_alpha * avg_loss
        if i >= period:
            if avg_loss == 0:
                out[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                out[i] = 100.0 - (100.0 / (1.0 + rs))

    return out
