# cython: boundscheck=False, wraparound=False, cdivision=True
"""Cython-accelerated tick backtesting hot paths.

Provides C-speed stop/TP scanning and bar aggregation over pre-extracted
numpy arrays. The strategy callbacks (on_bar, on_tick) remain in Python —
only the per-tick inner loop math (price comparisons, equity tracking,
bar boundary detection) is accelerated.

Typical speedup: 5-15x over the pure Python tick loop.
"""

import numpy as np
cimport numpy as np
from libc.math cimport fabs


def cy_tick_scan_stops(
    double[:] prices,
    double[:] stop_prices,
    double[:] tp_prices,
    long[:] sides,
    int n_ticks,
    int n_positions,
):
    """Scan tick prices against stop/TP levels for all open positions.

    For each tick, checks all open positions and returns the first tick index
    where each position's stop or TP is triggered.

    Parameters
    ----------
    prices : array of float64
        Tick prices (length n_ticks).
    stop_prices : array of float64
        Stop prices for each open position (length n_positions).
    tp_prices : array of float64
        Take-profit prices for each position (length n_positions). NaN = no TP.
    sides : array of int64
        Position sides (+1 long, -1 short) (length n_positions).
    n_ticks : int
        Number of ticks.
    n_positions : int
        Number of open positions.

    Returns
    -------
    exit_tick : ndarray of int32
        For each position, the tick index where stop/TP was hit. -1 if not hit.
    exit_type : ndarray of int32
        0 = not hit, 1 = stop, 2 = take-profit.
    """
    cdef np.ndarray[np.int32_t, ndim=1] exit_tick = np.full(n_positions, -1, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] exit_type = np.zeros(n_positions, dtype=np.int32)

    cdef int t, p
    cdef double px, stop, tp
    cdef long side
    cdef int[:] exit_tick_v = exit_tick
    cdef int[:] exit_type_v = exit_type

    for p in range(n_positions):
        stop = stop_prices[p]
        tp = tp_prices[p]
        side = sides[p]

        for t in range(n_ticks):
            px = prices[t]

            if side > 0:
                if px <= stop:
                    exit_tick_v[p] = t
                    exit_type_v[p] = 1
                    break
                if tp == tp and px >= tp:  # tp == tp is NaN check
                    exit_tick_v[p] = t
                    exit_type_v[p] = 2
                    break
            else:
                if px >= stop:
                    exit_tick_v[p] = t
                    exit_type_v[p] = 1
                    break
                if tp == tp and px <= tp:
                    exit_tick_v[p] = t
                    exit_type_v[p] = 2
                    break

    return exit_tick, exit_type


def cy_aggregate_ticks(
    double[:] prices,
    double[:] volumes,
    long long[:] bar_boundaries,
    int n_ticks,
):
    """Aggregate ticks into OHLCV bars using pre-computed bar boundaries.

    Parameters
    ----------
    prices : array of float64
        Tick prices.
    volumes : array of float64
        Tick volumes.
    bar_boundaries : array of int64
        Floored timestamp (as nanoseconds) for each tick's bar.
    n_ticks : int
        Number of ticks.

    Returns
    -------
    bar_open : ndarray of float64
    bar_high : ndarray of float64
    bar_low : ndarray of float64
    bar_close : ndarray of float64
    bar_volume : ndarray of float64
    bar_boundary_ns : ndarray of int64
        The bar boundary timestamp for each completed bar.
    bar_end_tick : ndarray of int32
        The tick index of the last tick in each completed bar.
    n_bars : int
        Number of completed bars.
    """
    # Worst case: every tick is its own bar
    cdef np.ndarray[np.float64_t, ndim=1] bar_open = np.empty(n_ticks, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] bar_high = np.empty(n_ticks, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] bar_low = np.empty(n_ticks, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] bar_close = np.empty(n_ticks, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] bar_volume = np.empty(n_ticks, dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim=1] bar_boundary_ns = np.empty(n_ticks, dtype=np.int64)
    cdef np.ndarray[np.int32_t, ndim=1] bar_end_tick = np.empty(n_ticks, dtype=np.int32)

    cdef double[:] bo = bar_open
    cdef double[:] bh = bar_high
    cdef double[:] bl = bar_low
    cdef double[:] bc = bar_close
    cdef double[:] bv = bar_volume
    cdef long long[:] bbn = bar_boundary_ns
    cdef int[:] bet = bar_end_tick

    cdef int n_bars = 0
    cdef int t
    cdef double px, vol
    cdef long long boundary, current_boundary

    if n_ticks == 0:
        return bar_open[:0], bar_high[:0], bar_low[:0], bar_close[:0], bar_volume[:0], bar_boundary_ns[:0], bar_end_tick[:0], 0

    # Start first bar
    current_boundary = bar_boundaries[0]
    cdef double cur_open = prices[0]
    cdef double cur_high = prices[0]
    cdef double cur_low = prices[0]
    cdef double cur_close = prices[0]
    cdef double cur_vol = volumes[0]

    for t in range(1, n_ticks):
        px = prices[t]
        vol = volumes[t]
        boundary = bar_boundaries[t]

        if boundary != current_boundary:
            # Emit completed bar
            bo[n_bars] = cur_open
            bh[n_bars] = cur_high
            bl[n_bars] = cur_low
            bc[n_bars] = cur_close
            bv[n_bars] = cur_vol
            bbn[n_bars] = current_boundary
            bet[n_bars] = t - 1
            n_bars += 1

            # Start new bar
            current_boundary = boundary
            cur_open = px
            cur_high = px
            cur_low = px
            cur_close = px
            cur_vol = vol
        else:
            if px > cur_high:
                cur_high = px
            if px < cur_low:
                cur_low = px
            cur_close = px
            cur_vol += vol

    # Emit final bar (partial)
    bo[n_bars] = cur_open
    bh[n_bars] = cur_high
    bl[n_bars] = cur_low
    bc[n_bars] = cur_close
    bv[n_bars] = cur_vol
    bbn[n_bars] = current_boundary
    bet[n_bars] = n_ticks - 1
    n_bars += 1

    return bar_open[:n_bars], bar_high[:n_bars], bar_low[:n_bars], bar_close[:n_bars], bar_volume[:n_bars], bar_boundary_ns[:n_bars], bar_end_tick[:n_bars], n_bars


def cy_compute_equity_curve(
    double[:] prices,
    double starting_cash,
    int n_ticks,
):
    """Compute equity curve over ticks when no positions are open.

    This is the fast path for the common case (most ticks have no open positions).
    Returns a flat equity array at starting_cash.

    For ticks WITH positions, the Python loop handles equity computation.
    """
    cdef np.ndarray[np.float64_t, ndim=1] equity = np.full(n_ticks, starting_cash, dtype=np.float64)
    return equity


def cy_find_stop_tp_tick(
    double[:] prices,
    int start_idx,
    int end_idx,
    double stop_price,
    double tp_price,
    int side,
):
    """Find the first tick where stop or TP is hit in a range.

    Parameters
    ----------
    prices : array of float64
    start_idx : int
        Start scanning from this tick index.
    end_idx : int
        Stop scanning at this tick index (exclusive).
    stop_price : float
    tp_price : float
        NaN means no take-profit.
    side : int
        +1 long, -1 short.

    Returns
    -------
    hit_idx : int
        Tick index where stop/TP was hit. -1 if not hit in range.
    hit_type : int
        0 = not hit, 1 = stop, 2 = take-profit.
    """
    cdef int t
    cdef double px
    cdef int has_tp = (tp_price == tp_price)  # False if NaN

    for t in range(start_idx, end_idx):
        px = prices[t]

        if side > 0:
            if px <= stop_price:
                return t, 1
            if has_tp and px >= tp_price:
                return t, 2
        else:
            if px >= stop_price:
                return t, 1
            if has_tp and px <= tp_price:
                return t, 2

    return -1, 0
