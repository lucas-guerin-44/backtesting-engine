"""Vectorized signal generators for each strategy.

These functions compute entry signals, stop prices, and take-profit prices
as numpy arrays — no Python loop, no per-bar objects. Designed for use
with ``VectorizedBacktester``.

Each function returns ``(entries, sides, stops, tps)`` arrays.
"""

import numpy as np
import pandas as pd

from backtesting.indicators import atr_array, ema_array, rsi_array
from backtesting.vectorized import shift


def trend_following_signals(
    open: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
    fast_period: int = 20, slow_period: int = 50,
    atr_period: int = 14, atr_stop_mult: float = 2.0, atr_target_mult: float = 4.0,
):
    """Dual-EMA crossover trend follower.

    Long when fast EMA crosses above slow and price > slow.
    Short when fast EMA crosses below slow and price < slow.
    """
    fast = ema_array(close, fast_period)
    slow = ema_array(close, slow_period)
    atr_val = atr_array(high, low, close, atr_period)

    prev_fast = shift(fast, 1)
    prev_slow = shift(slow, 1)

    bullish = (prev_fast <= prev_slow) & (fast > slow) & (close > slow)
    bearish = (prev_fast >= prev_slow) & (fast < slow) & (close < slow)

    entries = bullish | bearish
    sides = np.where(bullish, 1, np.where(bearish, -1, 0))
    stops = np.where(bullish, close - atr_val * atr_stop_mult,
                     np.where(bearish, close + atr_val * atr_stop_mult, np.nan))
    tps = np.where(bullish, close + atr_val * atr_target_mult,
                   np.where(bearish, close - atr_val * atr_target_mult, np.nan))

    return entries, sides, stops, tps


def mean_reversion_signals(
    open: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
    bb_period: int = 20, bb_std: float = 2.0,
    rsi_period: int = 14, rsi_oversold: float = 30.0, rsi_overbought: float = 70.0,
    atr_period: int = 14, atr_stop_mult: float = 2.5,
):
    """Bollinger Band + RSI mean reversion.

    Long at lower band + RSI oversold, short at upper band + RSI overbought.
    Target: middle band (SMA).
    """
    atr_val = atr_array(high, low, close, atr_period)
    rsi_val = rsi_array(close, rsi_period)

    # Bollinger Bands via pandas rolling (C-speed)
    s = pd.Series(close)
    sma = s.rolling(bb_period).mean().to_numpy()
    std = s.rolling(bb_period).std(ddof=0).to_numpy()
    bb_upper = sma + bb_std * std
    bb_lower = sma - bb_std * std

    bullish = (low <= bb_lower) & (rsi_val <= rsi_oversold)
    bearish = (high >= bb_upper) & (rsi_val >= rsi_overbought)

    entries = bullish | bearish
    sides = np.where(bullish, 1, np.where(bearish, -1, 0))
    stops = np.where(bullish, close - atr_val * atr_stop_mult,
                     np.where(bearish, close + atr_val * atr_stop_mult, np.nan))
    tps = np.where(bullish, sma, np.where(bearish, sma, np.nan))

    return entries, sides, stops, tps


def momentum_signals(
    open: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
    lookback: int = 20, entry_threshold: float = 0.03,
    atr_period: int = 14, atr_stop_mult: float = 2.0, atr_target_mult: float = 3.0,
):
    """Rate-of-change momentum.

    Long when N-bar return > threshold, short when < -threshold.
    """
    atr_val = atr_array(high, low, close, atr_period)
    roc = (close - shift(close, lookback)) / shift(close, lookback)

    bullish = roc > entry_threshold
    bearish = roc < -entry_threshold

    entries = bullish | bearish
    sides = np.where(bullish, 1, np.where(bearish, -1, 0))
    stops = np.where(bullish, close - atr_val * atr_stop_mult,
                     np.where(bearish, close + atr_val * atr_stop_mult, np.nan))
    tps = np.where(bullish, close + atr_val * atr_target_mult,
                   np.where(bearish, close - atr_val * atr_target_mult, np.nan))

    return entries, sides, stops, tps


def donchian_signals(
    open: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
    channel_period: int = 20, atr_period: int = 14,
    atr_stop_mult: float = 2.0, risk_reward: float = 2.0,
):
    """Donchian channel breakout (Turtle-style).

    Long when close > N-bar high, short when close < N-bar low.
    """
    atr_val = atr_array(high, low, close, atr_period)

    # Rolling channel high/low via pandas (C-speed), shifted by 1 to exclude current bar
    ch_high = shift(pd.Series(high).rolling(channel_period).max().to_numpy(), 1)
    ch_low = shift(pd.Series(low).rolling(channel_period).min().to_numpy(), 1)

    bullish = close > ch_high
    bearish = close < ch_low

    entries = bullish | bearish
    sides = np.where(bullish, 1, np.where(bearish, -1, 0))

    bull_stop = close - atr_val * atr_stop_mult
    bear_stop = close + atr_val * atr_stop_mult
    bull_tp = close + (close - bull_stop) * risk_reward
    bear_tp = close - (bear_stop - close) * risk_reward

    stops = np.where(bullish, bull_stop, np.where(bearish, bear_stop, np.nan))
    tps = np.where(bullish, bull_tp, np.where(bearish, bear_tp, np.nan))

    return entries, sides, stops, tps
