"""Donchian channel breakout strategy (Turtle Trading)."""

from collections import deque
from typing import Optional

from backtesting.indicators import ATR
from backtesting.strategy import Strategy
from backtesting.types import Bar, Trade

from strategies.base import TrendFilter, risk_adjusted_size


class DonchianBreakoutStrategy(Strategy):
    """Donchian channel breakout (Turtle Trading).

    Entry: price closes above N-bar high (long) or below N-bar low (short).
    Stop:  ATR-based.
    Exit:  take profit at R:R multiple.

    Made famous by Richard Dennis's Turtle Traders. The Donchian channel
    captures the highest high and lowest low over a lookback window.
    """

    def __init__(
        self,
        channel_period: int = 20,
        atr_period: int = 14,
        atr_stop_mult: float = 2.0,
        risk_reward: float = 2.0,
        risk_per_trade: float = 0.02,
        max_dd_halt: float = 0.15,
        cooldown_bars: int = 10,
        trend_filter_period: int = 0,
    ):
        self.channel_period = channel_period
        self.atr_stop_mult = atr_stop_mult
        self.risk_reward = risk_reward
        self.risk_per_trade = risk_per_trade
        self.max_dd_halt = max_dd_halt
        self.cooldown_bars = cooldown_bars

        self._atr = ATR(atr_period)
        self._highs: deque = deque(maxlen=channel_period + 1)
        self._lows: deque = deque(maxlen=channel_period + 1)
        self._trend_filter = TrendFilter(trend_filter_period)
        self._peak_equity = 0.0
        self._bars_since_trade = 999

    def on_bar(self, i: int, bar: Bar, equity: float) -> Optional[Trade]:
        self._highs.append(bar.high)
        self._lows.append(bar.low)
        atr_val = self._atr.update(bar.high, bar.low, bar.close)
        self._trend_filter.update(bar.close)
        self._peak_equity = max(self._peak_equity, equity)
        self._bars_since_trade += 1

        if len(self._highs) <= self.channel_period or atr_val is None or atr_val <= 0:
            return None
        if self._bars_since_trade < self.cooldown_bars:
            return None

        # Channel from lookback window (excluding current bar)
        highs_list = list(self._highs)
        lows_list = list(self._lows)
        ch_high = max(highs_list[:-1])
        ch_low = min(lows_list[:-1])

        if bar.close > ch_high:
            if not self._trend_filter.allows(1, bar.close):
                return None
            entry = bar.close
            stop = entry - atr_val * self.atr_stop_mult
            tp = entry + (entry - stop) * self.risk_reward
            size = risk_adjusted_size(equity, entry, stop, self.risk_per_trade,
                                      self._peak_equity, self.max_dd_halt)
            if size > 0:
                self._bars_since_trade = 0
                return Trade(entry_bar=bar, side=1, size=size,
                             entry_price=entry, stop_price=stop, take_profit=tp)

        if bar.close < ch_low:
            if not self._trend_filter.allows(-1, bar.close):
                return None
            entry = bar.close
            stop = entry + atr_val * self.atr_stop_mult
            tp = entry - (stop - entry) * self.risk_reward
            size = risk_adjusted_size(equity, entry, stop, self.risk_per_trade,
                                      self._peak_equity, self.max_dd_halt)
            if size > 0:
                self._bars_since_trade = 0
                return Trade(entry_bar=bar, side=-1, size=size,
                             entry_price=entry, stop_price=stop, take_profit=tp)

        return None
