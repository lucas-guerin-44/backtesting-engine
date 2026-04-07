"""Rate-of-change momentum strategy."""

from collections import deque
from typing import Optional

from backtesting.indicators import ATR
from backtesting.strategy import Strategy
from backtesting.types import Bar, Trade

from strategies.base import TrendFilter, risk_adjusted_size


class MomentumStrategy(Strategy):
    """Rate-of-change momentum strategy.

    Entry: N-bar return exceeds threshold -> enter in direction of move.
    Stop:  ATR-based.
    Exit:  ATR-based take profit.

    Based on the empirical observation that assets with strong recent
    performance tend to continue (Jegadeesh & Titman, 1993).
    """

    def __init__(
        self,
        lookback: int = 20,
        entry_threshold: float = 0.03,
        atr_period: int = 14,
        atr_stop_mult: float = 2.0,
        atr_target_mult: float = 3.0,
        risk_per_trade: float = 0.02,
        max_dd_halt: float = 0.15,
        cooldown_bars: int = 10,
        trend_filter_period: int = 0,
    ):
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult
        self.risk_per_trade = risk_per_trade
        self.max_dd_halt = max_dd_halt
        self.cooldown_bars = cooldown_bars

        self._atr = ATR(atr_period)
        self._closes: deque = deque(maxlen=lookback + 1)
        self._trend_filter = TrendFilter(trend_filter_period)
        self._peak_equity = 0.0
        self._bars_since_trade = 999

    def on_bar(self, i: int, bar: Bar, equity: float) -> Optional[Trade]:
        self._closes.append(bar.close)
        atr_val = self._atr.update(bar.high, bar.low, bar.close)
        self._trend_filter.update(bar.close)
        self._peak_equity = max(self._peak_equity, equity)
        self._bars_since_trade += 1

        if len(self._closes) <= self.lookback or atr_val is None or atr_val <= 0:
            return None
        if self._bars_since_trade < self.cooldown_bars:
            return None

        roc = (self._closes[-1] - self._closes[0]) / self._closes[0]

        if roc > self.entry_threshold:
            if not self._trend_filter.allows(1, bar.close):
                return None
            entry = bar.close
            stop = entry - atr_val * self.atr_stop_mult
            tp = entry + atr_val * self.atr_target_mult
            size = risk_adjusted_size(equity, entry, stop, self.risk_per_trade,
                                      self._peak_equity, self.max_dd_halt)
            if size > 0:
                self._bars_since_trade = 0
                return Trade(entry_bar=bar, side=1, size=size,
                             entry_price=entry, stop_price=stop, take_profit=tp)

        if roc < -self.entry_threshold:
            if not self._trend_filter.allows(-1, bar.close):
                return None
            entry = bar.close
            stop = entry + atr_val * self.atr_stop_mult
            tp = entry - atr_val * self.atr_target_mult
            size = risk_adjusted_size(equity, entry, stop, self.risk_per_trade,
                                      self._peak_equity, self.max_dd_halt)
            if size > 0:
                self._bars_since_trade = 0
                return Trade(entry_bar=bar, side=-1, size=size,
                             entry_price=entry, stop_price=stop, take_profit=tp)

        return None
