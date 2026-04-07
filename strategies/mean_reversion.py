"""Bollinger Band mean reversion strategy with RSI confirmation."""

from typing import Optional

from backtesting.indicators import ATR, RSI, BollingerBands
from backtesting.strategy import Strategy
from backtesting.types import Bar, Trade

from strategies.base import TrendFilter, risk_adjusted_size


class MeanReversionStrategy(Strategy):
    """Bollinger Band mean reversion with RSI confirmation.

    Entry: price touches outer band AND RSI confirms extreme.
    Stop:  ATR multiple beyond entry.
    Exit:  target the middle band (SMA).

    The opposite bet from trend-following. Works in ranging markets
    and during elevated but non-directional volatility.
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        atr_period: int = 14,
        atr_stop_mult: float = 2.5,
        risk_per_trade: float = 0.02,
        max_dd_halt: float = 0.15,
        cooldown_bars: int = 5,
        trend_filter_period: int = 0,
    ):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.atr_stop_mult = atr_stop_mult
        self.risk_per_trade = risk_per_trade
        self.max_dd_halt = max_dd_halt
        self.cooldown_bars = cooldown_bars

        self._bb = BollingerBands(bb_period, bb_std)
        self._rsi = RSI(rsi_period)
        self._atr = ATR(atr_period)
        self._trend_filter = TrendFilter(trend_filter_period)
        self._peak_equity = 0.0
        self._bars_since_trade = 999

    def on_bar(self, i: int, bar: Bar, equity: float) -> Optional[Trade]:
        bb_lower, bb_mid, bb_upper = self._bb.update(bar.close)
        rsi_val = self._rsi.update(bar.close)
        atr_val = self._atr.update(bar.high, bar.low, bar.close)
        self._trend_filter.update(bar.close)
        self._peak_equity = max(self._peak_equity, equity)
        self._bars_since_trade += 1

        if bb_lower is None or rsi_val is None or atr_val is None or atr_val <= 0:
            return None
        if self._bars_since_trade < self.cooldown_bars:
            return None

        # Long: lower band + oversold
        if bar.low <= bb_lower and rsi_val <= self.rsi_oversold:
            if not self._trend_filter.allows(1, bar.close):
                return None
            entry = bar.close
            stop = entry - atr_val * self.atr_stop_mult
            size = risk_adjusted_size(equity, entry, stop, self.risk_per_trade,
                                      self._peak_equity, self.max_dd_halt)
            if size > 0:
                self._bars_since_trade = 0
                return Trade(entry_bar=bar, side=1, size=size,
                             entry_price=entry, stop_price=stop, take_profit=bb_mid)

        # Short: upper band + overbought
        if bar.high >= bb_upper and rsi_val >= self.rsi_overbought:
            if not self._trend_filter.allows(-1, bar.close):
                return None
            entry = bar.close
            stop = entry + atr_val * self.atr_stop_mult
            size = risk_adjusted_size(equity, entry, stop, self.risk_per_trade,
                                      self._peak_equity, self.max_dd_halt)
            if size > 0:
                self._bars_since_trade = 0
                return Trade(entry_bar=bar, side=-1, size=size,
                             entry_price=entry, stop_price=stop, take_profit=bb_mid)

        return None
