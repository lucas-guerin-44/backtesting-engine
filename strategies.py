"""
Trading strategies for the backtesting engine.

Four strategies, each representing a distinct market thesis:
- TrendFollowing:   EMA crossover with ATR trailing stops
- MeanReversion:    Bollinger Band + RSI at band extremes
- Momentum:         N-bar rate-of-change breakout
- DonchianBreakout: Channel breakout (Turtle-style)

All strategies share a common risk model:
- Position size scaled to risk a fixed fraction of equity per trade (ATR-based)
- Equity curve circuit breaker: halts new entries when drawdown exceeds a threshold
- Cooldown between trades to prevent overtrading

Indicator computations use O(1) incremental classes from ``backtesting.indicators``
— no pandas Series or numpy array allocations per bar.
"""

from typing import List, Optional

from backtesting.indicators import ATR, RSI, EMA, BollingerBands
from backtesting.strategy import Strategy
from backtesting.types import Bar, Trade


# ---------------------------------------------------------------------------
# Drawdown-aware position sizing
# ---------------------------------------------------------------------------

def risk_adjusted_size(
    equity: float,
    entry_price: float,
    stop_price: float,
    risk_per_trade: float,
    peak_equity: float,
    max_dd_halt: float,
) -> float:
    """Compute position size risking a fixed fraction of equity, with a drawdown kill-switch.

    Parameters
    ----------
    equity : float
        Current portfolio equity (cash + open P&L).
    entry_price : float
        Intended entry price.
    stop_price : float
        Stop-loss price.
    risk_per_trade : float
        Fraction of equity to risk on this trade (e.g. 0.02 = 2%).
    peak_equity : float
        Highest equity value seen so far.
    max_dd_halt : float
        Maximum drawdown fraction before halting (e.g. 0.15 = 15%).

    Returns
    -------
    float
        Position size in units, or 0.0 if the trade should be skipped.
        The Broker's buying power check provides the final guard against
        oversizing beyond available capital.
    """
    if equity <= 0 or entry_price <= 0:
        return 0.0

    current_dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
    if current_dd >= max_dd_halt:
        return 0.0

    # Linear scale-down: full size at 0% DD, zero at max_dd_halt
    dd_scale = max(0.0, 1.0 - current_dd / max_dd_halt)

    risk_per_unit = abs(entry_price - stop_price)
    if risk_per_unit <= 0:
        return 0.0

    risk_capital = equity * risk_per_trade * dd_scale
    size = risk_capital / risk_per_unit
    return min(size, equity / entry_price)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

class TrendFollowingStrategy(Strategy):
    """Classic dual-EMA trend follower with ATR-based stops.

    Entry: fast EMA crosses slow EMA in direction of trend.
    Stop:  ATR multiple below/above entry.
    Exit:  take-profit at wider ATR multiple, or stop-loss.

    The managed-futures workhorse. Works best on H1+ timeframes
    in trending markets. Will bleed in sideways chop — the drawdown
    guard limits the damage.
    """

    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 50,
        atr_period: int = 14,
        atr_stop_mult: float = 2.0,
        atr_target_mult: float = 4.0,
        risk_per_trade: float = 0.02,
        max_dd_halt: float = 0.15,
        cooldown_bars: int = 5,
    ):
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult
        self.risk_per_trade = risk_per_trade
        self.max_dd_halt = max_dd_halt
        self.cooldown_bars = cooldown_bars

        self._fast_ema = EMA(fast_period)
        self._slow_ema = EMA(slow_period)
        self._atr = ATR(atr_period)
        self._prev_fast: Optional[float] = None
        self._prev_slow: Optional[float] = None
        self._peak_equity = 0.0
        self._bars_since_trade = 999

    def on_bar(self, i: int, bar: Bar, cash: float) -> Optional[Trade]:
        fast_val = self._fast_ema.update(bar.close)
        slow_val = self._slow_ema.update(bar.close)
        atr_val = self._atr.update(bar.high, bar.low, bar.close)
        self._peak_equity = max(self._peak_equity, cash)
        self._bars_since_trade += 1

        if fast_val is None or slow_val is None or atr_val is None or atr_val <= 0:
            self._prev_fast, self._prev_slow = fast_val, slow_val
            return None
        if self._prev_fast is None or self._prev_slow is None:
            self._prev_fast, self._prev_slow = fast_val, slow_val
            return None
        if self._bars_since_trade < self.cooldown_bars:
            self._prev_fast, self._prev_slow = fast_val, slow_val
            return None

        trade = None

        # Bullish crossover
        if self._prev_fast <= self._prev_slow and fast_val > slow_val and bar.close > slow_val:
            entry = bar.close
            stop = entry - atr_val * self.atr_stop_mult
            tp = entry + atr_val * self.atr_target_mult
            size = risk_adjusted_size(cash, entry, stop, self.risk_per_trade,
                                      self._peak_equity, self.max_dd_halt)
            if size > 0:
                self._bars_since_trade = 0
                trade = Trade(entry_bar=bar, side=1, size=size,
                              entry_price=entry, stop_price=stop, take_profit=tp)

        # Bearish crossover
        elif self._prev_fast >= self._prev_slow and fast_val < slow_val and bar.close < slow_val:
            entry = bar.close
            stop = entry + atr_val * self.atr_stop_mult
            tp = entry - atr_val * self.atr_target_mult
            size = risk_adjusted_size(cash, entry, stop, self.risk_per_trade,
                                      self._peak_equity, self.max_dd_halt)
            if size > 0:
                self._bars_since_trade = 0
                trade = Trade(entry_bar=bar, side=-1, size=size,
                              entry_price=entry, stop_price=stop, take_profit=tp)

        self._prev_fast, self._prev_slow = fast_val, slow_val
        return trade


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
        self._peak_equity = 0.0
        self._bars_since_trade = 999

    def on_bar(self, i: int, bar: Bar, cash: float) -> Optional[Trade]:
        bb_lower, bb_mid, bb_upper = self._bb.update(bar.close)
        rsi_val = self._rsi.update(bar.close)
        atr_val = self._atr.update(bar.high, bar.low, bar.close)
        self._peak_equity = max(self._peak_equity, cash)
        self._bars_since_trade += 1

        if bb_lower is None or rsi_val is None or atr_val is None or atr_val <= 0:
            return None
        if self._bars_since_trade < self.cooldown_bars:
            return None

        # Long: lower band + oversold
        if bar.low <= bb_lower and rsi_val <= self.rsi_oversold:
            entry = bar.close
            stop = entry - atr_val * self.atr_stop_mult
            size = risk_adjusted_size(cash, entry, stop, self.risk_per_trade,
                                      self._peak_equity, self.max_dd_halt)
            if size > 0:
                self._bars_since_trade = 0
                return Trade(entry_bar=bar, side=1, size=size,
                             entry_price=entry, stop_price=stop, take_profit=bb_mid)

        # Short: upper band + overbought
        if bar.high >= bb_upper and rsi_val >= self.rsi_overbought:
            entry = bar.close
            stop = entry + atr_val * self.atr_stop_mult
            size = risk_adjusted_size(cash, entry, stop, self.risk_per_trade,
                                      self._peak_equity, self.max_dd_halt)
            if size > 0:
                self._bars_since_trade = 0
                return Trade(entry_bar=bar, side=-1, size=size,
                             entry_price=entry, stop_price=stop, take_profit=bb_mid)

        return None


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
    ):
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult
        self.risk_per_trade = risk_per_trade
        self.max_dd_halt = max_dd_halt
        self.cooldown_bars = cooldown_bars

        self._atr = ATR(atr_period)
        self._closes: List[float] = []
        self._peak_equity = 0.0
        self._bars_since_trade = 999

    def on_bar(self, i: int, bar: Bar, cash: float) -> Optional[Trade]:
        self._closes.append(bar.close)
        atr_val = self._atr.update(bar.high, bar.low, bar.close)
        self._peak_equity = max(self._peak_equity, cash)
        self._bars_since_trade += 1

        if len(self._closes) <= self.lookback or atr_val is None or atr_val <= 0:
            return None
        if self._bars_since_trade < self.cooldown_bars:
            return None

        roc = (self._closes[-1] - self._closes[-self.lookback - 1]) / self._closes[-self.lookback - 1]

        if roc > self.entry_threshold:
            entry = bar.close
            stop = entry - atr_val * self.atr_stop_mult
            tp = entry + atr_val * self.atr_target_mult
            size = risk_adjusted_size(cash, entry, stop, self.risk_per_trade,
                                      self._peak_equity, self.max_dd_halt)
            if size > 0:
                self._bars_since_trade = 0
                return Trade(entry_bar=bar, side=1, size=size,
                             entry_price=entry, stop_price=stop, take_profit=tp)

        if roc < -self.entry_threshold:
            entry = bar.close
            stop = entry + atr_val * self.atr_stop_mult
            tp = entry - atr_val * self.atr_target_mult
            size = risk_adjusted_size(cash, entry, stop, self.risk_per_trade,
                                      self._peak_equity, self.max_dd_halt)
            if size > 0:
                self._bars_since_trade = 0
                return Trade(entry_bar=bar, side=-1, size=size,
                             entry_price=entry, stop_price=stop, take_profit=tp)

        return None


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
    ):
        self.channel_period = channel_period
        self.atr_stop_mult = atr_stop_mult
        self.risk_reward = risk_reward
        self.risk_per_trade = risk_per_trade
        self.max_dd_halt = max_dd_halt
        self.cooldown_bars = cooldown_bars

        self._atr = ATR(atr_period)
        self._highs: List[float] = []
        self._lows: List[float] = []
        self._peak_equity = 0.0
        self._bars_since_trade = 999

    def on_bar(self, i: int, bar: Bar, cash: float) -> Optional[Trade]:
        self._highs.append(bar.high)
        self._lows.append(bar.low)
        atr_val = self._atr.update(bar.high, bar.low, bar.close)
        self._peak_equity = max(self._peak_equity, cash)
        self._bars_since_trade += 1

        if len(self._highs) <= self.channel_period or atr_val is None or atr_val <= 0:
            return None
        if self._bars_since_trade < self.cooldown_bars:
            return None

        # Channel from lookback window (excluding current bar)
        ch_high = max(self._highs[-(self.channel_period + 1):-1])
        ch_low = min(self._lows[-(self.channel_period + 1):-1])

        if bar.close > ch_high:
            entry = bar.close
            stop = entry - atr_val * self.atr_stop_mult
            tp = entry + (entry - stop) * self.risk_reward
            size = risk_adjusted_size(cash, entry, stop, self.risk_per_trade,
                                      self._peak_equity, self.max_dd_halt)
            if size > 0:
                self._bars_since_trade = 0
                return Trade(entry_bar=bar, side=1, size=size,
                             entry_price=entry, stop_price=stop, take_profit=tp)

        if bar.close < ch_low:
            entry = bar.close
            stop = entry + atr_val * self.atr_stop_mult
            tp = entry - (stop - entry) * self.risk_reward
            size = risk_adjusted_size(cash, entry, stop, self.risk_per_trade,
                                      self._peak_equity, self.max_dd_halt)
            if size > 0:
                self._bars_since_trade = 0
                return Trade(entry_bar=bar, side=-1, size=size,
                             entry_price=entry, stop_price=stop, take_profit=tp)

        return None
