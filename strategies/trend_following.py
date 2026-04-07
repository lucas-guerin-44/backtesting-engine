"""Dual-EMA trend following strategy with ATR-based stops."""

from typing import Optional

from backtesting.indicators import ATR, EMA
from backtesting.strategy import Strategy
from backtesting.types import Bar, Trade

from strategies.base import TrendFilter, risk_adjusted_size


class TrendFollowingStrategy(Strategy):
    """Dual-EMA trend follower with ATR-based stops and trend re-entry.

    Evolution of the strategy
    -------------------------

    **v1 -- EMA crossover with fixed take-profit (baseline)**
    Classic dual-EMA crossover: enter when fast crosses slow, exit at a fixed
    ATR multiple take-profit. On XAUUSD D1 (2019-2024): +6% return, 0.28 Sharpe,
    19 trades. Gold gained +104% in that period -- the strategy captured almost
    none of it.

    **Problem identified: fixed TP caps profits.** Each trade grabs ~4x ATR
    ($80-100 on gold) and exits, while the underlying trend runs +$1500.
    The strategy takes small bites out of a massive move.

    **v2 -- Trailing stop (``use_trailing_stop=True``)**
    Replace the fixed TP with an ATR-based trailing stop that ratchets in the
    direction of profit. No profit cap -- the trade rides until the trend
    reverses. This is how managed-futures funds actually operate. Requires
    a wider stop distance (3.0x+ ATR) to survive normal pullbacks.

    **Problem identified: signal frequency.** Only ~6 EMA crossovers per year
    on D1. Once stopped out during a pullback, the strategy sits flat for
    months waiting for the next crossover -- while the trend continues.

    **v3 -- Trend re-entry (``allow_reentry=True``)**
    When stopped out but the trend is still intact (fast EMA > slow EMA and
    price above fast EMA), re-enter immediately without waiting for a full
    crossover. Roughly doubles the trade count (19 -> 31 on XAUUSD D1).

    **Combined result (trail + re-entry + optimized params):**
    XAUUSD D1: +12.6% return, 0.48 Sharpe, 31 trades. Improves 5 of 8
    instruments tested -- specifically the trending ones (gold, indices,
    crypto, oil). Hurts on mean-reverting pairs (EURUSD, GBPUSD) where
    re-entering the same direction is the wrong move.

    **v4 -- 200-bar trend filter (``trend_filter_period=200``)**
    The strategies were going both long and short on assets in secular
    uptrends (SPX500 +137%, NDX100 +238%, XAUUSD +103%). Every short
    trade was pure drag. A 200-bar EMA direction filter gates signals:
    only longs above the filter, only shorts below it.

    Result on the portfolio (7 assets, Equal Weight allocator):
    without filter -0.64%, with filter **+12.18%** (0.49 Sharpe, 3.25%
    max DD). The filter is the single biggest improvement -- it doesn't
    add complexity to the signal logic, it just stops taking the
    obviously wrong side of the market.

    Available on all four strategies via ``trend_filter_period``.

    Parameters
    ----------
    use_trailing_stop : bool
        If True, replace fixed TP with ATR trailing stop. Default False
        for backwards compatibility; set True for trend-following use.
    allow_reentry : bool
        If True, re-enter after stop-out when the trend is still intact.
        Default False; set True to roughly double signal frequency.
    trend_filter_period : int
        EMA period for long-term trend filter. When > 0, only allows long
        signals above the filter EMA and short signals below it. Set to
        200 for a standard trend filter. Default 0 (disabled).
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
        use_trailing_stop: bool = False,
        allow_reentry: bool = False,
        trend_filter_period: int = 0,
    ):
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult
        self.risk_per_trade = risk_per_trade
        self.max_dd_halt = max_dd_halt
        self.cooldown_bars = cooldown_bars
        self.use_trailing_stop = use_trailing_stop
        self.allow_reentry = allow_reentry

        self._fast_ema = EMA(fast_period)
        self._slow_ema = EMA(slow_period)
        self._atr = ATR(atr_period)
        self._trend_filter = TrendFilter(trend_filter_period)
        self._prev_fast: Optional[float] = None
        self._prev_slow: Optional[float] = None
        self._peak_equity = 0.0
        self._bars_since_trade = 999
        self._current_atr: float = 0.0
        self._has_position = False
        # Track the trend direction established by the last crossover.
        # +1 = bullish trend, -1 = bearish trend, 0 = no trend yet.
        self._trend_side: int = 0

    def on_bar(self, i: int, bar: Bar, equity: float) -> Optional[Trade]:
        fast_val = self._fast_ema.update(bar.close)
        slow_val = self._slow_ema.update(bar.close)
        atr_val = self._atr.update(bar.high, bar.low, bar.close)
        self._trend_filter.update(bar.close)
        self._peak_equity = max(self._peak_equity, equity)
        self._bars_since_trade += 1
        if atr_val is not None:
            self._current_atr = atr_val

        # Reset position flag -- manage_position() already set it True
        # this bar if a position still exists (it runs before on_bar).
        was_in_position = self._has_position
        self._has_position = False

        if fast_val is None or slow_val is None or atr_val is None or atr_val <= 0:
            self._prev_fast, self._prev_slow = fast_val, slow_val
            return None
        if self._prev_fast is None or self._prev_slow is None:
            self._prev_fast, self._prev_slow = fast_val, slow_val
            return None
        if self._bars_since_trade < self.cooldown_bars:
            self._prev_fast, self._prev_slow = fast_val, slow_val
            return None
        if was_in_position:
            self._prev_fast, self._prev_slow = fast_val, slow_val
            return None

        trade = None

        # Detect crossovers (primary entry signals)
        bullish_cross = self._prev_fast <= self._prev_slow and fast_val > slow_val
        bearish_cross = self._prev_fast >= self._prev_slow and fast_val < slow_val

        # Update trend direction on crossover
        if bullish_cross:
            self._trend_side = 1
        elif bearish_cross:
            self._trend_side = -1

        # Primary entry: crossover
        if bullish_cross and bar.close > slow_val:
            trade = self._build_trade(bar, 1, atr_val, equity)

        elif bearish_cross and bar.close < slow_val:
            trade = self._build_trade(bar, -1, atr_val, equity)

        # Re-entry: stopped out but trend is still intact
        elif self.allow_reentry and self._trend_side != 0:
            if (self._trend_side == 1
                    and fast_val > slow_val
                    and bar.close > fast_val):
                trade = self._build_trade(bar, 1, atr_val, equity)

            elif (self._trend_side == -1
                  and fast_val < slow_val
                  and bar.close < fast_val):
                trade = self._build_trade(bar, -1, atr_val, equity)

        self._prev_fast, self._prev_slow = fast_val, slow_val
        return trade

    def _build_trade(self, bar: Bar, side: int, atr_val: float,
                     equity: float) -> Optional[Trade]:
        """Construct a Trade with the appropriate stop and TP."""
        if not self._trend_filter.allows(side, bar.close):
            return None
        entry = bar.close
        stop = entry - side * atr_val * self.atr_stop_mult
        tp = (None if self.use_trailing_stop
              else entry + side * atr_val * self.atr_target_mult)
        size = risk_adjusted_size(equity, entry, stop, self.risk_per_trade,
                                  self._peak_equity, self.max_dd_halt)
        if size > 0:
            self._bars_since_trade = 0
            self._has_position = True
            return Trade(entry_bar=bar, side=side, size=size,
                         entry_price=entry, stop_price=stop, take_profit=tp)
        return None

    def manage_position(self, bar: Bar, trade: Trade) -> None:
        """Mark position as open; trail the stop if enabled."""
        self._has_position = True

        if not self.use_trailing_stop or self._current_atr <= 0:
            return

        trail_dist = self._current_atr * self.atr_stop_mult

        if trade.side > 0:
            new_stop = bar.close - trail_dist
            if new_stop > trade.stop_price:
                trade.stop_price = new_stop
        else:
            new_stop = bar.close + trail_dist
            if new_stop < trade.stop_price:
                trade.stop_price = new_stop
