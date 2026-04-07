"""Shared utilities for trading strategies."""

from typing import Optional

from backtesting.indicators import EMA


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
# Trend filter
# ---------------------------------------------------------------------------

class TrendFilter:
    """Long-term EMA direction filter. Gates signal direction.

    When active, only long signals are allowed when price is above the
    filter EMA, and only short signals when price is below it. This
    eliminates counter-trend trades on assets in secular trends.

    Set ``period=0`` to disable (all signals pass through).
    """

    def __init__(self, period: int = 0):
        self._ema = EMA(period) if period > 0 else None
        self._value: Optional[float] = None

    def update(self, close: float) -> None:
        if self._ema is not None:
            self._value = self._ema.update(close)

    def allows(self, side: int, close: float) -> bool:
        """Return True if the signal direction is allowed."""
        if self._value is None:
            return True
        if side > 0:
            return close > self._value
        return close < self._value
