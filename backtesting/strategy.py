"""Abstract base class for trading strategies."""

from abc import ABC, abstractmethod
from typing import Optional

from backtesting.types import Bar, Trade


class Strategy(ABC):
    """Base class for all trading strategies.

    Subclasses must implement ``on_bar()``, which is called once per bar
    during a backtest. Return a ``Trade`` to enter a position, or ``None``
    to do nothing.

    Signals are executed on the **next bar's open** (no lookahead bias).
    The strategy sees the current bar's close and decides whether to trade;
    the Backtester queues the signal and fills it at the following bar's
    open price.

    Example
    -------
    >>> class MyStrategy(Strategy):
    ...     def on_bar(self, i: int, bar: Bar, equity: float) -> Optional[Trade]:
    ...         if bar.close > bar.open:
    ...             return Trade(
    ...                 entry_bar=bar, side=1, size=equity * 0.1 / bar.close,
    ...                 entry_price=bar.close,
    ...                 stop_price=bar.close * 0.98,
    ...                 take_profit=bar.close * 1.04,
    ...             )
    ...         return None
    """

    @abstractmethod
    def on_bar(self, i: int, bar: Bar, equity: float) -> Optional[Trade]:
        """Generate a trading signal for the current bar.

        Parameters
        ----------
        i : int
            Zero-based bar index in the backtest.
        bar : Bar
            Current OHLC bar.
        equity : float
            Current portfolio equity (cash + unrealized P&L). Use this for
            position sizing and drawdown tracking.

        Returns
        -------
        Trade or None
            A Trade object to open a new position (executed at next bar's
            open), or None to skip.
        """
        ...

    def manage_position(self, bar: Bar, trade: Trade) -> None:
        """Called each bar for every open position. Override to implement
        trailing stops or other position management.

        The default implementation does nothing. Strategies that need to
        update stops or take-profits on open trades should override this.

        Parameters
        ----------
        bar : Bar
            Current OHLC bar.
        trade : Trade
            The open trade to manage (stop_price can be mutated directly).
        """
