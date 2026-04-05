"""Abstract base class for trading strategies."""

from abc import ABC, abstractmethod
from typing import Optional

from backtesting.types import Bar, Trade


class Strategy(ABC):
    """Base class for all trading strategies.

    Subclasses must implement ``on_bar()``, which is called once per bar
    during a backtest. Return a ``Trade`` to enter a position, or ``None``
    to do nothing.

    Example
    -------
    >>> class MyStrategy(Strategy):
    ...     def on_bar(self, i: int, bar: Bar, cash: float) -> Optional[Trade]:
    ...         if bar.close > bar.open:
    ...             return Trade(
    ...                 entry_bar=bar, side=1, size=cash * 0.1 / bar.close,
    ...                 entry_price=bar.close,
    ...                 stop_price=bar.close * 0.98,
    ...                 take_profit=bar.close * 1.04,
    ...             )
    ...         return None
    """

    @abstractmethod
    def on_bar(self, i: int, bar: Bar, cash: float) -> Optional[Trade]:
        """Generate a trading signal for the current bar.

        Parameters
        ----------
        i : int
            Zero-based bar index in the backtest.
        bar : Bar
            Current OHLC bar.
        cash : float
            Available cash (after accounting for open positions).

        Returns
        -------
        Trade or None
            A Trade object to open a new position, or None to skip.
        """
        ...
