"""Abstract base class for trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Union

from backtesting.types import Bar, Trade

if TYPE_CHECKING:
    from backtesting.order import Order


class Strategy(ABC):
    """Base class for all trading strategies.

    Subclasses must implement ``on_bar()``, which is called once per bar
    during a backtest. Return a ``Trade`` to enter a position, or ``None``
    to do nothing.

    Signals are executed on the **next bar's open** (no lookahead bias).
    The strategy sees the current bar's close and decides whether to trade;
    the Backtester queues the signal and fills it at the following bar's
    open price.

    For tick-level backtesting, strategies can also override ``on_tick()``
    to react to individual price updates within a bar. The default
    implementation delegates to ``on_bar()`` only when a bar completes —
    override it for intra-bar logic (limit orders, micro-structure signals,
    tighter stop management, etc.).

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
    def on_bar(self, i: int, bar: Bar, equity: float) -> Optional[Union[Trade, Order]]:
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

    def on_tick(self, tick, current_bar: Optional[Bar], equity: float) -> Optional[Union[Trade, Order]]:
        """Called on every tick during tick-level backtesting.

        Override this for intra-bar logic. The default does nothing — signal
        generation happens in ``on_bar()`` when a bar completes.

        Parameters
        ----------
        tick : Tick
            The raw tick (timestamp, price, volume, optional bid/ask).
        current_bar : Bar or None
            The in-progress partial bar (OHLC updated with each tick but
            not yet closed). None before the first tick.
        equity : float
            Current portfolio equity.

        Returns
        -------
        Trade or None
            A Trade to open (filled at tick price with slippage), or None.
        """
        return None

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

    def manage_position_tick(self, tick, trade: Trade) -> None:
        """Called on every tick for open positions during tick-level backtesting.

        Override for tick-granularity stop management (e.g., tighter trailing
        stops that react to every price update). The default does nothing.

        Parameters
        ----------
        tick : Tick
            The raw tick.
        trade : Trade
            The open trade to manage.
        """
