"""Abstract base class for trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from backtesting.types import Bar, Position, Trade

if TYPE_CHECKING:
    from backtesting.broker import Broker


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

    def __init__(self):
        self._broker: Optional[Broker] = None
        self._peak_equity: float = 0.0
        self._bars_since_trade: int = 999
        self.cooldown_bars: int = 0
        self.warmup_bars: int = 0
        self._contract_multiplier: float = 1.0

    def warmup_bar(self, bar: Bar) -> None:
        """Prime indicator state without generating or recording a trade.

        Walk-forward evaluation calls this for historical context bars before
        each OOS window. Strategies with incremental indicators should override
        it and update only indicator/signal history; portfolio, trade, and
        cooldown state should remain untouched. ``warmup_bars`` controls the
        default context length used by walk-forward evaluation.
        """

    def _update_tracking(self, equity: float) -> None:
        """Update peak equity and bars-since-trade counter. Call at start of on_bar."""
        self._peak_equity = max(self._peak_equity, equity)
        self._bars_since_trade += 1

    def _can_trade(self) -> bool:
        """Return True if cooldown has elapsed."""
        return self._bars_since_trade >= self.cooldown_bars

    def _record_trade_entry(self) -> None:
        """Reset cooldown counter. Call when a trade is submitted."""
        self._bars_since_trade = 0

    def _make_trade(self, bar: Bar, side: int, entry: float, stop: float,
                    tp: float, size: float) -> Optional[Trade]:
        """Build a Trade if size > 0, reset cooldown, and return it.

        Convenience helper that eliminates the repeated long/short symmetry
        blocks in strategy implementations.
        """
        if size <= 0:
            return None
        self._record_trade_entry()
        return Trade(
            entry_bar=bar, side=side, size=size,
            entry_price=entry, stop_price=stop, take_profit=tp,
        )

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

    # --- Position query API (available after backtester sets _broker) ---

    def get_position(self, symbol: str) -> Optional[Position]:
        """Return the Position object for a symbol, or None if flat."""
        if self._broker is None:
            return None
        return self._broker.positions.get(symbol)

    def net_size(self, symbol: str) -> float:
        """Return the signed net size for a symbol (+long, -short, 0 flat)."""
        pos = self.get_position(symbol)
        return pos.net_size if pos is not None else 0.0

    def net_side(self, symbol: str) -> int:
        """Return +1 long, -1 short, or 0 flat for a symbol."""
        pos = self.get_position(symbol)
        return pos.net_side if pos is not None else 0

    def average_entry(self, symbol: str) -> float:
        """Return the volume-weighted average entry price for a symbol."""
        pos = self.get_position(symbol)
        return pos.average_entry if pos is not None else 0.0

    def unrealized_pnl(self, symbol: str, current_price: float) -> float:
        """Return unrealized P&L for a symbol at the given price."""
        pos = self.get_position(symbol)
        return pos.unrealized_pnl(current_price) if pos is not None else 0.0

    def lot_count(self, symbol: str) -> int:
        """Return the number of open lots for a symbol."""
        pos = self.get_position(symbol)
        return len(pos) if pos is not None else 0
