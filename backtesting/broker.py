"""Broker simulation with leverage, margin, slippage, and commission support.

Provides the trade execution layer for the Backtester with:
- Buying power limits based on leverage
- Margin check before entry
- Gap-aware stop-loss execution (fills at open if price gaps past stop)
- Per-trade slippage and commission
"""

from typing import Dict, List, Optional

import pandas as pd

from backtesting.types import Bar, Trade


class Broker:
    """Simulated broker that executes trades against a Portfolio.

    Parameters
    ----------
    portfolio : Portfolio
        The portfolio instance this broker operates on.
    """

    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.positions: Dict[str, List[Trade]] = {}
        self.closed_trades: List[Trade] = []

    def _price_with_slippage(self, px: float, side: int) -> float:
        return px * (1 + side * self.portfolio.slippage_bps / 1e4)

    def _commission_cost(self, notional: float) -> float:
        return notional * (self.portfolio.commission_bps / 1e4)

    def _remaining_buying_power(self, current_prices: Dict[str, float]) -> float:
        equity = self.portfolio.compute_equity(current_prices)
        gross = self.portfolio.gross_notional(current_prices)
        limit = equity * self.portfolio.max_leverage
        return max(0.0, limit - gross)

    def has_open_position(self, symbol: str) -> bool:
        """Return True if there is at least one open trade for this symbol."""
        return symbol in self.positions and len(self.positions[symbol]) > 0

    def position_side(self, symbol: str) -> int:
        """Return the net side of open positions (+1 long, -1 short, 0 flat)."""
        if not self.has_open_position(symbol):
            return 0
        return self.positions[symbol][0].side

    def update_stop(self, symbol: str, new_stop: float) -> None:
        """Update the stop price for all open trades of a symbol."""
        if symbol not in self.positions:
            return
        for tr in self.positions[symbol]:
            tr.stop_price = new_stop

    def open_trade(self, symbol: str, bar: Bar, side: int, size: float,
                   stop: float, take_profit: float,
                   entry_price: Optional[float] = None) -> None:
        """Open a new trade, subject to buying power and margin checks.

        Parameters
        ----------
        symbol : str
            Instrument identifier.
        bar : Bar
            Current price bar (used for margin/buying-power checks).
        side : int
            +1 for long, -1 for short.
        size : float
            Number of units to trade.
        stop : float
            Stop-loss price.
        take_profit : float
            Take-profit price.
        entry_price : float, optional
            Desired fill price. If None, fills at ``bar.open`` (next-bar-open
            execution). When provided, the strategy's specified price is used
            (e.g., ``bar.close`` for same-bar execution).
        """
        if size <= 0:
            return

        raw_price = entry_price if entry_price is not None else bar.open

        notional = abs(raw_price * size)
        remaining_bp = self._remaining_buying_power({symbol: bar.close})
        if notional > remaining_bp:
            size = remaining_bp / raw_price
            notional = abs(raw_price * size)
        if size <= 0:
            return

        commission = self._commission_cost(notional)
        if self.portfolio.cash < commission:
            return

        entry_px = self._price_with_slippage(raw_price, side)

        # Margin check: ensure equity covers projected margin requirement
        projected_gross = self.portfolio.gross_notional({symbol: bar.close}) + notional
        projected_margin = projected_gross * self.portfolio.margin_rate
        projected_equity = self.portfolio.compute_equity({symbol: bar.close})
        if projected_equity < projected_margin:
            return

        trade = Trade(
            entry_bar=bar, side=side, size=size, entry_price=entry_px,
            stop_price=stop, take_profit=take_profit,
        )
        trade.bars_held = 0

        self.positions.setdefault(symbol, []).append(trade)
        self.portfolio.cash -= commission
        self.portfolio.trade_count += 1
        self.portfolio.record_trade(bar.ts, symbol, entry_px, size, side, tag="entry")

    def position_size(self, symbol: str, price: float, stop: float,
                      risk_fraction: float = 0.01) -> float:
        """Compute trade size based on a fixed-risk fraction of equity."""
        equity = self.portfolio.compute_equity({symbol: price})
        gross = self.portfolio.gross_notional({symbol: price})
        remaining_bp = max(0.0, equity * self.portfolio.max_leverage - gross)

        risk_per_unit = abs(price - stop)
        if risk_per_unit <= 0:
            return 0.0

        units = (equity * risk_fraction) / risk_per_unit
        return min(units, remaining_bp / price)

    def close_due_to_stop(self, symbol: str, bar: Bar) -> None:
        """Execute stop losses with gap-aware fill logic.

        If the bar opens past the stop (gap), the fill is at the open price
        (worse than the stop). Otherwise, the fill is at the stop price.
        """
        if symbol not in self.positions:
            return
        still_open = []
        for tr in self.positions[symbol]:
            exit_raw = None
            if tr.side > 0:
                if bar.open <= tr.stop_price:
                    exit_raw = bar.open  # Gapped through stop
                elif bar.low <= tr.stop_price <= bar.high:
                    exit_raw = tr.stop_price
            else:
                if bar.open >= tr.stop_price:
                    exit_raw = bar.open  # Gapped through stop
                elif bar.low <= tr.stop_price <= bar.high:
                    exit_raw = tr.stop_price

            if exit_raw is not None:
                self._close_position(tr, exit_raw, symbol, bar)
            else:
                still_open.append(tr)
        self.positions[symbol] = still_open

    def close_due_to_tp(self, symbol: str, bar: Bar) -> None:
        """Execute take-profit orders."""
        if symbol not in self.positions:
            return
        still_open = []
        for tr in self.positions[symbol]:
            exit_raw = None
            if tr.take_profit is not None:
                if tr.side > 0 and bar.high >= tr.take_profit:
                    exit_raw = tr.take_profit
                elif tr.side < 0 and bar.low <= tr.take_profit:
                    exit_raw = tr.take_profit

            if exit_raw is not None:
                self._close_position(tr, exit_raw, symbol, bar)
            else:
                still_open.append(tr)
        self.positions[symbol] = still_open

    def close_trade(self, symbol: str, tr: Trade, bar: Bar) -> None:
        """Force-close a specific trade at the bar's open price."""
        if symbol not in self.positions or tr not in self.positions[symbol]:
            return
        self._close_position(tr, bar.open, symbol, bar)
        self.positions[symbol].remove(tr)

    def _close_position(self, tr: Trade, raw_price: float, symbol: str, bar: Bar) -> None:
        """Apply slippage/commission and record the closed trade."""
        exit_px = self._price_with_slippage(raw_price, -tr.side)
        notional = abs(exit_px * tr.size)
        commission = self._commission_cost(notional)
        tr.exit_price = exit_px
        tr.pnl = (tr.exit_price - tr.entry_price) * tr.side * tr.size
        self.portfolio.cash += tr.pnl - commission
        self.closed_trades.append(tr)
        self.portfolio.record_trade(bar.ts, symbol, exit_px, tr.size, -tr.side, tag="exit")
