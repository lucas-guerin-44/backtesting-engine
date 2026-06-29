"""Broker simulation with leverage, margin, slippage, and commission support.

Provides the trade execution layer for the Backtester with:
- Buying power limits based on leverage
- Margin check before entry
- Gap-aware stop-loss execution (fills at open if price gaps past stop)
- Per-trade slippage and commission
- FIFO/LIFO netting via Position lot management
"""

from typing import Dict, List, Optional

import pandas as pd

from backtesting.types import Bar, Lot, Position, Trade


class Broker:
    """Simulated broker that executes trades against a Portfolio.

    Parameters
    ----------
    portfolio : Portfolio
        The portfolio instance this broker operates on.
    """

    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Trade] = []

    def _price_with_slippage(self, px: float, side: int, order_size: float = 0.0) -> float:
        slip_bps = self.portfolio.impact_slippage_bps(order_size)
        return px * (1 + side * slip_bps / 1e4)

    def _commission_cost(self, notional: float) -> float:
        return notional * (self.portfolio.commission_bps / 1e4)

    def _remaining_buying_power(self, current_prices: Dict[str, float]) -> float:
        equity = self.portfolio.compute_equity(current_prices)
        gross = self.portfolio.gross_notional(current_prices)
        limit = equity * self.portfolio.max_leverage
        return max(0.0, limit - gross)

    def has_open_position(self, symbol: str) -> bool:
        """Return True if there is at least one open lot for this symbol."""
        return symbol in self.positions and len(self.positions[symbol]) > 0

    def update_stop(self, symbol: str, new_stop: float) -> None:
        """Update the stop price for all open lots of a symbol."""
        if symbol not in self.positions:
            return
        for lot in self.positions[symbol]:
            lot.stop_price = new_stop

    def open_trade(self, symbol: str, bar: Bar, side: int, size: float,
                   stop: float, take_profit: float,
                   entry_price: Optional[float] = None,
                   current_prices: Optional[Dict[str, float]] = None) -> None:
        """Open a new lot, subject to buying power and margin checks.

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
        current_prices : dict, optional
            Current prices for all assets. Used for accurate buying power and
            margin checks in multi-asset portfolios. If None, falls back to
            ``{symbol: bar.close}``.
        """
        if size <= 0:
            return

        raw_price = entry_price if entry_price is not None else bar.open
        prices = current_prices if current_prices is not None else {symbol: bar.close}

        notional = abs(raw_price * size)
        remaining_bp = self._remaining_buying_power(prices)
        if notional > remaining_bp:
            size = remaining_bp / raw_price
            notional = abs(raw_price * size)
        if size <= 0:
            return

        commission = self._commission_cost(notional)
        if self.portfolio.cash < commission:
            return

        entry_px = self._price_with_slippage(raw_price, side, size)

        # Margin check: ensure equity covers projected margin requirement
        projected_gross = self.portfolio.gross_notional(prices) + notional
        projected_margin = projected_gross * self.portfolio.margin_rate
        projected_equity = self.portfolio.compute_equity(prices)
        if projected_equity < projected_margin:
            return

        lot = Lot(
            entry_bar=bar, side=side, size=size, entry_price=entry_px,
            stop_price=stop, take_profit=take_profit, symbol=symbol,
        )

        self.positions.setdefault(symbol, Position()).append(lot)
        self.portfolio.cash -= commission
        self.portfolio.trade_count += 1

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
        Each lot is checked individually (lot-level stop management).
        """
        if symbol not in self.positions:
            return
        still_open = []
        for lot in self.positions[symbol]:
            exit_raw = None
            if lot.side > 0:
                if bar.open <= lot.stop_price:
                    exit_raw = bar.open  # Gapped through stop
                elif bar.low <= lot.stop_price <= bar.high:
                    exit_raw = lot.stop_price
            else:
                if bar.open >= lot.stop_price:
                    exit_raw = bar.open  # Gapped through stop
                elif bar.low <= lot.stop_price <= bar.high:
                    exit_raw = lot.stop_price

            if exit_raw is not None:
                self._close_lot(lot, exit_raw, symbol, bar)
            else:
                still_open.append(lot)
        self.positions[symbol].lots = still_open

    def close_due_to_tp(self, symbol: str, bar: Bar) -> None:
        """Execute take-profit orders. Each lot checked individually."""
        if symbol not in self.positions:
            return
        still_open = []
        for lot in self.positions[symbol]:
            exit_raw = None
            if lot.take_profit is not None:
                if lot.side > 0 and bar.high >= lot.take_profit:
                    exit_raw = lot.take_profit
                elif lot.side < 0 and bar.low <= lot.take_profit:
                    exit_raw = lot.take_profit

            if exit_raw is not None:
                self._close_lot(lot, exit_raw, symbol, bar)
            else:
                still_open.append(lot)
        self.positions[symbol].lots = still_open

    def close_trade(self, symbol: str, lot: Lot, bar: Bar) -> None:
        """Force-close a specific lot at the bar's open price."""
        if symbol not in self.positions or lot not in self.positions[symbol]:
            return
        self._close_lot(lot, bar.open, symbol, bar)
        self.positions[symbol].remove(lot)

    def _close_lot(self, lot: Lot, raw_price: float, symbol: str, bar: Bar) -> None:
        """Apply slippage/commission, convert lot to trade, and record it."""
        exit_px = self._price_with_slippage(raw_price, -lot.side, lot.size)
        notional = abs(exit_px * lot.size)
        commission = self._commission_cost(notional)

        # Convert lot to a closed Trade record
        trade = Trade(
            entry_bar=lot.entry_bar, side=lot.side, size=lot.size,
            entry_price=lot.entry_price, stop_price=lot.stop_price,
            take_profit=lot.take_profit, symbol=symbol or lot.symbol,
            limit_price=None, bars_held=lot.bars_held,
            exit_price=exit_px, exit_ts=bar.ts,
            pnl=(exit_px - lot.entry_price) * lot.side * lot.size,
            metadata=lot.metadata,
        )
        self.portfolio.cash += trade.pnl - commission
        self.closed_trades.append(trade)

    def close_lot_at_price(self, lot: Lot, price: float, symbol: str, ts) -> None:
        """Close a lot at a given price without requiring a Bar object.

        Used by the tick engine for inline stop/TP execution.
        """
        synthetic_bar = Bar.at_price(ts, price)
        self._close_lot(lot, price, symbol, synthetic_bar)

    def check_stop_tp_tick(self, symbol: str, tick_price: float, ts) -> None:
        """Check all lots for stop/TP triggers at tick-level granularity.

        Used by the tick engine instead of inlining stop/TP logic.
        Each lot is checked individually; triggered lots are closed and removed.
        """
        if symbol not in self.positions:
            return
        position = self.positions[symbol]
        still_open = []
        for lot in position:
            stop_triggered = False
            tp_triggered = False

            if lot.side > 0:
                if tick_price <= lot.stop_price:
                    stop_triggered = True
                elif lot.take_profit is not None and tick_price >= lot.take_profit:
                    tp_triggered = True
            else:
                if tick_price >= lot.stop_price:
                    stop_triggered = True
                elif lot.take_profit is not None and tick_price <= lot.take_profit:
                    tp_triggered = True

            if stop_triggered or tp_triggered:
                self.close_lot_at_price(lot, tick_price, symbol, ts)
            else:
                still_open.append(lot)
        position.lots = still_open
