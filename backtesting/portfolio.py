"""Portfolio tracking with equity curve, drawdown, and margin-call liquidation.

Works with the Broker class for multi-asset portfolio-level backtests.
"""

from typing import Dict

import pandas as pd

from backtesting.broker import Broker
from backtesting.types import Bar


class Portfolio:
    """Tracks cash, equity, drawdown, and delegates trade execution to a Broker.

    Parameters
    ----------
    cash : float
        Initial cash balance.
    commission_bps : float
        Commission in basis points per trade.
    slippage_bps : float
        Slippage in basis points.
    max_leverage : float
        Maximum gross leverage allowed.
    margin_rate : float
        Margin requirement as a fraction of gross notional.
    """

    def __init__(self, cash: float = 100_000.0, commission_bps: float = 0.5,
                 slippage_bps: float = 1.0, max_leverage: float = 3.0,
                 margin_rate: float = 0.1):
        self.cash = cash
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.max_leverage = max_leverage
        self.margin_rate = margin_rate

        self.equity_curve = []
        self.peak_equity = cash
        self.max_drawdown = 0.0
        self.min_equity = cash

        self.broker = Broker(self)
        self.trade_count = 0
        self.trades = []

    def record_trade(self, ts, instrument: str, price: float, qty: float,
                     side: int, tag: str = None) -> None:
        """Append a trade record to the trade log."""
        self.trades.append({
            "ts": pd.to_datetime(ts),
            "instrument": instrument,
            "price": price,
            "qty": qty,
            "side": side,
            "tag": tag,
        })
        self.trade_count += 1

    def compute_equity(self, current_prices: Dict[str, float]) -> float:
        """Compute total equity: cash + mark-to-market of open positions."""
        open_pnl = 0.0
        for sym, trades in self.broker.positions.items():
            px = current_prices.get(sym)
            if px is None:
                continue
            for tr in trades:
                open_pnl += (px - tr.entry_price) * tr.side * tr.size
        return self.cash + open_pnl

    def gross_notional(self, current_prices: Dict[str, float]) -> float:
        """Compute total gross notional exposure across all open positions."""
        gross = 0.0
        for sym, trades in self.broker.positions.items():
            px = current_prices.get(sym)
            if px is None:
                continue
            for tr in trades:
                gross += abs(px * tr.size)
        return gross

    def update(self, ts: pd.Timestamp, current_prices: Dict[str, float]) -> None:
        """Update equity curve, drawdown tracking, and check for margin calls."""
        equity = self.compute_equity(current_prices)
        self.equity_curve.append((ts, equity))

        self.peak_equity = max(self.peak_equity, equity)
        if self.peak_equity > 0:
            dd = (self.peak_equity - equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, dd)
        self.min_equity = min(self.min_equity, equity)

        # Margin call: liquidate all positions if equity < 50% of margin requirement
        gross = self.gross_notional(current_prices)
        if gross > 0:
            margin_req = gross * self.margin_rate
            if equity < 0.5 * margin_req:
                for sym in list(self.broker.positions.keys()):
                    for tr in list(self.broker.positions[sym]):
                        synthetic_bar = Bar(ts, current_prices[sym], current_prices[sym],
                                            current_prices[sym], current_prices[sym])
                        self.broker.close_trade(sym, tr, synthetic_bar)
