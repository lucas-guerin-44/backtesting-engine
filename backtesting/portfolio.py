"""Portfolio tracking with equity curve, drawdown, and margin-call liquidation.

Works with the Broker class for multi-asset portfolio-level backtests.
"""

from typing import Dict, Optional

import numpy as np
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
                 margin_rate: float = 0.1,
                 typical_daily_volume: Optional[float] = None,
                 impact_scaling: float = 0.5,
                 daily_volatility: Optional[float] = None,
                 funding_rate_annual: float = 0.0,
                 funding_rate_short: float = 0.0):
        self.cash = cash
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.max_leverage = max_leverage
        self.margin_rate = margin_rate
        self.typical_daily_volume = typical_daily_volume
        self.impact_scaling = impact_scaling
        self.daily_volatility = daily_volatility
        self.funding_rate_annual = funding_rate_annual
        self.funding_rate_short = funding_rate_short

        self.equity_curve = []
        self.peak_equity = cash
        self.max_drawdown = 0.0
        self.min_equity = cash

        self.broker = Broker(self)
        self.trade_count = 0
        self.trades = []

    def impact_slippage_bps(self, order_size: float) -> float:
        if self.typical_daily_volume is None or self.daily_volatility is None:
            return self.slippage_bps
        if self.typical_daily_volume <= 0 or order_size <= 0:
            return self.slippage_bps
        participation = order_size / self.typical_daily_volume
        impact = self.daily_volatility * (participation ** 0.5) * self.impact_scaling
        return self.slippage_bps + impact

    def compute_daily_volatility(self, close_prices: np.ndarray) -> None:
        if len(close_prices) < 20:
            return
        log_returns = np.diff(np.log(close_prices))
        self.daily_volatility = float(np.std(log_returns[-60:]) * 10000)

    def accrue_funding(self, bar_hours: float) -> float:
        if self.funding_rate_annual == 0 and self.funding_rate_short == 0:
            return 0.0
        total = 0.0
        for sym, trades in self.broker.positions.items():
            for tr in trades:
                notional = abs(tr.entry_price * tr.size)
                rate = self.funding_rate_annual if tr.side > 0 else self.funding_rate_short
                accrual = notional * (rate / 100.0) * (bar_hours / 24.0) / 365.0
                total += accrual
        self.cash -= total
        return total

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
