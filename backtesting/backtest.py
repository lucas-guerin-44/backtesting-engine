"""Core backtesting engine that drives strategy evaluation over historical data.

The Backtester orchestrates the simulation loop, delegating trade execution
to the Broker (gap-aware stops, slippage, commission, leverage checks) and
equity/drawdown tracking to the Portfolio (margin calls, peak tracking).

Performance notes:
- Equity curve is pre-allocated as a numpy array.
- OHLC columns are extracted to numpy arrays upfront.
- Timestamps are pre-boxed to pd.Timestamp once in __init__ (not per-bar).
- Broker/Portfolio calls are skipped when there are no open positions.
- The main loop uses local variables to minimize attribute lookups.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd

from backtesting.data import validate_ohlc
from backtesting.portfolio import Portfolio
from backtesting.types import Bar, Trade
from utils import infer_freq_per_year


class Backtester:
    """Event-driven backtester that iterates over OHLC bars and manages positions.

    Delegates trade execution to ``Broker`` and equity tracking to ``Portfolio``.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC DataFrame indexed by timestamp with columns: open, high, low, close.
    strategy : object
        Any object implementing ``on_bar(i, bar, cash) -> Optional[Trade]``.
    starting_cash : float
        Initial portfolio cash.
    commission_bps : float
        Commission in basis points applied on entry and exit.
    slippage_bps : float
        Slippage in basis points applied to fill prices.
    max_leverage : float
        Maximum gross leverage (1.0 = no leverage, 3.0 = 3x).
    margin_rate : float
        Margin requirement as a fraction of gross notional. Set to 0.0 to disable
        margin calls.
    symbol : str
        Instrument identifier used to key positions in the Broker.
    """

    def __init__(self, df: pd.DataFrame, strategy, starting_cash: float = 10_000,
                 commission_bps: float = 0.0, slippage_bps: float = 0.0,
                 max_leverage: float = 1.0, margin_rate: float = 0.0,
                 symbol: str = "default"):
        self.strategy = strategy
        self.starting_cash = starting_cash
        self.symbol = symbol

        # Portfolio owns the Broker; together they handle execution and tracking
        self.portfolio = Portfolio(
            cash=starting_cash,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            max_leverage=max_leverage,
            margin_rate=margin_rate,
        )
        self.broker = self.portfolio.broker

        # Validate data before proceeding
        report = validate_ohlc(df)
        report.raise_if_invalid()

        # Pre-extract numpy arrays for fast bar construction
        self._open = df["open"].to_numpy(dtype=np.float64)
        self._high = df["high"].to_numpy(dtype=np.float64)
        self._low = df["low"].to_numpy(dtype=np.float64)
        self._close = df["close"].to_numpy(dtype=np.float64)
        # Pre-box all timestamps to pd.Timestamp once (avoids DatetimeIndex
        # __getitem__ boxing overhead per bar — was 46% of total runtime)
        self._ts = list(pd.DatetimeIndex(df.index))
        self.n = len(df)

        # Infer annualization factor from the data's actual frequency
        self.freq_per_year = infer_freq_per_year(self._ts)

        # Pre-allocate output array
        self.equity_curve = np.empty(self.n, dtype=np.float64)

    def run(self, execution_priority: str = "stop_first") -> Tuple[np.ndarray, List[Trade]]:
        """Run the backtest over all bars.

        Parameters
        ----------
        execution_priority : str
            How to resolve bars where both stop and TP are hit:
            ``"stop_first"`` (conservative default) or ``"tp_first"``.

        Returns
        -------
        tuple of (np.ndarray, list[Trade])
            The equity curve and list of closed trades.
        """
        if execution_priority == "stop_first":
            exit_order = (self.broker.close_due_to_stop, self.broker.close_due_to_tp)
        elif execution_priority == "tp_first":
            exit_order = (self.broker.close_due_to_tp, self.broker.close_due_to_stop)
        else:
            raise ValueError(f"Unknown execution_priority: {execution_priority}")

        # Local references for speed (avoids self.X lookups in tight loop)
        portfolio = self.portfolio
        broker = self.broker
        positions = broker.positions
        symbol = self.symbol
        on_bar = self.strategy.on_bar
        equity_curve = self.equity_curve
        o, h, lo, c, ts = self._open, self._high, self._low, self._close, self._ts
        n = self.n
        exit_first, exit_second = exit_order

        # Cache portfolio fields as locals for the fast path
        cash = portfolio.cash
        peak_equity = portfolio.peak_equity
        max_drawdown = portfolio.max_drawdown

        # Track whether margin checks are needed
        has_margin = portfolio.margin_rate > 0

        for i in range(n):
            bar = Bar(ts[i], o[i], h[i], lo[i], c[i])
            close_i = c[i]

            # Fast path: check if we have any open positions
            open_pos = positions.get(symbol)
            has_positions = open_pos is not None and len(open_pos) > 0

            if has_positions:
                # Process exits (gap-aware via Broker)
                exit_first(symbol, bar)
                exit_second(symbol, bar)

                # Recheck after exits
                open_pos = positions.get(symbol)
                has_positions = open_pos is not None and len(open_pos) > 0

            # Sync cash from portfolio (may have changed from exits)
            cash = portfolio.cash

            if has_positions:
                # Compute available cash (subtract reserved notional)
                reserved = 0.0
                open_pnl = 0.0
                for tr in open_pos:
                    reserved += tr.entry_price * tr.size
                    open_pnl += (close_i - tr.entry_price) * tr.side * tr.size
                available_cash = cash - reserved
                equity = cash + open_pnl
            else:
                available_cash = cash
                equity = cash

            # Ask strategy for a new signal (pass equity so the strategy
            # can correctly track peak equity and compute drawdown)
            new_trade = on_bar(i, bar, equity)

            # Execute entry via Broker
            if new_trade is not None and new_trade.size > 0:
                broker.open_trade(
                    symbol=symbol, bar=bar,
                    side=new_trade.side, size=new_trade.size,
                    stop=new_trade.stop_price, take_profit=new_trade.take_profit,
                    entry_price=new_trade.entry_price,
                )
                # Cash may have changed from commission
                cash = portfolio.cash

                # Recalculate equity with new position
                open_pos = positions.get(symbol)
                if open_pos:
                    open_pnl = 0.0
                    for tr in open_pos:
                        open_pnl += (close_i - tr.entry_price) * tr.side * tr.size
                    equity = cash + open_pnl
                    has_positions = True
                else:
                    equity = cash

            # Update drawdown tracking (inlined from Portfolio.update for speed)
            peak_equity = max(peak_equity, equity)
            if peak_equity > 0:
                dd = (peak_equity - equity) / peak_equity
                if dd > max_drawdown:
                    max_drawdown = dd

            # Margin call check (only when positions exist and margin is configured)
            if has_positions and has_margin:
                gross = 0.0
                for tr in open_pos:
                    gross += abs(close_i * tr.size)
                if gross > 0:
                    margin_req = gross * portfolio.margin_rate
                    if equity < 0.5 * margin_req:
                        current_prices = {symbol: close_i}
                        portfolio.update(ts[i], current_prices)
                        cash = portfolio.cash
                        equity = cash
                        # Portfolio.update handles the liquidation internally

            # Write equity to pre-allocated array
            equity_curve[i] = equity if equity > 0 else 0.0

        # Sync final state back to portfolio
        portfolio.peak_equity = peak_equity
        portfolio.max_drawdown = max_drawdown

        # Expose final state for external access
        self.cash = portfolio.cash
        self.positions = broker.positions.get(symbol, [])
        self.trades = broker.closed_trades
        self.max_drawdown = max_drawdown

        return equity_curve, broker.closed_trades
