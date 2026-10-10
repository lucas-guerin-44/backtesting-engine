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
from backtesting.types import BacktestConfig, Bar, Trade
from backtesting.utils import infer_freq_per_year


class Backtester:
    """Event-driven backtester that iterates over OHLC bars and manages positions.

    Delegates trade execution to ``Broker`` and equity tracking to ``Portfolio``.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC DataFrame indexed by timestamp with columns: open, high, low, close.
    strategy : object
        Any object implementing ``on_bar(i, bar, equity) -> Optional[Trade]``.
    config : BacktestConfig, optional
        Shared configuration (cash, commission, slippage, leverage, margin).
        If provided, individual kwargs are ignored.
    starting_cash : float
        Initial portfolio cash (ignored if ``config`` is provided).
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
    freq_per_year : int, optional
        Annualization periods used by downstream metrics. If omitted, estimated
        from the observed timestamps; specify explicitly for short samples or
        when using a chosen market-calendar convention.
    contract_multipliers : dict, optional
        Per-symbol multiplier converting a price change into account-currency
        P&L (e.g. ``{"default": 50}`` for an index future). Defaults to 1.0.
    """

    def __init__(self, df: pd.DataFrame, strategy,
                 config: BacktestConfig = None,
                  starting_cash: float = 10_000,
                  commission_bps: float = 0.0, slippage_bps: float = 0.0,
                  max_leverage: float = 1.0, margin_rate: float = 0.0,
                  symbol: str = "default", freq_per_year: int = None,
                  contract_multipliers: dict = None):
        if config is not None:
            starting_cash = config.starting_cash
            commission_bps = config.commission_bps
            slippage_bps = config.slippage_bps
            max_leverage = config.max_leverage
            margin_rate = config.margin_rate

        self.strategy = strategy
        self.starting_cash = starting_cash
        self.symbol = symbol

        # Portfolio owns the Broker; together they handle execution and tracking
        if config is not None:
            self.portfolio = Portfolio(**config.to_kwargs())
        else:
            self.portfolio = Portfolio(
                cash=starting_cash,
                commission_bps=commission_bps,
                slippage_bps=slippage_bps,
                max_leverage=max_leverage,
                margin_rate=margin_rate,
            )
        if contract_multipliers:
            self.portfolio.contract_multipliers = dict(contract_multipliers)
        self.broker = self.portfolio.broker
        strategy._broker = self.broker

        # Validate data before proceeding
        report = validate_ohlc(df)
        report.raise_if_invalid()

        # Pre-extract numpy arrays for fast bar construction
        self._open = df["open"].to_numpy(dtype=np.float64)
        self._high = df["high"].to_numpy(dtype=np.float64)
        self._low = df["low"].to_numpy(dtype=np.float64)
        self._close = df["close"].to_numpy(dtype=np.float64)
        # Resolve timestamps the same way validate_ohlc does: a DatetimeIndex
        # if present, otherwise a 'timestamp' column. Using df.index blindly
        # would coerce an integer RangeIndex into nonsense nanosecond times.
        if isinstance(df.index, pd.DatetimeIndex):
            ts_index = pd.DatetimeIndex(df.index)
        elif "timestamp" in df.columns:
            ts_index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"]))
        else:
            raise ValueError(
                "df must have a DatetimeIndex or a 'timestamp' column"
            )
        # Pre-box all timestamps to pd.Timestamp once (avoids DatetimeIndex
        # __getitem__ boxing overhead per bar — was 46% of total runtime)
        self._ts = list(ts_index)
        self.n = len(df)

        # Estimate annualization from observations/year (sessions included),
        # unless the caller selects an explicit market convention.
        if freq_per_year is not None and freq_per_year <= 0:
            raise ValueError("freq_per_year must be > 0")
        self.freq_per_year = (
            int(freq_per_year) if freq_per_year is not None
            else infer_freq_per_year(self._ts)
        )
        self._bar_hours = 24.0 * 365.0 / self.freq_per_year if self.freq_per_year > 0 else 1.0

        # Precompute per-bar elapsed hours from actual timestamp deltas.
        # This replaces the constant _bar_hours for funding accrual, so
        # weekend/holiday gaps accrue the correct amount of funding.
        if self.n > 1:
            deltas = np.array([
                (self._ts[i] - self._ts[i - 1]).total_seconds() / 3600.0
                for i in range(1, self.n)
            ], dtype=np.float64)
            self._bar_deltas_hours = np.empty(self.n, dtype=np.float64)
            self._bar_deltas_hours[0] = self._bar_hours
            self._bar_deltas_hours[1:] = deltas
        else:
            self._bar_deltas_hours = np.array([self._bar_hours], dtype=np.float64)

        # Pre-allocate output array
        self.equity_curve = np.empty(self.n, dtype=np.float64)
        self._has_run = False

    def run(self, execution_priority: str = "stop_first",
            liquidate_at_end: bool = False) -> Tuple[np.ndarray, List[Trade]]:
        """Run the backtest over all bars.

        Parameters
        ----------
        execution_priority : str
            How to resolve bars where both stop and TP are hit:
            ``"stop_first"`` (conservative default) or ``"tp_first"``.
        liquidate_at_end : bool
            If True, close remaining positions at the final bar's close and
            include exit costs. Useful for fold-by-fold OOS compounding.

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

        if self._has_run:
            raise RuntimeError(
                "Backtester instances are single-use; create a new instance to rerun"
            )
        self._has_run = True

        # Local references for speed (avoids self.X lookups in tight loop)
        portfolio = self.portfolio
        broker = self.broker
        positions = broker.positions
        symbol = self.symbol
        on_bar = self.strategy.on_bar
        manage_pos = self.strategy.manage_position
        equity_curve = self.equity_curve
        o, h, lo, c, ts = self._open, self._high, self._low, self._close, self._ts
        n = self.n
        exit_first, exit_second = exit_order

        # Cache portfolio fields as locals for the fast path
        cash = portfolio.cash
        peak_equity = portfolio.peak_equity
        max_drawdown = portfolio.max_drawdown

        # Contract multiplier for this symbol: a price move must be scaled to
        # account-currency P&L consistently with broker/portfolio accounting.
        multiplier = portfolio.contract_multiplier(symbol)

        # Track whether margin checks are needed
        has_margin = portfolio.margin_rate > 0

        # Pending trade: signal generated on bar i, executed on bar i+1 at open
        pending_trade = None

        for i in range(n):
            bar = Bar(ts[i], o[i], h[i], lo[i], c[i])
            close_i = c[i]

            # Funding for the elapsed interval belongs to positions carried
            # into this bar. Charge it before stops/exits and before opening
            # any order filled at this bar's open.
            portfolio.accrue_funding(self._bar_deltas_hours[i])

            # Fast path: check if we have any open positions
            open_pos = positions.get(symbol)
            has_positions = open_pos is not None and len(open_pos) > 0

            if has_positions:
                # Check levels that were active before this bar. A stop
                # updated from this bar's close cannot apply to earlier prices.
                exit_first(symbol, bar)
                exit_second(symbol, bar)

                # Management updates take effect on the next bar.
                open_pos = positions.get(symbol)
                has_positions = open_pos is not None and len(open_pos) > 0
                if has_positions:
                    for tr in open_pos:
                        manage_pos(bar, tr)

            # Execute pending trade from previous bar
            if pending_trade is not None:
                limit = getattr(pending_trade, 'limit_price', None)

                if limit is not None:
                    # Limit order: only fill if this bar's range includes the limit price
                    bar_reaches_limit = (
                        (pending_trade.side > 0 and lo[i] <= limit)
                        or (pending_trade.side < 0 and h[i] >= limit)
                    )
                    if bar_reaches_limit:
                        # Price improvement: if the bar opens through the
                        # limit, the fill is at the open (better), not at the
                        # limit price.
                        fill_price = (
                            min(limit, o[i]) if pending_trade.side > 0
                            else max(limit, o[i])
                        )
                        previous_count = len(positions.get(symbol, []))
                        broker.open_trade(
                            symbol=symbol, bar=bar,
                            side=pending_trade.side, size=pending_trade.size,
                            stop=pending_trade.stop_price,
                            take_profit=pending_trade.take_profit,
                            entry_price=fill_price,
                            current_prices={symbol: o[i]},
                        )
                        open_pos = positions.get(symbol)
                        if open_pos is not None and len(open_pos) > previous_count:
                            # A newly filled order is live from this bar's open.
                            # Check only its lots against this bar, so stops or
                            # targets updated for older positions at this bar's
                            # close are not applied retrospectively.
                            entry_lots = list(open_pos)[previous_count:]
                            exit_first(symbol, bar, lots=entry_lots)
                            exit_second(symbol, bar, lots=entry_lots)
                        pending_trade = None
                    # else: limit not reached, keep pending (will be checked next bar
                    # or replaced by a new signal from on_bar below)
                else:
                    # Market order: fill at this bar's open (standard next-bar-open)
                    previous_count = len(positions.get(symbol, []))
                    broker.open_trade(
                        symbol=symbol, bar=bar,
                        side=pending_trade.side, size=pending_trade.size,
                        stop=pending_trade.stop_price,
                        take_profit=pending_trade.take_profit,
                        entry_price=None,
                        current_prices={symbol: o[i]},
                    )
                    open_pos = positions.get(symbol)
                    if open_pos is not None and len(open_pos) > previous_count:
                        # The fill occurs at the open, so protective exits may
                        # trigger later in this same bar.
                        entry_lots = list(open_pos)[previous_count:]
                        exit_first(symbol, bar, lots=entry_lots)
                        exit_second(symbol, bar, lots=entry_lots)
                    pending_trade = None

                open_pos = positions.get(symbol)
                has_positions = open_pos is not None and len(open_pos) > 0

            # Sync cash from portfolio (may have changed from exits/entry)
            cash = portfolio.cash

            if has_positions:
                open_pnl = 0.0
                for tr in open_pos:
                    open_pnl += (close_i - tr.entry_price) * tr.side * tr.size * multiplier
                equity = cash + open_pnl
            else:
                equity = cash

            # Ask strategy for a new signal (pass equity so the strategy
            # can correctly track peak equity and compute drawdown).
            # Signal is stored as pending — executes on next bar's open.
            new_trade = on_bar(i, bar, equity)
            if new_trade is not None and new_trade.size > 0:
                pending_trade = new_trade

            # Update drawdown tracking
            portfolio.update_drawdown(equity)

            # Margin call check (only when positions exist and margin is configured)
            if has_positions and has_margin:
                if portfolio.check_margin_call(equity, {symbol: close_i}, ts[i]):
                    cash = portfolio.cash
                    equity = cash

            # Write equity to pre-allocated array
            equity_curve[i] = equity

        if liquidate_at_end and n > 0:
            final_bar = Bar.at_price(ts[-1], c[-1])
            for lot in list(positions.get(symbol, [])):
                broker.close_trade(symbol, lot, final_bar)
            equity_curve[-1] = portfolio.cash
            portfolio.update_drawdown(portfolio.cash)

        # Expose final state for external access
        self.cash = portfolio.cash
        self.positions = broker.positions.get(symbol, [])
        self.trades = broker.closed_trades
        self.max_drawdown = portfolio.max_drawdown

        return equity_curve, broker.closed_trades
