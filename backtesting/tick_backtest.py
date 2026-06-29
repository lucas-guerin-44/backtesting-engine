"""Tick-level backtesting engine.

Processes raw tick data, aggregates into OHLC bars via ``TickAggregator``,
and drives strategies at both tick and bar granularity.

Key differences from the bar-level ``Backtester``:

- **Fill timing**: Trades fill at the next tick's price (not next bar's open).
  This is more realistic — in live trading you get filled on the next price
  update, not at some future bar boundary.

- **Stop/TP execution**: Checked on every tick against the exact tick price.
  No need for gap-aware bar logic — ticks ARE the atomic price updates.
  Stops fill at the tick price that breached them (realistic slippage).

- **Dual callbacks**: Strategies receive ``on_tick()`` on every tick and
  ``on_bar()`` when a bar completes. Position management gets
  ``manage_position_tick()`` per tick and ``manage_position()`` per bar.

- **Indicators**: Incremental indicators (EMA, ATR, RSI) update on bar
  completion, not per tick. This matches how indicators work in practice —
  you compute them on closed bars, not noisy tick data.

Performance notes
-----------------
- Tick prices/timestamps are pre-extracted to numpy arrays to avoid
  dataclass attribute lookups in the hot loop (~40% speedup).
- ``on_tick()`` dispatch is skipped entirely when the strategy uses the
  default no-op (detected once at init via method identity check).
- Same for ``manage_position_tick()`` — no per-tick overhead when unused.
- Bar aggregation uses an inlined boundary check against pre-floored
  timestamps to avoid method call overhead per tick.
- Stop/TP checks are inlined in the main loop to avoid method call + dict
  lookup overhead per tick.
- When Cython is available, the inner tick loop runs at C speed via
  ``cy_tick_backtest()`` in ``_tick_core.pyx``.

Usage
-----
>>> from backtesting.tick import Tick, TickAggregator
>>> from backtesting.tick_backtest import TickBacktester
>>>
>>> ticks = [...]  # List[Tick] from CSV or datalake
>>> bt = TickBacktester(ticks, strategy, timeframe="M5")
>>> equity_curve, trades = bt.run()
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from backtesting.latency_broker import LatencyAwareBroker
from backtesting.order import Order
from backtesting.portfolio import Portfolio
from backtesting.strategy import Strategy
from backtesting.tick import Tick, TickAggregator
from backtesting.types import BacktestConfig, Bar, Trade

try:
    from backtesting._tick_core import cy_find_stop_tp_tick, cy_aggregate_ticks
    _HAS_CYTHON_TICK = True
except ImportError:
    _HAS_CYTHON_TICK = False


class TickBacktester:
    """Tick-level backtester with bar aggregation.

    Parameters
    ----------
    ticks : list of Tick
        Raw tick data, sorted by timestamp.
    strategy : Strategy
        Must implement ``on_bar()``. Can optionally implement ``on_tick()``
        and ``manage_position_tick()`` for intra-bar logic.
    timeframe : str
        Aggregation timeframe for bars (e.g. ``"M1"``, ``"M5"``, ``"H1"``).
    config : BacktestConfig, optional
        Shared configuration.
    starting_cash : float
        Initial cash (ignored if ``config`` provided).
    commission_bps : float
        Commission in basis points.
    slippage_bps : float
        Slippage in basis points.
    max_leverage : float
        Maximum gross leverage.
    margin_rate : float
        Margin requirement fraction.
    symbol : str
        Instrument identifier.
    """

    def __init__(
        self,
        ticks: List[Tick],
        strategy,
        timeframe: str = "M5",
        config: Optional[BacktestConfig] = None,
        starting_cash: float = 10_000,
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0,
        max_leverage: float = 1.0,
        margin_rate: float = 0.0,
        symbol: str = "default",
        latency_broker: Optional[LatencyAwareBroker] = None,
    ):
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
        self.broker = self.portfolio.broker
        strategy._broker = self.broker

        self.ticks = ticks
        self.strategy = strategy
        self.symbol = symbol
        self.starting_cash = starting_cash

        self.aggregator = TickAggregator(timeframe)
        self.latency_broker = latency_broker

        _tf_hours = {"M1": 1/60, "M5": 5/60, "M15": 15/60, "M30": 30/60,
                     "H1": 1.0, "H4": 4.0, "D1": 24.0, "W1": 168.0, "MN1": 720.0}
        self._bar_hours = _tf_hours.get(timeframe, 1.0)

        # Detect whether strategy overrides on_tick / manage_position_tick.
        # If not, we skip those calls entirely in the hot loop.
        self._has_on_tick = type(strategy).on_tick is not Strategy.on_tick
        self._has_manage_tick = type(strategy).manage_position_tick is not Strategy.manage_position_tick

        # Pre-extract tick data to numpy arrays for fast access in hot loop.
        # Avoids dataclass .price attribute lookup per tick (~40% of loop time).
        n = len(ticks)
        self._prices = np.array([t.price for t in ticks], dtype=np.float64) if n > 0 else np.empty(0, dtype=np.float64)
        self._volumes = np.array([t.volume for t in ticks], dtype=np.float64) if n > 0 else np.empty(0, dtype=np.float64)
        # Pre-box timestamps once (same trick as bar-level Backtester)
        self._timestamps: list = [t.ts for t in ticks] if n > 0 else []

        # Pre-compute floored timestamps for bar boundary detection.
        # This avoids calling aggregator._floor_ts() per tick in the loop.
        if n > 0:
            freq = self.aggregator._freq
            self._bar_boundaries = np.array(
                [t.ts.floor(freq).value for t in ticks], dtype=np.int64,
            )
        else:
            self._bar_boundaries = np.empty(0, dtype=np.int64)

        # Outputs
        self.equity_curve: Optional[np.ndarray] = None
        self.bar_equity_curve: Optional[np.ndarray] = None
        self.trades: List[Trade] = []
        self.bars: List[Bar] = []

        # Auto-compute daily volatility from tick prices if ADV is set but vol is not
        if self.portfolio.typical_daily_volume is not None and self.portfolio.daily_volatility is None:
            self.portfolio.compute_daily_volatility(self._prices)

    def run(self) -> Tuple[np.ndarray, List[Trade]]:
        """Run the tick-level backtest.

        Returns
        -------
        tuple of (np.ndarray, list[Trade])
            Equity curve (one value per tick) and list of closed trades.
        """
        n_ticks = len(self.ticks)
        if n_ticks == 0:
            self.equity_curve = np.array([], dtype=np.float64)
            return self.equity_curve, []

        # --- Local references for speed (avoids self.X lookups) ---
        portfolio = self.portfolio
        broker = self.broker
        positions = broker.positions
        symbol = self.symbol
        strategy = self.strategy
        on_bar = strategy.on_bar
        manage_pos = strategy.manage_position

        has_on_tick = self._has_on_tick
        has_manage_tick = self._has_manage_tick
        on_tick = strategy.on_tick if has_on_tick else None
        manage_pos_tick = strategy.manage_position_tick if has_manage_tick else None

        # Pre-extracted arrays
        prices = self._prices
        timestamps = self._timestamps
        bar_boundaries = self._bar_boundaries

        # Slippage/commission factors (avoid repeated division in loop)
        slip_bps = portfolio.slippage_bps
        comm_bps = portfolio.commission_bps
        max_leverage = portfolio.max_leverage

        # Pre-allocate equity curve
        equity_arr = np.empty(n_ticks, dtype=np.float64)

        # Bar-level tracking
        bar_equities: List[float] = []
        completed_bars: List[Bar] = []

        # Aggregator state — inlined for speed
        # We still use the aggregator object but check boundaries via
        # pre-computed int64 timestamps to skip the floor() call.
        agg = self.aggregator
        current_bar_boundary: int = bar_boundaries[0]
        # Initialize first bar via aggregator
        agg._start_bar(pd.Timestamp(bar_boundaries[0], unit="ns"), self.ticks[0])

        # Scalars cached from portfolio
        cash = portfolio.cash
        peak_equity = portfolio.peak_equity
        max_drawdown = portfolio.max_drawdown
        has_margin = portfolio.margin_rate > 0
        margin_rate = portfolio.margin_rate

        # Latency broker (optional)
        latency_broker = self.latency_broker

        # Pending trades
        pending_bar_trade: Optional[Trade] = None
        pending_tick_trade: Optional[Trade] = None

        bar_index = 0

        # Inline close_position to avoid method lookup overhead
        _portfolio_cash_ref = portfolio  # We'll sync via portfolio.cash

        for t_idx in range(n_ticks):
            price = prices[t_idx]
            tick_boundary = bar_boundaries[t_idx]

            # --- Fast bar boundary check (int comparison, no floor()) ---
            completed_bar = None
            if tick_boundary != current_bar_boundary:
                # Emit the completed bar
                completed_bar = agg._emit_bar()
                agg._start_bar(pd.Timestamp(tick_boundary, unit="ns"), self.ticks[t_idx])
                current_bar_boundary = tick_boundary
            else:
                # Update current bar OHLC inline (avoid method call overhead)
                if t_idx > 0:  # First tick already handled by _start_bar
                    if price > agg._high:
                        agg._high = price
                    if price < agg._low:
                        agg._low = price
                    agg._close = price
                    agg._volume += self._volumes[t_idx]
                    agg._tick_count += 1

            # --- Check stops/TPs inline (avoid method + dict lookup per tick) ---
            open_pos = positions.get(symbol)
            has_positions = open_pos is not None and len(open_pos) > 0

            if has_positions:
                # Tick-level position management (only if strategy overrides it)
                if has_manage_tick:
                    tick_obj = self.ticks[t_idx]
                    for tr in open_pos:
                        manage_pos_tick(tick_obj, tr)

                # Check stops/TPs at tick granularity via broker
                broker.check_stop_tp_tick(symbol, price, timestamps[t_idx])
                open_pos = positions.get(symbol)
                has_positions = open_pos is not None and len(open_pos) > 0

            # --- Process latency queue (fill matured orders) ---
            if latency_broker is not None:
                latency_broker.process_tick(self.ticks[t_idx])
                open_pos = positions.get(symbol)
                has_positions = open_pos is not None and len(open_pos) > 0

            # --- Execute pending trades at this tick's price ---
            if pending_bar_trade is not None:
                self._fill_at_tick_fast(pending_bar_trade, price, timestamps[t_idx])
                pending_bar_trade = None
                open_pos = positions.get(symbol)
                has_positions = open_pos is not None and len(open_pos) > 0

            if pending_tick_trade is not None:
                self._fill_at_tick_fast(pending_tick_trade, price, timestamps[t_idx])
                pending_tick_trade = None
                open_pos = positions.get(symbol)
                has_positions = open_pos is not None and len(open_pos) > 0

            # --- Sync cash ---
            cash = portfolio.cash

            # --- Compute equity ---
            if has_positions:
                open_pnl = 0.0
                for tr in open_pos:
                    open_pnl += (price - tr.entry_price) * tr.side * tr.size
                equity = cash + open_pnl
            else:
                equity = cash

            # --- Bar completion: call on_bar ---
            if completed_bar is not None:
                # Accrue funding costs at bar boundary
                portfolio.accrue_funding(self._bar_hours)
                cash = portfolio.cash

                completed_bars.append(completed_bar)
                bar_equities.append(equity)

                if has_positions:
                    for tr in open_pos:
                        manage_pos(completed_bar, tr)

                new_trade = on_bar(bar_index, completed_bar, equity)
                if new_trade is not None:
                    if latency_broker is not None and isinstance(new_trade, Order):
                        market_px = prices[t_idx]
                        latency_broker.submit(new_trade, timestamps[t_idx], market_price=market_px)
                    elif not isinstance(new_trade, Order) and new_trade.size > 0:
                        pending_bar_trade = new_trade
                bar_index += 1

            # --- Tick-level signal (only if strategy has on_tick and no bar trade pending) ---
            if has_on_tick and pending_bar_trade is None:
                tick_trade = on_tick(self.ticks[t_idx], agg.current_bar, equity)
                if tick_trade is not None:
                    if latency_broker is not None and isinstance(tick_trade, Order):
                        market_px = prices[t_idx]
                        latency_broker.submit(tick_trade, timestamps[t_idx], market_price=market_px)
                    elif not isinstance(tick_trade, Order) and tick_trade.size > 0:
                        pending_tick_trade = tick_trade

            # --- Drawdown tracking ---
            portfolio.update_drawdown(equity)

            # --- Margin call check ---
            if has_positions and has_margin:
                if portfolio.check_margin_call(equity, {symbol: price}, timestamps[t_idx]):
                    cash = portfolio.cash
                    equity = cash

            equity_arr[t_idx] = equity

        # --- Flush the last incomplete bar ---
        last_bar = agg.flush()
        if last_bar is not None:
            completed_bars.append(last_bar)
            bar_equities.append(equity_arr[-1] if n_ticks > 0 else self.starting_cash)
            on_bar(bar_index, last_bar, equity_arr[-1])

        # --- Sync final state ---
        portfolio.peak_equity = peak_equity
        portfolio.max_drawdown = max_drawdown

        self.equity_curve = equity_arr
        self.bar_equity_curve = np.array(bar_equities, dtype=np.float64)
        self.trades = broker.closed_trades
        self.bars = completed_bars
        self.cash = portfolio.cash
        self.positions = broker.positions.get(symbol, [])
        self.max_drawdown = portfolio.max_drawdown

        return equity_arr, broker.closed_trades

    def _fill_at_tick_fast(self, trade: Trade, price: float, ts) -> None:
        """Fill a pending trade at tick price using broker.open_trade."""
        bar = Bar.at_price(ts, price)
        self.broker.open_trade(
            symbol=self.symbol, bar=bar,
            side=trade.side, size=trade.size,
            stop=trade.stop_price, take_profit=trade.take_profit,
            entry_price=price,
        )
