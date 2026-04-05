"""Multi-asset portfolio backtester.

Runs multiple strategies on multiple assets with shared cash and risk management.
Reuses the existing Broker/Portfolio infrastructure — they already track positions
by symbol (``Dict[str, List[Trade]]``) and compute equity across all assets.

Timestamp alignment:
    Input DataFrames may have different timestamps. The backtester unions all
    indices into a master timeline and forward-fills missing bars. Strategies are
    only called on bars where real market data exists (``is_real_bar``).

Execution ordering:
    1. Process exits for ALL assets (stops, TPs) before any new entries.
    2. Recompute allocation weights at the configured frequency.
    3. Call each strategy with its allocated portion of available cash.
    4. Update portfolio-level equity and drawdown.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtesting.allocation import Allocator, AllocationWeights, EqualWeightAllocator
from backtesting.data import validate_ohlc
from backtesting.portfolio import Portfolio
from backtesting.types import Bar, Trade
from utils import infer_freq_per_year


@dataclass
class RiskLimits:
    """Portfolio-level risk constraints checked before each trade entry.

    Parameters
    ----------
    max_gross_exposure : float
        Maximum gross notional as a fraction of equity (e.g., 1.0 = 100%).
        Gross exposure = sum of |notional| across all positions.
    max_net_exposure : float
        Maximum absolute net notional as a fraction of equity.
        Net exposure = sum of signed notional (long - short).
    max_single_asset : float
        Maximum notional for any single asset as a fraction of equity.
    max_open_positions : int
        Maximum number of assets with open positions simultaneously.
        0 = unlimited.
    """
    max_gross_exposure: float = 1.0
    max_net_exposure: float = 1.0
    max_single_asset: float = 0.30
    max_open_positions: int = 0


@dataclass
class PortfolioBacktestResult:
    """Result from a multi-asset portfolio backtest."""
    equity_curve: np.ndarray
    per_asset_equity: Dict[str, np.ndarray]
    trades: List[Trade]
    per_asset_trades: Dict[str, List[Trade]]
    timestamps: List[pd.Timestamp]
    allocation_history: List[Dict[str, float]]


class PortfolioBacktester:
    """Multi-asset backtester that runs multiple strategies on a shared Portfolio.

    Parameters
    ----------
    dataframes : Dict[str, pd.DataFrame]
        Symbol -> OHLC DataFrame indexed by timestamp.
    strategies : Dict[str, Strategy]
        Symbol -> Strategy instance. Keys must match ``dataframes``.
    allocator : Allocator
        Allocation weight calculator.
    starting_cash : float
        Initial portfolio cash.
    commission_bps : float
        Commission in basis points per trade.
    slippage_bps : float
        Slippage in basis points.
    max_leverage : float
        Maximum gross leverage.
    margin_rate : float
        Margin requirement as fraction of gross notional (0 to disable).
    rebalance_frequency : int
        Recompute allocation weights every N bars. 0 = compute once at start.
    vol_lookback : int
        Lookback window for volatility/correlation estimation in allocators.
    """

    def __init__(
        self,
        dataframes: Dict[str, pd.DataFrame],
        strategies: Dict[str, "Strategy"],
        allocator: Optional[Allocator] = None,
        starting_cash: float = 10_000,
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0,
        max_leverage: float = 1.0,
        margin_rate: float = 0.0,
        rebalance_frequency: int = 0,
        vol_lookback: int = 60,
        risk_limits: Optional[RiskLimits] = None,
    ):
        if set(dataframes.keys()) != set(strategies.keys()):
            raise ValueError("dataframes and strategies must have the same keys")
        if not dataframes:
            raise ValueError("At least one asset is required")

        self.symbols = sorted(dataframes.keys())
        self.strategies = strategies
        self.allocator = allocator or EqualWeightAllocator()
        self.starting_cash = starting_cash
        self.rebalance_frequency = rebalance_frequency
        self.vol_lookback = vol_lookback
        self.risk_limits = risk_limits

        # Shared portfolio and broker
        self.portfolio = Portfolio(
            cash=starting_cash,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            max_leverage=max_leverage,
            margin_rate=margin_rate,
        )
        self.broker = self.portfolio.broker

        # Validate all DataFrames
        for sym in self.symbols:
            report = validate_ohlc(dataframes[sym])
            report.raise_if_invalid()

        # Build aligned master timeline
        self._align_data(dataframes)

        # Infer annualization factor from the aligned timeline
        self.freq_per_year = infer_freq_per_year(self._ts)

    def _align_data(self, dataframes: Dict[str, pd.DataFrame]) -> None:
        """Union all timestamps, forward-fill, and pre-extract numpy arrays."""
        # Build master timeline from union of all indices
        master_idx = dataframes[self.symbols[0]].index
        for sym in self.symbols[1:]:
            master_idx = master_idx.union(dataframes[sym].index)
        master_idx = master_idx.sort_values()
        self.n = len(master_idx)
        self._ts = list(pd.DatetimeIndex(master_idx))

        # Align each DataFrame and pre-extract arrays
        self._open: Dict[str, np.ndarray] = {}
        self._high: Dict[str, np.ndarray] = {}
        self._low: Dict[str, np.ndarray] = {}
        self._close: Dict[str, np.ndarray] = {}
        self._is_real_bar: Dict[str, np.ndarray] = {}
        self._local_idx: Dict[str, np.ndarray] = {}

        for sym in self.symbols:
            df = dataframes[sym]
            # Track which master timestamps have real data for this asset
            real_mask = master_idx.isin(df.index)
            self._is_real_bar[sym] = np.array(real_mask, dtype=bool)

            # Build local index mapping (master_idx -> contiguous strategy index)
            local_idx = np.full(self.n, -1, dtype=np.int64)
            counter = 0
            for i in range(self.n):
                if self._is_real_bar[sym][i]:
                    local_idx[i] = counter
                    counter += 1
            self._local_idx[sym] = local_idx

            # Reindex with forward-fill for mark-to-market on missing bars
            aligned = df.reindex(master_idx, method="ffill")
            # Back-fill any leading NaNs (if master starts before this asset)
            aligned = aligned.bfill()

            self._open[sym] = aligned["open"].to_numpy(dtype=np.float64)
            self._high[sym] = aligned["high"].to_numpy(dtype=np.float64)
            self._low[sym] = aligned["low"].to_numpy(dtype=np.float64)
            self._close[sym] = aligned["close"].to_numpy(dtype=np.float64)

    def run(self, execution_priority: str = "stop_first") -> PortfolioBacktestResult:
        """Run the multi-asset backtest.

        Parameters
        ----------
        execution_priority : str
            ``"stop_first"`` (default) or ``"tp_first"``.

        Returns
        -------
        PortfolioBacktestResult
        """
        broker = self.broker
        portfolio = self.portfolio
        positions = broker.positions
        symbols = self.symbols
        n = self.n
        risk_limits = self.risk_limits

        if execution_priority == "stop_first":
            exit_order = (broker.close_due_to_stop, broker.close_due_to_tp)
        elif execution_priority == "tp_first":
            exit_order = (broker.close_due_to_tp, broker.close_due_to_stop)
        else:
            raise ValueError(f"Unknown execution_priority: {execution_priority}")

        exit_first, exit_second = exit_order

        # Pre-allocate output arrays
        equity_curve = np.empty(n, dtype=np.float64)
        per_asset_equity: Dict[str, np.ndarray] = {
            sym: np.empty(n, dtype=np.float64) for sym in symbols
        }
        allocation_history: List[Dict[str, float]] = []
        per_asset_trades: Dict[str, List[Trade]] = {sym: [] for sym in symbols}
        closed_count = 0  # Track broker.closed_trades length to detect new closures

        # Local refs for speed
        strats = {sym: self.strategies[sym].on_bar for sym in symbols}
        o = self._open
        h = self._high
        lo = self._low
        c = self._close
        ts = self._ts
        is_real = self._is_real_bar
        local_idx = self._local_idx

        cash = portfolio.cash
        peak_equity = portfolio.peak_equity
        max_drawdown = portfolio.max_drawdown
        has_margin = portfolio.margin_rate > 0

        # Initial allocation weights
        close_arrays = self._close
        weights = self.allocator.compute_weights(
            symbols, close_arrays, self.vol_lookback, 0)
        current_weights = weights.weights
        allocation_history.append(dict(current_weights))

        for i in range(n):
            # PHASE 1: Process exits for ALL assets
            for sym in symbols:
                open_pos = positions.get(sym)
                if open_pos and len(open_pos) > 0 and is_real[sym][i]:
                    bar = Bar(ts[i], o[sym][i], h[sym][i], lo[sym][i], c[sym][i])
                    exit_first(sym, bar)
                    exit_second(sym, bar)
                    # Attribute newly closed trades to this symbol
                    new_closed = len(broker.closed_trades) - closed_count
                    for j in range(new_closed):
                        per_asset_trades[sym].append(
                            broker.closed_trades[closed_count + j])
                    closed_count += new_closed

            # Sync cash after all exits
            cash = portfolio.cash

            # PHASE 2: Recompute allocation weights if needed
            rebal = self.rebalance_frequency
            if rebal > 0 and i > 0 and i % rebal == 0:
                weights = self.allocator.compute_weights(
                    symbols, close_arrays, self.vol_lookback, i)
                current_weights = weights.weights
                allocation_history.append(dict(current_weights))

            # PHASE 3: Compute portfolio equity for strategy signals
            current_prices = {sym: c[sym][i] for sym in symbols}
            open_pnl_total = 0.0
            for sym in symbols:
                for tr in positions.get(sym, []):
                    open_pnl_total += (c[sym][i] - tr.entry_price) * tr.side * tr.size
            portfolio_equity = cash + open_pnl_total

            # PHASE 4: Call each strategy on its real bars
            # Pass equity * weight so strategies can correctly track peak
            # equity and drawdown. The Broker's buying power check prevents
            # oversizing beyond what's actually deployable.
            for sym in symbols:
                if not is_real[sym][i]:
                    continue

                bar = Bar(ts[i], o[sym][i], h[sym][i], lo[sym][i], c[sym][i])
                asset_equity = portfolio_equity * current_weights.get(sym, 0.0)

                li = local_idx[sym][i]
                new_trade = strats[sym](li, bar, asset_equity)

                if new_trade is not None and new_trade.size > 0:
                    # Risk limits check
                    if risk_limits is not None and not self._check_risk_limits(
                        risk_limits, sym, new_trade, positions, c, i,
                        symbols, portfolio_equity,
                    ):
                        continue

                    broker.open_trade(
                        symbol=sym, bar=bar,
                        side=new_trade.side, size=new_trade.size,
                        stop=new_trade.stop_price, take_profit=new_trade.take_profit,
                        entry_price=new_trade.entry_price,
                    )
                    cash = portfolio.cash
                    # Recalculate equity for remaining assets
                    open_pnl_total = 0.0
                    for s2 in symbols:
                        for tr in positions.get(s2, []):
                            open_pnl_total += (c[s2][i] - tr.entry_price) * tr.side * tr.size
                    portfolio_equity = cash + open_pnl_total

            # PHASE 5: Compute portfolio equity
            current_prices = {sym: c[sym][i] for sym in symbols}
            open_pnl = 0.0
            for sym in symbols:
                sym_pnl = 0.0
                for tr in positions.get(sym, []):
                    sym_pnl += (c[sym][i] - tr.entry_price) * tr.side * tr.size
                per_asset_equity[sym][i] = sym_pnl
                open_pnl += sym_pnl

            equity = cash + open_pnl
            equity_curve[i] = max(equity, 0.0)

            # Update drawdown tracking
            peak_equity = max(peak_equity, equity)
            if peak_equity > 0:
                dd = (peak_equity - equity) / peak_equity
                if dd > max_drawdown:
                    max_drawdown = dd

            # Margin call check
            if has_margin:
                gross = 0.0
                has_any_pos = False
                for sym in symbols:
                    for tr in positions.get(sym, []):
                        gross += abs(c[sym][i] * tr.size)
                        has_any_pos = True
                if has_any_pos and gross > 0:
                    margin_req = gross * portfolio.margin_rate
                    if equity < 0.5 * margin_req:
                        portfolio.update(ts[i], current_prices)
                        cash = portfolio.cash
                        equity = cash
                        equity_curve[i] = max(equity, 0.0)

        # Sync final state
        portfolio.peak_equity = peak_equity
        portfolio.max_drawdown = max_drawdown

        self.max_drawdown = max_drawdown
        self.cash = cash

        return PortfolioBacktestResult(
            equity_curve=equity_curve,
            per_asset_equity=per_asset_equity,
            trades=broker.closed_trades,
            per_asset_trades=per_asset_trades,
            timestamps=ts,
            allocation_history=allocation_history,
        )

    @staticmethod
    def _check_risk_limits(
        limits: RiskLimits,
        symbol: str,
        trade: "Trade",
        positions: Dict[str, List["Trade"]],
        close_arrays: Dict[str, np.ndarray],
        bar_idx: int,
        symbols: List[str],
        equity: float,
    ) -> bool:
        """Check if a proposed trade would violate portfolio risk limits.

        Returns True if the trade is allowed, False if it should be skipped.
        """
        if equity <= 0:
            return False

        new_notional = abs(trade.entry_price * trade.size)

        # Current exposure across all assets
        gross_exposure = 0.0
        net_exposure = 0.0
        asset_notional: Dict[str, float] = {}
        n_assets_with_positions = 0

        for sym in symbols:
            sym_notional = 0.0
            sym_signed = 0.0
            for tr in positions.get(sym, []):
                notional = abs(close_arrays[sym][bar_idx] * tr.size)
                sym_notional += notional
                sym_signed += close_arrays[sym][bar_idx] * tr.size * tr.side
            asset_notional[sym] = sym_notional
            gross_exposure += sym_notional
            net_exposure += sym_signed
            if sym_notional > 0:
                n_assets_with_positions += 1

        # Add the proposed trade's contribution
        projected_gross = gross_exposure + new_notional
        projected_net = net_exposure + trade.entry_price * trade.size * trade.side
        projected_asset = asset_notional.get(symbol, 0.0) + new_notional

        # Check: max gross exposure
        if projected_gross / equity > limits.max_gross_exposure:
            return False

        # Check: max net exposure
        if abs(projected_net) / equity > limits.max_net_exposure:
            return False

        # Check: max single asset concentration
        if projected_asset / equity > limits.max_single_asset:
            return False

        # Check: max open positions (count of assets with positions)
        if limits.max_open_positions > 0:
            will_open_new = asset_notional.get(symbol, 0.0) == 0.0
            if will_open_new and n_assets_with_positions >= limits.max_open_positions:
                return False

        return True
