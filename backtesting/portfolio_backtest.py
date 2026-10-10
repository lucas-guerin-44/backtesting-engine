"""Shared-account, multi-instrument bar backtester."""

from heapq import merge
from itertools import groupby
from dataclasses import dataclass
import math
from typing import Dict, Mapping, Optional, Tuple, List
import warnings

import numpy as np
import pandas as pd

from backtesting.data import validate_ohlc
from backtesting.instruments import InstrumentSpec
from backtesting.portfolio import Portfolio
from backtesting.types import BacktestConfig, Bar, Trade


def _symbol_events(symbol: str, timestamps_ns):
    return ((int(value), symbol, i) for i, value in enumerate(timestamps_ns))


@dataclass(frozen=True)
class PortfolioRiskLimits:
    """Optional gross stop-risk and per-symbol leverage caps.

    ``max_total_risk_fraction`` caps aggregate open loss to protective stops as
    a fraction of current portfolio equity. Open risk is gross-summed (no
    offsetting across symbols). ``max_symbol_gross_leverage`` caps one symbol's
    marked notional as a multiple of portfolio equity. Simultaneous eligible
    entries are scaled pro rata to fit these limits and shared buying power.
    """

    max_total_risk_fraction: Optional[float] = None
    max_symbol_gross_leverage: Optional[float] = None

    def __post_init__(self):
        if self.max_total_risk_fraction is not None and (
            not math.isfinite(self.max_total_risk_fraction)
            or self.max_total_risk_fraction < 0
        ):
            raise ValueError("max_total_risk_fraction must be finite and >= 0")
        if self.max_symbol_gross_leverage is not None and (
            not math.isfinite(self.max_symbol_gross_leverage)
            or self.max_symbol_gross_leverage < 0
        ):
            raise ValueError("max_symbol_gross_leverage must be finite and >= 0")


class PortfolioBacktester:
    """Run per-instrument strategies against one shared portfolio.

    Bars are processed in timestamp order. For a timestamp shared by multiple
    instruments, existing exits run first, then eligible simultaneous entries
    are scaled pro rata to fit shared buying power and configured risk limits,
    then close marks and strategy callbacks. Instruments without a bar retain
    their last close
    for portfolio marking.

    Instrument P&L and notional are ``price change * quantity * contract
    multiplier``. This first version requires instrument quote currencies to
    match ``account_currency``; it does not perform FX conversion.
    """

    def __init__(
        self,
        data: Mapping[str, pd.DataFrame],
        strategies: Mapping[str, object],
        config: Optional[BacktestConfig] = None,
        starting_cash: float = 10_000.0,
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0,
        spread_bps: float = 0.0,
        max_leverage: float = 1.0,
        margin_rate: float = 0.0,
        instrument_specs: Optional[Mapping[str, InstrumentSpec]] = None,
        account_currency: str = "USD",
        risk_limits: Optional[PortfolioRiskLimits] = None,
        stale_mark_policy: str = "warn",
    ) -> None:
        if not data:
            raise ValueError("data must contain at least one instrument")
        if set(data) != set(strategies):
            raise ValueError("data and strategies must have exactly the same symbols")

        self.symbols = tuple(sorted(data))
        self.account_currency = account_currency.strip().upper()
        if not self.account_currency:
            raise ValueError("account_currency must be non-empty")

        if instrument_specs is None:
            raise ValueError("instrument_specs must explicitly define every symbol")
        supplied_specs = dict(instrument_specs)
        unknown_specs = set(supplied_specs) - set(self.symbols)
        if unknown_specs:
            raise ValueError(f"instrument_specs contains unknown symbols: {sorted(unknown_specs)}")
        missing_specs = set(self.symbols) - set(supplied_specs)
        if missing_specs:
            raise ValueError(f"instrument_specs missing symbols: {sorted(missing_specs)}")
        self.instrument_specs = supplied_specs
        self.risk_limits = risk_limits or PortfolioRiskLimits()
        if stale_mark_policy not in ("allow", "warn", "error"):
            raise ValueError("stale_mark_policy must be 'allow', 'warn', or 'error'")
        self.stale_mark_policy = stale_mark_policy
        for symbol, spec in self.instrument_specs.items():
            if not isinstance(spec, InstrumentSpec):
                raise TypeError(f"instrument_specs[{symbol!r}] must be an InstrumentSpec")
            if spec.quote_currency != self.account_currency:
                raise ValueError(
                    f"{symbol} quote currency {spec.quote_currency} differs from account "
                    f"currency {self.account_currency}; FX conversion is not supported"
                )

        self.data: Dict[str, pd.DataFrame] = {}
        aware = set()
        for symbol in self.symbols:
            frame = data[symbol]
            if not isinstance(frame.index, pd.DatetimeIndex):
                raise ValueError(f"{symbol} data must have a DatetimeIndex")
            report = validate_ohlc(frame)
            report.raise_if_invalid()
            normalized = frame
            aware.add(frame.index.tz is not None)
            self.data[symbol] = normalized
        if len(aware) > 1:
            raise ValueError("all instrument timestamps must be either timezone-aware or naive")
        if aware == {True}:
            self.data = {
                symbol: frame.tz_convert("UTC")
                for symbol, frame in self.data.items()
            }

        self.strategies = dict(strategies)
        if config is not None:
            self.portfolio = Portfolio(**config.to_kwargs())
        else:
            self.portfolio = Portfolio(
                cash=starting_cash,
                commission_bps=commission_bps,
                slippage_bps=slippage_bps,
                spread_bps=spread_bps,
                max_leverage=max_leverage,
                margin_rate=margin_rate,
            )
        self.portfolio.instrument_specs = dict(self.instrument_specs)
        self.broker = self.portfolio.broker
        for symbol, strategy in self.strategies.items():
            strategy._broker = self.broker
            strategy._contract_multiplier = self.instrument_specs[symbol].contract_multiplier

        # Pre-extract arrays; the event merge visits O(total bars * log symbols),
        # not every symbol at every timestamp in the union calendar.
        self._arrays = {}
        self._timestamps = {}
        self._event_iters = []
        for symbol in self.symbols:
            frame = self.data[symbol]
            idx = pd.DatetimeIndex(frame.index)
            self._timestamps[symbol] = list(idx)
            self._arrays[symbol] = {
                "open": frame["open"].to_numpy(dtype=np.float64),
                "high": frame["high"].to_numpy(dtype=np.float64),
                "low": frame["low"].to_numpy(dtype=np.float64),
                "close": frame["close"].to_numpy(dtype=np.float64),
                "volume": (
                    frame["volume"].to_numpy(dtype=np.float64)
                    if "volume" in frame else None
                ),
            }
            ns = idx.asi8
            self._event_iters.append(_symbol_events(symbol, ns))

        self._latest_prices: Dict[str, float] = {}
        self._latest_bars: Dict[str, Bar] = {}
        self._bar_indices = {symbol: 0 for symbol in self.symbols}
        self._pending: Dict[str, Optional[Trade]] = {symbol: None for symbol in self.symbols}
        self._pending_symbols = set()
        self.stale_mark_events = []
        self.max_mark_age_hours = {symbol: 0.0 for symbol in self.symbols}
        self._stale_reported = set()
        self.equity_curve = np.array([], dtype=np.float64)
        self.equity_index = pd.DatetimeIndex([])
        self.trades: List[Trade] = []
        self.positions = self.broker.positions
        self.cash = self.portfolio.cash
        self.max_drawdown = 0.0
        self._has_run = False

    @staticmethod
    def _bar(symbol: str, i: int, ts, arrays) -> Bar:
        volume = arrays["volume"]
        return Bar(
            ts=ts,
            open=arrays["open"][i],
            high=arrays["high"][i],
            low=arrays["low"][i],
            close=arrays["close"][i],
            volume=None if volume is None else volume[i],
        )

    @staticmethod
    def _limit_reached(trade: Trade, bar: Bar) -> bool:
        if trade.limit_price is None:
            return True
        if trade.side > 0:
            return bar.low <= trade.limit_price
        return bar.high >= trade.limit_price

    def _open_stop_risk(self) -> float:
        risk = 0.0
        for symbol, position in self.broker.positions.items():
            mark = self._latest_prices.get(symbol)
            if mark is None:
                continue
            multiplier = self.instrument_specs[symbol].contract_multiplier
            for lot in position:
                if lot.stop_price is None:
                    return float("inf")
                distance = (
                    max(0.0, mark - lot.stop_price)
                    if lot.side > 0 else max(0.0, lot.stop_price - mark)
                )
                risk += distance * lot.size * multiplier
        return risk

    def _check_stale_marks(self, ts) -> None:
        for symbol, position in self.broker.positions.items():
            if not position:
                continue
            last_bar = self._latest_bars.get(symbol)
            if last_bar is None:
                continue
            age_hours = max(0.0, (ts - last_bar.ts).total_seconds() / 3600.0)
            self.max_mark_age_hours[symbol] = max(
                self.max_mark_age_hours[symbol], age_hours,
            )
            threshold = self.instrument_specs[symbol].max_stale_hours
            if threshold is None or age_hours <= threshold or symbol in self._stale_reported:
                continue
            event = {
                "symbol": symbol,
                "timestamp": ts,
                "last_bar_timestamp": last_bar.ts,
                "age_hours": age_hours,
                "threshold_hours": threshold,
            }
            self.stale_mark_events.append(event)
            self._stale_reported.add(symbol)
            message = (
                f"stale mark for {symbol}: last bar {last_bar.ts}, "
                f"age {age_hours:.2f}h exceeds {threshold:.2f}h"
            )
            if self.stale_mark_policy == "error":
                raise RuntimeError(message)
            if self.stale_mark_policy == "warn":
                warnings.warn(message, RuntimeWarning, stacklevel=2)

    def _allocate_pending(self, bars: Mapping[str, Bar], symbols):
        """Return eligible pending orders sized pro rata under shared limits."""
        candidates = []
        for symbol in sorted(symbols):
            trade = self._pending[symbol]
            bar = bars[symbol]
            if trade is None or not self._limit_reached(trade, bar):
                continue
            if trade.limit_price is not None:
                # Price improvement: if the bar opens through the limit, the
                # fill is at the (better) open, matching the single-asset engine.
                fill_reference = (
                    min(trade.limit_price, bar.open) if trade.side > 0
                    else max(trade.limit_price, bar.open)
                )
            else:
                fill_reference = bar.open
            multiplier = self.instrument_specs[symbol].contract_multiplier
            size = max(0.0, float(trade.size))
            notional = abs(fill_reference * size * multiplier)
            if self.risk_limits.max_total_risk_fraction is not None:
                if trade.stop_price is None:
                    raise ValueError(
                        f"Risk-limited order for {symbol} must define stop_price"
                    )
                stop_risk = abs(fill_reference - trade.stop_price) * size * multiplier
            else:
                stop_risk = 0.0
            candidates.append((symbol, trade, bar, fill_reference, size, notional, stop_risk))

        if not candidates:
            return []

        equity = self.portfolio.compute_equity(self._latest_prices)
        gross = self.portfolio.gross_notional(self._latest_prices)
        buying_power = max(0.0, equity * self.portfolio.max_leverage - gross)
        requested_notional = sum(item[5] for item in candidates)
        shared_scale = (
            min(1.0, buying_power / requested_notional)
            if requested_notional > 0 else 1.0
        )

        limits = self.risk_limits
        if limits.max_total_risk_fraction is not None:
            open_risk = self._open_stop_risk()
            risk_budget = max(0.0, equity * limits.max_total_risk_fraction - open_risk)
            requested_risk = sum(item[6] for item in candidates)
            if requested_risk > 0:
                shared_scale = min(shared_scale, risk_budget / requested_risk)

        symbol_scales = {}
        if limits.max_symbol_gross_leverage is not None:
            symbol_budget = max(0.0, equity * limits.max_symbol_gross_leverage)
            for symbol, _, _, _, _, requested, _ in candidates:
                mark = self._latest_prices.get(symbol)
                current = self.portfolio.gross_notional({symbol: mark}) if mark is not None else 0.0
                remaining = max(0.0, symbol_budget - current)
                symbol_scales[symbol] = (
                    min(1.0, remaining / requested) if requested > 0 else 1.0
                )

        allocated = []
        for symbol, trade, bar, fill_reference, size, _, _ in candidates:
            scale = min(shared_scale, symbol_scales.get(symbol, 1.0))
            self._pending[symbol] = None
            self._pending_symbols.discard(symbol)
            if size * scale > 0:
                allocated.append((symbol, trade, bar, size * scale, fill_reference))
        return allocated

    def run(
        self,
        execution_priority: str = "stop_first",
        liquidate_at_end: bool = False,
    ) -> Tuple[np.ndarray, List[Trade]]:
        """Run the shared-account simulation and return (equity, closed trades)."""
        if self._has_run:
            raise RuntimeError("PortfolioBacktester instances are single-use; create a new instance to rerun")
        if execution_priority == "stop_first":
            exit_methods = (self.broker.close_due_to_stop, self.broker.close_due_to_tp)
        elif execution_priority == "tp_first":
            exit_methods = (self.broker.close_due_to_tp, self.broker.close_due_to_stop)
        else:
            raise ValueError(f"Unknown execution_priority: {execution_priority}")
        self._has_run = True

        equity_values = []
        equity_timestamps = []
        previous_ts = None
        merged = merge(*self._event_iters, key=lambda event: event[0])

        for _ts_ns, same_time in groupby(merged, key=lambda event: event[0]):
            events = sorted(
                ((symbol, i) for _, symbol, i in same_time),
                key=lambda event: event[0],
            )
            ts = self._timestamps[events[0][0]][events[0][1]]

            if previous_ts is not None:
                elapsed_hours = (ts - previous_ts).total_seconds() / 3600.0
                if elapsed_hours > 0:
                    self.portfolio.accrue_funding(elapsed_hours)

            bars = {
                symbol: self._bar(symbol, i, self._timestamps[symbol][i], self._arrays[symbol])
                for symbol, i in events
            }
            # Open marks are used for gap-aware exits and shared buying-power
            # checks before the close marks arrive.
            for symbol, bar in bars.items():
                self._latest_prices[symbol] = bar.open
            self.portfolio.last_prices = dict(self._latest_prices)

            # Exits on existing lots, then close-based position management.
            for symbol, bar in bars.items():
                position = self.broker.positions.get(symbol)
                if position:
                    exit_methods[0](symbol, bar)
                    exit_methods[1](symbol, bar)
                    for lot in self.broker.positions.get(symbol, []):
                        self.strategies[symbol].manage_position(bar, lot)

            # Gather simultaneous eligible orders and scale them together so
            # alphabetic symbol ordering cannot consume the entire risk/cash budget.
            pending_symbols = self._pending_symbols.intersection(bars)
            for symbol, trade, bar, size, fill_reference in self._allocate_pending(bars, pending_symbols):
                previous_count = len(self.broker.positions.get(symbol, []))
                self.broker.open_trade(
                    symbol=symbol, bar=bar, side=trade.side, size=size,
                    stop=trade.stop_price, take_profit=trade.take_profit,
                    entry_price=fill_reference,
                    current_prices=self._latest_prices,
                )
                position = self.broker.positions.get(symbol)
                if position is not None and len(position) > previous_count:
                    new_lots = list(position)[previous_count:]
                    exit_methods[0](symbol, bar, lots=new_lots)
                    exit_methods[1](symbol, bar, lots=new_lots)

            # Mark all instruments at their latest close; symbols without a
            # bar at this event retain their last observed close.
            for symbol, bar in bars.items():
                self._latest_prices[symbol] = bar.close
                self._latest_bars[symbol] = bar
            self.portfolio.last_prices = dict(self._latest_prices)
            self._check_stale_marks(ts)

            equity = self.portfolio.compute_equity(self._latest_prices)
            self.portfolio.min_equity = min(self.portfolio.min_equity, equity)
            self.portfolio.update_drawdown(equity)
            if self.portfolio.margin_rate > 0:
                gross = self.portfolio.gross_notional(self._latest_prices)
                margin_req = gross * self.portfolio.margin_rate
                if gross > 0 and equity < self.portfolio.margin_call_threshold * margin_req:
                    for symbol, position in list(self.broker.positions.items()):
                        mark = self._latest_prices.get(symbol)
                        if mark is None:
                            continue
                        forced_bar = Bar.at_price(ts, mark)
                        for lot in list(position):
                            self.broker.close_trade(symbol, lot, forced_bar)
                    equity = self.portfolio.compute_equity(self._latest_prices)
                    self.portfolio.min_equity = min(self.portfolio.min_equity, equity)
                    self.portfolio.update_drawdown(equity)

            for symbol, bar in bars.items():
                signal = self.strategies[symbol].on_bar(
                    self._bar_indices[symbol], bar, equity,
                )
                if signal is not None and signal.size > 0:
                    self._pending[symbol] = signal
                    self._pending_symbols.add(symbol)
                self._bar_indices[symbol] += 1

            equity_values.append(equity)
            equity_timestamps.append(ts)
            self.portfolio.equity_curve.append((ts, equity))
            previous_ts = ts

        if liquidate_at_end and self._latest_bars:
            for symbol in self.symbols:
                bar = self._latest_bars.get(symbol)
                if bar is None:
                    continue
                for lot in list(self.broker.positions.get(symbol, [])):
                    self.broker.close_trade(symbol, lot, bar)
            if equity_values:
                equity_values[-1] = self.portfolio.cash
                self.portfolio.equity_curve[-1] = (
                    equity_timestamps[-1], self.portfolio.cash,
                )
                self.portfolio.update_drawdown(self.portfolio.cash)

        self.equity_curve = np.asarray(equity_values, dtype=np.float64)
        self.equity_index = pd.DatetimeIndex(equity_timestamps)
        self.trades = self.broker.closed_trades
        self.cash = self.portfolio.cash
        self.positions = self.broker.positions
        self.max_drawdown = self.portfolio.max_drawdown
        return self.equity_curve, self.trades
