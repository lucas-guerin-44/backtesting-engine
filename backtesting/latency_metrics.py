"""Latency execution metrics and Sharpe impact comparison.

Collects per-fill statistics from LatencyAwareBroker runs and exposes
summary metrics useful for evaluating execution quality.

Key metrics
-----------
avg_fill_latency_us
    Mean time from strategy submission to actual fill, in microseconds.
    Shows how much clock time the delay queue consumed on average.

p95_fill_latency_us
    95th percentile fill latency. Captures tail events — spikes caused
    by resting limit orders that waited many ticks before price crossed.

fill_rate
    Fraction of total requested qty that actually filled, across all
    orders. 1.0 = everything filled. < 1.0 = some orders partially
    filled or never crossed their limit price.

avg_slippage_bps
    Average price deterioration due to latency, in basis points.
    Measured as (fill_price - price_at_submit) / price_at_submit * 1e4
    for buys (positive = worse). Isolates the cost of the delay itself,
    separate from the bid/ask spread.

sharpe_impact
    Difference in annualized Sharpe between a zero-latency run and a
    latency-configured run of the same strategy on the same ticks.
    Negative = latency hurt the strategy.

Usage
-----
>>> from backtesting.latency_metrics import compare_latency_impact
>>>
>>> from backtesting.latency_models import GaussianLatency
>>>
>>> # Fixed latency
>>> result = compare_latency_impact(
...     ticks=ticks,
...     strategy_factory=lambda: MyStrategy(),
...     ack_latency_ns=50_000_000,
...     starting_cash=10_000,
...     symbol="XAUUSD",
... )
>>> result.print_summary()
>>>
>>> # Stochastic latency model
>>> result = compare_latency_impact(
...     ticks=ticks,
...     strategy_factory=lambda: MyStrategy(),
...     latency_model=GaussianLatency(mean_us=50_000, std_us=10_000),
...     starting_cash=10_000,
...     symbol="XAUUSD",
... )
>>> result.print_summary()
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from backtesting.statistics import compute_sharpe
from backtesting.tick import Tick


@dataclass
class FillRecord:
    """Per-fill record captured during a LatencyAwareBroker run.

    Parameters
    ----------
    order_id : str
    symbol : str
    side : int
        +1 long, -1 short.
    submitted_at : pd.Timestamp
        When the strategy submitted the order.
    filled_at : pd.Timestamp
        When the fill actually executed.
    fill_latency_us : float
        ``(filled_at - submitted_at)`` in microseconds.
    price_at_submit : float
        Market price at submission time (best ask for buys, best bid for
        sells). 0.0 if not captured.
    fill_price : float
        Actual execution price.
    fill_qty : float
        Units filled in this record (may be partial).
    order_qty : float
        Full size of the original order.
    """
    order_id: str
    symbol: str
    side: int
    submitted_at: pd.Timestamp
    filled_at: pd.Timestamp
    fill_latency_us: float
    price_at_submit: float
    fill_price: float
    fill_qty: float
    order_qty: float


class LatencyStats:
    """Aggregated statistics over a collection of FillRecords.

    Parameters
    ----------
    records : list of FillRecord
        Fill records collected by LatencyAwareBroker.
    unfilled_qty : float
        Total qty in orders that never filled (resting at run end).
    equity_curve : array-like, optional
        Equity curve from the same run. Used by ``compare_latency_impact``
        to attach Sharpe without recomputing.
    """

    def __init__(
        self,
        records: List[FillRecord],
        unfilled_qty: float = 0.0,
        equity_curve: Optional[np.ndarray] = None,
    ) -> None:
        self._records = records
        self._unfilled_qty = unfilled_qty
        self.equity_curve = equity_curve

    @property
    def fill_count(self) -> int:
        return len(self._records)

    @property
    def avg_fill_latency_us(self) -> float:
        if not self._records:
            return 0.0
        return float(np.mean([r.fill_latency_us for r in self._records]))

    @property
    def p95_fill_latency_us(self) -> float:
        if not self._records:
            return 0.0
        return float(np.percentile([r.fill_latency_us for r in self._records], 95))

    @property
    def fill_rate(self) -> float:
        """Fraction of requested qty that filled. 1.0 = fully filled."""
        total_requested = sum(r.order_qty for r in self._records) + self._unfilled_qty
        total_filled = sum(r.fill_qty for r in self._records)
        if total_requested <= 0:
            return 0.0
        return total_filled / total_requested

    @property
    def avg_slippage_bps(self) -> float:
        """Average price deterioration due to latency in basis points.

        For buys: (fill_price - price_at_submit) / price_at_submit * 1e4.
        For sells: (price_at_submit - fill_price) / price_at_submit * 1e4.
        Positive = you got a worse price than when you decided to trade.
        Returns 0.0 if no price_at_submit data was captured.
        """
        valid = [r for r in self._records if r.price_at_submit > 0]
        if not valid:
            return 0.0
        slippages = [
            r.side * (r.fill_price - r.price_at_submit) / r.price_at_submit * 1e4
            for r in valid
        ]
        return float(np.mean(slippages))

    @property
    def sharpe(self) -> float:
        if self.equity_curve is None or len(self.equity_curve) < 2:
            return float("nan")
        # Tick-level equity curve: annualise assuming ~252 trading days,
        # each with ~86400 ticks if 1-second data; this is approximate.
        # We keep freq_per_year=252 as a consistent baseline across runs.
        return compute_sharpe(self.equity_curve, annualize=True, freq_per_year=252)

    def summary(self) -> str:
        return textwrap.dedent(f"""
            Fill count          : {self.fill_count}
            Fill rate           : {self.fill_rate * 100:.1f}%
            Avg fill latency    : {self.avg_fill_latency_us:.1f} us
            P95 fill latency    : {self.p95_fill_latency_us:.1f} us
            Avg slippage (lat.) : {self.avg_slippage_bps:+.3f} bps
            Sharpe (annualised) : {self.sharpe:.4f}
        """).strip()


@dataclass
class LatencyImpactResult:
    """Result of compare_latency_impact().

    Attributes
    ----------
    zero_latency : LatencyStats
        Stats from the zero-latency (baseline) run.
    with_latency : LatencyStats
        Stats from the configured-latency run.
    sharpe_impact : float
        ``with_latency.sharpe - zero_latency.sharpe``.
        Negative = latency hurt the strategy.
    """
    zero_latency: LatencyStats
    with_latency: LatencyStats

    @property
    def sharpe_impact(self) -> float:
        return self.with_latency.sharpe - self.zero_latency.sharpe

    def print_summary(self) -> None:
        col = 26
        print()
        print("=" * 56)
        print(f"{'Latency impact analysis':^56}")
        print("=" * 56)
        print(f"{'':>{col}} {'Zero lat':>12} {'With lat':>12}")
        print("-" * 56)
        zl = self.zero_latency
        wl = self.with_latency
        print(f"{'Fill count':<{col}} {zl.fill_count:>12d} {wl.fill_count:>12d}")
        print(f"{'Fill rate':<{col}} {zl.fill_rate * 100:>11.1f}% {wl.fill_rate * 100:>11.1f}%")
        print(f"{'Avg fill latency (us)':<{col}} {zl.avg_fill_latency_us:>12.1f} {wl.avg_fill_latency_us:>12.1f}")
        print(f"{'P95 fill latency (us)':<{col}} {zl.p95_fill_latency_us:>12.1f} {wl.p95_fill_latency_us:>12.1f}")
        print(f"{'Avg slippage bps':<{col}} {zl.avg_slippage_bps:>+12.3f} {wl.avg_slippage_bps:>+12.3f}")
        print(f"{'Sharpe (annualised)':<{col}} {zl.sharpe:>12.4f} {wl.sharpe:>12.4f}")
        print("=" * 56)
        sign = "+" if self.sharpe_impact >= 0 else ""
        print(f"Sharpe impact of latency: {sign}{self.sharpe_impact:.4f}")
        print()


def compare_latency_impact(
    ticks: List[Tick],
    strategy_factory: Callable,
    ack_latency_ns: int = 0,
    fill_latency_ns: int = 0,
    latency_model=None,
    starting_cash: float = 10_000.0,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
    max_leverage: float = 1.0,
    symbol: str = "default",
    timeframe: str = "M1",
    order_book=None,
) -> LatencyImpactResult:
    """Run the same strategy twice — zero latency vs configured latency.

    Returns a ``LatencyImpactResult`` with per-run stats and the Sharpe
    difference. Both runs use fresh strategy and portfolio instances so
    there is no state bleed between runs.

    Parameters
    ----------
    ticks : list of Tick
        Tick data for both runs.
    strategy_factory : callable
        Called with no arguments to produce a fresh Strategy instance.
        Called twice — once per run.
    ack_latency_ns : int
        Acknowledgment delay for the latency run (nanoseconds).
        Ignored when ``latency_model`` is provided.
    fill_latency_ns : int
        Fill delay for the latency run (nanoseconds).
        Ignored when ``latency_model`` is provided.
    latency_model : LatencyModel, optional
        If provided, each order in the latency run samples its delay from
        this model instead of using the fixed ``ack_latency_ns +
        fill_latency_ns`` values. The zero-latency run always uses no delay.
        See ``backtesting.latency_models`` for available models.
    starting_cash : float
    commission_bps : float
    slippage_bps : float
    max_leverage : float
    symbol : str
    timeframe : str
    order_book : MatchingEngine, optional
        Shared engine config. A fresh OrderBook + MatchingEngine pair is
        constructed for the latency run if this is not None.
    """
    # Deferred import to avoid circular imports at module level
    from backtesting.latency_broker import LatencyAwareBroker
    from backtesting.portfolio import Portfolio
    from backtesting.tick_backtest import TickBacktester

    def _run(ack_ns: int, fill_ns: int, model=None) -> LatencyStats:
        portfolio = Portfolio(
            cash=starting_cash,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            max_leverage=max_leverage,
            margin_rate=0.0,
        )
        engine = None
        if order_book is not None:
            from backtesting.order_book import OrderBook, MatchingEngine
            engine = MatchingEngine(OrderBook(), order_book.max_qty_per_level)

        lb = LatencyAwareBroker(
            broker=portfolio.broker,
            ack_latency_ns=ack_ns,
            fill_latency_ns=fill_ns,
            order_book=engine,
            latency_model=model,
        )
        bt = TickBacktester(
            ticks=ticks,
            strategy=strategy_factory(),
            timeframe=timeframe,
            starting_cash=starting_cash,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            max_leverage=max_leverage,
            symbol=symbol,
            latency_broker=lb,
        )
        bt.portfolio = portfolio
        bt.broker = portfolio.broker
        equity_curve, _ = bt.run()

        unfilled = sum(qty for _, qty in lb._resting.values())
        return LatencyStats(
            records=lb.fill_records,
            unfilled_qty=unfilled,
            equity_curve=equity_curve,
        )

    zero = _run(0, 0)
    with_lat = _run(ack_latency_ns, fill_latency_ns, model=latency_model)
    return LatencyImpactResult(zero_latency=zero, with_latency=with_lat)
