"""Tests for latency metrics collection and LatencyImpactResult."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from backtesting.latency_metrics import (
    FillRecord,
    LatencyImpactResult,
    LatencyStats,
    compare_latency_impact,
)
from backtesting.order import Order, OrderType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(ns: int) -> pd.Timestamp:
    return pd.Timestamp(ns, unit="ns")


def _make_record(
    *,
    fill_latency_us: float = 50_000.0,
    price_at_submit: float = 100.0,
    fill_price: float = 100.5,
    fill_qty: float = 1.0,
    order_qty: float = 1.0,
    side: int = 1,
) -> FillRecord:
    return FillRecord(
        order_id="abc",
        symbol="TEST",
        side=side,
        submitted_at=_ts(0),
        filled_at=_ts(int(fill_latency_us * 1_000)),
        fill_latency_us=fill_latency_us,
        price_at_submit=price_at_submit,
        fill_price=fill_price,
        fill_qty=fill_qty,
        order_qty=order_qty,
    )


# ---------------------------------------------------------------------------
# FillRecord
# ---------------------------------------------------------------------------

def test_fill_record_fields():
    rec = _make_record(fill_latency_us=25_000.0, fill_price=101.0)
    assert rec.fill_latency_us == 25_000.0
    assert rec.fill_price == 101.0
    assert rec.side == 1


# ---------------------------------------------------------------------------
# LatencyStats — empty
# ---------------------------------------------------------------------------

def test_empty_stats_defaults():
    stats = LatencyStats(records=[])
    assert stats.fill_count == 0
    assert stats.avg_fill_latency_us == 0.0
    assert stats.p95_fill_latency_us == 0.0
    assert stats.fill_rate == 0.0
    assert stats.avg_slippage_bps == 0.0
    assert math.isnan(stats.sharpe)


# ---------------------------------------------------------------------------
# LatencyStats — avg_fill_latency_us / p95
# ---------------------------------------------------------------------------

def test_avg_fill_latency():
    records = [_make_record(fill_latency_us=x) for x in [10_000, 20_000, 30_000]]
    stats = LatencyStats(records=records)
    assert stats.avg_fill_latency_us == pytest.approx(20_000.0)


def test_p95_fill_latency_single_record():
    rec = _make_record(fill_latency_us=50_000.0)
    stats = LatencyStats(records=[rec])
    assert stats.p95_fill_latency_us == pytest.approx(50_000.0)


def test_p95_fill_latency_distribution():
    # 100 records: 95th percentile should be near the top
    records = [_make_record(fill_latency_us=float(i)) for i in range(1, 101)]
    stats = LatencyStats(records=records)
    assert stats.p95_fill_latency_us >= 94.0


# ---------------------------------------------------------------------------
# LatencyStats — fill_rate
# ---------------------------------------------------------------------------

def test_fill_rate_fully_filled():
    records = [_make_record(fill_qty=1.0, order_qty=1.0)]
    stats = LatencyStats(records=records)
    assert stats.fill_rate == pytest.approx(1.0)


def test_fill_rate_partial_fill():
    # fill_qty=0.5 of a 1.0 order; no separate unfilled_qty needed here
    records = [_make_record(fill_qty=0.5, order_qty=1.0)]
    stats = LatencyStats(records=records)
    assert stats.fill_rate == pytest.approx(0.5)


def test_fill_rate_with_unfilled_qty():
    # 2 units filled, 1 unit never filled (resting at run end)
    records = [_make_record(fill_qty=2.0, order_qty=2.0)]
    stats = LatencyStats(records=records, unfilled_qty=1.0)
    # total_requested = 2 + 1 = 3, total_filled = 2
    assert stats.fill_rate == pytest.approx(2.0 / 3.0)


# ---------------------------------------------------------------------------
# LatencyStats — avg_slippage_bps
# ---------------------------------------------------------------------------

def test_slippage_buy_positive_when_price_moved_up():
    # Bought at 100 at submit, filled at 100.5 (price moved against us)
    rec = _make_record(side=1, price_at_submit=100.0, fill_price=100.5)
    stats = LatencyStats(records=[rec])
    # slippage = (100.5 - 100.0) / 100.0 * 1e4 = 50 bps
    assert stats.avg_slippage_bps == pytest.approx(50.0)


def test_slippage_sell_positive_when_price_moved_against():
    # Sold at 100 at submit, filled at 99.5 (price moved against us)
    rec = _make_record(side=-1, price_at_submit=100.0, fill_price=99.5)
    stats = LatencyStats(records=[rec])
    # slippage = -1 * (99.5 - 100.0) / 100.0 * 1e4 = 50 bps
    assert stats.avg_slippage_bps == pytest.approx(50.0)


def test_slippage_zero_when_no_price_at_submit():
    rec = _make_record(price_at_submit=0.0)
    stats = LatencyStats(records=[rec])
    assert stats.avg_slippage_bps == 0.0


def test_slippage_averaged_across_fills():
    r1 = _make_record(side=1, price_at_submit=100.0, fill_price=100.2)
    r2 = _make_record(side=1, price_at_submit=100.0, fill_price=100.6)
    stats = LatencyStats(records=[r1, r2])
    # r1: 20 bps, r2: 60 bps -> avg = 40 bps
    assert stats.avg_slippage_bps == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# LatencyStats — sharpe
# ---------------------------------------------------------------------------

def test_sharpe_nan_without_equity_curve():
    stats = LatencyStats(records=[])
    assert math.isnan(stats.sharpe)


def test_sharpe_nan_with_single_point():
    import numpy as np
    stats = LatencyStats(records=[], equity_curve=np.array([10_000.0]))
    assert math.isnan(stats.sharpe)


def test_sharpe_finite_with_curve():
    import numpy as np
    # Steadily rising equity => positive Sharpe
    curve = np.linspace(10_000, 12_000, 300)
    stats = LatencyStats(records=[], equity_curve=curve)
    assert math.isfinite(stats.sharpe)
    assert stats.sharpe > 0


# ---------------------------------------------------------------------------
# LatencyStats — summary()
# ---------------------------------------------------------------------------

def test_summary_string_contains_key_fields():
    records = [_make_record()]
    stats = LatencyStats(records=records)
    s = stats.summary()
    assert "Fill count" in s
    assert "Fill rate" in s
    assert "Avg fill latency" in s
    assert "slippage" in s.lower()


# ---------------------------------------------------------------------------
# LatencyImpactResult
# ---------------------------------------------------------------------------

def test_sharpe_impact_negative_when_latency_hurts():
    import numpy as np
    rng = np.random.default_rng(42)
    # good run: high mean returns, modest vol -> high Sharpe
    good_ret = rng.normal(0.01, 0.002, 300)
    curve_good = np.cumprod(1 + good_ret) * 10_000
    # bad run: near-zero mean, same vol -> low Sharpe
    bad_ret = rng.normal(0.0001, 0.002, 300)
    curve_bad = np.cumprod(1 + bad_ret) * 10_000

    zero = LatencyStats(records=[], equity_curve=curve_good)
    with_lat = LatencyStats(records=[], equity_curve=curve_bad)
    result = LatencyImpactResult(zero_latency=zero, with_latency=with_lat)
    assert result.sharpe_impact < 0


def test_sharpe_impact_zero_when_identical():
    import numpy as np
    curve = np.linspace(10_000, 11_000, 300)
    zero = LatencyStats(records=[], equity_curve=curve)
    with_lat = LatencyStats(records=[], equity_curve=curve)
    result = LatencyImpactResult(zero_latency=zero, with_latency=with_lat)
    assert result.sharpe_impact == pytest.approx(0.0)


def test_print_summary_runs_without_error(capsys):
    import numpy as np
    curve = np.linspace(10_000, 11_000, 300)
    zero = LatencyStats(records=[], equity_curve=curve)
    with_lat = LatencyStats(records=[], equity_curve=curve.copy())
    result = LatencyImpactResult(zero_latency=zero, with_latency=with_lat)
    result.print_summary()
    out = capsys.readouterr().out
    assert "Latency impact analysis" in out
    assert "Sharpe impact" in out


# ---------------------------------------------------------------------------
# LatencyAwareBroker fill_records integration
# ---------------------------------------------------------------------------

def _make_ticks(n: int = 20, start_price: float = 100.0):
    """Produce a list of Tick objects with incrementing prices."""
    from backtesting.tick import Tick
    base_ns = pd.Timestamp("2024-01-01").value
    ticks = []
    for i in range(n):
        ticks.append(Tick(
            ts=pd.Timestamp(base_ns + i * 1_000_000_000, unit="ns"),
            price=start_price + i * 0.1,
            bid=start_price + i * 0.1 - 0.05,
            ask=start_price + i * 0.1 + 0.05,
            volume=100.0,
        ))
    return ticks


def test_fill_records_populated_on_market_order():
    from backtesting.latency_broker import LatencyAwareBroker
    from backtesting.portfolio import Portfolio
    from backtesting.order import Order, OrderType

    ticks = _make_ticks(n=20)
    portfolio = Portfolio(cash=10_000)
    lb = LatencyAwareBroker(
        broker=portfolio.broker,
        ack_latency_ns=0,
    )

    # Submit at tick 0 with zero latency -> should fill on first process_tick call
    order = Order(
        type=OrderType.MARKET,
        symbol="TEST",
        side=1,
        qty=1.0,
        protective_stop=90.0,
        take_profit=None,
        submitted_at=ticks[0].ts,
    )
    lb.submit(order, ticks[0].ts, market_price=ticks[0].ask)

    lb.process_tick(ticks[0])

    assert len(lb.fill_records) == 1
    rec = lb.fill_records[0]
    assert rec.order_id == order.order_id
    assert rec.symbol == "TEST"
    assert rec.fill_qty == pytest.approx(1.0)
    assert rec.price_at_submit == pytest.approx(ticks[0].ask)
    assert rec.fill_latency_us == pytest.approx(0.0)


def test_fill_records_latency_measured_correctly():
    from backtesting.latency_broker import LatencyAwareBroker
    from backtesting.portfolio import Portfolio
    from backtesting.order import Order, OrderType

    ticks = _make_ticks(n=20)
    # 5 seconds (in ns) ack latency; ticks are 1s apart
    ack_ns = 5 * 1_000_000_000
    portfolio = Portfolio(cash=10_000)
    lb = LatencyAwareBroker(broker=portfolio.broker, ack_latency_ns=ack_ns)

    order = Order(
        type=OrderType.MARKET,
        symbol="TEST",
        side=1,
        qty=1.0,
        protective_stop=90.0,
        take_profit=None,
        submitted_at=ticks[0].ts,
    )
    lb.submit(order, ticks[0].ts, market_price=100.0)

    for tick in ticks:
        lb.process_tick(tick)

    assert len(lb.fill_records) == 1
    rec = lb.fill_records[0]
    # fill should have happened at or after tick[5] (5s after submit)
    expected_min_latency_us = 5 * 1_000_000.0
    assert rec.fill_latency_us >= expected_min_latency_us


def test_fill_records_market_price_captured():
    from backtesting.latency_broker import LatencyAwareBroker
    from backtesting.portfolio import Portfolio
    from backtesting.order import Order, OrderType

    ticks = _make_ticks(n=5)
    portfolio = Portfolio(cash=10_000)
    lb = LatencyAwareBroker(broker=portfolio.broker, ack_latency_ns=0)

    order = Order(
        type=OrderType.MARKET,
        symbol="TEST",
        side=1,
        qty=1.0,
        protective_stop=90.0,
        take_profit=None,
        submitted_at=ticks[0].ts,
    )
    market_px = 100.05
    lb.submit(order, ticks[0].ts, market_price=market_px)
    lb.process_tick(ticks[0])

    assert lb.fill_records[0].price_at_submit == pytest.approx(market_px)


def test_fill_records_empty_when_no_orders():
    from backtesting.latency_broker import LatencyAwareBroker
    from backtesting.portfolio import Portfolio

    ticks = _make_ticks(n=5)
    portfolio = Portfolio(cash=10_000)
    lb = LatencyAwareBroker(broker=portfolio.broker)

    for tick in ticks:
        lb.process_tick(tick)

    assert lb.fill_records == []


# ---------------------------------------------------------------------------
# compare_latency_impact
# ---------------------------------------------------------------------------

def _build_strategy_factory():
    """Factory that returns a simple buy-and-hold strategy using Order."""
    from backtesting.strategy import Strategy
    from backtesting.order import Order, OrderType
    from backtesting.types import Bar
    import pandas as pd

    class _Simple(Strategy):
        _bought = False

        def on_bar(self, bar_index, bar: Bar, equity: float):
            if not self._bought and bar_index == 1:
                self._bought = True
                return Order(
                    type=OrderType.MARKET,
                    symbol="TEST",
                    side=1,
                    qty=0.1,
                    protective_stop=0.0,
                    take_profit=None,
                    submitted_at=bar.ts,
                )

    return _Simple


def test_compare_latency_impact_returns_result():
    from backtesting.tick import Tick

    ticks = _make_ticks(n=60, start_price=100.0)
    factory = _build_strategy_factory()

    result = compare_latency_impact(
        ticks=ticks,
        strategy_factory=factory,
        ack_latency_ns=2_000_000_000,   # 2s
        starting_cash=10_000,
        symbol="TEST",
    )

    assert isinstance(result, LatencyImpactResult)
    assert isinstance(result.zero_latency, LatencyStats)
    assert isinstance(result.with_latency, LatencyStats)


def test_compare_latency_impact_zero_run_has_zero_latency():
    ticks = _make_ticks(n=60)
    factory = _build_strategy_factory()

    result = compare_latency_impact(
        ticks=ticks,
        strategy_factory=factory,
        ack_latency_ns=1_000_000_000,
        starting_cash=10_000,
        symbol="TEST",
    )

    zero = result.zero_latency
    if zero.fill_count > 0:
        assert zero.avg_fill_latency_us == pytest.approx(0.0)


def test_compare_latency_impact_latency_run_has_nonzero_latency():
    ticks = _make_ticks(n=60)
    factory = _build_strategy_factory()

    result = compare_latency_impact(
        ticks=ticks,
        strategy_factory=factory,
        ack_latency_ns=1_000_000_000,
        starting_cash=10_000,
        symbol="TEST",
    )

    wl = result.with_latency
    if wl.fill_count > 0:
        assert wl.avg_fill_latency_us > 0.0


def test_compare_latency_impact_sharpe_impact_is_float():
    ticks = _make_ticks(n=60)
    factory = _build_strategy_factory()

    result = compare_latency_impact(
        ticks=ticks,
        strategy_factory=factory,
        ack_latency_ns=1_000_000_000,
        starting_cash=10_000,
        symbol="TEST",
    )

    assert isinstance(result.sharpe_impact, float)
