"""Tests for pluggable latency models."""

from __future__ import annotations

import math

import numpy as np
import pytest

from backtesting.latency_models import (
    ComponentLatency,
    FixedLatency,
    GaussianLatency,
    LatencyModel,
    LogNormalLatency,
)


# ---------------------------------------------------------------------------
# FixedLatency
# ---------------------------------------------------------------------------

def test_fixed_latency_constant():
    m = FixedLatency(total_us=500.0)
    samples = [m.sample_ns() for _ in range(10)]
    assert all(s == 500_000 for s in samples)


def test_fixed_latency_zero():
    m = FixedLatency(total_us=0.0)
    assert m.sample_ns() == 0


def test_fixed_latency_rejects_negative():
    with pytest.raises(ValueError):
        FixedLatency(total_us=-1.0)


def test_fixed_latency_sample_us():
    m = FixedLatency(total_us=250.0)
    assert m.sample_us() == pytest.approx(250.0)


def test_fixed_latency_repr():
    m = FixedLatency(total_us=100.0)
    assert "FixedLatency" in repr(m)
    assert "100" in repr(m)


# ---------------------------------------------------------------------------
# GaussianLatency
# ---------------------------------------------------------------------------

def test_gaussian_latency_mean_approx():
    rng = np.random.default_rng(0)
    m = GaussianLatency(mean_us=1_000.0, std_us=10.0, rng=rng)
    samples = [m.sample_us() for _ in range(2_000)]
    assert pytest.approx(np.mean(samples), rel=0.05) == 1_000.0


def test_gaussian_latency_nonnegative():
    # Use a large std to generate some near-zero values
    rng = np.random.default_rng(42)
    m = GaussianLatency(mean_us=50.0, std_us=100.0, rng=rng)
    samples = [m.sample_ns() for _ in range(500)]
    assert all(s >= 0 for s in samples)


def test_gaussian_latency_zero_std_is_fixed():
    rng = np.random.default_rng(1)
    m = GaussianLatency(mean_us=200.0, std_us=0.0, rng=rng)
    samples = {m.sample_ns() for _ in range(10)}
    assert samples == {200_000}


def test_gaussian_latency_rejects_negative_mean():
    with pytest.raises(ValueError):
        GaussianLatency(mean_us=-1.0, std_us=10.0)


def test_gaussian_latency_rejects_negative_std():
    with pytest.raises(ValueError):
        GaussianLatency(mean_us=100.0, std_us=-1.0)


def test_gaussian_latency_repr():
    m = GaussianLatency(mean_us=500.0, std_us=100.0)
    assert "GaussianLatency" in repr(m)


# ---------------------------------------------------------------------------
# LogNormalLatency
# ---------------------------------------------------------------------------

def test_lognormal_latency_median_approx():
    rng = np.random.default_rng(7)
    m = LogNormalLatency(median_us=500.0, sigma=0.3, rng=rng)
    samples = sorted(m.sample_us() for _ in range(5_000))
    empirical_median = samples[len(samples) // 2]
    assert pytest.approx(empirical_median, rel=0.05) == 500.0


def test_lognormal_latency_nonnegative():
    rng = np.random.default_rng(0)
    m = LogNormalLatency(median_us=100.0, sigma=2.0, rng=rng)
    samples = [m.sample_ns() for _ in range(500)]
    assert all(s >= 0 for s in samples)


def test_lognormal_latency_right_skewed():
    # Mean should be > median for a log-normal distribution
    rng = np.random.default_rng(3)
    m = LogNormalLatency(median_us=500.0, sigma=0.5, rng=rng)
    samples = [m.sample_us() for _ in range(5_000)]
    assert np.mean(samples) > 500.0


def test_lognormal_latency_rejects_nonpositive_median():
    with pytest.raises(ValueError):
        LogNormalLatency(median_us=0.0, sigma=0.5)
    with pytest.raises(ValueError):
        LogNormalLatency(median_us=-100.0, sigma=0.5)


def test_lognormal_latency_rejects_nonpositive_sigma():
    with pytest.raises(ValueError):
        LogNormalLatency(median_us=500.0, sigma=0.0)
    with pytest.raises(ValueError):
        LogNormalLatency(median_us=500.0, sigma=-0.1)


def test_lognormal_latency_repr():
    m = LogNormalLatency(median_us=400.0, sigma=0.5)
    r = repr(m)
    assert "LogNormalLatency" in r
    assert "400" in r
    assert "0.5" in r


# ---------------------------------------------------------------------------
# ComponentLatency
# ---------------------------------------------------------------------------

def test_component_latency_sums_legs():
    # Two fixed legs: 100us + 200us = 300us always
    m = ComponentLatency(
        network_out=FixedLatency(100.0),
        network_in=FixedLatency(200.0),
    )
    assert m.sample_ns() == 300_000


def test_component_latency_four_legs():
    m = ComponentLatency(
        network_out=FixedLatency(50.0),
        queue=FixedLatency(300.0),
        processing=FixedLatency(20.0),
        network_in=FixedLatency(50.0),
    )
    assert m.sample_ns() == 420_000


def test_component_latency_optional_legs():
    # Only two legs provided; None legs are skipped
    m = ComponentLatency(
        network_out=FixedLatency(100.0),
        network_in=FixedLatency(100.0),
    )
    assert m.sample_ns() == 200_000


def test_component_latency_rejects_empty():
    with pytest.raises(ValueError):
        ComponentLatency()


def test_component_latency_with_stochastic_legs():
    rng = np.random.default_rng(99)
    m = ComponentLatency(
        network_out=GaussianLatency(mean_us=50.0, std_us=5.0, rng=rng),
        queue=LogNormalLatency(median_us=300.0, sigma=0.4, rng=rng),
        processing=FixedLatency(20.0),
        network_in=GaussianLatency(mean_us=50.0, std_us=5.0, rng=rng),
    )
    samples = [m.sample_ns() for _ in range(1_000)]
    # All non-negative
    assert all(s >= 0 for s in samples)
    # Mean should be roughly 50+300+20+50 = 420us
    mean_us = np.mean(samples) / 1_000.0
    assert pytest.approx(mean_us, rel=0.15) == 420.0


def test_component_latency_repr():
    m = ComponentLatency(
        network_out=FixedLatency(50.0),
        network_in=FixedLatency(50.0),
    )
    assert "ComponentLatency" in repr(m)


# ---------------------------------------------------------------------------
# LatencyModel ABC
# ---------------------------------------------------------------------------

def test_latency_model_is_abstract():
    with pytest.raises(TypeError):
        LatencyModel()


def test_all_models_are_latency_model_instances():
    models = [
        FixedLatency(100.0),
        GaussianLatency(mean_us=100.0, std_us=10.0),
        LogNormalLatency(median_us=100.0, sigma=0.3),
        ComponentLatency(network_out=FixedLatency(100.0)),
    ]
    for m in models:
        assert isinstance(m, LatencyModel)


# ---------------------------------------------------------------------------
# LatencyAwareBroker integration with latency_model
# ---------------------------------------------------------------------------

def _make_ticks(n: int = 20):
    import pandas as pd
    from backtesting.tick import Tick
    base_ns = pd.Timestamp("2024-01-01").value
    return [
        Tick(
            ts=pd.Timestamp(base_ns + i * 1_000_000_000, unit="ns"),
            price=100.0 + i * 0.1,
            bid=100.0 + i * 0.1 - 0.05,
            ask=100.0 + i * 0.1 + 0.05,
            volume=100.0,
        )
        for i in range(n)
    ]


def test_latency_model_overrides_fixed_latency():
    """When latency_model is set, it replaces ack_latency_ns + fill_latency_ns."""
    import pandas as pd
    from backtesting.latency_broker import LatencyAwareBroker
    from backtesting.order import Order, OrderType
    from backtesting.portfolio import Portfolio

    ticks = _make_ticks(n=30)
    portfolio = Portfolio(cash=10_000)

    # Fixed model of 5s — should fill after 5 ticks (ticks are 1s apart)
    model = FixedLatency(total_us=5_000_000.0)  # 5s in microseconds

    lb = LatencyAwareBroker(
        broker=portfolio.broker,
        ack_latency_ns=0,   # would give instant fill if model not used
        latency_model=model,
    )

    order = Order(
        type=OrderType.MARKET,
        symbol="TEST",
        side=1,
        qty=1.0,
        protective_stop=0.0,
        take_profit=None,
        submitted_at=ticks[0].ts,
    )
    lb.submit(order, ticks[0].ts)

    # Process ticks 0-4 (5s not yet elapsed)
    for tick in ticks[:5]:
        lb.process_tick(tick)
    assert len(lb.fill_records) == 0

    # Tick 5 (5s elapsed) — should fill
    lb.process_tick(ticks[5])
    assert len(lb.fill_records) == 1


def test_gaussian_model_produces_varying_fill_times():
    """Gaussian model should produce different latencies across orders."""
    import pandas as pd
    from backtesting.latency_broker import LatencyAwareBroker
    from backtesting.order import Order, OrderType
    from backtesting.portfolio import Portfolio

    ticks = _make_ticks(n=200)
    portfolio = Portfolio(cash=100_000)

    rng = np.random.default_rng(42)
    model = GaussianLatency(mean_us=5_000_000.0, std_us=1_000_000.0, rng=rng)

    lb = LatencyAwareBroker(broker=portfolio.broker, latency_model=model)

    # Submit 5 orders at tick 0
    for i in range(5):
        order = Order(
            type=OrderType.MARKET,
            symbol="TEST",
            side=1,
            qty=0.1,
            protective_stop=0.0,
            take_profit=None,
            submitted_at=ticks[0].ts,
        )
        lb.submit(order, ticks[0].ts)

    for tick in ticks:
        lb.process_tick(tick)

    latencies = [r.fill_latency_us for r in lb.fill_records]
    # All should have filled (200 ticks = 200s, mean is 5s)
    assert len(latencies) == 5
    # With Gaussian jitter, latencies should not all be identical
    assert len(set(round(l, 0) for l in latencies)) > 1
