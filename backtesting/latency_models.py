"""Pluggable latency models for LatencyAwareBroker.

Each model implements a ``sample_ns() -> int`` method that returns a
nanosecond delay to add to an order's fill-eligibility timestamp. Pass
a model instance as ``latency_model`` to ``LatencyAwareBroker``; it
overrides the fixed ``ack_latency_ns + fill_latency_ns`` values.

Available models
----------------
FixedLatency
    Constant delay. Equivalent to setting ``ack_latency_ns`` directly.

GaussianLatency
    Normally distributed delay, floored at zero. Captures symmetric
    jitter around a mean round-trip time.

LogNormalLatency
    Log-normally distributed delay. Right-skewed - most fills are fast,
    rare tail events are slow. Closer to empirical network latency
    distributions.

ComponentLatency
    Sums independently sampled legs: network_out, exchange_queue,
    processing, network_in. Lets you model each stage separately. Each
    leg is itself a LatencyModel.

Usage
-----
>>> from backtesting.latency_models import GaussianLatency, ComponentLatency, FixedLatency
>>> from backtesting.latency_broker import LatencyAwareBroker
>>>
>>> model = GaussianLatency(mean_us=500, std_us=100)
>>> lb = LatencyAwareBroker(broker=portfolio.broker, latency_model=model)
>>>
>>> # Component model: four independently sampled legs
>>> model = ComponentLatency(
...     network_out=GaussianLatency(mean_us=50, std_us=10),
...     queue=LogNormalLatency(median_us=300, sigma=0.6),
...     processing=FixedLatency(total_us=20),
...     network_in=GaussianLatency(mean_us=50, std_us=10),
... )
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class LatencyModel(ABC):
    """Abstract base for latency samplers.

    Subclasses implement ``sample_ns()`` to return a single nanosecond
    delay drawn from whatever distribution they model. The returned
    value is always non-negative.
    """

    @abstractmethod
    def sample_ns(self) -> int:
        """Draw one latency sample, in nanoseconds."""

    def sample_us(self) -> float:
        """Convenience: return sample in microseconds."""
        return self.sample_ns() / 1_000.0


class FixedLatency(LatencyModel):
    """Constant latency - every order waits exactly ``total_us`` microseconds.

    Parameters
    ----------
    total_us : float
        Delay in microseconds.
    """

    def __init__(self, total_us: float) -> None:
        if total_us < 0:
            raise ValueError(f"total_us must be >= 0, got {total_us}")
        self._ns = int(total_us * 1_000)

    def sample_ns(self) -> int:
        return self._ns

    def __repr__(self) -> str:
        return f"FixedLatency(total_us={self._ns / 1_000:.1f})"


class GaussianLatency(LatencyModel):
    """Normally distributed latency, floored at zero.

    Models symmetric jitter around a mean RTT. Values below zero are
    clamped to zero (physical latency cannot be negative).

    Parameters
    ----------
    mean_us : float
        Mean latency in microseconds.
    std_us : float
        Standard deviation in microseconds. Must be >= 0.
    rng : numpy.random.Generator, optional
        Random number generator. Defaults to ``numpy.random.default_rng()``.
    """

    def __init__(
        self,
        mean_us: float,
        std_us: float,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if mean_us < 0:
            raise ValueError(f"mean_us must be >= 0, got {mean_us}")
        if std_us < 0:
            raise ValueError(f"std_us must be >= 0, got {std_us}")
        self._mean_ns = mean_us * 1_000.0
        self._std_ns = std_us * 1_000.0
        self._rng = rng or np.random.default_rng()

    def sample_ns(self) -> int:
        val = self._rng.normal(self._mean_ns, self._std_ns)
        return max(0, int(val))

    def __repr__(self) -> str:
        return (
            f"GaussianLatency(mean_us={self._mean_ns / 1_000:.1f}, "
            f"std_us={self._std_ns / 1_000:.1f})"
        )


class LogNormalLatency(LatencyModel):
    """Log-normally distributed latency.

    Right-skewed: most fills are fast, rare tail events are slow. This
    matches empirical network latency distributions better than Gaussian.

    The distribution is parameterised by its median (50th percentile)
    and the log-space standard deviation ``sigma``. A higher ``sigma``
    gives a heavier right tail.

    Parameters
    ----------
    median_us : float
        Median latency in microseconds (50th percentile of the distribution).
    sigma : float
        Shape parameter (log-space standard deviation). Must be > 0.
        Typical values: 0.3 (tight) to 1.0 (heavy tail).
    rng : numpy.random.Generator, optional
        Random number generator.
    """

    def __init__(
        self,
        median_us: float,
        sigma: float,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if median_us <= 0:
            raise ValueError(f"median_us must be > 0, got {median_us}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        # log-normal: mu_ln = log(median), so median = e^mu_ln
        self._mu_ln = math.log(median_us * 1_000.0)  # in ns
        self._sigma = sigma
        self._rng = rng or np.random.default_rng()

    def sample_ns(self) -> int:
        val = self._rng.lognormal(self._mu_ln, self._sigma)
        return max(0, int(val))

    def __repr__(self) -> str:
        median_us = math.exp(self._mu_ln) / 1_000.0
        return f"LogNormalLatency(median_us={median_us:.1f}, sigma={self._sigma})"


class ComponentLatency(LatencyModel):
    """Latency modelled as a sum of independently sampled legs.

    Each leg is a separate ``LatencyModel``. The total delay is the
    sum of one sample from each leg. Legs are optional - pass ``None``
    to skip a leg entirely.

    Parameters
    ----------
    network_out : LatencyModel, optional
        Client-to-exchange network leg.
    queue : LatencyModel, optional
        Exchange queue / order matching delay.
    processing : LatencyModel, optional
        Exchange order processing / acknowledgment.
    network_in : LatencyModel, optional
        Exchange-to-client network leg (fill notification).
    """

    def __init__(
        self,
        network_out: Optional[LatencyModel] = None,
        queue: Optional[LatencyModel] = None,
        processing: Optional[LatencyModel] = None,
        network_in: Optional[LatencyModel] = None,
    ) -> None:
        self._legs = [
            leg for leg in (network_out, queue, processing, network_in)
            if leg is not None
        ]
        if not self._legs:
            raise ValueError("ComponentLatency requires at least one leg.")

    def sample_ns(self) -> int:
        return sum(leg.sample_ns() for leg in self._legs)

    def __repr__(self) -> str:
        legs_repr = ", ".join(repr(leg) for leg in self._legs)
        return f"ComponentLatency([{legs_repr}])"
