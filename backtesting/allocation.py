"""Portfolio allocation weight calculators for multi-asset backtesting.

Four schemes:
- **Equal weight**: 1/N per asset
- **Risk parity**: Weight inversely by rolling volatility
- **Correlation-aware**: Risk parity scaled by inverse average pairwise correlation
- **Regime-aware**: Shifts weight between trend and mean-reversion assets based
  on a rolling volatility regime signal

All allocators gracefully degrade to simpler methods when insufficient data is
available (e.g., risk parity falls back to equal weight during warmup).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np
import pandas as pd


@dataclass
class AllocationWeights:
    """Result of an allocation calculation."""
    weights: Dict[str, float]  # symbol -> weight (sums to 1.0)
    method: str                # "equal", "risk_parity", "correlation_aware"


class Allocator(ABC):
    """Base class for portfolio allocation weight calculators."""

    @abstractmethod
    def compute_weights(
        self,
        symbols: List[str],
        close_arrays: Dict[str, np.ndarray],
        lookback: int,
        current_idx: int,
    ) -> AllocationWeights:
        """Compute allocation weights for each asset.

        Parameters
        ----------
        symbols : list of str
            Asset symbols.
        close_arrays : dict
            Symbol -> close price numpy array (aligned to master timeline).
        lookback : int
            Number of bars to use for vol/corr estimation.
        current_idx : int
            Current bar index in the aligned timeline.
        """
        ...


class EqualWeightAllocator(Allocator):
    """Each asset gets 1/N of the capital."""

    def compute_weights(self, symbols, close_arrays, lookback, current_idx):
        n = len(symbols)
        w = {s: 1.0 / n for s in symbols}
        return AllocationWeights(weights=w, method="equal")


class RiskParityAllocator(Allocator):
    """Weight inversely by rolling volatility so each asset contributes equal risk.

    ``weight_i = (1 / vol_i) / sum(1 / vol_j)``

    Falls back to equal weight when fewer than ``min_lookback`` bars are available.
    """

    def __init__(self, min_lookback: int = 20):
        self.min_lookback = min_lookback

    def compute_weights(self, symbols, close_arrays, lookback, current_idx):
        if current_idx < self.min_lookback:
            return EqualWeightAllocator().compute_weights(
                symbols, close_arrays, lookback, current_idx)

        start = max(0, current_idx - lookback)
        inv_vols = {}
        for sym in symbols:
            closes = close_arrays[sym][start:current_idx + 1]
            if len(closes) < 2:
                inv_vols[sym] = 1.0
                continue
            returns = np.diff(closes) / closes[:-1]
            vol = np.std(returns, ddof=1)
            inv_vols[sym] = 1.0 / vol if vol > 1e-10 else 1.0

        total = sum(inv_vols.values())
        w = {s: v / total for s, v in inv_vols.items()}
        return AllocationWeights(weights=w, method="risk_parity")


class CorrelationAwareAllocator(Allocator):
    """Risk parity scaled by inverse average pairwise correlation.

    For each asset *i*::

        corr_factor_i = 1 / mean(|corr(i, j)|)  for j != i

    Final weight = risk_parity_weight * corr_factor, then renormalized.

    Falls back to risk parity with fewer than 2 assets or insufficient data.
    """

    def __init__(self, min_lookback: int = 30):
        self.min_lookback = min_lookback

    def compute_weights(self, symbols, close_arrays, lookback, current_idx):
        n = len(symbols)
        if current_idx < self.min_lookback or n < 2:
            return RiskParityAllocator(self.min_lookback).compute_weights(
                symbols, close_arrays, lookback, current_idx)

        start = max(0, current_idx - lookback)

        # Build returns matrix (lookback x n_assets)
        returns_list = []
        for sym in symbols:
            closes = close_arrays[sym][start:current_idx + 1]
            if len(closes) < 2:
                return RiskParityAllocator(self.min_lookback).compute_weights(
                    symbols, close_arrays, lookback, current_idx)
            returns_list.append(np.diff(closes) / closes[:-1])

        # Truncate to same length
        min_len = min(len(r) for r in returns_list)
        if min_len < 5:
            return RiskParityAllocator(self.min_lookback).compute_weights(
                symbols, close_arrays, lookback, current_idx)

        returns_matrix = np.column_stack([r[-min_len:] for r in returns_list])
        corr_matrix = np.corrcoef(returns_matrix, rowvar=False)

        # Fall back to risk parity if correlation matrix contains NaN
        # (happens when an asset has zero variance in the lookback window)
        if np.any(np.isnan(corr_matrix)):
            return RiskParityAllocator(self.min_lookback).compute_weights(
                symbols, close_arrays, lookback, current_idx)

        # Get risk parity base weights
        rp_result = RiskParityAllocator(self.min_lookback).compute_weights(
            symbols, close_arrays, lookback, current_idx)

        # Compute correlation scaling factor for each asset
        corr_factors = {}
        for i, sym in enumerate(symbols):
            avg_abs_corr = np.mean([
                abs(corr_matrix[i, j]) for j in range(n) if j != i
            ])
            corr_factors[sym] = 1.0 / max(avg_abs_corr, 0.1)

        # Apply factor and renormalize
        raw = {s: rp_result.weights[s] * corr_factors[s] for s in symbols}
        total = sum(raw.values())
        w = {s: v / total for s, v in raw.items()}
        return AllocationWeights(weights=w, method="correlation_aware")


class RegimeAllocator(Allocator):
    """Shift allocation between trend-following and mean-reversion assets
    based on a rolling volatility regime signal.

    Measures the average cross-asset volatility percentile over a long history
    window. When current vol is elevated (above ``vol_threshold_pct``), the
    market is in a "trending" regime and trend-following assets get boosted.
    When vol is low, mean-reversion assets get boosted.

    Uses risk parity as the base weighting, then applies a regime multiplier.

    Parameters
    ----------
    trend_symbols : set of str
        Symbols running trend-following or momentum strategies.
    reversion_symbols : set of str
        Symbols running mean-reversion strategies.
    vol_lookback : int
        Short window for measuring current volatility.
    vol_history : int
        Long window for building the vol distribution (percentile baseline).
    vol_threshold_pct : float
        Percentile threshold (0-100). Above = trending regime, below = mean-rev.
    regime_boost : float
        Multiplier applied to the favored group's weights (>1.0). The
        unfavored group gets ``1 / regime_boost``. Weights are renormalized.
    min_lookback : int
        Minimum bars before regime detection activates (falls back to risk parity).
    """

    def __init__(
        self,
        trend_symbols: Set[str],
        reversion_symbols: Set[str],
        vol_lookback: int = 100,
        vol_history: int = 2000,
        vol_threshold_pct: float = 50.0,
        regime_boost: float = 2.0,
        min_lookback: int = 200,
    ):
        self.trend_symbols = set(trend_symbols)
        self.reversion_symbols = set(reversion_symbols)
        self.vol_lookback = vol_lookback
        self.vol_history = vol_history
        self.vol_threshold_pct = vol_threshold_pct
        self.regime_boost = regime_boost
        self.min_lookback = min_lookback

    def compute_weights(self, symbols, close_arrays, lookback, current_idx):
        # Fall back to risk parity during warmup
        if current_idx < self.min_lookback:
            return RiskParityAllocator(min(self.min_lookback, 20)).compute_weights(
                symbols, close_arrays, lookback, current_idx)

        # Measure current vol percentile across all assets
        regime = self._detect_regime(symbols, close_arrays, current_idx)

        # Start from risk parity base weights
        rp = RiskParityAllocator(20).compute_weights(
            symbols, close_arrays, lookback, current_idx)

        # Apply regime boost
        raw = {}
        for sym in symbols:
            base = rp.weights[sym]
            if regime == "trending" and sym in self.trend_symbols:
                raw[sym] = base * self.regime_boost
            elif regime == "ranging" and sym in self.reversion_symbols:
                raw[sym] = base * self.regime_boost
            elif regime == "trending" and sym in self.reversion_symbols:
                raw[sym] = base / self.regime_boost
            elif regime == "ranging" and sym in self.trend_symbols:
                raw[sym] = base / self.regime_boost
            else:
                # Symbol not in either group — keep base weight
                raw[sym] = base

        total = sum(raw.values())
        w = {s: v / total for s, v in raw.items()}
        return AllocationWeights(
            weights=w, method=f"regime:{regime}")

    def _detect_regime(
        self, symbols: List[str], close_arrays: Dict[str, np.ndarray],
        current_idx: int,
    ) -> str:
        """Classify current market regime based on cross-asset vol percentile.

        Returns ``"trending"`` or ``"ranging"``.
        """
        history_start = max(0, current_idx - self.vol_history)
        short_start = max(0, current_idx - self.vol_lookback)

        percentiles = []
        for sym in symbols:
            closes = close_arrays[sym][history_start:current_idx + 1]
            if len(closes) < self.vol_lookback + 10:
                continue

            returns = np.diff(closes) / closes[:-1]

            # Rolling vol over the full history window using a stride trick
            # would be ideal, but a simple expanding comparison is enough:
            # current short-window vol vs distribution of all short-window vols
            current_vol = np.std(returns[-self.vol_lookback:], ddof=1)

            # Build distribution of historical short-window vols
            n_windows = len(returns) - self.vol_lookback + 1
            if n_windows < 10:
                continue
            hist_vols = np.array([
                np.std(returns[i:i + self.vol_lookback], ddof=1)
                for i in range(0, n_windows, max(1, n_windows // 100))
            ])

            pct = np.sum(hist_vols <= current_vol) / len(hist_vols) * 100
            percentiles.append(pct)

        if not percentiles:
            return "trending"  # Default to trending when no data

        avg_pct = np.mean(percentiles)
        return "trending" if avg_pct >= self.vol_threshold_pct else "ranging"
