"""Portfolio allocation weight calculators for multi-asset backtesting.

Three schemes:
- **Equal weight**: 1/N per asset
- **Risk parity**: Weight inversely by rolling volatility
- **Correlation-aware**: Risk parity scaled by inverse average pairwise correlation

All allocators gracefully degrade to simpler methods when insufficient data is
available (e.g., risk parity falls back to equal weight during warmup).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

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
