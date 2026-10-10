#!/usr/bin/env python3
"""Benchmark the shared-account portfolio event engine.

Usage::

    python benchmarks/bench_portfolio.py --bars 50000 --symbols 4 --runs 3
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.instruments import InstrumentSpec
from backtesting.portfolio_backtest import PortfolioBacktester
from backtesting.strategy import Strategy


class NoTradeStrategy(Strategy):
    def on_bar(self, i, bar, equity):
        return None


def main():
    parser = argparse.ArgumentParser(description="Benchmark the portfolio backtester")
    parser.add_argument("--bars", type=int, default=50_000)
    parser.add_argument("--symbols", type=int, default=4)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.bars < 1 or args.symbols < 1 or args.runs < 1:
        parser.error("bars, symbols, and runs must all be >= 1")

    index = pd.date_range("2020-01-01", periods=args.bars, freq="min")
    prices = 100.0 + np.linspace(0.0, 1.0, args.bars)
    frame = pd.DataFrame({
        "open": prices,
        "high": prices + 0.1,
        "low": prices - 0.1,
        "close": prices,
    }, index=index)
    data = {f"S{i:03d}": frame for i in range(args.symbols)}
    specs = {symbol: InstrumentSpec() for symbol in data}

    samples = []
    for _ in range(args.runs):
        bt = PortfolioBacktester(
            data,
            {symbol: NoTradeStrategy() for symbol in data},
            instrument_specs=specs,
        )
        started = time.perf_counter()
        bt.run()
        samples.append(time.perf_counter() - started)

    elapsed = min(samples)
    total_bars = args.bars * args.symbols
    print(f"Symbols: {args.symbols}; bars/symbol: {args.bars:,}")
    print(f"Events:  {total_bars:,} bars across {args.bars:,} timestamps")
    print(f"Best:    {elapsed:.3f}s ({total_bars / elapsed:,.0f} bars/s)")


if __name__ == "__main__":
    main()
