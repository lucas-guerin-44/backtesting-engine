"""Backtesting engine for trading strategy evaluation.

The canonical engines are the event-driven ``Backtester`` (single symbol) and
``PortfolioBacktester`` (shared-account, multi-instrument). The pure-Python
``VectorizedBacktester`` is a fast screening/optimizer path.

Experimental execution-realism layers (tick engine, latency broker, FIFO order
book, Cython tick core) live on the ``experimental/tick-cython`` branch.
"""

from backtesting.backtest import Backtester
from backtesting.broker import Broker
from backtesting.instruments import InstrumentSpec
from backtesting.portfolio import Portfolio
from backtesting.portfolio_backtest import PortfolioBacktester, PortfolioRiskLimits
from backtesting.strategy import Strategy
from backtesting.types import Bar, Trade
from backtesting.statistics import (
    StatisticalReport,
    compute_sharpe,
    compute_statistical_report,
    compute_stats,
    passes_statistical_gate,
    print_report,
)

__all__ = [
    "Backtester",
    "PortfolioBacktester",
    "PortfolioRiskLimits",
    "InstrumentSpec",
    "Broker",
    "Portfolio",
    "Strategy",
    "Bar",
    "Trade",
    "compute_sharpe",
    "compute_stats",
    "compute_statistical_report",
    "passes_statistical_gate",
    "StatisticalReport",
    "print_report",
]
