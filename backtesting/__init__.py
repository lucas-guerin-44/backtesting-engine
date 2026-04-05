"""Backtesting engine for trading strategy evaluation."""

from backtesting.allocation import (
    Allocator,
    AllocationWeights,
    CorrelationAwareAllocator,
    EqualWeightAllocator,
    RegimeAllocator,
    RiskParityAllocator,
)
from backtesting.backtest import Backtester
from backtesting.broker import Broker
from backtesting.portfolio import Portfolio
from backtesting.portfolio_backtest import PortfolioBacktester, PortfolioBacktestResult
from backtesting.strategy import Strategy
from backtesting.types import Bar, Trade

__all__ = [
    "Allocator",
    "AllocationWeights",
    "Backtester",
    "Bar",
    "Broker",
    "CorrelationAwareAllocator",
    "EqualWeightAllocator",
    "Portfolio",
    "PortfolioBacktestResult",
    "PortfolioBacktester",
    "RegimeAllocator",
    "RiskParityAllocator",
    "Strategy",
    "Trade",
]
