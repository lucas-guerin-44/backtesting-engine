"""Backtesting engine for trading strategy evaluation."""

from backtesting.latency_broker import LatencyAwareBroker
from backtesting.latency_metrics import (
    FillRecord,
    LatencyImpactResult,
    LatencyStats,
    compare_latency_impact,
)
from backtesting.latency_models import (
    ComponentLatency,
    FixedLatency,
    GaussianLatency,
    LatencyModel,
    LogNormalLatency,
)
from backtesting.order import Order, OrderType, PendingOrder
from backtesting.order_book import Fill, MatchingEngine, OrderBook, PriceLevel
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
from backtesting.portfolio_backtest import (
    AuditEvent, PortfolioBacktester, PortfolioBacktestResult, RiskLimits,
)
from backtesting.strategy import Strategy
from backtesting.tick import Tick, TickAggregator
from backtesting.tick_backtest import TickBacktester
from backtesting.types import Bar, Trade

__all__ = [
    "Allocator",
    "AuditEvent",
    "Bar",
    "Backtester",
    "Broker",
    "compare_latency_impact",
    "ComponentLatency",
    "CorrelationAwareAllocator",
    "EqualWeightAllocator",
    "Fill",
    "FillRecord",
    "FixedLatency",
    "GaussianLatency",
    "LatencyAwareBroker",
    "LatencyImpactResult",
    "LatencyModel",
    "LatencyStats",
    "LogNormalLatency",
    "MatchingEngine",
    "Order",
    "OrderBook",
    "OrderType",
    "PendingOrder",
    "Portfolio",
    "PortfolioBacktestResult",
    "PortfolioBacktester",
    "PriceLevel",
    "RegimeAllocator",
    "AllocationWeights",
    "RiskLimits",
    "RiskParityAllocator",
    "Strategy",
    "Tick",
    "TickAggregator",
    "TickBacktester",
    "Trade",
]
