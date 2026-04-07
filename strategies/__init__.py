"""
Trading strategies for the backtesting engine.

Four strategies, each representing a distinct market thesis:
- TrendFollowing:   EMA crossover with ATR trailing stops
- MeanReversion:    Bollinger Band + RSI at band extremes
- Momentum:         N-bar rate-of-change breakout
- DonchianBreakout: Channel breakout (Turtle-style)

All strategies share a common risk model:
- Position size scaled to risk a fixed fraction of equity per trade (ATR-based)
- Equity curve circuit breaker: halts new entries when drawdown exceeds a threshold
- Cooldown between trades to prevent overtrading

Signals are generated at bar close and executed at the next bar's open
(no lookahead bias). The ``entry_price`` on the returned Trade is the
signal price used for sizing; the Backtester overrides the fill to the
next bar's open.

Indicator computations use O(1) incremental classes from ``backtesting.indicators``
--- no pandas Series or numpy array allocations per bar.
"""

from strategies.base import TrendFilter, risk_adjusted_size
from strategies.trend_following import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from strategies.donchian import DonchianBreakoutStrategy

__all__ = [
    "risk_adjusted_size",
    "TrendFilter",
    "TrendFollowingStrategy",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "DonchianBreakoutStrategy",
]
