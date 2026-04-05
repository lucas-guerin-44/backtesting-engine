# Backtesting Engine 2.0

[![Tests](https://github.com/lucas-guerin-44/backtesting-engine/actions/workflows/test.yml/badge.svg)](https://github.com/lucas-guerin-44/backtesting-engine/actions/workflows/test.yml)

An event-driven and vectorized backtesting engine for evaluating trading strategies on historical OHLC data. Built with Python, FastAPI, and Streamlit.

## What This Does

This engine lets you define trading strategies as Python classes, run them against historical price data, and analyze the results through either:

- A **REST API** (FastAPI) for programmatic access and batch optimization
- An **interactive dashboard** (Streamlit) for visual exploration and parameter tuning

The engine simulates realistic execution including:
- **Gap-aware stop losses** (fills at open price when price gaps past stop, not at the stop level)
- **Configurable slippage and commission** (basis-point level)
- **Leverage and margin** with automatic margin-call liquidation
- **Drawdown tracking** (peak-to-trough)

## Architecture

```
┌─────────────────────────────────────────────────┐
│  frontend.py (Streamlit Dashboard)              │
│  api.py      (FastAPI REST API)                 │
├─────────────────────────────────────────────────┤
│  strategy_registry.py                           │
│  strategies.py (8 strategy implementations)     │
├─────────────────────────────────────────────────┤
│  backtesting/                                   │
│    backtest.py   ← orchestration loop           │
│    broker.py     ← trade execution (fills,      │
│                     gap-aware stops, leverage)   │
│    portfolio.py  ← equity curve, drawdown,      │
│                     margin call liquidation      │
│    strategy.py   ← abstract base class          │
│    types.py      ← Bar, Trade dataclasses       │
├─────────────────────────────────────────────────┤
│  utils.py  (data fetching, Sharpe ratio, etc.)  │
│  config.py (env-based configuration)            │
└─────────────────────────────────────────────────┘
```

**Data flow:** Strategies extend `Strategy` and implement `on_bar(i, bar, cash) -> Optional[Trade]`. The `Backtester` iterates over OHLC bars, delegates exit execution to the `Broker` (gap-aware stops, take-profits), asks the strategy for entry signals, and tracks equity via the `Portfolio`.

## Included Strategies

| Strategy | Thesis | Description |
|---|---|---|
| **Trend Following** | Trend | Dual-EMA crossover with ATR stops. Managed-futures workhorse. |
| **Mean Reversion** | Mean Rev | Bollinger Band + RSI at extremes, targeting the middle band. |
| **Momentum** | Momentum | N-bar rate-of-change breakout. Based on Jegadeesh & Titman (1993). |
| **Donchian Breakout** | Breakout | Donchian channel breakout (Turtle Trading, Richard Dennis). |

All strategies share a common risk model:
- **ATR-based position sizing**: risk a fixed fraction of equity per trade (default 2%)
- **Drawdown-scaled sizing**: position size scales down linearly as drawdown deepens
- **Circuit breaker**: halts all new entries when drawdown exceeds a threshold (default 15%)
- **Cooldown**: minimum bars between trades to prevent overtrading

## Setup

### Prerequisites

- Python 3.11+
- An OHLC data source (the engine expects a datalake API, or local CSV files in `ohlc_data/`)

### Installation

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

Copy the example environment file and edit as needed:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `DATALAKE_URL` | `http://127.0.0.1:8888` | OHLC data API endpoint |
| `LOCAL_API_URL` | `http://127.0.0.1:8001` | Backtesting API URL |
| `LOCAL_API_PORT` | `8001` | Port for the FastAPI server |

### Quick Start

Run the end-to-end demo (backtests all strategies, optimizes, walk-forward validates):
```bash
python examples/demo.py
```

### Running

**API server:**
```bash
make backend
# or: uvicorn api:app --reload --port 8001
```

**Dashboard:**
```bash
make frontend
# or: streamlit run frontend.py
```

**Tests:**
```bash
python -m pytest tests/ -v
```

## Execution Model

The `Backtester` processes each bar in this order:

1. **Stop-loss exits** (via Broker) - gap-aware: if bar opens past stop, fills at open price (worse), not the stop level
2. **Take-profit exits** (via Broker)
3. **Strategy signal** - `on_bar()` receives available cash after accounting for open positions
4. **Entry execution** (via Broker) - with leverage/margin checks, slippage, commission
5. **Portfolio update** - equity curve, drawdown tracking, margin call check

When both stop and take-profit are hit in the same bar, `execution_priority` determines which fires first (default: `"stop_first"` for conservative estimates).

## Data Format

The engine expects OHLC data with these columns:

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime | Bar timestamp (UTC) |
| `open` | float | Open price |
| `high` | float | High price |
| `low` | float | Low price |
| `close` | float | Close price |

Local CSV files are cached in `ohlc_data/` with the naming convention `{INSTRUMENT}_{TIMEFRAME}.csv`.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/instruments` | List available instruments |
| `GET` | `/timeframes?instrument=X` | List timeframes for an instrument |
| `GET` | `/param_space/{strategy}` | Get parameter schema for a strategy |
| `POST` | `/backtest/run` | Run a backtest, returns metrics + equity curve + trades |

## Writing a New Strategy

```python
from typing import Optional
from backtesting.strategy import Strategy
from backtesting.types import Bar, Trade

class MyStrategy(Strategy):
    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def on_bar(self, i: int, bar: Bar, cash: float) -> Optional[Trade]:
        # Return a Trade to enter, or None to skip
        return Trade(
            entry_bar=bar,
            side=1,          # +1 long, -1 short
            size=cash * 0.1 / bar.close,
            entry_price=bar.close,
            stop_price=bar.close * 0.98,
            take_profit=bar.close * 1.04,
        )
```

Register it in `strategy_registry.py` to make it available in the API and dashboard.

## Parameter Optimizer

`optimizer.py` provides Bayesian parameter search (Optuna TPE) with walk-forward validation.

### How it works

The optimizer runs the Backtester internally for each trial — you don't need to run the optimizer and then re-run the backtest separately. Each trial constructs a strategy with sampled parameters, runs a full backtest, and scores the result. The best parameters and their backtest results are returned together.

### Objective functions

You choose what the optimizer maximizes:

| Objective | Formula | Best for |
|---|---|---|
| `"sharpe"` | mean(excess returns) / std(returns), annualized | General-purpose risk-adjusted performance |
| `"return"` | (final equity - initial) / initial | Maximizing raw P&L (ignores risk) |
| `"calmar"` | total return / max drawdown | Prioritizing drawdown protection |
| `"sortino"` | mean(returns) / std(negative returns), annualized | Penalizing downside volatility only |

### Single-period optimization

Fast parameter search over one date range. Good for exploration, but the results are overfit — the optimizer will find parameters that look great on the data it trained on.

```python
from optimizer import optimize
from strategies import TrendFollowingStrategy

result = optimize(
    strategy_cls=TrendFollowingStrategy,
    param_space={
        "fast_period": (5, 40),       # int range
        "slow_period": (20, 100),     # int range
        "atr_stop_mult": (1.0, 5.0),  # float range
        "risk_per_trade": (0.01, 0.05),
    },
    df=df,
    n_trials=100,
    objective="sharpe",            # or "return", "calmar", "sortino"
    commission_bps=5.0,
    slippage_bps=2.0,
)

print(result.best_params)   # {'fast_period': 39, 'slow_period': 88, ...}
print(result.best_score)    # 0.6492
print(result.all_trials)    # DataFrame of all 100 trials with params + scores
```

### Walk-forward validation

The only honest way to evaluate parameter tuning. Splits data into rolling train/test windows:

1. **Train**: optimize parameters on the training portion (in-sample)
2. **Test**: evaluate those parameters on unseen data (out-of-sample)

The gap between in-sample and out-of-sample performance is the **degradation** — a direct measure of overfitting.

```python
from optimizer import walk_forward

wf = walk_forward(
    strategy_cls=TrendFollowingStrategy,
    param_space={"fast_period": (5, 40), "slow_period": (20, 100)},
    df=df,
    n_splits=4,         # 4 rolling windows
    train_ratio=0.7,    # 70% train, 30% test per window
    n_trials=50,        # Optuna trials per training window
    objective="sharpe",
)

print(wf.summary)               # DataFrame with IS/OOS scores per split
print(wf.in_sample_mean)        # Average in-sample Sharpe
print(wf.out_of_sample_mean)    # Average out-of-sample Sharpe (the real number)
print(wf.degradation)           # IS - OOS (high = overfitting)
```

### Performance

Two execution tiers, both pure Python (no C extensions):

| Engine | Throughput | Use case |
|---|---|---|
| Event-driven (`Backtester`) | ~300,000 bars/sec | Any strategy, complex state allowed |
| Vectorized (`VectorizedBacktester`) | ~600,000 bars/sec | Array-expressible strategies (all 4 included) |

At vectorized speed, 1,000 Optuna trials on 3,000 bars completes in ~5 seconds.

## Visualization

```python
from backtesting.plot import plot_backtest, plot_strategy_comparison

# Single strategy equity curve with drawdown
plot_backtest(equity_curve, trades, timestamps=list(df.index),
              title="Trend Following — XAUUSD H1", save_path="chart.png")

# Compare multiple strategies
plot_strategy_comparison({
    "Trend Following": eq_trend,
    "Mean Reversion": eq_mean_rev,
    "Momentum": eq_momentum,
}, save_path="comparison.png")
```

## Testing

98 tests covering:

| Module | Tests | Coverage |
|---|---|---|
| `test_backtest.py` | 15 | Engine basics, trade execution, commission/slippage, gap-aware stops, drawdown |
| `test_broker.py` | 13 | Open/close trades, stop/TP execution, gap fills, buying power, costs |
| `test_portfolio.py` | 6 | Equity tracking, drawdown, margin call liquidation |
| `test_strategies.py` | 22 | ABC, incremental indicators, vectorized indicators, risk sizing, signals |
| `test_optimizer.py` | 8 | Single-period search, walk-forward validation, objective functions |
| `test_vectorized.py` | 13 | Vectorized backtester, signal generators, gap-aware stops |
| `test_utils.py` | 12 | Sharpe ratio, sanitize, timeframe normalization |
| `test_types.py` | 5 | Bar and Trade dataclass creation |

```bash
python -m pytest tests/ -v  # ~0.5s
```

## Known Limitations

- **Single-asset per backtest run.** The Broker/Portfolio support multi-asset positions, but the Backtester orchestrates one instrument at a time.
- **No volume data** used in execution simulation (fills are assumed regardless of volume).
- **Bar-level resolution** means intra-bar price path is unknown; execution priority resolves ambiguous bars.
