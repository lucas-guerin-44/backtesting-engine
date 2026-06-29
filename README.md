# Backtesting Engine

[![Tests](https://github.com/lucas-guerin-44/backtesting-engine/actions/workflows/test.yml/badge.svg)](https://github.com/lucas-guerin-44/backtesting-engine/actions/workflows/test.yml)

Event-driven backtesting engine built for execution realism.  
Tick-granularity fills, gap-aware stops, stochastic latency modeling, and a FIFO order book.

Data sourced from [lucas-guerin-44/datalake-api](https://github.com/lucas-guerin-44/datalake-api).

![Strategy Comparison](docs/strategy_comparison.png)
*Four strategies + buy & hold on XAUUSD D1 (2012-2025). 5bps commission, 0.4bps spread. Walk-forward validated — see [research process](docs/research.md).*

## Why this engine

Most backtesters optimize for speed or ease of use. This one optimizes for **not lying to you**.

- **No lookahead bias.** Signals fire on bar close, fills happen at next bar's open (or next tick). The engine never sees the future.
- **Execution realism.** Gap-aware stops fill at the open when price gaps through — not at the stop level. Stochastic latency models delay fills by configurable time. The FIFO order book handles partial fills and queue position.
- **Costs you can measure.** Spread modeled from MT5 M1 data, not guessed. Market impact scales with `sigma * sqrt(Q/ADV)`. Funding/swap accrual per bar. Commission on entry and exit.
- **Statistical honesty.** Walk-forward validation, Deflated Sharpe Ratio (corrects for multiple testing), bootstrap confidence intervals, permutation tests. The optimizer doesn't just find params — it tells you if they're real.

## Quick start

```bash
pip install -e .    # from backtesting-engine/
```

```python
from backtesting.backtest import Backtester
from backtesting.types import BacktestConfig
from strategies import MomentumStrategy

config = BacktestConfig(
    starting_cash=10_000,
    commission_bps=5.0,
    spread_bps=0.4,               # measured from MT5 M1 data
    funding_rate_annual=5.0,       # long swap rate %
    funding_rate_short=8.0,        # short swap rate %
)

bt = Backtester(df, MomentumStrategy(trend_filter_period=200), config=config, symbol="XAUUSD")
equity_curve, trades = bt.run()
```

## Architecture

```
strategies/              4 included (trend, reversion, momentum, donchian)
    ↓ on_bar()
backtesting/
    backtest.py          Event-driven engine (~300k bars/sec)
    tick_backtest.py     Tick-level engine (~2.4M ticks/sec)
    vectorized.py        Numpy engine (~700k bars/sec)
    ↓
    broker.py            Gap-aware stops, slippage, commission
    latency_broker.py    Stochastic delay (Gaussian, LogNormal, per-leg)
    order_book.py        FIFO matching, partial fills, limit resting
    ↓
    portfolio.py         Equity, drawdown, margin calls, funding accrual
    statistics.py        Sharpe, bootstrap CI, Deflated Sharpe Ratio
    optimizer.py         Optuna TPE + walk-forward validation
```

**Bar-level flow:** stop exits → TP exits → strategy signal → entry at next bar open → portfolio update. When both stop and TP fire in the same bar, stops execute first (conservative default).

**Tick-level flow:** stop/TP check at exact tick price → fill at next tick → aggregate into bar → `on_bar()` fires. No gap logic needed — ticks are the atomic price updates.

## Strategies

| Strategy | Description |
|---|---|
| **Trend Following** | Dual-EMA crossover with ATR trailing stops and trend re-entry |
| **Mean Reversion** | Bollinger Band + RSI at extremes, targeting the middle band |
| **Momentum** | N-bar rate-of-change breakout (Jegadeesh & Titman 1993) |
| **Donchian Breakout** | Channel breakout, Turtle Trading style (Richard Dennis) |

All share: ATR-based position sizing, drawdown-scaled sizing (linear scale-down), circuit breaker, and trade cooldown.

## Writing a strategy

```python
from backtesting.strategy import Strategy
from backtesting.types import Bar, Trade

class MyStrategy(Strategy):
    def on_bar(self, i: int, bar: Bar, equity: float):
        return Trade(entry_bar=bar, side=1, size=equity * 0.1 / bar.close,
                     entry_price=bar.close, stop_price=bar.close * 0.98,
                     take_profit=bar.close * 1.04)
```

Tick-level strategies can override `on_tick()` and `manage_position_tick()` for intra-bar logic — both default to no-ops so bar-only strategies work unchanged.

## Optimization

```python
from optimizer import optimize, walk_forward

result = optimize(MomentumStrategy, param_space={"lookback": (5, 40)},
                  df=df, n_trials=500, objective="sharpe")

wf = walk_forward(MomentumStrategy, param_space, df, n_splits=3, n_trials=200)
print(wf.degradation)  # IS - OOS: near-zero = real edge, large = overfitting
```

## Tests

337 tests, ~4s. Cross-engine consistency checks (event-driven = vectorized), walk-forward contamination regression, indicator edge cases.

```bash
python -m pytest tests/ -x -q
```

## See also

- **[datalake-api](https://github.com/lucas-guerin-44/datalake-api)** — Data pipeline for OHLC/tick data
- **[docs/research.md](docs/research.md)** — Research process: diagnosing overfitting, fixing methodology
- **[docs/backtester_engineering.md](docs/backtester_engineering.md)** — Full engineering walkthrough

## Known limitations

- Session filtering is at the data layer (`load_data()` in `utils.py`), not in the engine itself.
- No corporate action adjustments (splits, dividends) — use adjusted data.
