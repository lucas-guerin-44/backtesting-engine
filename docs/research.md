# From Backtester to Signal: A Walk-Forward Research Process

How I went from "every strategy looks overfit" to finding validated edges on gold and FX, and what I learned about the difference between optimizing and actually discovering signal.

## The Starting Point

I built a multi-asset backtesting engine with four strategies (Trend Following, Mean Reversion, Momentum, Donchian Breakout), a Bayesian optimizer, and walk-forward validation. The walk-forward test splits data into train/test windows, optimizes on training data, and evaluates on unseen test data. The gap between in-sample and out-of-sample performance, the **degradation**, directly measures overfitting.

The problem: everything I ran came back with "Large degradation, optimizer is curve-fitting."

## The Diagnosis

### Problem 1: Not Enough Data

With 6 years of daily data (2019-2024, ~1,500 bars) and 3 walk-forward splits, each out-of-sample window was ~7 months. A trend-following strategy with a 50-100 bar slow EMA generates maybe 2-3 crossover signals per year. Seven months might contain zero crossovers, not because the strategy is bad, but because there aren't enough bars to observe the signal.

**Fix:** Extended the dataset to 13 years (2012-2025, ~5,500 bars). Each OOS window now spans ~18 months with 12-30 trades, enough for a Sharpe ratio to stabilize.

### Problem 2: Degenerate Parameters

The optimizer was finding `fast_period=33, slow_period=37`, a 4-bar gap between moving averages. This is a near-degenerate crossover that fires only under very specific conditions. It looked great in-sample (few but "perfect" trades) and produced zero trades out-of-sample.

**Fix:** Added parameter constraints enforcing `slow_period - fast_period >= 15` to prevent the optimizer from exploiting narrow parameter pockets.

### Problem 3: Too Many Degrees of Freedom

Optimizing 5 parameters jointly with 30 trials gives the optimizer enough freedom to find noise. The curse of dimensionality: in a 5-dimensional space, 30 samples covers almost nothing, so the "best" trial is just the luckiest.

**Fix:** Two-stage approach. First, run a full 1,000-trial optimization to find the convergence region. Then lock down the converged parameters and only optimize `risk_per_trade` in walk-forward, reducing the search space to 1 dimension.

### Problem 4: Wrong Strategy on Wrong Asset

The original portfolio used Mean Reversion on EURUSD because "FX pairs mean-revert." Testing all four strategies on EURUSD told a different story:

| Strategy | Return | Sharpe | Trades |
|---|---|---|---|
| Mean Reversion (200 EMA filter) | -3.04% | -0.09 | 21 |
| Momentum (200 EMA filter) | +19.49% | 0.27 | 69 |
| Donchian Breakout (200 EMA filter) | +33.13% | 0.38 | 198 |

The "obvious" strategy was the worst. Donchian breakout, a trend strategy, outperformed on a pair that's supposed to mean-revert. EURUSD has long, slow directional moves (EUR weakening 2014-2015, strengthening 2017, weakening 2021-2022) that breakout strategies capture better than mean reversion.

## The Results

### Single-Asset: Momentum on XAUUSD

**Walk-forward (locked params, 1 free parameter):**

| Split | Train | Test | IS Sharpe | OOS Sharpe | OOS Return | OOS Max DD |
|---|---|---|---|---|---|---|
| 0 | 2012-2017 | 2017-2018 | 0.31 | 2.07 | +25.6% | 2.3% |
| 1 | 2012-2021 | 2021-2022 | 0.55 | 1.98 | +17.1% | 2.0% |
| 2 | 2012-2024 | 2024-2025 | 1.03 | 2.19 | +24.9% | 5.5% |

- OOS Sharpe: **2.0** (all 3 splits positive)
- Degradation: **-1.39** (negative = OOS outperformed IS)

**True holdout (2012-2022 train, 2023-2025 completely unseen):**
- +66% return, 6% max DD, Sharpe 1.7, 63 trades, 59% win rate
- vs. Buy & Hold: +136% return but 41% max drawdown

The strategy captures nearly half the buy & hold return with 3x less drawdown. The edge here isn't pure alpha generation but rather risk-adjusted returns.

**Statistical significance:**
- Bootstrap 95% CI: [0.37, 1.21], doesn't cross zero
- Survives Deflated Sharpe Ratio after 1,000 trials tested

### Single-Asset: Donchian on EURUSD

**Walk-forward (locked params, 1 free parameter):**

| Split | Test Period | OOS Sharpe | OOS Return | OOS Max DD |
|---|---|---|---|---|
| 0 | 2017-2018 | 0.80 | +5.2% | 1.7% |
| 1 | 2021-2022 | 1.27 | +7.1% | 1.4% |
| 2 | 2024-2025 | 1.69 | +14.8% | 1.3% |

- OOS Sharpe: **1.25**, Degradation: **-0.46**
- On a pair that *lost* 9% buy-and-hold over the same period

### Portfolio: 6 Assets, Locked-Param Walk-Forward

The portfolio optimizer hit the same overfitting wall as single assets, but at a larger scale. With 6 assets and 4-5 signal params each, the full search space is 25+ dimensions. Even 200 Optuna trials can't explore that meaningfully.

**Before locking params** (25+ free params, 30 trials):
- OOS Sharpe: 0.24, Degradation: +0.27

**After locking signal params** (only risk_per_trade per asset, 6 free params, 200 trials):
- OOS Sharpe: **0.73**, Degradation: **+0.14**

The remaining 0.14 degradation is expected. The single-asset tests got negative degradation because 1 param on 1 asset is trivial for the optimizer. At the portfolio level, 6 interacting risk sizing parameters is a harder problem, and 0.14 means the OOS Sharpe retained 84% of the IS Sharpe. That's a well-calibrated optimizer, not curve-fitting.

For reference, the unoptimized Equal Weight baseline (default params, no optimization):

| Metric | Value |
|---|---|
| Return | +83.2% |
| Sharpe | 0.99 |
| Max Drawdown | 7.0% |

vs. Buy & Hold at +152% return but 15%+ max drawdown. The portfolio captures half the return with less than half the risk.

## What Actually Mattered

In order of impact:

1. **More data**, going from 6 to 13 years was the single biggest unlock. Not because the strategies changed, but because you can't measure a Sharpe ratio from 2 trades.
2. **Locking parameters**, reducing the optimizer to 1 degree of freedom eliminated overfitting on single assets. The same approach at portfolio level (6 free params instead of 25+) took degradation from 0.27 to 0.14.
3. **Testing every strategy on every asset**, the "right" strategy for an asset class isn't always the obvious one. Mean Reversion on EURUSD was the "obvious" choice and the worst performer. Donchian Breakout was the actual winner.
4. **The 200 EMA filter**, on every single strategy and asset, adding a long-term trend filter improved results. It doesn't add complexity to the signal, it just prevents taking the wrong side of the market.
5. **Weight capping on allocators**, Risk Parity and Correlation-Aware allocators were putting 50-68% into low-volatility assets and starving the portfolio of exposure. Capping individual weights at 30% and redistributing the excess turned Risk Parity from 4 trades and -0.78% to 138 trades and +11.35%.

## What This Doesn't Prove

- These results don't account for funding costs, slippage beyond the modeled 1-5 bps, or execution risk during momentum spikes
- The strategies are long-only above the 200 EMA (by design), so they won't perform in sustained bear markets — they'll sit flat
- 13 years is better than 6, but it's still one sample of history — the 2012-2025 period includes a historically strong gold bull market
- Walk-forward validation reduces overfitting but doesn't eliminate it — paper trading is the next step
