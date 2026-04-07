# From Backtester to Signal: A Walk-Forward Research Process

How I went from "every strategy looks overfit" to understanding what's real and what isn't, and what I learned about the difference between optimizing and actually discovering signal.

## The Starting Point

I built a multi-asset backtesting engine with four strategies (Trend Following, Mean Reversion, Momentum, Donchian Breakout), a Bayesian optimizer, and walk-forward validation. The walk-forward test splits data into train/test windows, optimizes on training data, and evaluates on unseen test data. The gap between in-sample and out-of-sample performance, the **degradation** (IS - OOS), directly measures overfitting.

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

**Fix:** Two-stage walk-forward. For each split, run a full optimization on the training data to find signal params, then lock those and optimize only `risk_per_trade` on the same training data. The test window is never seen by either stage. This is critical: an earlier version locked params from a full-dataset optimization, which contaminates the walk-forward because the locked params were selected by a process that saw the OOS windows.

### Problem 4: Wrong Strategy on Wrong Asset

The original portfolio used Mean Reversion on EURUSD because "FX pairs mean-revert." Testing all four strategies on EURUSD told a different story:

| Strategy | Return | Sharpe | Trades |
|---|---|---|---|
| Mean Reversion (200 EMA filter) | -3.04% | -0.09 | 21 |
| Momentum (200 EMA filter) | +19.49% | 0.27 | 69 |
| Donchian Breakout (200 EMA filter) | +33.13% | 0.38 | 198 |

The "obvious" strategy was the worst. Donchian breakout, a trend strategy, outperformed on a pair that's supposed to mean-revert. EURUSD has long, slow directional moves (EUR weakening 2014-2015, strengthening 2017, weakening 2021-2022) that breakout strategies capture better than mean reversion.

### Problem 5: The 200 EMA Filter Does Most of the Work

Across every strategy and every asset, adding a 200-bar EMA trend filter was the single largest improvement. It doesn't refine the signal. It just prevents taking the wrong side of the market (no shorts in a bull market, no longs in a bear market).

This matters for interpreting the results: a momentum strategy with a 200 EMA filter on gold during a 13-year bull market is partially capturing beta to gold, not purely a momentum signal. The strategy is structurally long most of the time because gold has been above its 200 EMA for most of 2012-2025. The risk-adjusted improvement (lower drawdown than buy & hold) is real, but the raw return is largely riding the trend.

## The Results

### Single-Asset: Momentum on XAUUSD

**Two-stage walk-forward (clean, no data leakage):**

Each split finds its own signal params on training data only, then evaluates on unseen test data.

| Split | Train | Test | IS Sharpe | OOS Sharpe | OOS Return | OOS Max DD | OOS Trades |
|---|---|---|---|---|---|---|---|
| 0 | 2012-2017 | 2017-2018 | 0.48 | -0.07 | -0.5% | 4.9% | 3 |
| 1 | 2012-2021 | 2021-2022 | 0.81 | -0.35 | -3.7% | 7.8% | 23 |
| 2 | 2012-2024 | 2024-2025 | 0.88 | 1.26 | +20.5% | 6.4% | 33 |

- OOS Sharpe: **0.28** (1 of 3 splits positive)
- Degradation: **0.44** (positive, borderline)

The signal params did not converge across splits (lookback ranged from 6 to 32), which tells us the momentum signal on gold is regime-dependent, not a stable structural feature. It works during strong trending periods (2024-2025) and fails during choppy ones (2017-2018, 2021-2022).

**True holdout (2012-2022 train, 2023-2025 completely unseen):**
- +145% return, 5% max DD, Sharpe 1.98, 82 trades, 59% win rate
- vs. Buy & Hold: +136% return but 41% max drawdown

The holdout is clean and strong, but it covers a period (2023-2025) where gold was in a historic bull run. A momentum strategy with a long-only bias will naturally outperform during exactly this kind of regime.

**Statistical significance (on full 2012-2025 baseline, default params):**
- Bootstrap 95% CI: [0.37, 1.21], doesn't cross zero
- Survives Deflated Sharpe Ratio after 1,000 trials tested

### Single-Asset: Donchian on EURUSD

**Two-stage walk-forward (clean, no data leakage):**

| Split | Train | Test | IS Sharpe | OOS Sharpe | OOS Return | OOS Max DD | OOS Trades |
|---|---|---|---|---|---|---|---|
| 0 | 2012-2017 | 2017-2018 | 0.86 | 1.22 | +13.9% | 3.7% | 3 |
| 1 | 2012-2021 | 2021-2022 | 0.80 | 1.02 | +6.7% | 2.3% | 17 |
| 2 | 2012-2024 | 2024-2025 | 1.22 | 0.87 | +7.5% | 2.8% | 23 |

- OOS Sharpe: **1.04** (all 3 splits positive)
- Degradation: **-0.08** (negative, OOS slightly better than IS)
- On a pair that *lost* 9% buy-and-hold over the same period

Unlike XAUUSD momentum, the Donchian signal on EURUSD is more stable: `channel_period` converged to 40 in 2 of 3 splits, and all three OOS windows were positive across different market regimes (2017-2018 ranging, 2021-2022 USD strength, 2024-2025 EUR recovery). The OOS Sharpe decreases over time (1.22 → 1.02 → 0.87), consistent with a real edge that naturally erodes, as opposed to the contaminated version which showed monotonically *increasing* OOS Sharpe (0.80 → 1.27 → 1.69), a red flag for regime overfitting.

A note on negative degradation: while degradation of -0.08 looks encouraging (OOS better than IS), it's not unambiguously good. It can also indicate that the IS periods happened to include tougher regimes, making the IS score artificially low rather than the strategy being especially robust. The key metric is whether OOS is consistently positive across different market conditions, which it is here.

### Portfolio: 6 Assets

The portfolio optimizer hit the same overfitting wall as single assets, but at a larger scale. With 6 assets and 4-5 signal params each, the full search space is 25+ dimensions. Even 200 Optuna trials can't explore that meaningfully.

**Two-stage walk-forward (clean, no data leakage):**

Each split runs its own full optimization on training data to find signal params, locks those, optimizes risk_per_trade on training data, then evaluates on unseen test data.

| Split | Test | IS Sharpe | OOS Sharpe | OOS Return | OOS Max DD | OOS Trades |
|---|---|---|---|---|---|---|
| 1 | 2021-2022 | 1.05 | 1.19 | +9.6% | 2.7% | 36 |
| 2 | 2024-2025 | 0.95 | 0.11 | +0.3% | 1.9% | 39 |

- OOS Sharpe: **0.65**, Degradation: **+0.35**
- Both splits positive, with real trade counts (36-39)

The 0.35 degradation is higher than the single-asset EURUSD result but expected: optimizing 6 risk_per_trade parameters simultaneously is a harder problem than 1. The OOS Sharpe of 0.65 is the honest number, modest but positive.

For reference, the unoptimized Equal Weight baseline (default params, no optimization):

| Metric | Value |
|---|---|
| Return | +83.2% |
| Sharpe | 0.99 |
| Max Drawdown | 7.0% |

vs. Buy & Hold at +152% return but 15%+ max drawdown. The portfolio captures half the return with less than half the risk.

## What Actually Mattered

In order of impact:

1. **The 200 EMA filter** is the single most impactful variable. On every strategy and every asset, it improved results. It's not a signal refinement but a regime filter that keeps you from fighting the macro trend. It's also the main reason the strategies appear to have alpha: they're structurally long during bull markets and flat during bear markets. This is a form of market timing, not pure momentum/breakout alpha.
2. **More data**, going from 6 to 13 years was the second biggest unlock. Not because the strategies changed, but because you can't measure a Sharpe ratio from 2 trades.
3. **Clean walk-forward methodology**, running both optimization stages inside each training window eliminates data leakage. The earlier approach (locking params from full-dataset optimization) inflated single-asset OOS Sharpe from 0.28 to 2.0, a 7x overstatement. At the portfolio level, contaminated OOS Sharpe was 0.73 vs. the clean 0.65, a smaller gap, but the principle matters. The contamination is subtle and easy to miss.
4. **Testing every strategy on every asset**, the "right" strategy for an asset class isn't always the obvious one. Mean Reversion on EURUSD was the "obvious" choice and the worst performer. Donchian Breakout was the actual winner.
5. **Weight capping on allocators**, Risk Parity and Correlation-Aware allocators were putting 50-68% into low-volatility assets and starving the portfolio of exposure. Capping individual weights at 30% and redistributing the excess turned Risk Parity from 4 trades and -0.78% to 138 trades and +11.35%.

## What This Doesn't Prove

- **Gold beta, not alpha.** A long-only momentum strategy with a 200 EMA filter on gold during a 13-year bull market is partially capturing beta. The risk-adjusted improvement over buy & hold is real (much lower drawdown), but the raw return is largely riding the secular trend. The strategy would sit flat or produce small losses during a sustained gold bear market.
- **The signal is regime-dependent.** Walk-forward splits show the momentum signal works during strong trends (2024-2025) and fails during chop (2017-2018, 2021-2022). Signal params don't converge across splits, which means there isn't one stable "momentum signal" on gold, it's a strategy that works in certain environments.
- **Slippage is understated.** The model uses fixed 2-5 bps costs. Momentum strategies by design enter on breakouts, exactly when spreads widen and market impact is highest. A sensitivity analysis at 2-5x the modeled slippage would give a more realistic cost estimate.
- **n=3 splits is not a sample size.** The OOS Sharpe of 0.28 is a mean of three values (-0.07, -0.35, 1.26). The confidence interval around that mean is extremely wide. Three data points cannot establish statistical significance on their own.
- **The holdout is one period.** The 2023-2025 holdout shows strong results, but it's a single test on a period that happened to be one of the strongest gold rallies in history. One positive holdout doesn't validate a strategy.
- Walk-forward validation reduces overfitting but doesn't eliminate it. Paper trading is the next step.
