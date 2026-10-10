# Backtester Engineering: From Naive Loop to Execution Layer

This document covers the engineering decisions behind this backtesting engine,
why the abstractions exist, what problems each layer solves, and where each layer
still falls short. It is not about trading strategy or quant methodology. For that,
see `research.md`.

## Scope: canonical `main` vs experimental branch

`main` carries the **canonical** bar-level engines — the event-driven `Backtester`
(single symbol) and `PortfolioBacktester` (shared-account, multi-instrument) — plus
the pure-Python `VectorizedBacktester` used as an optimizer/screening fast path.

The **experimental execution-realism stack** — tick engine (`tick.py`,
`tick_backtest.py`), `LatencyAwareBroker`, FIFO order book (`order_book.py`),
stochastic latency models (`latency_models.py`), and the Cython tick core
(`_tick_core.*`) — lives on the **`experimental/tick-cython`** branch. That stack
does not depend on and is not depended on by the canonical engines, and it is not
built or imported on `main`.

The vectorized core `_core.*` is kept on `main`: it accelerates the canonical
indicator kernels (EMA/ATR/RSI) as well as the vectorized trade-chaining loop, and
degrades to pure Python when not compiled.

---

## 1. The Naive Backtester

The simplest possible backtester is a for-loop over OHLC bars:

```python
cash = 10_000
position = 0

for bar in bars:
    signal = strategy.decide(bar)
    if signal == "buy" and position == 0:
        position = cash / bar.close
        cash = 0
    elif signal == "sell" and position > 0:
        cash = position * bar.close
        position = 0

equity = cash + position * bars[-1].close
```

This works as a concept but is wrong in almost every way that matters for
producing reliable results.

### The core problem: decision price == fill price

The strategy sees `bar.close` to make its decision, then fills at `bar.close`.
This is **lookahead bias** at the bar level. In reality, by the time the close
price is known, the bar is over, you can't act on it until the next bar opens.
A strategy that fills at the decision price will always overstate its performance
because it captures the exact close-to-close move rather than the open-to-close
move of the next bar.

**Fix:** signal on bar N → fill at bar N+1's open. The strategy sees the close
and makes a decision; the execution happens at the next available price. This
single change, next-bar-open execution, is the most important correctness fix
in any backtester. Everything else is refinement.

### What "flat slippage" misses

Even after fixing fill timing, naive implementations apply a fixed percentage
to the fill price:

```python
fill_price = bar_open * (1 + side * slippage_bps / 10_000)
```

This is better than nothing but it's structurally wrong. Real slippage is not
proportional to price, it's a function of order size relative to available
liquidity, volatility at the moment of submission, and position in the queue.
A 5 bps slippage assumption on a $100 stock during a low-volume open is very
different from 5 bps during a news spike. The flat model is a useful sanity
check but it doesn't tell you what would happen in practice.

---

## 2. The Broker Layer: Gap-Aware Execution

Once next-bar execution is in place, the next problem is **stop-loss execution**.

A naive implementation does this:

```python
if position > 0 and bar.low <= stop_price:
    exit_price = stop_price  # fill at stop
```

This looks right but is wrong for the case where the bar **opens below the stop**
(a gap). If Friday closes at 100, the stop is at 98, and Monday opens at 95, the
naive implementation fills at 98, a price that never existed. The actual fill
would be at the open (95), 300 bps worse than expected.

The `Broker.close_due_to_stop()` implementation handles this explicitly:

```python
# broker.py
if tr.side > 0:
    if bar.open <= tr.stop_price:
        exit_raw = bar.open       # gapped through stop, fill at open
    elif bar.low <= tr.stop_price <= bar.high:
        exit_raw = tr.stop_price  # normal stop, fill at stop price
```

Two cases, two different outcomes. This distinction matters on daily bars for
assets that gap (equities, commodities) and is irrelevant for assets that
trade nearly 24 hours (spot FX, crypto). For daily gold data, gaps are common
enough that every strategy's drawdown numbers change materially when you
account for them correctly.

### Why a class, not a function

The broker is a class rather than a standalone function because it needs to
maintain **state across bars**: open positions, the list of closed trades,
and buying power. The gap-aware stop logic needs to read the current position's
stop price, which was set at entry. Putting all of this in a class with clear
ownership (positions, closed_trades) also makes the logic independently testable, you can unit test stop execution without running a full backtest.

---

## 3. The Portfolio Layer: Multi-Asset State

A single-asset backtester can track cash and equity as scalars. Once you add
multiple instruments, this breaks.

The `Portfolio` class owns three things the broker cannot:

**1. Equity computation across all open positions.**
Equity is `cash + sum(open_pnl for all positions across all symbols)`. The
broker only knows about the trades it has executed; the portfolio aggregates
across all brokers and computes the equity curve that strategies use for
position sizing.

**2. Buying power and leverage limits.**
`max_leverage` constrains total gross notional as a multiple of equity.
This is a portfolio-level constraint, not a per-trade one. A new trade must
check what all existing trades together already consume:

```python
# broker.py
def _remaining_buying_power(self, current_prices):
    equity = self.portfolio.compute_equity(current_prices)
    gross = self.portfolio.gross_notional(current_prices)
    limit = equity * self.portfolio.max_leverage
    return max(0.0, limit - gross)
```

Without this check, a strategy running on 6 assets simultaneously can
inadvertently lever up 6x because each trade only sees its own contribution.

**3. Margin calls.**
When equity falls below a fraction of the margin requirement, the portfolio
liquidates positions. This is not an edge case, during sharp drawdowns in
leveraged portfolios, margin calls happen and they happen at the worst possible
prices. Ignoring them produces equity curves that are physically impossible.

### The circular ownership design

`Portfolio` owns `Broker`, and `Broker` holds a reference back to `Portfolio`.
This is intentional. The broker needs portfolio-level state (cash, commission
rate, slippage rate, leverage limit) to execute trades. The portfolio needs
broker-level state (open positions, trade sizes) to compute equity. Rather than
pass these as arguments on every call, the circular reference keeps them in
sync. The broker never constructs a portfolio; the portfolio always constructs
the broker. Ownership is clear even if the reference is circular.

---

## 4. The Trade Type: Risk as a First-Class Object

A naive implementation might track a position as `{symbol: (size, entry_price)}`.
This is sufficient for a single open position per symbol but breaks immediately
when you want:

- **Multiple simultaneous trades** on the same symbol (scaling in, separate
  signals)
- **Per-trade stop and take-profit** prices that differ across trades
- **Trade-level P&L attribution** (which trade made money, not just total equity)
- **Bars-held tracking** for time-based position management

The `Trade` dataclass carries all of this:

```python
# types.py
@dataclass
class Trade:
    entry_bar: Bar
    side: int           # +1 long, -1 short
    size: float
    entry_price: float
    stop_price: float
    take_profit: float
    bars_held: int = 0
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
```

A trade is created at entry and mutated at exit. The broker maintains
`positions: Dict[str, Position]`, a `Position` per symbol that holds one or more
lots, not a scalar. This handles the multi-trade case correctly and makes
per-trade statistics (average hold time, win rate, average winner vs loser)
trivial to compute after the fact.

---

## 5. Experimental Execution-Realism Stack (separate branch)

`main` deliberately stops at the bar-level event engine. Three additional
questions are real but out of scope here, and are answered on the
**`experimental/tick-cython`** branch:

- **Intra-bar ambiguity.** OHLC bars cannot tell you whether a stop or a
  take-profit fired first when both sit inside the same bar's range. A tick
  engine resolves this by processing real price updates in order
  (`tick.py`, `tick_backtest.py`).
- **Order lifecycle.** A signal does not become a fill instantly: exchange
  acknowledgment and matching-engine queue time add latency, and limit/stop
  orders need conditional execution (`latency_broker.py`, `order_book.py`).
- **Latency as a distribution.** A single constant understates the tail;
  `latency_models.py` adds fixed, Gaussian, log-normal, and component models.

That stack is self-contained: it does not import the canonical bar engines'
private internals, and the canonical engines do not import it. It also carries
its own Cython core (`_tick_core.*`) and its own test suite, none of which are
built or collected on `main`. See the branch for those modules, tests,
examples, and benchmarks.

---

## 6. Design Decisions in Retrospect

**Event-driven vs vectorized:** The bar event-driven loop (`Backtester`) and
vectorized engine (`vectorized.py`) both generate signals from a completed bar
and fill at the next bar's open. The vectorized engine is faster but requires
careful index alignment to avoid accidentally using future data. They share
fill timing, but strategy state and position management can still differ; the
cross-engine tests check broad consistency rather than identical results.

**What the vectorized engine does *not* model** (use `Backtester` /
`PortfolioBacktester` instead): it holds only **one sequential position at a
time**; it supports fixed stop/take-profit prices and `risk_per_trade` sizing
with drawdown scaling; it applies commission + slippage in bps. It has no
funding/swap, no market impact, no contract multipliers, no multi-asset or
portfolio accounting, no trailing stops, and no partial fills or order-book /
limit-order mechanics. It is a screening/optimizer fast path, not an execution
model.

**Strategies return a `Trade`:** The strategy interface returns a `Trade`
object, a fully specified position with entry price, stop, and size already
calculated. This design keeps strategies simple: they compute what they want,
and the backtester handles the rest. Execution-realism extensions that would
return order objects live on the experimental branch and are not part of the
canonical interface.

**Run manifests:** The research layer owns experiment manifests; the engine should expose resolved run inputs and data identity for reproducibility.
