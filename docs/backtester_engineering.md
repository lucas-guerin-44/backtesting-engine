# Backtester Engineering: From Naive Loop to Execution Layer

This document covers the engineering decisions behind this backtesting engine,
why the abstractions exist, what problems each layer solves, and where each layer
still falls short. It is not about trading strategy or quant methodology. For that,
see `research.md`.

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
# broker.py:153-166
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
# broker.py:37-41
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
# types.py:35-45
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
`positions: Dict[str, List[Trade]]`, a list per symbol, not a scalar. This
handles the multi-trade case correctly and makes per-trade statistics
(average hold time, win rate, average winner vs loser) trivial to compute
after the fact.

---

## 5. Tick-Level Backtesting: The Bar Granularity Problem

Even with a correct bar-level backtester, OHLC bars lose information.

Consider a bar where `low=95, high=105, stop=98, take_profit=103`. Both the
stop and the take-profit are inside the bar's range. Which one fired? The OHLC
representation can't tell you, you only know the bar's low and high, not the
order in which those extremes were reached.

The standard bar-level approach assumes a fixed priority (usually stop before
TP on down moves) and documents that assumption. It's an approximation.

**The tick backtester eliminates this ambiguity.** Each tick is a real price
update. The stop and TP are checked against every single tick price, so the
order they fire is determined by the actual sequence of price updates. There is
no ambiguity about which triggered first because you process them chronologically.

The tick-level approach also changes **fill timing semantics**. At the bar level,
a signal fires and fills at the next bar's open, a delay of one bar (minutes to
days, depending on timeframe). At the tick level, a signal fires at tick N and
fills at tick N+1, a delay measured in milliseconds. This is closer to how
live execution actually works.

### Implementation note: pre-extracting tick arrays

The hot loop in `TickBacktester.run()` processes tens of thousands of ticks.
Every attribute lookup in a Python loop has overhead. The tick backtester
pre-extracts prices and timestamps into numpy arrays at init:

```python
# tick_backtest.py:137-140
self._prices = np.array([t.price for t in ticks], dtype=np.float64)
self._timestamps = [t.ts for t in ticks]
self._bar_boundaries = np.array(
    [t.ts.floor(freq).value for t in ticks], dtype=np.int64
)
```

Bar boundary detection in the hot loop becomes an integer comparison
(`tick_boundary != current_bar_boundary`) rather than a floor() call per tick.
This is approximately 50x faster per tick for known timeframes.

The `TickAggregator` similarly uses integer nanosecond arithmetic instead of
pandas `Timestamp.floor()`:

```python
# tick.py:131-133
if self._freq_ns is not None:
    ns = ts.value
    floored_ns = (ns // self._freq_ns) * self._freq_ns
```

These are not premature optimizations, the difference between a 30-second
backtest and a 3-second one determines whether iterative strategy development
is practical.

---

## 6. What the Tick Backtester Still Gets Wrong

Even with tick-level data, the `Broker` fills orders with two assumptions that
don't hold in practice:

**Assumption 1: Instant acknowledgment.**
A strategy signal at tick N fills at tick N+1. There is no model of the time
between submission and exchange acknowledgment. In reality, even a co-located
server has a round-trip latency of microseconds to milliseconds; a remote server
has tens of milliseconds. During that window, the price moves. The fill you get
is not the price you saw when you decided to trade.

**Assumption 2: Orders are market orders with no order type.**
The original broker has no concept of limit orders, stop orders, or any
conditional execution. Every entry is "fill me at approximately the current
price, adjusted for slippage bps." A strategy that wants to buy on a breakout
at exactly 102.50 cannot express that, it can only say "buy now, at whatever
the next tick's price is."

**Assumption 3: Flat slippage regardless of spread.**
The broker applies `slippage_bps` symmetrically to every fill. It doesn't
distinguish between buying (where you pay the ask) and selling (where you
receive the bid). On instruments with a real bid/ask spread, this understates
slippage for trades that cross the spread and overstates it for others.

---

## 7. The LatencyAwareBroker: Modeling Order Lifecycle

The `LatencyAwareBroker` wraps the existing `Broker` and adds two things the
original broker cannot express: **time** between submission and fill, and
**order types** with conditional execution.

### Order lifecycle

An order submitted at time T with `ack_latency_ns=10ms` and `fill_latency_ns=5ms`
is not eligible to fill until `T + 15ms`. Every tick processed before that
timestamp is scanned and skipped. This models the round-trip to the exchange
(ack) and the queue time at the matching engine (fill).

```python
# latency_broker.py:98-100
def submit(self, order: Order, ts: pd.Timestamp) -> None:
    fill_after_ns = ts.value + self.ack_latency_ns + self.fill_latency_ns
    self._pending.append(PendingOrder(order=order, fill_after_ns=fill_after_ns))
```

The queue is processed on every tick via `process_tick()`, which the
`TickBacktester` calls before the strategy signal block. Orders that haven't
matured stay in the queue. Orders that have matured are matched against the
current tick's price.

### Order types

**MARKET:** Fill at `tick.ask` (buy) or `tick.bid` (sell). When bid/ask are
present in the tick data (as they are in the datalake), this correctly prices
the spread. A buy order costs more than a sell order returns, by exactly the
spread, without needing to specify slippage bps.

**LIMIT:** Fill only if the relevant side of the market has crossed the limit
level. A limit buy at 100 does not fill if the ask is 101. It stays in the
queue until the ask drops to 100, at which point it fills at the limit price,
getting price improvement relative to a market order.

**STOP:** Inert until price crosses the `stop_trigger`. Once triggered, it
converts to a market order and fills on the next eligible tick. This separates
the activation event from the fill event, which matters when the trigger happens
in the middle of a fast-moving price sequence.

### Drop-in interface

`LatencyAwareBroker` exposes `positions` and `closed_trades` as pass-throughs
to the inner broker. Strategy code and the tick backtester's stop/TP logic
read these directly, they don't need to know whether they're talking to a
`Broker` or a `LatencyAwareBroker`. The latency layer is opt-in: passing
`latency_broker=None` to `TickBacktester` (the default) preserves the original
behavior exactly.

### What the latency layer still doesn't model

- **Queue position.** If 1000 limit buy orders are resting at price 100, your
  order is at the back of the queue. You won't fill until all 1000 ahead of you
  have been matched. The latency layer has no concept of queue depth or FIFO
  position. This is what the order book (Phase 2) addresses.

- **Partial fills.** A limit buy for 100 units at price 100 will fill fully if
  100+ units are available. If only 40 units trade at that price, you get a
  partial fill and the rest remains in the queue. The latency layer is all-or-
  nothing: the full size fills the moment the price condition is met.

- **Market impact.** Large orders move the price. A market order for 10,000
  units doesn't fill entirely at the best ask, it sweeps through multiple price
  levels, with each additional unit getting a worse price. The latency layer
  treats every order as price-taking.

- **Latency jitter.** Real network latency is not a fixed value, it's a
  distribution with variance, tail events, and occasional spikes. The latency
  layer uses a constant `ack_latency_ns`. A more realistic model would sample
  from a distribution on each submission.

---

## 8. Design Decisions in Retrospect

**Event-driven vs vectorized:** The engine has both an event-driven loop
(`Backtester`, `TickBacktester`) and a vectorized engine (`vectorized.py`). The
event-driven loop is correct by construction, it processes one bar at a time
with no access to future data, which prevents lookahead by design. The vectorized
engine is faster but requires careful index alignment to avoid accidentally using
a future value. Both are tested against each other (`test_cross_engine.py`) to
verify they produce identical results for the same strategy and data.

**`_fill_at_tick_fast()` as inlined broker logic:** The tick backtester's hot
loop does not call `broker.open_trade()` for fills. It inlines the equivalent
logic directly in the loop body. This is a deliberate performance trade-off: the
method call + dictionary lookup overhead on every tick was measurable (~15% of
loop time). The cost is that the fast-fill logic is duplicated from the broker.
Tests in `test_cross_engine.py` catch any divergence between the two paths.

**Strategies return `Trade | Order`:** The original strategy interface returns
a `Trade` object, a fully specified position with entry price, stop, and size
already calculated. This design keeps strategies simple: they compute what they
want, and the backtester handles the rest. The latency extension adds `Order` as
an alternative return type. Returning an `Order` routes the signal through the
latency queue; returning a `Trade` uses the original instant-fill path. Both are
valid and both are tested. Existing strategies require no changes.
