#!/usr/bin/env python3
"""
Order Book Demo
---------------
Shows the MatchingEngine in action: FIFO queue priority, partial fills,
and the effect of max_qty_per_level on thin vs deep markets.

Three scenarios, all using the same tick stream:

  Scenario A -- Unlimited depth (max_qty_per_level=inf)
    Full orders fill instantly when price crosses the limit.
    Baseline: represents a perfectly deep, liquid market.

  Scenario B -- Thin market (max_qty_per_level=0.5)
    Only 0.5 units fill per price level per tick. Large orders split
    across multiple ticks. Shows partial fill mechanics.

  Scenario C -- FIFO priority
    Two orders at the same limit price. The first submitted always fills
    before the second, even if the second is smaller.

Uses real tick data from the local datalake (localhost:8000/ticks) when
available, falling back to synthetic data if the server is not reachable.

Usage:
    python examples/order_book_demo.py
    python examples/order_book_demo.py --instrument EURUSD --n 2000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.order import Order, OrderType
from backtesting.order_book import Fill, MatchingEngine, OrderBook
from backtesting.tick import Tick


# ---------------------------------------------------------------------------
# Tick data helpers (shared with tick_latency_demo)
# ---------------------------------------------------------------------------

DATALAKE_URL = "http://localhost:8000"


def fetch_ticks(instrument: str = "XAUUSD", n: int = 2_000) -> list[Tick] | None:
    ticks: list[Tick] = []
    cursor = None
    while len(ticks) < n:
        batch = min(10_000, n - len(ticks))
        url = f"{DATALAKE_URL}/ticks?instrument={instrument}&limit={batch}"
        if cursor:
            url += f"&cursor={cursor}"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                payload = json.loads(resp.read())
        except Exception:
            return None
        for row in payload["data"]:
            ticks.append(Tick(
                ts=pd.Timestamp(row["timestamp"]),
                price=float(row["price"]),
                volume=float(row.get("volume") or 0.0),
                bid=float(row["bid"]) if row.get("bid") is not None else None,
                ask=float(row["ask"]) if row.get("ask") is not None else None,
            ))
        if not payload["pagination"]["has_more"]:
            break
        cursor = payload["pagination"]["next_cursor"]
    return ticks


def generate_ticks(
    n: int = 2_000,
    start: str = "2024-01-02 09:00:00",
    tick_ms: int = 100,
    base_price: float = 100.0,
    spread: float = 0.05,
    seed: int = 42,
) -> list[Tick]:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp(start)
    t = np.arange(n)
    prices = base_price + 2.0 * np.sin(2 * np.pi * t / 200) + rng.normal(0, 0.1, n)
    prices = np.maximum(prices, 1.0)
    return [
        Tick(
            ts=base + pd.Timedelta(milliseconds=i * tick_ms),
            price=float(prices[i]),
            volume=float(rng.integers(1, 20)),
            bid=float(prices[i] - spread / 2),
            ask=float(prices[i] + spread / 2),
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Scenario A & B: deep vs thin market with limit orders
# ---------------------------------------------------------------------------

def _make_order(i: int, side: int, limit: float, qty: float, symbol: str, ts) -> Order:
    return Order(
        type=OrderType.LIMIT,
        symbol=symbol,
        side=side,
        qty=qty,
        protective_stop=0.0,
        take_profit=None,
        submitted_at=ts,
        limit_price=limit,
        order_id=f"ord-{i:04d}",
    )


def run_depth_comparison(
    ticks: list[Tick],
    symbol: str,
    n_orders: int = 20,
    order_qty: float = 1.0,
) -> dict:
    """
    Submit `n_orders` limit buy orders spread across the tick range,
    with limit prices set just above the current ask (so they fill quickly).
    Compare fills and timing between unlimited and thin depth.
    """
    mid = ticks[0].price
    limit_price = mid * 1.002  # just above current price so they rest briefly

    results = {}
    for label, max_qty in [("deep (inf)", float("inf")), ("thin (0.5)", 0.5)]:
        book = OrderBook()
        engine = MatchingEngine(book, max_qty_per_level=max_qty)

        all_fills: list[Fill] = []
        submitted: list[tuple[str, float]] = []  # (order_id, submit_ns)

        # Space orders evenly across the first half of the tick stream
        submit_indices = sorted(set(
            int(i * len(ticks) / (2 * n_orders)) for i in range(n_orders)
        ))

        order_idx = 0
        for t_idx, tick in enumerate(ticks):
            # Submit orders at scheduled ticks
            if order_idx < len(submit_indices) and t_idx == submit_indices[order_idx]:
                order = _make_order(
                    i=order_idx,
                    side=1,
                    limit=limit_price,
                    qty=order_qty,
                    symbol=symbol,
                    ts=tick.ts,
                )
                fills = engine.submit(order, tick.ts)
                submitted.append((order.order_id, tick.ts.value))
                all_fills.extend(fills)
                order_idx += 1

            # Advance book
            tick_fills = engine.process_tick(tick)
            all_fills.extend(tick_fills)

        n_submitted = min(n_orders, len(submit_indices))
        total_requested = n_submitted * order_qty
        total_filled = sum(f.qty for f in all_fills)
        fill_rate = total_filled / total_requested if total_requested > 0 else 0.0

        # Track fills at the order level (not the fill-event level)
        filled_per_order: dict = {}
        for f in all_fills:
            filled_per_order[f.order_id] = filled_per_order.get(f.order_id, 0.0) + f.qty
        n_fully_filled = sum(1 for qty in filled_per_order.values() if qty >= order_qty - 1e-9)
        n_partial_orders = n_submitted - n_fully_filled

        results[label] = {
            "n_submitted": n_submitted,
            "total_requested": total_requested,
            "total_filled": total_filled,
            "fill_rate": fill_rate,
            "fill_count": len(all_fills),
            "n_fully_filled": n_fully_filled,
            "n_partial_orders": n_partial_orders,
        }

    return results


# ---------------------------------------------------------------------------
# Scenario C: FIFO priority
# ---------------------------------------------------------------------------

def run_fifo_demo(ticks: list[Tick], symbol: str) -> dict:
    """
    Submit two orders at the same limit price, first a large one then a small one.
    With max_qty_per_level=1.0 per tick, the first order should fill before the second.
    """
    mid = ticks[0].price
    limit = mid * 1.001  # slightly above mid so orders rest for a few ticks

    book = OrderBook()
    engine = MatchingEngine(book, max_qty_per_level=1.0)

    submit_tick = ticks[5]  # submit both at tick 5 so they're in the queue together

    order_a = Order(
        type=OrderType.LIMIT, symbol=symbol, side=1,
        qty=2.0, protective_stop=0.0, take_profit=None,
        submitted_at=submit_tick.ts, limit_price=limit,
        order_id="fifo-A",
    )
    order_b = Order(
        type=OrderType.LIMIT, symbol=symbol, side=1,
        qty=0.5, protective_stop=0.0, take_profit=None,
        submitted_at=submit_tick.ts, limit_price=limit,
        order_id="fifo-B",
    )

    # Submit A first, then B (same tick)
    engine.submit(order_a, submit_tick.ts)
    engine.submit(order_b, submit_tick.ts)

    fill_sequence: list[tuple[str, float, pd.Timestamp]] = []

    for tick in ticks:
        fills = engine.process_tick(tick)
        for f in fills:
            fill_sequence.append((f.order_id, f.qty, f.ts))

    fills_a = [(qty, ts) for oid, qty, ts in fill_sequence if oid == "fifo-A"]
    fills_b = [(qty, ts) for oid, qty, ts in fill_sequence if oid == "fifo-B"]

    first_fill_a = fills_a[0][1] if fills_a else None
    first_fill_b = fills_b[0][1] if fills_b else None

    fifo_respected = (
        first_fill_a is not None and first_fill_b is not None
        and first_fill_a <= first_fill_b
    )

    return {
        "order_a_fills": fills_a,
        "order_b_fills": fills_b,
        "first_fill_a": first_fill_a,
        "first_fill_b": first_fill_b,
        "fifo_respected": fifo_respected,
        "total_a": sum(qty for qty, _ in fills_a),
        "total_b": sum(qty for qty, _ in fills_b),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Order book demo")
    parser.add_argument("--instrument", default="XAUUSD")
    parser.add_argument("--n", type=int, default=2_000)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    ticks = None
    if not args.synthetic:
        print(f"Fetching {args.n} ticks for {args.instrument} from datalake...")
        ticks = fetch_ticks(args.instrument, args.n)
        if ticks:
            print(f"  {len(ticks)} ticks | "
                  f"{ticks[0].ts.strftime('%Y-%m-%d %H:%M:%S')} to "
                  f"{ticks[-1].ts.strftime('%Y-%m-%d %H:%M:%S')}\n")
        else:
            print("  Datalake unreachable -- falling back to synthetic ticks.\n")

    if not ticks:
        print(f"Generating {args.n} synthetic ticks...")
        ticks = generate_ticks(n=args.n)
        print(f"  {len(ticks)} ticks | "
              f"{ticks[0].ts.strftime('%H:%M:%S')} to "
              f"{ticks[-1].ts.strftime('%H:%M:%S')}\n")

    symbol = args.instrument

    # ------------------------------------------------------------------
    # Scenario A & B: deep vs thin
    # ------------------------------------------------------------------
    print("Running Scenarios A & B: deep vs thin market depth...")
    depth = run_depth_comparison(ticks, symbol=symbol, n_orders=20, order_qty=1.0)

    col = 30
    d = depth["deep (inf)"]
    t = depth["thin (0.5)"]
    print()
    print("=" * 60)
    print(f"{'Order book depth comparison':^60}")
    print("=" * 60)
    print(f"{'':>{col}} {'Deep (inf)':>14} {'Thin (0.5)':>14}")
    print("-" * 60)
    print(f"{'Orders submitted':<{col}} {d['n_submitted']:>14d} {t['n_submitted']:>14d}")
    print(f"{'Total qty requested':<{col}} {d['total_requested']:>14.2f} {t['total_requested']:>14.2f}")
    print(f"{'Total qty filled':<{col}} {d['total_filled']:>14.2f} {t['total_filled']:>14.2f}")
    print(f"{'Fill rate':<{col}} {d['fill_rate']*100:>13.1f}% {t['fill_rate']*100:>13.1f}%")
    print("-" * 60)
    print(f"{'Orders fully filled':<{col}} {d['n_fully_filled']:>14d} {t['n_fully_filled']:>14d}")
    print(f"{'Orders partially filled (open)':<{col}} {d['n_partial_orders']:>14d} {t['n_partial_orders']:>14d}")
    print(f"{'Fill events (broker callbacks)':<{col}} {d['fill_count']:>14d} {t['fill_count']:>14d}")
    print("=" * 60)
    print()
    print("  Deep market:  each order fills in one shot on the first eligible tick.")
    print(f"  Thin market:  max 0.5 units per tick, so a qty=1.0 order needs 2 ticks")
    print(f"                to fully fill -> 2x fill events, same total qty.")

    # ------------------------------------------------------------------
    # Scenario C: FIFO
    # ------------------------------------------------------------------
    print()
    print("Running Scenario C: FIFO priority...")
    fifo = run_fifo_demo(ticks, symbol=symbol)

    print()
    print("=" * 60)
    print(f"{'FIFO queue priority':^60}")
    print("=" * 60)
    print(f"  Order A (submitted first, qty=2.0):")
    if fifo["order_a_fills"]:
        for qty, ts in fifo["order_a_fills"]:
            print(f"    fill qty={qty:.2f}  at  {ts}")
    else:
        print("    (no fills)")
    print(f"  Order B (submitted second, qty=0.5):")
    if fifo["order_b_fills"]:
        for qty, ts in fifo["order_b_fills"]:
            print(f"    fill qty={qty:.2f}  at  {ts}")
    else:
        print("    (no fills)")
    print()
    print(f"  First fill A: {fifo['first_fill_a']}")
    print(f"  First fill B: {fifo['first_fill_b']}")
    verdict = "PASS -- A filled before B" if fifo["fifo_respected"] else "FAIL"
    print(f"  FIFO respected: {verdict}")
    print("=" * 60)
    print()
    print("  Order A was submitted first so it drains first from the level.")
    print("  max_qty_per_level=1.0 means A needs 2 ticks to fully fill,")
    print("  while B only starts filling after A is complete.")


if __name__ == "__main__":
    main()
