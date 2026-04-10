#!/usr/bin/env python3
"""
Tick Backtest + Latency Broker Demo
------------------------------------
Shows the difference between:
  - Instant fill (old Broker, flat slippage bps)
  - Latency-aware fill (LatencyAwareBroker, MARKET order queued with delay)
  - Limit order fill (only executes when price crosses the limit level)

Uses real tick data from the local datalake (localhost:8000/ticks) when
available, falling back to synthetic data if the server is not reachable.

Usage:
    python examples/tick_latency_demo.py
    python examples/tick_latency_demo.py --instrument EURUSD --n 5000
"""

import argparse
import os
import sys
import urllib.request
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting import LatencyAwareBroker, Portfolio, TickBacktester
from backtesting.latency_metrics import compare_latency_impact
from backtesting.order import Order, OrderType
from backtesting.strategy import Strategy
from backtesting.tick import Tick
from backtesting.types import Bar, Trade


# ---------------------------------------------------------------------------
# Live tick data from datalake
# ---------------------------------------------------------------------------

DATALAKE_URL = "http://localhost:8000"


def fetch_ticks_from_api(instrument: str = "XAUUSD", n: int = 3_000) -> list[Tick] | None:
    """Fetch real ticks from the local datalake.

    Returns None if the server is unreachable or the instrument has no ticks.
    Paginates automatically until ``n`` ticks are collected.
    """
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


# ---------------------------------------------------------------------------
# Synthetic tick data (fallback)
# ---------------------------------------------------------------------------

def generate_ticks(
    n: int = 3_000,
    start: str = "2024-01-02 09:00:00",
    tick_ms: int = 100,
    base_price: float = 100.0,
    spread: float = 0.05,
    seed: int = 42,
) -> list[Tick]:
    """Generate a sine-wave price series with Gaussian noise and a bid/ask spread."""
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
# Strategies
# ---------------------------------------------------------------------------

class InstantFillStrategy(Strategy):
    """Simple momentum strategy using the legacy Trade return (instant fill)."""

    def __init__(self):
        self._prev_close = None

    def on_bar(self, i: int, bar: Bar, equity: float):
        if self._prev_close is None:
            self._prev_close = bar.close
            return None

        # Buy when price ticked up, sell when it ticked down
        if bar.close > self._prev_close:
            side = 1
        elif bar.close < self._prev_close:
            side = -1
        else:
            self._prev_close = bar.close
            return None

        self._prev_close = bar.close
        stop = bar.close * (0.97 if side > 0 else 1.03)
        size = max(equity * 0.1 / bar.close, 0.01)
        return Trade(
            entry_bar=bar, side=side, size=size,
            entry_price=bar.close, stop_price=stop, take_profit=None,
        )


class LatencyMarketStrategy(Strategy):
    """Same momentum logic but returns an Order -- goes through the latency queue."""

    def __init__(self, symbol: str = "default"):
        self.symbol = symbol
        self._prev_close = None

    def on_bar(self, i: int, bar: Bar, equity: float):
        if self._prev_close is None:
            self._prev_close = bar.close
            return None

        if bar.close > self._prev_close:
            side = 1
        elif bar.close < self._prev_close:
            side = -1
        else:
            self._prev_close = bar.close
            return None

        self._prev_close = bar.close
        stop = bar.close * (0.97 if side > 0 else 1.03)
        size = max(equity * 0.1 / bar.close, 0.01)
        return Order(
            type=OrderType.MARKET,
            symbol=self.symbol,
            side=side,
            qty=size,
            protective_stop=stop,
            take_profit=None,
            submitted_at=bar.ts,
        )


class LimitOrderStrategy(Strategy):
    """Places limit orders slightly inside the spread to get a better fill."""

    def __init__(self, symbol: str = "default", offset: float = 0.03):
        self.symbol = symbol
        self.offset = offset  # price improvement vs close
        self._prev_close = None

    def on_bar(self, i: int, bar: Bar, equity: float):
        if self._prev_close is None:
            self._prev_close = bar.close
            return None

        if bar.close > self._prev_close:
            side = 1
        elif bar.close < self._prev_close:
            side = -1
        else:
            self._prev_close = bar.close
            return None

        self._prev_close = bar.close
        # Limit price is slightly better than close -- may not fill immediately
        limit = bar.close - side * self.offset
        stop = bar.close * (0.97 if side > 0 else 1.03)
        size = max(equity * 0.1 / bar.close, 0.01)
        return Order(
            type=OrderType.LIMIT,
            symbol=self.symbol,
            side=side,
            qty=size,
            protective_stop=stop,
            take_profit=None,
            submitted_at=bar.ts,
            limit_price=limit,
        )


# ---------------------------------------------------------------------------
# Run & compare
# ---------------------------------------------------------------------------

def run_instant(ticks: list[Tick]) -> dict:
    bt = TickBacktester(
        ticks, InstantFillStrategy(),
        timeframe="M1", starting_cash=10_000,
        commission_bps=1.0, slippage_bps=2.0,
        max_leverage=5.0,
    )
    equity_curve, trades = bt.run()
    open_pos = bt.broker.positions.get("default", [])
    return {"equity_curve": equity_curve, "trades": trades,
            "final": equity_curve[-1], "open_pos": open_pos,
            "last_price": ticks[-1].price}


def run_latency(ticks: list[Tick], ack_ms: float = 50.0) -> dict:
    portfolio = Portfolio(
        cash=10_000, commission_bps=1.0, slippage_bps=2.0,
        max_leverage=5.0, margin_rate=0.0,
    )
    lb = LatencyAwareBroker(
        broker=portfolio.broker,
        ack_latency_ns=int(ack_ms * 1_000_000),
    )
    bt = TickBacktester(
        ticks, LatencyMarketStrategy(),
        timeframe="M1", starting_cash=10_000,
        commission_bps=1.0, slippage_bps=2.0,
        max_leverage=5.0,
        latency_broker=lb,
    )
    bt.portfolio = portfolio
    bt.broker = portfolio.broker
    equity_curve, trades = bt.run()
    open_pos = portfolio.broker.positions.get("default", [])
    return {
        "equity_curve": equity_curve,
        "trades": lb.closed_trades,
        "final": equity_curve[-1],
        "pending_at_end": lb.pending_count,
        "open_pos": open_pos,
        "last_price": ticks[-1].price,
    }


def run_limit(ticks: list[Tick], ack_ms: float = 50.0, offset: float = 0.03) -> dict:
    portfolio = Portfolio(
        cash=10_000, commission_bps=1.0, slippage_bps=0.0,
        max_leverage=5.0, margin_rate=0.0,
    )
    lb = LatencyAwareBroker(
        broker=portfolio.broker,
        ack_latency_ns=int(ack_ms * 1_000_000),
    )
    bt = TickBacktester(
        ticks, LimitOrderStrategy(offset=offset),
        timeframe="M1", starting_cash=10_000,
        commission_bps=1.0, slippage_bps=0.0,
        max_leverage=5.0,
        latency_broker=lb,
    )
    bt.portfolio = portfolio
    bt.broker = portfolio.broker
    equity_curve, trades = bt.run()
    open_pos = portfolio.broker.positions.get("default", [])
    return {
        "equity_curve": equity_curve,
        "trades": lb.closed_trades,
        "final": equity_curve[-1],
        "pending_at_end": lb.pending_count,
        "open_pos": open_pos,
        "last_price": ticks[-1].price,
    }


def unrealized_pnl(open_pos: list, last_price: float) -> float:
    return sum((last_price - tr.entry_price) * tr.side * tr.size for tr in open_pos)


def main():
    parser = argparse.ArgumentParser(description="Tick backtest latency demo")
    parser.add_argument("--instrument", default="XAUUSD", help="Instrument to fetch (default: XAUUSD)")
    parser.add_argument("--n", type=int, default=3_000, help="Number of ticks (default: 3000)")
    parser.add_argument("--latency-ms", type=float, default=50.0, dest="latency_ms",
                        help="Ack latency in milliseconds for the latency runs (default: 50)")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data")
    args = parser.parse_args()

    ticks = None
    if not args.synthetic:
        print(f"Fetching {args.n} ticks for {args.instrument} from datalake...")
        ticks = fetch_ticks_from_api(args.instrument, args.n)
        if ticks:
            print(f"  {len(ticks)} ticks | "
                  f"{ticks[0].ts.strftime('%Y-%m-%d %H:%M:%S')} to "
                  f"{ticks[-1].ts.strftime('%Y-%m-%d %H:%M:%S')}\n")
        else:
            print("  Datalake unreachable or no data -- falling back to synthetic ticks.\n")

    if not ticks:
        print(f"Generating {args.n} synthetic ticks...")
        ticks = generate_ticks(n=args.n, tick_ms=100)
        print(f"  {len(ticks)} ticks | "
              f"{ticks[0].ts.strftime('%H:%M:%S')} to {ticks[-1].ts.strftime('%H:%M:%S')}\n")

    ack_ms = args.latency_ms

    print("Running instant-fill backtest (legacy Trade path)...")
    r_instant = run_instant(ticks)

    print(f"Running latency-aware backtest ({ack_ms:.0f}ms ack delay, MARKET orders)...")
    r_latency = run_latency(ticks, ack_ms=ack_ms)

    print(f"Running limit-order backtest ({ack_ms:.0f}ms delay, LIMIT 3 pips inside close)...")
    r_limit = run_limit(ticks, ack_ms=ack_ms, offset=0.03)

    n_open_i   = len(r_instant["open_pos"])
    n_open_l   = len(r_latency["open_pos"])
    n_open_lim = len(r_limit["open_pos"])

    upnl_i   = unrealized_pnl(r_instant["open_pos"],   r_instant["last_price"])
    upnl_l   = unrealized_pnl(r_latency["open_pos"],   r_latency["last_price"])
    upnl_lim = unrealized_pnl(r_limit["open_pos"],     r_limit["last_price"])

    total_i   = r_instant["final"] + upnl_i
    total_l   = r_latency["final"] + upnl_l
    total_lim = r_limit["final"]   + upnl_lim

    col = 32
    print()
    print("=" * 56)
    print(f"{'':>{col}} {'Instant':>8} {'Latency':>8} {'Limit':>8}")
    print("=" * 56)
    print(f"{'Cash + closed P&L':<{col}} {r_instant['final']:8.2f} {r_latency['final']:8.2f} {r_limit['final']:8.2f}")
    print(f"{'Unrealized P&L (open pos)':<{col}} {upnl_i:8.2f} {upnl_l:8.2f} {upnl_lim:8.2f}")
    print(f"{'Total equity (incl. open)':<{col}} {total_i:8.2f} {total_l:8.2f} {total_lim:8.2f}")
    print(f"{'Total return %':<{col}} "
          f"{(total_i/10_000-1)*100:7.2f}% "
          f"{(total_l/10_000-1)*100:7.2f}% "
          f"{(total_lim/10_000-1)*100:7.2f}%")
    print(f"{'Closed trades':<{col}} {len(r_instant['trades']):8d} {len(r_latency['trades']):8d} {len(r_limit['trades']):8d}")
    print(f"{'Open positions at end':<{col}} {n_open_i:8d} {n_open_l:8d} {n_open_lim:8d}")
    print(f"{'Orders still pending':<{col}} {'n/a':>8} {r_latency['pending_at_end']:8d} {r_limit['pending_at_end']:8d}")
    print("=" * 56)

    impact = total_l - total_i
    sign = "+" if impact >= 0 else ""
    print(f"\nLatency drag vs instant fill: {sign}{impact:.2f} ({sign}{(impact/10_000)*100:.3f}%)")

    # ------------------------------------------------------------------
    # Latency impact metrics
    # ------------------------------------------------------------------
    print(f"\nRunning latency impact analysis (zero-latency vs {ack_ms:.0f}ms)...")
    result = compare_latency_impact(
        ticks=ticks,
        strategy_factory=lambda: LatencyMarketStrategy(symbol="default"),
        ack_latency_ns=int(ack_ms * 1_000_000),
        starting_cash=10_000,
        commission_bps=1.0,
        slippage_bps=2.0,
        max_leverage=5.0,
        symbol="default",
        timeframe="M1",
    )
    result.print_summary()


if __name__ == "__main__":
    main()
