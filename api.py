"""FastAPI backend for running backtests via HTTP."""

import hashlib
import json

from datetime import date
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backtesting.backtest import Backtester
from strategy_registry import STRATEGY_REGISTRY, get_strategy_param_space
from backtesting.statistics import compute_sharpe
from utils import fetch_ohlc, sanitize

app = FastAPI(title="Backtesting API")

INSTRUMENTS = ["XAUUSD"]
TIMEFRAMES = ["M1", "M5", "M15", "H1", "H4", "D1"]


class BacktestRequest(BaseModel):
    strategy: str
    params: Dict[str, Any]
    instrument: str
    timeframe: str
    start: date
    end: date
    trend_tf: str = None
    trial_number: int
    starting_cash: int


class InstrumentsResponse(BaseModel):
    instruments: List[str]


class TimeframesResponse(BaseModel):
    timeframes: List[str]


class ParamSpaceResponse(BaseModel):
    strategy: str
    params: Dict[str, Dict[str, Any]]


def hash_params(params: dict) -> str:
    """Generate a deterministic hash for a parameter set (useful for caching)."""
    m = hashlib.sha256()
    m.update(json.dumps(params, sort_keys=True).encode())
    return m.hexdigest()


@app.get("/instruments", response_model=InstrumentsResponse)
def get_instruments():
    """Return the list of available instruments."""
    return {"instruments": INSTRUMENTS}


@app.get("/timeframes", response_model=TimeframesResponse)
def get_timeframes(instrument: str):
    """Return the available timeframes for a given instrument."""
    if instrument not in INSTRUMENTS:
        raise HTTPException(status_code=404, detail="Instrument not found")
    return {"timeframes": TIMEFRAMES}


@app.get("/param_space/{strategy_name}", response_model=ParamSpaceResponse)
def param_space(strategy_name: str):
    """Return the parameter space (names, types, defaults) for a strategy."""
    strategy_cls = STRATEGY_REGISTRY.get(strategy_name)
    if not strategy_cls:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return {"strategy": strategy_name, "params": get_strategy_param_space(strategy_cls)}


@app.post("/backtest/run")
def run_backtest(req: BacktestRequest):
    """Execute a backtest and return metrics, equity curve, and trade list."""
    strategy_cls = STRATEGY_REGISTRY.get(req.strategy)
    starting_cash = req.starting_cash if req.starting_cash > 0 else 10_000

    if not strategy_cls:
        raise HTTPException(status_code=404, detail="Strategy not found")

    df = fetch_ohlc(req.instrument, req.timeframe, req.start.isoformat(), req.end.isoformat())
    if df.empty:
        raise HTTPException(status_code=400, detail="No data for requested range")

    df = df.set_index(pd.to_datetime(df["timestamp"])).sort_index()

    strategy = strategy_cls(**req.params)

    bt = Backtester(df=df, strategy=strategy, starting_cash=starting_cash)
    equity_curve, trades = bt.run()

    equity_df = pd.DataFrame({
        "ts": df.index[:len(equity_curve)],
        "equity": equity_curve,
    })

    metrics = {
        "final_equity": float(equity_curve[-1]),
        "pct_return": (equity_curve[-1] - starting_cash) / starting_cash * 100,
        "trades": len(trades),
        "sharpe": compute_sharpe(pd.Series(equity_curve)),
    }

    trades_list = [
        {
            "entry_time": str(t.entry_bar.ts),
            "entry_price": float(t.entry_price),
            "exit_price": float(t.exit_price) if t.exit_price else None,
            "side": int(t.side),
            "size": float(t.size),
            "pnl": float(t.pnl) if t.pnl else None,
        }
        for t in trades
    ]

    return sanitize({
        "metrics": metrics,
        "equity_curve": equity_df.to_dict(orient="records"),
        "trades": trades_list,
    })
