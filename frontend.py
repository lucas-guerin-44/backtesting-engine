"""Streamlit dashboard for interactive backtesting."""

import ast
import inspect
import os
from datetime import datetime

import altair as alt
import pandas as pd
import requests
import streamlit as st

from backtesting.backtest import Backtester
from config import DATALAKE_URL
from strategy_registry import STRATEGY_REGISTRY
from utils import fetch_ohlc


def get_strategy_params(strategy_cls):
    """Extract parameter names and defaults from a strategy's __init__."""
    sig = inspect.signature(strategy_cls.__init__)
    params = {}
    for name, param in sig.parameters.items():
        if name == "self" or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        params[name] = param.default
    return params


def render_params_form(defaults: dict, strategy_name: str):
    """Render Streamlit sidebar widgets for each strategy parameter."""
    values = {}
    for key, default in defaults.items():
        widget_key = f"{strategy_name}_{key}"

        if widget_key not in st.session_state:
            if isinstance(default, (list, tuple)):
                st.session_state[widget_key] = str(default)
            else:
                st.session_state[widget_key] = default

        val = st.session_state[widget_key]
        if isinstance(default, bool):
            values[key] = st.sidebar.checkbox(key, value=bool(val), key=widget_key)
        elif isinstance(default, int):
            values[key] = st.sidebar.number_input(key, value=int(val), step=1, key=widget_key)
        elif isinstance(default, float):
            values[key] = st.sidebar.number_input(key, value=float(val), format="%.4f", key=widget_key)
        else:
            if isinstance(val, (list, tuple)):
                val = str(val)
            values[key] = st.sidebar.text_input(key, value=val, key=widget_key)

    return values


@st.cache_data
def get_instruments():
    """Fetch available instruments from the datalake API."""
    r = requests.get(f"{DATALAKE_URL}/instruments")
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        if "instruments" in data:
            return data["instruments"]
        return list(data.keys())
    elif isinstance(data, list):
        return data
    return []


@st.cache_data
def get_timeframes(instrument):
    """Fetch available timeframes for an instrument from the datalake API."""
    r = requests.get(f"{DATALAKE_URL}/timeframes", params={"instrument": instrument})
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        if "timeframes" in data:
            return data["timeframes"]
        return list(data.keys())
    elif isinstance(data, list):
        return data
    return []


def run_one(strategy_cls, param_set, df):
    """Run a single backtest and return results summary + Backtester instance."""
    strategy = strategy_cls(**param_set)

    bt = Backtester(df=df, strategy=strategy, starting_cash=10_000)
    equity_curve, trades = bt.run()

    if len(equity_curve) == 0:
        return None, bt

    start_eq = equity_curve[0]
    end_eq = equity_curve[-1]
    pct_return = (end_eq - start_eq) / start_eq * 100

    trade_stats = {
        "total_trades": len(trades),
        "winning_trades": len([t for t in trades if t.pnl and t.pnl > 0]),
        "losing_trades": len([t for t in trades if t.pnl and t.pnl < 0]),
    }
    if trade_stats["total_trades"] > 0:
        trade_stats["win_rate"] = round(trade_stats["winning_trades"] / trade_stats["total_trades"] * 100, 2)
    else:
        trade_stats["win_rate"] = 0.0

    results = {
        **param_set,
        "final_equity": end_eq,
        "pct_return": round(pct_return, 2),
        **trade_stats,
    }

    return results, bt


def main():
    st.title("Interactive Backtesting Dashboard")

    st.sidebar.header("Config")

    strategy_name = st.sidebar.selectbox("Strategy", list(STRATEGY_REGISTRY.keys()))
    strategy_cls = STRATEGY_REGISTRY[strategy_name]
    default_params = get_strategy_params(strategy_cls)
    params = render_params_form(default_params, strategy_name=strategy_name)

    instruments = get_instruments()
    if not instruments:
        st.error("No instruments returned by API")
        return
    instrument = st.sidebar.selectbox("Instrument", instruments)

    timeframes = get_timeframes(instrument)
    if not timeframes:
        st.error("No timeframes returned for this instrument")
        return
    timeframe = st.sidebar.selectbox("Timeframe", timeframes)

    start_date = st.sidebar.date_input("Start Date", datetime.fromisoformat("2024-01-01"))
    end_date = st.sidebar.date_input("End Date", datetime.fromisoformat("2025-08-01"))

    if st.sidebar.button("Run Backtest"):
        st.write(f"Fetching data for **{instrument} {timeframe}**...")
        df = fetch_ohlc(
            instrument,
            timeframe,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )
        if df.empty:
            st.error("No data returned.")
            return

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()

        for k, v in params.items():
            if isinstance(default_params[k], (list, tuple)):
                try:
                    params[k] = ast.literal_eval(v)
                except Exception:
                    st.warning(f"Failed to parse parameter {k}, using default")
                    params[k] = default_params[k]

        res, bt = run_one(strategy_cls, params, df)
        if res:
            st.success("Backtest complete")

            eq = pd.DataFrame({
                "ts": df.index[:len(bt.equity_curve)],
                "equity": bt.equity_curve,
            }).set_index("ts")

            chart = (
                alt.Chart(eq.reset_index())
                .mark_line()
                .encode(
                    x="ts:T",
                    y=alt.Y("equity:Q", scale=alt.Scale(zero=False)),
                    tooltip=["ts:T", "equity:Q"],
                )
                .interactive()
            )

            st.altair_chart(chart, use_container_width=True)

            st.write("Results:")
            st.json(res)

            os.makedirs("exports", exist_ok=True)
            filename = f"exports/{instrument}_{timeframe}_{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            pd.DataFrame([res]).to_csv(filename, index=False)
            st.info(f"Results saved to `{filename}`")

        else:
            st.warning("No valid trades fired.")


if __name__ == "__main__":
    main()
