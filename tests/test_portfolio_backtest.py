"""Tests for shared-account multi-instrument backtests."""

import pandas as pd
import pytest

from backtesting.instruments import InstrumentSpec
from backtesting.portfolio_backtest import PortfolioBacktester, PortfolioRiskLimits
from backtesting.strategy import Strategy
from backtesting.types import Trade


def _bars(index, opens, highs=None, lows=None, closes=None):
    highs = highs or [price + 1 for price in opens]
    lows = lows or [price - 1 for price in opens]
    closes = closes or opens
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=pd.DatetimeIndex(index),
    )


def _specs(data):
    return {symbol: InstrumentSpec() for symbol in data}


class _EnterOnce(Strategy):
    def __init__(self, size=1.0, stop=50.0):
        super().__init__()
        self.size = size
        self.stop = stop
        self.entered = False

    def on_bar(self, i, bar, equity):
        if self.entered:
            return None
        self.entered = True
        return Trade(
            entry_bar=bar,
            side=1,
            size=self.size,
            entry_price=bar.close,
            stop_price=self.stop,
            take_profit=None,
        )


class _NeverTrade(Strategy):
    def on_bar(self, i, bar, equity):
        return None


def test_pending_entries_fill_on_each_symbols_next_available_bar():
    t0, t1, t2 = pd.date_range("2024-01-02 09:00", periods=3, freq="h")
    data = {
        "A": _bars(
            [t0, t1], [100, 100], highs=[101, 111], lows=[99, 99],
            closes=[100, 110],
        ),
        "B": _bars([t0, t2], [200, 200]),
    }
    bt = PortfolioBacktester(
        data,
        {"A": _EnterOnce(size=1), "B": _NeverTrade()},
        starting_cash=10_000,
        instrument_specs=_specs(data),
    )

    equity, _ = bt.run()

    # A is marked at 110 while B has no bar at t1; its last mark remains 200.
    assert list(bt.equity_index) == [t0, t1, t2]
    assert equity[1] == pytest.approx(10_010.0)
    assert bt.broker.has_open_position("A")


def test_simultaneous_entries_share_buying_power_pro_rata():
    t0, t1 = pd.date_range("2024-01-02 09:00", periods=2, freq="h")
    data = {
        "B": _bars([t0, t1], [100, 100]),
        "A": _bars([t0, t1], [100, 100]),
    }
    bt = PortfolioBacktester(
        data,
        {"A": _EnterOnce(size=10), "B": _EnterOnce(size=10)},
        starting_cash=1_000,
        max_leverage=1.0,
        instrument_specs=_specs(data),
    )

    bt.run()

    assert bt.broker.has_open_position("A")
    assert bt.broker.has_open_position("B")
    assert bt.broker.positions["A"][0].size == pytest.approx(5.0)
    assert bt.broker.positions["B"][0].size == pytest.approx(5.0)
    assert bt.portfolio.gross_notional(bt._latest_prices) == pytest.approx(1_000)


def test_aggregate_stop_risk_budget_scales_simultaneous_orders():
    t0, t1 = pd.date_range("2024-01-02 09:00", periods=2, freq="h")
    data = {"A": _bars([t0, t1], [100, 100]), "B": _bars([t0, t1], [100, 100])}
    bt = PortfolioBacktester(
        data,
        {"A": _EnterOnce(size=20, stop=90), "B": _EnterOnce(size=20, stop=90)},
        starting_cash=10_000,
        risk_limits=PortfolioRiskLimits(max_total_risk_fraction=0.02),
        instrument_specs=_specs(data),
    )

    bt.run()

    assert bt.broker.positions["A"][0].size == pytest.approx(10.0)
    assert bt.broker.positions["B"][0].size == pytest.approx(10.0)
    total_stop_risk = sum(
        (100.0 - lot.stop_price) * lot.size
        for position in bt.broker.positions.values()
        for lot in position
    )
    assert total_stop_risk == pytest.approx(200.0)


def test_per_symbol_gross_cap_applies_to_all_simultaneous_entries():
    t0, t1 = pd.date_range("2024-01-02 09:00", periods=2, freq="h")
    data = {"A": _bars([t0, t1], [100, 100]), "B": _bars([t0, t1], [100, 100])}
    bt = PortfolioBacktester(
        data,
        {"A": _EnterOnce(size=10), "B": _EnterOnce(size=10)},
        starting_cash=1_000,
        max_leverage=10,
        risk_limits=PortfolioRiskLimits(max_symbol_gross_leverage=0.05),
        instrument_specs=_specs(data),
    )

    bt.run()

    assert bt.broker.positions["A"][0].size == pytest.approx(0.5)
    assert bt.broker.positions["B"][0].size == pytest.approx(0.5)


def test_stale_mark_policy_warns_for_held_symbol_without_new_bar():
    t0, t1, t2 = pd.date_range("2024-01-02 09:00", periods=3, freq="h")
    data = {
        "A": _bars([t0, t1, t2], [100, 100, 100]),
        "B": _bars([t0, t1], [200, 200]),
    }
    specs = _specs(data)
    specs["B"] = InstrumentSpec(max_stale_hours=0.5)
    bt = PortfolioBacktester(
        data,
        {"A": _NeverTrade(), "B": _EnterOnce(size=1)},
        starting_cash=10_000,
        instrument_specs=specs,
    )

    with pytest.warns(RuntimeWarning, match="stale mark for B"):
        bt.run()

    assert bt.stale_mark_events[0]["age_hours"] == pytest.approx(1.0)
    assert bt.max_mark_age_hours["B"] == pytest.approx(1.0)


def test_stale_mark_policy_can_fail_closed():
    t0, t1, t2 = pd.date_range("2024-01-02 09:00", periods=3, freq="h")
    data = {
        "A": _bars([t0, t1, t2], [100, 100, 100]),
        "B": _bars([t0, t1], [200, 200]),
    }
    specs = _specs(data)
    specs["B"] = InstrumentSpec(max_stale_hours=0.5)
    bt = PortfolioBacktester(
        data,
        {"A": _NeverTrade(), "B": _EnterOnce(size=1)},
        instrument_specs=specs,
        stale_mark_policy="error",
    )

    with pytest.raises(RuntimeError, match="stale mark for B"):
        bt.run()


def test_broker_position_sizing_uses_shared_portfolio_exposure():
    t0, t1 = pd.date_range("2024-01-02 09:00", periods=2, freq="h")
    data = {"A": _bars([t0, t1], [100, 100]), "B": _bars([t0, t1], [100, 100])}

    class CheckBuyingPower(Strategy):
        def __init__(self):
            super().__init__()
            self.available_size = None

        def on_bar(self, i, bar, equity):
            if i == 1:
                self.available_size = self._broker.position_size(
                    "B", price=bar.close, stop=90, risk_fraction=0.01,
                )
            return None

    strategy_b = CheckBuyingPower()
    bt = PortfolioBacktester(
        data,
        {"A": _EnterOnce(size=10), "B": strategy_b},
        starting_cash=1_000,
        max_leverage=1.0,
        instrument_specs=_specs(data),
    )

    bt.run()

    assert strategy_b.available_size == pytest.approx(0.0)


def test_new_position_can_hit_stop_inside_its_entry_bar():
    t0, t1 = pd.date_range("2024-01-02 09:00", periods=2, freq="h")
    data = {
        "A": _bars([t0, t1], [100, 100], highs=[101, 101], lows=[99, 80]),
    }
    bt = PortfolioBacktester(
        data, {"A": _EnterOnce(size=1, stop=90)}, starting_cash=10_000,
        instrument_specs=_specs(data),
    )

    _, trades = bt.run()

    assert len(trades) == 1
    assert trades[0].exit_ts == t1
    assert trades[0].exit_price == pytest.approx(90)
    assert trades[0].pnl == pytest.approx(-10)


def test_terminal_liquidation_uses_each_symbols_last_available_close():
    t0, t1, t2 = pd.date_range("2024-01-02 09:00", periods=3, freq="h")
    data = {
        "A": _bars(
            [t0, t1], [100, 100], highs=[101, 106], lows=[99, 99],
            closes=[100, 105],
        ),
        "B": _bars([t0, t2], [200, 200]),
    }
    bt = PortfolioBacktester(
        data,
        {"A": _EnterOnce(size=1), "B": _NeverTrade()},
        starting_cash=10_000,
        instrument_specs=_specs(data),
    )

    equity, trades = bt.run(liquidate_at_end=True)

    assert len(trades) == 1
    assert trades[0].symbol == "A"
    assert trades[0].exit_ts == t1
    assert equity[-1] == pytest.approx(bt.cash)
    assert bt.portfolio.equity_curve[-1][1] == pytest.approx(bt.cash)


def test_contract_multiplier_scales_sizing_pnl_and_notional():
    t0, t1, t2 = pd.date_range("2024-01-02 09:00", periods=3, freq="h")
    data = {
        "A": _bars([t0, t1, t2], [100, 100, 110], closes=[100, 100, 110]),
    }

    class RiskSized(Strategy):
        def __init__(self):
            super().__init__()
            self.entered = False

        def on_bar(self, i, bar, equity):
            if self.entered:
                return None
            self.entered = True
            risk_per_unit = abs(bar.close - 50) * self._contract_multiplier
            size = equity * 0.01 / risk_per_unit
            return Trade(bar, 1, size, bar.close, 50, None)

    bt = PortfolioBacktester(
        data,
        {"A": RiskSized()},
        starting_cash=10_000,
        max_leverage=10,
        instrument_specs={"A": InstrumentSpec(contract_multiplier=10)},
    )

    equity, _ = bt.run()

    assert bt.broker.positions["A"][0].size == pytest.approx(0.2)
    assert equity[-1] == pytest.approx(10_020)
    assert bt.portfolio.gross_notional(bt._latest_prices) == pytest.approx(220)


def test_instrument_specs_are_required_for_every_symbol():
    t0 = pd.Timestamp("2024-01-02")
    data = {"A": _bars([t0], [100]), "B": _bars([t0], [100])}
    with pytest.raises(ValueError, match="instrument_specs must explicitly"):
        PortfolioBacktester(
            data, {"A": _NeverTrade(), "B": _NeverTrade()},
        )


def test_incompatible_quote_currency_is_rejected():
    t0 = pd.Timestamp("2024-01-02")
    with pytest.raises(ValueError, match="FX conversion is not supported"):
        PortfolioBacktester(
            {"A": _bars([t0], [100])},
            {"A": _NeverTrade()},
            instrument_specs={"A": InstrumentSpec(quote_currency="EUR")},
            account_currency="USD",
        )


def test_runner_rejects_silent_reuse():
    t0 = pd.Timestamp("2024-01-02")
    data = {"A": _bars([t0], [100])}
    bt = PortfolioBacktester(
        data, {"A": _NeverTrade()}, instrument_specs=_specs(data),
    )
    bt.run()
    with pytest.raises(RuntimeError, match="single-use"):
        bt.run()


@pytest.mark.parametrize(
    "limits",
    [
        {"max_total_risk_fraction": -0.1},
        {"max_symbol_gross_leverage": float("inf")},
    ],
)
def test_risk_limits_reject_invalid_values(limits):
    with pytest.raises(ValueError):
        PortfolioRiskLimits(**limits)


def test_limit_buy_receives_price_improvement_like_single_asset_engine():
    """Portfolio fill semantics must match the single-asset engine: a buy limit
    that the bar opens through fills at the open, not at the limit price."""
    t0, t1, t2 = pd.date_range("2024-01-02 09:00", periods=3, freq="h")
    data = {
        "A": _bars(
            [t0, t1, t2], [100, 98, 98],
            highs=[101, 99, 99], lows=[99, 97, 97], closes=[100, 98, 98],
        ),
    }

    class _LimitBuy(Strategy):
        def __init__(self):
            super().__init__()
            self.entered = False

        def on_bar(self, i, bar, equity):
            if self.entered:
                return None
            self.entered = True
            return Trade(
                entry_bar=bar, side=1, size=1.0, entry_price=bar.close,
                stop_price=50.0, take_profit=None, limit_price=100.0,
            )

    bt = PortfolioBacktester(
        data, {"A": _LimitBuy()}, starting_cash=10_000,
        instrument_specs=_specs(data),
    )
    bt.run()

    lots = bt.broker.positions["A"]
    assert len(lots) == 1
    assert lots[0].entry_price == pytest.approx(98.0)
