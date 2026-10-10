"""Minimal instrument economics for portfolio-level backtests."""

from dataclasses import dataclass
import math
from typing import Optional


@dataclass(frozen=True)
class InstrumentSpec:
    """Contract economics for a linear instrument quoted in one currency.

    ``contract_multiplier`` converts one price point per unit into P&L and
    notional in ``quote_currency``. The initial portfolio runner requires all
    quote currencies to match its account currency; FX conversion is not
    silently guessed. ADV is in instrument units and ``daily_volatility`` is
    in basis points.
    """

    quote_currency: str = "USD"
    contract_multiplier: float = 1.0
    typical_daily_volume: Optional[float] = None
    daily_volatility: Optional[float] = None
    max_stale_hours: Optional[float] = None

    def __post_init__(self):
        currency = self.quote_currency.strip().upper()
        if not currency:
            raise ValueError("quote_currency must be non-empty")
        if not math.isfinite(self.contract_multiplier) or self.contract_multiplier <= 0:
            raise ValueError("contract_multiplier must be finite and > 0")
        if self.typical_daily_volume is not None and (
            not math.isfinite(self.typical_daily_volume) or self.typical_daily_volume <= 0
        ):
            raise ValueError("typical_daily_volume must be finite and > 0")
        if self.daily_volatility is not None and (
            not math.isfinite(self.daily_volatility) or self.daily_volatility < 0
        ):
            raise ValueError("daily_volatility must be finite and >= 0")
        if self.max_stale_hours is not None and (
            not math.isfinite(self.max_stale_hours) or self.max_stale_hours <= 0
        ):
            raise ValueError("max_stale_hours must be finite and > 0")
        object.__setattr__(self, "quote_currency", currency)
