"""
Single source of truth for the strategy registry.

Maps human-readable strategy names to their implementation classes.
Import STRATEGY_REGISTRY from here in both the API and frontend.
"""

import inspect
from typing import Any, Dict

from strategies import (
    DonchianBreakoutStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    TrendFollowingStrategy,
)

STRATEGY_REGISTRY: Dict[str, type] = {
    "Trend Following": TrendFollowingStrategy,
    "Mean Reversion": MeanReversionStrategy,
    "Momentum": MomentumStrategy,
    "Donchian Breakout": DonchianBreakoutStrategy,
}


def get_strategy_param_space(strategy_cls: type) -> Dict[str, Dict[str, Any]]:
    """Extract the parameter space from a strategy class's __init__ signature.

    Returns a dict mapping parameter names to their type and default value.
    """
    sig = inspect.signature(strategy_cls.__init__)
    space = {}
    for name, param in sig.parameters.items():
        if name == "self" or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        default = param.default
        space[name] = {"type": type(default).__name__, "default": default}
    return space
