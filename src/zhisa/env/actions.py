"""Action space definitions for the trading environment."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class DiscreteAction(IntEnum):
    """Discrete action set used by default."""

    SKIP = 0
    LONG_25 = 1     # open/close-to long with 25% of equity
    LONG_50 = 2
    LONG_100 = 3
    SHORT_25 = 4
    SHORT_50 = 5
    SHORT_100 = 6
    CLOSE = 7       # close any open position
    PARTIAL_CLOSE = 8


# A continuous alternative (direction in [-1, 1], size in [0, 1])
@dataclass
class ContinuousAction:
    direction: float = 0.0   # in [-1, 1] ; negative = short
    size: float = 0.0        # in [0, 1]  ; fraction of equity
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 3.0


DISCRETE_ACTION_NAMES = {a.name: int(a) for a in DiscreteAction}
