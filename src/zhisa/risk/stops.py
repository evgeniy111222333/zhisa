"""Stop-loss and take-profit computation: ATR-based, chandelier, time-stop."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class StopConfig:
    """Knobs for stop / TP calculation."""

    atr_period: int = 14
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 3.0
    trailing: bool = True
    time_stop_bars: int = 0  # 0 = disabled


def compute_stops(
    side: int,             # +1 long, -1 short
    entry_price: float,
    atr: float,
    cfg: Optional[StopConfig] = None,
) -> tuple[float, float]:
    """Return (stop_loss, take_profit) prices for a given entry and ATR."""
    cfg = cfg or StopConfig()
    if side == 0 or atr <= 0:
        return (float("nan"), float("nan"))
    if side > 0:
        sl = entry_price - cfg.sl_atr_mult * atr
        tp = entry_price + cfg.tp_atr_mult * atr
    else:
        sl = entry_price + cfg.sl_atr_mult * atr
        tp = entry_price - cfg.tp_atr_mult * atr
    return float(sl), float(tp)


def trail_stop(
    current_stop: float,
    side: int,
    price: float,
    atr: float,
    cfg: Optional[StopConfig] = None,
) -> float:
    """Tighten a trailing stop in the favourable direction only."""
    cfg = cfg or StopConfig()
    if atr <= 0 or side == 0:
        return current_stop
    if side > 0:
        candidate = price - cfg.sl_atr_mult * atr
        return max(current_stop, candidate)
    else:
        candidate = price + cfg.sl_atr_mult * atr
        return min(current_stop, candidate)
