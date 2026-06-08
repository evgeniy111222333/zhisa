"""Position sizing utilities: fixed-fractional, volatility-targeted, Kelly."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SizingConfig:
    method: str = "vol_target"   # one of: "fixed", "vol_target", "kelly_frac"
    risk_per_trade: float = 0.01
    target_annual_vol: float = 0.20
    kelly_fraction: float = 0.25
    max_leverage: float = 3.0
    max_position: float = 1.0


def fixed_fractional(
    equity: float,
    risk_per_trade: float,
    stop_distance: float,
) -> float:
    """Return the position size in units of equity.

    Args:
        equity: account equity.
        risk_per_trade: fraction of equity to risk (e.g. 0.01 = 1%).
        stop_distance: distance to stop-loss in price units (positive).
    """
    if stop_distance <= 0:
        return 0.0
    return equity * risk_per_trade / stop_distance


def volatility_targeted(
    equity: float,
    realised_vol: float,
    target_vol: float = 0.20,
    max_leverage: float = 3.0,
    periods_per_year: int = 365 * 24 * 12,
) -> float:
    """Position size (in equity units) such that annualised vol is `target_vol`.

    The model assumes a constant linear payoff (i.e. size scales with
    `target_vol / realised_vol`). Caller should clamp by max_leverage.
    """
    if realised_vol <= 0:
        return 0.0
    raw = target_vol / realised_vol
    return min(raw, max_leverage)


def kelly_fractional(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    kelly_fraction: float = 0.25,
) -> float:
    """Fractional-Kelly sizing (positive bets only).

    Kelly f* = (p/a) - q, where a = avg_win / avg_loss.
    Returns ``kelly_fraction * f*`` clamped to [0, 1].
    """
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    a = max(avg_win, 1e-12) / avg_loss
    f_star = (win_rate / a) - (1.0 - win_rate)
    f = max(0.0, kelly_fraction * f_star)
    return min(f, 1.0)
