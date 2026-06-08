"""Hard and soft risk constraints: position sizing, leverage, DD limits."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RiskLimits:
    """Hard risk limits. All values are in fractions of equity unless noted."""

    max_leverage: float = 3.0
    max_position_per_instrument: float = 1.0
    max_gross_exposure: float = 1.0
    max_per_trade_risk: float = 0.01
    daily_loss_limit: float = 0.03
    weekly_loss_limit: float = 0.05
    max_drawdown: float = 0.15
    max_orders_per_minute: int = 30
    target_annual_vol: float = 0.20

    def as_dict(self) -> dict:
        return {
            "max_leverage": self.max_leverage,
            "max_position_per_instrument": self.max_position_per_instrument,
            "max_gross_exposure": self.max_gross_exposure,
            "max_per_trade_risk": self.max_per_trade_risk,
            "daily_loss_limit": self.daily_loss_limit,
            "weekly_loss_limit": self.weekly_loss_limit,
            "max_drawdown": self.max_drawdown,
            "max_orders_per_minute": self.max_orders_per_minute,
            "target_annual_vol": self.target_annual_vol,
        }


@dataclass
class RiskState:
    """Live risk state tracked at runtime."""

    equity: float = 1.0
    peak_equity: float = 1.0
    day_start_equity: float = 1.0
    week_start_equity: float = 1.0
    realised_pnl_today: float = 0.0
    realised_pnl_week: float = 0.0
    gross_exposure: float = 0.0
    orders_this_minute: int = 0

    @property
    def drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)

    @property
    def day_pnl_fraction(self) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        return (self.equity - self.day_start_equity) / self.day_start_equity

    @property
    def week_pnl_fraction(self) -> float:
        if self.week_start_equity <= 0:
            return 0.0
        return (self.equity - self.week_start_equity) / self.week_start_equity

    def update_equity(self, new_equity: float) -> None:
        self.equity = float(new_equity)
        self.peak_equity = max(self.peak_equity, self.equity)
        self.realised_pnl_today = self.equity - self.day_start_equity
        self.realised_pnl_week = self.equity - self.week_start_equity
