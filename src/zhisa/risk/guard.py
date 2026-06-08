"""Runtime guard combining all risk checks into a single entry point."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from zhisa.risk.limits import RiskLimits, RiskState


@dataclass
class RiskDecision:
    allowed: bool
    reason: str = ""
    suggested_size: float = 1.0  # fraction of the requested size to apply


class RiskGuard:
    """Apply the hard limits to every order before it reaches the exchange."""

    def __init__(self, limits: Optional[RiskLimits] = None) -> None:
        self.limits = limits or RiskLimits()
        self.state = RiskState()

    def reset_state(self, equity: float = 1.0) -> None:
        self.state = RiskState(
            equity=equity,
            peak_equity=equity,
            day_start_equity=equity,
            week_start_equity=equity,
        )

    def check_order(
        self,
        *,
        requested_size_equity: float,
        instrument: str,
        positions: dict,
        current_price: float,
    ) -> RiskDecision:
        ls = self.limits
        st = self.state

        # Drawdown kill-switch
        if st.drawdown >= ls.max_drawdown:
            return RiskDecision(False, "max_drawdown_breach", 0.0)
        # Daily / weekly loss limits
        if st.day_pnl_fraction <= -ls.daily_loss_limit:
            return RiskDecision(False, "daily_loss_limit_breach", 0.0)
        if st.week_pnl_fraction <= -ls.weekly_loss_limit:
            return RiskDecision(False, "weekly_loss_limit_breach", 0.0)

        # Position cap per instrument
        cur_pos = positions.get(instrument, 0.0)
        new_pos = cur_pos + requested_size_equity
        if abs(new_pos) > ls.max_position_per_instrument:
            allowed = ls.max_position_per_instrument - abs(cur_pos)
            if allowed <= 0:
                return RiskDecision(False, "instrument_position_cap", 0.0)
            return RiskDecision(True, "size_clipped_to_cap",
                                suggested_size=allowed / max(abs(requested_size_equity), 1e-12))

        # Gross exposure cap
        gross = sum(abs(p) for k, p in positions.items() if k != instrument) + abs(new_pos)
        if gross > ls.max_gross_exposure:
            headroom = ls.max_gross_exposure - (gross - abs(new_pos))
            if headroom <= 0:
                return RiskDecision(False, "gross_exposure_cap", 0.0)
            return RiskDecision(True, "size_clipped_to_gross",
                                suggested_size=headroom / max(abs(requested_size_equity), 1e-12))

        # Leverage cap (gross / equity)
        leverage = gross / max(st.equity, 1e-12)
        if leverage > ls.max_leverage:
            scale = ls.max_leverage / leverage
            return RiskDecision(True, "size_clipped_to_leverage", suggested_size=scale)

        return RiskDecision(True)
