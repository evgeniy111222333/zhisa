"""Risk-shaped reward functions for the trading environment.

The default reward is a weighted sum of PnL, drawdown penalty, turnover
penalty, and a small Sharpe-increment bonus. All components are
expressed in equity-relative units to keep the scale interpretable.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import numpy as np
import pandas as pd

# Project-wide annualisation default for crypto bars: 5-minute bars per year.
# US-equities' ``252`` daily bars only applies to daily OHLC — using it on
# crypto intraday frames understates the Sharpe bonus by sqrt(252/periods).
PERIODS_PER_YEAR = 365 * 24 * 12


def infer_periods_per_year(index) -> float:
    """Bars per (365-day) year inferred from a DatetimeIndex's median gap.

    Returns :data:`PERIODS_PER_YEAR` when the frequency cannot be determined
    (irregular index, too few bars, non-datetime index).
    """
    try:
        if index is None or len(index) < 2:
            return float(PERIODS_PER_YEAR)
        gap = index.to_series().diff().median()
        if pd.isna(gap) or gap <= pd.Timedelta(0):
            return float(PERIODS_PER_YEAR)
        return float(pd.Timedelta(days=365) / gap)
    except (TypeError, AttributeError, ValueError):
        return float(PERIODS_PER_YEAR)


@dataclass
class RewardWeights:
    pnl: float = 1.0
    drawdown: float = 2.0
    turnover: float = 0.001
    sharpe_bonus: float = 0.05
    liquidation_penalty: float = 5.0
    slippage_penalty: float = 0.5
    survival_bonus: float = 0.0001
    cvar_penalty: float = 0.5


@dataclass
class RewardState:
    equity: float = 1.0
    peak_equity: float = 1.0
    last_position: float = 0.0
    last_equity: float = 1.0
    returns_window: Deque[float] = field(default_factory=lambda: deque(maxlen=128))
    liquidated: bool = False
    # Bars per year for Sharpe annualisation (crypto intraday != 252).
    periods_per_year: float = PERIODS_PER_YEAR


def reset_reward_state(
    initial_equity: float = 1.0,
    periods_per_year: float = PERIODS_PER_YEAR,
) -> RewardState:
    return RewardState(
        equity=initial_equity,
        peak_equity=initial_equity,
        last_equity=initial_equity,
        returns_window=deque(maxlen=128),
        periods_per_year=periods_per_year,
    )


def compute_reward(
    state: RewardState,
    *,
    new_equity: float,
    new_position: float,
    turnover: float,
    slippage_bps: float = 0.0,
    weights: Optional[RewardWeights] = None,
) -> tuple[float, RewardState]:
    """Compute one-step reward and update the reward state in-place."""
    w = weights or RewardWeights()
    prev_equity = state.equity
    state.equity = float(new_equity)
    state.peak_equity = max(state.peak_equity, state.equity)
    pnl_ret = (state.equity - prev_equity) / max(prev_equity, 1e-12)
    state.returns_window.append(pnl_ret)

    # CVaR penalty (5% tail)
    arr = np.fromiter(state.returns_window, dtype=np.float64)
    cvar = 0.0
    if arr.size >= 20:
        q = np.quantile(arr, 0.05)
        tail = arr[arr <= q]
        cvar = float(-tail.mean()) if tail.size > 0 else 0.0

    drawdown = (state.peak_equity - state.equity) / max(state.peak_equity, 1e-12)

    sharpe = 0.0
    if arr.size >= 8:
        std = arr.std(ddof=1) if arr.size > 1 else 0.0
        periods = max(float(state.periods_per_year), 1.0)
        sharpe = (arr.mean() / (std + 1e-9)) * np.sqrt(periods)

    # ``turnover`` is already executed notional divided by equity. Multiplying
    # it by the position delta again would square the size of partial trades.
    turnover_term = abs(turnover)
    slip_term = (slippage_bps / 1e4) * w.slippage_penalty

    reward = (
        w.pnl * pnl_ret
        - w.drawdown * drawdown
        - w.turnover * turnover_term
        + w.sharpe_bonus * sharpe
        - w.cvar_penalty * cvar
        - slip_term
        + w.survival_bonus
    )
    if state.liquidated:
        reward -= w.liquidation_penalty
        state.liquidated = False
    state.last_position = new_position
    state.last_equity = state.equity
    return float(reward), state
