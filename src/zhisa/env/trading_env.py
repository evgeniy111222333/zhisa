"""Trading environment (Gymnasium API).

A discrete-action, single-instrument perpetual-futures-style environment
backed by an OHLCV DataFrame. Includes:

- order execution with slippage and fees (see ``execution.py``);
- position management (open, scale, close) and PnL accounting;
- a risk-shaped reward (see ``rewards.py``);
- an info dict with realised returns, drawdown, and barrier hits.

The observation is a dict with ``chart``, ``numeric`` and ``context``
arrays, matching the model's input convention.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces

from zhisa.data.labeling import TripleBarrierConfig, triple_barrier
from zhisa.env.actions import DiscreteAction
from zhisa.env.execution import ExecutionConfig, execute_order
from zhisa.env.rewards import (
    RewardState,
    RewardWeights,
    compute_reward,
    reset_reward_state,
)
from zhisa.features.ohlcv import compute_ohlcv_features, normalize_feature_window
from zhisa.features.time import compute_time_features
from zhisa.rendering.chart_renderer import render_chart
from zhisa.risk.guard import RiskGuard
from zhisa.risk.limits import RiskLimits
from zhisa.utils.seeding import set_seed
from zhisa.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EnvConfig:
    initial_equity: float = 1.0
    fee_bps: float = 4.0
    slippage_bps_per_unit: float = 1.5
    market_depth_units: float = 100.0
    max_leverage: float = 3.0
    max_position: float = 1.0
    window: int = 32
    image_size: int = 64
    include_volume: bool = True
    include_indicators: bool = True
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    risk_limits: RiskLimits = field(default_factory=RiskLimits)
    seed: Optional[int] = 0
    # ---- Per-position exits (0 disables). All in fraction-of-entry. ----
    stop_loss_pct: float = 0.0       # e.g. 0.02 -> close if price moves -2% from entry
    take_profit_pct: float = 0.0     # e.g. 0.05 -> close if price moves +5% from entry
    trailing_stop_pct: float = 0.0   # e.g. 0.01 -> trail SL by 1% from peak price
    # ---- Episode control. ----
    episode_length: int = 0          # 0 -> no cap; >0 -> truncate after N steps from reset
    # ---- Drawdown kill-switch. ----
    kill_on_drawdown: bool = True    # terminate episode when max_drawdown is breached
    # ---- Conservative fill. ----
    # If both SL and TP are within a single bar, the conservative
    # assumption is that the unfavourable side was hit first. This
    # avoids hindsight bias in backtests.
    conservative_bar_fill: bool = True
    # ---- Funding rate (perpetual futures). ----
    # Funding is paid by longs to shorts when positive, and vice
    # versa. A typical 8h funding rate on BTC is 0.01% (1e-4).
    funding_rate: float = 0.0         # per-funding-interval, signed
    funding_interval: int = 0         # 0 -> disabled; e.g. 96 for 8h at 5min bars
    funding_column: str = ""          # if set, read from df[col] instead of fixed


class TradingEnv(gym.Env):
    """A single-instrument trading environment on top of an OHLCV frame."""

    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, cfg: Optional[EnvConfig] = None) -> None:
        super().__init__()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("df must have a DatetimeIndex")
        cfg = cfg or EnvConfig()
        self.cfg = cfg
        self.df = df.reset_index(drop=False)
        self._features = compute_ohlcv_features(
            df,
            include_volume=cfg.include_volume,
            include_indicators=cfg.include_indicators,
        ).reset_index(drop=True)
        self._time = compute_time_features(df).reset_index(drop=True)
        # Numeric obs carries only OHLCV-derived features. Cyclic time
        # embeddings go into ``context`` (last bar) so the env contract
        # matches :class:`PolicyConfig`'s ``in_numeric_features=32`` and
        # ``in_context_features=10`` defaults — i.e. ``build_default_policy()``
        # and ``TradingEnv()`` are wire-compatible out of the box.
        n_numeric_features = self._features.shape[1]
        n_context_features = self._time.shape[1]
        self.window = cfg.window
        # Observation spaces
        self.observation_space = spaces.Dict({
            "chart": spaces.Box(low=0.0, high=1.0, shape=(3, cfg.image_size, cfg.image_size), dtype=np.float32),
            "numeric": spaces.Box(low=-10.0, high=10.0,
                                  shape=(cfg.window, n_numeric_features), dtype=np.float32),
            "context": spaces.Box(low=-1.0, high=1.0, shape=(n_context_features,), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(len(DiscreteAction))
        self._exec_cfg = ExecutionConfig(
            taker_fee_bps=cfg.fee_bps,
            slippage_bps_per_unit=cfg.slippage_bps_per_unit,
            market_depth_units=cfg.market_depth_units,
        )
        # The env's max_leverage is the actual cap. Make sure the
        # risk guard's per-instrument cap is consistent with it so
        # LONG_100 doesn't get clipped to ~0.33 on a 3x setup.
        risk_limits = RiskLimits(
            max_leverage=cfg.risk_limits.max_leverage,
            max_position_per_instrument=max(
                cfg.risk_limits.max_position_per_instrument,
                cfg.max_leverage,
            ),
            max_gross_exposure=max(
                cfg.risk_limits.max_gross_exposure,
                cfg.max_leverage,
            ),
            max_per_trade_risk=cfg.risk_limits.max_per_trade_risk,
            daily_loss_limit=cfg.risk_limits.daily_loss_limit,
            weekly_loss_limit=cfg.risk_limits.weekly_loss_limit,
            max_drawdown=cfg.risk_limits.max_drawdown,
            max_orders_per_minute=cfg.risk_limits.max_orders_per_minute,
            target_annual_vol=cfg.risk_limits.target_annual_vol,
        )
        self._risk = RiskGuard(risk_limits)
        self._reward_state = reset_reward_state(cfg.initial_equity)
        self._rng = np.random.default_rng(cfg.seed)
        self._reset_state()

    # -------- internal helpers --------
    def _reset_state(self) -> None:
        cfg = self.cfg
        self._t = cfg.window
        self._t_start = cfg.window
        self._position = 0.0
        self._avg_entry = 0.0
        self._peak_price = 0.0
        self._equity = cfg.initial_equity
        self._peak_equity = cfg.initial_equity
        self._info_extra: dict = {}
        self._reward_state = reset_reward_state(cfg.initial_equity)
        self._risk.reset_state(cfg.initial_equity)
        self._turnover = 0.0
        self._last_position = 0.0
        self._last_exit_reason: str = ""
        # Funding counter (in bars since the last funding event).
        self._bars_since_funding: int = 0
        # Cumulative funding paid (positive = paid by trader).
        self._funding_paid: float = 0.0

    def _obs(self) -> dict:
        cfg = self.cfg
        start = self._t - cfg.window
        end = self._t
        window_df = self.df.iloc[start:end]
        chart = render_chart(window_df, size=cfg.image_size).numpy()
        feat = self._features.iloc[start:end].to_numpy(dtype=np.float32)
        tim = self._time.iloc[start:end].to_numpy(dtype=np.float32)
        # Apply the same z-score normalization using trailing window
        hist_start = max(0, self._t - 256)
        history_window = self._features.iloc[hist_start:end].to_numpy(dtype=np.float32)
        num = normalize_feature_window(feat, history_window)
        # ``context`` is the latest cyclic-time embedding (context encoder input).
        ctx = np.nan_to_num(tim[-1], nan=0.0, posinf=0.0, neginf=0.0)
        return {"chart": chart, "numeric": num, "context": ctx}

    # -------- public probe helpers --------
    @property
    def obs_numeric_dim(self) -> int:
        """Number of features in the ``numeric`` obs (== ``in_numeric_features``
        in :class:`PolicyConfig`)."""
        return int(self._features.shape[1])

    @property
    def obs_context_dim(self) -> int:
        """Number of features in the ``context`` obs (== ``in_context_features``
        in :class:`PolicyConfig`)."""
        return int(self._time.shape[1])

    def _mark_to_market(self, price: Optional[float] = None) -> float:
        """Mark-to-market equity at the given (or current) price.

        Defensive: if ``_avg_entry`` is non-positive (a corrupted state),
        we treat the position as flat rather than producing a runaway
        PnL. This should never happen in normal flow because the
        VWAP-update branch in :meth:`step` guards against it, but we
        keep the guard here as defence-in-depth.
        """
        cfg = self.cfg
        if self._position == 0:
            return self._equity
        if not (math.isfinite(self._avg_entry) and self._avg_entry > 0):
            # Corrupted entry price: fall back to "no unrealised PnL".
            return self._equity
        if price is None:
            price = float(self.df["close"].iloc[self._t])
        ret = (price / self._avg_entry) - 1.0
        return self._equity + self._position * cfg.max_leverage * ret

    def _check_barrier_exits(
        self, bar_low: float, bar_high: float,
    ) -> tuple[bool, float, str]:
        """Check whether the current bar would have triggered SL/TP/trailing.

        Returns ``(triggered, exit_price, reason)``. Conservative fill
        (the default) assumes the *unfavourable* side was hit first
        when both SL and TP are inside the bar — this avoids hindsight
        bias in backtests.
        """
        cfg = self.cfg
        if self._position == 0:
            return False, 0.0, ""
        is_long = self._position > 0
        # Update trailing-stop anchor.
        if cfg.trailing_stop_pct > 0:
            if is_long:
                self._peak_price = max(self._peak_price, bar_high)
            else:
                # For shorts, the "favourable" direction is *down*; track trough.
                # We re-use ``_peak_price`` but invert the meaning for shorts.
                self._peak_price = min(self._peak_price, bar_low)
        # Compute trigger levels.
        sl_price = None
        if cfg.stop_loss_pct > 0:
            sl_price = self._avg_entry * (1.0 - cfg.stop_loss_pct)
        if cfg.trailing_stop_pct > 0:
            if is_long:
                trail = self._peak_price * (1.0 - cfg.trailing_stop_pct)
            else:
                trail = self._peak_price * (1.0 + cfg.trailing_stop_pct)
            sl_price = trail if sl_price is None else (
                max(sl_price, trail) if is_long else min(sl_price, trail)
            )
        tp_price = None
        if cfg.take_profit_pct > 0:
            tp_price = self._avg_entry * (1.0 + cfg.take_profit_pct)
        # Check which barrier (if any) the bar actually touched.
        sl_hit = sl_price is not None and (
            (is_long and bar_low <= sl_price) or
            (not is_long and bar_high >= sl_price)
        )
        tp_hit = tp_price is not None and (
            (is_long and bar_high >= tp_price) or
            (not is_long and bar_low <= tp_price)
        )
        if not sl_hit and not tp_hit:
            return False, 0.0, ""
        # Determine the exit price. With both hit, conservative = worst
        # case for the trader; otherwise use the one that fired.
        if sl_hit and tp_hit:
            if cfg.conservative_bar_fill:
                exit_price = sl_price if is_long else tp_price
                reason = "stop_loss_conservative"
            else:
                exit_price = tp_price if is_long else sl_price
                reason = "take_profit_conservative" if is_long else "stop_loss_conservative"
        elif sl_hit:
            exit_price = sl_price
            reason = "stop_loss"
        else:
            exit_price = tp_price
            reason = "take_profit"
        return True, float(exit_price), reason

    # -------- gym API --------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[dict, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
            set_seed(int(seed))
        self._reset_state()
        return self._obs(), {}

    def step(self, action: int) -> Tuple[dict, float, bool, bool, dict]:
        cfg = self.cfg
        if not self.action_space.contains(int(action)):
            raise ValueError(f"Invalid action: {action}")

        price_now = float(self.df["close"].iloc[self._t])
        bar_high = float(self.df["high"].iloc[self._t])
        bar_low = float(self.df["low"].iloc[self._t])
        atr_val = float(self._features.iloc[self._t].get("atr_14", 0.0) or 0.0)

        # 1. Check barrier exits (SL/TP/trailing) for the open position
        #    using the *current* bar's high/low. If triggered, the agent's
        #    action is overridden and the position is closed at the
        #    barrier price.
        exit_triggered = False
        exit_price = 0.0
        exit_reason = ""
        if self._position != 0 and (
            cfg.stop_loss_pct > 0
            or cfg.take_profit_pct > 0
            or cfg.trailing_stop_pct > 0
        ):
            exit_triggered, exit_price, exit_reason = self._check_barrier_exits(
                bar_low, bar_high,
            )
        if exit_triggered:
            # Close the open position at the barrier price.
            ret = (exit_price / max(self._avg_entry, 1e-12)) - 1.0
            realised = self._position * cfg.max_leverage * ret
            self._equity += realised
            self._position = 0.0
            self._avg_entry = 0.0
            self._peak_price = 0.0
            price_now = exit_price
            self._last_exit_reason = exit_reason
            # Skip agent action; treat the bar as a forced close.
            action = int(DiscreteAction.CLOSE)

        # 2. Translate the (possibly overridden) action.
        target = self._position
        a = DiscreteAction(int(action))
        if a in (DiscreteAction.SKIP,):
            target = self._position
        elif a in (DiscreteAction.CLOSE,):
            target = 0.0
        elif a in (DiscreteAction.PARTIAL_CLOSE,):
            target = self._position * 0.5
        else:
            size_map = {
                DiscreteAction.LONG_25: 0.25,
                DiscreteAction.LONG_50: 0.50,
                DiscreteAction.LONG_100: 1.00,
                DiscreteAction.SHORT_25: -0.25,
                DiscreteAction.SHORT_50: -0.50,
                DiscreteAction.SHORT_100: -1.00,
            }
            target = size_map.get(a, self._position)

        # 3. Risk guard. If the action is *reducing* the absolute
        #    position (closing or partial-closing), always allow it
        #    in full — the trader must always be able to de-risk.
        requested_target = float(target)
        risk_allowed = True
        risk_reason = "de_risk_bypass"
        risk_suggested_size = 1.0
        delta = target - self._position
        book_top = 1.0
        if abs(delta) <= 1e-9:
            risk_reason = "no_order"
        elif abs(target) < abs(self._position) - 1e-9:
            # Reduction: no cap on size, no risk-guard check.
            pass
        else:
            risk_reason = ""
            decision = self._risk.check_order(
                requested_size_equity=abs(delta) * cfg.max_leverage,
                instrument="primary",
                positions={"primary": self._position * cfg.max_leverage},
                current_price=price_now,
            )
            risk_allowed = bool(decision.allowed)
            risk_reason = decision.reason
            risk_suggested_size = float(decision.suggested_size)
            if not decision.allowed:
                target = self._position
            else:
                target = self._position + delta * decision.suggested_size
        final_target = float(target)

        # 4. Execution
        size_units = abs(target - self._position) * cfg.max_leverage / max(price_now, 1e-12)
        side = int(np.sign(target - self._position)) if target != self._position else 0
        fill = execute_order(
            side=side,
            requested_size=size_units,
            ref_price=price_now,
            book_top_size=book_top,
            cfg=self._exec_cfg,
            rng=self._rng,
        )
        if fill.filled > 0:
            if self._position == 0 or np.sign(self._position) != np.sign(target):
                self._avg_entry = fill.price
                self._peak_price = fill.price
            else:
                # VWAP update
                old_notional = self._position * self._avg_entry
                added_notional = (target - self._position) * fill.price
                if target != 0 and abs(target) > 1e-12:
                    # Sign-preserving guard: keep target's sign, avoid /0.
                    denom = target if abs(target) > 1e-9 else (
                        1e-9 if target > 0 else -1e-9
                    )
                    new_avg = (old_notional + added_notional) / denom
                    # Sanity: avg_entry must be positive and finite.
                    if math.isfinite(new_avg) and new_avg > 0:
                        self._avg_entry = new_avg
                    # else: keep the old avg_entry; the position update
                    # below will not be poisoned.
                if is_long := (target > 0):
                    self._peak_price = max(self._peak_price, fill.price)
                else:
                    self._peak_price = min(self._peak_price, fill.price)
            self._position = target
            self._equity -= fill.fee
        elif target == 0.0 and self._position == 0.0:
            # Manual close without a fill — clear position bookkeeping.
            self._avg_entry = 0.0
            self._peak_price = 0.0

        # 5. Advance time
        self._t += 1
        terminated = self._t >= len(self.df) - 1
        truncated = False

        # 5b. Funding: if a funding event falls on this bar, settle it
        #     against the open position. Funding is per-bar here
        #     (multiplied by the equity-fraction position and the
        #     configured leverage, matching the PnL accounting in
        #     ``_mark_to_market``).
        funding_paid_this_step = 0.0
        if cfg.funding_interval > 0 and self._position != 0.0:
            self._bars_since_funding += 1
            if self._bars_since_funding >= cfg.funding_interval:
                # Get the funding rate (either from the df or fixed).
                if cfg.funding_column and cfg.funding_column in self.df.columns:
                    rate = float(
                        self.df[cfg.funding_column].iloc[self._t - 1]
                    )
                else:
                    rate = float(cfg.funding_rate)
                # A positive rate means longs pay shorts.
                # Equity change is negative for longs with positive rate.
                # ``paid`` is the absolute amount the trader transferred
                # (positive = we paid, negative = we received).
                equity_change = -self._position * cfg.max_leverage * rate
                paid = -equity_change
                self._equity += equity_change
                self._funding_paid += paid
                funding_paid_this_step = paid
                self._bars_since_funding = 0

        new_equity = self._mark_to_market()
        turnover = abs(fill.filled) * price_now / max(self._equity, 1e-12)
        self._turnover = turnover
        self._risk.state.update_equity(new_equity)

        # 6. Kill-switch on drawdown
        if (cfg.kill_on_drawdown
                and self._risk.state.drawdown >= cfg.risk_limits.max_drawdown):
            terminated = True
            self._last_exit_reason = "max_drawdown_kill_switch"

        # 7. Episode length cap (truncate, not terminate, so the
        #    PPO loop knows it's a forced boundary, not a crash).
        if cfg.episode_length > 0 and (self._t - self._t_start) >= cfg.episode_length:
            truncated = True
            self._last_exit_reason = self._last_exit_reason or "episode_length_cap"

        reward, self._reward_state = compute_reward(
            self._reward_state,
            new_equity=new_equity,
            new_position=self._position,
            turnover=turnover,
            slippage_bps=fill.slippage_bps,
            weights=cfg.reward_weights,
        )
        # ---- Defense-in-depth NaN/Inf guards ----
        # The reward function can produce non-finite values if equity or
        # price became non-finite (e.g. corruption upstream). We clip
        # the reward to a sane range so a single bad step does not
        # poison the PPO value head. We also log a warning so the
        # operator can see when this fires.
        if not math.isfinite(reward):
            logger.warning(
                "non-finite reward at t=%d (pos=%.4f equity=%.6e price=%.4f) -> clipped to 0",
                self._t, self._position, new_equity, price_now,
            )
            reward = 0.0
        else:
            reward = float(np.clip(reward, -10.0, 10.0))
        # Equity must always be finite and non-negative; clamp the
        # book value to a small positive floor to avoid divide-by-zero
        # downstream.
        if not math.isfinite(new_equity) or new_equity <= 0:
            new_equity = max(cfg.initial_equity * 0.01, 1e-9)
            self._equity = new_equity
        self._last_position = self._position
        info = {
            "equity": new_equity,
            "position": self._position,
            "fill_price": fill.price,
            "slippage_bps": fill.slippage_bps,
            "fee": fill.fee,
            "turnover": turnover,
            "price": price_now,
            "atr": atr_val,
            "exit_reason": self._last_exit_reason,
            "funding_paid": funding_paid_this_step,
            "cumulative_funding": self._funding_paid,
            "requested_position": requested_target,
            "target_position": final_target,
            "order_side": side,
            "requested_size": fill.requested,
            "filled_size": fill.filled,
            "risk_allowed": risk_allowed,
            "risk_reason": risk_reason,
            "risk_suggested_size": risk_suggested_size,
        }
        return self._obs(), float(reward), bool(terminated), bool(truncated), info

    def render(self) -> None:  # pragma: no cover
        return None


# Register with the Gymnasium registry so users can do
# ``gym.make("zhisa/Trading-v0", df=df, cfg=EnvConfig(...))``.
ZHISA_TRADING_ID = "zhisa/Trading-v0"

try:
    gym.register(
        id=ZHISA_TRADING_ID,
        entry_point="zhisa.env.trading_env:TradingEnv",
        vector_entry_point="zhisa.env.trading_env:TradingEnv",
        kwargs={},
        max_episode_steps=None,  # env terminates on its own
    )
except Exception as _exc:  # pragma: no cover
    # Re-registration in a re-imported session is fine.
    if "already registered" not in str(_exc).lower():
        raise
