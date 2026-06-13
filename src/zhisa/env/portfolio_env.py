"""Multi-instrument portfolio environment.

A natural generalisation of :class:`zhisa.env.trading_env.TradingEnv`
to N instruments traded simultaneously. Each instrument has its own
OHLCV DataFrame and runs the same machinery (mark-to-market, SL/TP,
trailing stop, funding, kill-switch), but the observation also
includes a *portfolio summary* (per-instrument equity contribution,
gross exposure, correlation matrix of recent returns) and the
action is the joint action across all instruments.

Action space
============

The action is a flat integer in ``[0, 9**N)`` (or a
:class:`gymnasium.spaces.MultiDiscrete` of N per-instrument
DiscreteAction indices). The default per-instrument action space
matches :class:`DiscreteAction` (LONG_25/50/100, SHORT_25/50/100,
PARTIAL_CLOSE, CLOSE), giving 9 actions per instrument.

Observation
===========

The observation is a dict with:

* ``instruments`` — list of per-instrument obs dicts (``chart``,
  ``numeric``, ``context``), in the same shape as
  :class:`TradingEnv._obs` returns.
* ``portfolio`` — a fixed-size numeric tensor summarising the
  portfolio state: per-instrument MTM equity fraction, position
  fraction, drawdown contribution, plus a flattened recent
  return-covariance vector.

Reward
======

The reward is the portfolio-level MTM delta minus penalties, as in
:func:`zhisa.env.rewards.compute_reward` (PnL − DD penalty − turnover
+ Sharpe bonus − CVaR − slippage + survival). The reward weights
are shared with the single-instrument case so the user can tune
risk appetite once and reuse it.

Risk
====

* Per-instrument SL/TP/trailing-stop and drawdown kill-switch are
  applied independently.
* A *gross leverage cap* is enforced across the whole portfolio.
* A *correlation cap* can be configured: if the rolling correlation
  between any two instruments exceeds the cap, the risk guard
  rejects new orders that would increase exposure in the correlated
  pair.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from zhisa.env.actions import DiscreteAction
from zhisa.env.rewards import (
    RewardState,
    RewardWeights,
    compute_reward,
    reset_reward_state,
)
from zhisa.env.trading_env import EnvConfig, TradingEnv
from zhisa.risk.limits import RiskLimits
from zhisa.utils.logging import get_logger
from zhisa.utils.seeding import set_seed

logger = get_logger(__name__)


def _discrete_action_to_fraction(
    action: DiscreteAction,
    current_position: float = 0.0,
) -> float:
    """Translate a :class:`DiscreteAction` to a position-fraction target.

    Mirrors the table in :class:`TradingEnv.step`. The portfolio
    version does NOT support ``PARTIAL_CLOSE`` because the position
    fraction lives in the env, not in the action — but we still
    accept the enum value for API symmetry (it maps to "keep 50% of
    the current position").
    """
    if action == DiscreteAction.SKIP:
        return float(current_position)
    if action == DiscreteAction.PARTIAL_CLOSE:
        return float(current_position) * 0.5
    table = {
        DiscreteAction.LONG_25: 0.25,
        DiscreteAction.LONG_50: 0.50,
        DiscreteAction.LONG_100: 1.00,
        DiscreteAction.SHORT_25: -0.25,
        DiscreteAction.SHORT_50: -0.50,
        DiscreteAction.SHORT_100: -1.00,
        DiscreteAction.CLOSE: 0.0,
    }
    return table.get(action, float(current_position))


def encode_multi_action(per_instrument: list[int]) -> int:
    """Encode a list of per-instrument DiscreteAction indices into a
    single flat integer (base-9 representation).
    """
    out = 0
    for a in per_instrument:
        out = out * 9 + int(a)
    return out


def decode_multi_action(code: int, n_instruments: int) -> list[int]:
    """Decode a flat integer into per-instrument DiscreteAction indices."""
    if n_instruments <= 0:
        return []
    digits: list[int] = []
    x = int(code)
    for _ in range(n_instruments):
        digits.append(x % 9)
        x //= 9
    # ``digits`` is in reverse order; restore the natural order.
    return list(reversed(digits))


@dataclass
class PortfolioConfig:
    """Per-portfolio-level configuration.

    Anything that varies between :class:`TradingEnv` instances
    (window size, image size, risk limits, etc.) lives in
    :class:`EnvConfig` and is shared across instruments.
    """

    instrument_names: list[str] = field(default_factory=list)
    n_instruments: int = 0
    env_cfg: EnvConfig = field(default_factory=EnvConfig)
    gross_leverage_cap: float = 6.0     # max sum(|position_i| * leverage)
    correlation_window: int = 50        # bars of return history for the cov vector
    correlation_cap: float = 0.95       # reject new orders above this abs corr
    n_corr_features: int = 10           # flattened upper-tri of cov matrix
    seed: int = 0


class PortfolioEnv(gym.Env):
    """Multi-instrument portfolio environment.

    Args:
        dataframes: a dict ``{name: DataFrame}`` with at least two
            instruments. All DataFrames must have a ``DatetimeIndex``.
        cfg: optional :class:`PortfolioConfig` overrides.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataframes: dict[str, pd.DataFrame],
        cfg: Optional[PortfolioConfig] = None,
    ) -> None:
        super().__init__()
        if not dataframes:
            raise ValueError("dataframes must be non-empty")
        if cfg is None:
            cfg = PortfolioConfig()
        names = list(dataframes.keys())
        if len(names) < 2:
            raise ValueError("PortfolioEnv needs at least 2 instruments")
        self.cfg = cfg
        if not cfg.instrument_names:
            cfg.instrument_names = names
        cfg.n_instruments = len(names)
        if cfg.n_corr_features <= 0:
            cfg.n_corr_features = min(10, cfg.n_instruments * (cfg.n_instruments - 1) // 2)
        # Build one TradingEnv per instrument, share the env config.
        self._envs: list[TradingEnv] = []
        for name, df in dataframes.items():
            sub_cfg = deepcopy(cfg.env_cfg)
            sub_cfg.seed = int(cfg.seed)
            env = TradingEnv(df, cfg=sub_cfg)
            self._envs.append(env)
        self.n_instruments = len(self._envs)
        # Per-instrument risk lives in the underlying env's RiskGuard.
        # Build a portfolio-level risk state for the gross cap.
        self._gross_cap = float(cfg.gross_leverage_cap)
        self._corr_cap = float(cfg.correlation_cap)
        self._corr_window = int(cfg.correlation_window)
        self._n_corr_features = int(cfg.n_corr_features)
        # Action space: a flat integer in [0, 9**N). We expose it as
        # both a Discrete and a MultiDiscrete so the user can pick.
        n_actions_per = 9
        self.n_total_actions = n_actions_per ** self.n_instruments
        self.action_space = spaces.Discrete(self.n_total_actions)
        self._multi_action_space = spaces.MultiDiscrete(
            [n_actions_per] * self.n_instruments
        )
        # Observation space.
        sample_obs = self._envs[0]._obs()
        chart_shape = sample_obs["chart"].shape
        num_shape = sample_obs["numeric"].shape
        ctx_shape = sample_obs["context"].shape
        # Portfolio summary: N fractions (positions) + N fractions
        # (drawdown contributions) + N fractions (equity shares) +
        # N_corr_features (cov vector) + 1 (gross leverage) +
        # 1 (unrealised PnL fraction).
        portfolio_dim = (
            3 * self.n_instruments
            + self._n_corr_features
            + 2
        )
        self.observation_space = spaces.Dict({
            "instruments": spaces.Tuple(
                tuple(
                    spaces.Dict({
                        "chart": spaces.Box(
                            low=0.0, high=1.0, shape=chart_shape, dtype=np.float32,
                        ),
                        "numeric": spaces.Box(
                            low=-10.0, high=10.0, shape=num_shape, dtype=np.float32,
                        ),
                        "context": spaces.Box(
                            low=-1.0, high=1.0, shape=ctx_shape, dtype=np.float32,
                        ),
                    })
                    for _ in range(self.n_instruments)
                )
            ),
            "portfolio": spaces.Box(
                low=-10.0, high=10.0, shape=(portfolio_dim,),
                dtype=np.float32,
            ),
        })
        # State.
        self._t: int = 0
        self._t_start: int = 0
        self._initial_equity: float = float(cfg.env_cfg.initial_equity)
        self._equity: float = self._initial_equity
        self._peak_equity: float = self._initial_equity
        self._reward_state: RewardState = reset_reward_state(self._initial_equity)
        self._return_history: list[np.ndarray] = [
            np.zeros(0, dtype=np.float32) for _ in range(self.n_instruments)
        ]
        self._rng = np.random.default_rng(cfg.seed)
        self._last_exit_reasons: list[str] = [""] * self.n_instruments

    # ------------------------------------------------------------------
    # Per-instrument helpers
    # ------------------------------------------------------------------
    def _instrument_position(self, i: int) -> float:
        return float(self._envs[i]._position)

    def _instrument_equity_share(self, i: int) -> float:
        """Mark-to-market equity of instrument ``i`` divided by the
        total portfolio equity (in fraction)."""
        mtm = self._envs[i]._mark_to_market()
        if self._equity <= 0:
            return 0.0
        return float(mtm / self._equity)

    def _gross_leverage(self) -> float:
        return float(sum(
            abs(self._envs[i]._position) * self.cfg.env_cfg.max_leverage
            for i in range(self.n_instruments)
        ))

    def _cov_vector(self) -> np.ndarray:
        """Flatten the upper triangle of the recent-return covariance
        matrix into a fixed-size vector, padded with zeros if there
        aren't enough instruments.
        """
        # Align history lengths.
        min_len = min(len(h) for h in self._return_history) if self._return_history else 0
        if min_len < 2:
            return np.zeros(self._n_corr_features, dtype=np.float32)
        n = min(self._corr_window, min_len)
        stacked = np.stack([h[-n:] for h in self._return_history], axis=1)  # (n, N)
        cov = np.cov(stacked, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        # Take upper triangle (i < j) in column-major order.
        iu = np.triu_indices(self.n_instruments, k=1)
        vals = cov[iu]
        out = np.zeros(self._n_corr_features, dtype=np.float32)
        m = min(len(vals), self._n_corr_features)
        out[:m] = vals[:m]
        return out

    def _portfolio_summary(self) -> np.ndarray:
        positions = np.array(
            [self._instrument_position(i) for i in range(self.n_instruments)],
            dtype=np.float32,
        )
        equity_shares = np.array(
            [self._instrument_equity_share(i) for i in range(self.n_instruments)],
            dtype=np.float32,
        )
        # Drawdown contribution per instrument = (equity_i - peak_equity_i) / peak_equity_i.
        dd = np.array(
            [
                (self._envs[i]._mark_to_market() - self._envs[i]._peak_equity)
                / max(self._envs[i]._peak_equity, 1e-12)
                for i in range(self.n_instruments)
            ],
            dtype=np.float32,
        )
        cov = self._cov_vector()
        gross = np.array([self._gross_leverage()], dtype=np.float32)
        unrealised = np.array(
            [(self._equity - self._initial_equity) / max(self._initial_equity, 1e-12)],
            dtype=np.float32,
        )
        return np.concatenate([positions, equity_shares, dd, cov, gross, unrealised])

    def _obs(self) -> dict:
        return {
            "instruments": tuple(env._obs() for env in self._envs),
            "portfolio": self._portfolio_summary(),
        }

    # ------------------------------------------------------------------
    # Action encoding
    # ------------------------------------------------------------------
    def per_instrument_actions(self, action: int) -> list[DiscreteAction]:
        """Decode a flat action into per-instrument DiscreteActions."""
        digits = decode_multi_action(int(action), self.n_instruments)
        return [DiscreteAction(d) for d in digits]

    # ------------------------------------------------------------------
    # Risk check at portfolio level
    # ------------------------------------------------------------------
    def _portfolio_risk_ok(self, proposed_positions: list[float]) -> bool:
        """Return True if the proposed per-instrument positions respect
        the gross leverage cap and the correlation cap.
        """
        proposed_lev = sum(
            abs(p) * self.cfg.env_cfg.max_leverage
            for p in proposed_positions
        )
        if proposed_lev > self._gross_cap:
            return False
        if self._n_corr_features == 0:
            return True
        # Correlation cap: the most-correlated pair must be below
        # ``correlation_cap`` *in absolute value* if we are increasing
        # exposure in both of them. If neither is changing, it's fine.
        cov = self._cov_vector()
        if cov.size == 0:
            return True
        max_abs = float(np.max(np.abs(cov))) if cov.size else 0.0
        if max_abs <= self._corr_cap:
            return True
        # Find the correlated pair. We only block if *both* positions
        # are moving in the same direction (|pos_new| > |pos_old|).
        current = [self._envs[i]._position for i in range(self.n_instruments)]
        same_dir_increase = False
        for i in range(self.n_instruments):
            for j in range(i + 1, self.n_instruments):
                if abs(cov[min(i + j - 1, cov.size - 1)]) > self._corr_cap:
                    if (np.sign(proposed_positions[i]) == np.sign(proposed_positions[j])
                            and abs(proposed_positions[i]) > abs(current[i])
                            and abs(proposed_positions[j]) > abs(current[j])):
                        same_dir_increase = True
        if same_dir_increase:
            return False
        return True

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
            set_seed(int(seed))
        # Reset each sub-env.
        for env in self._envs:
            env.reset(seed=int(self._rng.integers(0, 2**31 - 1)))
        self._t = self.cfg.env_cfg.window
        self._t_start = self._t
        self._equity = self._initial_equity
        self._peak_equity = self._initial_equity
        self._reward_state = reset_reward_state(self._initial_equity)
        self._return_history = [
            np.zeros(0, dtype=np.float32) for _ in range(self.n_instruments)
        ]
        self._last_exit_reasons = [""] * self.n_instruments
        return self._obs(), {}

    def step(
        self, action: int,
    ) -> tuple[dict, float, bool, bool, dict]:
        if not self.action_space.contains(int(action)):
            raise ValueError(f"Invalid action: {action}")
        actions = self.per_instrument_actions(action)
        # Pre-compute the proposed positions for portfolio-level risk.
        proposed_positions: list[float] = []
        for i, act in enumerate(actions):
            current = self._envs[i]._position
            proposed_positions.append(
                _discrete_action_to_fraction(act, current)
            )
        if not self._portfolio_risk_ok(proposed_positions):
            # The portfolio-level risk guard rejects the action: hold.
            actions = [DiscreteAction(0)] * self.n_instruments
            # Re-derive proposed positions after hold.
            proposed_positions = [self._envs[i]._position
                                  for i in range(self.n_instruments)]
        prev_equity = self._equity
        # Step each sub-env.
        rewards_per_instrument: list[float] = []
        infos: list[dict] = []
        for i, env in enumerate(self._envs):
            _, r, term_i, trunc_i, info_i = env.step(int(actions[i]))
            rewards_per_instrument.append(r)
            infos.append(info_i)
            self._last_exit_reasons[i] = info_i.get("exit_reason", "")
        # Aggregate equity.
        self._equity = sum(env._mark_to_market() for env in self._envs)
        self._peak_equity = max(self._peak_equity, self._equity)
        # Update return history.
        for i, env in enumerate(self._envs):
            ret = (env._mark_to_market() - self._initial_equity) \
                / max(self._initial_equity, 1e-12)
            self._return_history[i] = np.append(self._return_history[i], ret)
        # Portfolio-level reward.
        turnover = sum(
            abs(inf.get("turnover", 0.0)) for inf in infos
        ) / max(self.n_instruments, 1)
        slippage_bps = float(np.mean([inf.get("slippage_bps", 0.0)
                                      for inf in infos]))
        reward, self._reward_state = compute_reward(
            self._reward_state,
            new_equity=self._equity,
            new_position=sum(self._instrument_position(i)
                              for i in range(self.n_instruments)),
            turnover=turnover,
            slippage_bps=slippage_bps,
            weights=self.cfg.env_cfg.reward_weights,
        )
        terminated = any(
            self._envs[i]._t >= len(self._envs[i].df) - 1
            for i in range(self.n_instruments)
        )
        truncated = False
        if self.cfg.env_cfg.episode_length > 0:
            if (self._envs[0]._t - self._envs[0]._t_start) >= self.cfg.env_cfg.episode_length:
                truncated = True
        # Portfolio-level kill-switch on combined drawdown.
        dd = (self._peak_equity - self._equity) / max(self._peak_equity, 1e-12)
        if (self.cfg.env_cfg.kill_on_drawdown
                and dd >= self.cfg.env_cfg.risk_limits.max_drawdown):
            terminated = True
        info = {
            "equity": self._equity,
            "per_instrument_equity": [env._mark_to_market() for env in self._envs],
            "per_instrument_position": [self._instrument_position(i)
                                         for i in range(self.n_instruments)],
            "gross_leverage": self._gross_leverage(),
            "drawdown": dd,
            "exit_reasons": list(self._last_exit_reasons),
            "rewards_per_instrument": rewards_per_instrument,
        }
        return self._obs(), float(reward), bool(terminated), bool(truncated), info

    def render(self) -> None:  # pragma: no cover
        return None


# ---------------------------------------------------------------------------
# Gymnasium registration
# ---------------------------------------------------------------------------

ZHISA_PORTFOLIO_ID = "zhisa/Portfolio-v0"

try:
    gym.register(
        id=ZHISA_PORTFOLIO_ID,
        entry_point="zhisa.env.portfolio_env:PortfolioEnv",
        vector_entry_point="zhisa.env.portfolio_env:PortfolioEnv",
        kwargs={},
        max_episode_steps=None,
    )
except Exception as _exc:  # pragma: no cover
    if "already registered" not in str(_exc).lower():
        raise
