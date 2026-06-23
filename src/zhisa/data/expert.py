"""Rule-based expert policy generators for imitation learning.

These experts are **oracle-style** for the purposes of bootstrapping a
policy via behavioral cloning (BC) and DAgger. Each expert is a
callable that maps a market state at bar ``t`` to a
:class:`zhisa.env.actions.DiscreteAction`. Three families are shipped:

* :class:`TripleBarrierExpert` — uses the forward-looking triple-barrier
  outcome (computed once per market) to label each bar. The
  *actions are perfectly hindsight-optimal* in the sense that, at the
  time the bar is observed, the expert already knows whether the
  next ``max_holding`` bars will hit TP, SL or time out. This is
  the strongest possible "expert" we can get from price data alone
  and gives BC a strong gradient signal.
* :class:`MomentumExpert` — purely backward-looking: long when the
  trailing ``lookback`` log-return is positive, short when negative.
  No look-ahead bias. Useful as a noisier but realistic expert.
* :class:`SmaCrossExpert` — bullish/bearish SMA crossover signal.

All experts are deterministic, depend only on the input DataFrame
(plus an optional precomputed triple-barrier frame), and produce
valid :class:`DiscreteAction` indices in ``[0, 9)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from zhisa.data.labeling import (
    TripleBarrierConfig,
    triple_barrier,
)
from zhisa.env.actions import DiscreteAction


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class ExpertPolicy:
    """Abstract expert policy: ``predict(df, t) -> int``."""

    name: str = "base"

    def predict(self, df: pd.DataFrame, t: int) -> int:
        raise NotImplementedError

    def predict_array(self, df: pd.DataFrame, start: int = 0) -> np.ndarray:
        """Vectorised wrapper: return one action per bar in ``[start, len(df))``."""
        n = len(df)
        out = np.zeros(max(0, n - start), dtype=np.int64)
        for i, t in enumerate(range(start, n)):
            out[i] = int(self.predict(df, t))
        return out


# ---------------------------------------------------------------------------
# Triple-barrier expert
# ---------------------------------------------------------------------------


@dataclass
class TripleBarrierExpert(ExpertPolicy):
    """An expert that knows the forward triple-barrier outcome.

    Maps the triple-barrier ``label`` column (computed once via
    :func:`triple_barrier`) to discrete actions:

    * ``label == 1``  → ``LONG_50``
    * ``label == -1`` → ``SHORT_50``
    * ``label == 0``  → ``SKIP``

    The first ``chart_window`` bars (where the observation isn't
    fully formed) and the trailing ``max_holding`` bars (where the
    barrier label is undefined) are forced to ``SKIP``.
    """

    chart_window: int = 32
    max_holding: int = 16
    cfg: TripleBarrierConfig = field(default_factory=TripleBarrierConfig)
    long_action: int = field(default=int(DiscreteAction.LONG_50))
    short_action: int = field(default=int(DiscreteAction.SHORT_50))
    skip_action: int = field(default=int(DiscreteAction.SKIP))
    _tb: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    name: str = "triple_barrier"

    def __post_init__(self) -> None:
        self.cfg = TripleBarrierConfig(
            tp_atr_mult=self.cfg.tp_atr_mult,
            sl_atr_mult=self.cfg.sl_atr_mult,
            max_holding=int(self.max_holding),
            atr_window=self.cfg.atr_window,
        )

    def _ensure_labeled(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._tb is None or len(self._tb) != len(df):
            self._tb = triple_barrier(df, self.cfg)
        return self._tb

    def predict(self, df: pd.DataFrame, t: int) -> int:
        n = len(df)
        if t < self.chart_window or t >= n - self.cfg.max_holding:
            return int(self.skip_action)
        tb = self._ensure_labeled(df)
        lbl = int(tb["label"].iloc[t])
        if lbl > 0:
            return int(self.long_action)
        if lbl < 0:
            return int(self.short_action)
        return int(self.skip_action)


@dataclass
class SymmetricUtilityExpert(ExpertPolicy):
    """Cost-aware oracle for target-position imitation labels.

    Long and short are evaluated independently with symmetric barriers. A
    failed long is therefore never automatically relabelled as a short. The
    expert emits target exposure classes; ``CLOSE`` means a flat target and is
    intentionally used instead of ``SKIP`` because static BC observations do
    not contain a current portfolio position.
    """

    chart_window: int = 128
    horizons: tuple[int, ...] = (16, 32, 64)
    horizon_weights: tuple[float, ...] = (0.5, 0.3, 0.2)
    atr_window: int = 14
    take_profit_atr: float = 2.0
    stop_loss_atr: float = 2.0
    fee_bps: float = 4.0
    slippage_bps: float = 1.5
    downside_penalty: float = 0.15
    minimum_net_edge: float = 0.0005
    ambiguity_margin: float = 0.0002
    minimum_consensus: float = 2.0 / 3.0
    size_50_atr: float = 1.00
    size_100_atr: float = 1.70
    _actions: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _long_utility: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _short_utility: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _cache_key: Optional[tuple] = field(default=None, init=False, repr=False)

    name: str = "symmetric_utility"

    def __post_init__(self) -> None:
        self.horizons = tuple(int(h) for h in self.horizons)
        self.horizon_weights = tuple(float(w) for w in self.horizon_weights)
        if not self.horizons or len(self.horizons) != len(self.horizon_weights):
            raise ValueError("horizons and horizon_weights must be non-empty and aligned")
        if any(h <= 0 for h in self.horizons):
            raise ValueError("all horizons must be positive")
        total = sum(self.horizon_weights)
        if total <= 0:
            raise ValueError("horizon weights must have positive sum")
        self.horizon_weights = tuple(w / total for w in self.horizon_weights)
        if not 0.5 <= self.minimum_consensus <= 1.0:
            raise ValueError("minimum_consensus must be in [0.5, 1.0]")

    @staticmethod
    def _forward_extreme(values: np.ndarray, horizon: int, kind: str) -> np.ndarray:
        out = np.full(len(values), np.nan, dtype=np.float64)
        if len(values) <= horizon:
            return out
        windows = np.lib.stride_tricks.sliding_window_view(values[1:], horizon)
        reduced = windows.min(axis=1) if kind == "min" else windows.max(axis=1)
        out[: len(reduced)] = reduced
        return out

    def _ensure_actions(self, df: pd.DataFrame) -> np.ndarray:
        key = (id(df), len(df), str(df.index[0]), str(df.index[-1]))
        if self._actions is not None and self._cache_key == key:
            return self._actions
        required = {"high", "low", "close"}
        if not required.issubset(df.columns):
            raise ValueError(f"missing market columns: {sorted(required - set(df.columns))}")
        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        prev = np.r_[close[0], close[:-1]]
        true_range = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
        atr = pd.Series(true_range).rolling(self.atr_window, min_periods=1).mean().to_numpy()
        atr_pct = atr / np.maximum(close, 1e-12)
        n = len(df)
        long_score = np.zeros(n, dtype=np.float64)
        short_score = np.zeros(n, dtype=np.float64)
        long_votes = np.zeros(n, dtype=np.float64)
        short_votes = np.zeros(n, dtype=np.float64)
        valid_weight = np.zeros(n, dtype=np.float64)
        round_trip_cost = 2.0 * (self.fee_bps + self.slippage_bps) / 10_000.0

        for horizon, weight in zip(self.horizons, self.horizon_weights):
            future_low = self._forward_extreme(low, horizon, "min")
            future_high = self._forward_extreme(high, horizon, "max")
            future_close = np.full(n, np.nan, dtype=np.float64)
            future_close[:-horizon] = close[horizon:]
            valid = np.isfinite(future_close) & (close > 0) & (atr > 0)

            long_stop = close - self.stop_loss_atr * atr
            long_take = close + self.take_profit_atr * atr
            short_stop = close + self.stop_loss_atr * atr
            short_take = close - self.take_profit_atr * atr
            long_adverse = future_low <= long_stop
            long_favourable = future_high >= long_take
            short_adverse = future_high >= short_stop
            short_favourable = future_low <= short_take

            terminal_long = future_close / np.maximum(close, 1e-12) - 1.0
            terminal_short = 1.0 - future_close / np.maximum(close, 1e-12)
            long_ret = np.where(
                long_adverse,
                -self.stop_loss_atr * atr_pct,
                np.where(long_favourable, self.take_profit_atr * atr_pct, terminal_long),
            )
            short_ret = np.where(
                short_adverse,
                -self.stop_loss_atr * atr_pct,
                np.where(short_favourable, self.take_profit_atr * atr_pct, terminal_short),
            )
            long_mae = np.maximum(0.0, (close - future_low) / np.maximum(close, 1e-12))
            short_mae = np.maximum(0.0, (future_high - close) / np.maximum(close, 1e-12))
            long_utility = long_ret - round_trip_cost - self.downside_penalty * long_mae
            short_utility = short_ret - round_trip_cost - self.downside_penalty * short_mae
            long_utility[~valid] = 0.0
            short_utility[~valid] = 0.0
            long_score += weight * long_utility
            short_score += weight * short_utility
            long_votes += weight * (long_utility > short_utility)
            short_votes += weight * (short_utility > long_utility)
            valid_weight += weight * valid

        actions = np.full(n, int(DiscreteAction.CLOSE), dtype=np.int64)
        choose_long = long_score > short_score
        best = np.where(choose_long, long_score, short_score)
        gap = np.abs(long_score - short_score)
        consensus = np.where(choose_long, long_votes, short_votes)
        tradable = (
            (valid_weight > 0.999)
            & (best >= self.minimum_net_edge)
            & (gap >= self.ambiguity_margin)
            & (consensus >= self.minimum_consensus)
        )
        strength = best / np.maximum(atr_pct, 1e-8)
        long_action = np.where(
            strength >= self.size_100_atr,
            int(DiscreteAction.LONG_100),
            np.where(strength >= self.size_50_atr, int(DiscreteAction.LONG_50), int(DiscreteAction.LONG_25)),
        )
        short_action = np.where(
            strength >= self.size_100_atr,
            int(DiscreteAction.SHORT_100),
            np.where(strength >= self.size_50_atr, int(DiscreteAction.SHORT_50), int(DiscreteAction.SHORT_25)),
        )
        actions[tradable] = np.where(choose_long, long_action, short_action)[tradable]
        actions[: self.chart_window] = int(DiscreteAction.CLOSE)
        actions[n - max(self.horizons) :] = int(DiscreteAction.CLOSE)
        self._actions = actions
        self._long_utility = long_score
        self._short_utility = short_score
        self._cache_key = key
        return actions

    def predict(self, df: pd.DataFrame, t: int) -> int:
        if t < 0 or t >= len(df):
            return int(DiscreteAction.CLOSE)
        return int(self._ensure_actions(df)[t])

    def predict_array(self, df: pd.DataFrame, start: int = 0) -> np.ndarray:
        return self._ensure_actions(df)[start:].copy()


# ---------------------------------------------------------------------------
# Momentum expert (no look-ahead)
# ---------------------------------------------------------------------------


@dataclass
class MomentumExpert(ExpertPolicy):
    """Backward-looking momentum expert.

    Action is decided by the sign of the trailing ``lookback`` log-return
    and a deadband ``threshold`` (in log-return units). The expert also
    flips side on sign changes, which is a more aggressive but more
    *actionable* signal than the simpler sign-of-momentum rule.
    """

    lookback: int = 16
    threshold: float = 0.0
    chart_window: int = 32
    long_action: int = field(default=int(DiscreteAction.LONG_50))
    short_action: int = field(default=int(DiscreteAction.SHORT_50))
    skip_action: int = field(default=int(DiscreteAction.SKIP))

    name: str = "momentum"

    def predict(self, df: pd.DataFrame, t: int) -> int:
        n = len(df)
        if t < max(self.chart_window, self.lookback):
            return int(self.skip_action)
        if t >= n:
            return int(self.skip_action)
        prev_close = float(df["close"].iloc[t - self.lookback])
        cur_close = float(df["close"].iloc[t])
        if prev_close <= 0 or cur_close <= 0:
            return int(self.skip_action)
        r = float(np.log(cur_close / prev_close))
        if r > self.threshold:
            return int(self.long_action)
        if r < -self.threshold:
            return int(self.short_action)
        return int(self.skip_action)


# ---------------------------------------------------------------------------
# SMA crossover expert (no look-ahead)
# ---------------------------------------------------------------------------


@dataclass
class SmaCrossExpert(ExpertPolicy):
    """SMA fast/slow crossover expert.

    Long when ``SMA(fast) > SMA(slow)`` and the fast SMA is *rising*
    (its value one bar ago is smaller); short on the mirror condition.
    Otherwise skip.
    """

    fast: int = 10
    slow: int = 30
    chart_window: int = 32
    long_action: int = field(default=int(DiscreteAction.LONG_50))
    short_action: int = field(default=int(DiscreteAction.SHORT_50))
    skip_action: int = field(default=int(DiscreteAction.SKIP))

    name: str = "sma_cross"

    def predict(self, df: pd.DataFrame, t: int) -> int:
        n = len(df)
        warm = max(self.chart_window, self.slow + 2)
        if t < warm or t >= n:
            return int(self.skip_action)
        close = df["close"].to_numpy(dtype=np.float64)
        f_now = float(np.mean(close[t - self.fast + 1 : t + 1]))
        s_now = float(np.mean(close[t - self.slow + 1 : t + 1]))
        f_prev = float(np.mean(close[t - self.fast : t]))
        s_prev = float(np.mean(close[t - self.slow : t]))
        if f_now > s_now and f_prev <= s_prev:
            return int(self.long_action)
        if f_now < s_now and f_prev >= s_prev:
            return int(self.short_action)
        return int(self.skip_action)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


SUPPORTED_EXPERTS = {
    "triple_barrier": TripleBarrierExpert,
    "symmetric_utility": SymmetricUtilityExpert,
    "momentum": MomentumExpert,
    "sma_cross": SmaCrossExpert,
}


def build_expert(kind: str, **kwargs) -> ExpertPolicy:
    """Construct an :class:`ExpertPolicy` by name.

    Unknown names raise :class:`ValueError`. Keyword arguments are
    forwarded to the expert's constructor (which is a dataclass).
    """
    if kind not in SUPPORTED_EXPERTS:
        raise ValueError(
            f"unknown expert kind: {kind!r}. "
            f"Choose one of: {sorted(SUPPORTED_EXPERTS)}"
        )
    cls = SUPPORTED_EXPERTS[kind]
    return cls(**kwargs)
