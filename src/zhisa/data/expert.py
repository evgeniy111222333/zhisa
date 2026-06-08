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
