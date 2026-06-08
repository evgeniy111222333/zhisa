"""Backtest engine: replay an environment and collect equity / trades / metrics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from zhisa.backtest.metrics import Metrics, compute_metrics
from zhisa.env.trading_env import EnvConfig, TradingEnv
from zhisa.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    equity: np.ndarray
    positions: np.ndarray
    prices: np.ndarray
    timestamps: Optional[np.ndarray]
    rewards: np.ndarray
    info: list[dict]
    trade_returns: np.ndarray
    metrics: Metrics


PolicyFn = Callable[[dict], int]


def run_backtest(
    df: pd.DataFrame,
    policy: PolicyFn,
    cfg: Optional[EnvConfig] = None,
    *,
    seed: int = 0,
) -> BacktestResult:
    """Run a policy through the env and aggregate a backtest result.

    Args:
        df: OHLCV DataFrame.
        policy: callable ``policy(obs) -> action``.
        cfg: env configuration.

    Returns:
        A ``BacktestResult`` with the equity curve, positions, prices,
        rewards, per-step info, and computed metrics.
    """
    env = TradingEnv(df, cfg=cfg or EnvConfig(seed=seed))
    obs, _ = env.reset(seed=seed)
    done = False
    equity = [env._equity]
    positions = [env._position]
    prices = [float(df["close"].iloc[env._t])]
    rewards = [0.0]
    info_hist: list[dict] = []
    n_nan_clipped = 0
    n_extreme_clipped = 0
    # The env itself now has reward clipping in step() (defense in
    # depth), but the equity curve can still become pathological if a
    # downstream policy feeds garbage actions. We cap each step's
    # equity to a sane range and replace any non-finite values with
    # the previous known-good equity, so the metrics module receives
    # a usable array.
    last_good_equity = float(env._equity)
    equity_floor = 1e-12
    equity_ceiling_mult = 100.0  # 100x initial equity as a sanity cap
    init_equity = float(env._equity)
    while not done:
        action = int(policy(obs))
        obs, r, terminated, truncated, info = env.step(action)
        info_hist.append(info)
        # Guard the per-step equity.
        eq_raw = float(info["equity"])
        eq = eq_raw
        if not np.isfinite(eq) or eq <= 0:
            eq = last_good_equity
            n_nan_clipped += 1
        elif eq > init_equity * equity_ceiling_mult or eq < init_equity / equity_ceiling_mult:
            # Pathological: clamp to the last good equity, log a warning.
            logger.warning(
                "backtest: extreme equity %.3e at step %d (clamped to %.3e)",
                eq, len(equity) - 1, last_good_equity,
            )
            eq = last_good_equity
            n_extreme_clipped += 1
        else:
            last_good_equity = eq
        equity.append(eq)
        positions.append(info["position"])
        prices.append(info["price"])
        rewards.append(r if np.isfinite(r) else 0.0)
        done = bool(terminated or truncated)

    if n_nan_clipped or n_extreme_clipped:
        logger.warning(
            "backtest: %d NaN/Inf and %d extreme equity values were clipped",
            n_nan_clipped, n_extreme_clipped,
        )

    equity_arr = np.asarray(equity, dtype=np.float64)
    positions_arr = np.asarray(positions, dtype=np.float64)
    prices_arr = np.asarray(prices, dtype=np.float64)
    rewards_arr = np.asarray(rewards, dtype=np.float64)

    # Trade returns: change in PnL between position opens and closes
    trade_returns = _extract_trade_returns(positions_arr, equity_arr)

    timestamps = None
    if isinstance(df.index, pd.DatetimeIndex):
        timestamps = df.index[: len(equity_arr)].to_numpy()

    m = compute_metrics(equity_arr, trade_returns=trade_returns)
    return BacktestResult(
        equity=equity_arr,
        positions=positions_arr,
        prices=prices_arr,
        timestamps=timestamps,
        rewards=rewards_arr,
        info=info_hist,
        trade_returns=trade_returns,
        metrics=m,
    )


def _extract_trade_returns(positions: np.ndarray, equity: np.ndarray) -> np.ndarray:
    """Return per-trade PnL by detecting position changes.

    ``positions[i]`` is the position held starting at bar ``i``; the
    trade is active from the bar after a position change to the bar
    before the next change (or to the end of the series).
    """
    if positions.size < 2:
        return np.array([], dtype=np.float64)
    diffs = np.diff(positions)
    change_indices = np.where(diffs != 0)[0]
    if change_indices.size == 0:
        return np.array([], dtype=np.float64)
    rets: list[float] = []
    for i, idx in enumerate(change_indices):
        trade_start = idx + 1
        if i + 1 < change_indices.size:
            trade_end = change_indices[i + 1]
        else:
            trade_end = equity.size - 1
        if trade_end <= trade_start:
            continue
        e0 = equity[trade_start]
        e1 = equity[trade_end]
        if e0 > 0:
            rets.append((e1 - e0) / e0)
    return np.asarray(rets, dtype=np.float64)


def buy_and_hold_benchmark(df: pd.DataFrame) -> np.ndarray:
    """A simple buy & hold equity curve, normalised to start at 1.0."""
    close = df["close"].to_numpy(dtype=np.float64)
    if close.size < 2:
        return np.array([1.0])
    return close / close[0]


def random_policy(seed: int = 0) -> PolicyFn:
    """A uniformly random policy for smoke-testing."""
    rng = np.random.default_rng(seed)

    def _policy(_obs: dict) -> int:
        return int(rng.integers(0, 9))

    return _policy
