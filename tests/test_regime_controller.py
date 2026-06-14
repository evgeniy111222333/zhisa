"""Tests for the adaptive regime feedback controller."""
from __future__ import annotations

import numpy as np
import pandas as pd

from zhisa.backtest.regime_ab import RegimeABConfig, run_regime_ab_backtest
from zhisa.env.actions import DiscreteAction
from zhisa.env.trading_env import EnvConfig
from zhisa.regime import (
    RegimeAdaptiveController,
    RegimeFeedbackConfig,
)


def _ohlcv_from_close(close: np.ndarray, *, volume: float | np.ndarray = 100.0) -> pd.DataFrame:
    close = np.asarray(close, dtype=np.float64)
    open_ = np.r_[close[0], close[:-1]]
    spread = np.maximum(np.abs(close - open_) * 0.2, close * 0.001)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    if np.isscalar(volume):
        vol = np.full(close.size, float(volume))
    else:
        vol = np.asarray(volume, dtype=np.float64)
    idx = pd.date_range("2026-01-01", periods=close.size, freq="5min", tz="UTC")
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
    }, index=idx)


def _mixed_df(n: int = 180) -> pd.DataFrame:
    up = np.linspace(100.0, 122.0, n // 2)
    down = np.linspace(122.0, 92.0, n - len(up))
    close = np.r_[up, down] + 0.25 * np.sin(np.arange(n) / 4)
    volume = np.r_[np.full(len(up), 100.0), np.full(len(down), 350.0)]
    return _ohlcv_from_close(close, volume=volume)


class LongOnlyPolicy:
    def __call__(self, _obs):
        return int(DiscreteAction.LONG_100)


def test_regime_adaptive_controller_updates_memory_from_closed_outcomes() -> None:
    df = _mixed_df()
    controller = RegimeAdaptiveController(
        LongOnlyPolicy(),
        df,
        cfg=RegimeFeedbackConfig(outcome_horizon=4),
    )
    from zhisa.env.trading_env import TradingEnv

    env = TradingEnv(df, cfg=EnvConfig(seed=0, window=16, image_size=8, episode_length=24, kill_on_drawdown=False))
    obs, _ = env.reset(seed=0)
    for _ in range(18):
        action = controller.select_action(obs=obs, env=env)
        obs, reward, terminated, truncated, info = env.step(action)
        controller.observe_step(obs=obs, action=action, reward=float(reward), info=info, env=env)
        if terminated or truncated:
            break

    summary = controller.summary()

    assert summary["n_events"] > 0
    assert summary["n_closed_outcomes"] > 0
    assert summary["memory_updates"] == summary["n_closed_outcomes"]
    assert summary["memory_size"] == summary["n_closed_outcomes"]
    assert "recommended_playbooks" in summary
    assert any(event.closed and event.outcome is not None for event in controller.events)


def test_regime_ab_can_run_adaptive_feedback_controller() -> None:
    df = _mixed_df(220)

    result = run_regime_ab_backtest(
        df,
        LongOnlyPolicy(),
        env_cfg=EnvConfig(seed=0, window=16, image_size=8, episode_length=32, kill_on_drawdown=False),
        cfg=RegimeABConfig(
            adaptive_controller=True,
            feedback=RegimeFeedbackConfig(outcome_horizon=4),
        ),
        seed=0,
    )

    summary = result.gated.regime_summary
    assert result.gated.result.metrics.n_periods > 0
    assert summary["n_events"] > 0
    assert summary["memory_updates"] > 0
    assert summary["memory_size"] > 0
    assert "mean_memory_score_adjustment" in summary
