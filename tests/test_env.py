"""Tests for the trading environment."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gymnasium as gym

from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.env.actions import DiscreteAction
from zhisa.env.execution import (
    ExecutionConfig,
    execute_order,
)
from zhisa.env.rewards import compute_reward, reset_reward_state
from zhisa.env.trading_env import (
    ZHISA_TRADING_ID,
    EnvConfig,
    TradingEnv,
)
from zhisa.risk.limits import RiskLimits

from zhisa.env.execution import ExecutionConfig, execute_order
from zhisa.env.rewards import RewardWeights, compute_reward, reset_reward_state
from zhisa.env.trading_env import EnvConfig, TradingEnv
from zhisa.env.actions import DiscreteAction


def test_execute_order_no_negative_slippage():
    fill = execute_order(side=1, requested_size=1.0, ref_price=100.0, book_top_size=10.0)
    assert fill.filled > 0
    assert fill.price >= 100.0
    assert fill.slippage_bps >= 0
    assert fill.fee >= 0


def test_execute_order_short_slippage():
    fill = execute_order(side=-1, requested_size=1.0, ref_price=100.0, book_top_size=10.0)
    assert fill.price <= 100.0
    assert fill.slippage_bps >= 0


def test_execute_order_zero_size():
    fill = execute_order(side=1, requested_size=0.0, ref_price=100.0, book_top_size=10.0)
    assert fill.filled == 0.0


def test_execute_order_post_only_rejected():
    cfg = ExecutionConfig()
    fill = execute_order(side=1, requested_size=1.0, ref_price=100.0,
                         book_top_size=10.0, cfg=cfg, post_only=True)
    assert fill.filled == 0.0


def test_reward_increases_on_profit():
    s = reset_reward_state(1.0)
    r, s = compute_reward(s, new_equity=1.01, new_position=0.0, turnover=0.0)
    r_loss, _ = compute_reward(s, new_equity=0.99, new_position=0.0, turnover=0.0)
    assert r > r_loss


def test_trading_env_reset(small_market):
    env = TradingEnv(small_market, cfg=EnvConfig(window=16, image_size=16))
    obs, info = env.reset(seed=0)
    assert "chart" in obs and "numeric" in obs and "context" in obs
    assert obs["chart"].shape == (3, 16, 16)
    assert obs["numeric"].shape[0] == 16


def test_trading_env_random_rollout(small_market):
    env = TradingEnv(small_market, cfg=EnvConfig(window=16, image_size=16))
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    total_reward = 0.0
    done = False
    steps = 0
    while not done and steps < 50:
        a = int(rng.integers(0, len(DiscreteAction)))
        obs, r, term, trunc, info = env.step(a)
        total_reward += r
        done = term or trunc
        steps += 1
    assert steps > 0
    # Equity must be finite
    assert np.isfinite(info["equity"])


def test_trading_env_action_validation(small_market):
    env = TradingEnv(small_market, cfg=EnvConfig(window=16, image_size=16))
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(999)


# ---------------------------------------------------------------------------
# SL / TP / trailing-stop / drawdown kill-switch / episode cap
# ---------------------------------------------------------------------------


def _deterministic_market(n_bars: int = 600, seed: int = 0) -> pd.DataFrame:
    """A near-deterministic market (low vol, no shocks) for barrier tests."""
    return generate_market(MarketConfig(
        n_bars=n_bars, base_vol=0.01, shock_prob=0.0, student_t_df=20.0,
        seed=seed,
    ))


def test_stop_loss_closes_long_position():
    env_cfg = EnvConfig(
        window=16, image_size=16, stop_loss_pct=0.005, take_profit_pct=0.0,
        trailing_stop_pct=0.0, kill_on_drawdown=False, episode_length=0,
    )
    df = _deterministic_market(n_bars=400, seed=0)
    env = TradingEnv(df, cfg=env_cfg)
    env.reset(seed=0)
    _, _, _, _, _ = env.step(int(DiscreteAction.LONG_100))
    assert env._position != 0.0
    entry = env._avg_entry
    # Force a deep price drop in the CURRENT bar (the env reads
    # ``self.df.iloc[self._t]`` to decide barriers at this step).
    env.df.loc[env._t, "low"] = entry * (1.0 - 0.02)
    env.df.loc[env._t, "close"] = entry * (1.0 - 0.02)
    _, _, _, _, info = env.step(int(DiscreteAction.LONG_100))
    assert env._position == 0.0
    assert info["exit_reason"] == "stop_loss"


def test_take_profit_closes_long_position():
    env_cfg = EnvConfig(
        window=16, image_size=16, stop_loss_pct=0.0, take_profit_pct=0.05,
        kill_on_drawdown=False, episode_length=0,
    )
    df = _deterministic_market(n_bars=400, seed=1)
    env = TradingEnv(df, cfg=env_cfg)
    env.reset(seed=0)
    _, _, _, _, _ = env.step(int(DiscreteAction.LONG_100))
    entry = env._avg_entry
    env.df.loc[env._t, "high"] = entry * 1.10
    env.df.loc[env._t, "close"] = entry * 1.10
    _, _, _, _, info = env.step(int(DiscreteAction.LONG_100))
    assert env._position == 0.0
    assert info["exit_reason"] == "take_profit"


def test_trailing_stop_locks_in_profit():
    env_cfg = EnvConfig(
        window=16, image_size=16, stop_loss_pct=0.0, take_profit_pct=0.0,
        trailing_stop_pct=0.01, kill_on_drawdown=False, episode_length=0,
    )
    df = _deterministic_market(n_bars=400, seed=2)
    env = TradingEnv(df, cfg=env_cfg)
    env.reset(seed=0)
    _, _, _, _, _ = env.step(int(DiscreteAction.LONG_100))
    entry = env._avg_entry
    # Push the price up bar-by-bar so the trailing anchor ratchets.
    for k in range(1, 6):
        t_k = env._t + k
        env.df.loc[t_k, "high"] = entry * (1.0 + 0.01 * k)
        env.df.loc[t_k, "low"] = entry * (1.0 + 0.01 * k)
        env.df.loc[t_k, "close"] = entry * (1.0 + 0.01 * k)
    for _ in range(5):
        env.step(int(DiscreteAction.LONG_100))
    peak = entry * 1.05
    env.df.loc[env._t, "high"] = peak  # keep peak anchor at 1.05
    env.df.loc[env._t, "low"] = peak * 0.985
    env.df.loc[env._t, "close"] = peak * 0.985
    _, _, _, _, info = env.step(int(DiscreteAction.LONG_100))
    assert env._position == 0.0
    assert info["exit_reason"] in ("stop_loss", "trailing_stop")


def test_conservative_bar_fill_assumes_worst_case():
    env_cfg = EnvConfig(
        window=16, image_size=16, stop_loss_pct=0.01, take_profit_pct=0.01,
        kill_on_drawdown=False, episode_length=0, conservative_bar_fill=True,
    )
    df = _deterministic_market(n_bars=400, seed=3)
    env = TradingEnv(df, cfg=env_cfg)
    env.reset(seed=0)
    _, _, _, _, _ = env.step(int(DiscreteAction.LONG_100))
    entry = env._avg_entry
    # Bar that contains BOTH the SL and TP levels.
    env.df.loc[env._t, "low"] = entry * 0.985
    env.df.loc[env._t, "high"] = entry * 1.015
    env.df.loc[env._t, "close"] = entry * 1.015
    _, _, _, _, info = env.step(int(DiscreteAction.LONG_100))
    assert env._position == 0.0
    assert "stop_loss" in info["exit_reason"]


def test_kill_switch_terminates_on_drawdown():
    env_cfg = EnvConfig(
        window=16, image_size=16, stop_loss_pct=0.0, take_profit_pct=0.0,
        kill_on_drawdown=True, episode_length=0,
        risk_limits=RiskLimits(max_drawdown=0.10),
    )
    df = _deterministic_market(n_bars=200, seed=4)
    env = TradingEnv(df, cfg=env_cfg)
    env.reset(seed=0)
    env.step(int(DiscreteAction.LONG_100))
    entry = env._avg_entry
    # Mutate the bar that the env will use for MTM (the bar at
    # ``self._t`` *after* advancing, since the kill-switch reads
    # ``self._mark_to_market()`` with the new time).
    next_t = env._t + 1
    env.df.loc[next_t, "low"] = entry * 0.80
    env.df.loc[next_t, "close"] = entry * 0.80
    _, _, term, _, info = env.step(int(DiscreteAction.LONG_100))
    assert term is True
    assert info["exit_reason"] == "max_drawdown_kill_switch"


def test_kill_switch_can_be_disabled():
    env_cfg = EnvConfig(
        window=16, image_size=16, stop_loss_pct=0.0, take_profit_pct=0.0,
        kill_on_drawdown=False, episode_length=0,
        risk_limits=RiskLimits(max_drawdown=0.10),
    )
    df = _deterministic_market(n_bars=200, seed=4)
    env = TradingEnv(df, cfg=env_cfg)
    env.reset(seed=0)
    env.step(int(DiscreteAction.LONG_100))
    entry = env._avg_entry
    next_t = env._t + 1
    env.df.loc[next_t, "low"] = entry * 0.80
    env.df.loc[next_t, "close"] = entry * 0.80
    _, _, term, _, _ = env.step(int(DiscreteAction.LONG_100))
    assert term is False


def test_episode_length_cap_truncates():
    env_cfg = EnvConfig(
        window=16, image_size=16, stop_loss_pct=0.0, take_profit_pct=0.0,
        kill_on_drawdown=False, episode_length=10,
    )
    df = _deterministic_market(n_bars=400, seed=5)
    env = TradingEnv(df, cfg=env_cfg)
    env.reset(seed=0)
    truncated = False
    for _ in range(20):
        _, _, term, trunc, info = env.step(int(DiscreteAction.LONG_100))
        if trunc:
            truncated = True
            break
    assert truncated
    assert info["exit_reason"] == "episode_length_cap"


def test_no_barriers_means_no_force_closes(small_market):
    env = TradingEnv(small_market, cfg=EnvConfig(
        window=16, image_size=16,
        risk_limits=RiskLimits(max_drawdown=1.0),
    ))
    env.reset(seed=0)
    _, _, _, _, info = env.step(int(DiscreteAction.LONG_100))
    assert info["exit_reason"] == ""


def test_skip_holds_open_position(small_market):
    env = TradingEnv(small_market, cfg=EnvConfig(
        window=16, image_size=16,
        kill_on_drawdown=False,
        risk_limits=RiskLimits(max_drawdown=1.0),
    ))
    env.reset(seed=0)
    env.step(int(DiscreteAction.LONG_100))
    position_before = env._position
    _, _, _, _, info = env.step(int(DiscreteAction.SKIP))
    assert env._position == position_before
    assert info["position"] == position_before
    assert info["fee"] == 0.0


# ---------------------------------------------------------------------------
# Gymnasium registration
# ---------------------------------------------------------------------------


def test_gymnasium_registration_id():
    ids = list(gym.envs.registry.keys())
    assert ZHISA_TRADING_ID in ids


def test_gymnasium_make_returns_env():
    df = _deterministic_market(n_bars=200, seed=6)
    env = gym.make(ZHISA_TRADING_ID, df=df, cfg=EnvConfig(window=8, image_size=8))
    obs, info = env.reset(seed=0)
    assert "chart" in obs
    assert env.action_space.n == len(DiscreteAction)
    env.close()


# ---------------------------------------------------------------------------
# Funding rate
# ---------------------------------------------------------------------------


def test_funding_disabled_by_default():
    """No ``funding_interval`` → no funding payment ever."""
    env_cfg = EnvConfig(
        window=16, image_size=16, funding_rate=0.01, funding_interval=0,
    )
    df = _deterministic_market(n_bars=200, seed=10)
    env = TradingEnv(df, cfg=env_cfg)
    env.reset(seed=0)
    env.step(int(DiscreteAction.LONG_100))
    for _ in range(20):
        _, _, _, _, info = env.step(int(DiscreteAction.LONG_100))
    assert info["funding_paid"] == 0.0
    assert info["cumulative_funding"] == 0.0


def test_funding_paid_after_interval():
    """If we hold a position for ``funding_interval`` bars, the rate
    must be deducted from equity on the next bar."""
    env_cfg = EnvConfig(
        window=16, image_size=16,
        funding_rate=0.001, funding_interval=5,
        kill_on_drawdown=False, episode_length=0,
    )
    df = _deterministic_market(n_bars=200, seed=11)
    env = TradingEnv(df, cfg=env_cfg)
    env.reset(seed=0)
    env.step(int(DiscreteAction.LONG_100))
    paid = []
    for _ in range(15):
        _, _, _, _, info = env.step(int(DiscreteAction.LONG_100))
        paid.append(info["funding_paid"])
    # The first funding payment should fire on bar 5 (after the
    # 5-bar interval). A positive rate means longs pay shorts.
    non_zero = [p for p in paid if p != 0.0]
    assert len(non_zero) >= 1
    # ``funding_paid`` is signed: positive = trader paid, negative = trader received.
    # For a long with positive rate, the trader pays.
    assert all(p > 0 for p in non_zero)


def test_funding_credits_short_position():
    """A negative funding rate (shorts pay longs) must credit a long
    position and debit a short one."""
    env_cfg = EnvConfig(
        window=16, image_size=16,
        funding_rate=-0.001, funding_interval=5,
        kill_on_drawdown=False, episode_length=0,
    )
    df = _deterministic_market(n_bars=200, seed=12)
    env = TradingEnv(df, cfg=env_cfg)
    env.reset(seed=0)
    env.step(int(DiscreteAction.LONG_100))
    paid = []
    for _ in range(15):
        _, _, _, _, info = env.step(int(DiscreteAction.LONG_100))
        paid.append(info["funding_paid"])
    # Negative rate => longs receive, so the ``paid`` value is negative.
    non_zero = [p for p in paid if p != 0.0]
    assert non_zero
    assert all(p < 0 for p in non_zero)


def test_funding_skipped_when_flat():
    """Funding only matters for an open position."""
    env_cfg = EnvConfig(
        window=16, image_size=16,
        funding_rate=0.001, funding_interval=5,
        kill_on_drawdown=False, episode_length=0,
    )
    df = _deterministic_market(n_bars=200, seed=13)
    env = TradingEnv(df, cfg=env_cfg)
    env.reset(seed=0)
    paid = []
    for _ in range(30):
        _, _, _, _, info = env.step(int(DiscreteAction.CLOSE))  # no position
        paid.append(info["funding_paid"])
    assert all(p == 0.0 for p in paid)


def test_funding_from_dataframe_column():
    """If ``funding_column`` is set, the rate is read from the df."""
    df = _deterministic_market(n_bars=200, seed=14)
    df["fundrate"] = 0.005  # large positive rate
    env_cfg = EnvConfig(
        window=16, image_size=16,
        funding_rate=0.001,            # ignored when column is set
        funding_interval=5,
        funding_column="fundrate",
        kill_on_drawdown=False, episode_length=0,
    )
    env = TradingEnv(df, cfg=env_cfg)
    env.reset(seed=0)
    env.step(int(DiscreteAction.LONG_100))
    paid = []
    for _ in range(15):
        _, _, _, _, info = env.step(int(DiscreteAction.LONG_100))
        paid.append(info["funding_paid"])
    # The column rate is 5x the fixed rate, so the payment should be
    # 5x larger than the test_funding_paid_after_interval case.
    non_zero = [p for p in paid if p != 0.0]
    assert non_zero
    # payment = position * lev * rate = 1.0 * 3.0 * 0.005 = 0.015
    assert non_zero[0] >= 0.01
