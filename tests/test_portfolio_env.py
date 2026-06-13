"""Tests for the multi-instrument PortfolioEnv."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest

from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.env.actions import DiscreteAction
from zhisa.env.portfolio_env import (
    PortfolioConfig,
    PortfolioEnv,
    ZHISA_PORTFOLIO_ID,
    decode_multi_action,
    encode_multi_action,
)
from zhisa.env.trading_env import EnvConfig
from zhisa.risk.limits import RiskLimits


# ---------------------------------------------------------------------------
# Action encoding
# ---------------------------------------------------------------------------


def test_encode_decode_round_trip():
    for n in (1, 2, 3, 4):
        # Only codes that fit in base-9 with N digits round-trip.
        for code in (0, 1, 9 ** (n - 1), 9 ** n - 1, 9 ** n // 2):
            if code >= 9 ** n:
                continue  # not representable in N digits
            digits = decode_multi_action(code, n)
            assert len(digits) == n
            assert all(0 <= d < 9 for d in digits)
            assert encode_multi_action(digits) == code


def test_encode_zero_is_all_skip():
    """Action 0 must correspond to SKIP on every instrument."""
    digits = decode_multi_action(0, 3)
    assert digits == [0, 0, 0]
    assert all(DiscreteAction(d) == DiscreteAction.SKIP for d in digits)
    assert encode_multi_action(digits) == 0


def test_decode_single_instrument():
    """For N=1 the decode is just the identity."""
    for code in range(9):
        assert decode_multi_action(code, 1) == [code]


# ---------------------------------------------------------------------------
# Environment basics
# ---------------------------------------------------------------------------


def _two_markets(n_bars: int = 200, seeds=(0, 1)) -> dict:
    return {
        "a": generate_market(MarketConfig(n_bars=n_bars, seed=seeds[0])),
        "b": generate_market(MarketConfig(n_bars=n_bars, seed=seeds[1])),
    }


def test_portfolio_env_rejects_single_instrument():
    with pytest.raises(ValueError):
        PortfolioEnv({"only": generate_market(MarketConfig(n_bars=100, seed=0))})


def test_portfolio_env_rejects_empty_data():
    with pytest.raises(ValueError):
        PortfolioEnv({})


def test_portfolio_env_observation_shapes():
    cfg = PortfolioConfig(
        env_cfg=EnvConfig(window=8, image_size=8, risk_limits=RiskLimits(max_drawdown=1.0)),
    )
    env = PortfolioEnv(_two_markets(), cfg=cfg)
    obs, info = env.reset(seed=0)
    assert "instruments" in obs
    assert "portfolio" in obs
    assert len(obs["instruments"]) == 2
    for sub in obs["instruments"]:
        assert sub["chart"].shape == (3, 8, 8)
        assert sub["numeric"].shape[0] == 8
    assert obs["portfolio"].ndim == 1
    assert np.isfinite(obs["portfolio"]).all()


def test_portfolio_env_action_space():
    env = PortfolioEnv(_two_markets(), cfg=PortfolioConfig(
        env_cfg=EnvConfig(window=8, image_size=8, risk_limits=RiskLimits(max_drawdown=1.0)),
    ))
    assert env.action_space.n == 9 ** 2
    # Sanity: 0 and 80 (= 9^2 - 1) are both valid.
    assert env.action_space.contains(0)
    assert env.action_space.contains(9 ** 2 - 1)
    assert not env.action_space.contains(9 ** 2)


def test_portfolio_env_action_validation():
    env = PortfolioEnv(_two_markets(), cfg=PortfolioConfig(
        env_cfg=EnvConfig(window=8, image_size=8, risk_limits=RiskLimits(max_drawdown=1.0)),
    ))
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(9 ** 2 + 5)


def test_portfolio_env_random_rollout_completes():
    cfg = PortfolioConfig(
        env_cfg=EnvConfig(
            window=8, image_size=8,
            risk_limits=RiskLimits(max_drawdown=1.0),
            kill_on_drawdown=False,
        ),
    )
    env = PortfolioEnv(_two_markets(n_bars=200), cfg=cfg)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = int(rng.integers(0, 9 ** 2))
        obs, r, term, trunc, info = env.step(a)
        assert np.isfinite(r)
        assert np.isfinite(info["equity"])
        if term or trunc:
            break


def test_portfolio_env_close_action_closes_all():
    """A CLOSE action on every instrument closes all open positions."""
    cfg = PortfolioConfig(
        env_cfg=EnvConfig(window=8, image_size=8, risk_limits=RiskLimits(max_drawdown=1.0)),
    )
    env = PortfolioEnv(_two_markets(), cfg=cfg)
    env.reset(seed=0)
    # Open on both instruments.
    env.step(encode_multi_action([int(DiscreteAction.LONG_100),
                                    int(DiscreteAction.LONG_100)]))
    assert env._instrument_position(0) != 0
    assert env._instrument_position(1) != 0
    # Close.
    env.step(encode_multi_action([int(DiscreteAction.CLOSE),
                                    int(DiscreteAction.CLOSE)]))
    assert env._instrument_position(0) == 0
    assert env._instrument_position(1) == 0


def test_portfolio_env_skip_holds_open_positions():
    cfg = PortfolioConfig(
        env_cfg=EnvConfig(
            window=8, image_size=8,
            kill_on_drawdown=False,
            risk_limits=RiskLimits(max_drawdown=1.0),
        ),
    )
    env = PortfolioEnv(_two_markets(), cfg=cfg)
    env.reset(seed=0)
    env.step(encode_multi_action([int(DiscreteAction.LONG_100),
                                  int(DiscreteAction.SHORT_100)]))
    before = [env._instrument_position(0), env._instrument_position(1)]
    _, _, _, _, info = env.step(encode_multi_action([int(DiscreteAction.SKIP),
                                                     int(DiscreteAction.SKIP)]))
    assert [env._instrument_position(0), env._instrument_position(1)] == before
    assert info["per_instrument_position"] == before


def test_portfolio_env_gross_leverage_caps_action():
    """The gross leverage cap must reject an action that would push
    combined leverage above the limit."""
    cfg = PortfolioConfig(
        env_cfg=EnvConfig(
            window=8, image_size=8,
            max_leverage=3.0,
            risk_limits=RiskLimits(max_drawdown=1.0),
        ),
        gross_leverage_cap=2.0,   # both LONG_100 (3.0 each) would be 6.0
    )
    env = PortfolioEnv(_two_markets(), cfg=cfg)
    env.reset(seed=0)
    # Both LONG_100 -> proposed gross = 6.0, above cap of 2.0.
    # Should be held (positions stay at 0).
    env.step(encode_multi_action([int(DiscreteAction.LONG_100),
                                    int(DiscreteAction.LONG_100)]))
    assert env._instrument_position(0) == 0
    assert env._instrument_position(1) == 0


def test_portfolio_risk_rejection_holds_existing_positions():
    cfg = PortfolioConfig(
        env_cfg=EnvConfig(
            window=8, image_size=8,
            max_leverage=1.0,
            risk_limits=RiskLimits(max_drawdown=1.0),
            kill_on_drawdown=False,
        ),
        gross_leverage_cap=1.5,
    )
    env = PortfolioEnv(_two_markets(), cfg=cfg)
    env.reset(seed=0)
    env.step(encode_multi_action([int(DiscreteAction.LONG_50),
                                  int(DiscreteAction.LONG_50)]))
    before = [env._instrument_position(0), env._instrument_position(1)]
    assert before == [0.5, 0.5]

    # Moving both legs to LONG_100 would breach the gross cap and must
    # be rejected as a no-op, not converted to a close.
    env.step(encode_multi_action([int(DiscreteAction.LONG_100),
                                  int(DiscreteAction.LONG_100)]))
    assert [env._instrument_position(0), env._instrument_position(1)] == before


def test_portfolio_env_portfolio_summary_components():
    """The portfolio summary has the right dimensionality."""
    cfg = PortfolioConfig(
        env_cfg=EnvConfig(
            window=8, image_size=8, risk_limits=RiskLimits(max_drawdown=1.0),
        ),
        n_corr_features=1,   # 2-instrument -> 1 upper-tri element
    )
    env = PortfolioEnv(_two_markets(), cfg=cfg)
    obs, _ = env.reset(seed=0)
    # 2 instruments * 3 (pos / equity / dd) + 1 cov + 2 (gross + unrealised) = 9
    assert obs["portfolio"].shape == (9,)


def test_portfolio_env_correlation_feature_is_finite():
    cfg = PortfolioConfig(
        env_cfg=EnvConfig(window=8, image_size=8, risk_limits=RiskLimits(max_drawdown=1.0)),
        correlation_window=20,
    )
    env = PortfolioEnv(_two_markets(n_bars=200), cfg=cfg)
    env.reset(seed=0)
    for _ in range(30):
        env.step(encode_multi_action([int(DiscreteAction.LONG_100),
                                       int(DiscreteAction.LONG_100)]))
        obs, _, _, _, _ = env.step(encode_multi_action([int(DiscreteAction.LONG_100),
                                                          int(DiscreteAction.LONG_100)]))
    # No NaN/Inf in the portfolio summary after a few steps.
    assert np.isfinite(obs["portfolio"]).all()


def test_portfolio_env_per_instrument_equity_in_info():
    cfg = PortfolioConfig(
        env_cfg=EnvConfig(window=8, image_size=8, risk_limits=RiskLimits(max_drawdown=1.0)),
    )
    env = PortfolioEnv(_two_markets(), cfg=cfg)
    env.reset(seed=0)
    env.step(encode_multi_action([int(DiscreteAction.LONG_100),
                                    int(DiscreteAction.SHORT_100)]))
    _, _, _, _, info = env.step(0)
    assert "per_instrument_equity" in info
    assert len(info["per_instrument_equity"]) == 2
    assert "exit_reasons" in info
    assert len(info["exit_reasons"]) == 2


def test_portfolio_env_kill_switch_terminates():
    """A 20% drop on one instrument should trigger the portfolio
    drawdown kill-switch (after a 3x leveraged position the realised
    loss is ~60%, well past the 10% max_drawdown)."""
    cfg = PortfolioConfig(
        env_cfg=EnvConfig(
            window=8, image_size=8,
            risk_limits=RiskLimits(max_drawdown=0.10),
            kill_on_drawdown=True,
        ),
    )
    markets = _two_markets(n_bars=200)
    env = PortfolioEnv(markets, cfg=cfg)
    env.reset(seed=0)
    # Open a long on instrument "a" only.
    env.step(encode_multi_action([int(DiscreteAction.LONG_100), int(DiscreteAction.CLOSE)]))
    # Force a 20% drop on the bar the MTM will read (post-advance
    # index, which is ``sub._t + 1``).
    sub = env._envs[0]
    entry = sub._avg_entry
    target_bar = sub._t + 1
    sub.df.loc[target_bar, "low"] = entry * 0.80
    sub.df.loc[target_bar, "close"] = entry * 0.80
    _, _, term, _, info = env.step(encode_multi_action(
        [int(DiscreteAction.LONG_100), int(DiscreteAction.CLOSE)]))
    assert term is True
    assert info["drawdown"] >= 0.10


def test_portfolio_env_episode_length_truncates():
    cfg = PortfolioConfig(
        env_cfg=EnvConfig(
            window=8, image_size=8,
            risk_limits=RiskLimits(max_drawdown=1.0),
            episode_length=10,
        ),
    )
    env = PortfolioEnv(_two_markets(n_bars=400), cfg=cfg)
    env.reset(seed=0)
    truncated = False
    for _ in range(20):
        _, _, _, trunc, _ = env.step(0)
        if trunc:
            truncated = True
            break
    assert truncated


# ---------------------------------------------------------------------------
# Gymnasium registration
# ---------------------------------------------------------------------------


def test_gymnasium_registration_id():
    ids = list(gym.envs.registry.keys())
    assert ZHISA_PORTFOLIO_ID in ids


def test_gymnasium_make_returns_env():
    markets = _two_markets(n_bars=200, seeds=(0, 1))
    cfg = PortfolioConfig(env_cfg=EnvConfig(window=8, image_size=8))
    env = gym.make(ZHISA_PORTFOLIO_ID, dataframes=markets, cfg=cfg)
    obs, info = env.reset(seed=0)
    assert "instruments" in obs
    assert "portfolio" in obs
    assert env.action_space.n == 9 ** 2
    env.close()


# ---------------------------------------------------------------------------
# Three-instrument sanity (combinatorial action space)
# ---------------------------------------------------------------------------


def test_three_instrument_action_space_size():
    cfg = PortfolioConfig(
        env_cfg=EnvConfig(window=8, image_size=8, risk_limits=RiskLimits(max_drawdown=1.0)),
    )
    data = {
        "x": generate_market(MarketConfig(n_bars=100, seed=0)),
        "y": generate_market(MarketConfig(n_bars=100, seed=1)),
        "z": generate_market(MarketConfig(n_bars=100, seed=2)),
    }
    env = PortfolioEnv(data, cfg=cfg)
    assert env.action_space.n == 9 ** 3
    obs, _ = env.reset(seed=0)
    assert len(obs["instruments"]) == 3
