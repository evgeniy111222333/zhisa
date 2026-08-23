"""PortfolioTrunkPolicy: S1-trunk integration tests + local A/B probe."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from zhisa.env.trading_env import EnvConfig, TradingEnv
from zhisa.models.policy import PolicyConfig, PolicyNetwork
from zhisa.models.portfolio_trunk_policy import PortfolioTrunkConfig, PortfolioTrunkPolicy


def _backbone(**kw) -> PolicyConfig:
    defaults = dict(
        embed_dim=64, window=32, image_size=32, in_numeric_features=32,
        in_context_features=10, n_instruments=4, vision_mode="cnn",
        vision_channels=(16, 32), numeric_layers=1, fusion_layers=1,
        encoder_ff_mult=2.0, memory_layers=1,
    )
    defaults.update(kw)
    return PolicyConfig(**defaults)


def _policy(**kw) -> PortfolioTrunkPolicy:
    defaults = dict(n_instruments=3, portfolio_dim=8, fusion_hidden=32, n_actions_per=9)
    defaults.update(kw)
    cfg = PortfolioTrunkConfig(backbone=_backbone(), **defaults)
    return PortfolioTrunkPolicy(cfg)


def _obs(B=2, N=3, seed=0):
    rng = np.random.default_rng(seed)
    return (
        torch.as_tensor(rng.random((B, N, 3, 32, 32)), dtype=torch.float32),
        torch.as_tensor(rng.normal(0, 1, (B, N, 32, 32)), dtype=torch.float32),
        torch.as_tensor(rng.normal(0, 1, (B, N, 10)), dtype=torch.float32),
    )


def test_forward_shapes_and_determinism():
    p = _policy(cross_attn_depth=2)
    p.eval()
    c, n, x = _obs()
    with torch.no_grad():
        o1 = p(c, n, x, portfolio=torch.zeros(2, 8), instrument_ids=torch.arange(3).repeat(2, 1))
        o2 = p(c, n, x, portfolio=torch.zeros(2, 8), instrument_ids=torch.arange(3).repeat(2, 1))
    assert o1["action_logits"].shape == (2, 3, 9)
    assert o1["value"].shape == (2,)
    assert o1["regime_logits"].shape == (2, 4)
    assert o1["embedding"].shape == (2, 3, 64)
    assert torch.equal(o1["action_logits"], o2["action_logits"])


def test_no_attention_path_and_bias_guard():
    p = _policy(cross_attn_depth=0, n_instruments=2)
    c, n, x = _obs(B=1, N=2)
    with torch.no_grad():
        o = p(c, n, x, portfolio=torch.zeros(1, 8))
    assert o["action_logits"].shape == (1, 2, 9)
    with pytest.raises(ValueError, match="corr_bias"):
        p(c, n, x, portfolio=None, corr_bias=torch.zeros(1, 2, 2))
    with pytest.raises(ValueError, match="n_instruments"):
        p(*_obs(B=1, N=3), portfolio=torch.zeros(1, 8))


def test_warm_start_copies_s1_trunk():
    # 1) standalone S1-style policy (same shapes) -> fake checkpoint
    s1 = PolicyNetwork(_backbone())
    path = str(__import__("tempfile").mkdtemp()) + "/s1_fake.pt"
    torch.save({"model": s1.state_dict()}, path)
    # 2) portfolio with the same backbone
    p = _policy(cross_attn_depth=0)
    stats = p.warm_start_from_s1(path)
    assert stats["copied"] > 0
    # every copied trunk weight must be pixel-equal to the S1 source
    s1_sd = s1.state_dict()
    p_sd = p.state_dict()
    encoders = ("vision", "numeric", "context", "fusion")
    n_eq = 0
    for k, v in s1_sd.items():
        if k.split(".")[0] not in encoders:
            continue  # heads stay fresh by design
        t = "trunk." + k
        if t in p_sd and tuple(p_sd[t].shape) == tuple(v.shape):
            assert torch.equal(p_sd[t], v), f"trunk weight {k} not copied exactly"
            n_eq += 1
    assert n_eq == stats["copied"]


def _cross_dependence(p: PortfolioTrunkPolicy, bias_on: bool) -> float:
    """How much instrument-0 action logits move when instrument-2 numeric
    changes (strong bias seeds point attending instrument 2 to A)."""
    p.eval()
    c, n, x = _obs(B=1, N=3, seed=1)
    ids = torch.arange(3).repeat(1, 1)
    portf = torch.zeros(1, 8)
    with torch.no_grad():
        base = p(c, n, x, portfolio=portf, instrument_ids=ids,
                 corr_bias=(torch.zeros(1, 3, 3) if bias_on else None))["action_logits"]
        n2 = n.clone()
        n2[:, 2] = n2[:, 2] + 5.0
        bias = None
        if bias_on:
            bias = torch.zeros(1, 3, 3)
            bias[:, 0, 2] = 50.0  # instrument 0 heavily attends instrument 2
        pert = p(c, n2, x, portfolio=portf, instrument_ids=ids, corr_bias=bias)["action_logits"]
    d0 = float((pert[:, 0] - base[:, 0]).abs().mean().item())
    d1 = float((pert[:, 1] - base[:, 1]).abs().mean().item())
    return d0, max(d1, 1e-8)


def test_ab_attention_vs_bias_practical():
    """Local A/B: with cross-attention, instrument-0 reacts to instrument-2
    perturbations; an additive bias seeded ON instrument-2 amplifies it."""
    noattn = _policy(cross_attn_depth=0, use_attention_bias=False)
    attn = _policy(cross_attn_depth=2, use_attention_bias=False)
    attn_bias = _policy(cross_attn_depth=2, use_attention_bias=True, bias_gate=5.0)
    for m in (noattn, attn, attn_bias):
        m.eval()
    d_no, _ = _cross_dependence(noattn, False)
    d_att, _ = _cross_dependence(attn, False)
    d_bias, _ = _cross_dependence(attn_bias, True)
    # attention exposes the other instrument; bias seeds it further
    assert d_att > d_no, f"attention should increase cross-instrument dependence: {d_att} vs {d_no}"
    assert d_bias > d_att, f"bias should amplify the attended instrument: {d_bias} vs {d_att}"


def test_trading_env_prepared_numeric_mode():
    rng = np.random.default_rng(0)
    idx = pd = __import__("pandas").date_range("2024-01-01", periods=120, freq="1h", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.standard_normal(120) * 0.01))
    extra = {"beta_64": rng.normal(1, 0.2, 120), "corr_64": rng.uniform(-1, 1, 120),
             "rel_logret_1": rng.normal(0, 0.01, 120)}
    df = __import__("pandas").DataFrame(
        {"open": close, "high": close * 1.002, "low": close * 0.998, "close": close,
         "volume": rng.uniform(1, 2, 120), **extra}, index=idx)
    env = TradingEnv(df, cfg=EnvConfig(
        window=32, image_size=32, use_prepared_numeric=True,
        prepared_feature_columns=("beta_64", "corr_64", "rel_logret_1"),
    ))
    obs, _ = env.reset()
    assert env.obs_numeric_dim == 3
    assert obs["numeric"].shape == (32, 3)
    assert np.isfinite(obs["numeric"]).all()
    # missing column -> loud error
    with pytest.raises(ValueError, match="missing columns"):
        TradingEnv(df, cfg=EnvConfig(
            window=32, image_size=32, use_prepared_numeric=True,
            prepared_feature_columns=("beta_64", "ghost"),
        ))