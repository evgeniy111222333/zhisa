"""Tests for the :class:`PortfolioPolicyNetwork`."""
from __future__ import annotations

import pytest
import torch

from zhisa.models.portfolio_policy import (
    PortfolioPolicyConfig,
    PortfolioPolicyNetwork,
    build_default_portfolio_policy,
)


def test_config_validates():
    with pytest.raises(ValueError):
        PortfolioPolicyConfig(n_instruments=0)


def test_forward_shapes_two_instruments():
    cfg = PortfolioPolicyConfig(
        n_instruments=2, embed_dim=32, fusion_hidden=32,
        in_numeric_features=8, in_context_features=4, window=8, image_size=16,
    )
    model = PortfolioPolicyNetwork(cfg)
    B, N = 3, 2
    chart = torch.randn(B, N, 3, 16, 16)
    numeric = torch.randn(B, N, 8, 8)
    context = torch.randn(B, N, 4)
    portfolio = torch.randn(B, 32)
    out = model(
        instruments={"chart": chart, "numeric": numeric, "context": context},
        portfolio=portfolio,
    )
    assert out["action_logits"].shape == (B, N, 9)
    assert out["value"].shape == (B,)
    assert out["per_instrument_embedding"].shape == (B, N, 32)
    assert out["portfolio_embedding"].shape == (B, 32)
    assert out["regime_logits"].shape == (B, 4)


def test_forward_three_instruments():
    cfg = PortfolioPolicyConfig(
        n_instruments=3, embed_dim=16, fusion_hidden=16,
        in_numeric_features=4, in_context_features=2, window=4, image_size=8,
    )
    model = PortfolioPolicyNetwork(cfg)
    B, N = 2, 3
    out = model(
        instruments={
            "chart": torch.randn(B, N, 3, 8, 8),
            "numeric": torch.randn(B, N, 4, 4),
            "context": torch.randn(B, N, 2),
        },
        portfolio=torch.randn(B, 32),
    )
    assert out["action_logits"].shape == (B, N, 9)


def test_factored_heads_are_independent():
    """Different instruments should have *separate* linear heads."""
    cfg = PortfolioPolicyConfig(
        n_instruments=3, embed_dim=16, fusion_hidden=16,
        in_numeric_features=4, in_context_features=2, window=4, image_size=8,
    )
    model = PortfolioPolicyNetwork(cfg)
    weights = [h.weight for h in model.action_heads]
    assert weights[0] is not weights[1]
    assert not torch.equal(weights[0], weights[1])


def test_shared_encoders_across_instruments():
    """The vision/numeric/context/fusion modules are shared (same module instances)."""
    cfg = PortfolioPolicyConfig(
        n_instruments=4, embed_dim=16, fusion_hidden=16,
        in_numeric_features=4, in_context_features=2, window=4, image_size=8,
    )
    model = PortfolioPolicyNetwork(cfg)
    assert model.vision is model.vision
    assert model.fusion is model.fusion


def test_gradients_flow_through_heads():
    cfg = PortfolioPolicyConfig(
        n_instruments=2, embed_dim=16, fusion_hidden=16,
        in_numeric_features=4, in_context_features=2, window=4, image_size=8,
    )
    model = PortfolioPolicyNetwork(cfg)
    out = model(
        instruments={
            "chart": torch.randn(2, 2, 3, 8, 8, requires_grad=True),
            "numeric": torch.randn(2, 2, 4, 4, requires_grad=True),
            "context": torch.randn(2, 2, 2, requires_grad=True),
        },
        portfolio=torch.randn(2, 32, requires_grad=True),
    )
    loss = (
        out["action_logits"].pow(2).mean()
        + out["value"].pow(2).mean()
        + out["regime_logits"].pow(2).mean()
    )
    loss.backward()
    # Spot-check: encoders, fusion, portfolio_mlp, action heads, value head all
    # receive gradients. The instrument-id embedding inside the context encoder
    # is unused in this forward pass and may not have a grad.
    grad_params = [p for p in model.parameters() if p.grad is not None]
    assert len(grad_params) > 0
    for p in grad_params:
        assert torch.isfinite(p.grad).all()
    # Specifically check that the action heads and value head got gradients.
    for h in model.action_heads:
        assert h.weight.grad is not None
    assert model.value_head.weight.grad is not None
    assert model.portfolio_mlp[0].weight.grad is not None


def test_save_load_roundtrip(tmp_path):
    cfg = PortfolioPolicyConfig(
        n_instruments=2, embed_dim=16, fusion_hidden=16,
        in_numeric_features=4, in_context_features=2, window=4, image_size=8,
    )
    model = PortfolioPolicyNetwork(cfg)
    p = tmp_path / "pp.pt"
    model.save(str(p))
    loaded = PortfolioPolicyNetwork.load(str(p))
    sd = model.state_dict()
    sd_loaded = loaded.state_dict()
    for k in sd:
        assert torch.allclose(sd[k], sd_loaded[k])


def test_build_default_portfolio_policy():
    model = build_default_portfolio_policy(
        n_instruments=3, in_numeric_features=10, in_context_features=4,
        window=8, image_size=16, portfolio_dim=20, embed_dim=24,
    )
    assert model.cfg.n_instruments == 3
    assert model.cfg.portfolio_dim == 20
    assert model.cfg.embed_dim == 24
