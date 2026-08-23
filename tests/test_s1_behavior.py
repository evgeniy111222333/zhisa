"""Unit tests for the S1 behavioural battery helpers/threshold logic."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from zhisa.models.policy import PolicyConfig, PolicyNetwork
from zhisa.scripts.s1_behavior_check import _ang, _norm


def test_ang_and_norm():
    a = torch.zeros(4, 8)
    a[:, 0] = 1.0
    b = torch.zeros(4, 8)
    b[:, 0] = torch.cos(torch.tensor(0.2))
    b[:, 1] = torch.sin(torch.tensor(0.2))
    assert _ang(a, b) == pytest.approx(0.2 * 180.0 / np.pi, abs=0.02)
    assert torch.allclose(_norm(a).norm(dim=-1), torch.ones(4))


def test_determinism_and_no_collapse_on_tiny_model():
    torch.manual_seed(0)
    model = PolicyNetwork(PolicyConfig(
        image_size=8, in_numeric_features=12, in_context_features=6, window=8,
        embed_dim=16, n_actions=9, n_regime_classes=2, n_instruments=2,
        vision_channels=(4, 8), numeric_layers=1, fusion_layers=1,
        memory_layers=1, memory_max_len=8, vision_mode="cnn", use_memory=True,
    )).eval()
    chart = torch.randn(16, 3, 8, 8)
    num = torch.randn(16, 8, 12)
    ctx = torch.randn(16, 6)
    ids = torch.arange(16) % 2
    with torch.no_grad():
        z = model.encode(chart, num, ctx, instrument_id=ids)
        z2 = model.encode(chart, num, ctx, instrument_id=ids)
    assert float((z - z2).abs().max()) < 1e-6          # determinism (b8)
    norms = z.norm(dim=-1)
    assert float(norms.std() / norms.mean()) > 0.01    # not constant (b6)
    zn = _norm(z)
    off = (zn @ zn.t())[~torch.eye(16, dtype=torch.bool)]
    assert float(off.mean().abs()) < 0.99              # not collapsed (b6)


def test_instrument_probe_separable():
    """1-NN probe must hit >=0.9 on two separable embedding clusters (b4 logic)."""
    torch.manual_seed(1)
    za = torch.randn(24, 64) * 0.1 + torch.full((1, 64), 1.0)
    zb = torch.randn(24, 64) * 0.1 - torch.full((1, 64), 1.0)
    Z = torch.cat([_norm(za), _norm(zb)], dim=0)
    labels = torch.cat([torch.zeros(24), torch.ones(24)]).long()
    sim = Z @ Z.t()
    sim = sim - torch.eye(sim.size(0)) * 1e9
    pred = labels[sim.argmax(dim=-1)]
    assert float((pred == labels).float().mean()) >= 0.95