"""Tests for the model encoders, fusion, memory, heads, and policy."""
from __future__ import annotations

import pytest
import torch

from zhisa.models.encoders.context import ContextEncoder
from zhisa.models.encoders.numeric import NumericEncoder, NumericEncoderConfig
from zhisa.models.encoders.vision import VisionEncoder, VisionEncoderConfig
from zhisa.models.fusion import CrossModalFusion
from zhisa.models.heads import MultiTaskHeads
from zhisa.models.memory import WorkingMemory
from zhisa.models.policy import PolicyConfig, PolicyNetwork, build_default_policy


def test_vision_encoder_shapes():
    enc = VisionEncoder(VisionEncoderConfig(image_size=32, out_dim=64))
    x = torch.rand(4, 3, 32, 32)
    y = enc(x)
    assert y.shape == (4, 64)


def test_numeric_encoder_shapes():
    enc = NumericEncoder(NumericEncoderConfig(in_features=8, window=16, patch_size=4, out_dim=32))
    x = torch.rand(4, 16, 8)
    cls, tokens = enc(x)
    assert cls.shape == (4, 32)
    # tokens include CLS + n_patches; their feature dim is d_model (pre-output projection)
    assert tokens.shape[0] == 4
    assert tokens.shape[2] == 128  # d_model
    assert tokens.shape[1] == 16 // 4 + 1  # n_patches + 1


def test_context_encoder_shapes():
    enc = ContextEncoder()
    x = torch.rand(3, 10)
    y = enc(x)
    assert y.shape == (3, 64)


def test_fusion_shapes():
    f = CrossModalFusion()
    v = torch.rand(2, 128)
    n = torch.rand(2, 128)
    c = torch.rand(2, 128)
    out = f(v, n, c)
    assert out.shape == (2, 128)


def test_memory_shapes():
    m = WorkingMemory()
    x = torch.rand(2, 16, 128)
    y = m(x)
    assert y.shape == (2, 16, 128)


def test_heads_shapes():
    h = MultiTaskHeads()
    z = torch.rand(2, 128)
    out = h(z)
    assert out["direction"].shape == (2, 3)
    assert out["regime"].shape == (2, 4)
    assert out["policy_logits"].shape == (2, 9)
    assert out["value"].shape == (2,)
    assert out["volatility"].shape == (2,)


def test_policy_forward_shapes():
    model = build_default_policy(
        in_numeric_features=20, in_context_features=10,
        window=32, image_size=32, n_actions=9, n_regime_classes=4,
    )
    chart = torch.rand(2, 3, 32, 32)
    numeric = torch.rand(2, 32, 20)
    context = torch.rand(2, 10)
    out = model(chart=chart, numeric=numeric, context=context)
    assert "embedding" in out
    assert out["embedding"].shape == (2, 128)
    assert out["policy_logits"].shape == (2, 9)


def test_policy_backward_step():
    model = build_default_policy(in_numeric_features=12, in_context_features=8,
                                  window=16, image_size=16, n_actions=5)
    chart = torch.rand(3, 3, 16, 16)
    numeric = torch.rand(3, 16, 12)
    context = torch.rand(3, 8)
    out = model(chart=chart, numeric=numeric, context=context)
    loss = out["policy_logits"].sum() + out["value"].sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
