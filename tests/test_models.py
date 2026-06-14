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
from zhisa.models.regime_policy import (
    EXECUTION_ORDER_TYPES,
    EXECUTION_URGENCIES,
    POSITION_INTENTS,
    RegimeAwarePolicyConfig,
    RegimeAwarePolicyNetwork,
    RegimePolicyAuxLoss,
    build_regime_policy_targets,
)
from zhisa.regime.encoder import RegimeEncoderConfig
from zhisa.regime import RegimeIntelligence, RegimeIntelligenceConfig, plan_trade
from zhisa.training.losses import MultiTaskLoss


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


def test_regime_aware_policy_appends_regime_context_and_backpropagates():
    cfg = RegimeAwarePolicyConfig(
        base_policy=PolicyConfig(
            in_numeric_features=12,
            in_context_features=8,
            window=16,
            image_size=16,
            embed_dim=32,
            n_actions=5,
            use_memory=False,
            dropout=0.0,
        ),
        regime_encoder=RegimeEncoderConfig(embed_dim=8, hidden_dim=24, dropout=0.0),
    )
    model = RegimeAwarePolicyNetwork(cfg)
    chart = torch.rand(3, 3, 16, 16)
    numeric = torch.rand(3, 16, 12)
    context = torch.rand(3, 8)
    regime = torch.rand(3, model.regime_encoder.input_dim)

    out = model(chart=chart, numeric=numeric, context=context, regime=regime)
    loss = (
        out["policy_logits"].sum()
        + out["value"].sum()
        + out["regime_embedding"].sum()
        + out["regime_playbook_logits"].sum()
        + out["regime_risk_budget"].sum()
        + out["regime_no_trade"].sum()
    )
    loss.backward()

    assert model.policy.cfg.in_context_features == 16
    assert out["policy_logits"].shape == (3, 5)
    assert out["regime_embedding"].shape == (3, 8)
    assert out["regime_macro_logits"].shape[0] == 3
    assert out["regime_playbook_logits"].shape[0] == 3
    assert out["regime_risk_budget"].shape == (3,)
    assert out["regime_tradeability"].shape == (3,)
    assert out["regime_transition_wait"].shape == (3,)
    assert out["regime_no_trade"].shape == (3,)
    assert out["regime_playbook_prior"].shape == (3,)
    assert out["execution_order_type_logits"].shape == (3, len(EXECUTION_ORDER_TYPES))
    assert out["execution_urgency_logits"].shape == (3, len(EXECUTION_URGENCIES))
    assert out["execution_reduce_only"].shape == (3,)
    assert out["execution_scale_in"].shape == (3,)
    assert out["execution_max_slippage"].shape == (3,)
    assert out["position_intent_logits"].shape == (3, len(POSITION_INTENTS))
    regime_grads = [p.grad for p in model.regime_encoder.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in regime_grads)


def test_regime_policy_targets_and_aux_loss_penalize_blocked_actions():
    cfg = RegimeAwarePolicyConfig(
        base_policy=PolicyConfig(
            in_numeric_features=12,
            in_context_features=8,
            window=16,
            image_size=16,
            embed_dim=32,
            n_actions=9,
            use_memory=False,
            dropout=0.0,
        ),
        regime_encoder=RegimeEncoderConfig(embed_dim=8, hidden_dim=24, dropout=0.0),
    )
    model = RegimeAwarePolicyNetwork(cfg)
    analyzer = RegimeIntelligence(RegimeIntelligenceConfig(timeframes=("5m", "15m")))
    close = torch.linspace(130.0, 80.0, 180).numpy()
    open_ = close.copy()
    high = close + 0.5
    low = close - 0.5
    import pandas as pd

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 400.0},
        index=pd.date_range("2026-01-01", periods=len(close), freq="5min", tz="UTC"),
    )
    report = analyzer.analyze(df)
    plan = plan_trade(report, n_actions=9)
    targets = build_regime_policy_targets([report, report], plans=[plan, plan], n_actions=9)

    chart = torch.rand(2, 3, 16, 16)
    numeric = torch.rand(2, 16, 12)
    context = torch.rand(2, 8)
    regime = torch.rand(2, model.regime_encoder.input_dim)
    out = model(chart=chart, numeric=numeric, context=context, regime=regime)

    losses = RegimePolicyAuxLoss()(out, targets)
    losses["total"].backward()

    assert targets["regime_action_mask"].shape == (2, 9)
    assert targets["regime_playbook_prior"].shape == (2,)
    assert targets["execution_order_type_label"].shape == (2,)
    assert targets["execution_urgency_label"].shape == (2,)
    assert targets["execution_reduce_only"].shape == (2,)
    assert targets["position_intent_label"].shape == (2,)
    assert losses["total"].item() > 0.0
    assert "action_constraint" in losses
    assert "playbook_prior" in losses
    assert "execution_order_type" in losses
    assert "execution_urgency" in losses
    assert "position_intent" in losses
    assert torch.isfinite(losses["total"])
    head_grads = [p.grad for p in model.regime_heads.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in head_grads)


def test_multitask_loss_consumes_regime_policy_targets():
    cfg = RegimeAwarePolicyConfig(
        base_policy=PolicyConfig(
            in_numeric_features=8,
            in_context_features=4,
            window=8,
            image_size=8,
            embed_dim=24,
            n_actions=9,
            use_memory=False,
            dropout=0.0,
        ),
        regime_encoder=RegimeEncoderConfig(embed_dim=6, hidden_dim=16, dropout=0.0),
    )
    model = RegimeAwarePolicyNetwork(cfg)
    out = model(
        chart=torch.rand(2, 3, 8, 8),
        numeric=torch.rand(2, 8, 8),
        context=torch.rand(2, 4),
        regime=torch.rand(2, model.regime_encoder.input_dim),
    )
    targets = {
        "label_dir": torch.tensor([1, -1]),
        "label_vol": torch.tensor([0.01, 0.02]),
        "label_regime": torch.tensor([0, 1]),
        "label_ret": torch.tensor([0.01, -0.02]),
        "action": torch.tensor([0, 1]),
        "regime_playbook_label": torch.tensor([1, 2]),
        "regime_playbook_prior": torch.tensor([0.8, 0.35]),
        "regime_risk_budget": torch.tensor([0.2, 0.5]),
        "regime_tradeability": torch.tensor([0.7, 0.3]),
        "regime_transition_wait": torch.tensor([0.0, 1.0]),
        "regime_no_trade": torch.tensor([0.0, 1.0]),
        "regime_size_multiplier": torch.tensor([0.5, 0.25]),
        "execution_order_type_label": torch.tensor([1, 2]),
        "execution_urgency_label": torch.tensor([2, 1]),
        "execution_reduce_only": torch.tensor([0.0, 1.0]),
        "execution_scale_in": torch.tensor([0.3, 0.7]),
        "execution_max_slippage": torch.tensor([0.2, 0.5]),
        "position_intent_label": torch.tensor([2, 3]),
        "regime_action_mask": torch.tensor(
            [
                [True, True, False, False, True, True, True, True, True],
                [True, False, True, True, True, True, False, True, True],
            ]
        ),
    }

    losses = MultiTaskLoss()(out, targets)
    losses["total"].backward()

    assert "regime_playbook" in losses
    assert "regime_playbook_prior" in losses
    assert "regime_action_constraint" in losses
    assert "execution_order_type" in losses
    assert "execution_urgency" in losses
    assert "execution_reduce_only" in losses
    assert "execution_scale_in" in losses
    assert "execution_slippage" in losses
    assert "position_intent" in losses
    assert torch.isfinite(losses["total"])
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.regime_heads.parameters())
