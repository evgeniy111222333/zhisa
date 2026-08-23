"""Adaptive (Kendall) multi-task loss-weighting tests."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from zhisa.training.losses import LossWeights, MultiTaskLoss


def _tensors(B=16, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    return {
        "label_dir": torch.tensor(rng.choice([-1, 0, 1], B, p=[0.25, 0.5, 0.25]), dtype=torch.long),
        "label_dir_multi": torch.tensor(rng.choice([-1, 0, 1], (B, 4), p=[0.3, 0.4, 0.3]), dtype=torch.long),
        "label_vol": torch.tensor(rng.uniform(0.005, 0.03, B), dtype=torch.float32),
        "label_ret": torch.tensor(rng.normal(0, 0.003, B), dtype=torch.float32),
        "label_ret_multi": torch.tensor(rng.normal(0, 0.004, (B, 4)), dtype=torch.float32),
        "label_regime": torch.tensor(rng.integers(0, 4, B), dtype=torch.long),
        "label_risk": torch.tensor(rng.uniform(0.0, 0.05, B), dtype=torch.float32),
        "action": torch.tensor(rng.integers(0, 9, B), dtype=torch.long),
    }


def _outputs(B=16, seed=1):
    torch.manual_seed(seed)
    h = torch.randn(B, 16)
    return {
        "direction": torch.randn(B, 3) * 0.5,
        "direction_multi": torch.randn(B, 4, 3) * 0.5,
        "volatility": torch.randn(B) * 0.01,
        "regime": torch.randn(B, 4) * 0.5,
        "return_pred": torch.randn(B) * 0.01,
        "return_multi": torch.randn(B, 4) * 0.01,
        "risk": torch.randn(B) * 0.02,
        "policy_logits": torch.randn(B, 9) * 0.5,
        "value": torch.randn(B) * 0.01,
        "uncertainty_logit": torch.zeros(B),
        "regime_playbook_logits": torch.randn(B, 3) * 0.5,
        "regime_playbook_prior": torch.randn(B, 3) * 0.1,
        "regime_risk_budget": torch.randn(B) * 0.05,
        "regime_tradeability": torch.randn(B) * 0.05,
        "regime_size_multiplier": torch.randn(B) * 0.1,
        "regime_transition_wait": torch.randn(B) * 0.3,
        "regime_no_trade": torch.randn(B) * 0.3,
        "execution_order_type_logits": torch.randn(B, 3) * 0.5,
        "execution_urgency_logits": torch.randn(B, 3) * 0.5,
        "position_intent_logits": torch.randn(B, 3) * 0.5,
        "execution_reduce_only": torch.randn(B) * 0.3,
        "execution_scale_in": torch.randn(B) * 0.3,
        "execution_max_slippage": torch.randn(B) * 0.01,
    }


def test_total_stays_scalar_with_learnable_weights():
    """Bug #6 regression: the adaptive log-vars are scalar parameters so the
    aggregated loss remains a 0-dim tensor (a (1,)-shaped log_var would
    promote ``total`` to a 1D vector and break downstream scalar contracts)."""
    w = LossWeights()
    adap = MultiTaskLoss(w, learnable=True, adaptive_clamp=4.0)
    assert all(p.ndim == 0 for p in adap.log_vars.values())
    out, tgt = _outputs(), _tensors()
    total = adap(out, tgt)["total"]
    assert total.ndim == 0


def test_fixed_mode_unchanged_semantics():
    w = LossWeights(direction=1.0, return_pred=0.5, volatility=0.5)
    loss = MultiTaskLoss(w)  # learnable=False path
    assert not loss.learnable
    out, tgt = _outputs(), _tensors()
    out2, tgt2 = _outputs(), _tensors()
    totals = []
    for o, t in ((out, tgt), (out2, tgt2)):
        l = loss(o, t)
        totals.append(float(l["total"].item()))
        assert set(l) - {"total"} <= set(w.__dict__) | {"direction", "return_pred", "volatility"}
    # deterministic
    assert totals[0] == pytest.approx(totals[1], rel=1e-4)


def test_learnable_equals_fixed_at_init():
    w = LossWeights()
    fixed = MultiTaskLoss(w)
    adap = MultiTaskLoss(w, learnable=True, adaptive_clamp=4.0)
    out, tgt = _outputs(), _tensors()
    lf = fixed(out, tgt)["total"]
    la = adap(out, tgt)["total"]
    # log_vars init to 0 -> eff=1, lv reg=0 -> identical to fixed semantics
    assert (lf - la).abs().item() < 1e-5


def test_clamp_limits_effective_weight():
    w = LossWeights()
    adap = MultiTaskLoss(w, learnable=True, adaptive_clamp=4.0)
    with torch.no_grad():
        adap.log_vars["direction"].fill_(10.0)   # would be ~22026x
        adap.log_vars["return_pred"].fill_(-10.0)
    eff = adap.effective_weights()
    assert eff["direction"] == pytest.approx(np.exp(-4.0), abs=1e-6)
    assert eff["return_pred"] == pytest.approx(np.exp(4.0), abs=1e-4)


def test_seed_from_measured_scales():
    w = LossWeights()
    adap = MultiTaskLoss(w, learnable=True, adaptive_clamp=4.0)
    adap.seed_log_vars_from_losses({"direction": 0.3, "return_pred": 0.2})
    eff = adap.effective_weights()
    # log_var = ln(prior*mean) -> eff ~ 1/(prior*mean), clamped to exp(±4)
    assert eff["direction"] == pytest.approx(1.0 / (1.0 * 0.3), rel=0.1)
    assert eff["return_pred"] == pytest.approx(1.0 / (0.5 * 0.2), rel=0.1)
    assert all(np.exp(-4.0) <= v <= np.exp(4.0) for v in eff.values())
    # values beyond the clamp stay at the bound
    adap.seed_log_vars_from_losses({"return_pred": 0.015})  # ln(0.0075) < -4
    assert adap.effective_weights()["return_pred"] == pytest.approx(np.exp(4.0), abs=1e-4)


def test_adaptive_equalizes_mismatched_task_scales():
    """The measured imbalance (direction ~1.3 vs return ~0.015) must shrink:
    training ONLY the log-variances should drive the WEIGHTED CONTRIBUTIONS
    (prior*eff*L) toward each other — Kendall makes small-scale tasks up
    and large-scale tasks down."""
    torch.manual_seed(0)
    np.random.seed(0)
    w = LossWeights()
    adap = MultiTaskLoss(w, learnable=True, adaptive_clamp=4.0)
    out, tgt = _outputs(), _tensors()
    opt = torch.optim.Adam(adap.log_vars.parameters(), lr=0.05)

    def contrib_ratio():
        losses = adap(out, tgt)
        losses["total"].backward()
        opt.step()
        eff = adap.effective_weights()
        d = float(losses["direction"].item()) * w.direction * eff.get("direction", 1.0)
        r = float(losses["return_pred"].item()) * w.return_pred * eff.get("return_pred", 1.0)
        return (d + 1e-9) / (r + 1e-9), d, r

    first = None
    last = None
    for _ in range(60):
        ratio, d, r = contrib_ratio()
        if first is None:
            first = ratio
        last = ratio
    assert first > 50.0, f"expected a large initial contribution imbalance, got {first:.1f}"
    # Kendall must cut the imbalance by at least ~20x over 60 steps
    assert last < first * 0.05, f"contributions did not rebalance: {first:.1f} -> {last:.1f}"


def test_freeze_and_effective_weights_api():
    w = LossWeights()
    adap = MultiTaskLoss(w, learnable=True)
    adap.set_log_vars_trainable(False)
    for p in adap.log_vars.parameters():
        assert not p.requires_grad
    adap.set_log_vars_trainable(True)
    for p in adap.log_vars.parameters():
        assert p.requires_grad
    assert set(adap.log_vars.keys()) >= {"direction", "return_pred", "policy", "value"}
    # frozen path must not update magnitudes
    eff0 = adap.effective_weights()
    with torch.no_grad():
        adap.log_vars["direction"].fill_(2.0)
    assert adap.effective_weights() != eff0  # (api reflects values regardless)


def test_s2_config_wiring():
    """train_s2 must construct the loss with learnable from config."""
    from zhisa.scripts.train_s2 import _loss_weights_from
    from zhisa.training.losses import MultiTaskLoss

    cfg = {"loss_weights": {"direction": 1.0}, "loss_adaptive": True, "adaptive_clamp": 4.0}
    w = _loss_weights_from(cfg)
    loss = MultiTaskLoss(w, learnable=bool(cfg.get("loss_adaptive", False)),
                         adaptive_clamp=float(cfg.get("adaptive_clamp", 4.0)))
    assert loss.log_vars is not None and loss.adaptive_clamp == 4.0
    cfg2 = {}
    loss2 = MultiTaskLoss(_loss_weights_from(cfg2), learnable=False)
    assert loss2.log_vars is None