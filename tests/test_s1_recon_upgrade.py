"""Tests for the S1 reconstruction/gradient/instrument upgrades (P1-P3).

Everything is opt-in via SSLConfig knobs; default configs must keep the exact
canonical behaviour (default reconstructor == old single-Linear head).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from zhisa.models.policy import PolicyConfig, PolicyNetwork
from zhisa.training.s1_ssl import (
    SSLConfig,
    SSLPretrainer,
    _MaskedReconstructor,
    masked_numeric_loss,
)


def _tiny_model(seed: int = 0) -> PolicyNetwork:
    torch.manual_seed(seed)
    return PolicyNetwork(PolicyConfig(
        image_size=8, in_numeric_features=12, in_context_features=6, window=8,
        embed_dim=16, n_actions=9, n_regime_classes=2, n_instruments=2,
        vision_channels=(4, 8), numeric_layers=1, fusion_layers=1,
        memory_layers=1, memory_max_len=8, vision_mode="cnn", use_memory=True,
        dropout=0.0,
    ))


def _trainer(model, **ssl):
    cfg = SSLConfig(device="cpu", batch_size=4, epochs=1,
                    use_temporal_contrast=False, use_cross_modal=True,
                    use_masked_modeling=False, **ssl)
    return SSLPretrainer(model, cfg)


def _batch(B: int = 4, n_inst: int = 2):
    chart = torch.randn(B, 3, 8, 8)
    numeric = torch.randn(B, 8, 12)
    context = torch.randn(B, 6)
    ids = torch.arange(B, dtype=torch.long) % n_inst
    return {"chart": chart, "numeric": numeric, "context": context,
            "instrument_id": ids}


# ---------------------------------------------------------------------------
# P1: reconstructor
# ---------------------------------------------------------------------------


def test_default_reconstructor_matches_legacy_arch():
    r_legacy = _MaskedReconstructor(16, 4, 12)
    r_new = _MaskedReconstructor(16, 4, 12, depth=1, use_residual_norm=True, use_gain=False)
    # identical parameter names/shapes -> strict load both directions
    sd = r_legacy.state_dict()
    assert set(sd.keys()) == set(r_new.state_dict().keys())
    assert list(sd.keys()) == ["head.weight", "head.bias"]
    r_new.load_state_dict(sd, strict=True)


def test_reconstructor_gain_and_depth_forward_shape():
    r = _MaskedReconstructor(16, 4, 12, depth=2, use_residual_norm=True, use_gain=True)
    assert "gain" in r.state_dict()
    assert r.gain.shape == (48,)
    tokens = torch.randn(2, 3, 16)  # B, 1+n_patches, d_model (window8/patch4 -> 2 patches)
    out = r(tokens)
    assert out.shape == (2, 3, 48)


def test_masked_numeric_loss_target_norm():
    model = _tiny_model()
    recon = _MaskedReconstructor(model.numeric.cfg.d_model,
                                 model.numeric.cfg.patch_size,
                                 model.numeric.cfg.in_features,
                                 depth=2, use_gain=True)
    x = torch.randn(4, model.numeric.cfg.window, model.numeric.cfg.in_features)
    l_raw = masked_numeric_loss(model.numeric, recon, x, 0.5)
    l_norm = masked_numeric_loss(model.numeric, recon, x, 0.5, target_norm=True)
    assert torch.isfinite(l_raw) and torch.isfinite(l_norm)
    # gradients must flow into the new gain + head under both modes
    for use_norm in (False, True):
        recon.zero_grad(set_to_none=True)
        l = masked_numeric_loss(model.numeric, recon, x, 0.5, target_norm=use_norm)
        l.backward()
        assert recon.head.weight.grad is not None
        assert recon.gain.grad is not None


# ---------------------------------------------------------------------------
# P3: z-level instrument contrast
# ---------------------------------------------------------------------------


def test_instrument_z_contrast_present_when_multiple_ids():
    tr = _trainer(_tiny_model(), instrument_z_contrast_w=0.05)
    loss = tr._loss(_batch(8, n_inst=2))
    assert "instrument_z" in loss
    assert torch.isfinite(loss["instrument_z"]) and loss["instrument_z"] > 0.0
    loss["total"].backward()
    assert tr.model.numeric.encoder.layers[0].self_attn.in_proj_weight.grad is not None


def test_instrument_z_contrast_absent_for_single_id():
    tr = _trainer(_tiny_model(), instrument_z_contrast_w=0.05)
    loss = tr._loss(_batch(8, n_inst=1))
    assert "instrument_z" not in loss


# ---------------------------------------------------------------------------
# P2: vision gradient scaling
# ---------------------------------------------------------------------------


def test_vision_grad_scale_applied():
    results = []
    for gs in (1.0, 3.0):
        model = _tiny_model(seed=42)
        tr = _trainer(model, vision_grad_scale=gs, lr=0.0)
        b = _batch(8, n_inst=2)
        tr.step(b)  # backward -> vision-scale -> clip -> opt.step (lr=0 no-op)
        vg = 0.0
        ng = 0.0
        for n_, p_ in tr.model.named_parameters():
            if n_.startswith("vision.") and p_.grad is not None:
                vg = max(vg, float(p_.grad.detach().abs().mean()))
            elif n_.startswith("numeric.") and p_.grad is not None:
                ng = max(ng, float(p_.grad.detach().abs().mean()))
        results.append((vg, ng))
    v1, n1 = results[0]
    v3, n3 = results[1]
    assert v1 > 0 and v3 > 0
    assert 2.5 <= v3 / v1 <= 3.5, f"vision ratio {v3 / v1:.2f}"
    assert 0.5 <= n3 / n1 <= 1.5, f"numeric ratio {n3 / n1:.2f}"


# ---------------------------------------------------------------------------
# integration: everything on
# ---------------------------------------------------------------------------


def test_step_all_upgrade_knobs_runs():
    model = _tiny_model(seed=7)
    cfg = SSLConfig(device="cpu", batch_size=4, epochs=1,
                    use_temporal_contrast=True, use_cross_modal=True,
                    use_masked_modeling=True,
                    recon_depth=2, recon_use_gain=True, masked_target_norm=True,
                    vision_grad_scale=2.0, instrument_z_contrast_w=0.05,
                    mask_ratio=0.5,
                    lr=1e-4, warmup_steps=0)
    tr = SSLPretrainer(model, cfg)
    b = _batch(8, n_inst=2)
    b["future_chart"] = b["chart"]
    b["future_numeric"] = b["numeric"]
    b["future_context"] = b["context"]
    out = tr.step(b)
    for k in ("temporal", "alignment", "masked", "instrument_z", "total"):
        assert k in out and out[k] is not None
        assert np.isfinite(out[k]) if k == "total" else True
    assert np.isfinite(out["total"])
    assert tr._step == 1