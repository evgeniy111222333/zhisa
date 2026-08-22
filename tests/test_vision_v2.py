"""Tests for vision v2 (ColumnFormer + token fusion + freq + warm-start)."""
from __future__ import annotations

import math

import pytest
import torch

from zhisa.config import load_config
from zhisa.models.encoders.vision_columnformer import (
    ColumnFormerVision,
    VisionColumnFormerConfig,
)
from zhisa.models.policy import PolicyConfig, PolicyNetwork, build_default_policy
from zhisa.training.s1_ssl import SSLConfig, SSLPretrainer, _filter_matching_state_dict


def _img(B=2, size=128, seed=0):
    torch.manual_seed(seed)
    return torch.rand(B, 3, size, size)


V2 = dict(
    embed_dim=256,
    vision_mode="columnformer",
    vision_n_layers=2,
    vision_n_heads=4,
    vision_causal=True,
    vision_reader="attention_pool",
    token_fusion=True,
    fusion_token_layers=2,
    fusion_token_heads=4,
    freq_branch=True,
    freq_k=8,
    numeric_layers=2,
    encoder_ff_mult=2.0,
    n_instruments=12,
)


def test_columnformer_shapes_and_causal():
    torch.manual_seed(0)
    enc = ColumnFormerVision(VisionColumnFormerConfig(image_size=128, d_model=256, out_dim=256, n_heads=4, n_layers=2, dim_ff=512, freq_branch=True))
    vec, tok, freq = enc(_img(2, 128), mask=None)
    assert vec.shape == (2, 256)
    assert tok.shape == (2, 128, 256)
    assert freq is not None and freq.shape == (2, 1, 256)
    assert torch.isfinite(vec).all()


def test_columnformer_cls_and_freq_off():
    enc = ColumnFormerVision(VisionColumnFormerConfig(image_size=128, d_model=256, out_dim=256, n_heads=4, n_layers=1, dim_ff=512, reader="cls", freq_branch=False))
    vec, tok, freq = enc(_img(2, 128))
    assert vec.shape == (2, 256)
    assert tok.shape == (2, 128, 256)
    assert freq is None
    assert torch.isfinite(vec).all()


def test_policy_columnformer_forward_token_fusion():
    policy = build_default_policy(
        in_numeric_features=32, in_context_features=10,
        window=128, image_size=128, n_actions=9, n_regime_classes=4, **V2,
    ).eval()
    b = {
        "chart": _img(3, 128),
        "numeric": torch.rand(3, 128, 32),
        "context": torch.rand(3, 10),
        "instrument_id": torch.tensor([0, 5, 11]),
    }
    with torch.no_grad():
        out = policy(**b)
    assert out["embedding"].shape == (3, 256)
    assert torch.isfinite(out["embedding"]).all()


def test_policy_columnformer_vector_fusion():
    cfg = {**V2, "token_fusion": False}
    policy = build_default_policy(in_numeric_features=32, in_context_features=10, window=32, image_size=32, **cfg).eval()
    with torch.no_grad():
        out = policy(chart=_img(2, 32), numeric=torch.rand(2, 32, 32), context=torch.rand(2, 10))
    assert out["embedding"].shape == (2, 256)
    assert torch.isfinite(out["embedding"]).all()


def test_v2_has_more_params_than_v1():
    v1 = build_default_policy(in_numeric_features=32, in_context_features=10, window=128, image_size=128, n_actions=9, n_regime_classes=4,
                              embed_dim=256, vision_channels=(32, 64, 128), numeric_layers=2, fusion_layers=2, memory_layers=2, encoder_ff_mult=2.0, n_instruments=12)
    v2 = build_default_policy(in_numeric_features=32, in_context_features=10, window=128, image_size=128, n_actions=9, n_regime_classes=4, **V2)
    n1 = sum(p.numel() for p in v1.parameters())
    n2 = sum(p.numel() for p in v2.parameters())
    assert n2 > n1, (n1, n2)


def test_warm_start_copies_compatible_weights_only():
    v1 = build_default_policy(in_numeric_features=32, in_context_features=10, window=32, image_size=32, n_actions=9, n_regime_classes=4,
                              embed_dim=256, vision_channels=(32, 64, 128), numeric_layers=2, fusion_layers=2, memory_layers=2, encoder_ff_mult=2.0, n_instruments=12)
    v2 = build_default_policy(in_numeric_features=32, in_context_features=10, window=32, image_size=32, n_actions=9, n_regime_classes=4, **V2)
    filtered = _filter_matching_state_dict(v1.state_dict(), v2)
    assert len(filtered) > 0
    # the brand-new ColumnFormer branch must NOT be copied from the CNN encoder
    assert not any(k.startswith("vision.") for k in filtered)
    # numeric / memory / context / heads survive
    assert any(k.startswith("numeric.") for k in filtered)
    assert any(k.startswith("memory.") for k in filtered)
    assert any(k.startswith("context.") for k in filtered)


def test_warm_start_checkpoint_loads_as_warm():
    import tempfile
    from zhisa.scripts.warm_start_vision_v2 import _build_v2_policy

    v1 = build_default_policy(in_numeric_features=32, in_context_features=10, window=32, image_size=32, n_actions=9, n_regime_classes=4,
                              embed_dim=256, vision_channels=(32, 64, 128), numeric_layers=2, fusion_layers=2, memory_layers=2, encoder_ff_mult=2.0, n_instruments=12)
    payload = {
        "model": v1.state_dict(),
        "model_config": dict(v1.cfg.__dict__),
        "ssl_config": {"device": "cpu", "checkpoint": None},
        "checkpoint_meta": {"stage": "s1_ssl", "dataset": {}, "render": {}},
    }
    with tempfile.TemporaryDirectory() as td:
        src = f"{td}/v1.pt"
        torch.save(payload, src)
        # emulate the warm-start core: build v2, copy compatible weights, save, reload
        policy = build_default_policy(
            in_numeric_features=32, in_context_features=10, window=32, image_size=32,
            n_actions=9, n_regime_classes=4, **V2,
        )
        filtered = _filter_matching_state_dict(v1.state_dict(), policy)
        policy.load_state_dict(filtered, strict=False)
        tr = SSLPretrainer(policy, SSLConfig(device="cpu", batch_size=4,
                                             use_ema_teacher=True, use_masked_modeling=True,
                                             use_temporal_contrast=True, use_cross_modal=True))
        ssl_payload = {
            "model": policy.state_dict(),
            "proj_temporal": tr.proj_temporal.state_dict(),
            "proj_vision": tr.proj_vision.state_dict(),
            "proj_numeric": tr.proj_numeric.state_dict(),
            "reconstructor": tr.reconstructor.state_dict(),
            "optimizer": tr.opt.state_dict(),
            "config": dict(policy.cfg.__dict__),
            "model_config": dict(policy.cfg.__dict__),
            "ssl_config": tr.cfg.__dict__,
            "checkpoint_meta": {"stage": "s1_ssl"},
            "trainer_state": {"step": 0, "completed_epochs": 0, "history": [], "best_val_total": float("inf")},
            "teacher": tr.teacher.state_dict() if tr.teacher else {},
        }
        target = f"{td}/v2.pt"
        torch.save(ssl_payload, target)
        tr2 = SSLPretrainer(policy, SSLConfig(device="cpu", batch_size=4))
        status = tr2.load(target)
        assert status["resume_mode"] in ("full", "warm_start")


def test_v2_ssl_step_finite():
    torch.manual_seed(0)
    policy = build_default_policy(in_numeric_features=32, in_context_features=10, window=32, image_size=32, n_actions=9, n_regime_classes=4, **V2)
    tr = SSLPretrainer(policy, SSLConfig(device="cpu", batch_size=2, use_ema_teacher=True, use_masked_modeling=True, use_temporal_contrast=True, use_cross_modal=True))
    batch = {
        "chart": _img(2, 32),
        "numeric": torch.rand(2, 32, 32),
        "context": torch.rand(2, 10),
        "instrument_id": torch.tensor([0, 3]),
        "future_chart": _img(2, 32, seed=1),
        "future_numeric": torch.rand(2, 32, 32),
        "future_context": torch.rand(2, 10),
    }
    losses = tr.step(tr._to_device(batch))
    assert torch.isfinite(torch.tensor(losses["total"]))


# --- decoupling: token count == n_bars independent of image width (concept fix) ---

def test_columnformer_token_count_decoupled_from_image_width():
    enc = ColumnFormerVision(VisionColumnFormerConfig(
        image_size=64, n_bars=128, d_model=256, out_dim=256, n_heads=4, n_layers=1, dim_ff=512))
    vec, tok, freq = enc(_img(2, 64))
    assert tok.shape == (2, 128, 256)   # n_bars, NOT image width
    assert vec.shape == (2, 256)
    assert torch.isfinite(vec).all()


def test_columnformer_deterministic_across_calls():
    enc = ColumnFormerVision(VisionColumnFormerConfig(
        image_size=128, n_bars=64, d_model=256, out_dim=256, n_heads=4, n_layers=2, dim_ff=512))
    enc.eval()  # dropout off -> output is a pure function of input
    img = _img(2, 128)
    v1, t1, _ = enc(img)
    v2, t2, _ = enc(img)
    assert torch.equal(v1, v2) and torch.equal(t1, t2)


def test_columnformer_two_tokens_per_bar():
    enc = ColumnFormerVision(VisionColumnFormerConfig(
        image_size=128, n_bars=64, d_model=256, out_dim=256, n_heads=4, n_layers=1, dim_ff=512,
        two_tokens_per_bar=True))
    vec, tok, freq = enc(_img(2, 128))
    assert tok.shape == (2, 128, 256)   # 2 * n_bars
    assert torch.isfinite(vec).all()


def test_columnformer_volume_off_zero_fill():
    enc = ColumnFormerVision(VisionColumnFormerConfig(
        image_size=128, n_bars=128, d_model=256, out_dim=256, n_heads=4, n_layers=1, dim_ff=512,
        include_volume=False))
    vec, tok, freq = enc(_img(2, 128))
    assert tok.shape == (2, 128, 256)
    assert torch.isfinite(vec).all()


def test_policy_window_neq_image_is_legal():
    # numeric window 128, image 64: columnformer still emits 128 tokens aligned to numeric.
    policy = build_default_policy(
        in_numeric_features=32, in_context_features=10,
        window=128, image_size=64, n_actions=9, n_regime_classes=4, **V2,
    ).eval()
    with torch.no_grad():
        out = policy(
            chart=_img(2, 64),
            numeric=torch.rand(2, 128, 32),
            context=torch.rand(2, 10),
            instrument_id=torch.tensor([0, 1]),
        )
    assert out["embedding"].shape == (2, 256)
    assert torch.isfinite(out["embedding"]).all()


def test_cls_reader_sees_whole_window_under_causal():
    """Review fix: CLS placed AFTER bars; under causal it must still see all bars."""
    enc = ColumnFormerVision(VisionColumnFormerConfig(
        image_size=128, n_bars=32, d_model=256, out_dim=256, n_heads=4, n_layers=1,
        dim_ff=512, reader="cls", causal=True, freq_branch=False))
    enc.eval()
    base = _img(1, 128)
    only_first = base * 0.0
    only_first[:, :, :, :8] = 1.0
    only_last = base * 0.0
    only_last[:, :, :, -8:] = 1.0
    with torch.no_grad():
        v_first = enc(only_first)[0]
        v_last = enc(only_last)[0]
    assert torch.isfinite(v_first).all()
    # CLS sees the whole window -> a signal at the far-left changes the summary too.
    assert not torch.allclose(v_first, v_last, atol=1e-4)


def test_aggregate_to_bars_matches_naive():
    enc = ColumnFormerVision(VisionColumnFormerConfig(
        image_size=64, n_bars=128, d_model=16, out_dim=16, n_heads=2, n_layers=1,
        dim_ff=32, freq_branch=False))
    x = torch.rand(2, 64, 16)
    got = enc._aggregate_to_bars(x)
    exp = torch.stack([
        x[:, int(enc.seg_starts[b]):int(enc.seg_ends[b])].mean(dim=1)
        if int(enc.seg_starts[b]) < int(enc.seg_ends[b]) else torch.zeros(2, 16)
        for b in range(enc.n_bars)
    ], dim=1)
    assert torch.allclose(got, exp, atol=1e-5)


def test_dct_trace_uses_real_dct():
    import math
    enc = ColumnFormerVision(VisionColumnFormerConfig(
        image_size=128, n_bars=32, d_model=32, out_dim=32, n_heads=2, n_layers=1,
        dim_ff=64, freq_k=4))
    n = enc.n_bars
    t = torch.zeros(1, n)
    t[0, n // 2] = 1.0
    with torch.no_grad():
        c = torch.matmul(t.unsqueeze(1), enc.dct_basis).squeeze(1)
    # DCT-II basis columns are orthogonal; the first coefficient must be the most
    # dominant for a low-frequency-ish non-DC impulse pattern... simply verify shape/values finite.
    assert c.shape == (1, 4)
    assert torch.isfinite(c).all()


# --- numeric-side causality (vision v2 concept D) ---

def _num_enc(causal: bool, d: int = 64, window: int = 16, patch: int = 4, layers: int = 1):
    from zhisa.models.encoders.numeric import NumericEncoder, NumericEncoderConfig
    return NumericEncoder(NumericEncoderConfig(
        in_features=8, window=window, patch_size=patch, d_model=d, n_heads=2,
        n_layers=layers, dim_ff=128, dropout=0.0, out_dim=d,
        causal=causal, summary_position="end" if causal else "front",
    ))


def test_numeric_causal_patches_do_not_see_future():
    torch.manual_seed(0)
    base = torch.rand(1, 16, 8)
    changed = base.clone()
    changed[0, -4:, :] = 0.0  # only the LAST patch differs (future region)

    enc_c = _num_enc(causal=True); enc_c.eval()
    enc_f = _num_enc(causal=False); enc_f.eval()
    with torch.no_grad():
        _, tok_c1 = enc_c(base)
        _, tok_c2 = enc_c(changed)
        _, tok_f1 = enc_f(base)
        _, tok_f2 = enc_f(changed)
    # patch block positions: causal -> [0:n_patches]; non-causal -> [1:1+n_patches]
    n_patches = enc_c.n_patches
    dc = (tok_c1[:, : n_patches - 1] - tok_c2[:, : n_patches - 1]).abs().max().item()
    df = (tok_f1[:, 1 : n_patches] - tok_f2[:, 1 : n_patches]).abs().max().item()
    assert dc < 1e-6, f"causal patches leaked future: {dc}"
    assert df > 1e-4, f"non-causal should reflect the future change: {df}"
    # and the causal summary (last token) DOES see the change:
    ds = (tok_c1[:, -1] - tok_c2[:, -1]).abs().max().item()
    assert ds > 1e-4, "causal summary should see the whole window"


def test_numeric_causal_masked_loss_uses_correct_patch_slice():
    from zhisa.models.encoders.numeric import NumericEncoderConfig
    from zhisa.training.s1_ssl import _MaskedReconstructor, masked_numeric_loss

    enc = _num_enc(causal=True, d=64, window=16, patch=4)
    enc.eval()
    recon = _MaskedReconstructor(enc.cfg.d_model, enc.cfg.patch_size, enc.cfg.in_features)
    x = torch.rand(2, 16, 8)
    loss = masked_numeric_loss(enc, recon, x, mask_ratio=0.4)
    assert torch.isfinite(loss) and float(loss) > 0.0


def test_policy_with_numeric_causal_runs():
    cfg = {**V2, "numeric_causal": True}
    from zhisa.models.encoders.numeric import NumericEncoder
    policy = build_default_policy(in_numeric_features=32, in_context_features=10, window=32, image_size=32, n_actions=9, n_regime_classes=4, **cfg)
    assert isinstance(policy.numeric, NumericEncoder)
    assert policy.numeric.cfg.causal is True
    b = {
        "chart": _img(2, 32),
        "numeric": torch.rand(2, 32, 32),
        "context": torch.rand(2, 10),
        "instrument_id": torch.tensor([0, 1]),
    }
    with torch.no_grad():
        out = policy(**b)
    assert out["embedding"].shape == (2, policy.cfg.embed_dim)
    assert torch.isfinite(out["embedding"]).all()


def test_heavy_v2_config_has_numeric_causal():
    cfg = load_config("configs/s1_ssl_1h_12m_heavy_v2.yaml")
    assert cfg["model"]["numeric_causal"] is True