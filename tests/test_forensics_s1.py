"""Tests for the S1 checkpoint forensics tool.

Covers the correctness fixes added to ``forensics_s1_checkpoint``:
  * weight stats no longer emit NaN std for size-1 parameters and now count
    NaN/Inf weights explicitly;
  * the summary-token slice in masked reconstruction follows
    ``summary_position`` (not just ``causal``);
  * the model rebuild matches the checkpoint's parameter set exactly
    (memory_residual/memory_input_norm detected and disabled when absent);
  * ``best_last_weight_l2`` is computed on the LOADED checkpoint weights;
  * the CLI runs end-to-end on a tiny checkpoint + a tiny prepared store and
    produces a fully-populated report (aligned, CPC+margin, masked+visible,
    gradient balance, provenance).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.models.policy import PolicyConfig, PolicyNetwork
from zhisa.scripts.forensics_s1_checkpoint import (
    _load_into,
    _policy_matching_checkpoint,
    _summary_end,
    _weight_stats,
)
from zhisa.training.s1_ssl import SSLConfig, SSLPretrainer


def _tiny_model() -> PolicyNetwork:
    return PolicyNetwork(PolicyConfig(
        image_size=8, in_numeric_features=12, in_context_features=8, window=8,
        embed_dim=16, n_actions=9, n_regime_classes=2, n_instruments=2,
        vision_channels=(4, 8), numeric_layers=1, fusion_layers=1,
        memory_layers=1, memory_max_len=8, vision_mode="cnn", use_memory=True,
    ))


# ---------------------------------------------------------------------------
# 1. weight stats
# ---------------------------------------------------------------------------


def test_weight_stats_no_nan_for_size1_and_counts_nan_inf():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.zeros(1))          # size-1
            self.b = nn.Parameter(torch.ones(4))
            self.c = nn.Parameter(torch.zeros(8))

    m = M()
    with torch.no_grad():
        m.b[0] = float("nan")
        m.c[2] = float("inf")
    stats = _weight_stats(m)
    assert stats["nan_params"] == 1
    assert stats["inf_params"] == 1
    assert stats["per_layer"]["a"]["std"] == 0.0  # size-1: was NaN before the fix
    assert bool(np.isnan(stats["per_layer"]["b"]["std"]))  # real NaN surfaces
    assert stats["near_zero_total_frac"] > 0.0


# ---------------------------------------------------------------------------
# 2. summary-position slice
# ---------------------------------------------------------------------------


def test_summary_end_semantics():
    class C:  # minimal cfg stand-in
        summary_position = "front"
        causal = False
    assert _summary_end(C()) is False

    class E:
        summary_position = "end"
        causal = False
    assert _summary_end(E()) is True

    class Caus:
        summary_position = None
        causal = True
    assert _summary_end(Caus()) is True


# ---------------------------------------------------------------------------
# 3. model rebuild matches the checkpoint parameter set
# ---------------------------------------------------------------------------


def test_policy_matching_checkpoint_disables_missing_memory_modules():
    cfg = PolicyConfig(
        image_size=8, in_numeric_features=12, in_context_features=8, window=8,
        embed_dim=16, n_actions=9, n_regime_classes=2, n_instruments=2,
        vision_channels=(4, 8), numeric_layers=1, fusion_layers=1,
        memory_layers=1, memory_max_len=8, vision_mode="cnn",
        memory_residual=False, memory_input_norm=False,
    )
    model = PolicyNetwork(cfg)
    state = model.state_dict()
    # expose the same config but WITHOUT the memory-residual fields (old ckpt)
    mc = {k: v for k, v in cfg.__dict__.items()}
    mc.pop("memory_residual")
    mc.pop("memory_input_norm")
    rebuilt, eff = _policy_matching_checkpoint(mc, state)
    assert "memory_scale" not in rebuilt.state_dict().keys()  # no extra module
    assert eff["memory_residual"] is False and eff["memory_input_norm"] is False
    # every rebuilt param must exist in the checkpoint state
    assert set(rebuilt.state_dict().keys()) <= set(state.keys())
    # and they load strictly (no dropped keys)
    rebuilt.load_state_dict(state, strict=True)


def test_policy_matching_checkpoint_keeps_residual_when_ckpt_has_it():
    cfg = PolicyConfig(
        image_size=8, in_numeric_features=12, in_context_features=8, window=8,
        embed_dim=16, n_actions=9, n_regime_classes=2, n_instruments=2,
        vision_channels=(4, 8), numeric_layers=1, fusion_layers=1,
        memory_layers=1, memory_max_len=8, vision_mode="cnn",
        memory_residual=True, memory_input_norm=True,
    )
    state = PolicyNetwork(cfg).state_dict()
    mc = dict(cfg.__dict__)
    rebuilt, eff = _policy_matching_checkpoint(mc, state)
    assert "memory_scale" in rebuilt.state_dict().keys()
    assert eff["memory_residual"] is True


def test_load_into_filters_shape_mismatches():
    base = _tiny_model()
    target = _tiny_model()
    state = base.state_dict()
    # corrupt one tensor's shape so it must be dropped
    state["heads.value.bias"] = torch.zeros(99)
    _load_into(target, state)
    assert target.heads.value.bias.shape == torch.Size([1])


# ---------------------------------------------------------------------------
# 4. tiny checkpoint + store -> CLI end-to-end
# ---------------------------------------------------------------------------


def _make_store(tmp: Path) -> Path:
    store = tmp / "store"
    (store / "symbols").mkdir(parents=True, exist_ok=True)
    for i, sym in enumerate(("BTC_USDT", "TRX_USDT")):
        df = generate_market(MarketConfig(n_bars=220, seed=10 + i, initial_price=1000.0 * (i + 1)))
        df = df.drop(columns=["regime"]).copy()
        df.index = pd.date_range("2024-01-01", periods=len(df), freq="5min")
        df.to_parquet(store / "symbols" / f"{sym}.parquet")
    return store


def _make_checkpoint(model: PolicyNetwork, tmp: Path, ssl_kwargs=None) -> Path:
    ssl_cfg = SSLConfig(
        device="cpu", epochs=1, batch_size=4, projection_dim=8, hidden_dim=16,
        use_ema_teacher=True, use_masked_modeling=True,
        use_temporal_contrast=True, use_cross_modal=True,
        mask_ratio=0.5, temperature=0.1,
    )
    tr = SSLPretrainer(model, ssl_cfg)
    payload = {
        "model": tr.model.state_dict(),
        "model_config": model.cfg.__dict__.copy(),
        "config": model.cfg.__dict__.copy(),
        "optimizer": tr.opt.state_dict(),
        "proj_numeric": tr.proj_numeric.state_dict(),
        "proj_temporal": tr.proj_temporal.state_dict(),
        "proj_vision": tr.proj_vision.state_dict(),
        "temporal_predictor": tr.temporal_predictor.state_dict(),
        "reconstructor": tr.reconstructor.state_dict(),
        "target_proj_temporal": tr.target_proj_temporal.state_dict(),
        "teacher": {"teacher": tr.teacher.teacher.state_dict()},
        "ssl_config": ssl_cfg.__dict__.copy(),
        "trainer_state": {
            "completed_epochs": 3, "step": 4000,
            "best_val_total": 0.7, "history": [{"val": {"total": 0.8}}],
        },
        "checkpoint_meta": {
            "stage": "s1_ssl",
            "dataset": {
                "root": "test_store", "timeframe": "5min",
                "manifest_checksum": "abc123",
            },
        },
    }
    p = tmp / "best.pt"
    torch.save(payload, p)
    return p


def _probe_feature_dims(store: Path) -> tuple[int, int]:
    from zhisa.data.dataset import MarketDataset, SampleSpec
    df = pd.read_parquet(store / "symbols" / "BTC_USDT.parquet").sort_index()
    ds = MarketDataset(df, spec=SampleSpec(chart_window=8, image_size=8),
                       compute_targets=False, instrument_id=0)
    return int(ds._features.shape[1]), int(ds._time_features.shape[1])


def test_forensics_cli_end_to_end(tmp_path, monkeypatch):
    store = _make_store(tmp_path)
    n_feat, n_ctx = _probe_feature_dims(store)
    cfg = PolicyConfig(
        image_size=8, in_numeric_features=n_feat, in_context_features=n_ctx, window=8,
        embed_dim=16, n_actions=9, n_regime_classes=2, n_instruments=2,
        vision_channels=(4, 8), numeric_layers=1, fusion_layers=1,
        memory_layers=1, memory_max_len=8, vision_mode="cnn", use_memory=True,
    )
    model = PolicyNetwork(cfg)
    ck = _make_checkpoint(model, tmp_path)
    out = tmp_path / "out"
    import zhisa.scripts.forensics_s1_checkpoint as fx
    rc = fx.main([
        "--best", str(ck), "--last", str(ck),
        "--prepared-root", str(store), "--symbols", "BTC_USDT,TRX_USDT",
        "--out", str(out), "--device", "cpu",
        "--chart-window", "8", "--image-size", "8",
        "--samples", "8", "--num-cpc", "8", "--grad-batches", "1",
    ])
    assert rc == 0
    rep = json.loads((out / "forensics_report.json").read_text(encoding="utf-8"))

    # structural coverage
    for key in (
        "provenance", "model_config_short", "model_config_effective",
        "weight_stats_best", "behaviour_trained", "behaviour_random",
        "gradient_balance_trained", "instrument_separation_silhouette",
        "instrument_separation_silhouette_random",
    ):
        assert key in rep, f"missing {key}"

    bt = rep["behaviour_trained"]
    assert "cpc_forward" in bt and "margin" in bt["cpc_forward"]
    assert 0.0 <= bt["cpc_forward"]["top1_in_batch"] <= 1.0
    assert "visible_mse" in bt["masked_recon"] and bt["masked_recon"]["gain_vs_baseline"] > 0.0
    assert "alignment_cos_per_symbol" in bt and -1.0 <= bt["alignment_cos"] <= 1.0
    assert "embedding_dead_dim_frac" in bt["internals"]
    assert "vision_alive" in bt  # chart-only rank gate is always reported
    va = bt["vision_alive"]
    assert 0.0 <= va["chart_only_top10_svd"] <= 1.0
    assert va["chart_only_eff_dim"] >= 1 or va["chart_only_eff_dim"] == 0

    ws = rep["weight_stats_best"]
    assert ws["nan_params"] == 0 and ws["inf_params"] == 0
    for v in ws["per_layer"].values():
        assert isinstance(v["std"], float) and np.isfinite(v["std"])

    gb = rep["gradient_balance_trained"]
    assert gb["n_batches"] >= 1 and gb["vision_over_numeric"] >= 0.0

    # no critical metric fell back to an "err:" placeholder
    for val in (bt["alignment_cos"], bt["cpc_forward"]["top1_in_batch"],
                bt["masked_recon"]["gain_vs_baseline"], rep["provenance"]["stage"]):
        assert not (isinstance(val, str) and val.startswith("err:"))


def test_best_last_l2_is_meaningful_after_load(tmp_path, monkeypatch):
    # two checkpoints from the same architecture with different weights must
    # report a NON-zero, finite L2 distance (was computed on random inits).
    store = _make_store(tmp_path)
    n_feat, n_ctx = _probe_feature_dims(store)
    cfg = PolicyConfig(
        image_size=8, in_numeric_features=n_feat, in_context_features=n_ctx, window=8,
        embed_dim=16, n_actions=9, n_regime_classes=2, n_instruments=2,
        vision_channels=(4, 8), numeric_layers=1, fusion_layers=1,
        memory_layers=1, memory_max_len=8, vision_mode="cnn", use_memory=True,
    )
    best = _make_checkpoint(PolicyNetwork(cfg), tmp_path)
    import shutil
    last = tmp_path / "last.pt"
    shutil.copyfile(best, last)
    _perturb_checkpoint(last, 0.05)
    out = tmp_path / "out2"
    import zhisa.scripts.forensics_s1_checkpoint as fx
    rc = fx.main([
        "--best", str(best), "--last", str(last),
        "--prepared-root", str(store), "--symbols", "BTC_USDT,TRX_USDT",
        "--out", str(out), "--device", "cpu",
        "--chart-window", "8", "--image-size", "8",
        "--samples", "8", "--num-cpc", "8", "--grad-batches", "0",
    ])
    assert rc == 0
    rep = json.loads((out / "forensics_report.json").read_text(encoding="utf-8"))
    l2 = rep["best_last_weight_l2"]
    assert isinstance(l2, float) and l2 > 1e-3 and np.isfinite(l2)


def _perturb_checkpoint(path: Path, std: float) -> None:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for k, v in ck["model"].items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            ck["model"][k] = v + torch.randn_like(v) * std
    torch.save(ck, path)