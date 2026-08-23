"""Tests for the S1 roll-forward OOS evaluation (anti-overfitting gate)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.models.policy import PolicyConfig, PolicyNetwork

from tests.test_forensics_s1 import _make_checkpoint, _make_store, _probe_feature_dims


def test_rollforward_oos_eval(tmp_path):
    store = _make_store(tmp_path)          # 2 symbols, ~220 bars @5min, from 2024-01-01
    n_feat, n_ctx = _probe_feature_dims(store)
    cfg = PolicyConfig(
        image_size=8, in_numeric_features=n_feat, in_context_features=n_ctx, window=8,
        embed_dim=16, n_actions=9, n_regime_classes=2, n_instruments=2,
        vision_channels=(4, 8), numeric_layers=1, fusion_layers=1,
        memory_layers=1, memory_max_len=8, vision_mode="cnn", use_memory=True,
    )
    ck = _make_checkpoint(PolicyNetwork(cfg), tmp_path)

    df = pd.read_parquet(store / "symbols" / "BTC_USDT.parquet").sort_index()
    last = df.index.max()
    oos_start = (last - pd.Timedelta(hours=12)).isoformat()     # late slice
    oos_end = last.isoformat()

    import zhisa.scripts.s1_rollforward_eval as rf
    out = tmp_path / "oos"
    rc = rf.main([
        "--checkpoint", str(ck), "--prepared-root", str(store),
        "--symbols", "BTC_USDT,TRX_USDT",
        "--start", oos_start, "--end", oos_end,
        "--out", str(out), "--device", "cpu",
        "--chart-window", "8", "--image-size", "8",
        "--val-max-batches", "4", "--seed", "0",
    ])
    assert rc == 0
    rep = json.loads((out / "oos_report.json").read_text(encoding="utf-8"))
    assert rep["n_samples"] > 0
    for key in ("total", "masked"):
        assert key in rep and isinstance(rep[key], float) and rep[key] == rep[key]
    assert torch.isfinite(torch.tensor(rep["total"]))


def test_rollforward_empty_slice_returns_nonzero(tmp_path):
    store = _make_store(tmp_path)
    n_feat, n_ctx = _probe_feature_dims(store)
    cfg = PolicyConfig(
        image_size=8, in_numeric_features=n_feat, in_context_features=n_ctx, window=8,
        embed_dim=16, n_actions=9, n_regime_classes=2, n_instruments=2,
        vision_channels=(4, 8), numeric_layers=1, fusion_layers=1,
        memory_layers=1, memory_max_len=8, vision_mode="cnn", use_memory=True,
    )
    ck = _make_checkpoint(PolicyNetwork(cfg), tmp_path)
    import zhisa.scripts.s1_rollforward_eval as rf
    rc = rf.main([
        "--checkpoint", str(ck), "--prepared-root", str(store),
        "--symbols", "BTC_USDT,TRX_USDT",
        "--start", "1999-01-01", "--end", "1999-02-01",
        "--out", str(tmp_path / "oos2"), "--device", "cpu",
        "--chart-window", "8", "--image-size", "8", "--val-max-batches", "2",
    ])
    assert rc == 2  # explicitly gated: no OOS data