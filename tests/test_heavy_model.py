"""Tests for the heavy S1 model config and instrument identity wiring."""
from __future__ import annotations

import pytest
import torch

from zhisa.config import load_config
from zhisa.data.dataset import MarketDataset, SampleSpec, multimodal_collate
from zhisa.models.policy import build_default_policy
from zhisa.scripts.train_s1 import _policy_kwargs_from
from zhisa.training.s1_ssl import SSLConfig, SSLPretrainer


def _synth_ds(n=500, seed=7, instrument_id=0, window=32, image=32):
    from zhisa.data.synthetic import MarketConfig, generate_market
    from zhisa.utils.seeding import set_seed
    set_seed(seed)
    df = generate_market(MarketConfig(n_bars=n, freq="5min", seed=seed))
    return MarketDataset(
        df,
        spec=SampleSpec(chart_window=window, feature_window=window, image_size=image, horizons=(4, 16, 64)),
        cache_charts=False,
        compute_targets=False,
        instrument_id=instrument_id,
    )


HEAVY = {
    "embed_dim": 384,
    "vision_channels": [64, 128, 256, 384],
    "numeric_layers": 4,
    "fusion_layers": 4,
    "memory_layers": 4,
    "memory_max_len": 128,
    "n_instruments": 12,
}


def _heavy_policy(ds, window=32, image=32):
    return build_default_policy(
        in_numeric_features=int(ds._features_df.shape[1]),
        in_context_features=int(ds._time_features_df.shape[1]),
        window=window, image_size=image, n_actions=9, n_regime_classes=4,
        **HEAVY,
    )


def test_policy_kwargs_from_merges_config():
    cfg = {"model": {"embed_dim": 384, "vision_channels": [64, 128, 256, 384], "numeric_layers": 4}}
    class _Spec:
        chart_window = 32
        image_size = 32
        n_regime_states = 4
    kwargs = _policy_kwargs_from(cfg, n_instruments=12, spec=_Spec())
    assert kwargs["embed_dim"] == 384
    assert kwargs["vision_channels"] == (64, 128, 256, 384)
    assert kwargs["numeric_layers"] == 4
    assert kwargs["n_instruments"] == 12
    assert "in_numeric_features" not in kwargs  # guarded, set by caller


def test_heavy_policy_is_large():
    ds = _synth_ds()
    from zhisa.models.policy import PolicyNetwork
    policy = _heavy_policy(ds)
    assert isinstance(policy, PolicyNetwork)
    n = sum(p.numel() for p in policy.parameters())
    assert n > 15_000_000, f"heavy policy unexpectedly small: {n}"
    assert policy.cfg.n_instruments == 12


def test_heavy_policy_forward_runs():
    ds = _synth_ds()
    policy = _heavy_policy(ds).eval()
    b = multimodal_collate([ds[i] for i in range(4)])
    with torch.no_grad():
        out = policy(
            chart=b.chart, numeric=b.numeric, context=b.context,
            instrument_id=b.instrument_id,
        )
    assert out["embedding"].shape == (4, 384)
    assert torch.isfinite(out["embedding"]).all()


def test_dataset_instrument_id_wiring():
    ds0 = _synth_ds(instrument_id=0)
    ds5 = _synth_ds(instrument_id=5)
    assert int(ds0[10]["instrument_id"]) == 0
    assert int(ds5[10]["instrument_id"]) == 5
    batch = multimodal_collate([ds0[3], ds5[3], ds0[9]])
    assert batch.instrument_id.tolist() == [0, 5, 0]


def test_temporal_pair_collate_keeps_instrument():
    from zhisa.training.s1_ssl import TemporalPairDataset, temporal_pair_collate
    ds = _synth_ds(instrument_id=7)
    pairs = TemporalPairDataset(ds, horizon=1)
    cb, fb = pairs[5]
    batch = temporal_pair_collate([(cb, fb)])
    assert batch["instrument_id"].tolist() == [7]


def test_heavy_ssl_step_is_finite():
    from zhisa.utils.seeding import set_seed
    ds = _synth_ds(n=300, instrument_id=3)
    policy = _heavy_policy(ds)
    cfg = SSLConfig(
        device="cpu", batch_size=4, use_ema_teacher=True,
        use_masked_modeling=True, use_temporal_contrast=True, use_cross_modal=True,
    )
    tr = SSLPretrainer(policy, cfg)
    loader = tr._loader(ds, shuffle=True, epoch=0)
    batch = next(iter(loader))
    losses = tr.step(tr._to_device(batch))
    assert torch.isfinite(torch.tensor(losses["total"]))
    assert losses["temporal"] > 0.0  # instrument-wired temporal loss computed


def test_heavy_1h_config_loads():
    cfg = load_config("configs/s1_ssl_1h_12m_heavy.yaml")
    assert cfg["model"]["embed_dim"] == 384
    assert cfg["model"]["n_instruments"] == 12
    assert cfg["ssl"]["mask_ratio"] == 0.5
    assert cfg["batch_size"] == 64