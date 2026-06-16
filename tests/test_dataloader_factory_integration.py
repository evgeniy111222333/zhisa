"""Quick end-to-end smoke test for the optimised pipeline.

Creates a small MarketDataset, runs it through a build_dataloader-driven
training loop (single epoch, no backprop), and checks that all batch
fields are well-formed tensors. Designed to take ~10 seconds.
"""
from __future__ import annotations

import os

os.environ.setdefault("ZHISA_FAST_RENDER", "1")
os.environ.setdefault("ZHISA_TEST_DEVICE", "cpu")

import torch
import pytest

from zhisa.data.dataset import MarketDataset, SampleSpec, multimodal_collate
from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.training.dataloader_factory import build_dataloader
from zhisa.utils.seeding import set_seed


def test_end_to_end_dataloader():
    set_seed(42)
    df = generate_market(MarketConfig(n_bars=600, freq="5min", seed=42))
    spec = SampleSpec(chart_window=32, feature_window=32, image_size=32,
                       horizons=(4, 16))
    set_seed(42)
    ds = MarketDataset(df, spec=spec, cache_charts=True)
    assert ds.__fast_getitem__ is True
    n = len(ds)
    assert n > 0

    # Build loader via factory
    loader = build_dataloader(ds, batch_size=16, shuffle=True,
                              collate_fn=multimodal_collate, drop_last=True)
    # num_workers should be 0 because the dataset advertises __fast_getitem__
    assert loader.num_workers == 0
    # pin_memory: matches CUDA availability on this machine
    assert loader.pin_memory == torch.cuda.is_available()
    # persistent_workers must be False when num_workers == 0
    assert loader.persistent_workers is False

    # Run one fake epoch — just check shapes
    # The precompute path stores features as a numpy array; the legacy
    # DataFrame is kept as ``_features_df`` for fallback compatibility.
    feat_dim = ds._features_arr.shape[1] if ds._features_arr is not None else ds._features_df.shape[1]
    for i, batch in enumerate(loader):
        assert batch.chart.shape == (16, 3, 32, 32)
        assert batch.numeric.shape[0] == 16
        assert batch.numeric.shape[1:] == (32, feat_dim)
        assert batch.context.shape[0] == 16
        assert batch.label_dir.shape == (16,)
        assert batch.label_vol.shape == (16,)
        assert batch.label_regime.shape == (16,)
        assert batch.label_ret.shape == (16,)
        assert batch.label_risk.shape == (16,)
        assert batch.mask.shape == (16, 32)
        assert all(torch.isfinite(batch.label_ret))
        assert all(torch.isfinite(batch.label_vol))
        if i >= 2:
            break

    # Test explicit num_workers override
    loader2 = build_dataloader(ds, batch_size=16, num_workers=0,
                               collate_fn=multimodal_collate)
    assert loader2.num_workers == 0

    # Test that the dataset __fast_getitem__ is False when cache_charts=False
    set_seed(42)
    ds2 = MarketDataset(df, spec=spec, cache_charts=False)
    assert ds2.__fast_getitem__ is False
