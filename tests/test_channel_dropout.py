"""Tests for keyed, family-aware channel dropout (numeric augmentation)."""
from __future__ import annotations

import numpy as np
import torch

from zhisa.data.dataset import MarketDataset, SampleSpec
from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.features.channel_dropout import (
    ChannelDropoutSpec,
    apply_channel_dropout,
    channel_family,
    keyed_drop_mask,
    key_from_string,
)
from zhisa.utils.seeding import set_seed


COLS = ["logret_1", "rv_8", "vol_z_64", "rsi_14", "sma_dist_20",
        "beta_256", "corr_64", "funding", "open_oi", "ctx_vol"]


def test_key_stable_and_mask_deterministic():
    spec = ChannelDropoutSpec(p=0.5, max_channels=3)
    m1 = keyed_drop_mask(len(COLS), COLS, "cd:0:10", spec)
    m2 = keyed_drop_mask(len(COLS), COLS, "cd:0:10", spec)
    m3 = keyed_drop_mask(len(COLS), COLS, "cd:1:10", spec)
    assert np.array_equal(m1, m2)
    assert not np.array_equal(m1, m3)


def test_family_limit_respected():
    spec = ChannelDropoutSpec(p=1.0, max_channels=4)
    for trial in range(50):
        mask = keyed_drop_mask(len(COLS), COLS, f"cd:0:{trial}", spec)
        fams = {channel_family(COLS[i]) for i in np.where(mask)[0]}
        assert len(fams) == int(mask.sum())
        assert int(mask.sum()) <= spec.max_channels


def test_max_channels_cap():
    spec = ChannelDropoutSpec(p=1.0, max_channels=2)
    for trial in range(60):
        mask = keyed_drop_mask(len(COLS), COLS, f"cd:{trial}:0", spec)
        assert int(mask.sum()) <= 2


def test_apply_zeros_only_masked_channels():
    normed = np.random.default_rng(0).normal(size=(8, len(COLS)))
    spec = ChannelDropoutSpec(p=1.0, max_channels=2)
    out = apply_channel_dropout(normed, COLS, "cd:0:5", spec)
    mask = np.zeros(len(COLS), dtype=bool)
    mask += out[0] == 0.0
    assert mask.sum() == 2  # exactly the two family-max channels zeroed
    # untouched channels keep values
    assert np.allclose(out[:, ~np.any(out == 0.0, axis=0)], normed[:, ~np.any(out == 0.0, axis=0)])


def test_families_classified():
    assert channel_family("logret_8") == "logret"
    assert channel_family("corr_256") == "cross_asset"
    assert channel_family("rsix") == "other"
    assert channel_family("rsi_14") == "rsi"


def test_dataset_channel_dropout_applies_and_toggles():
    set_seed(11)
    df = generate_market(MarketConfig(n_bars=600, freq="5min", seed=11))
    spec = SampleSpec(chart_window=32, feature_window=32, image_size=32, horizons=(4, 16, 64))
    cd = ChannelDropoutSpec(p=1.0, max_channels=2)
    ds = MarketDataset(df, spec=spec, cache_charts=False, compute_targets=False, channel_dropout=cd)
    cols = ds._features_df.columns.tolist()

    def masked_cols(x):
        return {cols[i] for i in range(len(cols)) if bool((x[:, i] == 0.0).all())}

    s = ds[50]
    expected = {cols[i] for i in np.where(
        keyed_drop_mask(len(cols), cols, "cd:0:3", cd))[0]}
    assert expected.issubset(masked_cols(s["numeric"].numpy())), expected
    # deterministic for the same (salt, bucket)
    assert torch.equal(s["numeric"], ds[50]["numeric"])
    # different salt -> different mask
    ds.set_aug_salt(3)
    other = {cols[i] for i in np.where(
        keyed_drop_mask(len(cols), cols, "cd:3:3", cd))[0]}
    s3 = ds[50]
    assert (masked_cols(s3["numeric"].numpy()) & other) == other
    assert not torch.equal(s["numeric"], s3["numeric"])
    # disabled -> no dropout applied
    ds.set_channel_dropout_enabled(False)
    s4 = ds[50]
    assert not expected.issubset(masked_cols(s4["numeric"].numpy()))


def test_noise_fill_mode_is_keyed_and_bounded():
    normed = np.random.default_rng(0).normal(size=(8, len(COLS)))
    spec = ChannelDropoutSpec(p=1.0, max_channels=2, fill_mode="noise", noise_std=0.1)
    a = apply_channel_dropout(normed, COLS, "cd:0:5", spec)
    b = apply_channel_dropout(normed, COLS, "cd:0:5", spec)
    c = apply_channel_dropout(normed, COLS, "cd:1:5", spec)
    assert np.array_equal(a, b)          # deterministic per key
    assert not np.array_equal(a, c)      # different key -> different noise
    # masked positions carry noise (not exactly zero), others untouched
    assert np.allclose(a[:, ~(a != normed).any(axis=0)], normed[:, ~(a != normed).any(axis=0)])
    assert (a != normed).any(axis=0).sum() == 2
    masked_col_vals = a[:, (a != normed).any(axis=0)]
    std_obs = float(masked_col_vals.std())
    assert 0.02 < std_obs < 0.3


def test_mask_cache_reuses_object():
    spec = ChannelDropoutSpec(p=0.5, max_channels=3)
    m1 = keyed_drop_mask(len(COLS), COLS, "cd:0:7", spec)
    m2 = keyed_drop_mask(len(COLS), COLS, "cd:0:7", spec)
    assert m1 is m2  # lru_cache returns the same object for the same key


def test_temporal_pair_shares_mask_with_bucket():
    spec = SampleSpec(chart_window=16, feature_window=16, image_size=32, horizons=(4, 16, 64))
    set_seed(12)
    df = generate_market(MarketConfig(n_bars=400, freq="5min", seed=12))
    cd = ChannelDropoutSpec(p=0.9, max_channels=3, pair_bucket=16)
    ds = MarketDataset(df, spec=spec, cache_charts=False, compute_targets=False, channel_dropout=cd)
    a = ds[20]["numeric"]
    b = ds[24]["numeric"]  # same bucket (20//16 == 24//16 == 1)
    cols = ds._features_df.columns.tolist()
    expected = {cols[i] for i in np.where(
        keyed_drop_mask(len(cols), cols, "cd:0:1", cd))[0]}

    def masked_cols(x):
        return {cols[i] for i in range(len(cols)) if bool((x[:, i] == 0.0).all())}

    assert expected.issubset(masked_cols(a.numpy()))
    assert expected.issubset(masked_cols(b.numpy()))