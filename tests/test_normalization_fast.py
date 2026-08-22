"""Tests for the fast/robust numeric normalization layer."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from zhisa.data.dataset import MarketDataset, SampleSpec
from zhisa.features.normalization import (
    NormalizationSpec,
    PrefixStats,
    normalize_window,
    robust_z,
)
from zhisa.features.ohlcv import normalize_feature_window


def _table(n: int = 2048, f: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal((n, f)), axis=0)
    x[np.abs(x) < 0.02] = np.nan  # sprinkle non-finite like a real feature frame
    return x


def _windows(table, k: int = 40, window: int = 64, lookback: int = 256):
    rng = np.random.default_rng(1)
    out = []
    n = len(table)
    for t in sorted(rng.choice(max(1, n - window - lookback), size=k, replace=False)):
        start = int(t)
        end = start + window
        hist_start = max(0, start - lookback)
        fw = table[start:end]
        hw = table[hist_start:end]
        out.append((fw, hw, start, end, hist_start))
    return out


def test_prefix_equals_classic_zscore():
    table = _table()
    ps = PrefixStats(table)
    for fw, hw, start, end, hist_start in _windows(table):
        got = ps.zscore_window(fw, hist_start, end)
        exp = normalize_feature_window(fw, hw)
        assert np.allclose(got, exp, atol=1e-5, rtol=1e-4), (start, float(np.abs(got - exp).max()))


def test_prefix_stats_meanstd_matches_numpy():
    table = _table(100, 4)
    ps = PrefixStats(table)
    for lo, hi in [(0, 50), (10, 90), (99, 100)]:
        mu, sd = ps.mean_std(lo, hi)
        assert np.allclose(mu, np.nan_to_num(table[lo:hi]).mean(axis=0), atol=1e-9)
        # count=1 windows carry ~1e-13 f.p. round-trip noise; allow 2e-6
        assert np.allclose(sd - 1e-6, np.nan_to_num(table[lo:hi]).std(axis=0), atol=2e-6)


def test_prefix_clamps_bounds():
    table = _table(50, 3)
    ps = PrefixStats(table)
    mu0, sd0 = ps.mean_std(-100, 1000)
    assert np.allclose(mu0, np.nan_to_num(table).mean(axis=0))


def test_prefix_build_constant_column_not_nan():
    table = np.ones((200, 4)) * 5.0
    ps = PrefixStats(table)
    _, sd = ps.mean_std(0, 200)
    assert np.isfinite(sd).all() and (sd >= 1e-6).all()


def test_robust_z_matches_manual():
    table = _table(300, 4)
    fw, hw, *_ = _windows(table, k=1)[0]
    got = robust_z(fw, hw, eps=1e-6)
    hist = np.nan_to_num(hw)
    med = np.median(hist, axis=0)
    mad = np.median(np.abs(hist - med), axis=0)
    exp = (np.nan_to_num(fw) - med) / (1.4826 * mad + 1e-6)
    assert np.allclose(got, exp.astype(np.float32))


def test_robust_z_resists_outlier():
    base = np.random.default_rng(0).standard_normal((300, 1)) * 0.5
    clean = np.concatenate([base, base], axis=1)
    with_outlier = np.concatenate([base.copy(), base.copy()], axis=1)
    with_outlier[150, 1] = 500.0  # single huge spike in column 1

    fw_clean = np.concatenate([np.zeros((4, 1)), np.zeros((4, 1))], axis=1)
    z_clean = robust_z(fw_clean, clean)
    z_polluted = robust_z(fw_clean, with_outlier)
    # robust MAD barely moves; classic std would be dominated by the spike.
    assert abs(float(z_polluted[0, 1] - z_clean[0, 1])) < 1.0
    classic_clean = normalize_feature_window(fw_clean, clean)
    classic_poll = normalize_feature_window(fw_clean, with_outlier)
    assert abs(float(classic_poll[0, 1] - classic_clean[0, 1])) > abs(float(z_polluted[0, 1] - z_clean[0, 1]))


def test_normalization_spec_hash_and_meta():
    a = NormalizationSpec()
    assert a.content_hash() == NormalizationSpec().content_hash()
    assert a.content_hash() != NormalizationSpec(mode="robust_z").content_hash()
    assert NormalizationSpec.from_meta(a.to_meta()) == a
    with pytest.raises(ValueError):
        NormalizationSpec(mode="bogus")


def test_dataset_rolling_z_matches_classic():
    from zhisa.data.synthetic import MarketConfig, generate_market
    from zhisa.utils.seeding import set_seed
    set_seed(11)
    df = generate_market(MarketConfig(n_bars=800, freq="5min", seed=11))
    spec = SampleSpec(chart_window=64, feature_window=64, image_size=32, horizons=(4, 16, 64))
    set_seed(12)
    ds = MarketDataset(df, spec=spec, cache_charts=False, compute_targets=False)
    # reference computed the classic way
    from zhisa.features.ohlcv import compute_ohlcv_features
    feats = compute_ohlcv_features(df, include_volume=True, include_indicators=True).to_numpy(dtype=np.float32)
    for t in [0, 100, len(ds) - 1]:
        s = ds[t]
        start, end = t, t + spec.chart_window
        hist_start = max(0, t - 256)
        exp = normalize_feature_window(feats[start:end], feats[hist_start:end])
        assert np.allclose(s["numeric"].numpy(), exp, atol=1e-5), t


def test_dataset_robust_mode_works():
    from zhisa.data.synthetic import MarketConfig, generate_market
    from zhisa.utils.seeding import set_seed
    set_seed(13)
    df = generate_market(MarketConfig(n_bars=500, freq="5min", seed=13))
    spec = SampleSpec(chart_window=64, feature_window=64, image_size=32, horizons=(4, 16, 64))
    ds = MarketDataset(
        df, spec=spec, cache_charts=False, compute_targets=False,
        normalization=NormalizationSpec(mode="robust_z"),
    )
    num = ds[50]["numeric"].numpy()
    assert num.shape == (64, ds._features_df.shape[1])
    assert np.isfinite(num).all()


def test_dataset_default_equals_normalization_spec_rolling():
    assert NormalizationSpec().mode == "rolling_z"
    assert NormalizationSpec().lookback == 256