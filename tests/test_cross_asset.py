"""Tests for cross-asset / market-breadth enrichment."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zhisa.data.cross_asset import (
    build_market_index,
    enrich_frame,
    enrich_market_frames,
    enrich_market_frames_detailed,
    symbol_logret,
)


def _frame(n=600, drift=0.0, noise=1.0, seed=0, start="2024-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    close = 100 * np.exp(np.cumsum(drift + rng.standard_normal(n) * noise))
    return pd.DataFrame(
        {"open": close, "high": close * 1.002, "low": close * 0.998,
         "close": close, "volume": rng.uniform(100, 200, n)},
        index=idx,
    )


def test_index_excludes_self_and_is_deterministic():
    a, b, c = _frame(seed=1), _frame(seed=2), _frame(seed=3)
    lrs = {"A": symbol_logret(a), "B": symbol_logret(b), "C": symbol_logret(c)}
    ia = build_market_index({k: v for k, v in lrs.items() if k != "A"})
    ib = build_market_index({k: v for k, v in lrs.items() if k != "A"})
    assert ia.equals(ib)
    assert not ia.isna().all()
    assert set(ia.index) == set(a.index)


def test_enrich_adds_expected_columns_and_is_causal():
    asset, aux = _frame(seed=0, drift=0.001), _frame(seed=5, drift=0.001)
    ref = symbol_logret(aux)
    enriched = enrich_frame(asset, ref, windows=(64, 256))
    for col in ("rel_logret_1", "beta_64", "beta_256", "corr_64", "corr_256"):
        assert col in enriched.columns, col
    # causality: modifying FULLY-FUTURE bars must not change past values
    past = enriched.loc[enriched.index[100], ["beta_64", "beta_256", "corr_64", "corr_256"]].copy()
    tampered = asset.copy()
    tampered.iloc[300:, 3] *= 1.5  # future close differs
    enriched2 = enrich_frame(tampered, ref, windows=(64, 256))
    for col in past.index:
        assert np.isclose(enriched.loc[enriched.index[100], col], enriched2.loc[enriched2.index[100], col], atol=1e-12), col
    # but current-row value SHOULD reflect co-movement
    assert enriched["corr_64"].notna().mean() > 0.9


def test_beta_sanity_on_known_covariance():
    # A = 0.5*B + noise -> beta ~ 0.5 (when B is the ref)
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=800, freq="1h", tz="UTC")
    rb = rng.standard_normal(800) * 0.02
    ra = 0.5 * rb + rng.standard_normal(800) * 0.001
    close_a = 100 * np.exp(np.cumsum(ra))
    close_b = 100 * np.exp(np.cumsum(rb))
    frame_a = pd.DataFrame({"open": close_a, "high": close_a, "low": close_a,
                            "close": close_a, "volume": 1.0}, index=idx)
    frame_b = pd.DataFrame({"open": close_b, "high": close_b, "low": close_b,
                            "close": close_b, "volume": 1.0}, index=idx)
    enriched = enrich_frame(frame_a, symbol_logret(frame_b), windows=(256,))
    beta = enriched["beta_256"].dropna().median()
    assert 0.35 < beta < 0.65, beta


def test_enrich_market_frames_deterministic_and_self_free():
    frames = {"A": _frame(seed=1), "B": _frame(seed=2), "C": _frame(seed=3)}
    e1 = enrich_market_frames(frames)
    e2 = enrich_market_frames(frames)
    for sym in frames:
        assert e1[sym].equals(e2[sym])
        assert "rel_logret_1" in e1[sym].columns
    # A's reference is the mean of B and C — beta_256 of A should be finite
    assert e1["A"]["beta_256"].dropna().shape[0] > 0


def test_single_symbol_universe_schema():
    frames = {"A": _frame(seed=1)}
    e = enrich_market_frames(frames)
    for c in ("rel_logret_1", "beta_64", "beta_256", "corr_64", "corr_256"):
        assert c in e["A"].columns
        assert e["A"][c].isna().all()


def test_min_periods2_index_nan_on_single_source():
    a, b = _frame(seed=1), _frame(seed=2)
    lr = {"A": symbol_logret(a), "B": symbol_logret(b)}
    lr["B"].iloc[10] = np.nan
    idx = build_market_index(lr)
    assert pd.isna(idx.iloc[10])
    assert idx.notna().sum() > len(idx) - 10


def test_volume_ratios_gated():
    a, b = _frame(seed=1), _frame(seed=2)
    ref = symbol_logret(b)
    plain = enrich_frame(a, ref)
    assert not any(c.startswith("volume_ratio") or c.startswith("volvol_ratio") for c in plain.columns)
    with_v = enrich_frame(a, ref, with_volume_ratios=True, ref_volume=b["volume"])
    for w in (64, 256):
        assert f"volume_ratio_{w}" in with_v.columns
        assert f"volvol_ratio_{w}" in with_v.columns


def test_volume_ratios_causal():
    a, b = _frame(seed=1), _frame(seed=2)
    ref = symbol_logret(b)
    e1 = enrich_frame(a, ref, with_volume_ratios=True, ref_volume=b["volume"])
    past = float(e1.loc[e1.index[30], "volume_ratio_64"])
    tam = a.copy()
    tam.iloc[150:, -1] *= 3.0  # future volume different
    e2 = enrich_frame(tam, ref, with_volume_ratios=True, ref_volume=b["volume"])
    assert np.isclose(float(e2.loc[e2.index[30], "volume_ratio_64"]), past, atol=1e-12)


def test_detailed_audit_structure():
    frames = {"A": _frame(seed=1), "B": _frame(seed=2)}
    _, audit = enrich_market_frames_detailed(frames)
    for sym in frames:
        assert audit[sym]["refs"]
        assert hasattr(audit[sym]["index"], "index")
        assert isinstance(audit[sym]["na_frac"], float)
        assert "mean_beta_256" in audit[sym]


def test_enrich_on_15m_grid_with_gap():
    idx = pd.date_range("2024-01-01", periods=300, freq="15min", tz="UTC")
    rng = np.random.default_rng(0)
    frames = {}
    for s in ("A", "B", "C"):
        close = 100 * np.exp(np.cumsum(rng.standard_normal(300) * 0.01))
        frames[s] = pd.DataFrame(
            {"open": close, "high": close * 1.001, "low": close * 0.999,
             "close": close, "volume": rng.uniform(1, 2, 300)}, index=idx)
    frames["B"].drop(index=idx[50], inplace=True)  # a hole in one symbol only
    e = enrich_market_frames(frames)
    for s in frames:
        assert e[s].index.equals(frames[s].index)
        assert e[s]["beta_256"].notna().sum() > 100


def test_integration_dataset_and_ssl_step_with_cross_asset():
    import torch
    from torch.utils.data import ConcatDataset
    from zhisa.data.dataset import MarketDataset, SampleSpec
    from zhisa.models.policy import build_default_policy
    from zhisa.training.s1_ssl import SSLConfig, SSLPretrainer

    frames = {"A": _frame(seed=1, n=700), "B": _frame(seed=2, n=700)}
    e = enrich_market_frames(frames)
    spec = SampleSpec(chart_window=32, feature_window=32, image_size=32, horizons=(4, 16, 64))
    sets = [MarketDataset(e[s], spec=spec, cache_charts=False, compute_targets=False,
                          instrument_id=i) for i, s in enumerate(e)]
    base = MarketDataset(frames["A"], spec=spec, cache_charts=False, compute_targets=False)
    # +6: 5 cross-asset ctx features + global ctx_available_frac
    assert sets[0]._features_df.shape[1] == base._features_df.shape[1] + 6
    ds0 = sets[0]
    policy = build_default_policy(
        in_numeric_features=int(ds0._features_df.shape[1]),
        in_context_features=int(ds0._time_features_df.shape[1]),
        window=32, image_size=32, n_actions=9, n_regime_classes=4,
        embed_dim=96, vision_channels=(32, 64), numeric_layers=1,
        encoder_ff_mult=2.0, n_instruments=2,
    )
    tr = SSLPretrainer(policy, SSLConfig(device="cpu", batch_size=4, use_ema_teacher=True,
                                         use_masked_modeling=True, use_temporal_contrast=True,
                                         use_cross_modal=True))
    loader = tr._loader(ConcatDataset(sets), shuffle=True, epoch=0)
    b = next(iter(loader))
    losses = tr.step(tr._to_device(b))
    assert torch.isfinite(torch.tensor(losses["total"]))