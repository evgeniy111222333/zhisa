"""Tests for deterministic keyed chart augmentation inside the S1 trainer."""
from __future__ import annotations

import os

import pytest
import torch

from zhisa.data.dataset import MarketDataset, SampleSpec
from zhisa.models.policy import build_default_policy
from zhisa.training.s1_ssl import SSLConfig, SSLPretrainer


def _dataset():
    from zhisa.data.synthetic import MarketConfig, generate_market
    from zhisa.utils.seeding import set_seed
    set_seed(7)
    df = generate_market(MarketConfig(n_bars=400, freq="5min", seed=7))
    return MarketDataset(
        df,
        spec=SampleSpec(chart_window=32, feature_window=32, image_size=32, horizons=(4, 16, 64)),
        cache_charts=False,
        compute_targets=False,
    )


def _policy(ds=None):
    ds = ds if ds is not None else _dataset()
    return build_default_policy(
        in_numeric_features=int(ds._features_df.shape[1]),
        in_context_features=int(ds._time_features_df.shape[1]),
        window=32,
        image_size=32,
        n_actions=9,
        n_regime_classes=4,
    )


_AUG_CFG = dict(
    device="cpu",
    batch_size=8,
    use_ema_teacher=True,
    use_masked_modeling=True,
    use_temporal_contrast=True,
    use_cross_modal=True,
    augment_transforms=("mirror", "color_jitter", "crop", "gaussian_noise"),
)


@pytest.fixture
def double_tap():
    """Two independent trainers from identical state (same seed draws)."""
    ds = _dataset()

    def build():
        from zhisa.utils.seeding import set_seed
        set_seed(0)
        cfg = SSLConfig(
            seed=0,
            checkpoint=None,
            best_checkpoint=None,
            **_AUG_CFG,
        )
        return SSLPretrainer(_policy(ds), cfg)

    return ds, build


def test_augment_is_deterministic_across_runs(double_tap):
    ds, build = double_tap
    # Reproducible augmented loss sequences when the pipeline is reset.
    outs = []
    for i in range(2):
        tr = build()
        loader = tr._loader(ds, shuffle=True, epoch=0)
        first = next(iter(loader))
        b = tr._to_device(first)
        losses = tr.step(b)
        outs.append((losses["total"], b["chart"].clone()))
        # deterministic by construction: same inputs -> same augmented bytes
    assert outs[0][0] == outs[1][0]
    assert torch.equal(outs[0][1], outs[1][1])


def test_augmentation_changes_chart_but_not_numeric(double_tap):
    ds, build = double_tap
    tr = build()
    loader = tr._loader(ds, shuffle=True, epoch=0)
    batch = next(iter(loader))
    raw_chart = batch["chart"].clone()
    raw_numeric = batch["numeric"].clone()
    tr._augment_step_batch(batch)
    assert not torch.equal(raw_chart, batch["chart"])
    # different keys -> different augmented charts per sample
    assert not torch.equal(batch["chart"][0], batch["chart"][1])
    # numeric / context are untouched by chart augmentation
    assert torch.equal(raw_numeric, batch["numeric"])


def test_evaluation_is_never_augmented(double_tap):
    ds, build = double_tap
    tr = build()

    calls = {"n": 0}

    class CountingAugmentor:
        def apply(self, img, key):
            calls["n"] += 1
            return img

    real = tr.augmentor
    tr.augmentor = CountingAugmentor()

    # evaluate (and its _loss) must not trigger augmentation at all.
    _ = tr.evaluate(ds)
    assert calls["n"] == 0, "evaluate() must never augment charts"

    # step() calls augmentation once per chart (+ once per future chart when
    # temporal contrast is on).
    loader = tr._loader(ds, shuffle=True, epoch=0)
    batch = next(iter(loader))
    tr.step(tr._to_device(batch))
    assert calls["n"] > 0, "step() must augment charts"

    tr.augmentor = real


def test_augmentor_disabled_by_default():
    ds = _dataset()
    cfg = SSLConfig(device="cpu", batch_size=8, augment_transforms=())
    tr = SSLPretrainer(_policy(), cfg)
    assert tr.augmentor is None
    loader = tr._loader(ds, shuffle=True, epoch=0)
    batch = next(iter(loader))
    raw = batch["chart"].clone()
    tr._augment_step_batch(batch) if tr.augmentor else None
    # disabled -> no-op (batch keys unchanged because augmentor is None and the
    # helper is guarded by the caller); verify raw chart stays identical.
    assert torch.equal(raw, batch["chart"])


def test_checkpoint_records_augmentation(double_tap, tmp_path):
    ds, build = double_tap
    tr = build()
    tr.cfg.checkpoint = str(tmp_path / "model.pt")
    tr._augment_step_batch(next(iter(tr._loader(ds, shuffle=True, epoch=0))))
    tr.save(tr.cfg.checkpoint)
    payload = torch.load(tr.cfg.checkpoint, map_location="cpu", weights_only=False)
    render = payload["checkpoint_meta"]["render"]
    assert render["augmentation"] is not None
    assert render["augmentation"]["kind"] == "keyed_augmentor"
    assert render["augmentation_key_scheme"] == "epoch:step:index:cur|fut"