"""LR schedule tests: constant (canonical) vs cosine decay in SSLPretrainer."""
from __future__ import annotations

import tempfile

import numpy as np
import pytest
import torch
from torch.utils.data import ConcatDataset, Dataset

from zhisa.models.policy import PolicyConfig, PolicyNetwork
from zhisa.training.s1_ssl import SSLConfig, SSLPretrainer


class _FakeDS(Dataset):
    def __init__(self, n: int, instrument: int = 0):
        self.n, self.instrument = n, instrument

    def __len__(self):
        return self.n

    def __getitem__(self, t: int) -> dict:
        rng = np.random.default_rng(1000 * self.instrument + t)
        return {
            "chart": torch.from_numpy(rng.random((3, 64, 64), dtype=np.float32)),
            "numeric": torch.from_numpy(rng.normal(0.0, 1.0, (32, 32)).astype(np.float32)),
            "context": torch.from_numpy(rng.uniform(-1.0, 1.0, (10,)).astype(np.float32)),
            "label_dir": torch.tensor(0, dtype=torch.long),
            "label_dir_persistence": torch.tensor(0, dtype=torch.long),
            "label_ret": torch.tensor(0.0),
            "label_dir_multi": torch.zeros(4, dtype=torch.long),
            "label_dir_multi_persistence": torch.zeros(4, dtype=torch.long),
            "label_ret_multi": torch.zeros(4),
            "label_vol": torch.tensor(0.0),
            "label_risk": torch.tensor(0.0),
            "label_regime": torch.tensor(0, dtype=torch.long),
            "mask": torch.ones(32, dtype=torch.bool),
            "instrument_id": torch.tensor(self.instrument, dtype=torch.long),
            "meta": {"ts": "x", "t": t, "instrument": "FAKE"},
        }


def _plain_ds() -> Dataset:
    return ConcatDataset([_FakeDS(40, 0), _FakeDS(40, 1)])


def _tr(**overrides) -> SSLPretrainer:
    model = PolicyNetwork(PolicyConfig(n_instruments=2))
    kw = dict(
        device="cpu", batch_size=4, projection_dim=16, hidden_dim=32, lr=1e-3,
        use_ema_teacher=False, use_masked_modeling=False, use_cross_modal=False,
        use_temporal_contrast=False, warmup_steps=0, epochs=1,
    )
    kw.update(overrides)
    return SSLPretrainer(model, SSLConfig(**kw))


def test_constant_default_preserves_canonical_lr():
    tr = _tr()
    assert tr.cfg.lr_schedule == "constant"
    loader = tr._loader(ConcatDataset([_FakeDS(400, 0), _FakeDS(400, 1)]), shuffle=False)
    import itertools
    for b in itertools.islice(loader, 4):
        tr.step(b)
    assert tr.opt.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_warmup_still_linear_for_cosine_mode():
    tr = _tr(lr_schedule="cosine", warmup_steps=100, total_steps=200)
    loader = tr._loader(ConcatDataset([_FakeDS(400, 0), _FakeDS(400, 1)]), shuffle=False)
    it = iter(loader)
    tr.step(next(it))  # step 0 -> lr = 1e-3 * (1/100)
    assert tr.opt.param_groups[0]["lr"] == pytest.approx(1e-3 * 0.01)
    for _ in range(48):
        tr.step(next(it))
    assert tr.opt.param_groups[0]["lr"] == pytest.approx(1e-3 * 0.49)  # step 48 -> (48+1)/100


def test_cosine_decays_to_min_scale_and_is_monotonic():
    tr = _tr(lr_schedule="cosine", warmup_steps=0, total_steps=10, cosine_min_scale=0.003)
    scales = []
    for _ in range(12):
        scales.append(float(tr._lr_scale(step=tr._step)))
        tr._step += 1
    assert scales[0] == pytest.approx(1.0)
    assert scales[-1] == pytest.approx(0.003, rel=0.05)
    for a, b in zip(scales, scales[1:]):
        assert a >= b - 1e-6
    assert tr._lr_scale(step=10_000) == pytest.approx(0.003)


def test_auto_total_steps_from_epochs_and_loader():
    tr = _tr(lr_schedule="cosine", warmup_steps=0, total_steps=0, epochs=3)
    src = _plain_ds()
    assert tr._estimated_total_steps == 0
    tr.fit(src)
    assert tr._estimated_total_steps == 3 * 20  # 80 samples / batch 4
    assert tr._lr_scale(step=tr._estimated_total_steps) == pytest.approx(0.003, rel=0.05)


def test_cosine_resume_gets_fresh_budget():
    """Regression: on RESUME the cosine must restart its schedule from the
    resumed step (anchor), NOT inherit a floor from the previous run's budget
    (measured bug: lr stuck at 4.5e-07 = cosine_min*full-lr after resume)."""
    src = _plain_ds()
    tr = _tr(lr_schedule="cosine", warmup_steps=50, total_steps=200, cosine_min_scale=0.003)
    tr._step = 150  # simulate a long previous run
    # save a "checkpoint" as if mid-run, then resume fresh
    path = str(tempfile.mkdtemp()) + "/resume_lr.pt"
    tr.save(path)
    tr2 = _tr(lr_schedule="cosine", warmup_steps=50, total_steps=200, cosine_min_scale=0.003)
    tr2.load(path)  # optimizer restored from state -> resume anchor set
    assert tr2._resume_step == 150
    # at the first new step the warmup restarts (fresh budget)
    scale = tr2._lr_scale(step=150)
    assert scale == pytest.approx(1.0 / 50, rel=0.1), f"lr must restart warmup on resume, got {scale}"
    # and the mid-budget point is NOT the floor
    mid = tr2._lr_scale(step=150 + 100)
    assert mid > 0.5, f"cosine ate the budget on resume: {mid}"
    # cold start stays untouched
    tr3 = _tr(lr_schedule="cosine", warmup_steps=50, total_steps=200)
    assert tr3._resume_step == 0


def test_cosine_ends_low_and_constant_differs_over_steps():
    tr_c = _tr()
    tr_k = _tr(lr_schedule="cosine", warmup_steps=0, total_steps=20)
    it_c = iter(tr_c._loader(_plain_ds(), shuffle=False))
    it_k = iter(tr_k._loader(_plain_ds(), shuffle=False))
    for _ in range(20):
        tr_c.step(next(it_c))
        tr_k.step(next(it_k))
    assert tr_k.opt.param_groups[0]["lr"] < 0.2 * tr_c.opt.param_groups[0]["lr"]