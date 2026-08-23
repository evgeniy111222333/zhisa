"""Temporal CPC v2: memory bank + hard negatives (full case coverage)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset

from zhisa.models.policy import PolicyConfig, PolicyNetwork
from zhisa.training.s1_ssl import (
    SSLConfig,
    SSLPretrainer,
    TemporalPairDataset,
    info_nce,
    temporal_pair_collate,
)


def _norm_rand(b: int, d: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(b, d), dim=-1)


_WINDOW = int(PolicyConfig().window)
_IMG = 64


class FakeDS(Dataset):
    """Deterministic tiny dataset matching the multimodal collate contract."""

    def __init__(self, n: int, instrument: int = 0) -> None:
        super().__init__()
        self.n = n
        self.instrument = instrument

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, t: int) -> dict:
        rng = np.random.default_rng(1000 * self.instrument + t)
        return {
            "chart": torch.from_numpy(rng.random((3, _IMG, _IMG), dtype=np.float32) * 0.5 + 0.2),
            "numeric": torch.from_numpy(rng.normal(0.0, 1.0, (_WINDOW, 32)).astype(np.float32)),
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
            "mask": torch.ones(_WINDOW, dtype=torch.bool),
            "instrument_id": torch.tensor(self.instrument, dtype=torch.long),
            "meta": {"ts": "x", "t": t, "instrument": "FAKE"},
        }


def _concat(n_per_leaf: int = 40, n_leaves: int = 2, horizon: int = 2) -> Dataset:
    ds = ConcatDataset([FakeDS(n_per_leaf, i) for i in range(n_leaves)])
    return TemporalPairDataset(ds, horizon=horizon)


def _trainer(**ssl_kwargs) -> SSLPretrainer:
    model = PolicyNetwork(PolicyConfig(n_instruments=2))
    overrides = dict(
        device="cpu", batch_size=4, projection_dim=16, hidden_dim=32, lr=1e-4,
        use_ema_teacher=True, use_masked_modeling=False, use_cross_modal=False,
        use_temporal_contrast=True, temporal_horizon=2, epochs=1,
    )
    overrides.update(ssl_kwargs)
    return SSLPretrainer(model, SSLConfig(**overrides))


def _clone_ssl(tr: SSLPretrainer, **ssl_kwargs) -> SSLPretrainer:
    clone = _trainer(**ssl_kwargs)
    clone.model.load_state_dict(tr.model.state_dict())
    clone.teacher.teacher.load_state_dict(tr.teacher.teacher.state_dict())
    clone.proj_temporal.load_state_dict(tr.proj_temporal.state_dict())
    clone.temporal_predictor.load_state_dict(tr.temporal_predictor.state_dict())
    clone.target_proj_temporal.load_state_dict(tr.target_proj_temporal.state_dict())
    if tr._bank is not None and clone._bank is not None:
        clone._bank = tr._bank.clone()
        clone._bank_gids = list(tr._bank_gids)
    return clone


# ---------------------------------------------------------------------------
# 1. info_nce extended semantics
# ---------------------------------------------------------------------------


def test_info_nce_canonical_math_unchanged():
    a, p = _norm_rand(6, 16, seed=1), _norm_rand(6, 16, seed=2)
    plain = info_nce(a, p, temperature=0.1)
    same = info_nce(a, p, temperature=0.1, extra_negatives=None, row_mask=None)
    assert torch.equal(plain, same)


def test_info_nce_extra_rows_3d_and_flat_agree():
    a, p = _norm_rand(4, 16, seed=3), _norm_rand(4, 16, seed=4)
    extra = _norm_rand(12, 16, seed=5).view(4, 3, 16)
    l3 = info_nce(a, p, temperature=0.1, extra_negatives=extra)
    lf = info_nce(a, p, temperature=0.1, extra_negatives=extra.view(12, 16))
    assert torch.equal(l3, lf)


def test_info_nce_harder_negatives_raise_loss():
    # Constructed deterministic vectors, B=1, D=2, tau=0.1:
    # anchor a, positive p (cos=1); easy negatives orthogona (cos=0);
    # hard negatives identical to the positive (cos=1).
    a = torch.tensor([[1.0, 0.0]])
    p = torch.tensor([[1.0, 0.0]])          # cos(a,p) = 1  -> logit 10
    easy = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]).view(1, 3, 2)
    hard = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]).view(1, 3, 2)
    l_easy = info_nce(a, p, temperature=0.1, extra_negatives=easy)
    l_hard = info_nce(a, p, temperature=0.1, extra_negatives=hard)
    assert l_hard.item() > l_easy.item()
    # sanity: easy-only loss ~ -log(exp(10)/(exp(10)+2)) ~ 0
    assert l_easy.item() < 0.1


def test_info_nce_mask_excludes_duplicate_positive():
    a, p = _norm_rand(3, 16, seed=9), _norm_rand(3, 16, seed=10)
    evil = torch.cat([p[0:1], _norm_rand(2, 16, seed=11)], dim=0)
    extra = evil.view(1, 3, 16).expand(3, 3, 16)
    mask = torch.zeros(3, 3 + 9, dtype=torch.bool)  # (B, positive_block + flattened extra)
    mask[0, 3] = True  # mask the duplicate of anchor 0's positive
    l_unmasked = info_nce(a, p, temperature=0.1, extra_negatives=extra)
    l_masked = info_nce(a, p, temperature=0.1, extra_negatives=extra, row_mask=mask)
    assert torch.isfinite(l_masked)
    assert l_masked.item() < l_unmasked.item()


def test_info_nce_diagonal_is_never_masked():
    a, p = _norm_rand(2, 16, seed=12), _norm_rand(2, 16, seed=13)
    brutal = torch.ones(2, 2, dtype=torch.bool)
    loss = info_nce(a, p, temperature=0.1, row_mask=brutal)
    assert torch.isfinite(loss)
    assert abs(loss.item()) < 1.0  # all negatives masked -> loss ~ -logit_diag ~ 0


# ---------------------------------------------------------------------------
# 2. TemporalPairDataset gids + offsets
# ---------------------------------------------------------------------------


def test_pairs_carry_gids_and_never_cross_leaves():
    src = _concat(n_per_leaf=30, n_leaves=2, horizon=2)
    cur, fut = src[5]
    assert cur["meta"]["gid"] == (0, 5)
    assert fut["meta"]["gid"] == (0, 7)
    curL, futL = src[len(src) - 1]
    assert curL["meta"]["gid"][0] == 1
    assert futL["meta"]["gid"] == (1, 29)


def test_offset_item_bounds_and_determinism():
    src = _concat(n_per_leaf=30, n_leaves=1, horizon=2)
    assert src.offset_item(0, 10, -4)["meta"]["gid"] == (0, 6)
    assert src.offset_item(0, 1, -4) is None  # negative index -> None
    assert src.offset_item(0, 1, 40) is None  # beyond leaf end -> None
    assert src.offset_item(0, 29, 8) is None
    assert torch.equal(src.offset_item(0, 12, 3)["numeric"], src.offset_item(0, 12, 3)["numeric"])


def test_collate_carries_gids():
    src = _concat(n_per_leaf=30, n_leaves=2, horizon=2)
    b = temporal_pair_collate([src[i] for i in range(4)])
    assert len(b["gids"]) == 4 and len(b["future_gids"]) == 4
    assert b["gids"][0] == (0, 0)
    assert b["future_gids"][0] == (0, 2)  # horizon=2


# ---------------------------------------------------------------------------
# 3. Bank mechanics
# ---------------------------------------------------------------------------


def test_bank_push_eviction_and_gid_alignment():
    tr = _trainer(temporal_bank_size=8, temporal_bank_warmup=0)
    tr._loader(_concat(), shuffle=False)
    for i in range(4):
        b = {"future_gids": [(0, i), (1, i), (0, i + 1), (1, i + 1)]}
        tr._push_bank(_norm_rand(4, 16, seed=i), b)
    assert tr._bank.size(0) == 8
    assert len(tr._bank_gids) == 8
    for i in range(3):
        b = {"future_gids": [(0, 9 + i), (1, 9 + i), (0, 10 + i), (1, 10 + i)]}
        tr._push_bank(_norm_rand(4, 16, seed=50 + i), b)
    assert tr._bank.size(0) == 8
    assert len(tr._bank_gids) == 8
    assert tr._bank_gids[0] != (0, 0)  # the very first entries were evicted


def test_bank_guards_missing_gids_and_nonfinite():
    tr = _trainer(temporal_bank_size=8, temporal_bank_warmup=0)
    tr._push_bank(torch.zeros(4, 16), {"future_gids": None})
    assert tr._bank.size(0) == 0
    tr._push_bank(torch.full((4, 16), float("nan")), {"future_gids": [(0, i) for i in range(4)]})
    assert tr._bank.size(0) == 0
    tr._push_bank(torch.zeros(4, 16), {"future_gids": [(0, i) for i in range(3)]})
    assert tr._bank.size(0) == 0


def test_bank_mask_dedups_by_gid():
    tr = _trainer(temporal_bank_size=8, temporal_bank_warmup=0)
    tr._bank = torch.zeros(3, 16)
    tr._bank_gids = [(0, 5), (0, 6), (1, 2)]
    m = tr._bank_mask({"chart": torch.zeros(2, 3, 64, 64), "future_gids": [(0, 5), (1, 2)]}, 3)
    assert bool(m[0, 0]) and bool(m[1, 2])
    assert not bool(m[0, 1]) and not bool(m[1, 0])


def test_bank_rows_respect_warmup_and_emptiness():
    tr = _trainer(temporal_bank_size=8, temporal_bank_warmup=128)
    assert tr._bank_rows({}) is None  # warmup not reached (step 0)
    tr._step = 200
    assert tr._bank_rows({}) is None  # still empty bank
    tr._bank = _norm_rand(4, 16, seed=1)
    assert tr._bank_rows({}) is not None


# ---------------------------------------------------------------------------
# 4. Hard negatives through the trainer
# ---------------------------------------------------------------------------


def test_hard_negatives_smoke_shapes_and_masking():
    src = _concat(n_per_leaf=60, n_leaves=2, horizon=2)
    tr = _trainer(temporal_bank_size=0, temporal_bank_warmup=0, temporal_hard_offsets=(-1, 2))
    loader = tr._loader(src, shuffle=True, epoch=0)
    b = next(iter(loader))
    hard, mask = tr._temporal_hard_negatives(b)
    B = b["chart"].size(0)
    assert hard is not None and hard.shape == (B * 2, 16)
    assert mask.shape == (B, B * 2)
    assert bool((~mask).sum(dim=1).gt(0).all())  # every anchor has >=1 kept hard row
    loss = tr._loss(b)
    assert torch.isfinite(loss["temporal"])


def test_hard_negative_offsets_allinvalid_returns_none():
    src = _concat(n_per_leaf=4, n_leaves=1, horizon=2)  # tiny leaf: shifts out of bounds
    tr = _trainer(temporal_bank_size=0, temporal_bank_warmup=0, temporal_hard_offsets=(-4, 7))
    loader = tr._loader(src, shuffle=True, epoch=0)
    b = next(iter(loader))
    hard, mask = tr._temporal_hard_negatives(b)
    assert (hard is None) or bool((~mask).sum().item() == 0)
    assert torch.isfinite(tr._loss(b)["temporal"])


def test_hard_negatives_anchor_without_valid_offsets():
    """Regression: a batch anchor at a leaf edge with NO valid shift must not
    crash the tensor layout (rows B*K slots vs len(populated)*K ids) and its
    invalid rows must stay masked out."""
    src = _concat(n_per_leaf=10, n_leaves=2, horizon=2)
    # off=+2h only: the anchor at local=7 has future gid (0, 9); 9+4=13>=10 -> None.
    # local=0: 0+4=4 valid; local=5: 5+4=9 valid.
    last_pair_item = src[len(src) - 1]  # leaf 1, local 7 -> no valid shift
    middle = src[len(src) // 2]                 # valid shift exists
    first = src[0]                              # leaf 0 local 0 -> valid
    items = [first, middle, last_pair_item, middle]  # B=4, anchor#2 has zero rows
    b = temporal_pair_collate(items)
    tr = _trainer(temporal_bank_size=0, temporal_bank_warmup=0, temporal_hard_offsets=(2,))
    tr._pair_source = src
    hard, mask = tr._temporal_hard_negatives(b)
    B, K = 4, 1
    assert hard is not None and hard.shape == (B * K, 16)
    assert mask.shape == (B, B * K)
    assert bool(mask[2, 2])                  # anchor 2's own zero-row is excluded
    assert bool((~mask[0]).sum().item() > 0)  # anchors 0,1,3 keep their row
    loss = tr._loss(b)
    assert torch.isfinite(loss["temporal"])


def test_full_steps_with_bank_and_hard_deterministic():
    def run(seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        tr = _trainer(temporal_bank_size=16, temporal_bank_warmup=0, temporal_hard_offsets=(-1, 2))
        loader = tr._loader(_concat(n_per_leaf=60, n_leaves=2), shuffle=True, epoch=0)
        losses = []
        for b in loader:
            losses.append(tr.step(b)["temporal"])
            if tr._step >= 6:
                break
        return losses, int(tr._bank.size(0))

    l1, n1 = run(7)
    l2, n2 = run(7)
    assert n1 == n2 == 16
    assert len(l1) == len(l2)
    for x, y in zip(l1, l2):
        assert x == pytest.approx(y, abs=1e-6)


def test_eval_mode_ignores_bank_and_hard():
    src = _concat(n_per_leaf=60, n_leaves=2, horizon=2)
    tr = _trainer(temporal_bank_size=16, temporal_bank_warmup=0, temporal_hard_offsets=(-1, 2))
    loader = tr._loader(src, shuffle=True, epoch=0)
    b = next(iter(loader))
    tr.model.train()
    tr._loss(b)
    assert tr._bank.size(0) == 4
    tr.model.eval()
    bank_before = tr._bank.clone()
    l1 = tr._loss(b)["temporal"]
    l2 = tr._loss(b)["temporal"]
    assert torch.equal(tr._bank, bank_before)  # no push in eval
    assert l1.item() == pytest.approx(l2.item(), abs=1e-6)
    canonical_clone = _clone_ssl(tr, temporal_bank_size=0)
    canonical_clone.model.eval()
    canonical = canonical_clone._loss(b)["temporal"]
    assert l1.item() == pytest.approx(canonical.item(), abs=1e-5)


def test_train_loss_differs_from_canonical_when_features_on():
    src = _concat(n_per_leaf=60, n_leaves=2, horizon=2)
    tr = _trainer(temporal_bank_size=16, temporal_bank_warmup=0, temporal_hard_offsets=(-1, 2))
    loader = tr._loader(src, shuffle=True, epoch=0)
    b = next(iter(loader))
    tr.model.train()
    assert tr._bank is not None and tr._bank.size(0) == 0
    v2 = tr._loss(b)["temporal"]
    canonical = _clone_ssl(tr, temporal_bank_size=0)._loss(b)["temporal"]
    assert torch.isfinite(v2)
    assert v2.item() != pytest.approx(canonical.item(), abs=1e-5)


def test_hard_negatives_uses_batch_instrument_not_leaf_index():
    """Regression: with multiple segments per instrument, leaf index can
    exceed n_instruments; the teacher's context embedding must use the
    BATCH instrument id (segment 3 belongs to instrument 0)."""
    from torch.utils.data import ConcatDataset as CD
    ds = CD([FakeDS(30, 0), FakeDS(30, 1), FakeDS(30, 0)])
    src = TemporalPairDataset(ds, horizon=2)
    tr = _trainer(temporal_bank_size=0, temporal_bank_warmup=0, temporal_hard_offsets=(-1, 2))
    tr._pair_source = src
    b = temporal_pair_collate([src[66], src[70], src[74], src[78]])  # leaf 2 (instr 0)
    assert b["instrument_id"].tolist() == [0, 0, 0, 0]
    hard, mask = tr._temporal_hard_negatives(b)
    assert hard is not None
    assert torch.isfinite(tr._loss(b)["temporal"])


# ---------------------------------------------------------------------------
# 6. v2.5 loss terms (trunk-align + instrument contrast)
# ---------------------------------------------------------------------------


def test_trunk_align_term_moves_trunk_cos_up():
    """The trunk-level term must push raw trunk cos(v, n_cls) upward over
    steps, while the baseline (weight 0) shows only drift."""
    def run(weight: float, momentum: float):
        torch.manual_seed(0)
        np.random.seed(0)
        tr = _trainer(weight_trunk_align=weight, trunk_align_momentum=momentum,
                      use_cross_modal=True, use_masked_modeling=False)
        loader = tr._loader(_concat(n_per_leaf=60, n_leaves=2, horizon=2),
                            shuffle=True, epoch=0)
        src = tr._pair_source
        for i, b in enumerate(loader):
            tr.step(b)
            if tr._step >= 20:
                break
        with torch.no_grad():
            rng = np.random.default_rng(3)
            idx = rng.choice(len(src), size=32, replace=False)
            items = [src[int(i)] for i in idx]
            b = temporal_pair_collate(items)
            v = tr.model.plain_vision(b["chart"])
            n_cls, _ = tr.model.numeric(b["numeric"])
            return float(F.cosine_similarity(v.reshape(v.size(0), -1), n_cls, dim=-1).mean().item())

    b0 = run(0.0, 0.0)
    b1 = run(0.5, 0.99)
    assert b1 > b0  # the term moves trunk cos further after 20 steps


def test_instrument_contrast_term_pushes_embeddings_apart():
    def offdiag_mean(tr):
        emb = tr.model.context.instrument_emb.weight
        e = torch.nn.functional.normalize(emb, dim=-1)
        c = e @ e.t()
        off = c[~torch.eye(c.size(0), dtype=torch.bool)]
        return float(off.mean().item())

    torch.manual_seed(0)
    np.random.seed(0)
    plain = ConcatDataset([FakeDS(60, 0), FakeDS(60, 1)])
    tr = _trainer(instrument_contrast_w=0.5, use_cross_modal=False,
                  use_masked_modeling=False, use_temporal_contrast=False)
    loader = tr._loader(plain, shuffle=True, epoch=0)
    start = offdiag_mean(tr)
    for i, b in enumerate(loader):
        tr.step(b)
        if tr._step >= 15:
            break
    end = offdiag_mean(tr)
    assert end < start - 1e-4
    # disabled term must be a no-op for the loss dict (default trainer)
    tr0 = _trainer()
    loader0 = tr0._loader(_concat(n_per_leaf=60, n_leaves=2, horizon=2),
                          shuffle=True, epoch=0)
    b0 = next(iter(loader0))
    assert "instrument_contrast" not in tr0._loss(b0)


def test_v25_config_fields_parse():
    from zhisa.scripts.train_s1 import _ssl_config_from
    import yaml as _y
    cfg = _y.safe_load(open(r"D:\zhisa\configs\s1_ssl_1h_12m_heavy_v2_5.yaml",
                            encoding="utf-8"))
    ssl = _ssl_config_from(cfg)
    assert ssl.weight_trunk_align == 0.25
    assert ssl.trunk_align_momentum == 0.99
    assert ssl.instrument_contrast_w == 0.02
    assert ssl.lr_schedule == "cosine"
    assert ssl.cosine_min_scale == 0.003


def test_bank_survives_checkpoint_roundtrip():
    src = _concat(n_per_leaf=60, n_leaves=2, horizon=2)
    tr = _trainer(temporal_bank_size=16, temporal_bank_warmup=0)
    loader = tr._loader(src, shuffle=True, epoch=0)
    for b in loader:
        tr.step(b)
        if tr._step >= 4:
            break
    path = str(Path(tempfile.mkdtemp()) / "ssl_bank.pt")
    tr.save(path)
    tr2 = _trainer(temporal_bank_size=16, temporal_bank_warmup=0)
    tr2.load(path)
    assert tr2._bank is not None and tr2._bank.size(0) == tr._bank.size(0)
    assert tr2._bank_gids == tr._bank_gids
    torch.testing.assert_close(tr2._bank, tr._bank)


def test_disabled_features_behave_canonically():
    src = _concat(n_per_leaf=60, n_leaves=2, horizon=2)
    tr = _trainer()
    assert tr._bank is None
    loader = tr._loader(src, shuffle=True, epoch=0)
    b = next(iter(loader))
    assert tr._temporal_hard_negatives(b) == (None, None)
    assert torch.isfinite(tr._loss(b)["temporal"])


def test_checkpoint_load_tolerates_numeric_width_change():
    """v3.1 scenario: more numeric columns than the checkpoint. The numeric
    trunk re-initialises (width changed), the rest transfers shape-matched,
    and the reconstructor head (width-dependent shape) must not crash load."""
    import tempfile
    from pathlib import Path

    tr32 = _trainer()
    path = str(Path(tempfile.mkdtemp()) / "w32.pt")
    tr32.save(path)
    model43 = PolicyNetwork(PolicyConfig(n_instruments=2, in_numeric_features=43))
    tr43 = SSLPretrainer(
        model43,
        SSLConfig(device="cpu", batch_size=4, projection_dim=16, hidden_dim=32,
                  lr=1e-4, use_ema_teacher=True, use_masked_modeling=True,
                  use_cross_modal=False, use_temporal_contrast=True,
                  temporal_horizon=2, epochs=1),
    )
    tr43.load(path)
    # vision trunk transferred
    sd32 = tr32.model.vision.state_dict()
    sd43 = tr43.model.vision.state_dict()
    for k in sd32:
        torch.testing.assert_close(sd43[k], sd32[k], msg=f"vision/{k}")
    # numeric trunk re-initialised because input width differs
    assert tr32.model.numeric.patch_proj.weight.shape != tr43.model.numeric.patch_proj.weight.shape
    # reconstructor head restored shape-agnostically (no crash; width-adapted)
    assert tr43.reconstructor.head.out_features == (
        tr43.model.numeric.cfg.patch_size * tr43.model.numeric.cfg.in_features
    )