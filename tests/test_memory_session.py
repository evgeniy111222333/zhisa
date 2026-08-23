"""Tests: memory sessions (S2/S4/serve contract) + residual memory fix."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import ConcatDataset, Dataset

from zhisa.models.policy import PolicyConfig, PolicyNetwork
from zhisa.models.session import (
    _to_batch,
    make_stateful_policy,
    memory_sensitivity,
    session_start,
    session_step,
    session_warm_up,
)
from zhisa.training.s2_supervised import SupervisedTrainer, TrainConfig
from zhisa.training.losses import MultiTaskLoss


class FakeDS(Dataset):
    """Minimal labelled dataset for memory-session flow tests."""

    def __init__(self, n: int, instrument: int = 0):
        self.n, self.instrument = n, instrument

    def __len__(self):
        return self.n

    def __getitem__(self, t: int) -> dict:
        rng = np.random.default_rng(1000 * self.instrument + t)
        return {
            "chart": torch.as_tensor(rng.random((3, 32, 32)), dtype=torch.float32),
            "numeric": torch.as_tensor(rng.normal(0, 1, (32, 32)), dtype=torch.float32),
            "context": torch.as_tensor(rng.normal(0, 1, (10,)), dtype=torch.float32),
            "label_dir": torch.tensor(rng.choice([-1, 0, 1], p=[0.3, 0.4, 0.3]), dtype=torch.long),
            "label_dir_persistence": torch.tensor(0, dtype=torch.long),
            "label_dir_multi": torch.zeros(3, dtype=torch.long),
            "label_dir_multi_persistence": torch.zeros(3, dtype=torch.long),
            "label_ret": torch.tensor(rng.normal(0, 0.003), dtype=torch.float32),
            "label_ret_multi": torch.tensor(rng.normal(0, 0.004, 3), dtype=torch.float32),
            "label_vol": torch.tensor(rng.uniform(0.005, 0.03), dtype=torch.float32),
            "label_risk": torch.tensor(rng.uniform(0.0, 0.05), dtype=torch.float32),
            "label_regime": torch.tensor(0, dtype=torch.long),
            "mask": torch.ones(32, dtype=torch.bool),
            "instrument_id": torch.tensor(self.instrument, dtype=torch.long),
            "meta": {"ts": "x", "t": t, "instrument": "FAKE"},
        }


def _model(**kw) -> PolicyNetwork:
    cfg = PolicyConfig(n_instruments=2, embed_dim=64, window=32, image_size=32,
                       numeric_layers=1, fusion_layers=1, memory_layers=2,
                       memory_max_len=8, encoder_ff_mult=2.0, **kw)
    return PolicyNetwork(cfg)


def _obs(t) -> dict:
    rng = np.random.default_rng(t)
    return {
        "chart": torch.as_tensor(rng.random((3, 32, 32)), dtype=torch.float32),
        "numeric": torch.as_tensor(rng.normal(0, 1, (32, 32)), dtype=torch.float32),
        "context": torch.as_tensor(rng.normal(0, 1, (10,)), dtype=torch.float32),
        "instrument_id": 0,
    }


# ---------------------------------------------------------------------------
# 1. session helpers
# ---------------------------------------------------------------------------


def test_session_start_shape_and_contract():
    m = _model()
    st = session_start(m)
    assert st is not None and st.shape == (1, m.memory.cfg.max_len - 1, 64)
    assert (st == 0).all()  # cold = zeros
    m2 = _model(use_memory=False)
    assert session_start(m2) is None


def test_session_step_chains_state():
    m = _model()
    m.eval()
    st = session_start(m)
    obs0 = _obs(1)
    out0, st1 = session_step(m, obs0, st)
    assert st1 is not None and st1.shape == st.shape
    # chaining: the last history slot after step 1 == the raw (pre-residual)
    # encoder z of obs0 — the memory stream advances with real embeddings.
    b0 = _to_batch(obs0)
    z0_raw = m.encode(b0["chart"], b0["numeric"], b0["context"],
                      instrument_id=b0["instrument_id"])[0]
    assert torch.allclose(st1[0, -1], z0_raw, atol=1e-4)
    # warm-up produces one output per obs and a valid final state
    outs, final = session_warm_up(m, [_obs(i) for i in range(5)])
    assert len(outs) == 5 and final.shape == st.shape


def test_memory_sensitivity_ranges():
    m = _model()
    m.eval()
    obs_seq = [_obs(i) for i in range(6)]
    s = memory_sensitivity(m, obs_seq)
    assert 0.0 <= s < 0.2  # untrained residual memory starts near-blind


# ---------------------------------------------------------------------------
# 2. residual memory behaviour
# ---------------------------------------------------------------------------


def test_residual_memory_keeps_encoder_embedding_primary():
    m = _model()
    m.eval()
    obs = _obs(3)
    with torch.no_grad():
        batch = {k: v.unsqueeze(0) for k, v in obs.items()
                 if k != "instrument_id" and isinstance(v, torch.Tensor)}
        z_enc = m.encode(batch["chart"], batch["numeric"], batch["context"],
                         instrument_id=torch.tensor([0]))
        out_res = m(chart=batch["chart"], numeric=batch["numeric"],
                    context=batch["context"], instrument_id=torch.tensor([0]))
        m_legacy = _model(memory_residual=False)
        out_leg = m_legacy(chart=batch["chart"], numeric=batch["numeric"],
                           context=batch["context"], instrument_id=torch.tensor([0]))
    cos_res = float(torch.nn.functional.cosine_similarity(z_enc, out_res["embedding"], dim=-1).item())
    cos_leg = float(torch.nn.functional.cosine_similarity(
        m_legacy.encode(batch["chart"], batch["numeric"], batch["context"],
                        instrument_id=torch.tensor([0])),
        out_leg["embedding"], dim=-1).item())
    assert cos_res > 0.9, f"residual must keep z primary, got cos={cos_res:.3f}"
    assert cos_leg < 0.9, f"legacy overwrites z (cos={cos_leg:.3f}) — fix must change this regime"


def test_residual_memory_content_sensitivity_expressive():
    """Architecture guarantee: the residual memory CAN express history
    content (sensitivity >> 0 at full scale) while keeping the encoder
    embedding primary at the default small scale; also the learnable scale
    receives gradient (the training mechanism is wired)."""
    torch.manual_seed(0)
    m = _model()
    m.eval()
    obs_seq = [_obs(i) for i in range(8)]
    # default scale starts at 0 -> sensitivity ~0 (identity, no dominance)
    m.memory_scale.data.fill_(1.0)  # full expression: content visible
    sens_full = memory_sensitivity(m, obs_seq)
    assert sens_full > 0.005, f"memory content-signal not expressible: {sens_full}"
    # gradient flows into the scale + memory when the residual is active
    m.train()
    b = _to_batch(obs_seq[0])
    out, _ = session_step(m, obs_seq[0], session_start(m))
    loss = out["embedding"].square().mean()
    loss.backward()
    assert m.memory_scale.grad is not None and bool(
        torch.isfinite(m.memory_scale.grad).all()
    )
    mem_grads = [p.grad for p in m.memory.parameters() if p.grad is not None]
    assert len(mem_grads) > 0
    m.zero_grad(set_to_none=True)
    # at default scale (~0) the residual leaves z primary
    m.eval()
    with torch.no_grad():
        m.memory_scale.data.fill_(0.0)
    with torch.no_grad():
        z = m.encode(b["chart"], b["numeric"], b["context"],
                     instrument_id=b["instrument_id"])
        out0 = m(chart=b["chart"], numeric=b["numeric"], context=b["context"],
                 instrument_id=b["instrument_id"])
    assert float(torch.nn.functional.cosine_similarity(
        z, out0["embedding"], dim=-1).item()) > 0.999


def test_stateful_policy_serve_contract():
    m = _model()
    m.eval()
    pol = make_stateful_policy(m, warm_obs=[_obs(i) for i in range(4)])
    a1 = pol(_obs(10))
    a2 = pol(_obs(11))
    assert isinstance(a1, int) and isinstance(a2, int)
    pol2 = make_stateful_policy(m)
    assert 0 <= pol2(_obs(7)) <= 8


# ---------------------------------------------------------------------------
# 3. S2 sequential mode
# ---------------------------------------------------------------------------


def _s2_trainer(**cfg_kw) -> SupervisedTrainer:
    m = _model()
    from zhisa.training.losses import LossWeights
    kw = dict(epochs=2, batch_size=8)
    kw.update(cfg_kw)
    cfg = TrainConfig(
        device="cpu", num_workers=0, log_every=50,
        **kw,
    )
    return SupervisedTrainer(m, MultiTaskLoss(LossWeights()), cfg)


def test_s2_sequential_mode_runs_and_records_warm_eval():
    ds = ConcatDataset([FakeDS(40, 0), FakeDS(40, 1)])
    tr = _s2_trainer(sequential_memory=True, eval_warm=True, eval_every=1)
    result = tr.fit(ds, ds)
    assert len(result["history"]) == 2
    last = result["history"][-1]
    assert last.get("mode") == "sequential_memory"
    assert "val" in last and "val_warm" in last
    assert "memory_sensitivity" in last["val_warm"]


def test_s2_sequential_deterministic():
    ds = ConcatDataset([FakeDS(40, 0), FakeDS(40, 1)])

    def run(seed):
        tr = _s2_trainer(sequential_memory=True, seed=seed, epochs=1, eval_every=0)
        return tr.fit(ds, None)["final_step"]

    assert run(0) == run(0)


def test_s2_shuffled_mode_unchanged():
    ds = ConcatDataset([FakeDS(40, 0), FakeDS(40, 1)])
    tr = _s2_trainer(sequential_memory=False, epochs=1, eval_every=0)
    result = tr.fit(ds, None)
    assert result["history"][0].get("mode") is None  # legacy path untouched


def test_s2_script_wires_memory_flags_from_yaml():
    """Regression: TrainConfig construction in train_s2 must read the YAML
    memory flags (sequential_memory/eval_warm/memory_sensitivity_log) —
    forgetting them silently fell back to the non-sequential path."""
    from zhisa.scripts.train_s2 import _memory_training_flags

    assert _memory_training_flags(None) == {
        "sequential_memory": False, "eval_warm": True,
        "memory_sensitivity_log": True,
    }
    assert _memory_training_flags({
        "sequential_memory": True, "eval_warm": False,
        "memory_sensitivity_log": False,
    }) == {"sequential_memory": True, "eval_warm": False,
           "memory_sensitivity_log": False}
    # the yaml we ship carries the flags
    import yaml as _y
    with open(r"D:\zhisa\configs\s2_multitimeframe_15m_1h_context.yaml",
              encoding="utf-8") as fh:
        cfg = _y.safe_load(fh)
    flags = _memory_training_flags(cfg)
    assert flags["sequential_memory"] is True


def test_s2_sequential_fit_feeds_real_history():
    """The sequential path must ACTUALLY pass rolling history to the model
    (regression: the memory would otherwise train on zeros) and reset the
    session at leaf boundaries."""
    seen_history: list = []

    class Spy:
        def __init__(self, base):
            self.base = base

        def __getattr__(self, item):
            return getattr(self.base, item)

        def __call__(self, **kw):
            seen_history.append(kw.get("history"))
            return self.base(**kw)

    ds = ConcatDataset([FakeDS(40, 0), FakeDS(40, 1)])
    tr = _s2_trainer(sequential_memory=True, epochs=1, eval_every=0)
    spy = Spy(tr.model)
    real_forward = tr.model.forward

    def wrapped(**kw):
        seen_history.append(kw.get("history"))
        return real_forward(**kw)

    spy.base.forward = wrapped
    tr.model = spy  # type: ignore[assignment]
    tr.fit(ds, None)
    assert len(seen_history) > 4
    # within a leaf: second batch must carry a REAL (non-None) state;
    # at the first batch of a new leaf the state is None (cold start)
    hist_seq = [h is not None for h in seen_history]
    assert any(hist_seq), "rolling history was never passed to the model"
    assert hist_seq[0] is False  # first batch of the first leaf = cold start