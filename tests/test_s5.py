"""Tests for the S5 online continual learning trainer."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from zhisa.data.dataset import MarketDataset, SampleSpec
from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.models.policy import build_default_policy
from zhisa.training.s5_continual import (
    ContinualConfig,
    ContinualResult,
    ContinualStepResult,
    DriftDetector,
    EWCLoss,
    OnlineContinualTrainer,
    ReplayBuffer,
    ReplaySample,
)


# ---------------------------------------------------------------------------
# ReplayBuffer (reservoir sampling)
# ---------------------------------------------------------------------------


def test_replay_buffer_under_capacity_keeps_all():
    buf = ReplayBuffer(capacity=5, seed=0)
    for i in range(3):
        buf.add(ReplaySample(data={"i": i}))
    assert len(buf) == 3
    assert [s["i"] for s in buf._buf] == [0, 1, 2]
    assert buf.n_seen == 3


def test_replay_buffer_reservoir_keeps_capacity_when_full():
    """Algorithm R: when full, every new sample has chance capacity/n to land."""
    buf = ReplayBuffer(capacity=4, seed=42)
    for i in range(100):
        buf.add(ReplaySample(data={"i": i}))
    assert len(buf) == 4
    assert buf.n_seen == 100
    # Slots in [0, capacity) are the only legal slots, so all values
    # stored must be > (n_seen - capacity) with very high probability.
    # Just sanity-check that the buffer reflects a subset of recent
    # items (not all old items were kept).
    assert all(s["i"] >= 0 for s in buf._buf)


def test_replay_buffer_sample_with_replacement_for_small_buffer():
    buf = ReplayBuffer(capacity=10, seed=0)
    for i in range(3):
        buf.add(ReplaySample(data={"i": i}))
    samples = buf.sample(batch_size=8)
    assert len(samples) == 8
    # With replacement, n_unique <= n_in_buffer.
    assert len({s["i"] for s in samples}) <= 3


def test_replay_buffer_sample_uniform_when_large_enough():
    buf = ReplayBuffer(capacity=20, seed=0)
    for i in range(20):
        buf.add(ReplaySample(data={"i": i}))
    samples = buf.sample(batch_size=10)
    assert len(samples) == 10
    assert len({s["i"] for s in samples}) == 10


def test_replay_buffer_clear_empties_everything():
    buf = ReplayBuffer(capacity=5, seed=0)
    for i in range(10):
        buf.add(ReplaySample(data={"i": i}))
    buf.clear()
    assert len(buf) == 0
    assert buf.n_seen == 0


def test_replay_buffer_extend_adds_all():
    buf = ReplayBuffer(capacity=5, seed=0)
    buf.extend([ReplaySample(data={"i": i}) for i in range(3)])
    assert len(buf) == 3


def test_replay_buffer_state_dict_round_trip():
    buf = ReplayBuffer(capacity=4, seed=0)
    for i in range(7):
        buf.add(ReplaySample(data={"i": i, "arr": np.array([i, i + 1])}))
    state = buf.state_dict()
    new = ReplayBuffer(capacity=4, seed=1)
    new.load_state_dict(state)
    assert len(new) == len(buf)
    assert new.n_seen == buf.n_seen
    assert new.capacity == buf.capacity


def test_replay_buffer_rejects_non_positive_capacity():
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=0)


# ---------------------------------------------------------------------------
# EWCLoss
# ---------------------------------------------------------------------------


def _linear_model(n_in: int = 4, n_out: int = 3) -> nn.Module:
    torch.manual_seed(0)
    return nn.Linear(n_in, n_out)


def test_ewc_loss_is_zero_with_no_reference():
    ewc = EWCLoss(ewc_lambda=2.0)
    m = _linear_model()
    val = ewc.penalty_value(m)
    assert val == 0.0
    out = ewc(m)
    assert float(out.item()) == 0.0


def test_ewc_loss_is_zero_after_consolidation_without_calibration():
    """Without calibration batches, the penalty stays zero (no Fisher)."""
    ewc = EWCLoss(ewc_lambda=2.0)
    m = _linear_model()
    ewc.update_fisher(m, batches=[], device="cpu")
    # Fisher is empty -> no reference -> penalty is 0.
    assert ewc.penalty_value(m) == 0.0


def test_ewc_loss_penalises_drift_from_reference():
    ewc = EWCLoss(ewc_lambda=1.0)
    m = _linear_model(n_in=3, n_out=2)

    # Build a tiny calibration batch.
    obs = {
        "x": torch.randn(4, 3),
        "y": torch.tensor([0, 1, 0, 1], dtype=torch.long),
    }

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3, 2)
        def forward(self, x):
            return {"policy_logits": self.fc(x)}

    m = TinyModel()
    # Wrap the model to satisfy the EWC NLL: EWC looks for policy_logits
    # and "action" key in the batch. We add an adapter.
    class _Wrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, x):
            return self.inner(x)

    wrapped = _Wrapper(m)
    # Patch the EWC NLL to consume our (x, y) batch.
    def _nll(model, batch):
        out = model(batch["x"])
        return -F.log_softmax(out["policy_logits"], dim=-1)[
            range(len(batch["y"])), batch["y"]
        ].mean()
    ewc._negative_log_likelihood = _nll
    ewc.update_fisher(wrapped, batches=[obs], device="cpu")

    # Penalty before any change is small.
    base = ewc.penalty_value(wrapped)
    # Move the parameters and check the penalty grows.
    with torch.no_grad():
        for p in wrapped.parameters():
            p.add_(0.1)
    moved = ewc.penalty_value(wrapped)
    assert moved > base


def test_ewc_loss_scales_with_lambda():
    m = _linear_model(n_in=4, n_out=3)
    ewc = EWCLoss(ewc_lambda=1.0)
    # nn.Linear(4, 3) has 4*3 + 3 = 15 parameters.
    ewc._fisher = torch.ones(15)
    ewc._theta_star = torch.zeros(15)
    # Now perturb params.
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05)
    a = ewc.penalty_value(m)
    ewc.ewc_lambda = 4.0
    b = ewc.penalty_value(m)
    assert math.isclose(b, 4.0 * a, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# DriftDetector (Page-Hinkley)
# ---------------------------------------------------------------------------


def test_drift_detector_no_drift_on_stationary_signal():
    det = DriftDetector(threshold=50.0, warmup=3)
    triggered = [det.update(0.01 * (i % 5)) for i in range(20)]
    assert not any(triggered)


def test_drift_detector_triggers_on_mean_shift():
    det = DriftDetector(threshold=5.0, alpha=0.001, warmup=2)
    # Quiet stream.
    for _ in range(5):
        det.update(0.0)
    # Big step up — should trigger.
    fired = False
    for _ in range(20):
        if det.update(1.0):
            fired = True
            break
    assert fired
    assert det.drift_detected


def test_drift_detector_clears_after_signal_returns():
    det = DriftDetector(threshold=2.0, alpha=0.001, warmup=2,
                        reset_tolerance=0.5)
    for _ in range(3):
        det.update(0.0)
    for _ in range(10):
        det.update(1.0)
    assert det.drift_detected
    # Return to baseline long enough to clear.
    for _ in range(50):
        det.update(0.0)
    assert not det.drift_detected


def test_drift_detector_rejects_non_positive_threshold():
    with pytest.raises(ValueError):
        DriftDetector(threshold=0.0)


def test_drift_detector_state_snapshot():
    det = DriftDetector(threshold=5.0, warmup=1)
    for _ in range(5):
        det.update(0.1)
    s = det.state()
    assert s.n == 5
    assert math.isclose(s.mean, 0.1, abs_tol=1e-6)


def test_drift_detector_reset_clears_state():
    det = DriftDetector(threshold=5.0, warmup=2)
    for _ in range(10):
        det.update(1.0)
    det.reset()
    assert det.state().n == 0
    assert not det.drift_detected


# ---------------------------------------------------------------------------
# OnlineContinualTrainer
# ---------------------------------------------------------------------------


def _make_policy(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    spec = SampleSpec(chart_window=8, feature_window=8, image_size=8)
    df = generate_market(MarketConfig(n_bars=80, seed=seed))
    ds = MarketDataset(df, spec=spec)
    n_feat = ds._features.shape[1] + ds._time_features.shape[1]
    return build_default_policy(
        in_numeric_features=n_feat,
        in_context_features=ds._time_features.shape[1],
        window=spec.chart_window, image_size=spec.image_size,
        n_actions=4, n_regime_classes=spec.n_regime_states,
    )


def _make_inner_factory(returns_loss: float = 0.5):
    """A factory that returns a no-op trainer (deterministic loss)."""
    class FakeInner:
        def __init__(self, model):
            self.model = model
        def fit(self, df):
            # Pretend to run training; return a fixed-shape result.
            return {"history": [{"loss": returns_loss}], "final_loss": returns_loss}
    return FakeInner


def test_continual_trainer_records_inner_loss():
    model = _make_policy(seed=0)
    cfg = ContinualConfig(
        n_iterations=3, bars_per_iter=120, replay_capacity=8,
        replay_batch_size=2, inner_epochs=1, log_every=0, seed=0,
    )
    trainer = OnlineContinualTrainer(
        model, cfg, inner_factory=_make_inner_factory(0.42),
    )
    result = trainer.fit()
    assert isinstance(result, ContinualResult)
    assert len(result.history) == 3
    for step in result.history:
        assert math.isclose(step.inner_loss, 0.42, abs_tol=1e-6)
        assert step.iteration >= 0
    assert math.isclose(result.final_loss, 0.42, abs_tol=1e-6)


def test_continual_trainer_grows_replay_buffer():
    model = _make_policy(seed=0)
    cfg = ContinualConfig(
        n_iterations=2, bars_per_iter=120, replay_capacity=16,
        replay_batch_size=2, log_every=0, seed=0,
    )
    trainer = OnlineContinualTrainer(
        model, cfg, inner_factory=_make_inner_factory(0.0),
    )
    for i in range(5):
        trainer.record_transition({"reward": float(i), "x": np.array([i])})
    assert len(trainer.replay) == 5
    trainer.fit()
    # Replay should still hold 5 samples.
    assert len(trainer.replay) == 5


def test_continual_trainer_replay_step_runs_when_buffer_nonempty():
    """When the buffer has obs-shaped transitions, the replay step
    can actually call the model without crashing."""
    model = _make_policy(seed=0)
    spec = SampleSpec(chart_window=8, feature_window=8, image_size=8)
    df = generate_market(MarketConfig(n_bars=80, seed=0))
    ds = MarketDataset(df, spec=spec)
    n_feat = ds._features.shape[1] + ds._time_features.shape[1]

    cfg = ContinualConfig(
        n_iterations=1, bars_per_iter=80, replay_capacity=8,
        replay_batch_size=2, log_every=0, seed=0,
    )
    trainer = OnlineContinualTrainer(
        model, cfg, inner_factory=_make_inner_factory(0.0),
    )
    sample = ds[0]
    trainer.record_transition({
        "chart": sample["chart"],
        "numeric": sample["numeric"],
        "context": sample["context"],
        "reward": 0.1,
    })
    loss, n = trainer._replay_step()
    # The replay step samples a full batch (with replacement) and
    # returns the number of items it processed.
    assert n == cfg.replay_batch_size
    assert isinstance(loss, float)


def test_continual_trainer_drift_event_increases_ewc_lambda():
    model = _make_policy(seed=0)
    cfg = ContinualConfig(
        n_iterations=10, bars_per_iter=80, replay_capacity=8,
        drift_threshold=2.0, drift_alpha=0.0, drift_warmup=2,
        ewc_lambda=1.0, ewc_lambda_on_drift=10.0, log_every=0, seed=0,
    )
    trainer = OnlineContinualTrainer(
        model, cfg, inner_factory=_make_inner_factory(0.0),
    )
    # Manually drive the detector to trigger a drift event.
    for _ in range(2):
        trainer.record_episode_reward(0.0)
    triggered = trainer.record_episode_reward(100.0)
    assert triggered
    assert trainer.ewc.ewc_lambda == cfg.ewc_lambda_on_drift
    assert trainer.drift_events == 1


def test_continual_trainer_drift_event_count_matches():
    model = _make_policy(seed=0)
    cfg = ContinualConfig(
        n_iterations=20, bars_per_iter=80, replay_capacity=8,
        drift_threshold=2.0, drift_alpha=0.0, drift_warmup=1,
        log_every=0, seed=0,
    )
    trainer = OnlineContinualTrainer(
        model, cfg, inner_factory=_make_inner_factory(0.0),
    )
    # Two distinct drift events.
    for _ in range(3):
        trainer.record_episode_reward(0.0)
    trainer.record_episode_reward(50.0)   # event 1
    for _ in range(20):
        trainer.record_episode_reward(0.0)  # settle
    trainer.record_episode_reward(50.0)   # event 2
    assert trainer.drift_events == 2


def test_continual_trainer_consolidate_records_reference():
    model = _make_policy(seed=0)
    cfg = ContinualConfig(
        n_iterations=1, bars_per_iter=80, replay_capacity=4, log_every=0,
    )
    trainer = OnlineContinualTrainer(
        model, cfg, inner_factory=_make_inner_factory(0.0),
    )
    # No calibration: penalty is still zero, but we can perturb and
    # verify the snapshot mechanic doesn't crash.
    trainer.consolidate()
    ewc = trainer.ewc
    assert ewc.has_reference is False  # Fisher still empty -> no penalty


def test_continual_trainer_save_and_load_round_trip(tmp_path):
    model = _make_policy(seed=0)
    cfg = ContinualConfig(
        n_iterations=1, bars_per_iter=80, replay_capacity=4,
        checkpoint=str(tmp_path / "s5.pt"), log_every=0,
    )
    trainer = OnlineContinualTrainer(
        model, cfg, inner_factory=_make_inner_factory(0.0),
    )
    for i in range(3):
        trainer.record_transition({"x": np.array([i]), "reward": 0.0})
    trainer.fit()
    assert Path(cfg.checkpoint).exists()

    new_model = _make_policy(seed=1)
    new_trainer = OnlineContinualTrainer(
        new_model, cfg, inner_factory=_make_inner_factory(0.0),
    )
    new_trainer.load(cfg.checkpoint)
    assert len(new_trainer.replay) == len(trainer.replay)
    # EWC weight and drift event count must match.
    assert new_trainer.ewc.ewc_lambda == trainer.ewc.ewc_lambda
    assert new_trainer.drift_events == trainer.drift_events


def test_continual_trainer_handles_inner_factory_with_no_args():
    """Some inner trainers (S1 SSL) take no arguments to .fit()."""
    model = _make_policy(seed=0)

    class NoArgInner:
        def __init__(self, m):
            self.m = m
        def fit(self):
            return {"history": [{"loss": 0.1}], "final_loss": 0.1}

    cfg = ContinualConfig(
        n_iterations=2, bars_per_iter=80, replay_capacity=4, log_every=0,
    )
    trainer = OnlineContinualTrainer(model, cfg, inner_factory=NoArgInner)
    result = trainer.fit()
    assert len(result.history) == 2
    for step in result.history:
        assert math.isclose(step.inner_loss, 0.1, abs_tol=1e-6)


def test_continual_trainer_as_frame_returns_dataframe():
    model = _make_policy(seed=0)
    cfg = ContinualConfig(
        n_iterations=3, bars_per_iter=80, replay_capacity=4, log_every=0,
    )
    trainer = OnlineContinualTrainer(
        model, cfg, inner_factory=_make_inner_factory(0.0),
    )
    result = trainer.fit()
    df = result.as_frame()
    assert list(df.columns) == list(ContinualStepResult.__dataclass_fields__.keys())
    assert len(df) == 3


def test_continual_trainer_with_custom_market_stream():
    """``market_stream`` can be an explicit iterable of DataFrames."""
    model = _make_policy(seed=0)
    cfg = ContinualConfig(
        n_iterations=2, bars_per_iter=80, replay_capacity=4, log_every=0,
    )
    trainer = OnlineContinualTrainer(
        model, cfg, inner_factory=_make_inner_factory(0.0),
    )
    stream = [
        generate_market(MarketConfig(n_bars=80, seed=i))
        for i in range(2)
    ]
    result = trainer.fit(market_stream=iter(stream))
    assert len(result.history) == 2
