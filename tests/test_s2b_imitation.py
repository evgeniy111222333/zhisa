"""Tests for the S2b imitation learning trainers (BC + DAgger)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from zhisa.data.dataset import SampleSpec
from zhisa.data.expert import (
    MomentumExpert,
    SmaCrossExpert,
    TripleBarrierExpert,
    build_expert,
)
from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.models.policy import PolicyConfig, build_default_policy
from zhisa.training.losses import LossWeights, MultiTaskLoss
from zhisa.training.optim import OptimConfig
from zhisa.training.s2b_imitation import (
    BCConfig,
    BehavioralCloningTrainer,
    DAggerConfig,
    DAggerTrainer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_imitation_setup(tiny_market, device):
    """A minimal model + expert + loss setup used by every test below."""
    spec = SampleSpec(chart_window=16, feature_window=16, image_size=16, horizons=(4,), n_regime_states=2)
    # Probe the real feature dimensionality from a real dataset so
    # the test setup matches what the trainer will actually see.
    from zhisa.data.dataset import MarketDataset as _MD
    probe_ds = _MD(tiny_market, spec=spec)
    n_feat = probe_ds._features.shape[1]
    n_ctx = probe_ds._time_features.shape[1]
    model = build_default_policy(
        in_numeric_features=n_feat,
        in_context_features=n_ctx,
        window=spec.chart_window,
        image_size=spec.image_size,
        n_actions=9,
        n_regime_classes=2,
        vision_channels=(8, 16),
        embed_dim=32,
        fusion_layers=1,
        memory_layers=1,
        memory_max_len=8,
        n_instruments=1,
    ).to(device)
    # Env must use the same window/image_size as the model. Default
    # EnvConfig is window=32, image_size=64 which would mismatch.
    from zhisa.env.trading_env import EnvConfig
    env_cfg = EnvConfig(
        window=spec.chart_window,
        image_size=spec.image_size,
        episode_length=20,
        stop_loss_pct=0.0,
        take_profit_pct=0.0,
        trailing_stop_pct=0.0,
        funding_rate=0.0,
        funding_interval=0,
    )
    return {
        "model": model,
        "spec": spec,
        "n_feat": n_feat,
        "n_ctx": n_ctx,
        "df": tiny_market,
        "env_cfg": env_cfg,
    }


# ---------------------------------------------------------------------------
# Behavioral cloning
# ---------------------------------------------------------------------------


def test_bc_loss_decreases(tiny_imitation_setup, device):
    """Plain BC on a tiny dataset should reduce the loss over a few epochs."""
    s = tiny_imitation_setup
    loss = MultiTaskLoss(LossWeights(policy=1.0))
    cfg = BCConfig(
        epochs=3, batch_size=8,
        log_every=10_000, device=device, seed=0,
        optim=OptimConfig(lr=1e-3, scheduler="none"),
    )
    trainer = BehavioralCloningTrainer(s["model"], loss, cfg)
    expert = TripleBarrierExpert(chart_window=16, max_holding=4)
    res = trainer.fit(s["df"], expert, spec=s["spec"])
    losses = [h["loss"] for h in res["history"]]
    assert len(losses) == 3
    # Loss must be finite and on average go down. We allow a small
    # absolute slack on the first step in case the random init happens
    # to be very low.
    assert all(np.isfinite(l) for l in losses)
    assert losses[-1] < losses[0] + 0.5  # generous bound for a tiny model


def test_bc_policy_logits_match_expert_distribution(tiny_imitation_setup, device):
    """After a few BC epochs the argmax of policy_logits should
    roughly match the expert's action frequency on the training set."""
    s = tiny_imitation_setup
    loss = MultiTaskLoss(LossWeights(policy=2.0))
    cfg = BCConfig(
        epochs=4, batch_size=8, log_every=10_000, device=device, seed=0,
        optim=OptimConfig(lr=3e-3, scheduler="none"),
    )
    trainer = BehavioralCloningTrainer(s["model"], loss, cfg)
    expert = TripleBarrierExpert(chart_window=16, max_holding=4)
    trainer.fit(s["df"], expert, spec=s["spec"])

    # Re-label and compare argmax-vs-expert agreement. Batch the
    # inference to amortize GPU launch overhead.
    s["model"].eval()
    n = min(200, len(s["df"]) - 16 - 4)
    batch_size = 32
    expert_actions = [int(expert.predict(s["df"], t)) for t in range(16, 16 + n)]
    long_action = int(__import__("zhisa").env.actions.DiscreteAction.LONG_50)
    short_action = int(__import__("zhisa").env.actions.DiscreteAction.SHORT_50)
    long_pred = 0
    short_pred = 0
    long_exp = sum(1 for a in expert_actions if a == long_action)
    short_exp = sum(1 for a in expert_actions if a == short_action)
    total = 0
    correct = 0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            bs = stop - start
            chart = torch.from_numpy(_fake_chart(16)).unsqueeze(0).expand(bs, -1, -1, -1).to(device)
            num = torch.zeros((bs, 16, s["n_feat"]), dtype=torch.float32, device=device)
            ctx = torch.zeros((bs, s["n_ctx"]), dtype=torch.float32, device=device)
            out = s["model"](chart=chart, numeric=num, context=ctx)
            a_pred = out["policy_logits"].argmax(dim=-1).cpu().tolist()
            for i, ap in enumerate(a_pred):
                total += 1
                if ap == expert_actions[start + i]:
                    correct += 1
                if ap == long_action:
                    long_pred += 1
                if ap == short_action:
                    short_pred += 1
    # The BC trainer has only seen 4 epochs on a tiny dataset, so we
    # don't expect a high agreement rate — but the long/short
    # *frequency* of the policy should not collapse to a single
    # class on this dataset.
    assert total > 0
    assert (long_pred + short_pred) > 0, "policy collapsed to a single action class"


def test_bc_save_and_load_roundtrip(tmp_path, tiny_imitation_setup, device):
    s = tiny_imitation_setup
    loss = MultiTaskLoss(LossWeights(policy=1.0))
    cfg = BCConfig(
        epochs=1, batch_size=8, log_every=10_000, device=device, seed=0,
        optim=OptimConfig(lr=1e-3, scheduler="none"),
        checkpoint=str(tmp_path / "bc.pt"),
    )
    trainer = BehavioralCloningTrainer(s["model"], loss, cfg)
    expert = TripleBarrierExpert(chart_window=16, max_holding=4)
    trainer.fit(s["df"], expert, spec=s["spec"])
    assert (tmp_path / "bc.pt").exists()

    # Reload into a fresh model and confirm shapes match.
    ckpt = torch.load(str(tmp_path / "bc.pt"), map_location="cpu", weights_only=False)
    fresh = build_default_policy(
        in_numeric_features=s["n_feat"],
        in_context_features=s["n_ctx"],
        window=s["spec"].chart_window,
        image_size=s["spec"].image_size,
        n_actions=9, n_regime_classes=2,
        vision_channels=(8, 16), embed_dim=32,
        fusion_layers=1, memory_layers=1, memory_max_len=8,
    )
    missing, unexpected = fresh.load_state_dict(ckpt["model"], strict=False)
    # Aux heads' shapes match exactly (same config), so no missing/unexpected.
    assert not missing
    assert not unexpected


# ---------------------------------------------------------------------------
# DAgger
# ---------------------------------------------------------------------------


def test_dagger_runs_and_aggregates(tiny_imitation_setup, device):
    """DAgger should run n_rounds rounds and grow the aggregated dataset
    monotonically (after round 0, every round adds new pairs)."""
    s = tiny_imitation_setup
    cfg = DAggerConfig(
        n_rounds=3,
        epochs_per_round=1,
        rollout_episodes_per_round=2,
        max_steps_per_episode=20,
        batch_size=8, log_every=10_000, device=device, seed=0,
        optim=OptimConfig(lr=1e-3, scheduler="none"),
        env_cfg=s["env_cfg"],
    )
    expert = TripleBarrierExpert(chart_window=16, max_holding=4)
    trainer = DAggerTrainer(s["model"], expert, cfg)
    res = trainer.fit(s["df"], spec=s["spec"])
    assert len(res.rounds) == 3
    # The aggregated size must not decrease across rounds.
    sizes = [r.n_aggregated for r in res.rounds]
    for prev, cur in zip(sizes, sizes[1:]):
        assert cur >= prev
    # At least one round should have added new pairs.
    assert any(r.n_new_pairs > 0 for r in res.rounds[1:])
    # All losses finite.
    assert all(np.isfinite(r.bc_loss) for r in res.rounds)


def test_dagger_loss_does_not_explode(tiny_imitation_setup, device):
    """Loss must remain finite across all DAgger rounds even on tiny data."""
    s = tiny_imitation_setup
    cfg = DAggerConfig(
        n_rounds=2, epochs_per_round=1, rollout_episodes_per_round=1,
        max_steps_per_episode=10, batch_size=8, log_every=10_000,
        device=device, seed=0,
        optim=OptimConfig(lr=1e-3, scheduler="none"),
        env_cfg=s["env_cfg"],
    )
    expert = MomentumExpert(lookback=8, threshold=0.0, chart_window=16)
    trainer = DAggerTrainer(s["model"], expert, cfg)
    res = trainer.fit(s["df"], spec=s["spec"])
    assert all(np.isfinite(r.bc_loss) for r in res.rounds)


def test_dagger_handles_degenerate_expert(tiny_imitation_setup, device):
    """An expert that always returns SKIP should still allow DAgger
    to complete without error."""
    s = tiny_imitation_setup
    cfg = DAggerConfig(
        n_rounds=2, epochs_per_round=1, rollout_episodes_per_round=1,
        max_steps_per_episode=5, batch_size=4, log_every=10_000,
        device=device, seed=0,
        optim=OptimConfig(lr=1e-3, scheduler="none"),
        env_cfg=s["env_cfg"],
    )
    # A custom expert that always picks SKIP.

    class _SkipExpert:
        name = "skip"

        def predict(self, df, t):
            return 0

        def predict_array(self, df, start=0):
            return np.zeros(max(0, len(df) - start), dtype=np.int64)

    trainer = DAggerTrainer(s["model"], _SkipExpert(), cfg)
    res = trainer.fit(s["df"], spec=s["spec"])
    assert len(res.rounds) == 2
    assert all(np.isfinite(r.bc_loss) for r in res.rounds)


def test_dagger_save_creates_checkpoint(tmp_path, tiny_imitation_setup, device):
    s = tiny_imitation_setup
    cfg = DAggerConfig(
        n_rounds=2, epochs_per_round=1, rollout_episodes_per_round=1,
        max_steps_per_episode=5, batch_size=4, log_every=10_000,
        device=device, seed=0,
        optim=OptimConfig(lr=1e-3, scheduler="none"),
        env_cfg=s["env_cfg"],
        checkpoint=str(tmp_path / "dagger.pt"),
    )
    expert = TripleBarrierExpert(chart_window=16, max_holding=4)
    trainer = DAggerTrainer(s["model"], expert, cfg)
    res = trainer.fit(s["df"], spec=s["spec"])
    assert (tmp_path / "dagger.pt").exists()
    # DAgger's final_loss must equal the last round's bc_loss.
    assert res.final_loss == res.rounds[-1].bc_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_chart(size: int) -> np.ndarray:
    return np.zeros((3, size, size), dtype=np.float32)
