"""End-to-end pipeline test: data -> dataset -> model -> training -> backtest."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from zhisa.backtest.engine import run_backtest
from zhisa.data.dataset import MarketDataset, SampleSpec
from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.env.trading_env import EnvConfig
from zhisa.models.policy import build_default_policy
from zhisa.training.losses import LossWeights, MultiTaskLoss
from zhisa.training.optim import OptimConfig
from zhisa.training.s2_supervised import SupervisedTrainer, TrainConfig


@pytest.mark.slow
def test_end_to_end_pipeline(tmp_path: Path, device):
    """Run a tiny but complete training -> backtest loop."""
    # 1) Data
    df = generate_market(MarketConfig(n_bars=800, seed=42))
    spec = SampleSpec(chart_window=16, feature_window=16, image_size=16)
    ds = MarketDataset(df, spec=spec, cache_charts=True)
    n_feat = ds._features.shape[1] + ds._time_features.shape[1]
    n_ctx = ds._time_features.shape[1]

    # 2) Model
    model = build_default_policy(
        in_numeric_features=n_feat,
        in_context_features=n_ctx,
        window=spec.chart_window,
        image_size=spec.image_size,
        n_actions=9,
        n_regime_classes=spec.n_regime_states,
    ).to(device)

    # 3) Training (just 1 epoch for speed)
    loss_fn = MultiTaskLoss(LossWeights()).to(device)
    trainer = SupervisedTrainer(
        model, loss_fn,
        TrainConfig(epochs=1, batch_size=64, device=device,
                    optim=OptimConfig(lr=1e-3, scheduler="none", warmup_steps=0, t_max=10),
                    log_every=10, checkpoint=str(tmp_path / "model.pt")),
    )
    history = trainer.fit(ds)
    assert history["history"][-1]["loss"] is not None

    # 4) Backtest
    rng = np.random.default_rng(0)
    policy = lambda _obs: int(rng.integers(0, 9))  # random smoke policy
    cfg = EnvConfig(window=16, image_size=16, seed=0)
    result = run_backtest(df, policy, cfg=cfg)
    assert result.metrics.n_periods > 0
