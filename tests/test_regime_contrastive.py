"""Tests for contrastive regime representation learning."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from zhisa.regime import (
    PLAYBOOK_NAMES,
    RegimeAugmentationConfig,
    RegimeEncoder,
    RegimeEncoderConfig,
    RegimeIntelligenceConfig,
    RegimePositiveMaskConfig,
    RegimeSupervisionBatch,
    RegimeSupervisionConfig,
    RegimeSupervisionDataset,
    augment_regime_features,
    nt_xent_loss,
    regime_positive_mask,
    regime_supervision_collate,
    supervised_contrastive_loss,
)
from zhisa.training.optim import OptimConfig
from zhisa.training.regime_contrastive import (
    RegimeContrastiveTrainer,
    RegimeContrastiveTrainConfig,
    RegimeContrastiveWeights,
)


def _ohlcv_from_close(close: np.ndarray, *, volume: float | np.ndarray = 100.0) -> pd.DataFrame:
    close = np.asarray(close, dtype=np.float64)
    open_ = np.r_[close[0], close[:-1]]
    spread = np.maximum(np.abs(close - open_) * 0.2, close * 0.001)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    if np.isscalar(volume):
        vol = np.full(close.size, float(volume))
    else:
        vol = np.asarray(volume, dtype=np.float64)
    idx = pd.date_range("2026-01-01", periods=close.size, freq="5min", tz="UTC")
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
    }, index=idx)


def _mixed_df(n: int = 240) -> pd.DataFrame:
    up = np.linspace(100.0, 122.0, n // 3)
    range_ = 122.0 + 1.1 * np.sin(np.arange(n // 3) / 2.5)
    down = np.linspace(121.0, 96.0, n - len(up) - len(range_))
    close = np.r_[up, range_, down] + 0.18 * np.sin(np.arange(n) / 4)
    volume = np.r_[np.full(len(up), 120.0), np.full(len(range_), 90.0), np.full(len(down), 360.0)]
    return _ohlcv_from_close(close, volume=volume)


def _cfg() -> RegimeSupervisionConfig:
    return RegimeSupervisionConfig(
        horizon=6,
        stride=6,
        min_history=72,
        symbol="BTC/USDT",
        analyzer=RegimeIntelligenceConfig(timeframes=("5m", "15m")),
    )


def _batch_for_mask() -> RegimeSupervisionBatch:
    x = torch.randn(4, 8)
    return RegimeSupervisionBatch(
        x=x,
        macro=torch.tensor([0, 0, 0, 1]),
        meso=torch.tensor([1, 2, 1, 1]),
        risk_mode=torch.tensor([2, 2, 3, 2]),
        tradeability=torch.ones(4),
        transition_risk=torch.zeros(4),
        forward_return=torch.tensor([0.010, 0.014, -0.050, 0.011]),
        realized_vol=torch.tensor([0.010, 0.012, 0.040, 0.010]),
        max_drawdown=torch.tensor([-0.010, -0.012, -0.080, -0.011]),
        playbook_label=torch.tensor([1, 1, 2, 1]),
        playbook_scores=torch.zeros(4, len(PLAYBOOK_NAMES)),
        reports=[],
        outcomes=[],
        meta=[],
    )


def test_augment_regime_features_preserves_shape_and_category_defaults() -> None:
    x = torch.ones(3, 5)
    feature_names = (
        "scalar.confidence",
        "aggregate.trend_score",
        "context.state_space.entropy",
        "macro.bull_trend",
        "risk_mode.normal",
    )
    out = augment_regime_features(
        x,
        RegimeAugmentationConfig(feature_dropout=0.3, continuous_noise=0.05, categorical_dropout=0.0),
        feature_names=feature_names,
        generator=torch.Generator().manual_seed(7),
    )

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert not torch.allclose(out[:, :3], x[:, :3])
    torch.testing.assert_close(out[:, 3:], x[:, 3:])


def test_nt_xent_prefers_aligned_views() -> None:
    z = torch.eye(5)
    aligned = nt_xent_loss(z, z + 0.01 * torch.randn_like(z), temperature=0.08)
    shuffled = nt_xent_loss(z, torch.roll(z, shifts=1, dims=0), temperature=0.08)

    assert aligned.item() < shuffled.item()
    assert torch.isfinite(aligned)


def test_regime_positive_mask_is_outcome_aware() -> None:
    batch = _batch_for_mask()
    mask = regime_positive_mask(
        batch,
        RegimePositiveMaskConfig(
            same_macro=True,
            same_playbook=True,
            return_tolerance=0.01,
            drawdown_tolerance=0.01,
            vol_tolerance=0.01,
        ),
    )

    assert mask[0, 1]
    assert mask[1, 0]
    assert not mask[0, 2]
    assert not mask[0, 3]
    assert not mask.diag().any()


def test_supervised_contrastive_loss_backpropagates() -> None:
    z = torch.randn(4, 6, requires_grad=True)
    mask = torch.tensor(
        [
            [False, True, False, False],
            [True, False, False, False],
            [False, False, False, True],
            [False, False, True, False],
        ]
    )

    loss = supervised_contrastive_loss(z, mask, temperature=0.12)
    loss.backward()

    assert loss.item() > 0.0
    assert z.grad is not None
    assert z.grad.abs().sum() > 0.0


def test_regime_contrastive_trainer_runs_and_saves_checkpoint(tmp_path) -> None:
    ds = RegimeSupervisionDataset(_mixed_df(260), _cfg())
    batch = regime_supervision_collate([ds[0], ds[1], ds[2], ds[3]])
    model = RegimeEncoder(RegimeEncoderConfig(embed_dim=10, hidden_dim=32, dropout=0.0))
    ckpt = tmp_path / "regime_contrastive.pt"
    trainer = RegimeContrastiveTrainer(
        model,
        RegimeContrastiveTrainConfig(
            epochs=1,
            batch_size=8,
            device="cpu",
            checkpoint=str(ckpt),
            temperature=0.15,
            augmentation=RegimeAugmentationConfig(feature_dropout=0.05, continuous_noise=0.01),
            weights=RegimeContrastiveWeights(view_consistency=1.0, outcome_supervised=0.25, multitask=0.25),
            optim=OptimConfig(lr=2e-3, scheduler="none", warmup_steps=0),
        ),
    )

    result = trainer.fit(ds, val_ds=ds)
    metrics = trainer.evaluate(ds)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    out = model(batch.x)

    assert result["final_step"] > 0
    assert len(result["history"]) == 1
    assert metrics["total"] > 0.0
    assert metrics["view_consistency"] >= 0.0
    assert ckpt.exists()
    assert "model" in payload
    assert out["embedding"].shape == (4, 10)
