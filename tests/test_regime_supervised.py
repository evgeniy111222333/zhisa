"""Tests for supervised regime-intelligence datasets and training."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from zhisa.regime import (
    PLAYBOOK_NAMES,
    RegimeEncoder,
    RegimeEncoderConfig,
    RegimeFeatureVectorizer,
    RegimeIntelligence,
    RegimeIntelligenceConfig,
    RegimeSupervisionConfig,
    RegimeSupervisionDataset,
    regime_supervision_collate,
)
from zhisa.training.optim import OptimConfig
from zhisa.training.regime_supervised import (
    RegimeEncoderLoss,
    RegimeEncoderTrainer,
    RegimeTrainConfig,
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


def _mixed_df(n: int = 220) -> pd.DataFrame:
    up = np.linspace(100.0, 125.0, n // 2)
    down = np.linspace(125.0, 95.0, n - len(up))
    noise = 0.35 * np.sin(np.arange(n) / 3)
    close = np.r_[up, down] + noise
    volume = np.r_[np.full(len(up), 100.0), np.full(len(down), 350.0)]
    return _ohlcv_from_close(close, volume=volume)


def _cfg() -> RegimeSupervisionConfig:
    return RegimeSupervisionConfig(
        horizon=6,
        stride=8,
        min_history=72,
        symbol="BTC/USDT",
        analyzer=RegimeIntelligenceConfig(timeframes=("5m", "15m")),
    )


def test_regime_supervision_dataset_is_causal_and_complete() -> None:
    df = _mixed_df()
    ds = RegimeSupervisionDataset(df, _cfg())
    item = ds[0]
    t = item.meta["t"]
    analyzer = RegimeIntelligence(_cfg().analyzer)
    report = analyzer.analyze(df.iloc[: t + 1], symbol="BTC/USDT")
    expected_x = RegimeFeatureVectorizer().transform(report)

    assert len(ds) > 0
    assert item.x.shape == (ds.vectorizer.dim,)
    torch.testing.assert_close(item.x, torch.from_numpy(expected_x).float())
    assert item.meta["timestamp"] == report.features["timestamp"]
    assert torch.isfinite(item.forward_return)
    assert item.realized_vol.item() >= 0.0
    assert item.max_drawdown.item() <= 0.0
    assert 0 <= item.playbook_label.item() < len(PLAYBOOK_NAMES)
    assert item.playbook_scores.shape == (len(PLAYBOOK_NAMES),)
    assert item.outcome.label == item.meta["best_playbook"]


def test_regime_supervision_collate_batches_items() -> None:
    ds = RegimeSupervisionDataset(_mixed_df(), _cfg())
    batch = regime_supervision_collate([ds[0], ds[1], ds[2]])

    assert batch.x.shape == (3, ds.vectorizer.dim)
    assert batch.macro.shape == (3,)
    assert batch.meso.shape == (3,)
    assert batch.risk_mode.shape == (3,)
    assert batch.playbook_label.shape == (3,)
    assert batch.playbook_scores.shape == (3, len(PLAYBOOK_NAMES))
    assert len(batch.reports) == 3
    assert len(batch.outcomes) == 3


def test_regime_encoder_loss_backpropagates() -> None:
    ds = RegimeSupervisionDataset(_mixed_df(), _cfg())
    batch = regime_supervision_collate([ds[0], ds[1], ds[2], ds[3]])
    model = RegimeEncoder(RegimeEncoderConfig(embed_dim=12, hidden_dim=24, dropout=0.0))
    loss_fn = RegimeEncoderLoss()

    out = model(batch.x)
    losses = loss_fn(out, batch)
    losses["total"].backward()

    assert losses["total"].item() > 0.0
    assert torch.isfinite(losses["total"])
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_regime_encoder_trainer_runs_and_saves_checkpoint(tmp_path) -> None:
    ds = RegimeSupervisionDataset(_mixed_df(260), _cfg())
    model = RegimeEncoder(RegimeEncoderConfig(embed_dim=10, hidden_dim=32, dropout=0.0))
    ckpt = tmp_path / "regime_encoder.pt"
    trainer = RegimeEncoderTrainer(
        model,
        RegimeTrainConfig(
            epochs=2,
            batch_size=8,
            device="cpu",
            checkpoint=str(ckpt),
            optim=OptimConfig(lr=2e-3, scheduler="none", warmup_steps=0),
        ),
    )

    result = trainer.fit(ds, val_ds=ds)
    metrics = trainer.evaluate(ds)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)

    assert len(result["history"]) == 2
    assert result["final_step"] > 0
    assert ckpt.exists()
    assert "model" in payload
    assert "model_config" in payload
    assert metrics["total"] > 0.0
    assert 0.0 <= metrics["macro_acc"] <= 1.0
    assert 0.0 <= metrics["playbook_acc"] <= 1.0


def test_regime_supervision_dataloader_works_with_trainer_batch_contract() -> None:
    ds = RegimeSupervisionDataset(_mixed_df(), _cfg())
    loader = DataLoader(ds, batch_size=5, collate_fn=regime_supervision_collate)
    batch = next(iter(loader))
    model = RegimeEncoder()

    out = model(batch.x)

    assert out["macro_logits"].shape[0] == 5
    assert out["meso_logits"].shape[0] == 5
    assert out["risk_logits"].shape[0] == 5
