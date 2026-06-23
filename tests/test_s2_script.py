"""Regression tests for the production S1 -> S2 training path."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from zhisa.data.dataset import MarketDataset, MarketTargetConfig, SampleSpec
from zhisa.data.labeling import TripleBarrierConfig
from zhisa.models.policy import build_default_policy
from zhisa.scripts.train_s2 import (
    _load_s1_representation,
    _sample_spec,
    _sqrt_inverse_weights,
    _target_config_from,
    main,
)
from zhisa.training.s1_ssl import load_pretrained_into_policy


def _market(n: int, start: str = "2024-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    x = np.arange(n, dtype=np.float64)
    close = 100.0 + 0.02 * x + np.sin(x / 7.0)
    return pd.DataFrame(
        {
            "open": close - 0.05,
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": 1000.0 + 10.0 * np.cos(x / 5.0),
        },
        index=index,
    )


def _policy_for(df: pd.DataFrame, spec: SampleSpec):
    ds = MarketDataset(df, spec=spec, cache_charts=False, chart_cache_size=-1)
    return build_default_policy(
        in_numeric_features=ds._features_df.shape[1],
        in_context_features=ds._time_features_df.shape[1],
        window=spec.chart_window,
        image_size=spec.image_size,
        n_regime_classes=spec.n_regime_states,
    )


def test_s1_transfer_keeps_fresh_s2_heads(tmp_path: Path):
    spec = SampleSpec(chart_window=16, feature_window=16, image_size=16, horizons=(4, 8))
    frame = _market(280)
    torch.manual_seed(1)
    source = _policy_for(frame, spec)
    torch.manual_seed(2)
    target = _policy_for(frame, spec)
    target_head = target.heads.direction.weight.detach().clone()
    target_memory = target.memory.encoder.layers[0].linear1.weight.detach().clone()

    loaded = _load_s1_representation(target, {"model": source.state_dict()})

    assert loaded > 0
    torch.testing.assert_close(target.numeric.patch_proj.weight, source.numeric.patch_proj.weight)
    torch.testing.assert_close(target.heads.direction.weight, target_head)
    torch.testing.assert_close(
        target.memory.encoder.layers[0].linear1.weight,
        target_memory,
    )
    assert not torch.equal(target.heads.direction.weight, source.heads.direction.weight)

    checkpoint = tmp_path / "s1.pt"
    torch.save({"model": source.state_dict()}, checkpoint)
    torch.manual_seed(3)
    strict_target = _policy_for(frame, spec)
    strict_head = strict_target.heads.direction.weight.detach().clone()
    load_pretrained_into_policy(strict_target, str(checkpoint), strict=True)
    torch.testing.assert_close(strict_target.numeric.patch_proj.weight, source.numeric.patch_proj.weight)
    torch.testing.assert_close(strict_target.heads.direction.weight, strict_head)


def test_head_warmup_keeps_untrained_memory_trainable(tmp_path: Path):
    frame = _market(280)
    spec = SampleSpec(chart_window=16, feature_window=16, image_size=16, horizons=(4, 8))
    model = _policy_for(frame, spec)
    from zhisa.training.losses import MultiTaskLoss
    from zhisa.training.s2_supervised import SupervisedTrainer, TrainConfig

    trainer = SupervisedTrainer(model, MultiTaskLoss(), TrainConfig(device="cpu"))
    trainer._set_encoder_trainable(False)
    assert not next(model.vision.parameters()).requires_grad
    assert not next(model.numeric.parameters()).requires_grad
    assert next(model.memory.parameters()).requires_grad
    assert next(model.heads.parameters()).requires_grad


def test_s2_optimizer_uses_discriminative_learning_rates():
    frame = _market(280)
    spec = SampleSpec(chart_window=16, feature_window=16, image_size=16, horizons=(4, 8))
    model = _policy_for(frame, spec)
    from zhisa.training.losses import MultiTaskLoss
    from zhisa.training.optim import OptimConfig
    from zhisa.training.s2_supervised import SupervisedTrainer, TrainConfig

    trainer = SupervisedTrainer(
        model,
        MultiTaskLoss(),
        TrainConfig(
            device="cpu",
            encoder_lr_scale=0.25,
            optim=OptimConfig(lr=1e-4, scheduler="none"),
        ),
    )
    encoder_lrs = {
        group["lr"] for group in trainer.opt.param_groups
        if group["s2_encoder_group"]
    }
    task_lrs = {
        group["lr"] for group in trainer.opt.param_groups
        if not group["s2_encoder_group"]
    }
    assert encoder_lrs == {2.5e-5}
    assert task_lrs == {1e-4}


def test_sample_spec_follows_s1_checkpoint_contract():
    payload = {"model_config": {"window": 128, "image_size": 128}}
    spec = _sample_spec({"chart_window": 32, "image_size": 32}, payload)
    assert spec.chart_window == 128
    assert spec.image_size == 128


def test_s2_target_config_defaults_to_forward_return_and_symmetric_tb():
    target_cfg, tb_cfg = _target_config_from({})

    assert target_cfg == MarketTargetConfig(
        direction_mode="forward_return",
        flat_return_bps=1.0,
        use_log_return=False,
    )
    assert tb_cfg.tp_atr_mult == 2.0
    assert tb_cfg.sl_atr_mult == 2.0


def test_s2_rejects_asymmetric_triple_barrier_direction_without_override():
    cfg = {
        "targets": {
            "direction_mode": "triple_barrier",
            "triple_barrier": {"tp_atr_mult": 2.0, "sl_atr_mult": 1.0},
        }
    }

    import pytest

    with pytest.raises(ValueError, match="requires symmetric TP/SL"):
        _target_config_from(cfg)


def test_s2_class_balance_tolerates_empty_flat_class():
    weights = _sqrt_inverse_weights(np.array([120, 0, 80], dtype=np.int64))

    assert weights.tolist()[1] == 0.0
    assert weights[0] > 0
    assert weights[2] > 0


def test_prepared_s2_cli_writes_validated_checkpoints(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ZHISA_FAST_RENDER", "1")
    root = tmp_path / "prepared"
    (root / "splits").mkdir(parents=True)
    train = _market(300)
    val = _market(240, "2024-06-01")
    train_frame = pd.concat(
        [train.assign(symbol="BTC/USDT"), train.assign(symbol="ETH/USDT")]
    ).sort_index()
    val_frame = pd.concat(
        [val.assign(symbol="BTC/USDT"), val.assign(symbol="ETH/USDT")]
    ).sort_index()
    train_frame.to_parquet(root / "splits" / "train.parquet")
    val_frame.to_parquet(root / "splits" / "val.parquet")
    manifest = {
        "timeframe": "15m",
        "symbols": ["BTC/USDT", "ETH/USDT"],
        "output_checksum": "test-checksum",
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    spec = SampleSpec(
        chart_window=16,
        feature_window=16,
        image_size=16,
        horizons=(4, 8),
        n_regime_states=4,
    )
    s1_model = _policy_for(train, spec)
    s1_checkpoint = tmp_path / "s1.pt"
    torch.save(
        {
            "model": s1_model.state_dict(),
            "model_config": asdict(s1_model.cfg),
            "checkpoint_meta": {"stage": "s1_ssl"},
        },
        s1_checkpoint,
    )
    config = tmp_path / "s2.yaml"
    config.write_text(
        "\n".join(
            [
                "seed: 0",
                "device: cpu",
                "epochs: 1",
                "batch_size: 64",
                "workers: 0",
                "chart_window: 16",
                "image_size: 16",
                "horizons: [4, 8]",
                "class_balance: none",
                "eval_every: 1",
                "val_max_batches: 1",
                "champion_metric: s2_composite_score",
                "champion_mode: max",
                "targets:",
                "  direction_mode: forward_return",
                "  flat_return_bps: 1.0",
                "  use_log_return: false",
                "optim:",
                "  lr: 0.0003",
                "  scheduler: none",
                "  warmup_steps: 0",
            ]
        ),
        encoding="utf-8",
    )
    last = tmp_path / "s2_last.pt"
    best = tmp_path / "s2_best.pt"

    rc = main(
        [
            "--config", str(config),
            "--prepared-root", str(root),
            "--s1-checkpoint", str(s1_checkpoint),
            "--checkpoint", str(last),
            "--best-checkpoint", str(best),
            "--fast-render",
        ]
    )

    assert rc == 0
    assert last.exists() and best.exists()
    payload = torch.load(last, map_location="cpu", weights_only=False)
    assert payload["trainer_state"]["completed_epochs"] == 1
    val_metrics = payload["trainer_state"]["history"][0]["val"]
    assert val_metrics["n_samples"] > 0
    assert "direction_balanced_accuracy" in val_metrics
    assert "direction_macro_f1" in val_metrics
    assert "return_corr" in val_metrics
    assert "s2_composite_score" in val_metrics
    assert payload["checkpoint_meta"]["dataset"]["manifest_checksum"] == "test-checksum"
    assert payload["checkpoint_meta"]["target_config"]["direction_mode"] == "forward_return"
    assert payload["checkpoint_meta"]["champion_metric"] == "s2_composite_score"
    assert payload["checkpoint_meta"]["trading_policy_ready"] is False


def test_market_dataset_can_still_use_explicit_symmetric_triple_barrier():
    frame = _market(160)
    spec = SampleSpec(chart_window=16, feature_window=16, image_size=16, horizons=(4,))
    ds = MarketDataset(
        frame,
        spec=spec,
        cache_charts=False,
        chart_cache_size=-1,
        target_cfg=MarketTargetConfig(direction_mode="triple_barrier"),
        triple_barrier_cfg=TripleBarrierConfig(tp_atr_mult=2.0, sl_atr_mult=2.0),
    )

    assert ds.target_cfg.direction_mode == "triple_barrier"
    assert ds._tb_cfg_primary.tp_atr_mult == 2.0
    assert ds._tb_cfg_primary.sl_atr_mult == 2.0
