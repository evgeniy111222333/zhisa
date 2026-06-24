"""Regression tests for the production S1 -> S2 training path."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from zhisa.data.dataset import (
    MacroContextConfig,
    MarketDataset,
    MarketTargetConfig,
    SampleSpec,
    multimodal_collate,
)
from zhisa.data.labeling import TripleBarrierConfig
from zhisa.models.policy import PolicyConfig, PolicyNetwork, build_default_policy
from zhisa.scripts.train_s2 import (
    _class_weights,
    _critical_warm_start_missing_keys,
    _direction_sample_weights,
    _load_s1_representation,
    _load_macro_prepared_frames,
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


def test_s2_segment_guard_penalizes_hidden_weak_market():
    from zhisa.training.s2_supervised import TrainConfig, _segment_guard_score

    cfg = TrainConfig(
        guard_min_direction_balanced=0.34,
        guard_min_flat_recall=0.05,
        guard_min_flat_f1=0.08,
        guard_min_volatility_corr=0.45,
        guard_min_return_corr=-0.02,
        guard_min_persistence_lift=0.0,
        guard_max_prediction_share=0.62,
        guard_max_flat_prediction_share=0.36,
        guard_min_flat_pred_target_ratio=0.55,
        guard_max_flat_pred_target_ratio=2.0,
        guard_penalty_scale=0.5,
    )
    guarded, metrics = _segment_guard_score(
        0.25,
        {
            "good": {
                "direction_balanced_accuracy": 0.40,
                "direction_flat_recall": 0.10,
                "direction_flat_f1": 0.09,
                "direction_prediction_share": [0.4, 0.3, 0.3],
                "direction_flat_pred_target_ratio": 1.0,
                "direction_max_prediction_share": 0.4,
                "direction_lift_vs_persistence_balanced": 0.02,
                "volatility_corr": 0.55,
                "return_corr": 0.01,
            },
            "bad": {
                "direction_balanced_accuracy": 0.30,
                "direction_flat_recall": 0.00,
                "direction_flat_f1": 0.00,
                "direction_prediction_share": [0.05, 0.88, 0.07],
                "direction_flat_pred_target_ratio": 4.4,
                "direction_max_prediction_share": 0.88,
                "direction_lift_vs_persistence_balanced": -0.03,
                "volatility_corr": 0.35,
                "return_corr": -0.08,
            },
        },
        cfg,
    )

    assert guarded < 0.25
    assert metrics["s2_worst_segment_flat_recall"] == 0.0
    assert metrics["s2_worst_segment_flat_f1"] == 0.0
    assert metrics["s2_worst_segment_return_corr"] == -0.08
    assert metrics["s2_worst_segment_persistence_lift"] == -0.03
    assert metrics["s2_worst_segment_max_prediction_share"] == 0.88
    assert metrics["s2_worst_segment_flat_prediction_share"] == 0.88
    assert metrics["s2_worst_segment_flat_pred_target_ratio"] == 4.4


def test_s2_early_stopping_trend_defers_plateau_stop():
    from zhisa.training.s2_supervised import _recent_metric_trend_is_improving

    history = [
        {"val": {"s2_guarded_score": 0.210}},
        {"val": {"s2_guarded_score": 0.250}},
        {"val": {"s2_guarded_score": 0.230}},
        {"val": {"s2_guarded_score": 0.235}},
        {"val": {"s2_guarded_score": 0.241}},
    ]

    assert _recent_metric_trend_is_improving(
        history,
        metric="s2_guarded_score",
        mode="max",
        window=3,
        min_delta=0.005,
    )
    assert not _recent_metric_trend_is_improving(
        history,
        metric="s2_guarded_score",
        mode="max",
        window=3,
        min_delta=0.02,
    )


def test_s2_early_stopping_trend_supports_min_metrics():
    from zhisa.training.s2_supervised import _recent_metric_trend_is_improving

    history = [
        {"val": {"total": 1.0}},
        {"val": {"total": 0.9}},
        {"val": {"total": 0.88}},
    ]

    assert _recent_metric_trend_is_improving(
        history,
        metric="total",
        mode="min",
        window=3,
        min_delta=0.05,
    )


def test_market_dataset_exposes_causal_persistence_labels():
    frame = _market(120)
    spec = SampleSpec(chart_window=16, feature_window=16, image_size=16, horizons=(4, 8))
    ds = MarketDataset(frame, spec=spec, cache_charts=False, chart_cache_size=-1)

    sample_t = 20
    primary_idx = sample_t + spec.chart_window - 1
    sample = ds[sample_t]
    expected = np.asarray(
        [
            ds._tb_multi_label_arr[primary_idx - 4, 0],
            ds._tb_multi_label_arr[primary_idx - 8, 1],
        ],
        dtype=np.int64,
    )

    np.testing.assert_array_equal(sample["label_dir_multi_persistence"].numpy(), expected)
    assert int(sample["label_dir_persistence"]) == int(expected[len(spec.horizons) // 2])


def test_market_dataset_persistence_is_neutral_until_horizon_is_known():
    frame = _market(80)
    spec = SampleSpec(chart_window=4, feature_window=4, image_size=16, horizons=(8, 16))
    ds = MarketDataset(frame, spec=spec, cache_charts=False, chart_cache_size=-1)

    sample = ds[0]

    np.testing.assert_array_equal(
        sample["label_dir_multi_persistence"].numpy(),
        np.zeros(len(spec.horizons), dtype=np.int64),
    )
    assert int(sample["label_dir_persistence"]) == 0


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


def test_s2_target_config_parses_horizon_overrides():
    target_cfg, _ = _target_config_from(
        {
            "targets": {
                "direction_mode": "forward_return",
                "flat_return_bps": 8.0,
                "flat_volatility_mult": 0.10,
                "flat_min_bps": 8.0,
                "flat_max_bps": 35.0,
                "horizon_overrides": {
                    "16": {"flat_volatility_mult": 0.12, "flat_max_bps": 70.0},
                    64: {"flat_volatility_mult": 0.08, "flat_max_bps": 140.0},
                },
            }
        }
    )

    assert target_cfg.horizon_overrides == {
        16: {"flat_volatility_mult": 0.12, "flat_max_bps": 70.0},
        64: {"flat_volatility_mult": 0.08, "flat_max_bps": 140.0},
    }


def test_s2_target_config_rejects_bad_horizon_override_field():
    with pytest.raises(ValueError, match="unknown target horizon override fields"):
        _target_config_from(
            {
                "targets": {
                    "horizon_overrides": {
                        "16": {"definitely_not_a_target_field": 1.0},
                    }
                }
            }
        )


def test_market_dataset_applies_horizon_specific_forward_overrides():
    index = pd.date_range("2024-01-01", periods=220, freq="15min", tz="UTC")
    x = np.arange(len(index), dtype=np.float64)
    close = 100.0 + 0.02 * x + 2.5 * np.sin(x / 2.0)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1000.0 + 10.0 * np.cos(x / 3.0),
        },
        index=index,
    )
    spec = SampleSpec(chart_window=16, feature_window=16, image_size=16, horizons=(4, 16, 64))
    base_cfg = MarketTargetConfig(
        direction_mode="forward_return",
        flat_return_bps=8.0,
        flat_volatility_mult=0.10,
        flat_min_bps=8.0,
        flat_max_bps=35.0,
    )
    override_cfg = MarketTargetConfig(
        direction_mode="forward_return",
        flat_return_bps=8.0,
        flat_volatility_mult=0.10,
        flat_min_bps=8.0,
        flat_max_bps=35.0,
        horizon_overrides={
            16: {"flat_volatility_mult": 0.12, "flat_max_bps": 70.0},
            64: {"flat_volatility_mult": 0.08, "flat_max_bps": 140.0},
        },
    )
    base = MarketDataset(frame, spec=spec, cache_charts=False, chart_cache_size=-1, target_cfg=base_cfg)
    tuned = MarketDataset(frame, spec=spec, cache_charts=False, chart_cache_size=-1, target_cfg=override_cfg)

    start = spec.chart_window - 1
    stop = start + len(tuned)
    base_h64_flat = float((base._tb_multi_label_arr[start:stop, 2] == 0).mean())
    tuned_h64_flat = float((tuned._tb_multi_label_arr[start:stop, 2] == 0).mean())

    assert tuned_h64_flat > base_h64_flat
    assert not np.array_equal(
        base._tb_multi_label_arr[start:stop, 1],
        tuned._tb_multi_label_arr[start:stop, 1],
    )


def test_s2_class_balance_tolerates_empty_flat_class():
    weights = _sqrt_inverse_weights(np.array([120, 0, 80], dtype=np.int64))

    assert weights.tolist()[1] == 0.0
    assert weights[0] > 0
    assert weights[2] > 0


def test_s2_class_balance_modes_raise_rare_class_weight():
    counts = np.array([760_000, 136_000, 785_000], dtype=np.int64)

    sqrt_weights = _class_weights(counts, "sqrt_inverse")
    inverse_weights = _class_weights(counts, "inverse")

    assert sqrt_weights[1] > sqrt_weights[0]
    assert sqrt_weights[1] > sqrt_weights[2]
    assert inverse_weights[1] > sqrt_weights[1]


def test_s2_direction_class_balance_caps_flat_weight():
    counts = np.array([760_000, 70_000, 785_000], dtype=np.int64)

    uncapped = _class_weights(counts, "sqrt_inverse")
    capped = _class_weights(
        counts,
        "sqrt_inverse",
        max_weight=1.55,
        flat_max_weight=1.45,
    )

    assert uncapped[1] > 1.45
    assert capped[1] <= 1.45
    assert capped[1] > capped[0]
    assert capped[1] > capped[2]
    assert capped[0] > 0
    assert capped[2] > 0


def test_s2_direction_class_balance_keeps_empty_flat_zero_when_capped():
    weights = _class_weights(
        np.array([120, 0, 80], dtype=np.int64),
        "sqrt_inverse",
        max_weight=1.55,
        flat_max_weight=1.45,
    )

    assert weights.tolist()[1] == 0.0
    assert weights[0] > 0
    assert weights[2] > 0


def test_s2_per_symbol_direction_sampler_raises_rare_flat_exposure():
    spec = SampleSpec(chart_window=16, feature_window=16, image_size=16, horizons=(4, 8))
    first = MarketDataset(
        _market(180),
        spec=spec,
        cache_charts=False,
        chart_cache_size=-1,
        target_cfg=MarketTargetConfig(direction_mode="forward_return", flat_return_bps=1.0),
    )
    second = MarketDataset(
        _market(180, "2024-03-01"),
        spec=spec,
        cache_charts=False,
        chart_cache_size=-1,
        target_cfg=MarketTargetConfig(direction_mode="forward_return", flat_return_bps=1.0),
    )
    first.df.name = "A#segment-1"
    second.df.name = "B#segment-1"
    for ds, flat_count in ((first, 5), (second, 20)):
        start = ds.spec.chart_window - 1
        labels = np.full(len(ds), -1, dtype=np.int64)
        labels[:flat_count] = 0
        labels[flat_count: flat_count * 2] = 1
        ds._tb_label_arr[start : start + len(ds)] = labels

    weights = _direction_sample_weights(
        [first, second],
        mode="per_symbol_direction",
        power=1.0,
        max_weight=100.0,
    )

    assert weights is not None
    first_weights = weights[: len(first)]
    second_weights = weights[len(first) :]
    first_labels = first._tb_label_arr[first.spec.chart_window - 1 : first.spec.chart_window - 1 + len(first)]
    second_labels = second._tb_label_arr[second.spec.chart_window - 1 : second.spec.chart_window - 1 + len(second)]
    assert float(first_weights[first_labels == 0].mean()) > float(first_weights[first_labels == -1].mean())
    assert float(second_weights[second_labels == 0].mean()) > float(second_weights[second_labels == -1].mean())
    torch.testing.assert_close(first_weights.mean(), torch.tensor(1.0), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(second_weights.mean(), torch.tensor(1.0), atol=1e-6, rtol=1e-6)

    capped = _direction_sample_weights(
        [first],
        mode="per_symbol_direction",
        power=1.0,
        max_weight=1.5,
    )
    assert capped is not None
    assert float(capped.max()) <= 1.5


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
                "sample_balance: per_symbol_direction",
                "sample_balance_power: 0.75",
                "sample_balance_max_weight: 8.0",
                "eval_every: 1",
                "val_max_batches: 1",
                "champion_metric: s2_guarded_score",
                "champion_mode: max",
                "segment_validation: true",
                "guard_min_direction_balanced: 0.2",
                "guard_min_flat_recall: 0.0",
                "guard_min_flat_f1: 0.0",
                "guard_min_volatility_corr: -1.0",
                "guard_min_return_corr: -1.0",
                "guard_max_prediction_share: 1.0",
                "guard_max_flat_prediction_share: 1.0",
                "guard_min_flat_pred_target_ratio: 0.55",
                "guard_max_flat_pred_target_ratio: 2.0",
                "guard_penalty_scale: 0.5",
                "targets:",
                "  direction_mode: forward_return",
                "  flat_return_bps: 8.0",
                "  flat_volatility_mult: 0.10",
                "  flat_min_bps: 8.0",
                "  flat_max_bps: 35.0",
                "  use_log_return: false",
                "optim:",
                "  lr: 0.0003",
                "  scheduler: none",
                "  warmup_steps: 0",
                "loss_weights:",
                "  direction: 1.0",
                "  direction_multi: 0.25",
                "  return_pred: 0.8",
                "  return_multi: 0.25",
                "  volatility: 0.5",
                "  regime: 0.3",
                "  risk: 0.25",
                "  policy: 0.0",
                "  value: 0.0",
                "  uncertainty: 0.0",
                "direction_multi_horizon_weights: [1.0, 0.2]",
                "return_multi_horizon_weights: [1.0, 0.2]",
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
    assert "direction_flat_recall" in val_metrics
    assert "direction_flat_precision" in val_metrics
    assert "direction_flat_f1" in val_metrics
    assert "direction_prediction_share" in val_metrics
    assert "direction_max_prediction_share" in val_metrics
    assert "direction_flat_pred_target_ratio" in val_metrics
    assert "direction_persistence_balanced_accuracy" in val_metrics
    assert "direction_lift_vs_persistence_balanced" in val_metrics
    assert "direction_persistence_confusion" in val_metrics
    assert "return_corr" in val_metrics
    assert "value_corr" in val_metrics
    assert "volatility_corr" in val_metrics
    assert "risk_corr" in val_metrics
    assert "s2_composite_score" in val_metrics
    assert "s2_guarded_score" in val_metrics
    assert "segment_metrics" in val_metrics
    assert payload["checkpoint_meta"]["dataset"]["manifest_checksum"] == "test-checksum"
    assert payload["checkpoint_meta"]["target_config"]["direction_mode"] == "forward_return"
    assert payload["checkpoint_meta"]["target_config"]["flat_volatility_mult"] == 0.10
    assert payload["checkpoint_meta"]["target_config"]["horizons"] == [4, 8]
    assert payload["model_config"]["market_horizons"] == [4, 8]
    assert payload["checkpoint_meta"]["champion_metric"] == "s2_guarded_score"
    assert payload["checkpoint_meta"]["trading_policy_ready"] is False

    warm_last = tmp_path / "s2_warm_last.pt"
    warm_best = tmp_path / "s2_warm_best.pt"
    warm_rc = main(
        [
            "--config", str(config),
            "--prepared-root", str(root),
            "--warm-start-checkpoint", str(best),
            "--checkpoint", str(warm_last),
            "--best-checkpoint", str(warm_best),
            "--fast-render",
        ]
    )
    assert warm_rc == 0
    warm_payload = torch.load(warm_last, map_location="cpu", weights_only=False)
    assert warm_payload["trainer_state"]["completed_epochs"] == 1
    assert warm_payload["model_config"]["market_horizons"] == [4, 8]


def test_macro_context_is_causal_and_collates():
    frame = _market(240)
    spec = SampleSpec(chart_window=16, feature_window=16, image_size=16, horizons=(4,))
    macro_cfg = MacroContextConfig(enabled=True, window=6, resample_rule="1h")
    target_cfg = MarketTargetConfig(direction_mode="forward_return", flat_return_bps=1.0)
    ds = MarketDataset(
        frame,
        spec=spec,
        cache_charts=False,
        chart_cache_size=-1,
        target_cfg=target_cfg,
        macro_cfg=macro_cfg,
    )

    sample_idx = 32
    sample = ds[sample_idx]
    assert "macro_numeric" in sample
    assert sample["macro_numeric"].shape == (6, ds._features_df.shape[1])

    batch = multimodal_collate([ds[sample_idx], ds[sample_idx + 1]])
    assert batch.macro_numeric is not None
    assert batch.macro_numeric.shape == (2, 6, ds._features_df.shape[1])

    primary_idx = sample_idx + spec.chart_window - 1
    primary_ts = frame.index[primary_idx]
    mutated = frame.copy()
    mutated.loc[
        mutated.index >= primary_ts.floor("1h"),
        ["open", "high", "low", "close", "volume"],
    ] *= 10.0
    ds_mutated = MarketDataset(
        mutated,
        spec=spec,
        cache_charts=False,
        chart_cache_size=-1,
        target_cfg=target_cfg,
        macro_cfg=macro_cfg,
    )

    torch.testing.assert_close(sample["macro_numeric"], ds_mutated[sample_idx]["macro_numeric"])
    allowed_idx = int(ds._macro_primary_indices[primary_idx])
    assert allowed_idx >= 0
    assert ds._macro_features_df.index[allowed_idx] <= primary_ts.floor("1h") - pd.Timedelta("1h")


def test_prepared_macro_context_uses_external_1h_symbol_data(tmp_path: Path):
    primary = _market(240)
    macro = _market(80, start="2023-12-31 00:00").resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    macro["close"] += 50.0
    macro_root = tmp_path / "macro"
    (macro_root / "symbols").mkdir(parents=True)
    macro.to_parquet(macro_root / "symbols" / "BTC_USDT.parquet")
    macro_manifest = {
        "timeframe": "1h",
        "symbols": ["BTC/USDT"],
        "output_checksum": "macro-checksum",
    }
    (macro_root / "manifest.json").write_text(json.dumps(macro_manifest), encoding="utf-8")

    frames, manifest = _load_macro_prepared_frames(
        macro_root,
        primary_manifest={"symbols": ["BTC/USDT"]},
        expected_timeframe="1h",
    )
    assert manifest["output_checksum"] == "macro-checksum"
    assert set(frames) == {"BTC/USDT"}

    spec = SampleSpec(chart_window=16, feature_window=16, image_size=16, horizons=(4,))
    cfg = MacroContextConfig(enabled=True, source="prepared", window=8, resample_rule="1h")
    target_cfg = MarketTargetConfig(direction_mode="forward_return", flat_return_bps=1.0)
    ds = MarketDataset(
        primary,
        spec=spec,
        cache_charts=False,
        chart_cache_size=-1,
        target_cfg=target_cfg,
        macro_cfg=cfg,
        macro_df=frames["BTC/USDT"],
    )

    sample_idx = 32
    primary_idx = sample_idx + spec.chart_window - 1
    allowed_idx = int(ds._macro_primary_indices[primary_idx])
    assert allowed_idx >= 0
    assert ds._macro_features_df.index[allowed_idx] <= primary.index[primary_idx].floor("1h") - pd.Timedelta("1h")

    mutated_primary = primary.copy()
    mutated_primary[["open", "high", "low", "close", "volume"]] *= 100.0
    ds_mutated_primary = MarketDataset(
        mutated_primary,
        spec=spec,
        cache_charts=False,
        chart_cache_size=-1,
        target_cfg=target_cfg,
        macro_cfg=cfg,
        macro_df=frames["BTC/USDT"],
    )
    torch.testing.assert_close(
        ds[sample_idx]["macro_numeric"],
        ds_mutated_primary[sample_idx]["macro_numeric"],
    )

    mutated_macro = frames["BTC/USDT"].copy()
    allowed_ts = ds._macro_features_df.index[allowed_idx]
    mutated_macro.loc[allowed_ts, "close"] *= 2.0
    ds_mutated_macro = MarketDataset(
        primary,
        spec=spec,
        cache_charts=False,
        chart_cache_size=-1,
        target_cfg=target_cfg,
        macro_cfg=cfg,
        macro_df=mutated_macro,
    )
    assert not torch.allclose(
        ds[sample_idx]["macro_numeric"],
        ds_mutated_macro[sample_idx]["macro_numeric"],
    )


def test_prepared_macro_context_requires_1h_manifest(tmp_path: Path):
    root = tmp_path / "macro_bad"
    (root / "symbols").mkdir(parents=True)
    _market(80, start="2024-01-01").to_parquet(root / "symbols" / "BTC_USDT.parquet")
    (root / "manifest.json").write_text(
        json.dumps({"timeframe": "15m", "symbols": ["BTC/USDT"], "output_checksum": "bad"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="macro prepared timeframe"):
        _load_macro_prepared_frames(
            root,
            primary_manifest={"symbols": ["BTC/USDT"]},
            expected_timeframe="1h",
        )


def test_policy_network_accepts_optional_macro_context():
    cfg = PolicyConfig(
        image_size=16,
        window=16,
        in_numeric_features=8,
        in_macro_features=8,
        macro_window=8,
        in_context_features=4,
        embed_dim=32,
        vision_channels=(8, 16, 24, 32),
        n_regime_classes=3,
        market_horizons=(4, 8),
        use_macro_context=True,
        use_memory=False,
    )
    model = PolicyNetwork(cfg)
    chart = torch.rand(3, 3, 16, 16)
    numeric = torch.rand(3, 16, 8)
    context = torch.rand(3, 4)
    macro = torch.rand(3, 8, 8)

    out = model(chart=chart, numeric=numeric, context=context, macro_numeric=macro)
    assert out["direction"].shape == (3, 3)
    assert out["direction_multi"].shape == (3, 2, 3)
    assert out["embedding"].shape == (3, 32)

    out_zero_macro = model(chart=chart, numeric=numeric, context=context)
    assert out_zero_macro["direction"].shape == (3, 3)


def test_macro_warm_start_only_ignores_new_branch_missing_keys():
    missing = [
        "macro_numeric.patch_proj.weight",
        "timeframe_embed.weight",
        "macro_gate.0.weight",
        "macro_proj.1.bias",
        "macro_norm.weight",
        "heads.policy.weight",
        "numeric.patch_proj.weight",
    ]

    assert _critical_warm_start_missing_keys(missing) == ["numeric.patch_proj.weight"]


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
