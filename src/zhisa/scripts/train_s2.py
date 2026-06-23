"""Train the S2 supervised market heads on labelled market data."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from zhisa.config import load_config
from zhisa.data.dataset import MarketDataset, MarketTargetConfig, SampleSpec
from zhisa.data.labeling import TripleBarrierConfig
from zhisa.data.preparation import load_prepared_split
from zhisa.models.policy import PolicyConfig, PolicyNetwork, build_default_policy
from zhisa.scripts._real_data import add_market_data_args, load_market_dataframe
from zhisa.scripts.train_s1 import _concat, _market_datasets_from_frame
from zhisa.training.losses import LossWeights, MultiTaskLoss
from zhisa.training.optim import OptimConfig
from zhisa.training.s1_ssl import _filter_matching_state_dict
from zhisa.training.s2_supervised import SupervisedTrainer, TrainConfig
from zhisa.utils.seeding import set_seed


def _default_device() -> str:
    pref = os.environ.get("ZHISA_TEST_DEVICE", "auto").lower()
    if pref in {"cpu", "cuda"}:
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"


def _optim_config_from(cfg) -> OptimConfig:
    raw = (cfg.get("optim", {}) or {}) if cfg else {}
    return OptimConfig(
        lr=float(raw.get("lr", 3e-4)),
        weight_decay=float(raw.get("weight_decay", 1e-4)),
        betas=tuple(raw.get("betas", (0.9, 0.95))),
        scheduler=str(raw.get("scheduler", "cosine")),
        warmup_steps=int(raw.get("warmup_steps", 200)),
        step_size=int(raw.get("step_size", 1000)),
        step_gamma=float(raw.get("step_gamma", 0.5)),
        t_max=int(raw.get("t_max", 10_000)),
    )


def _loss_weights_from(cfg) -> LossWeights:
    raw = (cfg.get("loss_weights", {}) or {}) if cfg else {}
    defaults = LossWeights()
    values = {}
    for item in fields(LossWeights):
        default = getattr(defaults, item.name)
        value = raw.get(item.name, default)
        values[item.name] = int(value) if isinstance(default, int) else float(value)
    return LossWeights(**values)


def _sample_spec(cfg, s1_payload: dict | None) -> SampleSpec:
    chart_window = int(cfg.get("chart_window", 32)) if cfg else 32
    image_size = int(cfg.get("image_size", 32)) if cfg else 32
    model_cfg = (s1_payload or {}).get("model_config") or (s1_payload or {}).get("config")
    if model_cfg:
        chart_window = int(model_cfg.get("window", chart_window))
        image_size = int(model_cfg.get("image_size", image_size))
    horizons = tuple(int(x) for x in (cfg.get("horizons", (4, 16, 64)) if cfg else (4, 16, 64)))
    return SampleSpec(
        chart_window=chart_window,
        feature_window=chart_window,
        horizons=horizons,
        image_size=image_size,
        n_regime_states=int(cfg.get("n_regime_states", 4)) if cfg else 4,
    )


def _target_config_from(cfg) -> tuple[MarketTargetConfig, TripleBarrierConfig]:
    raw = (cfg.get("targets", {}) or {}) if cfg else {}
    direction_mode = str(raw.get("direction_mode", "forward_return"))
    target_cfg = MarketTargetConfig(
        direction_mode=direction_mode,
        flat_return_bps=float(raw.get("flat_return_bps", 1.0)),
        use_log_return=bool(raw.get("use_log_return", False)),
    )
    tb_raw = raw.get("triple_barrier", {}) or {}
    tb_cfg = TripleBarrierConfig(
        tp_atr_mult=float(tb_raw.get("tp_atr_mult", 2.0)),
        sl_atr_mult=float(tb_raw.get("sl_atr_mult", 2.0)),
        max_holding=int(tb_raw.get("max_holding", 32)),
        atr_window=int(tb_raw.get("atr_window", 14)),
    )
    allow_asym = bool(raw.get("allow_asymmetric_triple_barrier_direction", False))
    if (
        direction_mode == "triple_barrier"
        and not allow_asym
        and abs(tb_cfg.tp_atr_mult - tb_cfg.sl_atr_mult) > 1e-12
    ):
        raise ValueError(
            "S2 direction_mode='triple_barrier' requires symmetric TP/SL by default; "
            "set targets.allow_asymmetric_triple_barrier_direction=true only for an explicit legacy experiment"
        )
    return target_cfg, tb_cfg


def _build_policy(
    first_ds: MarketDataset,
    spec: SampleSpec,
    s1_payload: dict | None,
) -> PolicyNetwork:
    n_feat = first_ds._features_df.shape[1]
    n_ctx = first_ds._time_features_df.shape[1]
    raw = (s1_payload or {}).get("model_config") or (s1_payload or {}).get("config")
    if not raw:
        return build_default_policy(
            in_numeric_features=n_feat,
            in_context_features=n_ctx,
            window=spec.chart_window,
            image_size=spec.image_size,
            n_actions=9,
            n_regime_classes=spec.n_regime_states,
        )

    model_cfg = dict(raw)
    if isinstance(model_cfg.get("vision_channels"), list):
        model_cfg["vision_channels"] = tuple(model_cfg["vision_channels"])
    allowed = {item.name for item in fields(PolicyConfig)}
    model_cfg = {key: value for key, value in model_cfg.items() if key in allowed}
    expected = {
        "in_numeric_features": n_feat,
        "in_context_features": n_ctx,
        "window": spec.chart_window,
        "image_size": spec.image_size,
        "n_regime_classes": spec.n_regime_states,
    }
    mismatches = {
        key: (model_cfg.get(key), value)
        for key, value in expected.items()
        if key in model_cfg and int(model_cfg[key]) != int(value)
    }
    if mismatches:
        raise ValueError(f"S1 checkpoint is incompatible with the S2 dataset: {mismatches}")
    model_cfg.update(expected)
    return PolicyNetwork(PolicyConfig(**model_cfg))


def _load_s1_representation(model: PolicyNetwork, payload: dict) -> int:
    source = payload.get("model", payload)
    filtered = _filter_matching_state_dict(
        source,
        model,
        excluded_prefixes=("heads.", "memory."),
    )
    if not filtered:
        raise ValueError("S1 checkpoint contains no compatible representation weights")
    model.load_state_dict(filtered, strict=False)
    return len(filtered)


def _target_counts(
    datasets: list[MarketDataset],
    *,
    kind: str,
    n_classes: int,
) -> np.ndarray:
    counts = np.zeros(n_classes, dtype=np.int64)
    for ds in datasets:
        start = ds.spec.chart_window - 1
        stop = start + len(ds)
        if kind == "direction":
            values = ds._tb_label_arr[start:stop] + 1
        elif kind == "regime":
            values = ds._regime_arr[start:stop]
        else:
            raise ValueError(kind)
        counts += np.bincount(values, minlength=n_classes)[:n_classes]
    return counts


def _sqrt_inverse_weights(counts: np.ndarray) -> torch.Tensor:
    active = counts > 0
    if not np.any(active):
        raise ValueError(f"cannot balance classes with no targets: {counts.tolist()}")
    weights = np.zeros_like(counts, dtype=np.float64)
    weights[active] = np.sqrt(counts[active].sum() / (active.sum() * counts[active].astype(np.float64)))
    weights[active] /= weights[active].mean()
    return torch.tensor(weights, dtype=torch.float32)


def _target_contract_dict(
    target_cfg: MarketTargetConfig,
    tb_cfg: TripleBarrierConfig,
) -> dict:
    return {
        "direction_mode": target_cfg.direction_mode,
        "flat_return_bps": target_cfg.flat_return_bps,
        "use_log_return": target_cfg.use_log_return,
        "triple_barrier": {
            "tp_atr_mult": tb_cfg.tp_atr_mult,
            "sl_atr_mult": tb_cfg.sl_atr_mult,
            "max_holding": tb_cfg.max_holding,
            "atr_window": tb_cfg.atr_window,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train S2 supervised market heads.")
    parser.add_argument("--config", type=str, default="configs/s2_supervised.yaml")
    parser.add_argument("--bars", type=int, default=8000)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="artifacts/s2/model.pt")
    parser.add_argument("--best-checkpoint", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--s1-checkpoint", type=str, default=None)
    parser.add_argument("--prepared-root", type=str, default=None)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--prepared-max-bars-per-symbol", type=int, default=None)
    parser.add_argument("--fast-render", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--cache-charts", action="store_true")
    parser.add_argument("--chart-cache-size", type=int, default=-1)
    add_market_data_args(parser)
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path) if cfg_path.exists() else None
    seed = int(cfg.get("seed", 0)) if cfg else 0
    set_seed(seed)
    if args.fast_render:
        os.environ["ZHISA_FAST_RENDER"] = "1"

    s1_payload = None
    if args.s1_checkpoint:
        s1_path = Path(args.s1_checkpoint)
        if not s1_path.is_file():
            raise FileNotFoundError(f"S1 checkpoint not found: {s1_path}")
        s1_payload = torch.load(s1_path, map_location="cpu", weights_only=False)
        stage = (s1_payload.get("checkpoint_meta") or {}).get("stage")
        if stage not in (None, "s1_ssl"):
            raise ValueError(f"expected an S1 checkpoint, got stage={stage!r}")

    spec = _sample_spec(cfg, s1_payload)
    target_cfg, tb_cfg = _target_config_from(cfg)
    manifest = None
    val_ds: Dataset | None = None
    if args.prepared_root:
        prepared_root = Path(args.prepared_root)
        manifest_path = prepared_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"prepared manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        timeframe = str(manifest["timeframe"])
        train_frame = load_prepared_split(prepared_root, args.train_split)
        train_rows = len(train_frame)
        train_datasets = _market_datasets_from_frame(
            train_frame,
            spec=spec,
            cache_charts=args.cache_charts,
            chart_cache_size=args.chart_cache_size,
            max_bars_per_symbol=args.prepared_max_bars_per_symbol,
            timeframe=timeframe,
            compute_targets=True,
            target_cfg=target_cfg,
            triple_barrier_cfg=tb_cfg,
        )
        del train_frame
        if not args.no_validation:
            val_frame = load_prepared_split(prepared_root, args.val_split)
            val_datasets = _market_datasets_from_frame(
                val_frame,
                spec=spec,
                cache_charts=args.cache_charts,
                chart_cache_size=args.chart_cache_size,
                max_bars_per_symbol=args.prepared_max_bars_per_symbol,
                timeframe=timeframe,
                compute_targets=True,
                target_cfg=target_cfg,
                triple_barrier_cfg=tb_cfg,
            )
            del val_frame
            val_ds = _concat(val_datasets)
        train_ds = _concat(train_datasets)
        print(
            f"Prepared S2 data: train={train_rows:,} rows, "
            f"segments={len(train_datasets)}, validation={'on' if val_ds is not None else 'off'}"
        )
    else:
        df = load_market_dataframe(args, seed=seed, default_bars=args.bars)
        only_ds = MarketDataset(
            df,
            spec=spec,
            triple_barrier_cfg=tb_cfg,
            target_cfg=target_cfg,
            cache_charts=args.cache_charts,
            chart_cache_size=args.chart_cache_size,
        )
        train_datasets = [only_ds]
        train_ds = only_ds

    first_ds = train_datasets[0]
    model = _build_policy(first_ds, spec, s1_payload)
    if s1_payload is not None:
        loaded = _load_s1_representation(model, s1_payload)
        print(
            f"Loaded {loaded} S1 representation tensors; S2 market heads are freshly initialized. "
            f"Input contract: window={spec.chart_window}, image={spec.image_size}."
        )

    balance = str(cfg.get("class_balance", "none") if cfg else "none").lower()
    direction_weights = None
    regime_weights = None
    if balance == "sqrt_inverse":
        direction_counts = _target_counts(train_datasets, kind="direction", n_classes=3)
        regime_counts = _target_counts(
            train_datasets,
            kind="regime",
            n_classes=spec.n_regime_states,
        )
        direction_weights = _sqrt_inverse_weights(direction_counts)
        regime_weights = _sqrt_inverse_weights(regime_counts)
        print(
            f"Class balance: direction={direction_counts.tolist()} weights={direction_weights.tolist()} "
            f"regime={regime_counts.tolist()} weights={regime_weights.tolist()}"
        )
        if target_cfg.direction_mode == "forward_return":
            down, flat, up = direction_counts.tolist()
            directional = down + up
            if directional > 0:
                up_share = up / directional
                print(
                    "Direction target audit: "
                    f"mode=forward_return down={down} flat={flat} up={up} "
                    f"up_share_ex_flat={up_share:.4f}"
                )
    elif balance != "none":
        raise ValueError(f"unknown class_balance mode: {balance!r}")

    loss_weights = _loss_weights_from(cfg)
    loss = MultiTaskLoss(
        loss_weights,
        label_smoothing=float(cfg.get("label_smoothing", 0.05)) if cfg else 0.05,
        direction_class_weights=direction_weights,
        regime_class_weights=regime_weights,
    )
    epochs = args.epochs if args.epochs is not None else int(cfg.get("epochs", 2) if cfg else 2)
    batch_size = args.batch_size if args.batch_size is not None else int(cfg.get("batch_size", 32) if cfg else 32)
    device = args.device or str(cfg.get("device", _default_device()) if cfg else _default_device())
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    workers = args.workers if args.workers is not None else int(cfg.get("workers", 0) if cfg else 0)
    best_checkpoint = args.best_checkpoint
    if val_ds is not None and not best_checkpoint:
        checkpoint = Path(args.checkpoint)
        best_checkpoint = str(checkpoint.with_name(f"{checkpoint.stem}_best{checkpoint.suffix}"))
    train_cfg = TrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        grad_clip=float(cfg.get("grad_clip", 1.0)) if cfg else 1.0,
        num_workers=workers,
        log_every=int(cfg.get("log_every", 50)) if cfg else 50,
        eval_every=int(cfg.get("eval_every", 1)) if cfg else 1,
        val_max_batches=int(cfg.get("val_max_batches", 0)) if cfg else 0,
        checkpoint=args.checkpoint,
        best_checkpoint=best_checkpoint,
        checkpoint_every_steps=int(cfg.get("checkpoint_every_steps", 0)) if cfg else 0,
        freeze_encoder_epochs=int(cfg.get("freeze_encoder_epochs", 0)) if cfg else 0,
        encoder_lr_scale=float(cfg.get("encoder_lr_scale", 1.0)) if cfg else 1.0,
        early_stopping_patience=int(cfg.get("early_stopping_patience", 0)) if cfg else 0,
        early_stopping_min_delta=float(cfg.get("early_stopping_min_delta", 0.0)) if cfg else 0.0,
        device=device,
        seed=seed,
        dataset_root=str(Path(args.prepared_root).resolve()) if args.prepared_root else None,
        dataset_timeframe=str(manifest["timeframe"]) if manifest else None,
        dataset_manifest_checksum=str(manifest["output_checksum"]) if manifest else None,
        target_config=_target_contract_dict(target_cfg, tb_cfg),
        champion_metric=str(cfg.get("champion_metric", "s2_composite_score") if cfg else "s2_composite_score"),
        champion_mode=str(cfg.get("champion_mode", "max") if cfg else "max"),
        optim=_optim_config_from(cfg),
    )
    trainer = SupervisedTrainer(model, loss, train_cfg)
    if args.resume_from:
        if not Path(args.resume_from).is_file():
            raise FileNotFoundError(f"S2 resume checkpoint not found: {args.resume_from}")
        print(f"Resumed S2: {trainer.load(args.resume_from)}")
    result = trainer.fit(train_ds, val_ds=val_ds)
    final = result["history"][-1]
    message = f"S2 training complete. final_loss={final['loss']:.6f}"
    if "val" in final:
        message += (
            f" val_total={final['val']['total']:.6f}"
            f" direction_acc={final['val']['direction_accuracy']:.4f}"
            f" direction_bal_acc={final['val']['direction_balanced_accuracy']:.4f}"
            f" return_corr={final['val']['return_corr']:.4f}"
            f" s2_score={final['val']['s2_composite_score']:.4f}"
            f" regime_acc={final['val']['regime_accuracy']:.4f}"
        )
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
