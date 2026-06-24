"""Evaluate whether an S2 checkpoint is good for its own stage.

S2 is a supervised market-representation stage, not a live/paper policy.
This script therefore measures:

* primary heads used by the old diagnostics;
* auxiliary multi-horizon direction/return heads, if present;
* simple achievable baselines on the same labels;
* calibration, per-market weak spots, and a compact readiness verdict.

The goal is to avoid judging a noisy 15m forward-return task as if it were
an execution policy, while still catching genuine collapse and useless heads.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from zhisa.config import load_config
from zhisa.data.dataset import MarketDataset, SampleSpec, multimodal_collate
from zhisa.data.preparation import load_prepared_split
from zhisa.models.policy import PolicyConfig, PolicyNetwork
from zhisa.scripts.train_s1 import _market_datasets_from_frame
from zhisa.scripts.train_s2 import _target_config_from
from zhisa.training.dataloader_factory import build_dataloader


CLASS_NAMES = ("DOWN", "FLAT", "UP")


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    x = x.astype(np.float64) - float(x.mean())
    y = y.astype(np.float64) - float(y.mean())
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(x @ y / denom) if denom > 1e-12 else 0.0


def _class_report(confusion: np.ndarray) -> dict[str, Any]:
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    recalls: list[float] = []
    precisions: list[float] = []
    f1s: list[float] = []
    for idx in range(confusion.shape[0]):
        tp = float(confusion[idx, idx])
        recall = tp / max(1.0, float(support[idx]))
        precision = tp / max(1.0, float(predicted[idx]))
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
    active = support > 0
    target_share = support.astype(np.float64) / max(1.0, float(support.sum()))
    prediction_share = predicted.astype(np.float64) / max(1.0, float(predicted.sum()))
    return {
        "accuracy": float(np.trace(confusion) / max(1, int(confusion.sum()))),
        "balanced_accuracy": float(np.mean(np.asarray(recalls)[active])) if active.any() else 0.0,
        "macro_f1": float(np.mean(np.asarray(f1s)[active])) if active.any() else 0.0,
        "recall": dict(zip(CLASS_NAMES, recalls)),
        "precision": dict(zip(CLASS_NAMES, precisions)),
        "f1": dict(zip(CLASS_NAMES, f1s)),
        "target_counts": dict(zip(CLASS_NAMES, support.astype(int).tolist())),
        "prediction_counts": dict(zip(CLASS_NAMES, predicted.astype(int).tolist())),
        "target_share": dict(zip(CLASS_NAMES, target_share.tolist())),
        "prediction_share": dict(zip(CLASS_NAMES, prediction_share.tolist())),
        "confusion": confusion.astype(int).tolist(),
    }


def _direction_stats(
    logits: np.ndarray,
    labels_raw: np.ndarray,
    *,
    n_bins: int = 15,
) -> dict[str, Any]:
    labels = np.where(labels_raw == -1, 0, labels_raw + 1).astype(np.int64)
    logits = logits.astype(np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    pred = probs.argmax(axis=1)
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (labels, pred), 1)
    confidence = probs.max(axis=1)
    target_prob = probs[np.arange(labels.size), labels]
    one_hot = np.eye(3, dtype=np.float64)[labels]
    bins = np.minimum((confidence * n_bins).astype(np.int64), n_bins - 1)
    ece = 0.0
    calibration_bins = []
    for idx in range(n_bins):
        mask = bins == idx
        if not mask.any():
            continue
        acc = float((pred[mask] == labels[mask]).mean())
        conf = float(confidence[mask].mean())
        ece += float(mask.mean() * abs(acc - conf))
        calibration_bins.append({
            "bin": idx,
            "count": int(mask.sum()),
            "accuracy": acc,
            "confidence": conf,
        })
    report = _class_report(confusion)
    report.update({
        "nll": float(-np.log(np.maximum(target_prob, 1e-12)).mean()),
        "brier": float(((probs - one_hot) ** 2).sum(axis=1).mean()),
        "ece": ece,
        "mean_confidence": float(confidence.mean()),
        "mean_entropy": float(-(probs * np.log(np.maximum(probs, 1e-12))).sum(axis=1).mean()),
        "calibration_bins": calibration_bins,
    })
    return report


def _constant_baseline(labels_raw: np.ndarray, cls: int) -> dict[str, Any]:
    labels = np.where(labels_raw == -1, 0, labels_raw + 1).astype(np.int64)
    pred = np.full_like(labels, cls)
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (labels, pred), 1)
    return _class_report(confusion)


def _baseline_report(labels_raw: np.ndarray, persistence_raw: np.ndarray) -> dict[str, Any]:
    labels = np.where(labels_raw == -1, 0, labels_raw + 1).astype(np.int64)
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    prior = counts / max(1.0, float(counts.sum()))
    majority_cls = int(prior.argmax())
    entropy = float(-(prior[prior > 0] * np.log(prior[prior > 0])).sum())
    uniform_nll = float(np.log(3.0))
    prior_nll = float(-np.log(np.maximum(prior[labels], 1e-12)).mean())

    persistence_labels = np.where(persistence_raw == -1, 0, persistence_raw + 1).astype(np.int64)
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (labels, persistence_labels), 1)
    persistence = _class_report(confusion)
    return {
        "class_prior": dict(zip(CLASS_NAMES, prior.tolist())),
        "target_entropy_nats": entropy,
        "uniform_random_nll": uniform_nll,
        "class_prior_nll": prior_nll,
        "majority_class": CLASS_NAMES[majority_cls],
        "majority": _constant_baseline(labels_raw, majority_cls),
        "causal_persistence": persistence,
    }


def _return_report(pred: np.ndarray, target: np.ndarray, labels_raw: np.ndarray) -> dict[str, Any]:
    directional = labels_raw != 0
    sign_acc = (
        float((np.sign(pred[directional]) == np.sign(target[directional])).mean())
        if directional.any() else 0.0
    )
    zero_mae = float(np.abs(target).mean()) if target.size else 0.0
    return {
        "corr": _safe_corr(pred, target),
        "mae": float(np.abs(pred - target).mean()) if pred.size else 0.0,
        "zero_baseline_mae": zero_mae,
        "mae_improvement_vs_zero": zero_mae - float(np.abs(pred - target).mean()) if pred.size else 0.0,
        "sign_accuracy_directional": sign_acc,
        "target_mean": float(target.mean()) if target.size else 0.0,
        "target_std": float(target.std()) if target.size else 0.0,
        "pred_mean": float(pred.mean()) if pred.size else 0.0,
        "pred_std": float(pred.std()) if pred.size else 0.0,
    }


class Collector:
    def __init__(self, horizons: tuple[int, ...], n_regime: int) -> None:
        self.horizons = horizons
        self.n_regime = n_regime
        self.primary_logits: list[np.ndarray] = []
        self.primary_return_pred: list[np.ndarray] = []
        self.multi_logits: list[list[np.ndarray]] = [[] for _ in horizons]
        self.multi_return_pred: list[list[np.ndarray]] = [[] for _ in horizons]
        self.label_dir_multi: list[list[np.ndarray]] = [[] for _ in horizons]
        self.label_ret_multi: list[list[np.ndarray]] = [[] for _ in horizons]
        self.persistence_multi: list[list[np.ndarray]] = [[] for _ in horizons]
        self.regime_confusion = np.zeros((n_regime, n_regime), dtype=np.int64)
        self.vol_pred: list[np.ndarray] = []
        self.vol_target: list[np.ndarray] = []
        self.risk_pred: list[np.ndarray] = []
        self.risk_target: list[np.ndarray] = []
        self.value_pred: list[np.ndarray] = []

    def update(
        self,
        out: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        dataset: MarketDataset,
        meta: list[dict[str, Any]],
    ) -> None:
        self.primary_logits.append(out["direction"].detach().cpu().float().numpy())
        self.primary_return_pred.append(out["return_pred"].detach().cpu().float().numpy())

        label_dir_multi = batch["label_dir_multi"].detach().cpu().numpy()
        label_ret_multi = batch["label_ret_multi"].detach().cpu().numpy()
        if "direction_multi" in out:
            direction_multi = out["direction_multi"].detach().cpu().float().numpy()
            return_multi = out["return_multi"].detach().cpu().float().numpy()
        else:
            direction_multi = np.zeros((label_dir_multi.shape[0], len(self.horizons), 3), dtype=np.float32)
            return_multi = np.zeros((label_dir_multi.shape[0], len(self.horizons)), dtype=np.float32)

        indices = np.asarray([int(item["t"]) for item in meta], dtype=np.int64)
        for h_idx, horizon in enumerate(self.horizons):
            self.multi_logits[h_idx].append(direction_multi[:, h_idx, :])
            self.multi_return_pred[h_idx].append(return_multi[:, h_idx])
            self.label_dir_multi[h_idx].append(label_dir_multi[:, h_idx])
            self.label_ret_multi[h_idx].append(label_ret_multi[:, h_idx])
            past_idx = np.maximum(indices - int(horizon), 0)
            valid = indices >= int(horizon)
            persistence = dataset._tb_multi_label_arr[past_idx, h_idx].copy()
            persistence[~valid] = 0
            self.persistence_multi[h_idx].append(persistence)

        regime_target = batch["label_regime"].detach().cpu().numpy().astype(np.int64)
        regime_pred = out["regime"].argmax(dim=-1).detach().cpu().numpy().astype(np.int64)
        valid_regime = (
            (regime_target >= 0)
            & (regime_target < self.n_regime)
            & (regime_pred >= 0)
            & (regime_pred < self.n_regime)
        )
        np.add.at(self.regime_confusion, (regime_target[valid_regime], regime_pred[valid_regime]), 1)
        self.vol_pred.append(out["volatility"].detach().cpu().float().numpy())
        self.vol_target.append(batch["label_vol"].detach().cpu().float().numpy())
        self.risk_pred.append(out["risk"].detach().cpu().float().numpy())
        self.risk_target.append(batch["label_risk"].detach().cpu().float().numpy())
        self.value_pred.append(out["value"].detach().cpu().float().numpy())

    def report(self, primary_horizon_index: int) -> dict[str, Any]:
        horizon_reports = {}
        for h_idx, horizon in enumerate(self.horizons):
            logits = np.concatenate(self.multi_logits[h_idx], axis=0)
            labels = np.concatenate(self.label_dir_multi[h_idx], axis=0)
            rets = np.concatenate(self.label_ret_multi[h_idx], axis=0)
            ret_pred = np.concatenate(self.multi_return_pred[h_idx], axis=0)
            persistence = np.concatenate(self.persistence_multi[h_idx], axis=0)
            direction = _direction_stats(logits, labels)
            baselines = _baseline_report(labels, persistence)
            direction["lift_vs_majority_balanced"] = (
                direction["balanced_accuracy"] - baselines["majority"]["balanced_accuracy"]
            )
            direction["lift_vs_persistence_balanced"] = (
                direction["balanced_accuracy"] - baselines["causal_persistence"]["balanced_accuracy"]
            )
            direction["nll_improvement_vs_prior"] = baselines["class_prior_nll"] - direction["nll"]
            horizon_reports[str(horizon)] = {
                "direction": direction,
                "return": _return_report(ret_pred, rets, labels),
                "baselines": baselines,
            }

        primary_logits = np.concatenate(self.primary_logits, axis=0)
        primary_labels = np.concatenate(self.label_dir_multi[primary_horizon_index], axis=0)
        primary_ret_pred = np.concatenate(self.primary_return_pred, axis=0)
        primary_ret_target = np.concatenate(self.label_ret_multi[primary_horizon_index], axis=0)
        primary_direction = _direction_stats(primary_logits, primary_labels)
        primary_baselines = _baseline_report(
            primary_labels,
            np.concatenate(self.persistence_multi[primary_horizon_index], axis=0),
        )
        primary_direction["lift_vs_majority_balanced"] = (
            primary_direction["balanced_accuracy"] - primary_baselines["majority"]["balanced_accuracy"]
        )
        primary_direction["lift_vs_persistence_balanced"] = (
            primary_direction["balanced_accuracy"] - primary_baselines["causal_persistence"]["balanced_accuracy"]
        )
        primary_direction["nll_improvement_vs_prior"] = primary_baselines["class_prior_nll"] - primary_direction["nll"]

        vol_pred = np.concatenate(self.vol_pred)
        vol_target = np.concatenate(self.vol_target)
        risk_pred = np.concatenate(self.risk_pred)
        risk_target = np.concatenate(self.risk_target)
        value_pred = np.concatenate(self.value_pred)
        stage = _stage_verdict(
            primary_direction=primary_direction,
            primary_return=_return_report(primary_ret_pred, primary_ret_target, primary_labels),
            horizon_reports=horizon_reports,
            volatility_corr=_safe_corr(vol_pred, vol_target),
            risk_corr=_safe_corr(risk_pred, risk_target),
            value_corr=_safe_corr(value_pred, primary_ret_target),
        )
        return {
            "samples": int(primary_labels.size),
            "primary_horizon": int(self.horizons[primary_horizon_index]),
            "primary": {
                "direction": primary_direction,
                "return": _return_report(primary_ret_pred, primary_ret_target, primary_labels),
                "baselines": primary_baselines,
            },
            "multi_horizon": horizon_reports,
            "regime": _class_report(self.regime_confusion),
            "volatility": {
                "corr": _safe_corr(vol_pred, vol_target),
                "mae": float(np.abs(vol_pred - vol_target).mean()),
                "target_std": float(vol_target.std()),
                "pred_std": float(vol_pred.std()),
            },
            "risk": {
                "corr": _safe_corr(risk_pred, risk_target),
                "mae": float(np.abs(risk_pred - risk_target).mean()),
                "target_std": float(risk_target.std()),
                "pred_std": float(risk_pred.std()),
            },
            "value_head_note": {
                "corr": _safe_corr(value_pred, primary_ret_target),
                "trained_in_s2": False,
                "reason": "configs/s2_supervised_15m_12markets.yaml sets loss_weights.value=0.0",
            },
            "stage_readiness": stage,
        }


def _stage_verdict(
    *,
    primary_direction: dict[str, Any],
    primary_return: dict[str, Any],
    horizon_reports: dict[str, Any],
    volatility_corr: float,
    risk_corr: float,
    value_corr: float,
) -> dict[str, Any]:
    scores = {
        "volatility_risk": max(0.0, min(1.0, (volatility_corr + risk_corr) / 1.30)),
        "direction_lift": max(
            0.0,
            min(1.0, (primary_direction["lift_vs_persistence_balanced"] + 0.02) / 0.08),
        ),
        "direction_shape": max(0.0, min(1.0, (primary_direction["macro_f1"] - 0.333) / 0.09)),
        "calibration": max(0.0, min(1.0, (0.10 - primary_direction["ece"]) / 0.10)),
        "return_signal": max(0.0, min(1.0, (primary_return["corr"] + 0.005) / 0.05)),
    }
    multi_lifts = [
        float(item["direction"]["lift_vs_persistence_balanced"])
        for item in horizon_reports.values()
    ]
    scores["multi_horizon_consistency"] = max(
        0.0,
        min(1.0, (float(np.mean(multi_lifts)) + 0.02) / 0.08 if multi_lifts else 0.0),
    )
    overall = (
        0.30 * scores["volatility_risk"]
        + 0.20 * scores["direction_lift"]
        + 0.15 * scores["direction_shape"]
        + 0.15 * scores["calibration"]
        + 0.10 * scores["return_signal"]
        + 0.10 * scores["multi_horizon_consistency"]
    )
    blockers = []
    cautions = []
    if volatility_corr < 0.55 or risk_corr < 0.55:
        blockers.append("volatility_or_risk_representation_too_weak")
    if primary_direction["balanced_accuracy"] < 0.34:
        blockers.append("primary_direction_below_minimum_random_plus_noise_floor")
    if primary_direction["lift_vs_persistence_balanced"] < -0.01:
        blockers.append("primary_direction_worse_than_causal_persistence")
    if primary_direction["ece"] > 0.10:
        cautions.append("direction_calibration_is_poor")
    if primary_return["corr"] < 0.01:
        cautions.append("signed_return_signal_is_very_weak")
    if value_corr < 0.0:
        cautions.append("value_head_is_untrained_in_s2_ignore_until_s4")
    if overall >= 0.70 and not blockers:
        verdict = "good_for_s2_stage"
    elif overall >= 0.50 and not blockers:
        verdict = "usable_but_weak_direction"
    elif not blockers:
        verdict = "borderline_representation_only"
    else:
        verdict = "not_ready_without_fix"
    return {
        "verdict": verdict,
        "score_0_1": float(overall),
        "component_scores": scores,
        "blockers": blockers,
        "cautions": cautions,
        "interpretation": (
            "Use this as an S2 representation checkpoint only. S2 direction/return labels are noisy; "
            "the key question is whether the checkpoint beats causal baselines without collapsing, "
            "while preserving strong volatility/risk structure."
        ),
    }


def _model_from_checkpoint(payload: dict[str, Any], device: str) -> PolicyNetwork:
    raw = dict(payload.get("model_config") or payload.get("config") or {})
    if isinstance(raw.get("vision_channels"), list):
        raw["vision_channels"] = tuple(raw["vision_channels"])
    if isinstance(raw.get("market_horizons"), list):
        raw["market_horizons"] = tuple(int(x) for x in raw["market_horizons"])
    allowed = {item.name for item in fields(PolicyConfig)}
    cfg = PolicyConfig(**{k: v for k, v in raw.items() if k in allowed})
    model = PolicyNetwork(cfg).to(device)
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected[:20]}")
    critical_missing = [
        key for key in missing
        if not key.startswith("heads.policy")
    ]
    if critical_missing:
        raise RuntimeError(f"missing model keys: {critical_missing[:20]}")
    model.eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate S2 stage readiness.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/s2_supervised_15m_12markets.yaml")
    parser.add_argument("--prepared-root", default="data/prepared/s1_15m_12m_v2")
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--max-batches-per-segment", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("ZHISA_FAST_RENDER", "1")
    cfg = load_config(args.config)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = _model_from_checkpoint(payload, args.device)
    horizons = tuple(int(x) for x in cfg.get("horizons", (4, 16, 64)))
    if model.cfg.market_horizons:
        horizons = tuple(int(x) for x in model.cfg.market_horizons)
    primary_horizon_index = len(horizons) // 2
    spec = SampleSpec(
        chart_window=model.cfg.window,
        feature_window=model.cfg.window,
        horizons=horizons,
        image_size=model.cfg.image_size,
        n_regime_states=model.cfg.n_regime_classes,
    )
    target_cfg, tb_cfg = _target_config_from(cfg)
    prepared_root = Path(args.prepared_root)
    manifest = json.loads((prepared_root / "manifest.json").read_text(encoding="utf-8"))
    frame = load_prepared_split(prepared_root, args.split)
    datasets = _market_datasets_from_frame(
        frame,
        spec=spec,
        cache_charts=False,
        chart_cache_size=-1,
        timeframe=str(manifest["timeframe"]),
        compute_targets=True,
        target_cfg=target_cfg,
        triple_barrier_cfg=tb_cfg,
    )
    if args.max_segments > 0:
        datasets = datasets[:args.max_segments]

    overall = Collector(horizons=horizons, n_regime=model.cfg.n_regime_classes)
    per_market: dict[str, Any] = {}
    with torch.inference_mode():
        for idx, dataset in enumerate(datasets, start=1):
            local = Collector(horizons=horizons, n_regime=model.cfg.n_regime_classes)
            loader = build_dataloader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                collate_fn=multimodal_collate,
                drop_last=False,
            )
            for batch_idx, batch in enumerate(loader):
                if args.max_batches_per_segment > 0 and batch_idx >= args.max_batches_per_segment:
                    break
                batch_d = {
                    "chart": batch.chart.to(args.device, non_blocking=True),
                    "numeric": batch.numeric.to(args.device, non_blocking=True),
                    "context": batch.context.to(args.device, non_blocking=True),
                    "label_dir": batch.label_dir.to(args.device, non_blocking=True),
                    "label_ret": batch.label_ret.to(args.device, non_blocking=True),
                    "label_dir_multi": batch.label_dir_multi.to(args.device, non_blocking=True),
                    "label_ret_multi": batch.label_ret_multi.to(args.device, non_blocking=True),
                    "label_vol": batch.label_vol.to(args.device, non_blocking=True),
                    "label_risk": batch.label_risk.to(args.device, non_blocking=True),
                    "label_regime": batch.label_regime.to(args.device, non_blocking=True),
                }
                out = model(
                    chart=batch_d["chart"],
                    numeric=batch_d["numeric"],
                    context=batch_d["context"],
                )
                local.update(out, batch_d, dataset, batch.meta)
                overall.update(out, batch_d, dataset, batch.meta)
            name = str(getattr(dataset.df, "name", f"segment-{idx}"))
            per_market[name] = local.report(primary_horizon_index)
            print(f"evaluated {idx}/{len(datasets)} {name} samples={per_market[name]['samples']}", flush=True)

    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "timeframe": manifest["timeframe"],
        "frame_start": str(frame.index.min()),
        "frame_end": str(frame.index.max()),
        "segments": len(datasets),
        "horizons": list(horizons),
        "trainer_state_summary": {
            "completed_epochs": payload.get("trainer_state", {}).get("completed_epochs"),
            "step": payload.get("trainer_state", {}).get("step"),
            "best_val_metric": payload.get("trainer_state", {}).get("best_val_metric"),
            "best_val_total": payload.get("trainer_state", {}).get("best_val_total"),
            "early_stopping_bad_epochs": payload.get("trainer_state", {}).get("early_stopping_bad_epochs"),
        },
        "target_config": payload.get("checkpoint_meta", {}).get("target_config"),
        "overall": overall.report(primary_horizon_index),
        "per_market": per_market,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["overall"]["stage_readiness"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
