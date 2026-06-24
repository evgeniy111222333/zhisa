"""Deep diagnostics for S2 supervised checkpoints on prepared market splits.

The report is meant to answer whether S2 learned useful market heads:
direction balance/calibration, return correlation, regime quality, per-market
weak spots, and obvious collapse/bias risks.
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
from zhisa.data.dataset import SampleSpec, multimodal_collate
from zhisa.data.preparation import load_prepared_split
from zhisa.models.policy import PolicyConfig, PolicyNetwork
from zhisa.scripts.train_s1 import _market_datasets_from_frame
from zhisa.scripts.train_s2 import _target_config_from
from zhisa.training.dataloader_factory import build_dataloader


def _model_from_checkpoint(payload: dict[str, Any], device: str) -> PolicyNetwork:
    raw = dict(payload.get("model_config") or payload.get("config") or {})
    if isinstance(raw.get("vision_channels"), list):
        raw["vision_channels"] = tuple(raw["vision_channels"])
    allowed = {item.name for item in fields(PolicyConfig)}
    cfg = PolicyConfig(**{k: v for k, v in raw.items() if k in allowed})
    model = PolicyNetwork(cfg).to(device)
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected[:20]}")
    critical_missing = [
        key for key in missing
        if not key.startswith("loss.") and not key.startswith("heads.policy")
    ]
    if critical_missing:
        raise RuntimeError(f"missing model keys: {critical_missing[:20]}")
    model.eval()
    return model


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(x @ y / denom) if denom > 1e-12 else 0.0


class Metrics:
    def __init__(self, n_regime: int, n_bins: int = 15) -> None:
        self.n = 0
        self.n_regime = n_regime
        self.direction_confusion = np.zeros((3, 3), dtype=np.int64)
        self.regime_confusion = np.zeros((n_regime, n_regime), dtype=np.int64)
        self.direction_nll_sum = 0.0
        self.direction_brier_sum = 0.0
        self.confidence_sum = 0.0
        self.entropy_sum = 0.0
        self.bin_count = np.zeros(n_bins, dtype=np.int64)
        self.bin_conf = np.zeros(n_bins, dtype=np.float64)
        self.bin_correct = np.zeros(n_bins, dtype=np.float64)
        self.return_pred: list[np.ndarray] = []
        self.return_target: list[np.ndarray] = []
        self.value_pred: list[np.ndarray] = []
        self.vol_pred: list[np.ndarray] = []
        self.vol_target: list[np.ndarray] = []
        self.risk_pred: list[np.ndarray] = []
        self.risk_target: list[np.ndarray] = []
        self.return_abs_sum = 0.0
        self.vol_abs_sum = 0.0
        self.risk_abs_sum = 0.0

    def update(self, out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> None:
        direction_target = torch.where(
            batch["label_dir"] == -1,
            torch.zeros_like(batch["label_dir"]),
            batch["label_dir"] + 1,
        ).long()
        direction_probs = torch.softmax(out["direction"], dim=-1)
        direction_pred = direction_probs.argmax(dim=-1)
        confidence = direction_probs.max(dim=-1).values
        entropy = -(direction_probs * direction_probs.clamp_min(1e-12).log()).sum(dim=-1)
        target_prob = direction_probs.gather(1, direction_target[:, None]).squeeze(1)
        one_hot = torch.nn.functional.one_hot(direction_target, num_classes=3).float()

        y = direction_target.detach().cpu().numpy()
        p = direction_pred.detach().cpu().numpy()
        np.add.at(self.direction_confusion, (y, p), 1)
        conf = confidence.detach().cpu().numpy()
        correct = (p == y).astype(np.float64)
        bins = np.minimum((conf * len(self.bin_count)).astype(np.int64), len(self.bin_count) - 1)
        np.add.at(self.bin_count, bins, 1)
        np.add.at(self.bin_conf, bins, conf)
        np.add.at(self.bin_correct, bins, correct)

        regime_target = batch["label_regime"].long()
        regime_pred = out["regime"].argmax(dim=-1)
        rt = regime_target.detach().cpu().numpy()
        rp = regime_pred.detach().cpu().numpy()
        valid = (rt >= 0) & (rt < self.n_regime) & (rp >= 0) & (rp < self.n_regime)
        np.add.at(self.regime_confusion, (rt[valid], rp[valid]), 1)

        n = int(direction_target.numel())
        self.n += n
        self.direction_nll_sum += float((-target_prob.clamp_min(1e-12).log()).sum().item())
        self.direction_brier_sum += float(((direction_probs - one_hot) ** 2).sum(dim=-1).sum().item())
        self.confidence_sum += float(confidence.sum().item())
        self.entropy_sum += float(entropy.sum().item())
        ret_pred = out["return_pred"].detach().cpu().flatten().float().numpy()
        ret_target = batch["label_ret"].detach().cpu().flatten().float().numpy()
        value_pred = out["value"].detach().cpu().flatten().float().numpy()
        vol_pred = out["volatility"].detach().cpu().flatten().float().numpy()
        vol_target = batch["label_vol"].detach().cpu().flatten().float().numpy()
        risk_pred = out["risk"].detach().cpu().flatten().float().numpy()
        risk_target = batch["label_risk"].detach().cpu().flatten().float().numpy()
        self.return_pred.append(ret_pred)
        self.return_target.append(ret_target)
        self.value_pred.append(value_pred)
        self.vol_pred.append(vol_pred)
        self.vol_target.append(vol_target)
        self.risk_pred.append(risk_pred)
        self.risk_target.append(risk_target)
        self.return_abs_sum += float(np.abs(ret_pred - ret_target).sum())
        self.vol_abs_sum += float(np.abs(vol_pred - vol_target).sum())
        self.risk_abs_sum += float(np.abs(risk_pred - risk_target).sum())

    @staticmethod
    def _class_report(confusion: np.ndarray) -> dict[str, Any]:
        support = confusion.sum(axis=1)
        predicted = confusion.sum(axis=0)
        recalls = []
        precisions = []
        f1s = []
        for idx in range(confusion.shape[0]):
            if support[idx] <= 0:
                recalls.append(0.0)
                precisions.append(0.0)
                f1s.append(0.0)
                continue
            tp = float(confusion[idx, idx])
            recall = tp / max(1.0, float(support[idx]))
            precision = tp / max(1.0, float(predicted[idx]))
            f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)
        active = support > 0
        return {
            "accuracy": float(np.trace(confusion) / max(1, int(confusion.sum()))),
            "balanced_accuracy": float(np.mean(np.asarray(recalls)[active])) if active.any() else 0.0,
            "macro_f1": float(np.mean(np.asarray(f1s)[active])) if active.any() else 0.0,
            "recall": recalls,
            "precision": precisions,
            "f1": f1s,
            "target_counts": support.tolist(),
            "prediction_counts": predicted.tolist(),
            "confusion": confusion.tolist(),
        }

    def report(self) -> dict[str, Any]:
        direction = self._class_report(self.direction_confusion)
        regime = self._class_report(self.regime_confusion)
        ece = 0.0
        calibration_bins = []
        for idx, count in enumerate(self.bin_count):
            if count <= 0:
                continue
            acc = float(self.bin_correct[idx] / count)
            conf = float(self.bin_conf[idx] / count)
            ece += float(count / max(1, self.n) * abs(acc - conf))
            calibration_bins.append({
                "bin": int(idx),
                "count": int(count),
                "accuracy": acc,
                "confidence": conf,
            })
        ret_pred = np.concatenate(self.return_pred) if self.return_pred else np.asarray([])
        ret_target = np.concatenate(self.return_target) if self.return_target else np.asarray([])
        value_pred = np.concatenate(self.value_pred) if self.value_pred else np.asarray([])
        vol_pred = np.concatenate(self.vol_pred) if self.vol_pred else np.asarray([])
        vol_target = np.concatenate(self.vol_target) if self.vol_target else np.asarray([])
        risk_pred = np.concatenate(self.risk_pred) if self.risk_pred else np.asarray([])
        risk_target = np.concatenate(self.risk_target) if self.risk_target else np.asarray([])
        nonflat = np.abs(ret_target) > 1e-8
        sign_acc = (
            float((np.sign(ret_pred[nonflat]) == np.sign(ret_target[nonflat])).mean())
            if nonflat.any() else 0.0
        )
        pred_counts = np.asarray(direction["prediction_counts"], dtype=np.float64)
        pred_share = pred_counts / max(1.0, float(pred_counts.sum()))
        risks = []
        if pred_share.max(initial=0.0) > 0.75:
            risks.append(f"direction_prediction_collapse:{int(pred_share.argmax())}:{pred_share.max():.3f}")
        if direction["balanced_accuracy"] < 0.42:
            risks.append("low_direction_balanced_accuracy")
        if direction["macro_f1"] < 0.38:
            risks.append("low_direction_macro_f1")
        ret_corr = _safe_corr(ret_pred, ret_target)
        value_corr = _safe_corr(value_pred, ret_target)
        vol_corr = _safe_corr(vol_pred, vol_target)
        risk_corr = _safe_corr(risk_pred, risk_target)
        if ret_corr < 0.0:
            risks.append("negative_return_correlation")
        if value_corr < 0.0:
            risks.append("negative_value_correlation")
        if ece > 0.12:
            risks.append("poor_direction_calibration")

        return {
            "samples": int(self.n),
            "direction": direction,
            "regime": regime,
            "direction_nll": self.direction_nll_sum / max(1, self.n),
            "direction_brier": self.direction_brier_sum / max(1, self.n),
            "direction_ece": ece,
            "mean_confidence": self.confidence_sum / max(1, self.n),
            "mean_entropy": self.entropy_sum / max(1, self.n),
            "calibration_bins": calibration_bins,
            "return_corr": ret_corr,
            "return_mae": self.return_abs_sum / max(1, self.n),
            "return_sign_accuracy_nonflat": sign_acc,
            "return_target_mean": float(ret_target.mean()) if ret_target.size else 0.0,
            "return_target_std": float(ret_target.std()) if ret_target.size else 0.0,
            "return_pred_mean": float(ret_pred.mean()) if ret_pred.size else 0.0,
            "return_pred_std": float(ret_pred.std()) if ret_pred.size else 0.0,
            "value_corr": value_corr,
            "value_mae": float(np.abs(value_pred - ret_target).mean()) if value_pred.size else 0.0,
            "value_pred_mean": float(value_pred.mean()) if value_pred.size else 0.0,
            "value_pred_std": float(value_pred.std()) if value_pred.size else 0.0,
            "volatility_corr": vol_corr,
            "volatility_mae": self.vol_abs_sum / max(1, self.n),
            "volatility_target_mean": float(vol_target.mean()) if vol_target.size else 0.0,
            "volatility_target_std": float(vol_target.std()) if vol_target.size else 0.0,
            "volatility_pred_mean": float(vol_pred.mean()) if vol_pred.size else 0.0,
            "volatility_pred_std": float(vol_pred.std()) if vol_pred.size else 0.0,
            "risk_corr": risk_corr,
            "risk_mae": self.risk_abs_sum / max(1, self.n),
            "risk_target_mean": float(risk_target.mean()) if risk_target.size else 0.0,
            "risk_target_std": float(risk_target.std()) if risk_target.size else 0.0,
            "risk_pred_mean": float(risk_pred.mean()) if risk_pred.size else 0.0,
            "risk_pred_std": float(risk_pred.std()) if risk_pred.size else 0.0,
            "risk_flags": risks,
        }


def _merge(dst: Metrics, src: Metrics) -> None:
    dst.n += src.n
    dst.direction_confusion += src.direction_confusion
    dst.regime_confusion += src.regime_confusion
    dst.direction_nll_sum += src.direction_nll_sum
    dst.direction_brier_sum += src.direction_brier_sum
    dst.confidence_sum += src.confidence_sum
    dst.entropy_sum += src.entropy_sum
    dst.bin_count += src.bin_count
    dst.bin_conf += src.bin_conf
    dst.bin_correct += src.bin_correct
    dst.return_pred.extend(src.return_pred)
    dst.return_target.extend(src.return_target)
    dst.value_pred.extend(src.value_pred)
    dst.vol_pred.extend(src.vol_pred)
    dst.vol_target.extend(src.vol_target)
    dst.risk_pred.extend(src.risk_pred)
    dst.risk_target.extend(src.risk_target)
    dst.return_abs_sum += src.return_abs_sum
    dst.vol_abs_sum += src.vol_abs_sum
    dst.risk_abs_sum += src.risk_abs_sum


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/s2_supervised_15m_12markets.yaml")
    parser.add_argument("--prepared-root", default="data/prepared/s1_15m_12m_v2")
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--max-batches-per-segment", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("ZHISA_FAST_RENDER", "1")
    cfg = load_config(args.config)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = _model_from_checkpoint(payload, args.device)
    model_cfg = model.cfg
    spec = SampleSpec(
        chart_window=model_cfg.window,
        feature_window=model_cfg.window,
        horizons=tuple(int(x) for x in cfg.get("horizons", (4, 16, 64))),
        image_size=model_cfg.image_size,
        n_regime_states=model_cfg.n_regime_classes,
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

    overall = Metrics(n_regime=model_cfg.n_regime_classes)
    per_segment: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for idx, dataset in enumerate(datasets, start=1):
            local = Metrics(n_regime=model_cfg.n_regime_classes)
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
                    "label_vol": batch.label_vol.to(args.device, non_blocking=True),
                    "label_risk": batch.label_risk.to(args.device, non_blocking=True),
                    "label_regime": batch.label_regime.to(args.device, non_blocking=True),
                    "label_ret": batch.label_ret.to(args.device, non_blocking=True),
                }
                out = model(
                    chart=batch_d["chart"],
                    numeric=batch_d["numeric"],
                    context=batch_d["context"],
                )
                local.update(out, batch_d)
            _merge(overall, local)
            name = str(getattr(dataset.df, "name", f"segment-{idx}"))
            per_segment[name] = local.report()
            print(f"evaluated {idx}/{len(datasets)} {name} samples={local.n}", flush=True)

    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "timeframe": manifest["timeframe"],
        "frame_start": str(frame.index.min()),
        "frame_end": str(frame.index.max()),
        "segments": len(datasets),
        "trainer_state": payload.get("trainer_state", {}),
        "train_config": payload.get("train_config", {}),
        "overall": overall.report(),
        "per_segment": per_segment,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
