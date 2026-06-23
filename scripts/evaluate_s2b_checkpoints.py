"""Evaluate one or more S2b checkpoints on a prepared held-out split."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from zhisa.config import load_config
from zhisa.data.dataset import SampleSpec
from zhisa.data.expert import build_expert
from zhisa.data.preparation import load_prepared_split
from zhisa.scripts._rl_training import build_policy_from_checkpoint, load_trading_checkpoint
from zhisa.scripts.train_s1 import _market_datasets_from_frame
from zhisa.training.dataloader_factory import build_dataloader
from zhisa.training.s2b_imitation import (
    _LabeledMarketDataset,
    _batched_collate,
    _expert_actions_for_dataset,
)


@dataclass
class Metrics:
    confusion: np.ndarray = field(default_factory=lambda: np.zeros((9, 9), dtype=np.int64))
    nll_sum: float = 0.0
    brier_sum: float = 0.0
    confidence_sum: float = 0.0
    entropy_sum: float = 0.0
    n: int = 0
    bin_count: np.ndarray = field(default_factory=lambda: np.zeros(15, dtype=np.int64))
    bin_confidence: np.ndarray = field(default_factory=lambda: np.zeros(15, dtype=np.float64))
    bin_correct: np.ndarray = field(default_factory=lambda: np.zeros(15, dtype=np.float64))

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        probs = torch.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)
        confidence = probs.max(dim=-1).values
        target_probs = probs.gather(1, target[:, None]).squeeze(1)
        self.nll_sum += float((-target_probs.clamp_min(1e-12).log()).sum().item())
        one_hot = torch.nn.functional.one_hot(target, num_classes=probs.shape[-1])
        self.brier_sum += float(((probs - one_hot) ** 2).sum(dim=-1).sum().item())
        self.confidence_sum += float(confidence.sum().item())
        self.entropy_sum += float(
            (-(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)).sum().item()
        )
        y = target.cpu().numpy()
        p = pred.cpu().numpy()
        conf = confidence.cpu().numpy()
        np.add.at(self.confusion, (y, p), 1)
        bins = np.minimum((conf * 15).astype(np.int64), 14)
        np.add.at(self.bin_count, bins, 1)
        np.add.at(self.bin_confidence, bins, conf)
        np.add.at(self.bin_correct, bins, (p == y).astype(np.float64))
        self.n += len(y)

    def report(self) -> dict:
        support = self.confusion.sum(axis=1)
        predicted = self.confusion.sum(axis=0)
        active = np.flatnonzero(support)
        recalls = []
        f1s = []
        for cls in active:
            tp = float(self.confusion[cls, cls])
            recall = tp / max(1.0, float(support[cls]))
            precision = tp / max(1.0, float(predicted[cls]))
            recalls.append(recall)
            f1s.append(2 * precision * recall / max(1e-12, precision + recall))
        direction = np.zeros((3, 3), dtype=np.int64)
        mapping = np.array([0, 1, 1, 1, 2, 2, 2, 0, 0], dtype=np.int64)
        for actual in range(9):
            for pred in range(9):
                direction[mapping[actual], mapping[pred]] += self.confusion[actual, pred]
        direction_recalls = []
        direction_f1s = []
        for cls in range(3):
            tp = float(direction[cls, cls])
            recall = tp / max(1.0, float(direction[cls].sum()))
            precision = tp / max(1.0, float(direction[:, cls].sum()))
            direction_recalls.append(recall)
            direction_f1s.append(2 * precision * recall / max(1e-12, precision + recall))
        ece = 0.0
        for idx in np.flatnonzero(self.bin_count):
            ece += self.bin_count[idx] / max(1, self.n) * abs(
                self.bin_correct[idx] / self.bin_count[idx]
                - self.bin_confidence[idx] / self.bin_count[idx]
            )
        accuracy = float(np.trace(self.confusion) / max(1, self.n))
        nll = self.nll_sum / max(1, self.n)
        balanced = float(np.mean(recalls)) if recalls else 0.0
        macro_f1 = float(np.mean(f1s)) if f1s else 0.0
        dir_balanced = float(np.mean(direction_recalls))
        dir_macro_f1 = float(np.mean(direction_f1s))
        composite = (
            0.25 * macro_f1 + 0.20 * balanced + 0.25 * dir_macro_f1
            + 0.20 * dir_balanced + 0.05 * accuracy - 0.03 * nll - 0.02 * ece
        )
        return {
            "samples": self.n,
            "nll": nll,
            "brier": self.brier_sum / max(1, self.n),
            "accuracy": accuracy,
            "balanced_accuracy": balanced,
            "macro_f1": macro_f1,
            "direction_balanced_accuracy": dir_balanced,
            "direction_macro_f1": dir_macro_f1,
            "ece": float(ece),
            "mean_confidence": self.confidence_sum / max(1, self.n),
            "mean_entropy": self.entropy_sum / max(1, self.n),
            "composite_score": composite,
            "target_counts": support.tolist(),
            "prediction_counts": predicted.tolist(),
            "direction_confusion": direction.tolist(),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.environ.setdefault("ZHISA_FAST_RENDER", "1")

    cfg = load_config(args.config)
    payloads = [load_trading_checkpoint(path) for path in args.checkpoints]
    models = [build_policy_from_checkpoint(payload).to(args.device).eval() for payload in payloads]
    model_cfg = models[0].cfg
    if any(model.cfg != model_cfg for model in models[1:]):
        raise ValueError("all checkpoints must use the same model architecture")
    spec = SampleSpec(
        chart_window=model_cfg.window,
        feature_window=model_cfg.window,
        image_size=model_cfg.image_size,
        n_regime_states=model_cfg.n_regime_classes,
    )
    root = Path(args.prepared_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    frame = load_prepared_split(root, args.split)
    datasets = _market_datasets_from_frame(
        frame, spec=spec, cache_charts=False, chart_cache_size=-1,
        timeframe=str(manifest["timeframe"]), compute_targets=True,
    )
    expert_kind = str(cfg.get("expert", "symmetric_utility"))
    expert_kwargs = dict(cfg.get("expert_kwargs", {}) or {})
    expert_kwargs.setdefault("chart_window", model_cfg.window)
    global_metrics = [Metrics() for _ in models]
    market_metrics: list[dict[str, Metrics]] = [{} for _ in models]

    with torch.inference_mode():
        for dataset_idx, dataset in enumerate(datasets):
            name = str(getattr(dataset.df, "name", f"segment-{dataset_idx}"))
            actions = _expert_actions_for_dataset(
                dataset, build_expert(expert_kind, **expert_kwargs),
            )
            labeled = _LabeledMarketDataset(dataset, actions)
            loader = build_dataloader(
                labeled, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, collate_fn=_batched_collate, drop_last=False,
            )
            local = [Metrics() for _ in models]
            for batch in loader:
                chart = batch["chart"].to(args.device, non_blocking=True)
                numeric = batch["numeric"].to(args.device, non_blocking=True)
                context = batch["context"].to(args.device, non_blocking=True)
                target = batch["action"].to(args.device, non_blocking=True)
                for idx, model in enumerate(models):
                    logits = model(chart=chart, numeric=numeric, context=context)["policy_logits"]
                    global_metrics[idx].update(logits, target)
                    local[idx].update(logits, target)
            for idx in range(len(models)):
                market_metrics[idx][name] = local[idx]
            print(f"evaluated {dataset_idx + 1}/{len(datasets)}: {name}", flush=True)

    report = {
        "split": args.split,
        "timeframe": manifest["timeframe"],
        "start": str(frame.index.min()),
        "end": str(frame.index.max()),
        "checkpoints": {},
    }
    for idx, path in enumerate(args.checkpoints):
        report["checkpoints"][str(Path(path).resolve())] = {
            "overall": global_metrics[idx].report(),
            "markets": {name: metric.report() for name, metric in market_metrics[idx].items()},
        }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({path: data["overall"] for path, data in report["checkpoints"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
