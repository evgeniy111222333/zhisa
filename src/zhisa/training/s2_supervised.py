"""S2: Supervised multi-task trainer.

This is the MVP trainer. It uses the labelled ``MarketDataset`` and
trains a ``PolicyNetwork`` with a multi-task loss. It is designed to
be a solid baseline that subsequent phases (SSL pretrain, RL, online)
can build on.
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW

from zhisa.data.dataset import MarketDataset, multimodal_collate
from zhisa.models.policy import PolicyConfig, PolicyNetwork
from zhisa.training.dataloader_factory import build_dataloader
from zhisa.training.losses import LossWeights, MultiTaskLoss
from zhisa.training.optim import OptimConfig, build_scheduler
from zhisa.utils.logging import get_logger
from zhisa.utils.timing import Timer

logger = get_logger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 64
    grad_clip: float = 1.0
    num_workers: int = 0
    log_every: int = 50
    eval_every: int = 1          # 0 = no eval
    val_max_batches: int = 0     # 0 = full validation split
    checkpoint: Optional[str] = None
    best_checkpoint: Optional[str] = None
    checkpoint_every_steps: int = 0
    freeze_encoder_epochs: int = 0
    encoder_lr_scale: float = 1.0
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    early_stopping_min_epochs: int = 0
    early_stopping_trend_window: int = 0
    early_stopping_trend_min_delta: float = 0.0
    device: str = "cpu"
    seed: int = 0
    dataset_root: Optional[str] = None
    dataset_timeframe: Optional[str] = None
    dataset_manifest_checksum: Optional[str] = None
    target_config: dict = field(default_factory=dict)
    champion_metric: str = "s2_composite_score"
    champion_mode: str = "max"
    segment_validation: bool = False
    guard_min_direction_balanced: float = 0.0
    guard_min_flat_recall: float = 0.0
    guard_min_flat_f1: float = 0.0
    guard_min_volatility_corr: float = -1.0
    guard_min_return_corr: float = -1.0
    guard_min_persistence_lift: float = -1.0
    guard_max_prediction_share: float = 1.0
    guard_max_flat_prediction_share: float = 1.0
    guard_min_flat_pred_target_ratio: float = 0.0
    guard_max_flat_pred_target_ratio: float = 10.0
    guard_penalty_scale: float = 0.0
    optim: OptimConfig = field(default_factory=OptimConfig)


def _segment_guard_score(
    base_score: float,
    segment_metrics: dict[str, dict],
    cfg: TrainConfig,
) -> tuple[float, dict[str, float]]:
    """Penalise champions that hide weak markets behind average metrics."""
    if not segment_metrics:
        return float(base_score), {
            "s2_segment_guard_penalty": 0.0,
            "s2_worst_segment_direction_balanced_accuracy": 0.0,
            "s2_worst_segment_flat_recall": 0.0,
            "s2_worst_segment_flat_f1": 0.0,
            "s2_worst_segment_volatility_corr": 0.0,
            "s2_worst_segment_return_corr": 0.0,
            "s2_worst_segment_persistence_lift": 0.0,
            "s2_worst_segment_max_prediction_share": 0.0,
            "s2_worst_segment_flat_prediction_share": 0.0,
            "s2_worst_segment_flat_pred_target_ratio": 0.0,
        }
    worst_dir = min(
        float(m.get("direction_balanced_accuracy", 0.0))
        for m in segment_metrics.values()
    )
    worst_flat = min(float(m.get("direction_flat_recall", 0.0)) for m in segment_metrics.values())
    worst_flat_f1 = min(float(m.get("direction_flat_f1", 0.0)) for m in segment_metrics.values())
    worst_vol = min(float(m.get("volatility_corr", 0.0)) for m in segment_metrics.values())
    worst_ret = min(float(m.get("return_corr", 0.0)) for m in segment_metrics.values())
    worst_persistence_lift = min(
        float(m.get("direction_lift_vs_persistence_balanced", 0.0))
        for m in segment_metrics.values()
    )
    worst_max_pred_share = max(
        float(m.get("direction_max_prediction_share", 0.0))
        for m in segment_metrics.values()
    )
    worst_flat_pred_share = max(
        float((m.get("direction_prediction_share") or [0.0, 0.0, 0.0])[1])
        for m in segment_metrics.values()
    )
    worst_flat_ratio = max(
        float(m.get("direction_flat_pred_target_ratio", 0.0))
        for m in segment_metrics.values()
    )
    best_flat_ratio_floor = min(
        float(m.get("direction_flat_pred_target_ratio", 0.0))
        for m in segment_metrics.values()
    )
    penalty = (
        max(0.0, float(cfg.guard_min_direction_balanced) - worst_dir)
        + max(0.0, float(cfg.guard_min_flat_recall) - worst_flat)
        + max(0.0, float(cfg.guard_min_flat_f1) - worst_flat_f1)
        + max(0.0, float(cfg.guard_min_volatility_corr) - worst_vol)
        + max(0.0, float(cfg.guard_min_return_corr) - worst_ret)
        + max(0.0, float(cfg.guard_min_persistence_lift) - worst_persistence_lift)
        + max(0.0, worst_max_pred_share - float(cfg.guard_max_prediction_share))
        + max(0.0, worst_flat_pred_share - float(cfg.guard_max_flat_prediction_share))
        + max(0.0, float(cfg.guard_min_flat_pred_target_ratio) - best_flat_ratio_floor)
        + max(0.0, worst_flat_ratio - float(cfg.guard_max_flat_pred_target_ratio))
    )
    penalty *= max(0.0, float(cfg.guard_penalty_scale))
    guarded = float(base_score) - penalty
    return guarded, {
        "s2_segment_guard_penalty": penalty,
        "s2_worst_segment_direction_balanced_accuracy": worst_dir,
        "s2_worst_segment_flat_recall": worst_flat,
        "s2_worst_segment_flat_f1": worst_flat_f1,
        "s2_worst_segment_volatility_corr": worst_vol,
        "s2_worst_segment_return_corr": worst_ret,
        "s2_worst_segment_persistence_lift": worst_persistence_lift,
        "s2_worst_segment_max_prediction_share": worst_max_pred_share,
        "s2_worst_segment_flat_prediction_share": worst_flat_pred_share,
        "s2_worst_segment_flat_pred_target_ratio": worst_flat_ratio,
    }


def _history_metric_values(history: list[dict], metric: str) -> list[float]:
    values: list[float] = []
    for record in history:
        val = record.get("val")
        if not isinstance(val, dict):
            continue
        value = float(val.get(metric, float("nan")))
        if math.isfinite(value):
            values.append(value)
    return values


def _recent_metric_trend_is_improving(
    history: list[dict],
    *,
    metric: str,
    mode: str,
    window: int,
    min_delta: float,
) -> bool:
    """Return True when recent validation metrics still show useful momentum."""
    if window <= 1:
        return False
    values = _history_metric_values(history, metric)
    if len(values) < window:
        return False
    recent = values[-window:]
    if mode == "max":
        return recent[-1] - recent[0] > float(min_delta)
    if mode == "min":
        return recent[0] - recent[-1] > float(min_delta)
    raise ValueError(f"champion_mode must be 'max' or 'min', got {mode!r}")


class SupervisedTrainer:
    """The S2 supervised multi-task trainer."""

    def __init__(
        self,
        model: PolicyNetwork,
        loss: MultiTaskLoss,
        cfg: TrainConfig,
        train_sample_weights: Optional[torch.Tensor] = None,
    ) -> None:
        self.model = model
        self.loss = loss
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model.to(self.device)
        self.loss.to(self.device)
        self.opt = self._build_optimizer()
        self.sched = build_scheduler(self.opt, cfg.optim)
        self.train_sample_weights = train_sample_weights.detach().cpu().float() if train_sample_weights is not None else None
        self._step = 0
        self._completed_epochs = 0
        self._history: list[dict] = []
        self._best_val_total = float("inf")
        self._best_val_metric = float("-inf") if cfg.champion_mode == "max" else float("inf")
        self._early_stopping_bad_epochs = 0

    def _build_optimizer(self) -> AdamW:
        encoder_prefixes = (
            "vision.",
            "numeric.",
            "context.",
            "fusion.",
            "macro_numeric.",
            "timeframe_embed.",
            "macro_gate.",
            "macro_proj.",
            "macro_norm.",
        )
        groups: dict[tuple[bool, bool], list[torch.nn.Parameter]] = {}
        named = list(self.model.named_parameters()) + [
            (f"loss.{name}", value) for name, value in self.loss.named_parameters()
        ]
        for name, parameter in named:
            if not parameter.requires_grad:
                continue
            is_encoder = name.startswith(encoder_prefixes)
            lname = name.lower()
            use_decay = not (
                parameter.ndim < 2
                or lname.endswith(".bias")
                or "norm" in lname
                or "bn" in lname
            )
            groups.setdefault((is_encoder, use_decay), []).append(parameter)
        param_groups = []
        for (is_encoder, use_decay), parameters in groups.items():
            param_groups.append({
                "params": parameters,
                "lr": self.cfg.optim.lr
                * (self.cfg.encoder_lr_scale if is_encoder else 1.0),
                "weight_decay": self.cfg.optim.weight_decay if use_decay else 0.0,
                "s2_encoder_group": is_encoder,
            })
        return AdamW(param_groups, betas=self.cfg.optim.betas)

    def _move_batch(self, batch) -> dict:
        return {
            "chart": batch.chart.to(self.device, non_blocking=True),
            "numeric": batch.numeric.to(self.device, non_blocking=True),
            "context": batch.context.to(self.device, non_blocking=True),
            "label_dir": batch.label_dir.to(self.device, non_blocking=True),
            "label_dir_persistence": batch.label_dir_persistence.to(self.device, non_blocking=True)
            if batch.label_dir_persistence is not None else None,
            "label_vol": batch.label_vol.to(self.device, non_blocking=True),
            "label_risk": batch.label_risk.to(self.device, non_blocking=True),
            "label_regime": batch.label_regime.to(self.device, non_blocking=True),
            "label_ret": batch.label_ret.to(self.device, non_blocking=True),
            "label_dir_multi": batch.label_dir_multi.to(self.device, non_blocking=True)
            if batch.label_dir_multi is not None else None,
            "label_dir_multi_persistence": batch.label_dir_multi_persistence.to(self.device, non_blocking=True)
            if batch.label_dir_multi_persistence is not None else None,
            "label_ret_multi": batch.label_ret_multi.to(self.device, non_blocking=True)
            if batch.label_ret_multi is not None else None,
            "macro_numeric": batch.macro_numeric.to(self.device, non_blocking=True)
            if batch.macro_numeric is not None else None,
        }

    def fit(self, train_ds: MarketDataset, val_ds: Optional[MarketDataset] = None) -> dict:
        cfg = self.cfg
        sampler = None
        shuffle = True
        if self.train_sample_weights is not None:
            if int(self.train_sample_weights.numel()) != len(train_ds):
                raise ValueError(
                    "train_sample_weights length must match train dataset length: "
                    f"{int(self.train_sample_weights.numel())} != {len(train_ds)}"
                )
            sampler = WeightedRandomSampler(
                self.train_sample_weights,
                num_samples=len(train_ds),
                replacement=True,
            )
            shuffle = False
        train_loader = build_dataloader(
            train_ds, batch_size=cfg.batch_size, shuffle=shuffle,
            num_workers=cfg.num_workers, collate_fn=multimodal_collate, drop_last=True,
            sampler=sampler,
        )
        timer = Timer()
        for epoch in range(self._completed_epochs, cfg.epochs):
            self.model.train()
            encoder_trainable = epoch >= cfg.freeze_encoder_epochs
            self._set_encoder_trainable(encoder_trainable)
            ep_sums: dict[str, float] = {}
            ep_count = 0
            timer.start()
            for it, batch in enumerate(train_loader):
                batch_d = self._move_batch(batch)
                out = self.model(
                    chart=batch_d["chart"],
                    numeric=batch_d["numeric"],
                    context=batch_d["context"],
                    macro_numeric=batch_d["macro_numeric"],
                )
                losses = self.loss(out, batch_d)
                loss = losses["total"]
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.opt.step()
                if self.sched is not None:
                    self.sched.step()
                self._step += 1
                batch_n = batch_d["chart"].size(0)
                for name, value in losses.items():
                    ep_sums[name] = ep_sums.get(name, 0.0) + float(value.item()) * batch_n
                ep_count += batch_n
                if (it + 1) % cfg.log_every == 0:
                    avg = ep_sums.get("total", 0.0) / max(1, ep_count)
                    lr = max(group["lr"] for group in self.opt.param_groups)
                    logger.info(
                        "epoch=%d iter=%d step=%d loss=%.5f lr=%.2e elapsed=%.1fs",
                        epoch, it, self._step, avg, lr, timer.elapsed,
                    )
                if (
                    cfg.checkpoint
                    and cfg.checkpoint_every_steps > 0
                    and self._step % cfg.checkpoint_every_steps == 0
                ):
                    self.save(cfg.checkpoint)
            if ep_count == 0:
                raise RuntimeError(
                    "S2 training produced no batches; reduce batch_size or add data"
                )
            averages = {name: value / ep_count for name, value in ep_sums.items()}
            avg = averages["total"]
            timer.stop()
            record = {
                "epoch": epoch,
                "loss": avg,
                "elapsed_s": timer.elapsed,
                "encoder_trainable": encoder_trainable,
                "components": {k: v for k, v in averages.items() if k != "total"},
            }
            logger.info("epoch %d done in %.1fs, avg_loss=%.5f", epoch, timer.elapsed, avg)
            timer.reset()
            if val_ds is not None and cfg.eval_every and (epoch + 1) % cfg.eval_every == 0:
                val_metrics = self.evaluate(val_ds)
                record["val"] = val_metrics
            self._completed_epochs = epoch + 1
            self._history.append(record)
            val_metrics = record.get("val", {})
            val_total = float(val_metrics.get("total", float("inf")))
            val_champion = float(val_metrics.get(cfg.champion_metric, val_total))
            previous_best = self._best_val_metric
            improved = self._is_better(val_champion, previous_best)
            if math.isfinite(val_champion):
                delta = (
                    val_champion - previous_best
                    if cfg.champion_mode == "max"
                    else previous_best - val_champion
                )
                if delta > cfg.early_stopping_min_delta:
                    self._early_stopping_bad_epochs = 0
                else:
                    self._early_stopping_bad_epochs += 1
            if improved:
                self._best_val_metric = val_champion
                self._best_val_total = val_total
                if cfg.best_checkpoint:
                    self.save(cfg.best_checkpoint)
            if cfg.checkpoint:
                self.save(cfg.checkpoint)
            min_epochs_met = self._completed_epochs >= max(0, int(cfg.early_stopping_min_epochs))
            trend_improving = _recent_metric_trend_is_improving(
                self._history,
                metric=cfg.champion_metric,
                mode=cfg.champion_mode,
                window=int(cfg.early_stopping_trend_window),
                min_delta=float(cfg.early_stopping_trend_min_delta),
            )
            should_stop = (
                cfg.early_stopping_patience > 0
                and min_epochs_met
                and self._early_stopping_bad_epochs >= cfg.early_stopping_patience
                and not trend_improving
            )
            if should_stop:
                logger.info(
                    "S2 early stopping after epoch %d: best_%s=%.6f bad_epochs=%d",
                    epoch,
                    cfg.champion_metric,
                    self._best_val_metric,
                    self._early_stopping_bad_epochs,
                )
                break
            if (
                cfg.early_stopping_patience > 0
                and self._early_stopping_bad_epochs >= cfg.early_stopping_patience
                and trend_improving
            ):
                logger.info(
                    "S2 early stopping deferred after epoch %d: recent %s trend is still improving",
                    epoch,
                    cfg.champion_metric,
                )
        self._set_encoder_trainable(True)
        return {"history": list(self._history), "final_step": self._step}

    def _is_better(self, value: float, best: float) -> bool:
        if not math.isfinite(value):
            return False
        if self.cfg.champion_mode == "max":
            return value > best
        if self.cfg.champion_mode == "min":
            return value < best
        raise ValueError(f"champion_mode must be 'max' or 'min', got {self.cfg.champion_mode!r}")

    @torch.no_grad()
    def evaluate(self, ds: MarketDataset) -> dict:
        metrics = self._evaluate_single(ds)
        if self.cfg.segment_validation and isinstance(ds, ConcatDataset):
            segment_metrics: dict[str, dict] = {}
            for idx, child in enumerate(ds.datasets):
                name = str(getattr(getattr(child, "df", None), "name", f"segment-{idx}"))
                segment_metrics[name] = self._evaluate_single(child)
            guarded, guard_metrics = _segment_guard_score(
                float(metrics.get("s2_composite_score", 0.0)),
                segment_metrics,
                self.cfg,
            )
            metrics.update(guard_metrics)
            metrics["s2_guarded_score"] = guarded
            metrics["segment_metrics"] = segment_metrics
        else:
            metrics["s2_guarded_score"] = metrics.get("s2_composite_score", 0.0)
        return metrics

    @torch.no_grad()
    def _evaluate_single(self, ds: MarketDataset) -> dict:
        loader = build_dataloader(
            ds, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.num_workers, collate_fn=multimodal_collate,
        )
        self.model.eval()
        agg: dict = {}
        n = 0
        direction_correct = 0
        regime_correct = 0
        return_abs = 0.0
        volatility_abs = 0.0
        risk_abs = 0.0
        direction_confusion = torch.zeros((3, 3), dtype=torch.long)
        persistence_confusion = torch.zeros((3, 3), dtype=torch.long)
        return_pred_values: list[torch.Tensor] = []
        return_target_values: list[torch.Tensor] = []
        value_pred_values: list[torch.Tensor] = []
        volatility_pred_values: list[torch.Tensor] = []
        volatility_target_values: list[torch.Tensor] = []
        risk_pred_values: list[torch.Tensor] = []
        risk_target_values: list[torch.Tensor] = []
        for batch_idx, batch in enumerate(loader):
            if self.cfg.val_max_batches > 0 and batch_idx >= self.cfg.val_max_batches:
                break
            batch_d = self._move_batch(batch)
            out = self.model(
                chart=batch_d["chart"],
                numeric=batch_d["numeric"],
                context=batch_d["context"],
                macro_numeric=batch_d["macro_numeric"],
            )
            losses = self.loss(out, batch_d)
            for k, v in losses.items():
                agg[k] = agg.get(k, 0.0) + float(v.item()) * batch_d["chart"].size(0)
            batch_n = batch_d["chart"].size(0)
            direction_target = torch.where(
                batch_d["label_dir"] == -1,
                torch.zeros_like(batch_d["label_dir"]),
                batch_d["label_dir"] + 1,
            )
            direction_correct += int((out["direction"].argmax(-1) == direction_target).sum())
            direction_pred = out["direction"].argmax(-1)
            for target, pred in zip(direction_target.cpu(), direction_pred.cpu()):
                direction_confusion[int(target), int(pred)] += 1
            if batch_d.get("label_dir_persistence") is not None:
                persistence_pred = torch.where(
                    batch_d["label_dir_persistence"] == -1,
                    torch.zeros_like(batch_d["label_dir_persistence"]),
                    batch_d["label_dir_persistence"] + 1,
                )
                for target, pred in zip(direction_target.cpu(), persistence_pred.cpu()):
                    persistence_confusion[int(target), int(pred)] += 1
            regime_correct += int((out["regime"].argmax(-1) == batch_d["label_regime"]).sum())
            return_abs += float((out["return_pred"] - batch_d["label_ret"]).abs().sum())
            return_pred_values.append(out["return_pred"].detach().cpu().flatten())
            return_target_values.append(batch_d["label_ret"].detach().cpu().flatten())
            value_pred_values.append(out["value"].detach().cpu().flatten())
            volatility_abs += float((out["volatility"] - batch_d["label_vol"]).abs().sum())
            volatility_pred_values.append(out["volatility"].detach().cpu().flatten())
            volatility_target_values.append(batch_d["label_vol"].detach().cpu().flatten())
            risk_abs += float((out["risk"] - batch_d["label_risk"]).abs().sum())
            risk_pred_values.append(out["risk"].detach().cpu().flatten())
            risk_target_values.append(batch_d["label_risk"].detach().cpu().flatten())
            n += batch_n
        if n == 0:
            raise RuntimeError("S2 validation produced no samples")
        support = direction_confusion.sum(dim=1).float()
        predicted = direction_confusion.sum(dim=0).float()
        target_share = (support / support.sum().clamp_min(1.0)).tolist()
        prediction_share = (predicted / predicted.sum().clamp_min(1.0)).tolist()
        max_prediction_share = max(float(x) for x in prediction_share)
        flat_pred_target_ratio = (
            float(prediction_share[1]) / max(float(target_share[1]), 1e-8)
        )
        active = support > 0
        recalls = []
        precisions = []
        f1_scores = []
        for cls in range(3):
            if not bool(active[cls]):
                continue
            tp = float(direction_confusion[cls, cls])
            recall = tp / max(1.0, float(support[cls]))
            precision = tp / max(1.0, float(predicted[cls]))
            recalls.append(recall)
            precisions.append(precision)
            f1_scores.append(2.0 * precision * recall / max(1e-12, precision + recall))
        direction_balanced = float(sum(recalls) / max(1, len(recalls)))
        direction_macro_f1 = float(sum(f1_scores) / max(1, len(f1_scores)))
        persistence_support = persistence_confusion.sum(dim=1).float()
        persistence_predicted = persistence_confusion.sum(dim=0).float()
        persistence_recalls = []
        for cls in range(3):
            if not bool(persistence_support[cls] > 0):
                continue
            tp = float(persistence_confusion[cls, cls])
            persistence_recalls.append(tp / max(1.0, float(persistence_support[cls])))
        persistence_total = int(persistence_confusion.sum().item())
        persistence_accuracy = (
            float(persistence_confusion.diag().sum().item()) / max(1, persistence_total)
        )
        persistence_balanced = float(
            sum(persistence_recalls) / max(1, len(persistence_recalls))
        )
        persistence_lift = direction_balanced - persistence_balanced
        flat_recall = (
            float(direction_confusion[1, 1])
            / max(1.0, float(direction_confusion[1].sum()))
        )
        flat_precision = (
            float(direction_confusion[1, 1])
            / max(1.0, float(direction_confusion[:, 1].sum()))
        )
        flat_f1 = 2.0 * flat_precision * flat_recall / max(1e-12, flat_precision + flat_recall)

        def _corr(pred_values: list[torch.Tensor], target_values: list[torch.Tensor]) -> float:
            if not pred_values or not target_values:
                return 0.0
            pred = torch.cat(pred_values).float()
            target = torch.cat(target_values).float()
            pred_centered = pred - pred.mean()
            target_centered = target - target.mean()
            denom = pred_centered.norm() * target_centered.norm()
            return float((pred_centered @ target_centered / denom).item()) if float(denom) > 0 else 0.0

        return_corr = _corr(return_pred_values, return_target_values)
        value_corr = _corr(value_pred_values, return_target_values)
        volatility_corr = _corr(volatility_pred_values, volatility_target_values)
        risk_corr = _corr(risk_pred_values, risk_target_values)
        metrics = {k: v / n for k, v in agg.items()}
        total_for_score = float(metrics.get("total", 0.0))
        regime_accuracy = regime_correct / n
        return_mae = return_abs / n
        volatility_mae = volatility_abs / n
        distribution_penalty = max(0.0, max_prediction_share - 0.62)
        flat_collapse_penalty = max(0.0, float(prediction_share[1]) - 0.35)
        flat_ratio_penalty = max(0.0, flat_pred_target_ratio - 1.8)
        persistence_penalty = max(0.0, -persistence_lift)
        s2_composite = (
            0.40 * direction_balanced
            + 0.35 * direction_macro_f1
            + 0.15 * max(-1.0, min(1.0, return_corr))
            + 0.10 * regime_accuracy
            + 0.05 * flat_f1
            + 0.05 * max(-0.10, min(0.10, persistence_lift))
            - 0.03 * total_for_score
            - 0.02 * return_mae
            - 0.20 * distribution_penalty
            - 0.25 * flat_collapse_penalty
            - 0.03 * flat_ratio_penalty
            - 0.20 * persistence_penalty
        )
        metrics.update({
            "direction_accuracy": direction_correct / n,
            "direction_balanced_accuracy": direction_balanced,
            "direction_macro_f1": direction_macro_f1,
            "direction_flat_recall": flat_recall,
            "direction_flat_precision": flat_precision,
            "direction_flat_f1": flat_f1,
            "direction_target_share": target_share,
            "direction_prediction_share": prediction_share,
            "direction_max_prediction_share": max_prediction_share,
            "direction_flat_pred_target_ratio": flat_pred_target_ratio,
            "direction_confusion": direction_confusion.tolist(),
            "direction_persistence_accuracy": persistence_accuracy,
            "direction_persistence_balanced_accuracy": persistence_balanced,
            "direction_lift_vs_persistence_balanced": persistence_lift,
            "direction_persistence_prediction_share": (
                persistence_predicted / persistence_predicted.sum().clamp_min(1.0)
            ).tolist(),
            "direction_persistence_confusion": persistence_confusion.tolist(),
            "regime_accuracy": regime_correct / n,
            "return_mae": return_mae,
            "return_corr": return_corr,
            "value_corr": value_corr,
            "volatility_mae": volatility_mae,
            "volatility_corr": volatility_corr,
            "risk_mae": risk_abs / n,
            "risk_corr": risk_corr,
            "s2_composite_score": s2_composite,
            "n_samples": n,
        })
        return metrics

    def _set_encoder_trainable(self, trainable: bool) -> None:
        # S1 trains encode() only, so WorkingMemory is still randomly
        # initialised and must warm up together with the supervised heads.
        for name in ("vision", "numeric", "context", "fusion", "macro_numeric", "timeframe_embed", "macro_gate", "macro_proj", "macro_norm"):
            module = getattr(self.model, name, None)
            if module is None:
                continue
            module.requires_grad_(trainable)
            if not trainable:
                module.eval()

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Convert the config to a fully-JSON-serialisable dict. The
        # ``vision_channels`` field is a tuple, which torch.save will
        # happily pickle but json.dumps will not. We also store the
        # ``PolicyConfig`` as a dict under ``model_config`` so downstream
        # tools (explain, backtest, evaluate) can rebuild an identical
        # model without re-deriving the hyperparameters from the YAML.
        cfg_dict = self.model.cfg.__dict__.copy()
        if "vision_channels" in cfg_dict and isinstance(cfg_dict["vision_channels"], tuple):
            cfg_dict["vision_channels"] = list(cfg_dict["vision_channels"])
        if "market_horizons" in cfg_dict and isinstance(cfg_dict["market_horizons"], tuple):
            cfg_dict["market_horizons"] = list(cfg_dict["market_horizons"])
        payload = {
            "model": self.model.state_dict(),
            "loss": self.loss.state_dict(),
            "config": cfg_dict,
            "model_config": cfg_dict,  # canonical name
            "train_config": asdict(self.cfg),
            "optimizer": self.opt.state_dict(),
            "scheduler": self.sched.state_dict() if self.sched is not None else None,
            "trainer_state": {
                "step": self._step,
                "completed_epochs": self._completed_epochs,
                "history": self._history,
                "best_val_total": self._best_val_total,
                "best_val_metric": self._best_val_metric,
                "early_stopping_bad_epochs": self._early_stopping_bad_epochs,
            },
            "checkpoint_meta": {
                "stage": "s2_supervised",
                "trading_policy_ready": False,
                "policy_head_trained": False,
                "reason": "S2 trains supervised market heads; use S2b/S4+ for paper trading policies.",
                "resume_granularity": "epoch",
                "dataset": {
                    "root": self.cfg.dataset_root,
                    "timeframe": self.cfg.dataset_timeframe,
                    "manifest_checksum": self.cfg.dataset_manifest_checksum,
                },
                "target_config": self.cfg.target_config,
                "champion_metric": self.cfg.champion_metric,
                "champion_mode": self.cfg.champion_mode,
            },
        }
        tmp = p.with_name(f".{p.name}.tmp-{os.getpid()}")
        try:
            torch.save(payload, tmp)
            os.replace(tmp, p)
        finally:
            if tmp.exists():
                tmp.unlink()
        logger.info("checkpoint saved to %s", p)

    def load(self, path: str) -> dict:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(payload["model"], strict=True)
        if "loss" in payload:
            self.loss.load_state_dict(payload["loss"], strict=False)
        optimizer_restored = False
        if "optimizer" in payload:
            try:
                self.opt.load_state_dict(payload["optimizer"])
                optimizer_restored = True
            except (ValueError, RuntimeError) as exc:
                logger.warning("could not restore S2 optimizer state: %s", exc)
        if optimizer_restored and self.sched is not None and payload.get("scheduler"):
            self.sched.load_state_dict(payload["scheduler"])
        state = payload.get("trainer_state", {}) if optimizer_restored else {}
        self._step = int(state.get("step", 0))
        self._completed_epochs = int(state.get("completed_epochs", 0))
        self._history = list(state.get("history", []))
        self._best_val_total = float(state.get("best_val_total", float("inf")))
        self._best_val_metric = float(
            state.get(
                "best_val_metric",
                float("-inf") if self.cfg.champion_mode == "max" else float("inf"),
            )
        )
        self._early_stopping_bad_epochs = int(
            state.get("early_stopping_bad_epochs", 0)
        )
        status = {
            "optimizer_restored": optimizer_restored,
            "resume_mode": "full" if optimizer_restored else "warm_start",
            "step": self._step,
            "completed_epochs": self._completed_epochs,
        }
        logger.info("S2 checkpoint loaded from %s | %s", path, status)
        return status
