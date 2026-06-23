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
from torch.utils.data import DataLoader
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
    device: str = "cpu"
    seed: int = 0
    dataset_root: Optional[str] = None
    dataset_timeframe: Optional[str] = None
    dataset_manifest_checksum: Optional[str] = None
    target_config: dict = field(default_factory=dict)
    champion_metric: str = "s2_composite_score"
    champion_mode: str = "max"
    optim: OptimConfig = field(default_factory=OptimConfig)


class SupervisedTrainer:
    """The S2 supervised multi-task trainer."""

    def __init__(
        self,
        model: PolicyNetwork,
        loss: MultiTaskLoss,
        cfg: TrainConfig,
    ) -> None:
        self.model = model
        self.loss = loss
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model.to(self.device)
        self.loss.to(self.device)
        self.opt = self._build_optimizer()
        self.sched = build_scheduler(self.opt, cfg.optim)
        self._step = 0
        self._completed_epochs = 0
        self._history: list[dict] = []
        self._best_val_total = float("inf")
        self._best_val_metric = float("-inf") if cfg.champion_mode == "max" else float("inf")
        self._early_stopping_bad_epochs = 0

    def _build_optimizer(self) -> AdamW:
        encoder_prefixes = ("vision.", "numeric.", "context.", "fusion.")
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
            "label_vol": batch.label_vol.to(self.device, non_blocking=True),
            "label_risk": batch.label_risk.to(self.device, non_blocking=True),
            "label_regime": batch.label_regime.to(self.device, non_blocking=True),
            "label_ret": batch.label_ret.to(self.device, non_blocking=True),
        }

    def fit(self, train_ds: MarketDataset, val_ds: Optional[MarketDataset] = None) -> dict:
        cfg = self.cfg
        train_loader = build_dataloader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, collate_fn=multimodal_collate, drop_last=True,
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
            if (
                cfg.early_stopping_patience > 0
                and self._early_stopping_bad_epochs >= cfg.early_stopping_patience
            ):
                logger.info(
                    "S2 early stopping after epoch %d: best_%s=%.6f bad_epochs=%d",
                    epoch,
                    cfg.champion_metric,
                    self._best_val_metric,
                    self._early_stopping_bad_epochs,
                )
                break
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
        return_pred_values: list[torch.Tensor] = []
        return_target_values: list[torch.Tensor] = []
        for batch_idx, batch in enumerate(loader):
            if self.cfg.val_max_batches > 0 and batch_idx >= self.cfg.val_max_batches:
                break
            batch_d = self._move_batch(batch)
            out = self.model(
                chart=batch_d["chart"],
                numeric=batch_d["numeric"],
                context=batch_d["context"],
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
            regime_correct += int((out["regime"].argmax(-1) == batch_d["label_regime"]).sum())
            return_abs += float((out["return_pred"] - batch_d["label_ret"]).abs().sum())
            return_pred_values.append(out["return_pred"].detach().cpu().flatten())
            return_target_values.append(batch_d["label_ret"].detach().cpu().flatten())
            volatility_abs += float((out["volatility"] - batch_d["label_vol"]).abs().sum())
            risk_abs += float((out["risk"] - batch_d["label_risk"]).abs().sum())
            n += batch_n
        if n == 0:
            raise RuntimeError("S2 validation produced no samples")
        support = direction_confusion.sum(dim=1).float()
        predicted = direction_confusion.sum(dim=0).float()
        active = support > 0
        recalls = []
        f1_scores = []
        for cls in range(3):
            if not bool(active[cls]):
                continue
            tp = float(direction_confusion[cls, cls])
            recall = tp / max(1.0, float(support[cls]))
            precision = tp / max(1.0, float(predicted[cls]))
            recalls.append(recall)
            f1_scores.append(2.0 * precision * recall / max(1e-12, precision + recall))
        direction_balanced = float(sum(recalls) / max(1, len(recalls)))
        direction_macro_f1 = float(sum(f1_scores) / max(1, len(f1_scores)))
        if return_pred_values:
            pred_ret = torch.cat(return_pred_values).float()
            target_ret = torch.cat(return_target_values).float()
            pred_centered = pred_ret - pred_ret.mean()
            target_centered = target_ret - target_ret.mean()
            denom = pred_centered.norm() * target_centered.norm()
            return_corr = float((pred_centered @ target_centered / denom).item()) if float(denom) > 0 else 0.0
        else:
            return_corr = 0.0
        metrics = {k: v / n for k, v in agg.items()}
        total_for_score = float(metrics.get("total", 0.0))
        regime_accuracy = regime_correct / n
        return_mae = return_abs / n
        volatility_mae = volatility_abs / n
        s2_composite = (
            0.45 * direction_balanced
            + 0.25 * direction_macro_f1
            + 0.15 * max(-1.0, min(1.0, return_corr))
            + 0.10 * regime_accuracy
            - 0.03 * total_for_score
            - 0.02 * return_mae
        )
        metrics.update({
            "direction_accuracy": direction_correct / n,
            "direction_balanced_accuracy": direction_balanced,
            "direction_macro_f1": direction_macro_f1,
            "direction_confusion": direction_confusion.tolist(),
            "regime_accuracy": regime_correct / n,
            "return_mae": return_mae,
            "return_corr": return_corr,
            "volatility_mae": volatility_mae,
            "risk_mae": risk_abs / n,
            "s2_composite_score": s2_composite,
            "n_samples": n,
        })
        return metrics

    def _set_encoder_trainable(self, trainable: bool) -> None:
        # S1 trains encode() only, so WorkingMemory is still randomly
        # initialised and must warm up together with the supervised heads.
        for name in ("vision", "numeric", "context", "fusion"):
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
