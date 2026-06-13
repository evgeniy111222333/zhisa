"""Supervised training for the RegimeEncoder."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from zhisa.regime.dataset import (
    RegimeSupervisionBatch,
    RegimeSupervisionDataset,
    regime_supervision_collate,
)
from zhisa.regime.encoder import RegimeEncoder
from zhisa.training.optim import OptimConfig, build_optimizer, build_scheduler
from zhisa.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RegimeLossWeights:
    macro: float = 1.0
    meso: float = 0.75
    risk_mode: float = 0.5
    playbook: float = 0.4
    tradeability: float = 0.5
    transition_risk: float = 0.5


class RegimeEncoderLoss(nn.Module):
    """Multi-task loss for regime representation learning."""

    def __init__(self, weights: Optional[RegimeLossWeights] = None) -> None:
        super().__init__()
        self.weights = weights or RegimeLossWeights()

    def forward(self, outputs: dict[str, torch.Tensor], batch: RegimeSupervisionBatch) -> dict[str, torch.Tensor]:
        losses = {
            "macro": F.cross_entropy(outputs["macro_logits"], batch.macro),
            "meso": F.cross_entropy(outputs["meso_logits"], batch.meso),
            "risk_mode": F.cross_entropy(outputs["risk_logits"], batch.risk_mode),
            "tradeability": F.smooth_l1_loss(outputs["tradeability"], batch.tradeability),
            "transition_risk": F.smooth_l1_loss(outputs["transition_risk"], batch.transition_risk),
        }
        if "playbook_logits" in outputs:
            losses["playbook"] = F.cross_entropy(outputs["playbook_logits"], batch.playbook_label)
        total = torch.zeros((), device=outputs["embedding"].device)
        for name, loss in losses.items():
            total = total + float(getattr(self.weights, name)) * loss
        losses["total"] = total
        return losses


@dataclass
class RegimeTrainConfig:
    epochs: int = 5
    batch_size: int = 64
    grad_clip: float = 1.0
    num_workers: int = 0
    log_every: int = 50
    checkpoint: Optional[str] = None
    device: str = "cpu"
    seed: int = 0
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss_weights: RegimeLossWeights = field(default_factory=RegimeLossWeights)


def _move_batch(batch: RegimeSupervisionBatch, device: torch.device) -> RegimeSupervisionBatch:
    return RegimeSupervisionBatch(
        x=batch.x.to(device, non_blocking=True),
        macro=batch.macro.to(device, non_blocking=True),
        meso=batch.meso.to(device, non_blocking=True),
        risk_mode=batch.risk_mode.to(device, non_blocking=True),
        tradeability=batch.tradeability.to(device, non_blocking=True),
        transition_risk=batch.transition_risk.to(device, non_blocking=True),
        forward_return=batch.forward_return.to(device, non_blocking=True),
        realized_vol=batch.realized_vol.to(device, non_blocking=True),
        max_drawdown=batch.max_drawdown.to(device, non_blocking=True),
        playbook_label=batch.playbook_label.to(device, non_blocking=True),
        playbook_scores=batch.playbook_scores.to(device, non_blocking=True),
        reports=batch.reports,
        outcomes=batch.outcomes,
        meta=batch.meta,
    )


def _accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == target).float().mean().item())


class RegimeEncoderTrainer:
    """Train and evaluate a RegimeEncoder on regime supervision data."""

    def __init__(
        self,
        model: RegimeEncoder,
        cfg: Optional[RegimeTrainConfig] = None,
        *,
        loss: RegimeEncoderLoss | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg or RegimeTrainConfig()
        self.device = torch.device(self.cfg.device)
        self.model.to(self.device)
        self.loss = loss or RegimeEncoderLoss(self.cfg.loss_weights)
        self.loss.to(self.device)
        self.opt = build_optimizer(self.model, self.cfg.optim)
        self.sched = build_scheduler(self.opt, self.cfg.optim)
        self._step = 0

    def fit(
        self,
        train_ds: RegimeSupervisionDataset,
        val_ds: Optional[RegimeSupervisionDataset] = None,
    ) -> dict:
        torch.manual_seed(int(self.cfg.seed))
        loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            collate_fn=regime_supervision_collate,
            drop_last=len(train_ds) >= self.cfg.batch_size,
        )
        history: list[dict] = []
        for epoch in range(self.cfg.epochs):
            self.model.train()
            totals: dict[str, float] = {}
            n = 0
            for it, batch in enumerate(loader):
                batch_d = _move_batch(batch, self.device)
                out = self.model(batch_d.x)
                loss_d = self.loss(out, batch_d)
                self.opt.zero_grad(set_to_none=True)
                loss_d["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.opt.step()
                if self.sched is not None:
                    self.sched.step()
                self._step += 1
                bs = batch_d.x.size(0)
                n += bs
                for key, val in loss_d.items():
                    totals[key] = totals.get(key, 0.0) + float(val.item()) * bs
                if self.cfg.log_every and (it + 1) % self.cfg.log_every == 0:
                    logger.info(
                        "regime epoch=%d iter=%d step=%d loss=%.5f",
                        epoch, it, self._step, totals["total"] / max(1, n),
                    )
            row = {"epoch": epoch, "step": self._step}
            row.update({k: v / max(1, n) for k, v in totals.items()})
            if val_ds is not None:
                row["val"] = self.evaluate(val_ds)
            history.append(row)
        if self.cfg.checkpoint:
            self.save(self.cfg.checkpoint)
        return {"history": history, "final_step": self._step}

    @torch.no_grad()
    def evaluate(self, ds: RegimeSupervisionDataset) -> dict:
        loader = DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=regime_supervision_collate,
        )
        self.model.eval()
        totals: dict[str, float] = {}
        n = 0
        for batch in loader:
            batch_d = _move_batch(batch, self.device)
            out = self.model(batch_d.x)
            loss_d = self.loss(out, batch_d)
            bs = batch_d.x.size(0)
            n += bs
            for key, val in loss_d.items():
                totals[key] = totals.get(key, 0.0) + float(val.item()) * bs
            totals["macro_acc"] = totals.get("macro_acc", 0.0) + _accuracy(out["macro_logits"], batch_d.macro) * bs
            totals["meso_acc"] = totals.get("meso_acc", 0.0) + _accuracy(out["meso_logits"], batch_d.meso) * bs
            totals["risk_mode_acc"] = totals.get("risk_mode_acc", 0.0) + _accuracy(out["risk_logits"], batch_d.risk_mode) * bs
            if "playbook_logits" in out:
                totals["playbook_acc"] = totals.get("playbook_acc", 0.0) + _accuracy(out["playbook_logits"], batch_d.playbook_label) * bs
        return {k: v / max(1, n) for k, v in totals.items()}

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "model_config": asdict(self.model.cfg),
            "train_config": asdict(self.cfg),
            "step": self._step,
        }, p)
        logger.info("regime encoder checkpoint saved to %s", p)


__all__ = [
    "RegimeEncoderLoss",
    "RegimeEncoderTrainer",
    "RegimeLossWeights",
    "RegimeTrainConfig",
]
