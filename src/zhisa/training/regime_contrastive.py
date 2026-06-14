"""Self-supervised and outcome-aware contrastive training for RegimeEncoder."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from zhisa.regime.contrastive import (
    RegimeAugmentationConfig,
    RegimePositiveMaskConfig,
    augment_regime_features,
    nt_xent_loss,
    regime_positive_mask,
    supervised_contrastive_loss,
)
from zhisa.regime.dataset import (
    RegimeSupervisionBatch,
    RegimeSupervisionDataset,
    regime_supervision_collate,
)
from zhisa.regime.encoder import RegimeEncoder
from zhisa.training.optim import OptimConfig, build_optimizer, build_scheduler
from zhisa.training.regime_supervised import RegimeEncoderLoss, RegimeLossWeights, _move_batch
from zhisa.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RegimeContrastiveWeights:
    view_consistency: float = 1.0
    outcome_supervised: float = 0.35
    multitask: float = 0.5


@dataclass
class RegimeContrastiveTrainConfig:
    epochs: int = 5
    batch_size: int = 64
    grad_clip: float = 1.0
    num_workers: int = 0
    log_every: int = 50
    checkpoint: Optional[str] = None
    device: str = "cpu"
    seed: int = 0
    temperature: float = 0.12
    optim: OptimConfig = field(default_factory=OptimConfig)
    augmentation: RegimeAugmentationConfig = field(default_factory=RegimeAugmentationConfig)
    positive_mask: RegimePositiveMaskConfig = field(default_factory=RegimePositiveMaskConfig)
    weights: RegimeContrastiveWeights = field(default_factory=RegimeContrastiveWeights)
    multitask_weights: RegimeLossWeights = field(default_factory=RegimeLossWeights)


class RegimeContrastiveTrainer:
    """Pretrain/fine-tune RegimeEncoder embeddings with robust regime views."""

    def __init__(
        self,
        model: RegimeEncoder,
        cfg: Optional[RegimeContrastiveTrainConfig] = None,
        *,
        multitask_loss: RegimeEncoderLoss | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg or RegimeContrastiveTrainConfig()
        self.device = torch.device(self.cfg.device)
        self.model.to(self.device)
        self.multitask_loss = multitask_loss or RegimeEncoderLoss(self.cfg.multitask_weights)
        self.multitask_loss.to(self.device)
        self.opt = build_optimizer(self.model, self.cfg.optim)
        self.sched = build_scheduler(self.opt, self.cfg.optim)
        self._step = 0
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(int(self.cfg.seed))

    def _views(self, batch: RegimeSupervisionBatch) -> tuple[torch.Tensor, torch.Tensor]:
        names = self.model.vectorizer.feature_names
        x_a = augment_regime_features(
            batch.x,
            self.cfg.augmentation,
            feature_names=names,
            generator=self._generator,
        )
        x_b = augment_regime_features(
            batch.x,
            self.cfg.augmentation,
            feature_names=names,
            generator=self._generator,
        )
        return x_a, x_b

    def _losses(self, batch: RegimeSupervisionBatch) -> dict[str, torch.Tensor]:
        x_a, x_b = self._views(batch)
        out_clean = self.model(batch.x)
        out_a = self.model(x_a)
        out_b = self.model(x_b)

        losses: dict[str, torch.Tensor] = {}
        losses["view_consistency"] = nt_xent_loss(
            out_a["embedding"],
            out_b["embedding"],
            temperature=self.cfg.temperature,
        )

        pos_mask = regime_positive_mask(batch, self.cfg.positive_mask)
        if bool(pos_mask.any()):
            z = torch.cat([out_a["embedding"], out_b["embedding"]], dim=0)
            pair_mask = torch.cat(
                [
                    torch.cat([pos_mask, pos_mask], dim=1),
                    torch.cat([pos_mask, pos_mask], dim=1),
                ],
                dim=0,
            )
            pair_mask.fill_diagonal_(False)
            losses["outcome_supervised"] = supervised_contrastive_loss(
                z,
                pair_mask,
                temperature=self.cfg.temperature,
            )
        else:
            losses["outcome_supervised"] = batch.x.new_zeros(())

        multitask = self.multitask_loss(out_clean, batch)
        losses["multitask"] = multitask["total"]

        total = batch.x.new_zeros(())
        total = total + self.cfg.weights.view_consistency * losses["view_consistency"]
        total = total + self.cfg.weights.outcome_supervised * losses["outcome_supervised"]
        total = total + self.cfg.weights.multitask * losses["multitask"]
        losses["total"] = total
        return losses

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
                loss_d = self._losses(batch_d)
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
                        "regime-contrastive epoch=%d iter=%d step=%d loss=%.5f",
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
            losses = self._losses(batch_d)
            bs = batch_d.x.size(0)
            n += bs
            for key, val in losses.items():
                totals[key] = totals.get(key, 0.0) + float(val.item()) * bs
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
        logger.info("regime contrastive checkpoint saved to %s", p)


__all__ = [
    "RegimeContrastiveTrainer",
    "RegimeContrastiveTrainConfig",
    "RegimeContrastiveWeights",
]
