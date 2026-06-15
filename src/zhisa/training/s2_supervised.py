"""S2: Supervised multi-task trainer.

This is the MVP trainer. It uses the labelled ``MarketDataset`` and
trains a ``PolicyNetwork`` with a multi-task loss. It is designed to
be a solid baseline that subsequent phases (SSL pretrain, RL, online)
can build on.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from zhisa.data.dataset import MarketDataset, multimodal_collate
from zhisa.models.policy import PolicyConfig, PolicyNetwork
from zhisa.training.losses import LossWeights, MultiTaskLoss
from zhisa.training.optim import OptimConfig, build_optimizer, build_scheduler
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
    eval_every: int = 0          # 0 = no eval
    checkpoint: Optional[str] = None
    device: str = "cpu"
    seed: int = 0
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
        self.opt = build_optimizer(
            [p for p in model.parameters() if p.requires_grad]
            + [p for p in loss.parameters() if p.requires_grad],
            cfg.optim,
        )
        self.sched = build_scheduler(self.opt, cfg.optim)
        self._step = 0

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
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, collate_fn=multimodal_collate, drop_last=True,
        )
        history: list[dict] = []
        timer = Timer()
        for epoch in range(cfg.epochs):
            self.model.train()
            ep_loss = 0.0
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
                ep_loss += float(loss.item()) * batch_d["chart"].size(0)
                ep_count += batch_d["chart"].size(0)
                if (it + 1) % cfg.log_every == 0:
                    avg = ep_loss / max(1, ep_count)
                    lr = self.opt.param_groups[0]["lr"]
                    logger.info(
                        "epoch=%d iter=%d step=%d loss=%.5f lr=%.2e elapsed=%.1fs",
                        epoch, it, self._step, avg, lr, timer.elapsed,
                    )
            avg = ep_loss / max(1, ep_count)
            timer.stop()
            history.append({"epoch": epoch, "loss": avg, "elapsed_s": timer.elapsed})
            logger.info("epoch %d done in %.1fs, avg_loss=%.5f", epoch, timer.elapsed, avg)
            timer.reset()
            if val_ds is not None and cfg.eval_every and (epoch + 1) % cfg.eval_every == 0:
                val_metrics = self.evaluate(val_ds)
                history[-1]["val"] = val_metrics
        if cfg.checkpoint:
            self.save(cfg.checkpoint)
        return {"history": history, "final_step": self._step}

    @torch.no_grad()
    def evaluate(self, ds: MarketDataset) -> dict:
        loader = DataLoader(
            ds, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.num_workers, collate_fn=multimodal_collate,
        )
        self.model.eval()
        agg: dict = {}
        n = 0
        for batch in loader:
            batch_d = self._move_batch(batch)
            out = self.model(
                chart=batch_d["chart"],
                numeric=batch_d["numeric"],
                context=batch_d["context"],
            )
            losses = self.loss(out, batch_d)
            for k, v in losses.items():
                if k == "total":
                    continue
                agg[k] = agg.get(k, 0.0) + float(v.item()) * batch_d["chart"].size(0)
            n += batch_d["chart"].size(0)
        return {k: v / max(1, n) for k, v in agg.items()}

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
        torch.save({
            "model": self.model.state_dict(),
            "loss": self.loss.state_dict(),
            "config": cfg_dict,
            "model_config": cfg_dict,  # canonical name
            "train_config": self.cfg.__dict__,
            "checkpoint_meta": {
                "stage": "s2_supervised",
                "trading_policy_ready": False,
                "policy_head_trained": False,
                "reason": "S2 trains supervised market heads; use S2b/S4+ for paper trading policies.",
            },
        }, p)
        logger.info("checkpoint saved to %s", p)
