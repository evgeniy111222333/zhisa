"""World Model trainer.

Trains :class:`zhisa.models.world_model.WorldModel` on
:class:`Trajectory` data using the standard three-loss objective
introduced in :class:`zhisa.models.latent_dynamics.LatentDynamics`:

* **state MSE** — predict ``z_{t+1}`` from ``(z_t, a_t, h_t)``
* **reward MSE** — predict the realised reward at ``t+1``
* **done BCE** — predict episode termination

The encoder is **frozen**: state embeddings are pre-computed
upstream by :func:`zhisa.training.decision_transformer.embed_trajectories`
and stored in ``obs["state_emb"]``. This avoids the moving-target
problem where an evolving encoder destabilises the dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from zhisa.data.trajectory import Trajectory, TrajectoryWindowDataset, compute_returns_to_go
from zhisa.models.world_model import WorldModel
from zhisa.utils.logging import get_logger
from zhisa.utils.seeding import set_seed


# ---------------------------------------------------------------------------
# Dataset of (z_t, a_t, z_{t+1}, r_t, d_t) tuples
# ---------------------------------------------------------------------------


class WorldModelDataset(Dataset):
    """Yield single-step transition tuples for world-model training.

    A trajectory of length ``T`` yields ``T`` transitions (with the
    final transition's ``done`` set to True). All inputs are
    pre-computed ``state_emb`` (frozen encoder output).
    """

    def __init__(self, trajectories: list[Trajectory]) -> None:
        self._items: list[dict] = []
        for traj in trajectories:
            if traj.is_empty():
                continue
            for t in range(len(traj)):
                emb = traj.obs[t].get("state_emb")
                if emb is None:
                    raise ValueError(
                        "Trajectory observations must have 'state_emb' (run "
                        "zhisa.training.embed_trajectories first)."
                    )
                next_emb = (
                    traj.obs[t + 1]["state_emb"]
                    if t + 1 < len(traj)
                    else traj.obs[t]["state_emb"]
                )
                self._items.append({
                    "z": np.asarray(emb, dtype=np.float32),
                    "a": int(traj.actions[t]),
                    "z_next": np.asarray(next_emb, dtype=np.float32),
                    "r": float(traj.rewards[t]) if t < len(traj.rewards) else 0.0,
                    "d": bool(traj.dones[t]) if t < len(traj.dones) else False,
                })

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        it = self._items[idx]
        return {
            "z": torch.from_numpy(it["z"]).float(),
            "a": torch.tensor(it["a"], dtype=torch.long),
            "z_next": torch.from_numpy(it["z_next"]).float(),
            "r": torch.tensor(it["r"], dtype=torch.float32),
            "d": torch.tensor(float(it["d"]), dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class WorldModelTrainerConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    epochs: int = 5
    state_loss_weight: float = 1.0
    reward_loss_weight: float = 1.0
    done_loss_weight: float = 0.1
    grad_clip_norm: float = 1.0
    reward_clip: float = 5.0
    log_every: int = 1
    seed: int = 0
    device: str = "cpu"
    verbose: bool = True


# ---------------------------------------------------------------------------
# Result + Trainer
# ---------------------------------------------------------------------------


@dataclass
class WorldModelTrainResult:
    history: list[dict] = field(default_factory=list)
    final_state_mse: float = 0.0
    final_reward_mse: float = 0.0
    final_done_bce: float = 0.0


class WorldModelTrainer:
    """Trains a :class:`WorldModel` on a list of :class:`Trajectory`."""

    def __init__(
        self,
        model: WorldModel,
        cfg: WorldModelTrainerConfig,
        logger=None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        if cfg.device:
            self.model.to(cfg.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        self.logger = logger or get_logger("zhisa.world_model")
        self.history: list[dict] = []

    def _move(self, batch: dict) -> dict:
        return {k: (v.to(self.cfg.device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()}

    def _step(self, batch: dict) -> dict:
        batch = self._move(batch)
        z = batch["z"]
        a = batch["a"]
        z_next_target = batch["z_next"]
        r_target = batch["r"].clamp(-self.cfg.reward_clip, self.cfg.reward_clip)
        d_target = batch["d"]
        out = self.model.step(z, a)
        z_pred = out["z_next"]
        r_pred = out["r_pred"]
        d_logit = out["d_logit"]
        state_mse = F.mse_loss(z_pred, z_next_target)
        reward_mse = F.mse_loss(r_pred, r_target)
        done_bce = F.binary_cross_entropy_with_logits(d_logit, d_target)
        loss = (
            self.cfg.state_loss_weight * state_mse
            + self.cfg.reward_loss_weight * reward_mse
            + self.cfg.done_loss_weight * done_bce
        )
        return {
            "loss": loss,
            "state_mse": state_mse.detach(),
            "reward_mse": reward_mse.detach(),
            "done_bce": done_bce.detach(),
        }

    def _train_step(self, batch: dict) -> dict:
        out = self._step(batch)
        loss = out["loss"]
        if not torch.isfinite(loss):
            return {k: (float("nan") if not torch.is_tensor(v) else float("nan")) for k, v in out.items()}
        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optimizer.step()
        return {k: (v.item() if torch.is_tensor(v) else v) for k, v in out.items()}

    @torch.no_grad()
    def _eval(self, dataset: WorldModelDataset) -> dict:
        if len(dataset) == 0:
            return {"state_mse": float("nan"), "reward_mse": float("nan"), "done_bce": float("nan")}
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=False)
        ss, rs, ds, n = 0.0, 0.0, 0.0, 0
        for batch in loader:
            out = self._step(batch)
            ss += float(out["state_mse"].item()) * batch["z"].size(0)
            rs += float(out["reward_mse"].item()) * batch["z"].size(0)
            ds += float(out["done_bce"].item()) * batch["z"].size(0)
            n += batch["z"].size(0)
        return {
            "state_mse": ss / max(n, 1),
            "reward_mse": rs / max(n, 1),
            "done_bce": ds / max(n, 1),
        }

    def fit(
        self,
        dataset: WorldModelDataset,
        val_dataset: Optional[WorldModelDataset] = None,
    ) -> WorldModelTrainResult:
        if len(dataset) == 0:
            raise ValueError("Cannot fit WorldModel on an empty dataset")
        set_seed(self.cfg.seed)
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)
        result = WorldModelTrainResult()
        for epoch in range(self.cfg.epochs):
            agg = {"loss": [], "state_mse": [], "reward_mse": [], "done_bce": []}
            for batch in loader:
                out = self._train_step(batch)
                for k, v in out.items():
                    if not (v != v):  # NaN check
                        agg[k].append(float(v))
            mean = {k: float(np.mean(v)) if v else float("nan") for k, v in agg.items()}
            val = self._eval(val_dataset) if (val_dataset is not None and len(val_dataset) > 0) else {
                "state_mse": float("nan"), "reward_mse": float("nan"), "done_bce": float("nan")
            }
            entry = {**{f"train_{k}": v for k, v in mean.items()},
                     **{f"val_{k}": v for k, v in val.items()}, "epoch": epoch}
            self.history.append(entry)
            if self.cfg.verbose and (epoch % max(1, self.cfg.log_every) == 0):
                self.logger.info(
                    "epoch=%d loss=%.4f state_mse=%.4f reward_mse=%.4f done_bce=%.4f val_state_mse=%.4f",
                    epoch, mean["loss"], mean["state_mse"], mean["reward_mse"],
                    mean["done_bce"], val["state_mse"],
                )
        result.history = list(self.history)
        if self.history:
            def _fallback(metric: str) -> float:
                val = self.history[-1].get(f"val_{metric}", float("nan"))
                if val != val:  # NaN check
                    return float(self.history[-1].get(f"train_{metric}", 0.0))
                return float(val)
            result.final_state_mse = _fallback("state_mse")
            result.final_reward_mse = _fallback("reward_mse")
            result.final_done_bce = _fallback("done_bce")
        return result

    # ------------------------------------------------------------------ IO
    def save(self, path: str, extra: Optional[dict] = None) -> None:
        payload = {
            "model": self.model.state_dict(),
            "config": {
                "wm": self.model.cfg.__dict__,
                "trainer": self.cfg.__dict__,
            },
            "history": self.history,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @staticmethod
    def load(path: str, map_location: str = "cpu") -> tuple[WorldModel, WorldModelTrainerConfig]:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        from zhisa.models.world_model import WorldModelConfig
        wm_cfg = WorldModelConfig(**payload["config"]["wm"])
        model = WorldModel(wm_cfg)
        model.load_state_dict(payload["model"])
        trainer_cfg = WorldModelTrainerConfig(**payload["config"]["trainer"])
        return model, trainer_cfg


__all__ = [
    "WorldModelDataset",
    "WorldModelTrainer",
    "WorldModelTrainerConfig",
    "WorldModelTrainResult",
]
