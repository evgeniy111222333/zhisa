"""Decision Transformer (offline RL via sequence modelling).

The DT consumes fixed-length windows of ``(return-to-go, state, action)``
triples produced by :class:`zhisa.data.trajectory.TrajectoryWindowDataset`
and is trained to predict the next action via cross-entropy on the
*last* token of each window. This is the canonical formulation
(Chen et al. 2021) but specialised for discrete action spaces.

The state embedding is **not** produced by the DT — it is computed
upfront by :func:`embed_trajectories` using a frozen
:class:`PolicyNetwork`. The DT only owns the per-step Transformer
over the token stream, the action head, and the optional
return-to-go regression head.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from zhisa.data.trajectory import Trajectory, TrajectoryWindowDataset
from zhisa.models.policy import PolicyConfig, PolicyNetwork
from zhisa.utils.logging import get_logger
from zhisa.utils.seeding import set_seed


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DecisionTransformerConfig:
    """Hyper-parameters for the Decision Transformer body."""

    state_dim: int = 128
    n_actions: int = 9
    context_length: int = 8
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    max_timestep: int = 1024
    max_rtg_clip: float = 10.0

    def __post_init__(self) -> None:
        if self.context_length <= 0:
            raise ValueError(f"context_length must be positive, got {self.context_length}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )


@dataclass
class DTConfig:
    """Outer training config (optim, schedule, IO)."""

    state_dim: int = 128
    n_actions: int = 9
    context_length: int = 8
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 32
    epochs: int = 5
    rtg_loss_weight: float = 0.0
    max_rtg_clip: float = 10.0
    grad_clip_norm: float = 1.0
    log_every: int = 1
    seed: int = 0
    device: str = "cpu"
    # Whether to print epoch summaries
    verbose: bool = True


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class DecisionTransformer(nn.Module):
    """A small causal Transformer over (rtg, state, action, t) tokens."""

    def __init__(self, cfg: DecisionTransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.state_proj = nn.Linear(cfg.state_dim, cfg.d_model)
        self.action_emb = nn.Embedding(cfg.n_actions, cfg.d_model)
        self.rtg_proj = nn.Linear(1, cfg.d_model)
        self.timestep_emb = nn.Embedding(cfg.max_timestep, cfg.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.action_head = nn.Linear(cfg.d_model, cfg.n_actions)
        self.rtg_head = nn.Linear(cfg.d_model, 1)

    def forward(
        self,
        state: torch.Tensor,   # (B, T, state_dim)
        rtg: torch.Tensor,     # (B, T)
        action: torch.Tensor,  # (B, T) long
        timesteps: torch.Tensor,  # (B, T) long
        mask: Optional[torch.Tensor] = None,  # (B, T) True = valid
    ) -> dict:
        s = self.state_proj(state)
        a = self.action_emb(action.clamp(min=0))
        r = self.rtg_proj(rtg.unsqueeze(-1))
        t = self.timestep_emb(timesteps.clamp(min=0, max=self.cfg.max_timestep - 1))
        x = s + a + r + t
        T = x.size(1)
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        kpm = (~mask) if mask is not None else None
        out = self.transformer(x, mask=causal, src_key_padding_mask=kpm)
        return {
            "action_logits": self.action_head(out),
            "rtg_pred": self.rtg_head(out).squeeze(-1),
            "hidden": out,
        }

    def predict_action(
        self,
        state: torch.Tensor,
        rtg: torch.Tensor,
        action: torch.Tensor,
        timesteps: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the argmax action at the *last* token of the window."""
        out = self.forward(state, rtg, action, timesteps, mask=mask)
        return out["action_logits"][:, -1, :].argmax(dim=-1)


# ---------------------------------------------------------------------------
# Pre-compute state embeddings using a frozen PolicyNetwork
# ---------------------------------------------------------------------------


def embed_trajectories(
    trajectories: Sequence[Trajectory],
    policy: PolicyNetwork,
    device: str = "cpu",
    batch_size: int = 64,
) -> list[Trajectory]:
    """Annotate each ``obs`` with ``obs["state_emb"]`` via ``policy.encode``.

    Mutates the input trajectories in place and returns them. The
    policy is moved to ``device`` in ``eval()`` mode and is **not**
    updated by this call — gradients are disabled.
    """
    policy = policy.to(device).eval()
    flat_obs: list[dict] = []
    index_map: list[tuple[int, int]] = []
    for ti, traj in enumerate(trajectories):
        for oi, o in enumerate(traj.obs):
            flat_obs.append(o)
            index_map.append((ti, oi))
    if not flat_obs:
        return list(trajectories)
    for start in range(0, len(flat_obs), batch_size):
        batch = flat_obs[start : start + batch_size]
        chart = torch.from_numpy(np.stack([np.asarray(o["chart"], dtype=np.float32) for o in batch])).to(device)
        numeric = torch.from_numpy(np.stack([np.asarray(o["numeric"], dtype=np.float32) for o in batch])).to(device)
        context = torch.from_numpy(np.stack([np.asarray(o["context"], dtype=np.float32) for o in batch])).to(device)
        with torch.no_grad():
            z = policy.encode(chart, numeric, context)
        z_np = z.detach().cpu().numpy().astype(np.float32)
        for k, (ti, oi) in enumerate(index_map[start : start + batch_size]):
            trajectories[ti].obs[oi]["state_emb"] = z_np[k]
    return list(trajectories)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


@dataclass
class DTTrainResult:
    history: list[dict] = field(default_factory=list)
    final_loss: float = 0.0
    n_steps: int = 0


class DecisionTransformerTrainer:
    """Offline trainer for the Decision Transformer body."""

    def __init__(
        self,
        model: DecisionTransformer,
        cfg: DTConfig,
        logger=None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        if cfg.device:
            self.model.to(cfg.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        self.logger = logger or get_logger("zhisa.s6_dt")
        self._global_step = 0
        self.history: list[dict] = []

    def _move_batch(self, batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.cfg.device)
            else:
                out[k] = v
        return out

    def _train_step(self, batch: dict) -> float:
        batch = self._move_batch(batch)
        state = batch["state"]
        rtg = batch["rtg"].clamp(-self.cfg.max_rtg_clip, self.cfg.max_rtg_clip)
        action = batch["action"]
        mask = batch["mask"]
        target_action = batch["target_action"]
        T = state.size(1)
        timesteps = torch.arange(T, device=state.device).unsqueeze(0).expand(state.size(0), -1)
        out = self.model(state, rtg, action, timesteps, mask=mask)
        logits_last = out["action_logits"][:, -1, :]
        loss = F.cross_entropy(logits_last, target_action)
        if self.cfg.rtg_loss_weight > 0.0 and "target_rtg" in batch:
            target_rtg = batch["target_rtg"].clamp(-self.cfg.max_rtg_clip, self.cfg.max_rtg_clip)
            rtg_last = out["rtg_pred"][:, -1]
            loss = loss + self.cfg.rtg_loss_weight * F.mse_loss(rtg_last, target_rtg)
        if not torch.isfinite(loss):
            return float("nan")
        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optimizer.step()
        self._global_step += 1
        return float(loss.detach().cpu().item())

    @torch.no_grad()
    def _eval_loss(self, dataset: TrajectoryWindowDataset) -> float:
        if len(dataset) == 0:
            return float("nan")
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=False)
        losses: list[float] = []
        for batch in loader:
            batch = self._move_batch(batch)
            state = batch["state"]
            rtg = batch["rtg"].clamp(-self.cfg.max_rtg_clip, self.cfg.max_rtg_clip)
            action = batch["action"]
            mask = batch["mask"]
            target_action = batch["target_action"]
            T = state.size(1)
            timesteps = torch.arange(T, device=state.device).unsqueeze(0).expand(state.size(0), -1)
            out = self.model(state, rtg, action, timesteps, mask=mask)
            logits_last = out["action_logits"][:, -1, :]
            l = F.cross_entropy(logits_last, target_action)
            if torch.isfinite(l):
                losses.append(float(l.detach().cpu().item()))
        return float(np.mean(losses)) if losses else float("nan")

    def fit(
        self,
        dataset: TrajectoryWindowDataset,
        val_dataset: Optional[TrajectoryWindowDataset] = None,
    ) -> DTTrainResult:
        if len(dataset) == 0:
            raise ValueError("Cannot fit DT on an empty dataset")
        set_seed(self.cfg.seed)
        loader = DataLoader(
            dataset, batch_size=self.cfg.batch_size, shuffle=True, drop_last=False
        )
        result = DTTrainResult()
        for epoch in range(self.cfg.epochs):
            losses: list[float] = []
            for batch in loader:
                l = self._train_step(batch)
                if not (l != l):  # NaN check
                    losses.append(l)
            mean_loss = float(np.mean(losses)) if losses else float("nan")
            val_loss = (
                self._eval_loss(val_dataset) if val_dataset is not None else float("nan")
            )
            # Guard: if the val set is empty (or all batches produced
            # non-finite loss), fall back to the training loss and
            # emit a one-time warning so the operator can see that the
            # reported val_loss is actually the train loss.
            if val_loss != val_loss:  # NaN
                if not getattr(self, "_val_warned", False):
                    self.logger.warning(
                        "DT: val set is empty or all-NaN; falling back to train_loss "
                        "for val_loss reporting (epoch %d)", epoch,
                    )
                    self._val_warned = True
                val_loss = mean_loss
            entry = {"epoch": epoch, "loss": mean_loss, "val_loss": val_loss}
            self.history.append(entry)
            if self.cfg.verbose and (epoch % max(1, self.cfg.log_every) == 0):
                self.logger.info(
                    "epoch=%d loss=%.4f val_loss=%.4f", epoch, mean_loss, val_loss
                )
        result.history = list(self.history)
        result.final_loss = self.history[-1]["loss"] if self.history else float("nan")
        result.n_steps = self._global_step
        return result

    # ------------------------------------------------------------------ IO
    def save(self, path: str, extra: Optional[dict] = None) -> None:
        payload = {
            "model": self.model.state_dict(),
            "config": {
                "dt": self.model.cfg.__dict__,
                "trainer": self.cfg.__dict__,
            },
            "history": self.history,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @staticmethod
    def load(path: str, map_location: str = "cpu") -> tuple[DecisionTransformer, DTConfig]:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        dt_cfg = DecisionTransformerConfig(**payload["config"]["dt"])
        model = DecisionTransformer(dt_cfg)
        model.load_state_dict(payload["model"])
        trainer_cfg = DTConfig(**payload["config"]["trainer"])
        return model, trainer_cfg


# ---------------------------------------------------------------------------
# Convenience: build a fresh DT from a PolicyConfig
# ---------------------------------------------------------------------------


def build_default_dt(policy_cfg: PolicyConfig, dt_cfg: Optional[DTConfig] = None) -> DecisionTransformer:
    """Build a DT whose ``state_dim`` matches the policy's ``embed_dim``."""
    dt_cfg = dt_cfg or DTConfig()
    dt_cfg.state_dim = int(policy_cfg.embed_dim)
    body_cfg = DecisionTransformerConfig(
        state_dim=dt_cfg.state_dim,
        n_actions=dt_cfg.n_actions,
        context_length=dt_cfg.context_length,
        d_model=dt_cfg.d_model,
        n_heads=dt_cfg.n_heads,
        n_layers=dt_cfg.n_layers,
        dropout=dt_cfg.dropout,
        max_rtg_clip=dt_cfg.max_rtg_clip,
    )
    return DecisionTransformer(body_cfg), dt_cfg


__all__ = [
    "DecisionTransformer",
    "DecisionTransformerConfig",
    "DecisionTransformerTrainer",
    "DTConfig",
    "DTTrainResult",
    "embed_trajectories",
    "build_default_dt",
]
