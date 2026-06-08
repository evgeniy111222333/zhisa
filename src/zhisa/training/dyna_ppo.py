"""Dyna-style PPO trainer that learns a :class:`LatentActorCritic` in imagination.

The Dyna loop is the textbook recipe:

1. **Collect** a real episode in :class:`TradingEnv` using a
   behaviour policy and embed it via a frozen
   :class:`PolicyNetwork` (delegated to the caller).
2. **Train** the :class:`WorldModel` on the new (and replayed)
   trajectories.
3. **Imagine** rollouts in :class:`DreamEnv`: sample initial
   ``(z, h)`` pairs from the pool of real states, then roll out
   in latent space for ``H`` steps using the world model and the
   current :class:`LatentActorCritic`.
4. **PPO update** the actor-critic on the imagined trajectories
   using the world-model's predicted reward and a learned value
   baseline.
5. **Repeat** with the policy now updated.

This module owns only steps 3-4; step 1-2 are explicit calls
(see :class:`zhisa.scripts.train_s7`) so the user can mix and
match world models and policies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from zhisa.models.latent_actor_critic import LatentActorCritic
from zhisa.models.world_model import WorldModel
from zhisa.utils.logging import get_logger
from zhisa.utils.seeding import set_seed


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DynaPPOConfig:
    horizon: int = 16
    n_imagined_rollouts: int = 32
    ppo_epochs: int = 4
    ppo_minibatch_size: int = 32
    ppo_clip: float = 0.2
    ppo_entropy_coef: float = 0.01
    ppo_value_coef: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    grad_clip_norm: float = 1.0
    seed: int = 0
    device: str = "cpu"
    verbose: bool = True


@dataclass
class DynaPPOResult:
    history: list[dict] = field(default_factory=list)
    final_imagined_return: float = 0.0
    n_ppo_steps: int = 0


# ---------------------------------------------------------------------------
# Imagined rollout buffer
# ---------------------------------------------------------------------------


@dataclass
class ImaginedBatch:
    z: torch.Tensor            # (N, T, D) current states
    h: torch.Tensor            # (n_layers, N, T, H) recurrent states
    a: torch.Tensor            # (N, T) long
    logp: torch.Tensor         # (N, T) float
    r: torch.Tensor            # (N, T) float
    d: torch.Tensor            # (N, T) float  (1 = terminate)
    z_next: torch.Tensor       # (N, T, D) predicted next states
    v_old: torch.Tensor        # (N, T) float — values before update
    adv: torch.Tensor          # (N, T) float — GAE advantage
    ret: torch.Tensor          # (N, T) float — GAE return

    def size(self) -> int:
        return int(self.z.size(0) * self.z.size(1))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class DynaPPOTrainer:
    """Dyna-style PPO trainer operating purely in imagination."""

    def __init__(
        self,
        world_model: WorldModel,
        actor_critic: LatentActorCritic,
        cfg: DynaPPOConfig,
        logger=None,
    ) -> None:
        self.world_model = world_model.to(cfg.device).eval()
        self.actor_critic = actor_critic.to(cfg.device)
        self.cfg = cfg
        self.optimizer = torch.optim.AdamW(
            actor_critic.parameters(), lr=cfg.learning_rate
        )
        self.logger = logger or get_logger("zhisa.dyna_ppo")
        self.history: list[dict] = []

    # ----------------------------------------------------------------- imagine
    @torch.no_grad()
    def imagine_rollouts(
        self,
        initial_z: torch.Tensor,  # (N, D)
        initial_h: torch.Tensor,  # (n_layers, N, H)
    ) -> ImaginedBatch:
        """Sample actions from the actor-critic and roll them out in the world model.

        Initial states should come from the pool of *real* states
        observed in the market — sampling from a wider distribution
        than the WM has been trained on causes catastrophic
        hallucination.
        """
        device = self.cfg.device
        N = int(initial_z.size(0))
        T = int(self.cfg.horizon)
        D = int(initial_z.size(1))
        L = int(initial_h.size(0))
        H = int(initial_h.size(-1))
        z = initial_z.to(device).unsqueeze(1).expand(N, T, D).contiguous()
        h = initial_h.to(device).unsqueeze(2).expand(L, N, T, H).contiguous()
        # Per-step sampled actions / log-probs.
        a = torch.zeros(N, T, dtype=torch.long, device=device)
        logp = torch.zeros(N, T, device=device)
        for t in range(T):
            z_t = z[:, t]
            h_t = h[:, :, t].contiguous()
            act_t, lp_t = self.actor_critic.act(z_t, deterministic=False)
            a[:, t] = act_t
            logp[:, t] = lp_t
        # Rollout the world model with these actions.
        a_one_seq = a  # (N, T)
        rollout = self.world_model.rollout(z[:, 0], a_one_seq, h0=h[:, :, 0].contiguous())
        z_next_seq = rollout["z_seq"]  # (N, T, D)
        r_seq = rollout["r_seq"]        # (N, T)
        d_seq = rollout["d_seq"]        # (N, T) probabilities
        # Hard done = horizon or d_prob > 0.5
        d = (d_seq > 0.5).float()
        # Forward through actor-critic on the *predicted* next states
        # (used as bootstrap values).
        v_old = self.actor_critic.value(z.reshape(N * T, D)).reshape(N, T)
        v_next = self.actor_critic.value(z_next_seq.reshape(N * T, D)).reshape(N, T)
        # GAE per rollout.
        adv = torch.zeros(N, T, device=device)
        last_adv = torch.zeros(N, device=device)
        for t in reversed(range(T)):
            nonterminal = 1.0 - d[:, t]
            delta = r_seq[:, t] + self.cfg.gamma * v_next[:, t] * nonterminal - v_old[:, t]
            last_adv = delta + self.cfg.gamma * self.cfg.gae_lambda * nonterminal * last_adv
            adv[:, t] = last_adv
        ret = adv + v_old
        return ImaginedBatch(
            z=z, h=h, a=a, logp=logp, r=r_seq, d=d,
            z_next=z_next_seq, v_old=v_old, adv=adv, ret=ret,
        )

    # ----------------------------------------------------------------- PPO
    def _ppo_loss(self, batch: ImaginedBatch) -> dict:
        N, T, D = batch.z.shape
        flat_z = batch.z.reshape(N * T, D)
        flat_a = batch.a.reshape(N * T)
        flat_logp_old = batch.logp.reshape(N * T)
        flat_adv = batch.adv.reshape(N * T)
        flat_ret = batch.ret.reshape(N * T)
        # Normalise advantages
        if flat_adv.numel() > 1:
            flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-6)
        # Minibatch SGD over the flattened (N*T) buffer.
        idx = torch.randperm(N * T, device=flat_z.device)
        mb = max(1, int(self.cfg.ppo_minibatch_size))
        total_policy = 0.0
        total_value = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        n_mb = 0
        for start in range(0, N * T, mb):
            mb_idx = idx[start : start + mb]
            logits, value = self.actor_critic(flat_z[mb_idx])
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(flat_a[mb_idx])
            ratio = torch.exp(logp - flat_logp_old[mb_idx])
            adv = flat_adv[mb_idx]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.cfg.ppo_clip, 1.0 + self.cfg.ppo_clip) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value, flat_ret[mb_idx])
            entropy = dist.entropy().mean()
            loss = (
                policy_loss
                + self.cfg.ppo_value_coef * value_loss
                - self.cfg.ppo_entropy_coef * entropy
            )
            if not torch.isfinite(loss):
                continue
            self.optimizer.zero_grad()
            loss.backward()
            if self.cfg.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.grad_clip_norm)
            self.optimizer.step()
            total_policy += float(policy_loss.detach().item())
            total_value += float(value_loss.detach().item())
            total_entropy += float(entropy.detach().item())
            total_loss += float(loss.detach().item())
            n_mb += 1
        if n_mb == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "loss": 0.0}
        return {
            "policy_loss": total_policy / n_mb,
            "value_loss": total_value / n_mb,
            "entropy": total_entropy / n_mb,
            "loss": total_loss / n_mb,
        }

    # ----------------------------------------------------------------- Dyna step
    def update(
        self,
        initial_z: torch.Tensor,
        initial_h: torch.Tensor,
    ) -> dict:
        """Run one Dyna update: imagine rollouts, then ``ppo_epochs`` PPO passes."""
        set_seed(self.cfg.seed)
        batch = self.imagine_rollouts(initial_z, initial_h)
        last_metrics: dict = {}
        for ep in range(self.cfg.ppo_epochs):
            metrics = self._ppo_loss(batch)
            last_metrics = metrics
        ret_mean = float(batch.ret.mean().item())
        adv_std = float(batch.adv.std().item()) if batch.adv.numel() > 1 else 0.0
        n_steps = batch.size()
        summary = {
            "n_steps": n_steps,
            "imagined_return": ret_mean,
            "adv_std": adv_std,
            **last_metrics,
        }
        self.history.append(summary)
        if self.cfg.verbose:
            self.logger.info(
                "ppo update | steps=%d ret=%.3f policy=%.3f value=%.3f ent=%.3f",
                n_steps, ret_mean,
                last_metrics.get("policy_loss", 0.0),
                last_metrics.get("value_loss", 0.0),
                last_metrics.get("entropy", 0.0),
            )
        return summary

    # ----------------------------------------------------------------- IO
    def save(self, path: str, extra: Optional[dict] = None) -> None:
        payload = {
            "actor_critic": self.actor_critic.state_dict(),
            "world_model": self.world_model.state_dict(),
            "config": {
                "dyna": self.cfg.__dict__,
                "ac": self.actor_critic.cfg.__dict__,
            },
            "history": self.history,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @staticmethod
    def load(path: str, map_location: str = "cpu") -> tuple[LatentActorCritic, DynaPPOConfig, WorldModel]:
        from zhisa.models.latent_actor_critic import LatentActorCritic, LatentActorCriticConfig
        from zhisa.models.world_model import WorldModel, WorldModelConfig
        payload = torch.load(path, map_location=map_location, weights_only=False)
        ac_cfg = LatentActorCriticConfig(**payload["config"]["ac"])
        ac = LatentActorCritic(ac_cfg)
        ac.load_state_dict(payload["actor_critic"])
        dyna_cfg = DynaPPOConfig(**payload["config"]["dyna"])
        wm_cfg = WorldModelConfig(**payload["config"].get("wm", {
            "state_dim": ac_cfg.state_dim, "n_actions": ac_cfg.n_actions,
        }))
        wm = WorldModel(wm_cfg)
        if "world_model" in payload:
            try:
                wm.load_state_dict(payload["world_model"])
            except Exception:
                pass
        return ac, dyna_cfg, wm


__all__ = [
    "DynaPPOConfig",
    "DynaPPOResult",
    "DynaPPOTrainer",
    "ImaginedBatch",
]
