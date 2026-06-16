"""CVaR-Constrained PPO via a Lagrangian dual multiplier.

The standard PPO trainer optimises expected return. For trading we
also want to control *tail* risk — the average loss in the worst
``alpha``-fraction of episodes. This module adds a Lagrangian
constraint ``-CVaR_alpha <= threshold`` to the PPO objective.

The full loss for the policy is::

    L(pi, lambda) = L_ppo - lambda * max(0, -CVaR_alpha - threshold)

and ``lambda`` is updated after each rollout by *dual ascent*::

    lambda <- clip(lambda + lr_lambda * violation, 0, lambda_max)

where ``violation = max(0, -CVaR_alpha - threshold)`` is computed
on the (non-differentiable) per-episode returns of the rollout.

The :class:`CVaRPPOConfig` is a drop-in extension of
:class:`zhisa.training.s4_rl.PPOConfig` — all original fields keep
their defaults. The :class:`CVaRPPOTrainer` is a drop-in extension
of :class:`zhisa.training.s4_rl.PPOTrainer`; only ``fit()`` is
overridden to add the dual update and the CVaR penalty.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from zhisa.env.trading_env import TradingEnv
from zhisa.models.policy import PolicyNetwork
from zhisa.risk.cvar import cvar_numpy, cvar_torch
from zhisa.training.s4_rl import (
    PPOConfig,
    PPOTrainer,
    RolloutBuffer,
    Transition,
    compute_gae,
    ppo_loss,
)
from zhisa.utils.logging import get_logger
from zhisa.utils.timing import Timer

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CVaRPPOConfig(PPOConfig):
    """Extends :class:`PPOConfig` with CVaR-constraint parameters."""

    cvar_alpha: float = 0.1
    cvar_threshold: float = 0.1
    cvar_lambda_init: float = 1.0
    cvar_lambda_lr: float = 0.05
    cvar_lambda_max: float = 100.0
    cvar_warmup_iters: int = 0
    n_iterations: int = 10

    def __post_init__(self) -> None:
        if not 0.0 < self.cvar_alpha <= 1.0:
            raise ValueError(f"cvar_alpha must be in (0, 1], got {self.cvar_alpha}")
        if self.cvar_threshold < 0.0:
            raise ValueError(f"cvar_threshold must be >= 0, got {self.cvar_threshold}")
        if self.cvar_lambda_lr <= 0.0:
            raise ValueError(f"cvar_lambda_lr must be positive, got {self.cvar_lambda_lr}")


# ---------------------------------------------------------------------------
# Episode-return helper
# ---------------------------------------------------------------------------


def _per_episode_returns(rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
    """Group rewards by episode and return the per-episode sums."""
    if rewards.size == 0:
        return np.zeros(0, dtype=np.float32)
    out: list[float] = []
    cur = 0.0
    for r, d in zip(rewards, dones):
        cur += float(r)
        if d > 0.5:
            out.append(cur)
            cur = 0.0
    if cur != 0.0 or not out:
        out.append(cur)
    return np.array(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class CVaRPPOTrainer(PPOTrainer):
    """PPO with a CVaR risk constraint enforced via a Lagrangian dual.

    Behaviour matches :class:`PPOTrainer` when all CVaR fields are at
    their defaults *and* the dual multiplier stays at zero (which it
    does if the constraint is never violated).
    """

    def __init__(
        self,
        model: PolicyNetwork,
        cfg: Optional[CVaRPPOConfig] = None,
    ) -> None:
        cfg = cfg or CVaRPPOConfig()
        super().__init__(model, cfg)
        self.cfg: CVaRPPOConfig = cfg
        self.lambda_cvar: float = float(cfg.cvar_lambda_init)
        self.cvar_history: list[dict] = []

    # ------------------------------------------------------------------
    # PPO update with CVaR penalty
    # ------------------------------------------------------------------

    def _cvar_ppo_update(
        self,
        buf: RolloutBuffer,
        ep_returns_t: torch.Tensor,
    ) -> dict:
        """PPO update with CVaR constraint enforced via Advantage Penalization on tail episodes."""
        cfg = self.cfg
        if len(buf) == 0:
            return {
                "policy": 0.0, "value": 0.0, "entropy": 0.0, "total": 0.0,
                "cvar_penalty": 0.0, "cvar_value": 0.0,
            }

        stacked = buf.stack_tensors()
        rewards = stacked["reward"]
        values = stacked["value"]
        dones = stacked["done"]
        advantages, returns = compute_gae(
            rewards, values, dones, last_value=0.0,
            gamma=cfg.gamma, lam=cfg.gae_lambda,
        )
        # -----------------------------------------------------------
        # CVaR Advantage Penalization (The Math Fix)
        # -----------------------------------------------------------
        ep_returns_np = ep_returns_t.cpu().numpy()
        var_threshold = float(np.percentile(ep_returns_np, cfg.cvar_alpha * 100))
        
        T = len(rewards)
        step_to_ep_return = np.zeros(T, dtype=np.float32)
        ep_idx = 0
        for t in range(T):
            step_to_ep_return[t] = ep_returns_np[ep_idx]
            if dones[t] > 0.5 and ep_idx < len(ep_returns_np) - 1:
                ep_idx += 1
                
        # Penalize steps belonging to tail episodes that violate the threshold
        violation_mask = (step_to_ep_return <= var_threshold) & (step_to_ep_return < -cfg.cvar_threshold)
        if np.any(violation_mask) and self.lambda_cvar > 0:
            advantages[violation_mask] -= float(self.lambda_cvar)

        to_t = lambda a: (  # noqa: E731
            a.to(self.device, non_blocking=True)
            if torch.is_tensor(a)
            else torch.from_numpy(a).to(self.device, non_blocking=True)
        )
        adv_t = to_t(advantages)
        ret_t = to_t(returns)
        old_logp_t = to_t(stacked["log_prob"])
        action_t = to_t(stacked["action"])
        chart_t = to_t(stacked["chart"]).float()
        num_t = to_t(stacked["numeric"]).float()
        ctx_t = to_t(stacked["context"]).float()

        has_history = "history" in stacked
        if has_history:
            history_t = to_t(stacked["history"]).float()

        # For logging purposes only (the actual penalty is in adv_t)
        with torch.no_grad():
            cvar_value_const = cvar_torch(ep_returns_t, cfg.cvar_alpha)
        cvar_penalty_const = F.relu(-cvar_value_const - cfg.cvar_threshold)

        stats = {
            "policy": [], "value": [], "entropy": [], "total": [],
            "cvar_penalty": [], "cvar_value": [],
        }
        for epoch in range(cfg.n_epochs):
            for idx in buf.minibatch_indices(cfg.minibatch_size, self._rng):
                hist_mb = history_t[idx] if has_history else None
                out = self.model(
                    chart=chart_t[idx],
                    numeric=num_t[idx],
                    context=ctx_t[idx],
                    history=hist_mb
                )
                logits = out["policy_logits"]
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(action_t[idx])
                entropy = dist.entropy()
                values_pred = out["value"]
                losses = ppo_loss(
                    new_log_probs=new_logp,
                    old_log_probs=old_logp_t[idx],
                    advantages=adv_t[idx],
                    values=values_pred,
                    returns=ret_t[idx],
                    entropy=entropy,
                    clip_ratio=cfg.clip_ratio,
                    value_coef=cfg.value_coef,
                    entropy_coef=cfg.entropy_coef,
                )
                # CVaR penalty is already applied to advantages, so we just use PPO total
                total = losses["total"]
                if not torch.isfinite(total):
                    logger.warning("cvar-ppo step %d: non-finite loss, skipping", self._step)
                    continue
                self.opt.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.opt.step()
                self._step += 1
                stats["policy"].append(float(losses["policy"].item()))
                stats["value"].append(float(losses["value"].item()))
                stats["entropy"].append(float(losses["entropy"].item()))
                stats["total"].append(float(total.item()))
                stats["cvar_penalty"].append(float(cvar_penalty_const.item()))
                stats["cvar_value"].append(float(cvar_value_const.item()))
                with torch.no_grad():
                    kl = (old_logp_t[idx] - new_logp).mean().item()
                if abs(kl) > cfg.target_kl:
                    logger.info("cvar-ppo early-stop at epoch %d: KL=%.4f", epoch, kl)
                    break
            else:
                continue
            break
        if not stats["total"]:
            return {
                "policy": 0.0, "value": 0.0, "entropy": 0.0, "total": 0.0,
                "cvar_penalty": 0.0, "cvar_value": 0.0,
            }
        return {k: float(np.mean(v)) for k, v in stats.items()}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> dict:
        cfg = self.cfg
        env = TradingEnv(df, cfg=cfg.env_cfg)
        history: list[dict] = []
        timer = Timer()
        for it in range(cfg.n_iterations):
            timer.start()
            buf, rollout_stats = self._collect_rollout(env)
            stacked = buf.stack_tensors()
            ep_returns_np = _per_episode_returns(stacked["reward"], stacked["done"])
            cvar_value = cvar_numpy(ep_returns_np, cfg.cvar_alpha)
            violation = max(0.0, -cvar_value - cfg.cvar_threshold)
            mean_ep_return = float(np.mean(rollout_stats["ep_returns"])) if rollout_stats["ep_returns"] else 0.0
            mean_ep_return = float(np.clip(mean_ep_return, -1e6, 1e6))
            if it >= cfg.cvar_warmup_iters:
                self.lambda_cvar = float(np.clip(
                    self.lambda_cvar + cfg.cvar_lambda_lr * violation,
                    0.0, cfg.cvar_lambda_max,
                ))
            ep_returns_t = torch.from_numpy(ep_returns_np).to(self.device)
            losses = self._cvar_ppo_update(buf, ep_returns_t)
            timer.stop()
            entry = {
                "iteration": it,
                "n_episodes": len(rollout_stats["ep_returns"]),
                "rollout_steps": len(buf),
                "mean_return": mean_ep_return,
                "cvar": float(cvar_value),
                "cvar_violation": float(violation),
                "lambda_cvar": float(self.lambda_cvar),
                "policy_loss": losses["policy"],
                "value_loss": losses["value"],
                "entropy": losses["entropy"],
                "total_loss": losses["total"],
                "cvar_penalty": losses.get("cvar_penalty", 0.0),
                "elapsed_s": timer.elapsed,
            }
            history.append(entry)
            self.cvar_history.append(entry)
            if (it + 1) % cfg.log_every == 0:
                logger.info(
                    "cvar-ppo it=%d episodes=%d steps=%d mean_return=%.4f "
                    "cvar=%.4f violation=%.4f lambda=%.4f policy=%.4f value=%.4f ent=%.4f total=%.4f",
                    it, len(rollout_stats["ep_returns"]), len(buf),
                    mean_ep_return, cvar_value, violation, self.lambda_cvar,
                    losses["policy"], losses["value"],
                    losses["entropy"], losses["total"],
                )
                if cfg.checkpoint and (it + 1) % 10 == 0:
                    inter_path = str(cfg.checkpoint).replace(".pt", f"_iter_{it+1}.pt")
                    self.save(inter_path)
            timer.reset()
        if cfg.checkpoint:
            self.save(cfg.checkpoint)
        return {"history": history, "cvar_history": list(self.cvar_history)}

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cfg_dict = self.model.cfg.__dict__.copy()
        if "vision_channels" in cfg_dict and isinstance(cfg_dict["vision_channels"], tuple):
            cfg_dict["vision_channels"] = list(cfg_dict["vision_channels"])
        torch.save({
            "model": self.model.state_dict(),
            "config": cfg_dict,
            "model_config": cfg_dict,  # canonical name
            "ppo_config": self.cfg.__dict__,
            "lambda_cvar": self.lambda_cvar,
            "cvar_history": self.cvar_history,
            "checkpoint_meta": {
                "stage": "s4_cvar_ppo",
                "trading_policy_ready": True,
                "policy_head_trained": True,
                "policy_training": "cvar_constrained_ppo",
            },
        }, p)
        logger.info("cvar-ppo checkpoint saved to %s", p)


__all__ = [
    "CVaRPPOConfig",
    "CVaRPPOTrainer",
    "_per_episode_returns",
]
