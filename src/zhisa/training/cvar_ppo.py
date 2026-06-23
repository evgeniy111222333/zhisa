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

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

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
    _chart_tensor,
    _release_rollout_memory,
    compute_gae,
    approximate_kl,
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
        if self.cvar_lambda_max <= 0.0:
            raise ValueError(f"cvar_lambda_max must be positive, got {self.cvar_lambda_max}")
        if self.cvar_lambda_init < 0.0 or self.cvar_lambda_init > self.cvar_lambda_max:
            raise ValueError("cvar_lambda_init must be within [0, cvar_lambda_max]")
        if self.cvar_warmup_iters < 0:
            raise ValueError("cvar_warmup_iters must be non-negative")


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


def _tail_loss_excess(
    episode_returns: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Loss severity beyond the allowed CVaR threshold for tail episodes."""
    return np.maximum(0.0, -np.asarray(episode_returns, dtype=np.float32) - threshold)


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
        self._best_val_feasible = False

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
        # Likelihood-ratio CVaR gradient estimate. When the rollout violates
        # the constraint, actions from tail-loss episodes receive a downside
        # adjustment proportional to loss severity and 1 / alpha.
        # -----------------------------------------------------------
        ep_returns_np = ep_returns_t.cpu().numpy()
        cvar_value = cvar_numpy(ep_returns_np, cfg.cvar_alpha)
        var_threshold = float(np.percentile(ep_returns_np, cfg.cvar_alpha * 100))
        
        T = len(rewards)
        step_to_ep_return = np.zeros(T, dtype=np.float32)
        ep_idx = 0
        for t in range(T):
            step_to_ep_return[t] = ep_returns_np[ep_idx]
            if dones[t] > 0.5 and ep_idx < len(ep_returns_np) - 1:
                ep_idx += 1
                
        rollout_violates = -cvar_value > cfg.cvar_threshold
        tail_mask = step_to_ep_return <= var_threshold
        if rollout_violates and np.any(tail_mask) and self.lambda_cvar > 0:
            tail_losses = _tail_loss_excess(
                step_to_ep_return[tail_mask],
                cfg.cvar_threshold,
            )
            advantages[tail_mask] -= (
                float(self.lambda_cvar) * tail_losses / max(cfg.cvar_alpha, 1e-6)
            )

        to_t = lambda a: (  # noqa: E731
            a.to(self.device, non_blocking=True)
            if torch.is_tensor(a)
            else torch.from_numpy(a).to(self.device, non_blocking=True)
        )
        adv_t = to_t(advantages)
        ret_t = to_t(returns)
        old_logp_t = to_t(stacked["log_prob"])
        action_t = to_t(stacked["action"])
        chart_t = _chart_tensor(stacked["chart"], self.device)
        num_t = to_t(stacked["numeric"]).float()
        ctx_t = to_t(stacked["context"]).float()

        has_history = "history" in stacked
        if has_history:
            history_t = to_t(stacked["history"]).float()

        # For logging purposes only (the actual penalty is in adv_t)
        with torch.no_grad():
            cvar_value_const = cvar_torch(ep_returns_t, cfg.cvar_alpha)
        cvar_penalty_const = self.lambda_cvar * F.relu(
            -cvar_value_const - cfg.cvar_threshold
        )

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
                    value_loss_scale=cfg.value_loss_scale,
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
                    kl = approximate_kl(old_logp_t[idx], new_logp).item()
                if kl > cfg.target_kl:
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

    def fit(
        self,
        df: pd.DataFrame | Sequence[pd.DataFrame],
        val_df: Optional[pd.DataFrame | Sequence[pd.DataFrame]] = None,
    ) -> dict:
        cfg = self.cfg
        frames = list(df) if isinstance(df, Sequence) and not isinstance(df, pd.DataFrame) else [df]
        envs = [TradingEnv(frame, cfg=cfg.env_cfg) for frame in frames]
        val_envs: list[TradingEnv] = []
        if val_df is not None:
            val_frames = list(val_df) if isinstance(val_df, Sequence) and not isinstance(val_df, pd.DataFrame) else [val_df]
            val_envs = [TradingEnv(frame, cfg=cfg.env_cfg) for frame in val_frames]
        history: list[dict] = []
        timer = Timer()
        for it in range(self._iteration, cfg.n_iterations):
            is_best = False
            timer.start()
            buf, rollout_stats = self._collect_rollout(envs)
            ep_returns_np = np.asarray(rollout_stats["ep_equity_returns"], dtype=np.float32)
            cvar_value = cvar_numpy(ep_returns_np, cfg.cvar_alpha)
            violation = max(0.0, -cvar_value - cfg.cvar_threshold)
            mean_ep_return = float(np.mean(rollout_stats["ep_returns"])) if rollout_stats["ep_returns"] else 0.0
            mean_ep_return = float(np.clip(mean_ep_return, -1e6, 1e6))
            if it >= cfg.cvar_warmup_iters:
                signed_violation = -cvar_value - cfg.cvar_threshold
                self.lambda_cvar = float(np.clip(
                    self.lambda_cvar + cfg.cvar_lambda_lr * signed_violation,
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
                "mean_equity_return": float(np.mean(ep_returns_np)),
                "mean_max_drawdown": float(np.mean(rollout_stats["ep_max_drawdowns"])),
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
            if val_envs and cfg.eval_every_iterations > 0 and (it + 1) % cfg.eval_every_iterations == 0:
                entry["val"] = self._evaluate_policy(
                    val_envs, cfg.eval_episodes, cfg.seed + 100_000,
                    cvar_alpha=cfg.cvar_alpha,
                )
                val_cvar = entry["val"]["cvar"]
                val_feasible = val_cvar >= -cfg.cvar_threshold
                val_score = (
                    entry["val"]["mean_equity_return"] if val_feasible else val_cvar
                )
                improved = (
                    (val_feasible and not self._best_val_feasible)
                    or (
                        val_feasible == self._best_val_feasible
                        and val_score > self._best_val_score + cfg.early_stopping_min_delta
                    )
                )
                entry["val"]["constraint_feasible"] = val_feasible
                entry["val"]["selection_score"] = val_score
                if improved:
                    self._best_val_feasible = val_feasible
                    self._best_val_score = val_score
                    self._bad_evals = 0
                    is_best = True
                else:
                    self._bad_evals += 1
            history.append(entry)
            self.cvar_history.append(entry)
            self._iteration = it + 1
            if val_envs and is_best and cfg.best_checkpoint:
                self.save(cfg.best_checkpoint)
            if (it + 1) % cfg.log_every == 0:
                logger.info(
                    "cvar-ppo it=%d episodes=%d steps=%d shaped_return=%.4f "
                    "equity_return=%.5f max_dd=%.5f cvar=%.5f violation=%.5f "
                    "lambda=%.4f policy=%.4f value=%.4f ent=%.4f total=%.4f",
                    it, len(rollout_stats["ep_returns"]), len(buf),
                    mean_ep_return, entry["mean_equity_return"], entry["mean_max_drawdown"],
                    cvar_value, violation, self.lambda_cvar,
                    losses["policy"], losses["value"],
                    losses["entropy"], losses["total"],
                )
                if cfg.checkpoint and (it + 1) % 10 == 0:
                    inter_path = str(cfg.checkpoint).replace(".pt", f"_iter_{it+1}.pt")
                    self.save(inter_path)
            timer.reset()
            if cfg.checkpoint_every_iterations > 0 and (it + 1) % cfg.checkpoint_every_iterations == 0:
                checkpoint = Path(cfg.checkpoint or "artifacts/s4_cvar/model.pt")
                self.save(str(checkpoint.with_name(f"{checkpoint.stem}_iter{it + 1}{checkpoint.suffix}")))
            should_stop = (
                bool(val_envs)
                and cfg.early_stopping_patience > 0
                and self._bad_evals >= cfg.early_stopping_patience
            )
            del buf, rollout_stats, ep_returns_t
            _release_rollout_memory()
            if should_stop:
                logger.info(
                    "cvar-ppo early stopping at iteration %d; feasible=%s best_val_score=%.6f",
                    it, self._best_val_feasible, self._best_val_score,
                )
                break
        if cfg.checkpoint:
            self.save(cfg.checkpoint)
        return {"history": history, "cvar_history": list(self.cvar_history)}

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cfg_dict = self.model.cfg.__dict__.copy()
        if "vision_channels" in cfg_dict and isinstance(cfg_dict["vision_channels"], tuple):
            cfg_dict["vision_channels"] = list(cfg_dict["vision_channels"])
        payload = {
            "model": self.model.state_dict(),
            "config": cfg_dict,
            "model_config": cfg_dict,  # canonical name
            "ppo_config": self.cfg.__dict__,
            "optimizer": self.opt.state_dict(),
            "trainer_state": {
                "step": self._step,
                "iteration": self._iteration,
                "best_val_score": self._best_val_score,
                "best_val_feasible": self._best_val_feasible,
                "bad_evals": self._bad_evals,
                "numpy_rng_state": self._rng.bit_generator.state,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "lambda_cvar": self.lambda_cvar,
            "cvar_history": self.cvar_history,
            "checkpoint_meta": {
                "stage": "s4_cvar_ppo",
                "trading_policy_ready": True,
                "policy_head_trained": True,
                "policy_training": "cvar_constrained_ppo",
                "source_checkpoint": self.cfg.source_checkpoint,
                "dataset": {
                    "root": self.cfg.dataset_root,
                    "manifest_checksum": self.cfg.dataset_manifest_checksum,
                },
            },
        }
        tmp = p.with_name(f".{p.name}.tmp-{os.getpid()}")
        try:
            torch.save(payload, tmp)
            os.replace(tmp, p)
        finally:
            if tmp.exists():
                tmp.unlink()
        logger.info("cvar-ppo checkpoint saved to %s", p)

    def load(self, path: str) -> dict:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"])
        if payload.get("optimizer"):
            self.opt.load_state_dict(payload["optimizer"])
        state = payload.get("trainer_state") or {}
        self._step = int(state.get("step", 0))
        self._iteration = int(state.get("iteration", 0))
        self._best_val_score = float(
            state.get("best_val_score", state.get("best_val_cvar", float("-inf")))
        )
        self._best_val_feasible = bool(state.get("best_val_feasible", False))
        self._bad_evals = int(state.get("bad_evals", 0))
        if state.get("numpy_rng_state"):
            self._rng.bit_generator.state = state["numpy_rng_state"]
        if state.get("torch_rng_state") is not None:
            torch.set_rng_state(state["torch_rng_state"].cpu())
        if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda_rng_state"]])
        self.lambda_cvar = float(payload.get("lambda_cvar", self.lambda_cvar))
        self.cvar_history = list(payload.get("cvar_history", []))
        return {
            "step": self._step,
            "iteration": self._iteration,
            "lambda_cvar": self.lambda_cvar,
        }


__all__ = [
    "CVaRPPOConfig",
    "CVaRPPOTrainer",
    "_per_episode_returns",
    "_tail_loss_excess",
]
