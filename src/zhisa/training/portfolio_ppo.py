"""PPO trainer for the :class:`PortfolioEnv` using :class:`PortfolioPolicyNetwork`.

This trainer mirrors :class:`zhisa.training.s4_rl.PPOTrainer` but
operates on the multi-instrument portfolio env and produces **N
factored** actions per step. The gross-leverage cap is enforced
*at action sampling time* by a per-instrument mask, so the policy
never produces an action tuple that would breach the cap.

The trainer is **drop-in** with the rest of the PPO machinery:
* GAE is computed on the per-step reward (sum of per-instrument
  PnL effects aggregated by the env).
* The policy loss is the **sum** of the per-instrument PPO
  clipped-surrogate objectives.
* Action masking is applied identically at sample time and
  during the PPO update — otherwise the ratio ``pi_new/pi_old``
  would see an apparent change in policy induced by the mask.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from zhisa.env.portfolio_action_mask import (
    compute_gross_leverage_mask,
    mask_logits,
)
from zhisa.env.portfolio_env import (
    PortfolioConfig,
    PortfolioEnv,
    encode_multi_action,
)
from zhisa.models.portfolio_policy import PortfolioPolicyNetwork
from zhisa.training.optim import OptimConfig, build_optimizer
from zhisa.training.s4_rl import (
    PPOConfig,
    RolloutBuffer,
    Transition,
    compute_gae,
    ppo_loss,
)
from zhisa.utils.logging import get_logger
from zhisa.utils.timing import Timer

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Multi-instrument transition
# ---------------------------------------------------------------------------


@dataclass
class PortfolioTransition(Transition):
    """Like :class:`Transition` but stores one (B, N, ...) tensor per modality."""

    chart: np.ndarray = field(default_factory=lambda: np.zeros(0))        # (N, 3, H, W)
    numeric: np.ndarray = field(default_factory=lambda: np.zeros(0))      # (N, T, F)
    context: np.ndarray = field(default_factory=lambda: np.zeros(0))      # (N, C)
    action: int = 0
    actions_per_instrument: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))  # (N,)
    portfolio: np.ndarray = field(default_factory=lambda: np.zeros(0))    # (portfolio_dim,)
    log_prob: float = 0.0
    log_prob_per_instrument: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))  # (N,)
    action_mask: np.ndarray = field(default_factory=lambda: np.zeros((0, 9), dtype=bool))  # (N, 9)
    reward: float = 0.0
    value: float = 0.0
    done: bool = False


class PortfolioRolloutBuffer:
    """Buffer for :class:`PortfolioTransition` records."""

    def __init__(self) -> None:
        self._data: list[PortfolioTransition] = []

    def add(self, t: PortfolioTransition) -> None:
        self._data.append(t)

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[PortfolioTransition]:
        return iter(self._data)

    def minibatch_indices(self, batch_size: int, rng: np.random.Generator) -> Iterator[np.ndarray]:
        n = len(self._data)
        order = rng.permutation(n)
        for start in range(0, n - batch_size + 1, batch_size):
            yield order[start:start + batch_size]

    def stack_tensors(self) -> dict:
        if not self._data:
            return {}
        N = int(self._data[0].actions_per_instrument.shape[0])
        A = int(self._data[0].action_mask.shape[1])
        return {
            "chart": np.stack([t.chart for t in self._data], axis=0).astype(np.float32),
            "numeric": np.stack([t.numeric for t in self._data], axis=0).astype(np.float32),
            "context": np.stack([t.context for t in self._data], axis=0).astype(np.float32),
            "portfolio": np.stack([t.portfolio for t in self._data], axis=0).astype(np.float32),
            "action": np.array([t.action for t in self._data], dtype=np.int64),
            "actions_per_instrument": np.stack(
                [t.actions_per_instrument for t in self._data], axis=0
            ).astype(np.int64),
            "reward": np.array([t.reward for t in self._data], dtype=np.float32),
            "value": np.array([t.value for t in self._data], dtype=np.float32),
            "log_prob": np.array([t.log_prob for t in self._data], dtype=np.float32),
            "log_prob_per_instrument": np.stack(
                [t.log_prob_per_instrument for t in self._data], axis=0
            ).astype(np.float32),
            "action_mask": np.stack(
                [t.action_mask for t in self._data], axis=0
            ).astype(bool),
            "done": np.array([t.done for t in self._data], dtype=np.float32),
        }

    def clear(self) -> None:
        self._data.clear()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PortfolioPPOConfig(PPOConfig):
    """Extends :class:`PPOConfig` for the portfolio env."""

    n_instruments: int = 2
    portfolio_dim: int = 32
    n_iterations: int = 5
    instruments: list[str] = field(default_factory=lambda: ["primary", "secondary"])

    def __post_init__(self) -> None:
        if self.n_instruments < 2:
            raise ValueError(f"n_instruments must be >= 2 for portfolio, got {self.n_instruments}")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class PortfolioPPOTrainer:
    """PPO over a :class:`PortfolioEnv` with a :class:`PortfolioPolicyNetwork`."""

    def __init__(
        self,
        model: PortfolioPolicyNetwork,
        cfg: Optional[PortfolioPPOConfig] = None,
    ) -> None:
        self.model = model
        self.cfg = cfg or PortfolioPPOConfig()
        self.device = torch.device(self.cfg.device)
        self.model.to(self.device)
        params = [p for p in model.parameters() if p.requires_grad]
        self.opt = build_optimizer(model, self.cfg.optim)
        self._rng = np.random.default_rng(self.cfg.seed)
        self._step = 0
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def _select_action(
        self,
        obs: dict,
        env: PortfolioEnv,
    ) -> tuple[int, np.ndarray, np.ndarray, float, torch.Tensor]:
        """Sample a joint action.

        Returns ``(flat_action, actions_per_instrument, log_prob_per_instrument,
        joint_log_prob, value)``.
        """
        instr = obs["instruments"]
        portfolio = obs["portfolio"]
        chart = np.stack([o["chart"] for o in instr], axis=0)
        numeric = np.stack([o["numeric"] for o in instr], axis=0)
        context = np.stack([o["context"] for o in instr], axis=0)
        chart_t = torch.from_numpy(chart).unsqueeze(0).float().to(self.device)
        numeric_t = torch.from_numpy(numeric).unsqueeze(0).float().to(self.device)
        context_t = torch.from_numpy(context).unsqueeze(0).float().to(self.device)
        portfolio_t = torch.from_numpy(portfolio).unsqueeze(0).float().to(self.device)
        # Current positions -> mask.
        current_positions = np.array(
            [env._instrument_position(i) for i in range(env.n_instruments)],
            dtype=np.float32,
        )
        mask = compute_gross_leverage_mask(
            current_positions, env.cfg.gross_leverage_cap, n_actions_per=self.cfg.n_actions_per_instrument  # noqa: E731
        ) if hasattr(self.cfg, "n_actions_per_instrument") else compute_gross_leverage_mask(
            current_positions, env.cfg.gross_leverage_cap, n_actions_per=self.model.cfg.n_actions_per,
        )
        with torch.no_grad():
            out = self.model(
                instruments={"chart": chart_t, "numeric": numeric_t, "context": context_t},
                portfolio=portfolio_t,
            )
            logits = out["action_logits"].squeeze(0)  # (N, A)
            mask_t = torch.from_numpy(mask).to(self.device)
            if not torch.isfinite(logits).all():
                # Degenerate policy fallback: uniform random over valid actions.
                a = torch.distributions.Categorical(logits=torch.zeros_like(logits).masked_fill(~mask_t, -1e9))
                actions = a.sample()
                log_prob_pi = a.log_prob(actions)
            else:
                masked = mask_logits(logits, mask_t)
                dist = torch.distributions.Categorical(logits=masked)
                actions = dist.sample()
                log_prob_pi = dist.log_prob(actions)
            value = out["value"].squeeze(0)
        actions_np = actions.detach().cpu().numpy().astype(np.int64)
        log_prob_np = log_prob_pi.detach().cpu().numpy().astype(np.float32)
        joint_log_prob = float(log_prob_pi.sum().item())
        flat = encode_multi_action(list(actions_np))
        return int(flat), actions_np, log_prob_np, joint_log_prob, value

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _collect_rollout(self, env: PortfolioEnv) -> tuple[PortfolioRolloutBuffer, dict]:
        buf = PortfolioRolloutBuffer()
        ep_returns: list[float] = []
        ep_lengths: list[int] = []
        ep_gross: list[float] = []
        ep_mask_violations: list[int] = []
        for ep in range(self.cfg.n_episodes):
            obs, _ = env.reset(seed=int(self._rng.integers(0, 2**31 - 1)))
            ep_return = 0.0
            steps = 0
            mask_violations = 0
            for _ in range(self.cfg.max_steps_per_episode):
                flat, actions_pi, log_prob_pi, joint_lp, value = self._select_action(obs, env)
                instr = obs["instruments"]
                chart = np.stack([o["chart"] for o in instr], axis=0)
                numeric = np.stack([o["numeric"] for o in instr], axis=0)
                context = np.stack([o["context"] for o in instr], axis=0)
                # Build the action mask used at this timestep (for PPO consistency).
                current_positions = np.array(
                    [env._envs[i]._position for i in range(env.n_instruments)],
                    dtype=np.float32,
                )
                mask = compute_gross_leverage_mask(
                    current_positions,
                    env.cfg.gross_leverage_cap,
                    n_actions_per=self.model.cfg.n_actions_per,
                )
                next_obs, reward, terminated, truncated, info = env.step(flat)
                buf.add(PortfolioTransition(
                    chart=chart,
                    numeric=numeric,
                    context=context,
                    action=flat,
                    actions_per_instrument=actions_pi,
                    portfolio=obs["portfolio"].astype(np.float32),
                    log_prob=joint_lp,
                    log_prob_per_instrument=log_prob_pi,
                    action_mask=mask,
                    reward=float(reward),
                    value=float(value.item()),
                    done=bool(terminated or truncated),
                ))
                ep_return += float(reward)
                steps += 1
                obs = next_obs
                if terminated or truncated:
                    break
            ep_returns.append(ep_return)
            ep_lengths.append(steps)
            ep_gross.append(float(info.get("gross_leverage", 0.0)))
            ep_mask_violations.append(mask_violations)
        return buf, {
            "ep_returns": ep_returns,
            "ep_lengths": ep_lengths,
            "ep_gross": ep_gross,
            "ep_mask_violations": ep_mask_violations,
        }

    # ------------------------------------------------------------------
    # PPO update with factored heads
    # ------------------------------------------------------------------

    def _ppo_update(self, buf: PortfolioRolloutBuffer) -> dict:
        cfg = self.cfg
        if len(buf) == 0:
            return {"policy": 0.0, "value": 0.0, "entropy": 0.0, "total": 0.0}
        stacked = buf.stack_tensors()
        N = int(stacked["actions_per_instrument"].shape[1])
        A = int(stacked["action_mask"].shape[-1])
        # GAE on the joint reward (env already aggregates).
        advantages, returns = compute_gae(
            stacked["reward"], stacked["value"], stacked["done"], last_value=0.0,
            gamma=cfg.gamma, lam=cfg.gae_lambda,
        )
        to_t = lambda a: torch.from_numpy(a).to(self.device)  # noqa: E731
        adv_t = to_t(advantages)
        ret_t = to_t(returns)
        old_lp_pi = to_t(stacked["log_prob_per_instrument"])  # (T, N)
        old_lp_joint = to_t(stacked["log_prob"])              # (T,)
        action_pi_t = to_t(stacked["actions_per_instrument"])  # (T, N)
        chart_t = to_t(stacked["chart"]).float()
        num_t = to_t(stacked["numeric"]).float()
        ctx_t = to_t(stacked["context"]).float()
        port_t = to_t(stacked["portfolio"]).float()
        mask_t = to_t(stacked["action_mask"]).bool()  # (T, N, A)
        stats = {"policy": [], "value": [], "entropy": [], "total": []}
        for epoch in range(cfg.n_epochs):
            for idx in buf.minibatch_indices(cfg.minibatch_size, self._rng):
                mb_chart = chart_t[idx]
                mb_num = num_t[idx]
                mb_ctx = ctx_t[idx]
                mb_port = port_t[idx]
                mb_mask = mask_t[idx]
                mb_act = action_pi_t[idx]
                mb_old_lp = old_lp_pi[idx]
                mb_old_lp_joint = old_lp_joint[idx]
                out = self.model(
                    instruments={"chart": mb_chart, "numeric": mb_num, "context": mb_ctx},
                    portfolio=mb_port,
                )
                logits = out["action_logits"]  # (B, N, A)
                masked_logits = mask_logits(logits, mb_mask)
                dist = torch.distributions.Categorical(logits=masked_logits)
                new_lp_pi = dist.log_prob(mb_act)              # (B, N)
                entropy_pi = dist.entropy()                    # (B, N)
                new_lp_joint = new_lp_pi.sum(dim=-1)          # (B,)
                # PPO loss summed across instruments: per-instrument surrogate
                # contribution averaged.
                adv_mb = adv_t[idx]
                if adv_mb.numel() > 1:
                    adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)
                # Per-instrument ratio, clipped.
                ratio = torch.exp(new_lp_pi - mb_old_lp)        # (B, N)
                surr1 = ratio * adv_mb.unsqueeze(-1)
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv_mb.unsqueeze(-1)
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(out["value"], ret_t[idx])
                entropy_scalar = entropy_pi.mean()
                total = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy_scalar
                )
                if not torch.isfinite(total):
                    logger.warning("portfolio-ppo step %d: non-finite loss, skipping", self._step)
                    continue
                self.opt.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.opt.step()
                self._step += 1
                stats["policy"].append(float(policy_loss.item()))
                stats["value"].append(float(value_loss.item()))
                stats["entropy"].append(float(entropy_scalar.item()))
                stats["total"].append(float(total.item()))
                with torch.no_grad():
                    kl = (mb_old_lp_joint - new_lp_joint).mean().item()
                if abs(kl) > cfg.target_kl:
                    logger.info("portfolio-ppo early-stop at epoch %d: KL=%.4f", epoch, kl)
                    break
            else:
                continue
            break
        if not stats["total"]:
            return {"policy": 0.0, "value": 0.0, "entropy": 0.0, "total": 0.0}
        return {k: float(np.mean(v)) for k, v in stats.items()}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fit(self, dataframes: dict[str, pd.DataFrame], env_cfg: Optional[PortfolioConfig] = None) -> dict:
        cfg = self.cfg
        env_cfg = env_cfg or PortfolioConfig(
            n_instruments=cfg.n_instruments,
            instrument_names=list(dataframes.keys()),
            env_cfg=cfg.env_cfg,
        )
        env = PortfolioEnv(dataframes, cfg=env_cfg)
        history: list[dict] = []
        timer = Timer()
        for it in range(cfg.n_iterations):
            timer.start()
            buf, rollout_stats = self._collect_rollout(env)
            losses = self._ppo_update(buf)
            timer.stop()
            mean_return = float(np.mean(rollout_stats["ep_returns"])) if rollout_stats["ep_returns"] else 0.0
            mean_return = float(np.clip(mean_return, -1e6, 1e6))
            mean_gross = float(np.mean(rollout_stats["ep_gross"])) if rollout_stats["ep_gross"] else 0.0
            entry = {
                "iteration": it,
                "n_episodes": len(rollout_stats["ep_returns"]),
                "rollout_steps": len(buf),
                "mean_return": mean_return,
                "mean_gross_leverage": mean_gross,
                "policy_loss": losses["policy"],
                "value_loss": losses["value"],
                "entropy": losses["entropy"],
                "total_loss": losses["total"],
                "elapsed_s": timer.elapsed,
            }
            history.append(entry)
            if (it + 1) % cfg.log_every == 0:
                logger.info(
                    "portfolio-ppo it=%d episodes=%d steps=%d mean_return=%.4f "
                    "gross=%.4f policy=%.4f value=%.4f ent=%.4f total=%.4f",
                    it, len(rollout_stats["ep_returns"]), len(buf),
                    mean_return, mean_gross, losses["policy"], losses["value"],
                    losses["entropy"], losses["total"],
                )
            timer.reset()
        self.history = history
        if cfg.checkpoint:
            self.save(cfg.checkpoint)
        return {"history": history}

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "config": self.model.cfg.__dict__,
            "ppo_config": self.cfg.__dict__,
            "history": self.history,
        }, p)
        logger.info("portfolio-ppo checkpoint saved to %s", p)


__all__ = [
    "PortfolioPPOConfig",
    "PortfolioPPOTrainer",
    "PortfolioRolloutBuffer",
    "PortfolioTransition",
]
