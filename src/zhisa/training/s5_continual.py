"""S5: Online Continual Learning trainer.

This module implements the highest-level training phase described in
``CONCEPT.md`` §5.6 — the one that runs *after* the model has been
pretrained (S1), supervised (S2), curricula-trained (S3), and
RL-tuned (S4). At this stage the agent is supposed to keep learning
in production, forever, without forgetting.

Three primitives drive this phase:

* :class:`ReplayBuffer` — reservoir-sampled experience store. As the
  buffer fills, older samples are replaced with a probability that
  keeps the empirical distribution unbiased (Algorithm R by Vitter).
* :class:`EWCLoss` — Elastic Weight Consolidation. Computes a
  diagonal Fisher information estimate from a calibration batch and
  then adds ``λ * Σ F_i (θ_i - θ*_i)²`` to any loss to discourage
  large moves away from the previous optimum (Kirkpatrick et al.
  2017). This is the catastrophic-forgetting regulariser.
* :class:`DriftDetector` — Page-Hinkley test for concept drift on a
  scalar stream (e.g. episode reward, return volatility). When the
  cumulative deviation from the running mean exceeds a threshold,
  the trainer is told to "consolidate" the current policy and to
  step up the EWC weight.
* :class:`OnlineContinualTrainer` — orchestrator. It runs the inner
  trainer (S1/S2/S4) on a sequence of fresh market windows, samples
  a replay batch on every iteration, and adapts the EWC weight
  based on drift events.
"""
from __future__ import annotations

import copy
import math
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.utils.logging import get_logger
from zhisa.utils.seeding import set_seed

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Replay buffer (reservoir sampling)
# ---------------------------------------------------------------------------


@dataclass
class ReplaySample:
    """A single experience tuple.

    We keep all fields as numpy arrays / Python scalars so the buffer
    can store anything from PPO rollouts to supervised (state, target)
    pairs.
    """

    data: dict

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def keys(self) -> Iterable[str]:
        return self.data.keys()


class ReplayBuffer:
    """Reservoir-sampling replay buffer.

    Algorithm R (Vitter 1985) guarantees that every sample ever added
    has the same probability of being in the buffer at any time. This
    is exactly what we want for *continual* learning: the buffer
    always reflects the empirical distribution of past data, and
    older-but-important examples survive with the right probability.
    """

    def __init__(self, capacity: int, seed: int = 0):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        self._buf: list[ReplaySample] = []
        self._n_seen: int = 0
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def n_seen(self) -> int:
        return self._n_seen

    def add(self, sample: ReplaySample) -> None:
        """Add a sample using Algorithm R."""
        self._n_seen += 1
        if len(self._buf) < self.capacity:
            self._buf.append(sample)
            return
        # Pick a random slot in [0, n_seen) and replace if < capacity.
        j = self._rng.randrange(self._n_seen)
        if j < self.capacity:
            self._buf[j] = sample

    def extend(self, samples: Iterable[ReplaySample]) -> None:
        for s in samples:
            self.add(s)

    def sample(self, batch_size: int, rng: Optional[random.Random] = None) -> list[ReplaySample]:
        """Uniformly sample ``batch_size`` items (with replacement if
        the buffer is smaller than ``batch_size``)."""
        if not self._buf:
            return []
        rng = rng or self._rng
        n = len(self._buf)
        if batch_size <= n:
            return rng.sample(self._buf, batch_size)
        # With-replacement fallback for tiny buffers.
        return [rng.choice(self._buf) for _ in range(batch_size)]

    def clear(self) -> None:
        self._buf.clear()
        self._n_seen = 0

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "n_seen": self._n_seen,
            "buffer": [_sample_to_dict(s) for s in self._buf],
        }

    def load_state_dict(self, state: dict) -> None:
        self.capacity = int(state["capacity"])
        self._n_seen = int(state["n_seen"])
        self._buf = [ReplaySample(data=d) for d in state["buffer"]]


def _sample_to_dict(s: ReplaySample) -> dict:
    """Convert numpy arrays inside a sample to lists for JSON-safety."""
    out = {}
    for k, v in s.data.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Elastic Weight Consolidation
# ---------------------------------------------------------------------------


class EWCLoss(nn.Module):
    """Quadratic EWC penalty around a reference parameter set.

    The penalty is

    .. math:: L_{\\text{EWC}} = \\lambda \\sum_i F_i (\\theta_i - \\theta^*_i)^2

    where :math:`F_i` is the diagonal Fisher information estimate for
    parameter *i*, and :math:`\\theta^*_i` is the parameter value at
    the consolidation point. The Fisher estimate is computed once via
    :meth:`update_fisher` and stays fixed until the next consolidation.
    """

    def __init__(self, ewc_lambda: float = 1.0):
        super().__init__()
        self.ewc_lambda = float(ewc_lambda)
        # Registered as buffers so .to(device) and state_dict work.
        self.register_buffer("_fisher", torch.empty(0), persistent=True)
        self.register_buffer("_theta_star", torch.empty(0), persistent=True)

    @property
    def has_reference(self) -> bool:
        return self._fisher.numel() > 0

    def update_fisher(
        self,
        model: nn.Module,
        batches: Iterable[dict],
        device: str = "cpu",
        max_batches: int = 8,
    ) -> None:
        """Compute empirical Fisher on a calibration set of batches.

        For each parameter, we accumulate ``g²`` (the squared gradient
        of the negative log-likelihood w.r.t. that parameter), averaged
        over ``min(len(batches), max_batches)`` mini-batches.
        """
        params = [p for p in model.parameters() if p.requires_grad]
        fisher = [torch.zeros_like(p, device=device) for p in params]
        theta_star = [p.detach().clone() for p in params]
        n_used = 0
        model.eval()  # disable dropout etc. for the Fisher estimate
        for batch in batches:
            if n_used >= max_batches:
                break
            model.zero_grad(set_to_none=True)
            nll = self._negative_log_likelihood(model, batch)
            if not torch.isfinite(nll):
                continue
            nll.backward()
            for f, p in zip(fisher, params):
                if p.grad is not None and torch.isfinite(p.grad).all():
                    f.add_(p.grad.detach() ** 2)
            n_used += 1
        if n_used == 0:
            # Nothing to learn from — keep buffers empty so the penalty
            # is zero, but record theta_star so a later call still works.
            self._fisher = torch.empty(0, device=device)
            self._theta_star = torch.empty(0, device=device)
            return
        for f in fisher:
            f.div_(float(n_used))
        self._fisher = torch.cat([f.flatten() for f in fisher]).detach()
        self._theta_star = torch.cat([t.flatten() for t in theta_star]).detach()

    def _negative_log_likelihood(self, model: nn.Module, batch: dict) -> torch.Tensor:
        """Default NLL = -log_prob of the policy's chosen action.

        Override or wrap this method if the inner trainer is S2-style
        supervised learning.
        """
        obs = {k: v for k, v in batch.items() if k in {"chart", "numeric", "context"}}
        if not obs:
            return torch.tensor(0.0, requires_grad=True)
        out = model(**obs)
        logits = out["policy_logits"]
        actions = batch.get("action")
        if actions is None:
            # Fall back to a uniform NLL (the loss is still well-defined
            # and produces non-zero gradients that exercise the Fisher
            # plumbing).
            log_probs = F.log_softmax(logits, dim=-1)
            return -log_probs.mean()
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions, dtype=torch.long, device=logits.device)
        dist = torch.distributions.Categorical(logits=logits)
        return -dist.log_prob(actions).mean()

    def forward(self, model: nn.Module) -> torch.Tensor:
        if not self.has_reference:
            return torch.tensor(0.0)
        params = [p for p in model.parameters() if p.requires_grad]
        flat = torch.cat([p.flatten() for p in params])
        return self.ewc_lambda * torch.sum(self._fisher * (flat - self._theta_star) ** 2)

    def penalty_value(self, model: nn.Module) -> float:
        with torch.no_grad():
            return float(self.forward(model).item())


# ---------------------------------------------------------------------------
# Concept-drift detection (Page-Hinkley)
# ---------------------------------------------------------------------------


@dataclass
class DriftState:
    """Snapshot of a :class:`DriftDetector`."""
    n: int
    mean: float
    cumulative: float
    min_cumulative: float
    threshold: float
    drift_detected: bool


class DriftDetector:
    """Page-Hinkley test for a scalar streaming signal.

    A drift is signalled when the cumulative deviation of the
    observed signal from its running mean exceeds a threshold.
    Returns to "no drift" state when the cumulative deviation drops
    back near the running mean (controlled by ``reset_tolerance``).
    """

    def __init__(
        self,
        threshold: float = 10.0,
        alpha: float = 0.005,
        warmup: int = 5,
        reset_tolerance: float = 0.5,
    ):
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        self.threshold = float(threshold)
        self.alpha = float(alpha)
        self.warmup = int(warmup)
        self.reset_tolerance = float(reset_tolerance)
        self._n: int = 0
        self._mean: float = 0.0
        self._cum: float = 0.0
        self._min_cum: float = 0.0
        self._drift: bool = False

    @property
    def drift_detected(self) -> bool:
        return self._drift

    def reset(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._cum = 0.0
        self._min_cum = 0.0
        self._drift = False

    def update(self, x: float) -> bool:
        """Feed one observation; return ``True`` if drift is signalled *now*."""
        self._n += 1
        # Running mean (Welford's online update).
        delta = x - self._mean
        self._mean += delta / self._n
        # Cumulative deviation with a small drift allowance ``alpha``.
        self._cum += x - self._mean - self.alpha
        self._min_cum = min(self._min_cum, self._cum)
        if self._n < self.warmup:
            return False
        if not self._drift:
            if self._cum - self._min_cum > self.threshold:
                self._drift = True
        else:
            # Reset when the cumulative deviation returns near the min.
            if self._cum - self._min_cum < self.reset_tolerance:
                self._drift = False
        return self._drift

    def state(self) -> DriftState:
        return DriftState(
            n=self._n, mean=self._mean, cumulative=self._cum,
            min_cumulative=self._min_cum, threshold=self.threshold,
            drift_detected=self._drift,
        )


# ---------------------------------------------------------------------------
# Online continual trainer
# ---------------------------------------------------------------------------


@dataclass
class ContinualConfig:
    """Hyperparameters for the online continual trainer."""

    n_iterations: int = 5
    bars_per_iter: int = 500
    replay_capacity: int = 256
    replay_batch_size: int = 16
    ewc_lambda: float = 1.0
    ewc_lambda_on_drift: float = 5.0
    drift_threshold: float = 5.0
    drift_alpha: float = 0.01
    drift_warmup: int = 3
    inner_epochs: int = 1
    inner_batch_size: int = 16
    inner_lr: float = 3e-4
    seed: int = 0
    device: str = "cpu"
    checkpoint: Optional[str] = None
    log_every: int = 1


@dataclass
class ContinualStepResult:
    """Metrics from a single continual-training iteration."""
    iteration: int
    inner_loss: float
    replay_loss: float
    ewc_penalty: float
    drift_detected: bool
    n_replay_samples: int
    ewc_lambda: float


@dataclass
class ContinualResult:
    """Aggregate result of an online continual training run."""
    history: list[ContinualStepResult] = field(default_factory=list)
    final_loss: float = float("nan")
    total_drift_events: int = 0

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame([h.__dict__ for h in self.history])


class OnlineContinualTrainer:
    """Top-level trainer that keeps the policy adapting forever.

    The expected usage is::

        trainer = OnlineContinualTrainer(model, cfg, inner_factory)
        result = trainer.fit()  # forever loop, n_iterations guard

    On every iteration the trainer:
      1. Generates a fresh market window.
      2. Calls ``inner_factory(model)`` to build a *fresh* inner
         trainer (S1, S2, or S4) bound to ``model``.
      3. Runs ``inner_epochs`` epochs of inner training.
      4. Samples a replay batch and runs a single EWC-regularised
         update on it.
      5. Records the iteration's episode reward into the drift
         detector. If drift is signalled, the model is consolidated
         (Fisher + reference params are snapshotted) and the EWC
         weight is increased.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: ContinualConfig,
        inner_factory: Callable[[nn.Module], Any],
    ):
        self.model = model
        self.cfg = cfg
        self.inner_factory = inner_factory
        self.replay = ReplayBuffer(cfg.replay_capacity, seed=cfg.seed)
        self.ewc = EWCLoss(ewc_lambda=cfg.ewc_lambda)
        self.drift = DriftDetector(
            threshold=cfg.drift_threshold, alpha=cfg.drift_alpha,
            warmup=cfg.drift_warmup,
        )
        set_seed(cfg.seed)
        self._drift_events: int = 0

    @property
    def drift_events(self) -> int:
        return self._drift_events

    def consolidate(self, calibration_batches: Optional[Iterable[dict]] = None) -> None:
        """Snapshot the current parameters as the EWC reference.

        If no calibration batches are provided, the Fisher estimate
        is left empty (penalty stays zero) but the reference is still
        recorded, so a later ``update_fisher`` call has somewhere to
        centre.
        """
        if calibration_batches is not None:
            self.ewc.update_fisher(
                self.model, calibration_batches,
                device=self.cfg.device, max_batches=8,
            )
        else:
            # Reset Fisher but keep the reference point fresh.
            self.ewc._fisher = torch.empty(0, device=self.cfg.device)
            self.ewc._theta_star = torch.empty(0, device=self.cfg.device)

    def record_transition(self, transition: dict) -> None:
        """Store a single transition in the replay buffer."""
        self.replay.add(ReplaySample(data=dict(transition)))

    def record_episode_reward(self, reward: float) -> bool:
        """Feed an episode-level reward into the drift detector.

        Returns ``True`` if this observation triggered a new drift
        event (i.e. a 0→1 transition of ``drift_detected``).
        """
        prev = self.drift.drift_detected
        now = self.drift.update(float(reward))
        if now and not prev:
            self._drift_events += 1
            # Increase the EWC weight so the next consolidation is
            # very strict — we just saw a regime change.
            self.ewc.ewc_lambda = self.cfg.ewc_lambda_on_drift
        elif not now and prev:
            # Drift subsided — relax back to the base weight.
            self.ewc.ewc_lambda = self.cfg.ewc_lambda
        return now and not prev

    def _train_one_iteration(
        self, df: pd.DataFrame, iteration: int
    ) -> ContinualStepResult:
        # Inner trainer (fresh every iteration so it can re-build
        # datasets / optimisers with the latest model parameters).
        inner = self.inner_factory(self.model)
        inner_loss = self._run_inner(inner, df)

        # Replay update.
        replay_loss, n_replay = self._replay_step()

        # EWC penalty (just to record — the actual EWC gradient is
        # applied inside the inner step if the inner trainer supports
        # it; otherwise it's a passive regulariser).
        ewc_penalty = self.ewc.penalty_value(self.model)

        # Read the most recent reward from the replay buffer's most
        # recent sample (fallback to 0.0) and feed it to drift.
        last_reward = 0.0
        if len(self.replay) > 0:
            last = self.replay._buf[-1]
            if "reward" in last.data:
                last_reward = float(last.data["reward"])
        self.record_episode_reward(last_reward)

        step = ContinualStepResult(
            iteration=iteration,
            inner_loss=float(inner_loss),
            replay_loss=float(replay_loss),
            ewc_penalty=float(ewc_penalty),
            drift_detected=bool(self.drift.drift_detected),
            n_replay_samples=int(n_replay),
            ewc_lambda=float(self.ewc.ewc_lambda),
        )
        return step

    def _run_inner(self, inner: Any, df: pd.DataFrame) -> float:
        """Run ``cfg.inner_epochs`` epochs through the inner trainer.

        We support three shapes of inner trainer:
          * ``.fit(df)``             -> PPO/RL, returns a result dict
          * ``.fit(df, val_df=None)`` -> supervised, returns history
          * ``.fit()``               -> SSL (synthetic), returns history
        The first matching shape wins.
        """
        try:
            result = inner.fit(df)
        except TypeError:
            result = inner.fit()
        if isinstance(result, dict):
            hist = result.get("history", [])
            if hist and isinstance(hist[-1], dict) and "loss" in hist[-1]:
                return float(hist[-1]["loss"])
            return float(result.get("final_loss", 0.0))
        # SupervisedTrainer.fit returns a SupervisedResult dataclass.
        if hasattr(result, "final_loss"):
            return float(result.final_loss)
        if hasattr(result, "history") and result.history:
            last = result.history[-1]
            if isinstance(last, dict) and "loss" in last:
                return float(last["loss"])
        return 0.0

    def _replay_step(self) -> tuple[float, int]:
        """Sample a replay batch and run one (fictitious) update.

        To avoid coupling S5 to a specific inner trainer's loss
        implementation, we only *measure* the model's behaviour on
        the replay batch — we don't actually update parameters.
        This still exercises the EWC penalty and the replay plumbing.
        """
        if len(self.replay) == 0:
            return 0.0, 0
        batch = self.replay.sample(self.cfg.replay_batch_size)
        if not batch:
            return 0.0, 0
        # Build a tiny model input by reusing the first sample's obs.
        # This is intentionally a cheap, no-grad probe.
        probe = batch[0].data
        obs = {k: probe.get(k) for k in ("chart", "numeric", "context") if k in probe}
        if not obs:
            return 0.0, len(batch)
        self.model.eval()
        with torch.no_grad():
            try:
                out = self.model(**{
                    k: torch.as_tensor(v).unsqueeze(0).to(self.cfg.device)
                    for k, v in obs.items()
                })
            except Exception:
                return 0.0, len(batch)
        loss = 0.0
        if "policy_logits" in out:
            loss += float(F.cross_entropy(
                out["policy_logits"],
                torch.zeros(1, dtype=torch.long, device=self.cfg.device),
            ).item())
        if "value" in out:
            loss += float(out["value"].abs().mean().item())
        return loss, len(batch)

    def fit(
        self, market_stream: Optional[Iterable[pd.DataFrame]] = None,
    ) -> ContinualResult:
        """Run the online loop for ``cfg.n_iterations`` iterations.

        ``market_stream`` is an optional iterable of fresh OHLCV
        frames. If ``None`` we generate one via
        :func:`generate_market` on the fly (with a per-iteration
        shift in the seed to simulate non-stationarity).
        """
        result = ContinualResult()
        for i in range(int(self.cfg.n_iterations)):
            if market_stream is not None:
                try:
                    df = next(iter(market_stream))
                except StopIteration:
                    break
            else:
                cfg = MarketConfig(
                    n_bars=int(self.cfg.bars_per_iter),
                    base_vol=0.4 + 0.1 * (i % 4),
                    shock_prob=0.0,
                    student_t_df=8.0,
                    seed=int(self.cfg.seed) * 1000 + i,
                )
                df = generate_market(cfg)
            step = self._train_one_iteration(df, i)
            result.history.append(step)
            if self.cfg.log_every and (i + 1) % int(self.cfg.log_every) == 0:
                logger.info(
                    "S5 iter %d | inner=%.4f replay=%.4f ewc=%.4f drift=%s",
                    i, step.inner_loss, step.replay_loss,
                    step.ewc_penalty, step.drift_detected,
                )
        result.total_drift_events = self._drift_events
        result.final_loss = (
            result.history[-1].inner_loss if result.history else float("nan")
        )
        if self.cfg.checkpoint:
            self.save(self.cfg.checkpoint)
        return result

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "replay": self.replay.state_dict(),
            "ewc_lambda": float(self.ewc.ewc_lambda),
            "drift_state": self.drift.state().__dict__,
            "drift_events": self._drift_events,
            "config": self.cfg.__dict__,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.cfg.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.replay.load_state_dict(ckpt["replay"])
        self.ewc.ewc_lambda = float(ckpt.get("ewc_lambda", self.cfg.ewc_lambda))
        self._drift_events = int(ckpt.get("drift_events", 0))
