"""S2b: Imitation learning (Behavioral Cloning + DAgger).

This module implements the imitation-learning stage from
``CONCEPT.md`` §5.3. It is positioned between S2 (supervised
multi-task) and S4 (RL): the goal is to give the policy a strong
initialisation that already produces non-random trades, so that
S4's PPO loop starts from a meaningful point and explores
meaningful neighbourhoods of action space.

Two trainers are provided:

* :class:`BehavioralCloningTrainer` — supervised learning of
  ``policy_logits`` against the action chosen by a rule-based
  :class:`ExpertPolicy`. We also co-train the auxiliary heads
  (direction, vol, regime) using the same multi-task loss that
  S2 uses, but ``policy`` is the dominant term.

* :class:`DAggerTrainer` — DAgger (Dataset Aggregation). On each
  round the current policy is rolled out inside a
  :class:`TradingEnv`, the expert is queried on every visited
  state, the resulting ``(obs, expert_action)`` pairs are
  aggregated with the previous dataset, and the policy is
  retrained for a few epochs. This addresses the standard
  covariate-shift failure mode of plain BC.

The trainers share helpers so DAgger can reuse BC's training step.
The S4 PPO trainer can load checkpoints produced here directly
since the model state-dict shape is the same.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.optim import AdamW

from zhisa.data.dataset import MarketDataset, MarketTargetConfig, SampleSpec
from zhisa.training.dataloader_factory import build_dataloader


def _batched_collate(batch: list[dict]) -> dict:
    """Collate that handles the standard multimodal keys plus ``action``.

    The stock :func:`zhisa.data.dataset._batched_collate` is hardcoded
    to the supervised keys, so BC and DAgger need a variant that also
    stacks the per-sample ``action`` tensor.
    """
    keys_tensor = (
        "chart", "numeric", "context",
        "label_dir", "label_vol", "label_risk", "label_regime", "label_ret", "mask",
        "action",
    )
    out: dict = {}
    for k in keys_tensor:
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["meta"] = [b.get("meta", {}) for b in batch]
    return out
from zhisa.data.expert import ExpertPolicy, SymmetricUtilityExpert, TripleBarrierExpert
from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.env.actions import DiscreteAction
from zhisa.env.trading_env import EnvConfig, TradingEnv
from zhisa.models.policy import PolicyConfig, PolicyNetwork
from zhisa.training.losses import LossWeights, MultiTaskLoss
from zhisa.training.optim import OptimConfig, build_optimizer, build_scheduler
from zhisa.utils.logging import get_logger
from zhisa.utils.seeding import set_seed
from zhisa.utils.timing import Timer

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Labeled dataset (BC)
# ---------------------------------------------------------------------------


class _LabeledMarketDataset(Dataset):
    """A thin wrapper over :class:`MarketDataset` that adds an action label.

    ``actions`` is a 1-D ``int64`` array aligned to the same bar
    indices the underlying dataset yields. Bars with no valid
    observation (the warmup window and the trailing ``max_holding``
    bars) are masked out — the dataloader will skip them.
    """

    def __init__(self, base: MarketDataset, actions: np.ndarray):
        if len(actions) < len(base):
            raise ValueError(
                f"actions array is too short: got {len(actions)}, need >= {len(base)}"
            )
        self.base = base
        # ``actions`` is indexed by absolute bar index; we store the
        # slice that aligns with the dataset's yielded indices.
        # MarketDataset yields t in [0, len(df) - chart_window - horizon_max - 1).
        # We let the caller pre-trim the array, so we just keep it as-is.
        self.actions = actions

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, t: int) -> dict:
        sample = self.base[t]
        sample["action"] = torch.tensor(int(self.actions[t]), dtype=torch.long)
        return sample


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BCConfig:
    """Hyperparameters for behavioral cloning."""

    epochs: int = 3
    batch_size: int = 32
    grad_clip: float = 1.0
    log_every: int = 50
    checkpoint: Optional[str] = None
    device: str = "cpu"
    seed: int = 0
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss_weights: LossWeights = field(default_factory=LossWeights)
    # The ``policy`` weight is overridden to 1.0 by default; the user
    # can still turn it down via the config if imitation is auxiliary.
    use_expert_actions_only: bool = False  # if True, do NOT co-train aux heads
    num_workers: int = 0
    source_checkpoint: Optional[str] = None
    dataset_root: Optional[str] = None
    dataset_manifest_checksum: Optional[str] = None
    best_checkpoint: Optional[str] = None
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    freeze_backbone_epochs: int = 0
    backbone_lr_scale: float = 1.0
    policy_class_balance: str = "none"
    champion_metric: str = "composite_score"
    expert_name: str = "unknown"


@dataclass
class DAggerConfig:
    """Hyperparameters for DAgger (Dataset Aggregation)."""

    n_rounds: int = 3
    epochs_per_round: int = 1
    rollout_episodes_per_round: int = 2
    max_steps_per_episode: int = 200
    batch_size: int = 32
    grad_clip: float = 1.0
    log_every: int = 50
    checkpoint: Optional[str] = None
    device: str = "cpu"
    seed: int = 0
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss_weights: LossWeights = field(default_factory=LossWeights)
    env_cfg: EnvConfig = field(default_factory=EnvConfig)


@dataclass
class DAggerRoundMetrics:
    round_idx: int
    n_aggregated: int
    n_new_pairs: int
    bc_loss: float
    elapsed_s: float


@dataclass
class DAggerResult:
    rounds: list[DAggerRoundMetrics]
    final_loss: float

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self.rounds])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(
    df: pd.DataFrame,
    spec: SampleSpec,
    expert: ExpertPolicy,
) -> tuple[MarketDataset, np.ndarray]:
    """Build a MarketDataset and an aligned action-label array.

    Bars outside the valid observation window (warmup or trailing
    ``max_holding`` for the triple-barrier expert) are labelled
    ``SKIP`` (= 0) and excluded by the dataset's own length.
    """
    target_cfg = None
    triple_barrier_cfg = None
    if isinstance(expert, TripleBarrierExpert):
        target_cfg = MarketTargetConfig(direction_mode="triple_barrier")
        triple_barrier_cfg = expert.cfg
    ds = MarketDataset(
        df,
        spec=spec,
        triple_barrier_cfg=triple_barrier_cfg,
        target_cfg=target_cfg,
        cache_charts=True,
    )
    n = len(ds)
    actions = _expert_actions_for_dataset(ds, expert)
    return ds, actions


def _expert_actions_for_dataset(dataset: MarketDataset, expert: ExpertPolicy) -> np.ndarray:
    """Return actions aligned to each sample's final visible bar."""
    n = len(dataset)
    if isinstance(expert, SymmetricUtilityExpert):
        start = dataset.spec.chart_window - 1
        return expert.predict_array(dataset.df, start=start)[:n]
    if isinstance(expert, TripleBarrierExpert):
        cached = dataset._tb_cfg_primary
        same_barriers = (
            cached.max_holding == expert.cfg.max_holding
            and cached.atr_window == expert.cfg.atr_window
            and cached.tp_atr_mult == expert.cfg.tp_atr_mult
            and cached.sl_atr_mult == expert.cfg.sl_atr_mult
        )
        cached_is_triple_barrier = (
            getattr(getattr(dataset, "target_cfg", None), "direction_mode", None)
            == "triple_barrier"
        )
        if cached_is_triple_barrier and same_barriers and dataset._tb_label_arr is not None:
            start = dataset.spec.chart_window - 1
            labels = dataset._tb_label_arr[start:start + n]
            actions = np.full(n, int(expert.skip_action), dtype=np.int64)
            actions[labels > 0] = int(expert.long_action)
            actions[labels < 0] = int(expert.short_action)
            # The expert contract requires t >= chart_window.
            if n and start < expert.chart_window:
                actions[:expert.chart_window - start] = int(expert.skip_action)
            return actions
    return np.asarray([
        int(expert.predict(dataset.df, i + dataset.spec.chart_window - 1))
        for i in range(n)
    ], dtype=np.int64)


def _train_bc_one_epoch(
    model: PolicyNetwork,
    loss_fn: MultiTaskLoss,
    opt: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    grad_clip: float,
    log_every: int,
    epoch: int,
    scheduler=None,
    policy_only: bool = False,
) -> float:
    """Standard BC step: cross-entropy on ``policy_logits`` (+ aux heads)."""
    model.train()
    if policy_only:
        model.eval()
        model.heads.policy_logits.train()
    total = 0.0
    n = 0
    for it, batch in enumerate(loader):
        chart = batch["chart"].to(device, non_blocking=device.type == "cuda")
        num = batch["numeric"].to(device, non_blocking=device.type == "cuda")
        ctx = batch["context"].to(device, non_blocking=device.type == "cuda")
        action = batch["action"].to(device, non_blocking=device.type == "cuda")
        out = model(chart=chart, numeric=num, context=ctx)
        targets = {
            "label_dir": batch["label_dir"].to(device, non_blocking=device.type == "cuda"),
            "label_vol": batch["label_vol"].to(device, non_blocking=device.type == "cuda"),
            "label_risk": batch["label_risk"].to(device, non_blocking=device.type == "cuda"),
            "label_regime": batch["label_regime"].to(device, non_blocking=device.type == "cuda"),
            "label_ret": batch["label_ret"].to(device, non_blocking=device.type == "cuda"),
            "action": action,
        }
        losses = loss_fn(out, targets)
        loss = losses["total"]
        if not torch.isfinite(loss):
            logger.warning("bc epoch %d iter %d: non-finite loss, skipping", epoch, it)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        if scheduler is not None:
            scheduler.step()
        bs = chart.size(0)
        total += float(loss.item()) * bs
        n += bs
        if (it + 1) % log_every == 0:
            logger.info(
                "bc epoch=%d iter=%d loss=%.4f",
                epoch, it, total / max(1, n),
            )
    return total / max(1, n)


@torch.no_grad()
def _evaluate_bc(
    model: PolicyNetwork,
    loss_fn: MultiTaskLoss,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    policy_loss = 0.0
    correct = 0
    n = 0
    confusion: np.ndarray | None = None
    bin_count = np.zeros(15, dtype=np.int64)
    bin_confidence = np.zeros(15, dtype=np.float64)
    bin_correct = np.zeros(15, dtype=np.float64)
    for batch in loader:
        chart = batch["chart"].to(device, non_blocking=device.type == "cuda")
        num = batch["numeric"].to(device, non_blocking=device.type == "cuda")
        ctx = batch["context"].to(device, non_blocking=device.type == "cuda")
        action = batch["action"].to(device, non_blocking=device.type == "cuda")
        out = model(chart=chart, numeric=num, context=ctx)
        probs = torch.softmax(out["policy_logits"], dim=-1)
        predicted = probs.argmax(dim=-1)
        targets = {
            "label_dir": batch["label_dir"].to(device),
            "label_vol": batch["label_vol"].to(device),
            "label_risk": batch["label_risk"].to(device),
            "label_regime": batch["label_regime"].to(device),
            "label_ret": batch["label_ret"].to(device),
            "action": action,
        }
        loss = loss_fn(out, targets)["total"]
        bs = chart.size(0)
        total_loss += float(loss.item()) * bs
        policy_loss += float(F.cross_entropy(out["policy_logits"], action).item()) * bs
        correct += int((predicted == action).sum().item())
        if confusion is None:
            confusion = np.zeros((probs.shape[-1], probs.shape[-1]), dtype=np.int64)
        y_np = action.cpu().numpy()
        pred_np = predicted.cpu().numpy()
        np.add.at(confusion, (y_np, pred_np), 1)
        confidence = probs.max(dim=-1).values.cpu().numpy()
        bins = np.minimum((confidence * 15).astype(np.int64), 14)
        np.add.at(bin_count, bins, 1)
        np.add.at(bin_confidence, bins, confidence)
        np.add.at(bin_correct, bins, (pred_np == y_np).astype(np.float64))
        n += bs
    confusion = confusion if confusion is not None else np.zeros((1, 1), dtype=np.int64)
    support = confusion.sum(axis=1)
    active = np.flatnonzero(support > 0)
    recalls: list[float] = []
    f1_scores: list[float] = []
    for cls in active:
        tp = float(confusion[cls, cls])
        recall = tp / max(1.0, float(support[cls]))
        precision = tp / max(1.0, float(confusion[:, cls].sum()))
        recalls.append(recall)
        f1_scores.append(2.0 * precision * recall / max(1e-12, precision + recall))
    ece = 0.0
    for idx in np.flatnonzero(bin_count):
        ece += (bin_count[idx] / max(1, n)) * abs(
            bin_correct[idx] / bin_count[idx] - bin_confidence[idx] / bin_count[idx]
        )
    accuracy = correct / max(1, n)
    balanced_accuracy = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    direction_confusion = np.zeros((3, 3), dtype=np.int64)
    action_to_direction = np.zeros(confusion.shape[0], dtype=np.int64)
    action_to_direction[1:4] = 1
    action_to_direction[4:7] = 2
    for target_action in range(confusion.shape[0]):
        for predicted_action in range(confusion.shape[1]):
            direction_confusion[
                action_to_direction[target_action], action_to_direction[predicted_action]
            ] += confusion[target_action, predicted_action]
    direction_recalls: list[float] = []
    direction_f1: list[float] = []
    for cls in range(3):
        tp = float(direction_confusion[cls, cls])
        recall = tp / max(1.0, float(direction_confusion[cls].sum()))
        precision = tp / max(1.0, float(direction_confusion[:, cls].sum()))
        direction_recalls.append(recall)
        direction_f1.append(2.0 * precision * recall / max(1e-12, precision + recall))
    direction_balanced_accuracy = float(np.mean(direction_recalls))
    direction_macro_f1 = float(np.mean(direction_f1))
    nll = policy_loss / max(1, n)
    composite = (
        0.25 * macro_f1
        + 0.20 * balanced_accuracy
        + 0.25 * direction_macro_f1
        + 0.20 * direction_balanced_accuracy
        + 0.05 * accuracy
        - 0.03 * nll
        - 0.02 * ece
    )
    return {
        "loss": total_loss / max(1, n),
        "policy_loss": nll,
        "action_accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "direction_balanced_accuracy": direction_balanced_accuracy,
        "direction_macro_f1": direction_macro_f1,
        "ece": float(ece),
        "composite_score": composite,
        "target_counts": support.tolist(),
        "prediction_counts": confusion.sum(axis=0).tolist(),
    }


# ---------------------------------------------------------------------------
# Behavioral Cloning
# ---------------------------------------------------------------------------


class BehavioralCloningTrainer:
    """Supervised BC on ``(obs, expert_action)`` pairs.

    The trainer is intentionally similar in shape to
    :class:`SupervisedTrainer` (S2) so that an S4 PPO load can
    accept its checkpoint without any code change.
    """

    def __init__(
        self,
        model: PolicyNetwork,
        loss: MultiTaskLoss,
        cfg: BCConfig,
    ) -> None:
        self.model = model
        self.loss = loss
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model.to(self.device)
        self.loss.to(self.device)
        self.opt = self._build_optimizer()
        self.sched = build_scheduler(self.opt, cfg.optim)
        self.history: list[dict] = []
        self.action_counts: list[int] = []

    def _build_optimizer(self) -> AdamW:
        groups: dict[tuple[bool, bool], list[torch.nn.Parameter]] = {}
        named = list(self.model.named_parameters()) + [
            (f"loss.{name}", value) for name, value in self.loss.named_parameters()
        ]
        for name, parameter in named:
            if not parameter.requires_grad:
                continue
            is_policy = name.startswith("heads.policy_logits.")
            lname = name.lower()
            use_decay = not (
                parameter.ndim < 2 or lname.endswith(".bias") or "norm" in lname or "bn" in lname
            )
            groups.setdefault((is_policy, use_decay), []).append(parameter)
        param_groups = [{
            "params": parameters,
            "lr": self.cfg.optim.lr * (1.0 if is_policy else self.cfg.backbone_lr_scale),
            "weight_decay": self.cfg.optim.weight_decay if use_decay else 0.0,
        } for (is_policy, use_decay), parameters in groups.items()]
        return AdamW(param_groups, betas=self.cfg.optim.betas)

    def _set_transfer_trainable(self, policy_only: bool) -> None:
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(not policy_only or name.startswith("heads.policy_logits."))

    def fit(
        self,
        df: pd.DataFrame,
        expert: ExpertPolicy,
        spec: Optional[SampleSpec] = None,
    ) -> dict:
        """Run BC on ``df`` labelled by ``expert``."""
        set_seed(self.cfg.seed)
        spec = spec or SampleSpec()
        ds, actions = _make_dataset(df, spec, expert)
        return self.fit_labeled([_LabeledMarketDataset(ds, actions)])

    def fit_market_datasets(
        self,
        datasets: list[MarketDataset],
        expert_factory,
        val_datasets: Optional[list[MarketDataset]] = None,
    ) -> dict:
        """Train jointly on independent market segments.

        A fresh expert is required per segment because forward-label experts
        cache labels internally and equal-length symbols must never share them.
        """
        def label_all(items: list[MarketDataset]) -> list[Dataset]:
            labeled: list[Dataset] = []
            for dataset in items:
                expert = expert_factory()
                actions = _expert_actions_for_dataset(dataset, expert)
                labeled.append(_LabeledMarketDataset(dataset, actions))
            return labeled

        return self.fit_labeled(
            label_all(datasets),
            label_all(val_datasets) if val_datasets else None,
        )

    def fit_labeled(
        self,
        datasets: list[Dataset],
        val_datasets: Optional[list[Dataset]] = None,
    ) -> dict:
        set_seed(self.cfg.seed)
        labeled: Dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
        raw_counts = np.zeros(self.model.cfg.n_actions, dtype=np.int64)
        for dataset in datasets:
            actions = np.asarray(dataset.actions, dtype=np.int64)
            raw_counts += np.bincount(actions, minlength=len(raw_counts))[:len(raw_counts)]
        self.action_counts = raw_counts.tolist()
        if self.cfg.policy_class_balance == "sqrt_inverse":
            counts = raw_counts
            active = counts > 0
            weights = np.zeros_like(counts, dtype=np.float32)
            weights[active] = np.sqrt(counts[active].sum() / (active.sum() * counts[active]))
            weights[active] /= weights[active].mean()
            self.loss.set_policy_class_weights(torch.tensor(weights, device=self.device))
            logger.info("BC action counts=%s weights=%s", counts.tolist(), weights.tolist())
        elif self.cfg.policy_class_balance != "none":
            raise ValueError(f"unknown policy_class_balance={self.cfg.policy_class_balance!r}")
        loader = build_dataloader(
            labeled, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=self.cfg.num_workers, collate_fn=_batched_collate, drop_last=True,
        )
        val_loader = None
        if val_datasets:
            val_labeled: Dataset = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
            val_loader = build_dataloader(
                val_labeled, batch_size=self.cfg.batch_size, shuffle=False,
                num_workers=self.cfg.num_workers, collate_fn=_batched_collate, drop_last=False,
            )
        history: list[dict] = []
        timer = Timer()
        best_val = float("-inf")
        bad_epochs = 0
        for epoch in range(self.cfg.epochs):
            self._set_transfer_trainable(epoch < self.cfg.freeze_backbone_epochs)
            timer.start()
            avg = _train_bc_one_epoch(
                self.model, self.loss, self.opt, loader,
                self.device, self.cfg.grad_clip, self.cfg.log_every, epoch, self.sched,
                policy_only=epoch < self.cfg.freeze_backbone_epochs,
            )
            timer.stop()
            record = {"epoch": epoch, "loss": avg, "elapsed_s": timer.elapsed}
            if val_loader is not None:
                record["val"] = _evaluate_bc(self.model, self.loss, val_loader, self.device)
                metric = self.cfg.champion_metric
                if metric not in record["val"]:
                    raise ValueError(f"unknown S2b champion metric: {metric}")
                val_score = float(record["val"][metric])
                if val_score > best_val + self.cfg.early_stopping_min_delta:
                    best_val = val_score
                    bad_epochs = 0
                    if self.cfg.best_checkpoint:
                        self.save(self.cfg.best_checkpoint)
                else:
                    bad_epochs += 1
            history.append(record)
            self.history = history
            logger.info(
                "bc epoch %d done in %.1fs, loss=%.5f%s",
                epoch, timer.elapsed, avg,
                (
                    f" val_policy={record['val']['policy_loss']:.5f}"
                    f" val_action_acc={record['val']['action_accuracy']:.4f}"
                    f" val_bal_acc={record['val']['balanced_accuracy']:.4f}"
                    f" val_macro_f1={record['val']['macro_f1']:.4f}"
                    f" val_dir_bal_acc={record['val']['direction_balanced_accuracy']:.4f}"
                    f" val_dir_macro_f1={record['val']['direction_macro_f1']:.4f}"
                    f" val_ece={record['val']['ece']:.4f}"
                    f" val_composite={record['val']['composite_score']:.4f}"
                    if "val" in record else ""
                ),
            )
            timer.reset()
            if val_loader is not None and self.cfg.early_stopping_patience > 0 and bad_epochs >= self.cfg.early_stopping_patience:
                logger.info("bc early stopping at epoch %d; best_%s=%.5f", epoch, self.cfg.champion_metric, best_val)
                break
        if self.cfg.checkpoint:
            self.save(self.cfg.checkpoint)
        return {"history": history}

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cfg_dict = self.model.cfg.__dict__.copy()
        if "vision_channels" in cfg_dict and isinstance(cfg_dict["vision_channels"], tuple):
            cfg_dict["vision_channels"] = list(cfg_dict["vision_channels"])
        torch.save({
            "model": self.model.state_dict(),
            "loss": self.loss.state_dict(),
            "config": cfg_dict,
            "model_config": cfg_dict,  # canonical name
            "trainer_state": {"history": self.history},
            "checkpoint_meta": {
                "stage": "s2b_imitation",
                "trading_policy_ready": True,
                "policy_head_trained": True,
                "policy_training": "behavioral_cloning",
                "expert": self.cfg.expert_name,
                "action_counts": self.action_counts,
                "action_space_note": (
                    "Symmetric utility BC trains flat and target exposure at 25/50/100; "
                    "SKIP and PARTIAL_CLOSE remain state-dependent S4 actions."
                    if self.cfg.expert_name == "symmetric_utility"
                    else "Legacy BC expert action space."
                ),
                "source_checkpoint": self.cfg.source_checkpoint,
                "dataset": {
                    "root": self.cfg.dataset_root,
                    "manifest_checksum": self.cfg.dataset_manifest_checksum,
                },
            },
        }, p)
        logger.info("bc checkpoint saved to %s", p)


# ---------------------------------------------------------------------------
# DAgger (Dataset Aggregation)
# ---------------------------------------------------------------------------


def _rollout_policy_for_aggregation(
    model: PolicyNetwork,
    env: TradingEnv,
    expert: ExpertPolicy,
    n_episodes: int,
    max_steps: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[list[dict], list[int], list[float]]:
    """Roll out the current policy in ``env`` and collect
    ``(obs, expert_action)`` pairs on every visited state.

    Returns the list of observation dicts, the list of expert
    actions, and the per-episode returns (clipped to a finite range
    so a degenerate policy does not poison the logging statistics).
    """
    obs_list: list[dict] = []
    actions_list: list[int] = []
    ep_returns: list[float] = []
    model.eval()
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        ep_ret = 0.0
        steps = 0
        for _ in range(max_steps):
            t = env._t  # absolute bar index in env.df
            expert_action = int(expert.predict(env.df, t))
            obs_list.append(obs)
            actions_list.append(expert_action)
            chart = torch.from_numpy(obs["chart"]).unsqueeze(0).to(device)
            num = torch.from_numpy(obs["numeric"]).unsqueeze(0).to(device)
            ctx = torch.from_numpy(obs["context"]).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(chart=chart, numeric=num, context=ctx)
                logits = out["policy_logits"]
                if not torch.isfinite(logits).all():
                    action = int(rng.integers(0, env.action_space.n))
                else:
                    action = int(torch.distributions.Categorical(logits=logits).sample().item())
            obs, reward, terminated, truncated, _ = env.step(action)
            if np.isfinite(reward):
                ep_ret += float(reward)
            steps += 1
            if terminated or truncated:
                break
        # Clip pathological returns so logs stay readable. The DAgger
        # loop cares about the *aggregated dataset*, not the reward.
        if not np.isfinite(ep_ret):
            ep_ret = 0.0
        ep_ret = float(np.clip(ep_ret, -1e6, 1e6))
        ep_returns.append(ep_ret)
    return obs_list, actions_list, ep_returns


class _AggregatedPairs(Dataset):
    """Holds aggregated ``(obs, action)`` pairs across DAgger rounds.

    Internally stores the multimodal tensors plus the per-pair
    action label. The auxiliary head labels are filled with safe
    defaults so the multi-task loss can run without per-pair
    triples — only the ``policy`` term is meaningful for these
    aggregated pairs.
    """

    def __init__(self, charts: np.ndarray, nums: np.ndarray, ctxs: np.ndarray, actions: np.ndarray):
        if not (len(charts) == len(nums) == len(ctxs) == len(actions)):
            raise ValueError("charts / nums / ctxs / actions must have the same length")
        self.charts = charts
        self.nums = nums
        self.ctxs = ctxs
        self.actions = actions

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, i: int) -> dict:
        T = self.nums.shape[1]
        F = self.nums.shape[2]
        return {
            "chart": torch.from_numpy(self.charts[i]).float(),
            "numeric": torch.from_numpy(self.nums[i]).float(),
            "context": torch.from_numpy(self.ctxs[i]).float(),
            "label_dir": torch.tensor(1, dtype=torch.long),  # 1 == "0" after offset
            "label_vol": torch.tensor(0.0, dtype=torch.float32),
            "label_risk": torch.tensor(0.0, dtype=torch.float32),
            "label_regime": torch.tensor(0, dtype=torch.long),
            "label_ret": torch.tensor(0.0, dtype=torch.float32),
            "mask": torch.ones(T, dtype=torch.bool),
            "action": torch.tensor(int(self.actions[i]), dtype=torch.long),
            "meta": {"t": -1, "ts": "", "instrument": "agg"},
        }


class DAggerTrainer:
    """DAgger (Dataset Aggregation) trainer.

    Algorithm:

    1. **Round 0 (BC warmup).**  Standard BC on the static expert
       labels. Produces a policy that already trades non-randomly.
    2. **For each round k = 1, ..., n_rounds-1:**

       a. Roll out the current policy in a :class:`TradingEnv`
          for ``rollout_episodes_per_round`` short episodes.
       b. At every visited state, query the expert for what it
          *would have done*. Append these ``(obs, expert_action)``
          pairs to the aggregated dataset.
       c. Re-train (BC) for ``epochs_per_round`` epochs on the
          aggregated dataset.

    The aggregated dataset is **the union** of the static BC
    examples and all DAgger-augmented pairs from previous rounds.
    """

    def __init__(
        self,
        model: PolicyNetwork,
        expert: ExpertPolicy,
        cfg: DAggerConfig,
    ) -> None:
        self.model = model
        self.expert = expert
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model.to(self.device)
        # BC sub-trainer (re-built each round so it gets a fresh
        # optimiser / scheduler bound to the latest model state).
        self._bc_loss_fn: Optional[MultiTaskLoss] = None
        self._bc_trainer: Optional[BehavioralCloningTrainer] = None
        # Aggregated data (filled across rounds).
        self._agg_charts: list[np.ndarray] = []
        self._agg_nums: list[np.ndarray] = []
        self._agg_ctxs: list[np.ndarray] = []
        self._agg_actions: list[int] = []
        self._rng = np.random.default_rng(cfg.seed)

    def _build_static_loader(
        self,
        df: pd.DataFrame,
        spec: SampleSpec,
    ) -> DataLoader:
        """The static BC dataset (computed once)."""
        ds, actions = _make_dataset(df, spec, self.expert)
        labeled = _LabeledMarketDataset(ds, actions)
        return build_dataloader(
            labeled, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=0, collate_fn=_batched_collate, drop_last=True,
        ), ds, labeled

    def fit(
        self,
        df: pd.DataFrame,
        spec: Optional[SampleSpec] = None,
    ) -> DAggerResult:
        """Run the DAgger loop on ``df`` and return aggregate metrics."""
        set_seed(self.cfg.seed)
        spec = spec or SampleSpec()
        rounds: list[DAggerRoundMetrics] = []

        # --- Round 0: BC warmup on the static dataset ---
        static_loader, ds, labeled = self._build_static_loader(df, spec)
        n_static = len(labeled)
        self._bc_loss_fn = MultiTaskLoss(self.cfg.loss_weights)
        bc_cfg = BCConfig(
            epochs=self.cfg.epochs_per_round,
            batch_size=self.cfg.batch_size,
            grad_clip=self.cfg.grad_clip,
            log_every=self.cfg.log_every,
            device=self.cfg.device,
            seed=self.cfg.seed,
            optim=self.cfg.optim,
        )
        self._bc_trainer = BehavioralCloningTrainer(self.model, self._bc_loss_fn, bc_cfg)
        # Train the BC sub-trainer on the static loader.
        timer = Timer()
        timer.start()
        # Drive BC manually so we can use the pre-built loader.
        for epoch in range(self.cfg.epochs_per_round):
            avg = _train_bc_one_epoch(
                self.model, self._bc_loss_fn,
                self._bc_trainer.opt, static_loader,
                self.device, self.cfg.grad_clip, self.cfg.log_every, epoch,
                self._bc_trainer.sched,
            )
        timer.stop()
        rounds.append(DAggerRoundMetrics(
            round_idx=0, n_aggregated=n_static, n_new_pairs=0,
            bc_loss=avg, elapsed_s=timer.elapsed,
        ))

        # --- Rounds 1..n_rounds-1: roll out + aggregate + retrain ---
        for k in range(1, self.cfg.n_rounds):
            env = TradingEnv(df, cfg=self.cfg.env_cfg)
            new_obs, new_actions, ep_returns = _rollout_policy_for_aggregation(
                self.model, env, self.expert,
                n_episodes=self.cfg.rollout_episodes_per_round,
                max_steps=self.cfg.max_steps_per_episode,
                rng=self._rng,
                device=self.device,
            )
            n_new = len(new_actions)
            # Append to the aggregated pool.
            for o, a in zip(new_obs, new_actions):
                self._agg_charts.append(o["chart"])
                self._agg_nums.append(o["numeric"])
                self._agg_ctxs.append(o["context"])
                self._agg_actions.append(int(a))
            n_agg = n_static + len(self._agg_actions)
            # Build a combined DataLoader that mixes static and
            # aggregated data. We re-build it from scratch each round
            # so the (obs, action) pairs are seen in random order.
            if self._agg_charts:
                agg_charts = np.stack(self._agg_charts, axis=0)
                agg_nums = np.stack(self._agg_nums, axis=0)
                agg_ctxs = np.stack(self._agg_ctxs, axis=0)
                agg_acts = np.asarray(self._agg_actions, dtype=np.int64)
            else:
                # Empty aggregated pool: use the static loader's shapes
                # for the bogus placeholders (it'll never be sampled).
                sample = labeled[0]
                agg_charts = np.zeros((0,) + tuple(sample["chart"].shape), dtype=np.float32)
                agg_nums = np.zeros((0,) + tuple(sample["numeric"].shape), dtype=np.float32)
                agg_ctxs = np.zeros((0,) + tuple(sample["context"].shape), dtype=np.float32)
                agg_acts = np.zeros(0, dtype=np.int64)
            agg_ds = _AggregatedPairs(agg_charts, agg_nums, agg_ctxs, agg_acts)
            # Concat-loader: chain static and aggregated.
            combined = torch.utils.data.ConcatDataset([labeled, agg_ds])
            combined_loader = build_dataloader(
                combined, batch_size=self.cfg.batch_size, shuffle=True,
                num_workers=0, collate_fn=_batched_collate, drop_last=True,
            )
            timer.start()
            # Refresh the BC trainer's optimiser (so the BC loss's
            # running state is consistent).
            self._bc_trainer.opt = build_optimizer(
                [p for p in self.model.parameters() if p.requires_grad]
                + [p for p in self._bc_loss_fn.parameters() if p.requires_grad],
                self.cfg.optim,
            )
            self._bc_trainer.sched = build_scheduler(self._bc_trainer.opt, self.cfg.optim)
            avg = 0.0
            for epoch in range(self.cfg.epochs_per_round):
                avg = _train_bc_one_epoch(
                    self.model, self._bc_loss_fn,
                    self._bc_trainer.opt, combined_loader,
                    self.device, self.cfg.grad_clip, self.cfg.log_every, epoch,
                    self._bc_trainer.sched,
                )
            timer.stop()
            rounds.append(DAggerRoundMetrics(
                round_idx=k, n_aggregated=int(n_agg), n_new_pairs=int(n_new),
                bc_loss=avg, elapsed_s=timer.elapsed,
            ))
            mean_ret = float(np.mean(ep_returns)) if ep_returns else 0.0
            logger.info(
                "dagger round %d: n_aggregated=%d n_new=%d bc_loss=%.4f mean_ep_return=%.4f",
                k, n_agg, n_new, avg, mean_ret,
            )
        if self.cfg.checkpoint:
            self._bc_trainer.save(self.cfg.checkpoint)
        return DAggerResult(
            rounds=rounds,
            final_loss=rounds[-1].bc_loss if rounds else float("nan"),
        )


__all__ = [
    "BCConfig",
    "BehavioralCloningTrainer",
    "DAggerConfig",
    "DAggerResult",
    "DAggerRoundMetrics",
    "DAggerTrainer",
]
