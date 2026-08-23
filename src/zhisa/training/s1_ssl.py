"""S1: Self-supervised pretraining for the multimodal policy.

This module implements the S1 stage from ``CONCEPT.md`` (§5.2). The
goal is to pretrain the multimodal encoder stack on **unlabelled**
market data so that the S2 supervised trainer starts from a richer
initialisation. Four complementary objectives are combined:

1. **Temporal contrastive (CPC-style).**  The model encodes the current
   market state into ``z_t`` and predicts the projected next state. The
   next-bar state is encoded by an exponential moving average (EMA)
   teacher into ``z_{t+1}``. We
   maximise the cosine similarity of matched (t, t+1) pairs against
   the rest of the in-batch negatives via InfoNCE.
2. **Masked numeric modeling.**  A random fraction of the numeric
   encoder's input patches are zeroed out, the encoder is forced to
   reconstruct their values from the surrounding context. This teaches
   the numeric encoder local temporal structure.
3. **Cross-modal alignment.**  Vision and numeric embeddings are
   pulled together (positive pair) and pushed apart from the rest of
   the batch (negatives) via symmetric InfoNCE. This builds a shared
   semantic space between the chart and the OHLCV feature stream.
4. **EMA teacher.**  A momentum copy of the student encoders produces
   stable targets for the contrastive losses and acts as a regulariser.

The implementation reuses the project's :class:`PolicyNetwork` so the
S2 trainer can ``load_state_dict`` the pretrained encoder weights
directly.
"""
from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from zhisa.data.dataset import multimodal_collate
from zhisa.models.policy import PolicyNetwork
from zhisa.rendering.augmentations import KeyedAugmentor
from zhisa.utils.logging import get_logger
from zhisa.utils.timing import Timer


def _iter_leaf_datasets(dataset):
    """Yield the innermost (MarketDataset-like) datasets of a train/val tree."""
    if isinstance(dataset, ConcatDataset):
        for sub in dataset.datasets:
            yield from _iter_leaf_datasets(sub)
    else:
        yield dataset

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SSLConfig:
    """Hyperparameters for the S1 self-supervised pretraining."""

    projection_dim: int = 64
    hidden_dim: int = 128
    temperature: float = 0.1
    mask_ratio: float = 0.4
    ema_decay: float = 0.996
    weight_temporal: float = 1.0
    weight_masked: float = 1.0
    weight_alignment: float = 0.5
    epochs: int = 1
    batch_size: int = 32
    grad_clip: float = 1.0
    log_every: int = 50
    checkpoint: Optional[str] = None
    device: str = "cpu"
    seed: int = 0
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 100
    temporal_horizon: int = 1
    val_max_batches: int = 32
    checkpoint_every_steps: int = 500
    best_checkpoint: Optional[str] = None
    dataset_root: Optional[str] = None
    dataset_timeframe: Optional[str] = None
    dataset_manifest_checksum: Optional[str] = None
    # Render provenance / byte-equivalence contract (see data.chart_store).
    renderer_version: Optional[str] = None
    render_spec_hash: Optional[str] = None
    render_fingerprint: Optional[str] = None
    render_store_checksum: Optional[str] = None
    # Deterministic keyed chart augmentation (see rendering.augmentations).
    # Empty tuple disables augmentation; keys derive from (epoch, step, index).
    augment_transforms: tuple = ()
    augment_strength: float = 0.05
    augment_crop_frac: float = 0.85
    augment_noise_std: float = 0.01
    use_ema_teacher: bool = True
    use_masked_modeling: bool = True
    use_temporal_contrast: bool = True
    use_cross_modal: bool = True
    # Weight of the TRUNK-level alignment term (cos similarity on raw
    # vision-pooled vs numeric-CLS embeddings, NO projections). Projection
    # heads can satisfy the proj-space InfoNCE alone (measured: proj-cos
    # ~0.95 while trunk-cos stays negative); this term forces the trunk
    # itself to align. 0.0 disables (canonical behaviour).
    weight_trunk_align: float = 0.0
    trunk_align_momentum: float = 0.0  # >0: soft target from EMA of numeric CLS
    # Instrument-embedding spread: penalises the mean off-diagonal cosine of
    # ``context.instrument_emb`` so symbols stop colliding (measured pairs
    # at 0.44-0.56 with silhouette 0.007). 0.0 disables.
    instrument_contrast_w: float = 0.0
    # ---- Reconstruction upgrade (P1, evidence-gated) -------------------------
    # The masked reconstructor on heavy checkpoints collapses to *shrinkage*:
    # linear fit pred~0.28*target (corr 0.60) -> gain stuck ~1.2-1.4 vs 2.0+.
    # Fixes (all opt-in, default = canonical behaviour):
    #   * recon_depth>1 : deeper residual MLP head (capacity -> higher corr);
    #   * recon_use_gain: learnable per-feature output scale (init 1) that lets
    #     the head calibrate raw amplitude instead of shrinking toward the mean;
    #   * masked_target_norm: train in per-feature standardized space so the
    #     loss is not dominated by a few large-variance channels.
    recon_depth: int = 1
    recon_use_norm: bool = True
    recon_use_gain: bool = False
    masked_target_norm: bool = False
    # ---- Gradient rebalance (P2) ---------------------------------------------
    # vision.* grads are scaled by this factor AFTER backward (measured: vision
    # gets ~2% of grad while numeric gets ~40% -> z is numeric-dominated, which
    # drives both the numeric-perturb angle and the recon weakness). 1.0 = off.
    vision_grad_scale: float = 1.0
    # ---- z-level instrument contrast (P3) ------------------------------------
    # InfoNCE on the TRUNK embedding z keyed by instrument identity: forces the
    # *encoded* embedding (not just the instrument_emb table) to separate
    # symbols. Measured: flipping id moves z by ~5 deg, hence silhouette ~0.
    instrument_z_contrast_w: float = 0.0
    # ---- Temporal CPC v2 (negatives engineering, all train-only). ----
    # MoCo-style memory bank of L2-normalised teacher target projections.
    # 0 disables the bank (canonical v1 behaviour: in-batch negatives only).
    temporal_bank_size: int = 0
    # Steps before the bank is consulted (avoids a near-empty queue).
    temporal_bank_warmup: int = 128
    # Hard-negative offsets in units of temporal_horizon, sampled from the
    # SAME instrument, e.g. (-1, 2) => windows at t-h and t+2h. Empty
    # tuple disables hard negatives. Costs one extra teacher forward
    # (2B windows concatenated into a single call).
    temporal_hard_offsets: tuple = ()
    # ---- LR schedule. "constant" reproduces the canonical v1 behaviour
    # (linear warmup then fixed lr). "cosine" decays to
    # ``cosine_min_scale * lr`` over ``total_steps`` (auto-estimated from
    # epochs x steps-per-epoch when total_steps == 0).
    lr_schedule: str = "constant"
    cosine_min_scale: float = 0.003
    total_steps: int = 0


class TemporalPairDataset(Dataset):
    """Expose causal ``(sample[t], sample[t+horizon])`` pairs.

    A ``ConcatDataset`` is handled component-by-component so a pair can never
    cross from the end of one instrument into the start of another.

    Every returned item is stamped with a global id ``meta["gid"] =
    (leaf_index, local_index)`` so the contrastive loss can (a) build a
    deduplicated memory bank and (b) sample same-instrument hard negatives
    at arbitrary step offsets (see :meth:`offset_item`).
    """

    def __init__(self, dataset: Dataset, horizon: int = 1) -> None:
        if horizon < 1:
            raise ValueError("temporal horizon must be >= 1")
        self.horizon = int(horizon)
        self.datasets = (
            list(dataset.datasets) if isinstance(dataset, ConcatDataset) else [dataset]
        )
        self.lengths = [max(0, len(ds) - self.horizon) for ds in self.datasets]
        total = 0
        self.cumulative_sizes: list[int] = []
        for length in self.lengths:
            total += length
            self.cumulative_sizes.append(total)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def _stamp(self, dataset_idx: int, local_idx: int, horizon_shift: int = 0):
        """Fetch ``dataset[local_idx + shift]`` with a global id stamped.

        Items are expected to be dicts (MarketDataset contract); any other
        item type (legacy tuple-based fakes) is passed through untouched so
        old callers keep their behaviour.
        """
        ds = self.datasets[dataset_idx]
        idx = local_idx + horizon_shift
        item = ds[idx]
        if not isinstance(item, dict):
            return item
        item = dict(item)
        meta = dict(item.get("meta", {}) or {})
        meta["gid"] = (int(dataset_idx), int(idx))
        item["meta"] = meta
        return item

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        dataset_idx = bisect_right(self.cumulative_sizes, index)
        previous = self.cumulative_sizes[dataset_idx - 1] if dataset_idx > 0 else 0
        local_idx = index - previous
        current = self._stamp(dataset_idx, local_idx, 0)
        future = self._stamp(dataset_idx, local_idx, self.horizon)
        return current, future

    def offset_item(self, dataset_idx: int, local_idx: int, step_offset: int):
        """Same-instrument shifted window for hard negatives, or None.

        ``step_offset`` is in raw sample steps (NOT horizon units); the caller
        multiplies by ``horizon``. Bounds-checked per leaf; deterministic —
        it depends only on ``(dataset_idx, local_idx, step_offset)``.
        """
        ds = self.datasets[dataset_idx]
        idx = local_idx + step_offset
        if idx < 0 or idx >= len(ds):
            return None
        return self._stamp(dataset_idx, idx, 0)


def temporal_pair_collate(batch) -> dict:
    current, future = zip(*batch)
    current_batch = multimodal_collate(current)
    future_batch = multimodal_collate(future)
    gids = [it["meta"].get("gid") for it in current]
    future_gids = [it["meta"].get("gid") for it in future]
    return {
        "chart": current_batch.chart,
        "numeric": current_batch.numeric,
        "context": current_batch.context,
        "instrument_id": current_batch.instrument_id,
        "future_chart": future_batch.chart,
        "future_numeric": future_batch.numeric,
        "future_context": future_batch.context,
        "gids": gids,
        "future_gids": future_gids,
    }


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _ProjectionHead(nn.Module):
    """A 2-layer MLP projection head used by all three objectives."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _MaskedReconstructor(nn.Module):
    """Predicts the original patch contents from the encoder's outputs.

    The numeric encoder produces a sequence of tokens (CLS + patches).
    We attach a head that maps each token back to the flattened patch
    values. Only the masked positions contribute to the loss.

    ``depth=1, use_gain=False`` reproduces the canonical single-Linear head
    exactly (state-dict compatible). ``depth>1`` adds residual MLP blocks,
    and ``use_gain=True`` adds a learnable per-feature output scale (init 1)
    that lets the head calibrate raw amplitude instead of shrinking toward
    zero (measured slope ~0.28 / corr ~0.60 in heavy checkpoints).
    """

    def __init__(
        self,
        d_model: int,
        patch_size: int,
        in_features: int,
        *,
        depth: int = 1,
        use_residual_norm: bool = True,
        use_gain: bool = False,
        gain_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_features = in_features
        self.depth = int(depth)
        self.use_gain = bool(use_gain)
        blocks = []
        prev = d_model
        for _ in range(max(0, int(depth) - 1)):
            blocks.append(nn.Linear(prev, d_model))
            if use_residual_norm:
                blocks.append(nn.LayerNorm(d_model))
            blocks.append(nn.GELU())
            prev = d_model
        self.net = nn.Sequential(*blocks) if blocks else nn.Identity()
        self.head = nn.Linear(prev, patch_size * in_features)
        if self.use_gain:
            self.gain = nn.Parameter(
                torch.full((patch_size * in_features,), float(gain_init))
            )
        else:
            self.register_parameter("gain", None)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, 1 + n_patches, d_model)
        out = self.net(tokens)
        out = self.head(out)
        if self.gain is not None:
            out = out * self.gain
        return out


class EMATeacher:
    """Maintains a momentum copy of the student's encoder parameters.

    Only the encoders (vision, numeric, context, fusion) are tracked;
    the heads and SSL-specific projections are student-only. The
    teacher is updated as ``teacher = decay * teacher + (1-decay) * student``
    after every optimisation step.
    """

    def __init__(self, model: PolicyNetwork, decay: float = 0.996) -> None:
        self.decay = decay
        self.teacher = deepcopy(model)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    @torch.no_grad()
    def update(self, model: PolicyNetwork) -> None:
        d = self.decay
        for tp, sp in zip(self.teacher.parameters(), model.parameters()):
            if not sp.requires_grad:
                continue
            tp.mul_(d).add_(sp.detach(), alpha=1.0 - d)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "teacher": self.teacher.state_dict()}

    def load_state_dict(self, sd: dict) -> None:
        self.decay = float(sd["decay"])
        # The teacher is a copy of the student policy; head shapes can
        # differ from the current model (e.g. n_actions). Use strict=False
        # so we tolerate such mismatches — the SSL trainer will refresh
        # the teacher in the next step anyway.
        self.teacher.load_state_dict(sd["teacher"], strict=False)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------


def info_nce(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    temperature: float = 0.1,
    max_logit: float = 50.0,
    extra_negatives: Optional[torch.Tensor] = None,
    row_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """InfoNCE between two L2-normalised projection batches.

    Both ``anchor`` and ``positive`` are expected to be shape ``(B, D)``.
    The positive pair is the diagonal ``(i, i)``; all other entries
    are negatives. ``extra_negatives`` may be shape ``(B, K, D)`` (or
    ``(B*K, D)``) — additional rows concatenated after the positive
    block (memory-bank queue and/or hard negatives). ``row_mask``, when
    given, is a boolean tensor ``(B, B + n_extra_rows)`` marking rows to
    EXCLUDE (logits set to a large negative so the row cannot win). The
    diagonal positive row is never excluded (callers must leave it
    unmasked). The logits are clamped to ``[-max_logit, max_logit]``
    to keep cross-entropy numerically stable when the projection head
    has not yet been warmed up.
    """
    a = F.normalize(anchor, dim=-1)
    p = F.normalize(positive, dim=-1)
    blocks = [p]
    if extra_negatives is not None:
        if extra_negatives.dim() == 3:
            extra_negatives = extra_negatives.reshape(-1, extra_negatives.size(-1))
        blocks.append(F.normalize(extra_negatives, dim=-1))
    neg = torch.cat(blocks, dim=0)
    logits = a @ neg.t() / max(temperature, 1e-6)
    logits = logits.clamp(min=-max_logit, max=max_logit)
    if row_mask is not None:
        if row_mask.dtype != torch.bool:
            row_mask = row_mask.bool()
        # Never mask the diagonal (the positive row itself): the mask built
        # by callers exempts it by construction, but we enforce it here as
        # defence-in-depth (a fully-masked row would produce a NaN CE).
        # The diagonal lives in the FIRST B columns (positive block).
        diag = torch.zeros_like(row_mask)
        diag[:, :a.size(0)] = torch.eye(a.size(0), dtype=torch.bool, device=a.device)
        row_mask = row_mask & ~diag
        logits = logits.masked_fill(row_mask, -1e9)
    labels = torch.arange(a.size(0), device=a.device)
    return F.cross_entropy(logits, labels)


def masked_numeric_loss(
    numeric_encoder: nn.Module,
    reconstructor: _MaskedReconstructor,
    x: torch.Tensor,
    mask_ratio: float = 0.4,
    *,
    target_norm: bool = False,
    target_mean: Optional[torch.Tensor] = None,
    target_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mask random patches of the numeric input, encode, and predict them.

    The numeric encoder is :class:`zhisa.models.encoders.numeric.NumericEncoder`
    which returns ``(cls, tokens)`` of shape ``(B, 1+n_patches, d_model)``.
    Only the non-CLS positions are considered for masking.

    ``target_norm=True`` standardizes patch targets per feature channel using
    the batch stats (or the caller-provided ``target_mean``/``target_scale``).
    This removes the amplitude bias that dominates the raw MSE (a few
    large-variance channels) and is the training-side half of the anti-shrinkage
    fix — the reconstruction head then optimises correlation, not raw scale.
    """
    B, T, F_ = x.shape
    cfg = numeric_encoder.cfg
    n_patches = cfg.window // cfg.patch_size
    patch = cfg.patch_size

    # Patchify the input so we can mask and reconstruct at the patch level.
    patches = x.view(B, n_patches, patch, F_).reshape(B, n_patches, -1)

    # Random per-patch binary mask. 0 = masked, 1 = visible.
    mask = torch.bernoulli(
        torch.full((B, n_patches), 1.0 - mask_ratio, device=x.device)
    )
    # Guarantee at least one visible patch so the encoder has signal.
    visible_any = mask.sum(dim=1) > 0
    if not bool(visible_any.all()):
        for i in torch.where(~visible_any)[0].tolist():
            mask[i, 0] = 1.0
    mask_expanded = mask.unsqueeze(-1)  # (B, n_patches, 1)
    masked_patches = patches * mask_expanded

    # Rebuild the masked window and re-encode.
    masked_window = masked_patches.view(B, T, F_)
    _, tokens = numeric_encoder(masked_window)

    # Predict original patch values at all positions.
    pred = reconstructor(tokens)  # (B, 1+n_patches, patch*F)
    # Drop the summary slot. Its position is governed by
    # ``NumericEncoderConfig.summary_position`` ("end" = LAST token when the
    # encoder is causal, otherwise "front" = FIRST/CLS). Branching on the raw
    # ``causal`` flag is fragile: the encoder may legally use summary_position
    # "end" with causal=False, which would slice the wrong slot.
    summary_end = (
        getattr(cfg, "summary_position", None) == "end"
        or getattr(cfg, "causal", False)
    )
    if summary_end:
        pred_patches = pred[:, :n_patches, :]
    else:
        pred_patches = pred[:, 1:, :]
    target = patches.view(B, n_patches, -1)
    if target_norm:
        if target_mean is None or target_scale is None:
            tm = target.mean(dim=(0, 1), keepdim=True)
            ts = target.std(dim=(0, 1), keepdim=True).clamp_min(1e-3) + 1e-6
        else:
            tm = target_mean.to(x.device)
            ts = target_scale.to(x.device)
        pred_patches = (pred_patches - tm) / ts
        target = (target - tm) / ts
    # MSE only on masked positions.
    loss_per_patch = (pred_patches - target).pow(2).mean(dim=-1)  # (B, n_patches)
    masked_positions = mask < 0.5
    n_masked = masked_positions.float().sum().clamp_min(1.0)
    return (loss_per_patch * masked_positions.float()).sum() / n_masked


# ---------------------------------------------------------------------------
# Pretrainer
# ---------------------------------------------------------------------------


class SSLPretrainer:
    """The full S1 self-supervised pretrainer.

    Holds the :class:`PolicyNetwork` and augments it with the SSL
    projection heads, masked reconstructor, and EMA teacher. The
    public method :meth:`fit` runs a standard training loop on a
    :class:`MarketDataset` (or any compatible dataset).
    """

    def __init__(
        self,
        model: PolicyNetwork,
        cfg: Optional[SSLConfig] = None,
    ) -> None:
        self.cfg = cfg or SSLConfig()
        self.model = model
        self.device = torch.device(self.cfg.device)
        self.model.to(self.device)

        D = model.cfg.embed_dim

        # Three projection heads feeding the three InfoNCE losses.
        self.proj_temporal = _ProjectionHead(D, self.cfg.hidden_dim, self.cfg.projection_dim)
        self.temporal_predictor = _ProjectionHead(
            self.cfg.projection_dim,
            self.cfg.hidden_dim,
            self.cfg.projection_dim,
        )
        self.proj_vision = _ProjectionHead(D, self.cfg.hidden_dim, self.cfg.projection_dim)
        self.proj_numeric = _ProjectionHead(D, self.cfg.hidden_dim, self.cfg.projection_dim)

        # Masked numeric reconstructor.
        self.reconstructor = _MaskedReconstructor(
            d_model=model.numeric.cfg.d_model,
            patch_size=model.numeric.cfg.patch_size,
            in_features=model.numeric.cfg.in_features,
            depth=int(self.cfg.recon_depth),
            use_residual_norm=bool(self.cfg.recon_use_norm),
            use_gain=bool(self.cfg.recon_use_gain),
        )

        self.proj_temporal.to(self.device)
        self.temporal_predictor.to(self.device)
        self.proj_vision.to(self.device)
        self.proj_numeric.to(self.device)
        self.reconstructor.to(self.device)

        # The temporal target projection must move with the EMA teacher, not
        # with the student optimizer. Otherwise the supposedly stable target
        # changes immediately after every student update.
        self.target_proj_temporal = deepcopy(self.proj_temporal).to(self.device)
        for p in self.target_proj_temporal.parameters():
            p.requires_grad_(False)
        self.target_proj_temporal.eval()

        # EMA teacher.
        self.teacher: Optional[EMATeacher] = None
        if self.cfg.use_ema_teacher:
            self.teacher = EMATeacher(model, decay=self.cfg.ema_decay)
            self.teacher.teacher.to(self.device)

        # Optimiser & LR schedule.
        params = (
            list(model.parameters())
            + list(self.proj_temporal.parameters())
            + list(self.temporal_predictor.parameters())
            + list(self.proj_vision.parameters())
            + list(self.proj_numeric.parameters())
            + list(self.reconstructor.parameters())
        )
        self.opt = torch.optim.AdamW(
            params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        self._step = 0
        self._completed_epochs = 0
        self._history: list[dict] = []
        self._best_val_total = float("inf")
        # Deterministic keyed chart augmentation (training only).
        self.augmentor: Optional[KeyedAugmentor] = None
        if self.cfg.augment_transforms:
            self.augmentor = KeyedAugmentor(
                transforms=tuple(self.cfg.augment_transforms),
                strength=float(self.cfg.augment_strength),
                crop_frac=float(self.cfg.augment_crop_frac),
                noise_std=float(self.cfg.augment_noise_std),
            )

        # ---- Temporal CPC v2 state ------------------------------------
        # Memory bank (MoCo-style queue) of L2-normalised teacher target
        # projections + the global ids of the samples that produced them,
        # so the contrastive loss can exclude exact duplicates by id.
        self._bank: Optional[torch.Tensor] = None
        self._bank_gids: list = []
        if self.cfg.temporal_bank_size > 0:
            cap = int(self.cfg.temporal_bank_size)
            self._bank = torch.zeros(0, self.cfg.projection_dim, device=self.device)
            self._bank_capacity = cap
        # Active TemporalPairDataset for hard-negative sampling (set by
        # ``_loader``; sampled deterministically per (epoch, step, index)).
        self._pair_source: Optional[TemporalPairDataset] = None
        # Total optimisation steps for the cosine schedule; auto-estimated
        # at fit() start when cfg.total_steps == 0.
        self._estimated_total_steps: int = 0
        # Anchor for the LR schedule: the step where THIS run started
        # (0 for a cold start; the restored step after a resume). Keeps a
        # cosine decay from consuming the previous run's budget on resume.
        self._resume_step: int = 0

    # ------------------------------------------------------------------
    # Temporal CPC v2: memory bank + hard negatives
    # ------------------------------------------------------------------

    def _temporal_hard_negatives(self, batch: dict) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Same-instrument shifted-window negatives (train only).

        For each anchor ``(leaf, local)`` encode the windows at
        ``local + off*h`` (``off`` in :attr:`SSLConfig.temporal_hard_offsets`,
        h = temporal_horizon) with the EMA teacher, in ONE concatenated
        forward. Returns ``(rows (B*K, proj_dim), mask (B, B*K))`` or
        ``(None, None)`` when disabled or no shift is valid. Rows that
        could not be produced (out-of-bounds) are zero-padded and masked
        out; the only *unmasked* rows are valid shifted windows whose
        global id differs from the anchor's own positive.
        """
        offsets = tuple(int(o) for o in (self.cfg.temporal_hard_offsets or ()))
        if not offsets or self._pair_source is None:
            return None, None
        h = int(self.cfg.temporal_horizon)
        B = int(batch["chart"].size(0))
        future_gids = batch.get("future_gids") or [None] * B
        chart_rows: list = []
        numeric_rows: list = []
        context_rows: list = []
        row_gids: list = []
        populated: list = []  # anchors (in batch order) that produced rows
        for i in range(B):
            gid = future_gids[i]
            if gid is None:
                continue
            leaf, local = gid
            c_rows, n_rows, x_rows, g_rows = [], [], [], []
            for off in offsets:
                item = self._pair_source.offset_item(leaf, local, off * h)
                if item is None:
                    continue
                c_rows.append(item["chart"].to(self.device, non_blocking=True))
                n_rows.append(item["numeric"].to(self.device, non_blocking=True))
                x_rows.append(item["context"].to(self.device, non_blocking=True))
                g_rows.append(item["meta"]["gid"])
            if not c_rows:
                continue
            chart_rows.append(c_rows)
            numeric_rows.append(n_rows)
            context_rows.append(x_rows)
            row_gids.append(g_rows)
            populated.append(i)
        if not chart_rows:
            return None, None
        K = max(len(r) for r in row_gids)
        ld = self.device

        def _pad(lst: list, dtype: torch.dtype, shape: tuple) -> torch.Tensor:
            out = torch.zeros(B, K, *shape, dtype=dtype, device=ld)
            for pos, rows in enumerate(lst):
                a = populated[pos]  # original batch anchor slot
                for j, t in enumerate(rows):
                    out[a, j] = t
            return out

        c_shape = chart_rows[0][0].shape
        n_shape = numeric_rows[0][0].shape
        x_shape = context_rows[0][0].shape
        chart = _pad(chart_rows, torch.float32, c_shape)
        numeric = _pad(numeric_rows, torch.float32, n_shape)
        context = _pad(context_rows, torch.float32, x_shape)
        # Instrument ids per hard row, aligned with the (B, K) layout above:
        # rows live at slot ``anchor*K + j``; anchors without valid offsets
        # keep zero-id rows (they are masked out of the logits anyway).
        # NOTE: the instrument id comes from the BATCH tensor, NOT from the
        # leaf index — a dataset may contain many segments per instrument
        # (leaf idx can exceed n_instruments and embedding would OOR).
        batch_inst = batch.get("instrument_id")
        if batch_inst is None:
            return None, None
        inst = torch.zeros(B * K, dtype=torch.long, device=ld)
        for anchor in populated:
            inst[anchor * K:(anchor + 1) * K] = int(batch_inst[anchor])
        with torch.no_grad():
            z_hn = self.teacher.teacher.encode(
                chart.view(-1, *c_shape),
                numeric.view(-1, *n_shape),
                context.view(-1, *x_shape),
                instrument_id=inst,
            ).view(B, K, -1)
            rows = F.normalize(self.target_proj_temporal(z_hn).detach(), dim=-1)
        mask = torch.ones(B, B * K, dtype=torch.bool, device=ld)
        for a in range(B):
            pos = future_gids[a]
            for i, gids_i in enumerate(row_gids):
                anchor = populated[i]
                for j, gid in enumerate(gids_i):
                    col = anchor * K + j
                    if gid is not None and pos is not None and gid != pos:
                        mask[a, col] = False
        return rows.view(B * K, -1), mask

    def _bank_rows(self, batch: dict) -> Optional[torch.Tensor]:
        """The current memory-bank tensor (L2-normalised), or None."""
        if self._bank is None:
            return None
        if self._step < int(self.cfg.temporal_bank_warmup) or self._bank.size(0) == 0:
            return None
        return F.normalize(self._bank, dim=-1).detach()

    def _bank_mask(self, batch: dict, n_rows: int) -> torch.Tensor:
        """(B, n_rows) exclusion mask: True where a bank row is the SAME
        sample as the anchor's own positive (dedup by global id)."""
        B = int(batch["chart"].size(0))
        mask = torch.zeros(B, n_rows, dtype=torch.bool, device=self.device)
        future_gids = batch.get("future_gids") or [None] * B
        bg = list(self._bank_gids)
        if not bg:
            return mask
        for i in range(B):
            pos = future_gids[i]
            if pos is None:
                continue
            for j, gid in enumerate(bg[:n_rows]):
                if gid == pos:
                    mask[i, j] = True
        return mask

    def _push_bank(self, target: torch.Tensor, batch: dict) -> None:
        """Append this batch's normalised target projections to the queue."""
        if self._bank is None:
            return
        target = target.detach()
        if not bool(torch.isfinite(target).all()):
            logger.warning("ssl step %d: skipping bank push (non-finite targets)", self._step)
            return
        future_gids = batch.get("future_gids")
        B = int(target.size(0))
        if not future_gids or len(future_gids) != B:
            return
        cap = int(self._bank_capacity)
        if self._bank.size(0) + B <= cap:
            self._bank = torch.cat([self._bank, target], dim=0)
            self._bank_gids.extend(future_gids)
        else:
            drop = min(B, cap)
            self._bank = torch.cat([self._bank[drop:], target], dim=0)[:cap]
            self._bank_gids = (self._bank_gids[drop:] + list(future_gids))[-cap:]
            if self._bank.size(0) > cap:
                self._bank = self._bank[:cap]

    # ------------------------------------------------------------------
    # Single-batch loss
    # ------------------------------------------------------------------

    def _loss(self, batch: dict) -> dict:
        chart = batch["chart"].to(self.device, non_blocking=True)
        numeric = batch["numeric"].to(self.device, non_blocking=True)
        context = batch["context"].to(self.device, non_blocking=True)

        losses: dict[str, torch.Tensor] = {}

        # --- 1) Temporal contrastive (CPC) --------------------------------
        if self.cfg.use_temporal_contrast:
            assert self.teacher is not None, "temporal contrast requires EMA teacher"
            z_t = self.model.encode(
                chart, numeric, context, instrument_id=batch.get("instrument_id")
            )
            future_chart = batch.get("future_chart", chart).to(
                self.device, non_blocking=True
            )
            future_numeric = batch.get("future_numeric", numeric).to(
                self.device, non_blocking=True
            )
            future_context = batch.get("future_context", context).to(
                self.device, non_blocking=True
            )
            with torch.no_grad():
                z_tp1 = self.teacher.teacher.encode(
                    future_chart, future_numeric, future_context,
                    instrument_id=batch.get("instrument_id"),
                ).detach()
            # Project both sides to the common contrast space.
            p_t = self.temporal_predictor(self.proj_temporal(z_t))
            with torch.no_grad():
                p_tp1 = F.normalize(self.target_proj_temporal(z_tp1).detach(), dim=-1)
            train_mode = bool(self.model.training)
            hard = None
            hard_mask = None
            bank = None
            bank_mask = None
            if train_mode:
                hard, hard_mask = self._temporal_hard_negatives(batch)
                bank = self._bank_rows(batch)
                if bank is not None:
                    bank_mask = self._bank_mask(batch, bank.size(0))
            n0 = p_tp1.size(0)
            extra_blocks = [x for x in (hard, bank) if x is not None]
            row_mask = None
            if extra_blocks:
                extra = torch.cat(extra_blocks, dim=0)
                n_hard = hard.size(0) if hard is not None else 0
                row_mask = torch.zeros(
                    n0, n0 + extra.size(0), dtype=torch.bool, device=self.device
                )
                if hard_mask is not None:
                    row_mask[:, n0:n0 + hard_mask.size(1)] = hard_mask
                if bank_mask is not None and n_hard + bank.size(0) <= extra.size(0):
                    row_mask[:, n0 + n_hard:n0 + n_hard + bank_mask.size(1)] = bank_mask
            if extra_blocks:
                losses["temporal"] = info_nce(
                    p_t, p_tp1, self.cfg.temperature,
                    extra_negatives=extra, row_mask=row_mask,
                )
            else:
                losses["temporal"] = info_nce(p_t, p_tp1, self.cfg.temperature)
            if train_mode and self._bank is not None:
                self._push_bank(p_tp1, batch)

        # --- 2) Cross-modal alignment ------------------------------------
        if self.cfg.use_cross_modal:
            v = self.model.plain_vision(chart)
            n_cls, _ = self.model.numeric(numeric)
            v_proj = self.proj_vision(v)
            n_proj = self.proj_numeric(n_cls)
            # Symmetric InfoNCE: vision <-> numeric (projection space).
            loss_v2n = info_nce(v_proj, n_proj, self.cfg.temperature)
            loss_n2v = info_nce(n_proj, v_proj, self.cfg.temperature)
            losses["alignment"] = 0.5 * (loss_v2n + loss_n2v)
            # Trunk-level alignment (no projections): forces the raw
            # embeddings to agree, not just their heads. Soft-target mode
            # detaches the numeric side (momentum>0) to keep the stronger
            # modality stable while pushing vision toward it.
            if self.cfg.weight_trunk_align > 0.0:
                vn = F.normalize(v.reshape(v.size(0), -1), dim=-1)
                nn = F.normalize(n_cls.detach() if self.cfg.trunk_align_momentum > 0.0 else n_cls, dim=-1)
                trunk_cos = (vn * nn).sum(dim=-1).mean()
                losses["trunk_align"] = self.cfg.weight_trunk_align * (1.0 - trunk_cos)

        # --- 3) Masked numeric modeling -----------------------------------
        if self.cfg.use_masked_modeling:
            losses["masked"] = masked_numeric_loss(
                self.model.numeric, self.reconstructor, numeric, self.cfg.mask_ratio,
                target_norm=bool(self.cfg.masked_target_norm),
            )

        # --- 3b) Instrument-embedding spread (fixes symbol collisions) ----
        if self.cfg.instrument_contrast_w > 0.0 and self.model.context.instrument_emb is not None:
            emb = self.model.context.instrument_emb.weight
            n_inst = emb.size(0)
            if n_inst > 1:
                e = F.normalize(emb, dim=-1)
                off = e @ e.t()
                mask_off = ~torch.eye(n_inst, dtype=torch.bool, device=off.device)
                losses["instrument_contrast"] = (
                    self.cfg.instrument_contrast_w * off[mask_off].mean()
                )

        # --- 3c) z-level instrument contrast (P3) -------------------------
        # InfoNCE on the TRUNK embedding z keyed by instrument identity so the
        # ENCODED representation separates symbols (the embedding-table loss
        # above leaves z only ~5 deg separated, hence silhouette ~0.008).
        if self.cfg.instrument_z_contrast_w > 0.0:
            ids = batch.get("instrument_id")
            if ids is not None and ids.numel() > 1 and torch.unique(ids).numel() > 1:
                with torch.no_grad():
                    zc = self.model.encode(chart, numeric, context, instrument_id=ids)
                zn = F.normalize(zc, dim=-1)
                sim = zn @ zn.t() / max(self.cfg.temperature, 1e-3)
                eye = torch.eye(ids.numel(), dtype=torch.bool, device=ids.device)
                same = ids.view(-1, 1) == ids.view(1, -1)
                peers = same & ~eye
                if bool(peers.sum(-1).ge(1).all().item()):
                    target_idx = peers.float().argmax(dim=-1)
                    losses["instrument_z"] = (
                        self.cfg.instrument_z_contrast_w * F.cross_entropy(sim, target_idx)
                    )

        # Total = weighted sum.
        total = (
            self.cfg.weight_temporal * losses.get("temporal", torch.zeros((), device=self.device))
            + self.cfg.weight_alignment * losses.get("alignment", torch.zeros((), device=self.device))
            + self.cfg.weight_masked * losses.get("masked", torch.zeros((), device=self.device))
            + losses.get("trunk_align", torch.zeros((), device=self.device))
            + losses.get("instrument_contrast", torch.zeros((), device=self.device))
            + losses.get("instrument_z", torch.zeros((), device=self.device))
        )
        losses["total"] = total
        return losses

    def _augment_step_batch(self, batch: dict) -> None:
        """Deterministically augment ``chart``/``future_chart`` in-place.

        Keys are derived from ``(epoch, global_step, within-batch index, tag)``
        so a run with the same seed/data reproduces the exact augmented bytes,
        and every epoch sees a different augmentation per sample. Evaluation is
        never augmented (see ``evaluate``), keeping the base render identical
        train-vs-serve.
        """
        assert self.augmentor is not None
        epoch, step = int(self._completed_epochs), int(self._step)
        chart = batch["chart"]
        bs = chart.size(0)
        keys_cur = [f"cur:e{epoch}:s{step}:{i}" for i in range(bs)]
        batch["chart"] = torch.stack(
            [self.augmentor.apply(chart[i], k) for i, k in enumerate(keys_cur)], dim=0
        )
        if "future_chart" in batch:
            fut = batch["future_chart"]
            keys_fut = [f"fut:e{epoch}:s{step}:{i}" for i in range(bs)]
            batch["future_chart"] = torch.stack(
                [self.augmentor.apply(fut[i], k) for i, k in enumerate(keys_fut)], dim=0
            )

    def _lr_scale(self, *, step: int) -> float:
        """LR multiplier: linear warmup followed by the configured schedule.

        ``constant`` (canonical v1): 1.0 after warmup.
        ``cosine``: decays from 1.0 post-warmup to ``cosine_min_scale`` over
        the RUN's own budget. The step anchor is the resume point (if any):
        two measurings MUST NOT consume the cosine budget of the previous run
        (a resume with a tiny floor-lr was a real bug measured at 4.5e-07).
        """
        eff = int(step) - int(self._resume_step)
        warmup = max(0, int(self.cfg.warmup_steps))
        if warmup > 0 and eff < warmup:
            return min(1.0, float(eff + 1) / warmup)
        if self.cfg.lr_schedule != "cosine":
            return 1.0
        total = max(0, int(self.cfg.total_steps))
        if total <= 0:
            total = int(self._estimated_total_steps)
        remaining = total - warmup
        if remaining <= 0:
            return min(self.cfg.cosine_min_scale, 1.0)
        progress = min(1.0, max(0.0, float(eff - warmup) / remaining))
        scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min(self.cfg.cosine_min_scale, 1.0) + (1.0 - min(self.cfg.cosine_min_scale, 1.0)) * scale

    def step(self, batch: dict) -> dict:
        """Run one optimisation step on a single batch."""
        self.model.train()
        if self.augmentor is not None:
            self._augment_step_batch(batch)
        batch = self._to_device(batch)
        lr_scale = self._lr_scale(step=self._step)
        for group in self.opt.param_groups:
            group["lr"] = self.cfg.lr * lr_scale
        losses = self._loss(batch)
        loss = losses["total"]
        do_update = bool(loss.requires_grad) and bool(torch.isfinite(loss))
        self.opt.zero_grad(set_to_none=True)
        if do_update:
            loss.backward()
            # P2: rebalance modalities. The measured grad share is numeric ~40%
            # vs vision ~2%, which makes z numeric-dominated (and drives the
            # numeric-perturb angle + recon weakness). Scale vision grads AFTER
            # backward, BEFORE clipping. 1.0 disables.
            if self.cfg.vision_grad_scale != 1.0:
                for name, p in self.model.named_parameters():
                    if name.startswith("vision.") and p.grad is not None:
                        p.grad.mul_(self.cfg.vision_grad_scale)
            # Clip gradients across the full SSL parameter set (model +
            # projection heads + reconstructor) — gradient explosion in
            # the projection heads was a known failure mode in v0.1.
            all_params = (
                list(self.model.parameters())
                + list(self.proj_temporal.parameters())
                + list(self.temporal_predictor.parameters())
                + list(self.proj_vision.parameters())
                + list(self.proj_numeric.parameters())
                + list(self.reconstructor.parameters())
            )
            torch.nn.utils.clip_grad_norm_(all_params, self.cfg.grad_clip)
            self.opt.step()
        else:
            logger.warning("ssl step %d: non-finite/no-grad loss, skipping update", self._step)
        if self.teacher is not None:
            self.teacher.update(self.model)
            decay = self.teacher.decay
            with torch.no_grad():
                for target, student in zip(
                    self.target_proj_temporal.parameters(),
                    self.proj_temporal.parameters(),
                ):
                    target.mul_(decay).add_(student.detach(), alpha=1.0 - decay)
        self._step += 1
        # Replace any non-finite values with 0.0 for clean averaging.
        return {
            k: float(v.detach().item()) if torch.isfinite(v).all() else 0.0
            for k, v in losses.items()
        }

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def _paired_dataset(self, dataset: Dataset) -> Dataset:
        if self.cfg.use_temporal_contrast:
            if isinstance(dataset, TemporalPairDataset):
                return dataset  # already paired (idempotent)
            return TemporalPairDataset(dataset, horizon=self.cfg.temporal_horizon)
        return dataset

    def _loader(
        self,
        dataset: Dataset,
        *,
        shuffle: bool,
        epoch: Optional[int] = None,
    ) -> DataLoader:
        source = self._paired_dataset(dataset)
        if len(source) == 0:
            raise ValueError("S1 dataset has no temporal pairs after window/horizon trimming")
        if self.cfg.use_temporal_contrast and isinstance(source, TemporalPairDataset):
            # Stash the active pair source so the loss can sample
            # deterministic same-instrument hard negatives by gid.
            self._pair_source = source
        use_cuda = self.device.type == "cuda"
        workers = int(os.environ.get("ZHISA_SSL_WORKERS", "0"))
        generator = None
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.cfg.seed + int(epoch or 0))
        return DataLoader(
            source,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=workers,
            collate_fn=(
                temporal_pair_collate
                if self.cfg.use_temporal_contrast
                else multimodal_collate
            ),
            drop_last=shuffle and len(source) >= self.cfg.batch_size,
            pin_memory=use_cuda,
            persistent_workers=workers > 0,
            generator=generator,
        )

    @torch.no_grad()
    def evaluate(self, dataset: Dataset) -> dict:
        # Sequential adjacent windows are almost identical and become false
        # negatives for one another. A fixed random order measures the same
        # objective without this ordering artefact and remains reproducible.
        # Channel dropout is a TRAIN-only regulariser: disable it for eval.
        leaves = list(_iter_leaf_datasets(dataset))
        for leaf in leaves:
            if hasattr(leaf, "set_channel_dropout_enabled"):
                leaf.set_channel_dropout_enabled(False)
        loader = self._loader(dataset, shuffle=True, epoch=10_000)
        self.model.eval()
        if self.teacher is not None:
            self.teacher.teacher.eval()
        totals: dict[str, float] = {}
        count = 0
        max_batches = int(self.cfg.val_max_batches)
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.cfg.seed + 10_000)
            for batch_idx, batch in enumerate(loader):
                if max_batches > 0 and batch_idx >= max_batches:
                    break
                batch_d = self._to_device(batch)
                losses = self._loss(batch_d)
                bs = batch_d["chart"].size(0)
                for key, value in losses.items():
                    totals[key] = totals.get(key, 0.0) + float(value.item()) * bs
                count += bs
        for leaf in leaves:
            if hasattr(leaf, "set_channel_dropout_enabled"):
                leaf.set_channel_dropout_enabled(True)
        if count == 0:
            raise RuntimeError("S1 validation produced no batches")
        return {key: value / count for key, value in totals.items()} | {
            "n_samples": int(count)
        }

    def fit(self, train_ds: Dataset, val_ds: Optional[Dataset] = None) -> dict:
        cfg = self.cfg
        history: list[dict] = []
        timer = Timer()
        if cfg.lr_schedule == "cosine" and self._estimated_total_steps <= 0:
            steps_per_epoch = max(1, len(self._paired_dataset(train_ds)) // max(1, cfg.batch_size))
            self._estimated_total_steps = int(cfg.total_steps) or (cfg.epochs * steps_per_epoch)
        for _ in range(cfg.epochs):
            epoch = self._completed_epochs
            for leaf in _iter_leaf_datasets(train_ds):
                if hasattr(leaf, "set_aug_salt"):
                    leaf.set_aug_salt(epoch)
                if hasattr(leaf, "set_channel_dropout_enabled"):
                    leaf.set_channel_dropout_enabled(True)
            loader = self._loader(train_ds, shuffle=True, epoch=epoch)
            self.model.train()
            ep_agg: dict[str, float] = {}
            ep_count = 0
            timer.start()
            for it, batch in enumerate(loader):
                b = self._to_device(batch)
                losses = self.step(b)
                bs = b["chart"].size(0)
                for k, v in losses.items():
                    ep_agg[k] = ep_agg.get(k, 0.0) + v * bs
                ep_count += bs
                if (it + 1) % cfg.log_every == 0:
                    avg = {k: v / max(1, ep_count) for k, v in ep_agg.items()}
                    lr = self.opt.param_groups[0]["lr"]
                    logger.info(
                        "ssl epoch=%d iter=%d step=%d %s lr=%.2e elapsed=%.1fs",
                        epoch, it, self._step,
                        " ".join(f"{k}={v:.4f}" for k, v in avg.items()),
                        lr, timer.elapsed,
                    )
                if (
                    cfg.checkpoint
                    and cfg.checkpoint_every_steps > 0
                    and self._step % cfg.checkpoint_every_steps == 0
                ):
                    self.save(cfg.checkpoint)
            avg = {k: v / max(1, ep_count) for k, v in ep_agg.items()}
            if ep_count == 0:
                raise RuntimeError("S1 epoch produced no batches")
            timer.stop()
            record = {"epoch": epoch, **avg, "elapsed_s": timer.elapsed}
            if val_ds is not None:
                val_metrics = self.evaluate(val_ds)
                record["val"] = val_metrics
                logger.info(
                    "ssl epoch %d validation | %s",
                    epoch,
                    " ".join(f"{key}={value:.4f}" for key, value in val_metrics.items() if key != "n_samples"),
                )
            history.append(record)
            self._history.append(record)
            self._completed_epochs += 1
            logger.info(
                "ssl epoch %d done in %.1fs | %s",
                epoch, timer.elapsed,
                " ".join(f"{k}={v:.4f}" for k, v in avg.items()),
            )
            timer.reset()
            score = (
                record.get("val", {}).get("total", float("inf"))
                if val_ds is not None
                else record["total"]
            )
            if score < self._best_val_total:
                self._best_val_total = float(score)
                if cfg.best_checkpoint:
                    self.save(cfg.best_checkpoint)
            if cfg.checkpoint:
                # Save checkpoint after every epoch to prevent data loss!
                self.save(cfg.checkpoint)
        return {"history": history, "final_step": self._step}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cfg_dict = self.model.cfg.__dict__.copy()
        if "vision_channels" in cfg_dict and isinstance(cfg_dict["vision_channels"], tuple):
            cfg_dict["vision_channels"] = list(cfg_dict["vision_channels"])
        payload = {
            "model": self.model.state_dict(),
            "proj_temporal": self.proj_temporal.state_dict(),
            "temporal_predictor": self.temporal_predictor.state_dict(),
            "proj_vision": self.proj_vision.state_dict(),
            "proj_numeric": self.proj_numeric.state_dict(),
            "reconstructor": self.reconstructor.state_dict(),
            "target_proj_temporal": self.target_proj_temporal.state_dict(),
            "optimizer": self.opt.state_dict(),
            "config": cfg_dict,
            "model_config": cfg_dict,  # canonical name
            "ssl_config": self.cfg.__dict__,
            "checkpoint_meta": {
                "stage": "s1_ssl",
                "trading_policy_ready": False,
                "policy_head_trained": False,
                "reason": "S1 is representation pretraining; fine-tune with S2b/S4+ before paper trading.",
                "temporal_pairing": "causal_adjacent_sample",
                "temporal_objective": "student_predictor_to_ema_target",
                "temporal_horizon": self.cfg.temporal_horizon,
                "resume_granularity": "completed_epoch",
                "dataset": {
                    "root": self.cfg.dataset_root,
                    "timeframe": self.cfg.dataset_timeframe,
                    "manifest_checksum": self.cfg.dataset_manifest_checksum,
                },
                "render": {
                    "renderer_version": self.cfg.renderer_version,
                    "render_spec_hash": self.cfg.render_spec_hash,
                    "render_fingerprint": self.cfg.render_fingerprint,
                    "render_store_checksum": self.cfg.render_store_checksum,
                    "augmentation": (
                        self.augmentor.to_meta() if self.augmentor is not None else None
                    ),
                    "augmentation_key_scheme": (
                        "epoch:step:index:cur|fut" if self.augmentor is not None else None
                    ),
                },
            },
            "trainer_state": {
                "step": self._step,
                "completed_epochs": self._completed_epochs,
                "history": self._history,
                "best_val_total": self._best_val_total,
            },
        }
        if self.teacher is not None:
            payload["teacher"] = self.teacher.state_dict()
        if self._bank is not None:
            payload["temporal_bank"] = {
                "tensor": self._bank,
                "gids": list(self._bank_gids),
            }
        tmp = p.with_name(f".{p.name}.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, p)
        logger.info("ssl checkpoint saved to %s", p)

    def load(self, path: str, *, restore_optimizer: bool = True) -> dict:
        sd = torch.load(path, map_location=self.device, weights_only=False)
        # The saved model may have head shapes that differ from the current
        # model (e.g. n_actions, n_regime_classes). We cannot use
        # ``load_state_dict(strict=False)`` alone because PyTorch still
        # raises on size mismatches; we must filter the checkpoint to
        # only contain keys with matching shapes.
        filtered_model = _filter_matching_state_dict(sd["model"], self.model)
        model_exact = len(filtered_model) == len(sd["model"])
        self.model.load_state_dict(filtered_model, strict=False)
        self.proj_temporal.load_state_dict(sd["proj_temporal"])
        if "temporal_predictor" in sd:
            self.temporal_predictor.load_state_dict(sd["temporal_predictor"])
        self.proj_vision.load_state_dict(sd["proj_vision"])
        self.proj_numeric.load_state_dict(sd["proj_numeric"])
        # The reconstructor head shape follows the numeric input width
        # (patch_size * in_features); tolerate width changes (e.g. a
        # cross-asset dataset with more feature columns) by shape-filtering.
        if "reconstructor" in sd:
            filtered_recon = _filter_matching_state_dict(
                sd["reconstructor"], self.reconstructor
            )
            self.reconstructor.load_state_dict(filtered_recon, strict=False)
        if "target_proj_temporal" in sd:
            self.target_proj_temporal.load_state_dict(sd["target_proj_temporal"])
        else:
            self.target_proj_temporal.load_state_dict(self.proj_temporal.state_dict())
        if self.teacher is not None and "teacher" in sd:
            filtered_teacher = _filter_matching_state_dict(
                sd["teacher"]["teacher"], self.teacher.teacher
            )
            self.teacher.teacher.load_state_dict(filtered_teacher, strict=False)

        optimizer_restored = False
        if restore_optimizer and "optimizer" in sd and model_exact:
            try:
                self.opt.load_state_dict(sd["optimizer"])
                optimizer_restored = True
            except (ValueError, RuntimeError) as exc:
                logger.warning("could not restore S1 optimizer state: %s", exc)

        trainer_state = sd.get("trainer_state", {}) if optimizer_restored else {}
        self._step = int(trainer_state.get("step", 0))
        self._completed_epochs = int(trainer_state.get("completed_epochs", 0))
        self._history = list(trainer_state.get("history", []))
        self._best_val_total = float(
            trainer_state.get("best_val_total", float("inf"))
        )
        if optimizer_restored:
            # Anchor the LR schedule at the resumed step so a cosine decay
            # gets THIS run's full budget (old bug: floor-lr on resume).
            self._resume_step = self._step
        else:
            self._resume_step = 0
        bank_payload = sd.get("temporal_bank")
        if bank_payload is not None and self._bank is not None:
            bank_t = bank_payload.get("tensor")
            if (
                torch.is_tensor(bank_t)
                and bank_t.numel() > 0
                and bank_t.shape[-1] == self.cfg.projection_dim
            ):
                self._bank = bank_t.to(self.device)
                self._bank_gids = list(bank_payload.get("gids", []))
                logger.info("ssl restored temporal bank (%d rows)", self._bank.size(0))
            else:
                self._bank = torch.zeros(0, self.cfg.projection_dim, device=self.device)
                self._bank_gids = []
        status = {
            "optimizer_restored": optimizer_restored,
            "legacy_warm_start": "trainer_state" not in sd,
            "resume_mode": "full" if optimizer_restored else "warm_start",
            "step": self._step,
            "completed_epochs": self._completed_epochs,
        }
        logger.info("ssl checkpoint loaded from %s | %s", path, status)
        return status

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_device(self, batch) -> dict:
        if isinstance(batch, dict):
            out: dict = {}
            for key, value in batch.items():
                if torch.is_tensor(value):
                    out[key] = value.to(self.device, non_blocking=True)
                else:
                    out[key] = value  # keep gids/meta lists as-is
            return out
        return {
            "chart": batch.chart.to(self.device, non_blocking=True),
            "numeric": batch.numeric.to(self.device, non_blocking=True),
            "context": batch.context.to(self.device, non_blocking=True),
            **{
                k: getattr(batch, k).to(self.device, non_blocking=True)
                for k in ("future_chart", "future_numeric", "future_context",
                          "instrument_id", "macro_numeric")
                if getattr(batch, k, None) is not None
            },
        }


def load_pretrained_into_policy(
    policy: PolicyNetwork,
    ssl_checkpoint: str,
    strict: bool = False,
) -> PolicyNetwork:
    """Load the encoder weights from an S1 checkpoint into a fresh policy.

    Only the encoder / fusion / memory parameters are restored (the S2
    trainer will freshly initialise the heads). Heads and SSL-specific
    projections are filtered out so the load tolerates shape mismatches
    (e.g. different ``n_actions`` between pretraining and S2).

    Returns the policy in-place for convenience.
    """
    sd = torch.load(ssl_checkpoint, map_location="cpu", weights_only=False)
    enc_sd = sd["model"] if "model" in sd else sd
    filtered = _filter_matching_state_dict(
        enc_sd,
        policy,
        excluded_prefixes=("heads.", "memory."),
    )
    incompatible = policy.load_state_dict(filtered, strict=False)
    if strict:
        missing_trunk = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(("heads.", "memory."))
        ]
        if missing_trunk or incompatible.unexpected_keys:
            raise RuntimeError(
                "S1 representation checkpoint is not strictly compatible: "
                f"missing={missing_trunk}, unexpected={incompatible.unexpected_keys}"
            )
    return policy


def _filter_matching_state_dict(
    sd: dict,
    model: nn.Module,
    *,
    excluded_prefixes: tuple[str, ...] = (),
) -> dict:
    """Return a state_dict containing only entries with shapes matching
    ``model``'s parameters.

    This is the standard workaround for PyTorch's :meth:`load_state_dict`
    which raises on size mismatches even when ``strict=False``. We
    need this because the SSL trainer can be re-instantiated with
    different head shapes (e.g. ``n_actions``, ``n_regime_classes``)
    than the model that produced the checkpoint.
    """
    ref = {k: v.shape for k, v in model.named_parameters()}
    ref.update({k: v.shape for k, v in model.named_buffers()})
    out = {}
    for k, v in sd.items():
        if excluded_prefixes and k.startswith(excluded_prefixes):
            continue
        if k in ref and tuple(v.shape) == tuple(ref[k]):
            out[k] = v
    return out
