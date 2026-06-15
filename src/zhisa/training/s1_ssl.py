"""S1: Self-supervised pretraining for the multimodal policy.

This module implements the S1 stage from ``CONCEPT.md`` (§5.2). The
goal is to pretrain the multimodal encoder stack on **unlabelled**
market data so that the S2 supervised trainer starts from a richer
initialisation. Four complementary objectives are combined:

1. **Temporal contrastive (CPC-style).**  The model encodes the current
   market state into ``z_t``. The next-bar state is encoded by an
   exponential moving average (EMA) teacher into ``z_{t+1}``. We
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

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from zhisa.data.dataset import MarketDataset, multimodal_collate
from zhisa.models.policy import PolicyNetwork
from zhisa.utils.logging import get_logger
from zhisa.utils.timing import Timer

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
    use_ema_teacher: bool = True
    use_masked_modeling: bool = True
    use_temporal_contrast: bool = True
    use_cross_modal: bool = True


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
    We attach a small linear head that maps each token back to the
    flattened patch values. Only the masked positions contribute to
    the loss.
    """

    def __init__(self, d_model: int, patch_size: int, in_features: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_features = in_features
        self.head = nn.Linear(d_model, patch_size * in_features)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, 1 + n_patches, d_model)
        return self.head(tokens)


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
) -> torch.Tensor:
    """Symmetric InfoNCE between two L2-normalised projection batches.

    Both ``anchor`` and ``positive`` are expected to be shape ``(B, D)``.
    The positive pair is the diagonal ``(i, i)``; all other entries
    are negatives. The logits are clamped to ``[-max_logit, max_logit]``
    to keep cross-entropy numerically stable when the projection
    head has not yet been warmed up.
    """
    a = F.normalize(anchor, dim=-1)
    p = F.normalize(positive, dim=-1)
    logits = a @ p.t() / max(temperature, 1e-6)
    logits = logits.clamp(min=-max_logit, max=max_logit)
    labels = torch.arange(a.size(0), device=a.device)
    return F.cross_entropy(logits, labels)


def masked_numeric_loss(
    numeric_encoder: nn.Module,
    reconstructor: _MaskedReconstructor,
    x: torch.Tensor,
    mask_ratio: float = 0.4,
) -> torch.Tensor:
    """Mask random patches of the numeric input, encode, and predict them.

    The numeric encoder is :class:`zhisa.models.encoders.numeric.NumericEncoder`
    which returns ``(cls, tokens)`` of shape ``(B, 1+n_patches, d_model)``.
    Only the non-CLS positions are considered for masking.
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
    # Drop the CLS slot: tokens[:, 0] is CLS, tokens[:, 1:] are patches.
    pred_patches = pred[:, 1:, :]
    target = patches.view(B, n_patches, -1)
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
        self.proj_vision = _ProjectionHead(D, self.cfg.hidden_dim, self.cfg.projection_dim)
        self.proj_numeric = _ProjectionHead(D, self.cfg.hidden_dim, self.cfg.projection_dim)

        # Masked numeric reconstructor.
        self.reconstructor = _MaskedReconstructor(
            d_model=model.numeric.cfg.d_model,
            patch_size=model.numeric.cfg.patch_size,
            in_features=model.numeric.cfg.in_features,
        )

        self.proj_temporal.to(self.device)
        self.proj_vision.to(self.device)
        self.proj_numeric.to(self.device)
        self.reconstructor.to(self.device)

        # EMA teacher.
        self.teacher: Optional[EMATeacher] = None
        if self.cfg.use_ema_teacher:
            self.teacher = EMATeacher(model, decay=self.cfg.ema_decay)
            self.teacher.teacher.to(self.device)

        # Optimiser & LR schedule.
        params = (
            list(model.parameters())
            + list(self.proj_temporal.parameters())
            + list(self.proj_vision.parameters())
            + list(self.proj_numeric.parameters())
            + list(self.reconstructor.parameters())
        )
        self.opt = torch.optim.AdamW(
            params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        self._step = 0

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
            z_t = self.model.encode(chart, numeric, context)
            with torch.no_grad():
                z_tp1 = self.teacher.teacher.encode(chart, numeric, context).detach()
            # Project both sides to the common contrast space.
            p_t = self.proj_temporal(z_t)
            with torch.no_grad():
                p_tp1 = self.proj_temporal(z_tp1).detach()
            losses["temporal"] = info_nce(p_t, p_tp1, self.cfg.temperature)

        # --- 2) Cross-modal alignment ------------------------------------
        if self.cfg.use_cross_modal:
            v = self.model.vision(chart)
            n_cls, _ = self.model.numeric(numeric)
            v_proj = self.proj_vision(v)
            n_proj = self.proj_numeric(n_cls)
            # Symmetric InfoNCE: vision <-> numeric.
            loss_v2n = info_nce(v_proj, n_proj, self.cfg.temperature)
            loss_n2v = info_nce(n_proj, v_proj, self.cfg.temperature)
            losses["alignment"] = 0.5 * (loss_v2n + loss_n2v)

        # --- 3) Masked numeric modeling -----------------------------------
        if self.cfg.use_masked_modeling:
            losses["masked"] = masked_numeric_loss(
                self.model.numeric, self.reconstructor, numeric, self.cfg.mask_ratio
            )

        # Total = weighted sum.
        total = (
            self.cfg.weight_temporal * losses.get("temporal", torch.zeros((), device=self.device))
            + self.cfg.weight_alignment * losses.get("alignment", torch.zeros((), device=self.device))
            + self.cfg.weight_masked * losses.get("masked", torch.zeros((), device=self.device))
        )
        losses["total"] = total
        return losses

    def step(self, batch: dict) -> dict:
        """Run one optimisation step on a single batch."""
        self.model.train()
        losses = self._loss(batch)
        loss = losses["total"]
        do_update = bool(loss.requires_grad) and bool(torch.isfinite(loss))
        self.opt.zero_grad(set_to_none=True)
        if do_update:
            loss.backward()
            # Clip gradients across the full SSL parameter set (model +
            # projection heads + reconstructor) — gradient explosion in
            # the projection heads was a known failure mode in v0.1.
            all_params = (
                list(self.model.parameters())
                + list(self.proj_temporal.parameters())
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
        self._step += 1
        # Replace any non-finite values with 0.0 for clean averaging.
        return {
            k: float(v.detach().item()) if torch.isfinite(v).all() else 0.0
            for k, v in losses.items()
        }

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def fit(self, train_ds: MarketDataset) -> dict:
        cfg = self.cfg
        use_cuda = isinstance(self.device, torch.device) and self.device.type == "cuda" \
            or (isinstance(self.device, str) and self.device.startswith("cuda"))
        loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=int(os.environ.get("ZHISA_SSL_WORKERS", "0")),
            collate_fn=multimodal_collate,
            drop_last=True,
            pin_memory=use_cuda,
            persistent_workers=int(os.environ.get("ZHISA_SSL_WORKERS", "0")) > 0,
        )
        history: list[dict] = []
        timer = Timer()
        for epoch in range(cfg.epochs):
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
            avg = {k: v / max(1, ep_count) for k, v in ep_agg.items()}
            timer.stop()
            history.append({"epoch": epoch, **avg, "elapsed_s": timer.elapsed})
            logger.info(
                "ssl epoch %d done in %.1fs | %s",
                epoch, timer.elapsed,
                " ".join(f"{k}={v:.4f}" for k, v in avg.items()),
            )
            timer.reset()
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
            "proj_vision": self.proj_vision.state_dict(),
            "proj_numeric": self.proj_numeric.state_dict(),
            "reconstructor": self.reconstructor.state_dict(),
            "config": cfg_dict,
            "model_config": cfg_dict,  # canonical name
            "ssl_config": self.cfg.__dict__,
            "checkpoint_meta": {
                "stage": "s1_ssl",
                "trading_policy_ready": False,
                "policy_head_trained": False,
                "reason": "S1 is representation pretraining; fine-tune with S2b/S4+ before paper trading.",
            },
        }
        if self.teacher is not None:
            payload["teacher"] = self.teacher.state_dict()
        torch.save(payload, p)
        logger.info("ssl checkpoint saved to %s", p)

    def load(self, path: str) -> None:
        sd = torch.load(path, map_location=self.device, weights_only=False)
        # The saved model may have head shapes that differ from the current
        # model (e.g. n_actions, n_regime_classes). We cannot use
        # ``load_state_dict(strict=False)`` alone because PyTorch still
        # raises on size mismatches; we must filter the checkpoint to
        # only contain keys with matching shapes.
        filtered_model = _filter_matching_state_dict(sd["model"], self.model)
        self.model.load_state_dict(filtered_model, strict=False)
        self.proj_temporal.load_state_dict(sd["proj_temporal"])
        self.proj_vision.load_state_dict(sd["proj_vision"])
        self.proj_numeric.load_state_dict(sd["proj_numeric"])
        self.reconstructor.load_state_dict(sd["reconstructor"])
        if self.teacher is not None and "teacher" in sd:
            filtered_teacher = _filter_matching_state_dict(
                sd["teacher"]["teacher"], self.teacher.teacher
            )
            self.teacher.teacher.load_state_dict(filtered_teacher, strict=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_device(self, batch) -> dict:
        return {
            "chart": batch.chart.to(self.device, non_blocking=True),
            "numeric": batch.numeric.to(self.device, non_blocking=True),
            "context": batch.context.to(self.device, non_blocking=True),
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
    filtered = _filter_matching_state_dict(enc_sd, policy)
    missing, unexpected = policy.load_state_dict(filtered, strict=strict)
    return policy


def _filter_matching_state_dict(
    sd: dict, model: nn.Module
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
        if k in ref and tuple(v.shape) == tuple(ref[k]):
            out[k] = v
    return out
