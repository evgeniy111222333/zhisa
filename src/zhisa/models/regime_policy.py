"""Policy wrapper that injects Regime Intelligence into context."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from zhisa.models.policy import PolicyConfig, PolicyNetwork
from zhisa.regime.dataset import PLAYBOOK_NAMES, PLAYBOOK_TO_ID
from zhisa.regime.encoder import RegimeEncoder, RegimeEncoderConfig, append_regime_context
from zhisa.regime.gating import regime_action_mask, regime_position_size_multiplier
from zhisa.regime.planner import TradePlan
from zhisa.regime.schema import RegimeReport


EXECUTION_ORDER_TYPES: tuple[str, ...] = ("none", "market_or_limit", "limit", "post_only_limit")
EXECUTION_URGENCIES: tuple[str, ...] = ("wait", "passive", "normal", "aggressive")
POSITION_INTENTS: tuple[str, ...] = ("hold_or_wait", "hold", "add", "reduce")
SLIPPAGE_BUCKETS_BPS: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0)
ORDER_TYPE_TO_ID = {name: i for i, name in enumerate(EXECUTION_ORDER_TYPES)}
URGENCY_TO_ID = {name: i for i, name in enumerate(EXECUTION_URGENCIES)}
POSITION_INTENT_TO_ID = {name: i for i, name in enumerate(POSITION_INTENTS)}


@dataclass
class RegimeAwarePolicyConfig:
    base_policy: PolicyConfig = field(default_factory=PolicyConfig)
    regime_encoder: RegimeEncoderConfig = field(default_factory=RegimeEncoderConfig)
    append_regime_embedding: bool = True
    freeze_regime_encoder: bool = False
    enable_regime_heads: bool = True
    n_playbooks: int = len(PLAYBOOK_NAMES)


@dataclass(frozen=True)
class RegimePolicyHeadConfig:
    embed_dim: int = 128
    regime_embed_dim: int = 32
    hidden_dim: int = 128
    n_playbooks: int = len(PLAYBOOK_NAMES)
    dropout: float = 0.1


@dataclass(frozen=True)
class RegimePolicyLossWeights:
    playbook: float = 0.5
    risk_budget: float = 0.5
    tradeability: float = 0.35
    transition_wait: float = 0.35
    no_trade: float = 0.35
    size_multiplier: float = 0.35
    action_constraint: float = 0.5
    playbook_prior: float = 0.2
    execution_order_type: float = 0.3
    execution_urgency: float = 0.25
    execution_reduce_only: float = 0.25
    execution_scale_in: float = 0.2
    execution_slippage: float = 0.2
    position_intent: float = 0.25


@dataclass(frozen=True)
class RegimePolicyTargetConfig:
    transition_wait_threshold: float = 0.55
    no_trade_tradeability_threshold: float = 0.25
    default_playbook_prior: float = 0.5
    unknown_playbook_fallback: str = "no_trade_wait"
    max_scale_in_steps: float = 3.0
    max_slippage_bps: float = SLIPPAGE_BUCKETS_BPS[-1]

    def __post_init__(self) -> None:
        if self.max_scale_in_steps <= 0:
            raise ValueError("max_scale_in_steps must be positive")
        if self.max_slippage_bps <= 0:
            raise ValueError("max_slippage_bps must be positive")
        if self.unknown_playbook_fallback not in PLAYBOOK_TO_ID:
            raise ValueError(f"unknown_playbook_fallback must be known, got {self.unknown_playbook_fallback!r}")


class RegimePolicyHeads(nn.Module):
    """Auxiliary regime-control heads on top of the policy/regime state."""

    def __init__(self, cfg: RegimePolicyHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.shared = nn.Sequential(
            nn.Linear(cfg.embed_dim + cfg.regime_embed_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.playbook = nn.Linear(cfg.hidden_dim, cfg.n_playbooks)
        self.risk_budget = nn.Linear(cfg.hidden_dim, 1)
        self.tradeability = nn.Linear(cfg.hidden_dim, 1)
        self.transition_wait = nn.Linear(cfg.hidden_dim, 1)
        self.no_trade = nn.Linear(cfg.hidden_dim, 1)
        self.size_multiplier = nn.Linear(cfg.hidden_dim, 1)
        self.playbook_prior = nn.Linear(cfg.hidden_dim, 1)
        self.execution_order_type = nn.Linear(cfg.hidden_dim, len(EXECUTION_ORDER_TYPES))
        self.execution_urgency = nn.Linear(cfg.hidden_dim, len(EXECUTION_URGENCIES))
        self.execution_reduce_only = nn.Linear(cfg.hidden_dim, 1)
        self.execution_scale_in = nn.Linear(cfg.hidden_dim, 1)
        self.execution_slippage = nn.Linear(cfg.hidden_dim, 1)
        self.position_intent = nn.Linear(cfg.hidden_dim, len(POSITION_INTENTS))

    def forward(self, policy_embedding: torch.Tensor, regime_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.shared(torch.cat([policy_embedding, regime_embedding], dim=-1))
        return {
            "regime_playbook_logits": self.playbook(h),
            "regime_risk_budget": torch.sigmoid(self.risk_budget(h)).squeeze(-1),
            "regime_tradeability": torch.sigmoid(self.tradeability(h)).squeeze(-1),
            "regime_transition_wait": torch.sigmoid(self.transition_wait(h)).squeeze(-1),
            "regime_no_trade": torch.sigmoid(self.no_trade(h)).squeeze(-1),
            "regime_size_multiplier": torch.sigmoid(self.size_multiplier(h)).squeeze(-1),
            "regime_playbook_prior": torch.sigmoid(self.playbook_prior(h)).squeeze(-1),
            "execution_order_type_logits": self.execution_order_type(h),
            "execution_urgency_logits": self.execution_urgency(h),
            "execution_reduce_only": torch.sigmoid(self.execution_reduce_only(h)).squeeze(-1),
            "execution_scale_in": torch.sigmoid(self.execution_scale_in(h)).squeeze(-1),
            "execution_max_slippage": torch.sigmoid(self.execution_slippage(h)).squeeze(-1),
            "position_intent_logits": self.position_intent(h),
        }


def _as_batch_tensor(values: Sequence[float], *, device: torch.device | str | None = None) -> torch.Tensor:
    return torch.as_tensor(list(values), dtype=torch.float32, device=device)


def build_regime_policy_targets(
    reports: Sequence[RegimeReport],
    *,
    plans: Sequence[TradePlan] | None = None,
    current_positions: Sequence[float] | None = None,
    n_actions: int = 9,
    device: torch.device | str | None = None,
    cfg: RegimePolicyTargetConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Convert regime reports/plans into auxiliary policy-training targets."""
    cfg = cfg or RegimePolicyTargetConfig()
    playbook_labels: list[int] = []
    playbook_fallback: list[float] = []
    playbook_prior: list[float] = []
    risk_budget: list[float] = []
    tradeability: list[float] = []
    transition_wait: list[float] = []
    no_trade: list[float] = []
    size_multiplier: list[float] = []
    order_type_labels: list[int] = []
    urgency_labels: list[int] = []
    reduce_only: list[float] = []
    scale_in: list[float] = []
    slippage: list[float] = []
    position_intent_labels: list[int] = []
    masks: list[np.ndarray] = []
    positions = list(current_positions or [0.0] * len(reports))
    for i, report in enumerate(reports):
        plan = plans[i] if plans is not None and i < len(plans) else None
        model_outputs = report.features.get("model_outputs", {}) if report.features else {}
        playbook_utility = report.features.get("playbook_utility", {}) if report.features else {}
        playbook = plan.recommended_playbook if plan is not None else ""
        if not playbook and isinstance(playbook_utility, dict) and playbook_utility:
            playbook = max(playbook_utility, key=lambda k: float(playbook_utility.get(k, -1e9)))
        fallback_used = False
        if playbook not in PLAYBOOK_TO_ID:
            replacement = next((p for p in report.allowed_playbooks if p in PLAYBOOK_TO_ID), cfg.unknown_playbook_fallback)
            fallback_used = replacement == cfg.unknown_playbook_fallback
            playbook = replacement
        playbook_labels.append(int(PLAYBOOK_TO_ID.get(playbook, PLAYBOOK_TO_ID["no_trade_wait"])))
        playbook_fallback.append(float(fallback_used))
        prior_dict = {}
        if plan is not None and plan.setups:
            for setup in plan.setups:
                if setup.playbook == playbook:
                    prior_dict = setup.memory_prior or {}
                    break
        if isinstance(playbook_utility, dict) and playbook in playbook_utility:
            prior_score = float(1.0 / (1.0 + np.exp(-float(playbook_utility[playbook]))))
        else:
            prior_score = float(prior_dict.get("reliability", cfg.default_playbook_prior)) if prior_dict else cfg.default_playbook_prior
        playbook_prior.append(float(np.clip(prior_score, 0.0, 1.0)))
        rb = float(model_outputs.get("risk_budget", plan.risk_budget if plan is not None else regime_position_size_multiplier(report))) if isinstance(model_outputs, dict) else (float(plan.risk_budget) if plan is not None else regime_position_size_multiplier(report))
        risk_budget.append(float(np.clip(rb, 0.0, 1.0)))
        trade = float(model_outputs.get("tradeability", report.tradeability_score)) if isinstance(model_outputs, dict) else float(report.tradeability_score)
        transition_score = float(model_outputs.get("transition_hazard", report.transition_risk)) if isinstance(model_outputs, dict) else float(report.transition_risk)
        no_trade_prob = float(model_outputs.get("no_trade_probability", -1.0)) if isinstance(model_outputs, dict) else -1.0
        tradeability.append(float(np.clip(trade, 0.0, 1.0)))
        transition_wait.append(float(transition_score >= cfg.transition_wait_threshold or "transition_wait" in report.allowed_playbooks))
        no_trade.append(float((plan.status == "no_trade") if plan is not None else (no_trade_prob >= 0.5 or "no_trade_wait" in report.allowed_playbooks or trade < cfg.no_trade_tradeability_threshold)))
        size_multiplier.append(float(np.clip(report.position_size_multiplier, 0.0, 1.0)))
        if plan is not None:
            order_type_labels.append(ORDER_TYPE_TO_ID.get(plan.execution.order_type, ORDER_TYPE_TO_ID["none"]))
            urgency_labels.append(URGENCY_TO_ID.get(plan.execution.urgency, URGENCY_TO_ID["wait"]))
            reduce_only.append(float(plan.execution.reduce_only))
            scale_in.append(float(np.clip(plan.execution.scale_in_steps / cfg.max_scale_in_steps, 0.0, 1.0)))
            max_bucket = max(cfg.max_slippage_bps, 1e-12)
            slippage.append(float(np.clip(plan.execution.max_slippage_bps / max_bucket, 0.0, 1.0)))
            position_intent_labels.append(POSITION_INTENT_TO_ID.get(plan.position_management.intent, POSITION_INTENT_TO_ID["hold"]))
        else:
            order_type_labels.append(ORDER_TYPE_TO_ID["none"])
            urgency_labels.append(URGENCY_TO_ID["wait"])
            reduce_only.append(float(no_trade[-1]))
            scale_in.append(0.0)
            slippage.append(0.0)
            position_intent_labels.append(POSITION_INTENT_TO_ID["hold_or_wait"] if no_trade[-1] else POSITION_INTENT_TO_ID["hold"])
        pos = float(positions[i]) if i < len(positions) else 0.0
        masks.append(regime_action_mask(report, current_position=pos, n_actions=n_actions))
    return {
        "regime_playbook_label": torch.as_tensor(playbook_labels, dtype=torch.long, device=device),
        "regime_playbook_fallback": _as_batch_tensor(playbook_fallback, device=device),
        "regime_playbook_prior": _as_batch_tensor(playbook_prior, device=device),
        "regime_risk_budget": _as_batch_tensor(risk_budget, device=device),
        "regime_tradeability": _as_batch_tensor(tradeability, device=device),
        "regime_transition_wait": _as_batch_tensor(transition_wait, device=device),
        "regime_no_trade": _as_batch_tensor(no_trade, device=device),
        "regime_size_multiplier": _as_batch_tensor(size_multiplier, device=device),
        "execution_order_type_label": torch.as_tensor(order_type_labels, dtype=torch.long, device=device),
        "execution_urgency_label": torch.as_tensor(urgency_labels, dtype=torch.long, device=device),
        "execution_reduce_only": _as_batch_tensor(reduce_only, device=device),
        "execution_scale_in": _as_batch_tensor(scale_in, device=device),
        "execution_max_slippage": _as_batch_tensor(slippage, device=device),
        "position_intent_label": torch.as_tensor(position_intent_labels, dtype=torch.long, device=device),
        "regime_action_mask": torch.as_tensor(np.stack(masks), dtype=torch.bool, device=device),
    }


class RegimePolicyAuxLoss(nn.Module):
    """Auxiliary loss that teaches the policy to respect regime intelligence."""

    def __init__(self, weights: RegimePolicyLossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or RegimePolicyLossWeights()

    def forward(self, outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        device = outputs["policy_logits"].device
        if "regime_playbook_logits" in outputs and "regime_playbook_label" in targets:
            losses["playbook"] = F.cross_entropy(
                outputs["regime_playbook_logits"],
                targets["regime_playbook_label"].to(device=device),
            )
        for name in ("risk_budget", "tradeability", "size_multiplier"):
            out_key = f"regime_{name}"
            if out_key in outputs and out_key in targets:
                losses[name] = F.smooth_l1_loss(outputs[out_key], targets[out_key].to(device=device, dtype=outputs[out_key].dtype))
        if "regime_playbook_prior" in outputs and "regime_playbook_prior" in targets:
            losses["playbook_prior"] = F.smooth_l1_loss(
                outputs["regime_playbook_prior"],
                targets["regime_playbook_prior"].to(device=device, dtype=outputs["regime_playbook_prior"].dtype),
            )
        for name in ("transition_wait", "no_trade"):
            out_key = f"regime_{name}"
            if out_key in outputs and out_key in targets:
                pred = outputs[out_key].clamp(1e-6, 1.0 - 1e-6)
                losses[name] = F.binary_cross_entropy(pred, targets[out_key].to(device=device, dtype=pred.dtype))
        if "execution_order_type_logits" in outputs and "execution_order_type_label" in targets:
            losses["execution_order_type"] = F.cross_entropy(
                outputs["execution_order_type_logits"],
                targets["execution_order_type_label"].to(device=device),
            )
        if "execution_urgency_logits" in outputs and "execution_urgency_label" in targets:
            losses["execution_urgency"] = F.cross_entropy(
                outputs["execution_urgency_logits"],
                targets["execution_urgency_label"].to(device=device),
            )
        if "position_intent_logits" in outputs and "position_intent_label" in targets:
            losses["position_intent"] = F.cross_entropy(
                outputs["position_intent_logits"],
                targets["position_intent_label"].to(device=device),
            )
        for name in ("execution_reduce_only", "execution_scale_in", "execution_max_slippage"):
            if name in outputs and name in targets:
                pred = outputs[name].clamp(1e-6, 1.0 - 1e-6) if name == "execution_reduce_only" else outputs[name]
                target = targets[name].to(device=device, dtype=pred.dtype)
                loss_name = "execution_slippage" if name == "execution_max_slippage" else name
                losses[loss_name] = (
                    F.binary_cross_entropy(pred, target)
                    if name == "execution_reduce_only"
                    else F.smooth_l1_loss(pred, target)
                )
        if "policy_logits" in outputs and "regime_action_mask" in targets:
            mask = targets["regime_action_mask"].to(device=device, dtype=torch.bool)
            logits = outputs["policy_logits"]
            if mask.shape == logits.shape:
                probs = torch.softmax(logits, dim=-1)
                blocked_mass = probs.masked_fill(mask, 0.0).sum(dim=-1)
                losses["action_constraint"] = blocked_mass.mean()
        total = torch.zeros((), device=device)
        for name, loss in losses.items():
            total = total + float(getattr(self.weights, name)) * loss
        losses["total"] = total
        return losses


class RegimeAwarePolicyNetwork(nn.Module):
    """PolicyNetwork with a trainable regime embedding appended to context."""

    def __init__(self, cfg: Optional[RegimeAwarePolicyConfig] = None) -> None:
        super().__init__()
        cfg = cfg or RegimeAwarePolicyConfig()
        self.cfg = cfg
        self.regime_encoder = RegimeEncoder(cfg.regime_encoder)
        extra_context = cfg.regime_encoder.embed_dim if cfg.append_regime_embedding else 0
        policy_cfg = replace(
            cfg.base_policy,
            in_context_features=cfg.base_policy.in_context_features + extra_context,
        )
        self.policy = PolicyNetwork(policy_cfg)
        self.regime_heads = None
        if cfg.enable_regime_heads:
            self.regime_heads = RegimePolicyHeads(
                RegimePolicyHeadConfig(
                    embed_dim=policy_cfg.embed_dim,
                    regime_embed_dim=cfg.regime_encoder.embed_dim,
                    hidden_dim=max(policy_cfg.embed_dim, cfg.regime_encoder.hidden_dim),
                    n_playbooks=cfg.n_playbooks,
                    dropout=policy_cfg.dropout,
                )
            )
        if cfg.freeze_regime_encoder:
            for param in self.regime_encoder.parameters():
                param.requires_grad_(False)

    def _encode_regime(
        self,
        context: torch.Tensor,
        regime: torch.Tensor | np.ndarray | Sequence[RegimeReport] | None,
        regime_embedding: torch.Tensor | np.ndarray | None,
    ) -> dict[str, torch.Tensor]:
        if regime_embedding is not None:
            emb = regime_embedding if isinstance(regime_embedding, torch.Tensor) else torch.as_tensor(regime_embedding, dtype=context.dtype, device=context.device)
            emb = emb.to(device=context.device, dtype=context.dtype)
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            return {"embedding": emb}
        if regime is None:
            zeros = torch.zeros(
                context.size(0),
                self.regime_encoder.input_dim,
                device=context.device,
                dtype=context.dtype,
            )
            return self.regime_encoder(zeros)
        return self.regime_encoder(regime)

    def forward(
        self,
        chart: torch.Tensor,
        numeric: torch.Tensor,
        context: torch.Tensor,
        *,
        regime: torch.Tensor | np.ndarray | Sequence[RegimeReport] | None = None,
        regime_embedding: torch.Tensor | np.ndarray | None = None,
        history: Optional[torch.Tensor] = None,
        instrument_id: Optional[torch.Tensor] = None,
    ) -> dict:
        regime_out = self._encode_regime(context, regime, regime_embedding)
        policy_context = context
        if self.cfg.append_regime_embedding:
            policy_context = append_regime_context(context, regime_out["embedding"])
        out = self.policy(
            chart=chart,
            numeric=numeric,
            context=policy_context,
            history=history,
            instrument_id=instrument_id,
        )
        out["regime_embedding"] = regime_out["embedding"]
        for key, value in regime_out.items():
            if key != "embedding":
                out[f"regime_{key}"] = value
        if self.regime_heads is not None:
            out.update(self.regime_heads(out["embedding"], regime_out["embedding"]))
        return out


def build_regime_aware_policy(
    in_numeric_features: int = 32,
    in_context_features: int = 10,
    n_actions: int = 9,
    n_regime_classes: int = 4,
    regime_embed_dim: int = 32,
    **kwargs,
) -> RegimeAwarePolicyNetwork:
    base_cfg = PolicyConfig(
        in_numeric_features=in_numeric_features,
        in_context_features=in_context_features,
        n_actions=n_actions,
        n_regime_classes=n_regime_classes,
        **kwargs,
    )
    cfg = RegimeAwarePolicyConfig(
        base_policy=base_cfg,
        regime_encoder=RegimeEncoderConfig(embed_dim=regime_embed_dim),
    )
    return RegimeAwarePolicyNetwork(cfg)


__all__ = [
    "EXECUTION_ORDER_TYPES",
    "EXECUTION_URGENCIES",
    "ORDER_TYPE_TO_ID",
    "POSITION_INTENTS",
    "POSITION_INTENT_TO_ID",
    "RegimePolicyAuxLoss",
    "RegimeAwarePolicyConfig",
    "RegimeAwarePolicyNetwork",
    "RegimePolicyHeadConfig",
    "RegimePolicyHeads",
    "RegimePolicyLossWeights",
    "RegimePolicyTargetConfig",
    "SLIPPAGE_BUCKETS_BPS",
    "URGENCY_TO_ID",
    "build_regime_policy_targets",
    "build_regime_aware_policy",
]
