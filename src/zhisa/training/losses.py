"""Multi-task losses: direction, volatility, regime, return, action imitation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossWeights:
    direction: float = 1.0
    volatility: float = 0.5
    regime: float = 0.3
    return_pred: float = 0.5
    risk: float = 0.25
    policy: float = 1.0
    value: float = 0.5
    uncertainty: float = 0.05
    regime_playbook: float = 0.5
    regime_playbook_prior: float = 0.2
    regime_risk_budget: float = 0.5
    regime_tradeability: float = 0.35
    regime_transition_wait: float = 0.35
    regime_no_trade: float = 0.35
    regime_size_multiplier: float = 0.35
    regime_action_constraint: float = 0.5
    execution_order_type: float = 0.3
    execution_urgency: float = 0.25
    execution_reduce_only: float = 0.25
    execution_scale_in: float = 0.2
    execution_slippage: float = 0.2
    position_intent: float = 0.25
    uncertainty_warmup: int = 200


class MultiTaskLoss(nn.Module):
    """A weighted sum of task-specific losses, with optional uncertainty
    weighting (Kendall et al. 2018).

    For the MVP we use fixed weights; the uncertainty-weighting version
    is exposed via the `learnable` flag.
    """

    def __init__(
        self,
        weights: Optional[LossWeights] = None,
        learnable: bool = False,
        label_smoothing: float = 0.05,
        direction_class_weights: Optional[torch.Tensor] = None,
        regime_class_weights: Optional[torch.Tensor] = None,
        policy_class_weights: Optional[torch.Tensor] = None,
        policy_focal_gamma: float = 0.0,
        policy_direction_aux_weight: float = 0.0,
        policy_size_aux_weight: float = 0.0,
    ) -> None:
        super().__init__()
        w = weights or LossWeights()
        self.label_smoothing = label_smoothing
        self.policy_focal_gamma = float(policy_focal_gamma)
        self.policy_direction_aux_weight = float(policy_direction_aux_weight)
        self.policy_size_aux_weight = float(policy_size_aux_weight)
        self.register_buffer(
            "direction_class_weights",
            None if direction_class_weights is None else direction_class_weights.float(),
        )
        self.register_buffer(
            "regime_class_weights",
            None if regime_class_weights is None else regime_class_weights.float(),
        )
        self.register_buffer(
            "policy_class_weights",
            None if policy_class_weights is None else policy_class_weights.float(),
        )
        if learnable:
            self.log_vars = nn.ParameterDict({
                k: nn.Parameter(torch.zeros(1)) for k in (
                    "direction", "volatility", "regime", "return_pred",
                    "risk", "policy", "value", "uncertainty",
                    "regime_playbook", "regime_playbook_prior", "regime_risk_budget",
                    "regime_tradeability", "regime_transition_wait",
                    "regime_no_trade", "regime_size_multiplier",
                    "regime_action_constraint",
                    "execution_order_type", "execution_urgency",
                    "execution_reduce_only", "execution_scale_in",
                    "execution_slippage", "position_intent",
                )
            })
        else:
            self.log_vars = None
            self.register_buffer("_direction_w", torch.tensor(w.direction))
            self.register_buffer("_volatility_w", torch.tensor(w.volatility))
            self.register_buffer("_regime_w", torch.tensor(w.regime))
            self.register_buffer("_return_pred_w", torch.tensor(w.return_pred))
            self.register_buffer("_risk_w", torch.tensor(w.risk))
            self.register_buffer("_policy_w", torch.tensor(w.policy))
            self.register_buffer("_value_w", torch.tensor(w.value))
            self.register_buffer("_uncertainty_w", torch.tensor(w.uncertainty))
            self.register_buffer("_regime_playbook_w", torch.tensor(w.regime_playbook))
            self.register_buffer("_regime_playbook_prior_w", torch.tensor(w.regime_playbook_prior))
            self.register_buffer("_regime_risk_budget_w", torch.tensor(w.regime_risk_budget))
            self.register_buffer("_regime_tradeability_w", torch.tensor(w.regime_tradeability))
            self.register_buffer("_regime_transition_wait_w", torch.tensor(w.regime_transition_wait))
            self.register_buffer("_regime_no_trade_w", torch.tensor(w.regime_no_trade))
            self.register_buffer("_regime_size_multiplier_w", torch.tensor(w.regime_size_multiplier))
            self.register_buffer("_regime_action_constraint_w", torch.tensor(w.regime_action_constraint))
            self.register_buffer("_execution_order_type_w", torch.tensor(w.execution_order_type))
            self.register_buffer("_execution_urgency_w", torch.tensor(w.execution_urgency))
            self.register_buffer("_execution_reduce_only_w", torch.tensor(w.execution_reduce_only))
            self.register_buffer("_execution_scale_in_w", torch.tensor(w.execution_scale_in))
            self.register_buffer("_execution_slippage_w", torch.tensor(w.execution_slippage))
            self.register_buffer("_position_intent_w", torch.tensor(w.position_intent))
        self.weights = w

    def set_policy_class_weights(self, weights: Optional[torch.Tensor]) -> None:
        """Update policy class weights while preserving buffer semantics."""
        if weights is None:
            self._buffers["policy_class_weights"] = None
            return
        ref = next(self.parameters(), None)
        if ref is None:
            ref = next((buf for buf in self.buffers() if buf is not None), None)
        device = ref.device if ref is not None else weights.device
        self._buffers["policy_class_weights"] = weights.detach().to(
            device=device,
            dtype=torch.float32,
        )

    def _w(self, key: str) -> torch.Tensor:
        if self.log_vars is not None:
            return torch.exp(-self.log_vars[key])
        return getattr(self, f"_{key}_w")

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        # Direction: labels in {-1, 0, +1} -> map to {0, 1, 2}
        if "direction" in outputs and "label_dir" in targets:
            tgt = targets["label_dir"].clone()
            tgt = torch.where(tgt == -1, torch.zeros_like(tgt), tgt + 1)
            losses["direction"] = F.cross_entropy(
                outputs["direction"], tgt,
                weight=self.direction_class_weights,
                label_smoothing=self.label_smoothing,
            )
        # Volatility: regression
        if "volatility" in outputs and "label_vol" in targets:
            losses["volatility"] = F.smooth_l1_loss(
                outputs["volatility"], targets["label_vol"]
            )
        # Regime: classification
        if "regime" in outputs and "label_regime" in targets:
            losses["regime"] = F.cross_entropy(
                outputs["regime"],
                targets["label_regime"],
                weight=self.regime_class_weights,
            )
        # Return prediction
        if "return_pred" in outputs and "label_ret" in targets:
            losses["return_pred"] = F.smooth_l1_loss(
                outputs["return_pred"], targets["label_ret"]
            )
        # Risk head: explicit downside-risk label when available; legacy
        # batches derive it from downside return plus realised volatility.
        if "risk" in outputs:
            risk_tgt = targets.get("label_risk")
            if risk_tgt is None and "label_ret" in targets:
                risk_tgt = torch.relu(-targets["label_ret"])
                if "label_vol" in targets:
                    risk_tgt = risk_tgt + torch.relu(targets["label_vol"])
            if risk_tgt is not None:
                losses["risk"] = F.smooth_l1_loss(
                    outputs["risk"], risk_tgt.to(outputs["risk"].device)
                )
        # Uncertainty head: predict log-variance for the return head.
        if (
            "uncertainty_logit" in outputs
            and "return_pred" in outputs
            and "label_ret" in targets
        ):
            log_var = outputs["uncertainty_logit"].clamp(min=-10.0, max=10.0)
            err = outputs["return_pred"] - targets["label_ret"].to(log_var.device)
            losses["uncertainty"] = 0.5 * torch.mean(
                torch.exp(-log_var) * err.pow(2) + log_var
            )
        # Policy imitation: cross-entropy
        if "policy_logits" in outputs and "action" in targets:
            logits = outputs["policy_logits"]
            action_target = targets["action"]
            if self.policy_focal_gamma > 0:
                ce = F.cross_entropy(
                    logits, action_target,
                    reduction="none",
                )
                target_prob = torch.softmax(logits, dim=-1).gather(
                    1, action_target.unsqueeze(1)
                ).squeeze(1)
                focal_weight = (1.0 - target_prob).pow(self.policy_focal_gamma)
                if self.policy_class_weights is not None:
                    sample_weight = self.policy_class_weights.to(logits.device)[action_target]
                    losses["policy"] = (
                        sample_weight * focal_weight * ce
                    ).sum() / sample_weight.sum().clamp_min(1e-8)
                else:
                    losses["policy"] = (focal_weight * ce).mean()
            else:
                losses["policy"] = F.cross_entropy(
                    logits, action_target,
                    weight=self.policy_class_weights,
                    label_smoothing=self.label_smoothing,
                )
            if self.policy_direction_aux_weight > 0:
                direction_logits = torch.stack([
                    torch.logsumexp(logits[:, [0, 7, 8]], dim=1),
                    torch.logsumexp(logits[:, [1, 2, 3]], dim=1),
                    torch.logsumexp(logits[:, [4, 5, 6]], dim=1),
                ], dim=1)
                direction_target = torch.zeros_like(action_target)
                direction_target[(action_target >= 1) & (action_target <= 3)] = 1
                direction_target[(action_target >= 4) & (action_target <= 6)] = 2
                losses["policy"] = losses["policy"] + self.policy_direction_aux_weight * F.cross_entropy(
                    direction_logits, direction_target
                )
            if self.policy_size_aux_weight > 0:
                trade = (action_target >= 1) & (action_target <= 6)
                if trade.any():
                    is_long = action_target[trade] <= 3
                    trade_logits = logits[trade]
                    size_logits = torch.where(
                        is_long.unsqueeze(1), trade_logits[:, [1, 2, 3]], trade_logits[:, [4, 5, 6]]
                    )
                    size_target = torch.where(
                        is_long, action_target[trade] - 1, action_target[trade] - 4
                    )
                    losses["policy"] = losses["policy"] + self.policy_size_aux_weight * F.cross_entropy(
                        size_logits, size_target
                    )
        # Value: regression to label_ret
        if "value" in outputs and "label_ret" in targets:
            losses["value"] = F.smooth_l1_loss(outputs["value"], targets["label_ret"])

        if "regime_playbook_logits" in outputs and "regime_playbook_label" in targets:
            losses["regime_playbook"] = F.cross_entropy(
                outputs["regime_playbook_logits"],
                targets["regime_playbook_label"].to(outputs["regime_playbook_logits"].device),
            )
        if "regime_playbook_prior" in outputs and "regime_playbook_prior" in targets:
            losses["regime_playbook_prior"] = F.smooth_l1_loss(
                outputs["regime_playbook_prior"],
                targets["regime_playbook_prior"].to(outputs["regime_playbook_prior"].device, dtype=outputs["regime_playbook_prior"].dtype),
            )
        for name in ("risk_budget", "tradeability", "size_multiplier"):
            out_key = f"regime_{name}"
            if out_key in outputs and out_key in targets:
                losses[f"regime_{name}"] = F.smooth_l1_loss(
                    outputs[out_key],
                    targets[out_key].to(outputs[out_key].device, dtype=outputs[out_key].dtype),
                )
        for name in ("transition_wait", "no_trade"):
            out_key = f"regime_{name}"
            if out_key in outputs and out_key in targets:
                pred = outputs[out_key].clamp(1e-6, 1.0 - 1e-6)
                losses[f"regime_{name}"] = F.binary_cross_entropy(
                    pred,
                    targets[out_key].to(pred.device, dtype=pred.dtype),
                )
        if "policy_logits" in outputs and "regime_action_mask" in targets:
            mask = targets["regime_action_mask"].to(outputs["policy_logits"].device, dtype=torch.bool)
            if mask.shape == outputs["policy_logits"].shape:
                probs = torch.softmax(outputs["policy_logits"], dim=-1)
                losses["regime_action_constraint"] = probs.masked_fill(mask, 0.0).sum(dim=-1).mean()
        if "execution_order_type_logits" in outputs and "execution_order_type_label" in targets:
            losses["execution_order_type"] = F.cross_entropy(
                outputs["execution_order_type_logits"],
                targets["execution_order_type_label"].to(outputs["execution_order_type_logits"].device),
            )
        if "execution_urgency_logits" in outputs and "execution_urgency_label" in targets:
            losses["execution_urgency"] = F.cross_entropy(
                outputs["execution_urgency_logits"],
                targets["execution_urgency_label"].to(outputs["execution_urgency_logits"].device),
            )
        if "position_intent_logits" in outputs and "position_intent_label" in targets:
            losses["position_intent"] = F.cross_entropy(
                outputs["position_intent_logits"],
                targets["position_intent_label"].to(outputs["position_intent_logits"].device),
            )
        if "execution_reduce_only" in outputs and "execution_reduce_only" in targets:
            pred = outputs["execution_reduce_only"].clamp(1e-6, 1.0 - 1e-6)
            losses["execution_reduce_only"] = F.binary_cross_entropy(
                pred,
                targets["execution_reduce_only"].to(pred.device, dtype=pred.dtype),
            )
        if "execution_scale_in" in outputs and "execution_scale_in" in targets:
            losses["execution_scale_in"] = F.smooth_l1_loss(
                outputs["execution_scale_in"],
                targets["execution_scale_in"].to(outputs["execution_scale_in"].device, dtype=outputs["execution_scale_in"].dtype),
            )
        if "execution_max_slippage" in outputs and "execution_max_slippage" in targets:
            losses["execution_slippage"] = F.smooth_l1_loss(
                outputs["execution_max_slippage"],
                targets["execution_max_slippage"].to(outputs["execution_max_slippage"].device, dtype=outputs["execution_max_slippage"].dtype),
            )

        total = torch.zeros((), device=outputs["direction"].device)
        for k, v in losses.items():
            total = total + self._w(k) * v
        losses["total"] = total
        return losses
