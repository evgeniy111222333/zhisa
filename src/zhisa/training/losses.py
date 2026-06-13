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
    ) -> None:
        super().__init__()
        w = weights or LossWeights()
        self.label_smoothing = label_smoothing
        if learnable:
            self.log_vars = nn.ParameterDict({
                k: nn.Parameter(torch.zeros(1)) for k in (
                    "direction", "volatility", "regime", "return_pred",
                    "risk", "policy", "value", "uncertainty",
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
        self.weights = w

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
                label_smoothing=self.label_smoothing,
            )
        # Volatility: regression
        if "volatility" in outputs and "label_vol" in targets:
            losses["volatility"] = F.smooth_l1_loss(
                outputs["volatility"], targets["label_vol"]
            )
        # Regime: classification
        if "regime" in outputs and "label_regime" in targets:
            losses["regime"] = F.cross_entropy(outputs["regime"], targets["label_regime"])
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
            losses["policy"] = F.cross_entropy(
                outputs["policy_logits"], targets["action"],
                label_smoothing=self.label_smoothing,
            )
        # Value: regression to label_ret
        if "value" in outputs and "label_ret" in targets:
            losses["value"] = F.smooth_l1_loss(outputs["value"], targets["label_ret"])

        total = torch.zeros((), device=outputs["direction"].device)
        for k, v in losses.items():
            total = total + self._w(k) * v
        losses["total"] = total
        return losses
