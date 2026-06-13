"""Regime-aware action gating and risk sizing."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from zhisa.env.actions import DiscreteAction
from zhisa.env.portfolio_action_mask import mask_logits
from zhisa.regime.schema import MacroRegime, MesoRegime, RegimeReport, RiskMode


@dataclass(frozen=True)
class RegimeActionGateConfig:
    min_tradeability_for_new_risk: float = 0.25
    defensive_max_abs_target: float = 0.25
    reduced_max_abs_target: float = 0.5
    normal_max_abs_target: float = 1.0
    aggressive_max_abs_target: float = 1.0
    block_countertrend_full_size: bool = True
    block_new_risk_in_crash: bool = False


ACTION_TARGETS: dict[int, float] = {
    int(DiscreteAction.LONG_25): 0.25,
    int(DiscreteAction.LONG_50): 0.50,
    int(DiscreteAction.LONG_100): 1.00,
    int(DiscreteAction.SHORT_25): -0.25,
    int(DiscreteAction.SHORT_50): -0.50,
    int(DiscreteAction.SHORT_100): -1.00,
    int(DiscreteAction.CLOSE): 0.0,
}


def _target(action: int, current_position: float) -> float:
    if int(action) == int(DiscreteAction.SKIP):
        return float(current_position)
    if int(action) == int(DiscreteAction.PARTIAL_CLOSE):
        return 0.5 * float(current_position)
    return ACTION_TARGETS.get(int(action), 0.0)


def _is_de_risking(action: int, current_position: float) -> bool:
    if int(action) in {int(DiscreteAction.CLOSE), int(DiscreteAction.PARTIAL_CLOSE)}:
        return True
    target = _target(action, current_position)
    current = float(current_position)
    if abs(current) <= 1e-9:
        return False
    if abs(target) <= 1e-9:
        return True
    return np.sign(target) == np.sign(current) and abs(target) < abs(current) - 1e-9


def _max_abs_target(report: RegimeReport, cfg: RegimeActionGateConfig) -> float:
    risk_mode = report.risk_mode
    if risk_mode == RiskMode.OFF.value:
        return 0.0
    if risk_mode == RiskMode.DEFENSIVE.value:
        return cfg.defensive_max_abs_target
    if risk_mode == RiskMode.REDUCED.value:
        return cfg.reduced_max_abs_target
    if risk_mode == RiskMode.AGGRESSIVE.value:
        return cfg.aggressive_max_abs_target
    return cfg.normal_max_abs_target


def regime_action_mask(
    report: RegimeReport,
    *,
    current_position: float = 0.0,
    n_actions: int = 9,
    cfg: RegimeActionGateConfig | None = None,
) -> np.ndarray:
    """Return boolean mask for discrete actions under the current regime."""
    cfg = cfg or RegimeActionGateConfig()
    mask = np.ones(int(n_actions), dtype=bool)
    max_abs = _max_abs_target(report, cfg)
    macro = report.primary_regime
    meso = report.secondary_regime
    risk_mode = report.risk_mode
    tradeability = float(report.tradeability_score)

    for a in range(int(n_actions)):
        action = int(a)
        target = _target(action, current_position)
        de_risking = _is_de_risking(action, current_position)
        increasing = abs(target) > abs(float(current_position)) + 1e-9
        flips_direction = (
            abs(float(current_position)) > 1e-9
            and abs(target) > 1e-9
            and np.sign(target) != np.sign(float(current_position))
        )
        new_risk = increasing or flips_direction

        if de_risking or action == int(DiscreteAction.SKIP):
            mask[a] = True
            continue
        if (
            not increasing
            and abs(target) <= abs(float(current_position)) + 1e-9
            and (abs(target) <= 1e-9 or np.sign(target) == np.sign(float(current_position)))
        ):
            mask[a] = True
            continue
        if risk_mode == RiskMode.OFF.value:
            mask[a] = False
            continue
        if tradeability < cfg.min_tradeability_for_new_risk and new_risk:
            mask[a] = False
            continue
        if abs(target) > max_abs + 1e-9:
            mask[a] = False
            continue
        if cfg.block_new_risk_in_crash and macro == MacroRegime.HIGH_VOL_CRASH.value and new_risk:
            mask[a] = False
            continue
        if macro == MacroRegime.HIGH_VOL_CRASH.value and target > 0:
            mask[a] = False
            continue
        if meso == MesoRegime.COMPRESSION.value and abs(target) >= 1.0:
            mask[a] = False
            continue
        if cfg.block_countertrend_full_size:
            if macro == MacroRegime.BULL_TREND.value and target <= -1.0:
                mask[a] = False
            if macro == MacroRegime.BEAR_TREND.value and target >= 1.0:
                mask[a] = False
    return mask


def apply_regime_action_mask(
    logits: torch.Tensor,
    report: RegimeReport,
    *,
    current_position: float = 0.0,
    cfg: RegimeActionGateConfig | None = None,
    neg_value: float = -1e9,
) -> torch.Tensor:
    """Mask policy logits using the regime-aware discrete action gate."""
    mask_np = regime_action_mask(
        report,
        current_position=current_position,
        n_actions=logits.shape[-1],
        cfg=cfg,
    )
    mask_t = torch.as_tensor(mask_np, dtype=torch.bool, device=logits.device)
    while mask_t.dim() < logits.dim():
        mask_t = mask_t.unsqueeze(0)
    return mask_logits(logits, mask_t.expand_as(logits), neg_value=neg_value)


def regime_position_size_multiplier(
    report: RegimeReport,
    *,
    cfg: RegimeActionGateConfig | None = None,
) -> float:
    """Blend report sizing, tradeability, uncertainty, and transition risk."""
    cfg = cfg or RegimeActionGateConfig()
    max_abs = _max_abs_target(report, cfg)
    base = float(report.position_size_multiplier)
    quality = max(0.0, float(report.tradeability_score))
    uncertainty_penalty = 1.0 - min(1.0, float(report.uncertainty))
    transition_penalty = 1.0 - 0.5 * min(1.0, float(report.transition_risk))
    out = base * quality * uncertainty_penalty * transition_penalty
    return float(np.clip(out, 0.0, max_abs))


__all__ = [
    "ACTION_TARGETS",
    "RegimeActionGateConfig",
    "apply_regime_action_mask",
    "regime_action_mask",
    "regime_position_size_multiplier",
]
