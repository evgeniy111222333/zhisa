"""A/B backtests for baseline vs regime-aware policies."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from zhisa.backtest.engine import BacktestResult, PolicyFn, run_backtest
from zhisa.env.actions import DiscreteAction
from zhisa.env.trading_env import EnvConfig, TradingEnv
from zhisa.regime.controller import RegimeAdaptiveController, RegimeFeedbackConfig
from zhisa.regime.detector import RegimeIntelligence, RegimeIntelligenceConfig
from zhisa.regime.gating import RegimeActionGateConfig, apply_regime_action_mask, regime_action_mask
from zhisa.regime.planner import RegimeTradePlanner, TradePlan
from zhisa.regime.profiles import build_regime_profile_config, resolve_regime_profile
from zhisa.regime.schema import RegimeReport


@dataclass(frozen=True)
class RegimeABConfig:
    analyzer: RegimeIntelligenceConfig = field(default_factory=RegimeIntelligenceConfig)
    analyzer_profile: str = ""
    symbol: str = ""
    asset_class: str = ""
    venue: str = ""
    benchmark_symbol: str = ""
    gate: RegimeActionGateConfig = field(default_factory=RegimeActionGateConfig)
    gated_name: str = "regime_gated"
    baseline_name: str = "baseline"
    fallback_action: int = int(DiscreteAction.CLOSE)
    adaptive_controller: bool = False
    feedback: RegimeFeedbackConfig = field(default_factory=RegimeFeedbackConfig)


@dataclass(frozen=True)
class RegimeVariantResult:
    name: str
    result: BacktestResult
    regime_summary: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegimeABResult:
    baseline: RegimeVariantResult
    gated: RegimeVariantResult
    comparison: dict[str, Any]


def _metric_delta(baseline: BacktestResult, gated: BacktestResult) -> dict[str, float]:
    b = baseline.metrics.to_dict()
    g = gated.metrics.to_dict()
    keys = (
        "total_return",
        "annualised_return",
        "annualised_vol",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "n_trades",
    )
    return {f"delta_{k}": float(g.get(k, 0.0) - b.get(k, 0.0)) for k in keys}


class RegimeGatedPolicy:
    """State-aware policy wrapper that masks actions using RegimeReport."""

    def __init__(
        self,
        base_policy: PolicyFn,
        df: pd.DataFrame,
        *,
        analyzer: RegimeIntelligence | None = None,
        gate_cfg: RegimeActionGateConfig | None = None,
        fallback_action: int = int(DiscreteAction.CLOSE),
        symbol: str = "",
    ) -> None:
        self.base_policy = base_policy
        self.df = df
        self.analyzer = analyzer or RegimeIntelligence()
        self.gate_cfg = gate_cfg or RegimeActionGateConfig()
        self.fallback_action = int(fallback_action)
        self.symbol = symbol
        self.reports: list[RegimeReport] = []
        self.plans: list[TradePlan] = []
        self.actions_raw: list[int] = []
        self.actions_final: list[int] = []
        self.masked_count = 0

    def _report(self, env: TradingEnv) -> RegimeReport:
        # Observation at env._t contains bars [env._t-window, env._t),
        # so use env._t - 1 to avoid leaking the execution bar.
        t = max(0, min(int(env._t) - 1, len(self.df) - 1))
        return self.analyzer.analyze(self.df, t=t, symbol=self.symbol)

    def _valid_fallback(self, mask: np.ndarray, current_position: float) -> int:
        preferred = [self.fallback_action, int(DiscreteAction.PARTIAL_CLOSE), int(DiscreteAction.SKIP)]
        if abs(float(current_position)) <= 1e-9:
            preferred = [int(DiscreteAction.SKIP), self.fallback_action]
        for action in preferred:
            if 0 <= action < mask.size and bool(mask[action]):
                return int(action)
        valid = np.flatnonzero(mask)
        return int(valid[0]) if valid.size else int(DiscreteAction.SKIP)

    def select_action(self, *, obs: dict, env: TradingEnv) -> int:
        report = self._report(env)
        current_position = float(env._position)
        plan = RegimeTradePlanner().plan(report, current_position=current_position, n_actions=env.action_space.n)
        mask = regime_action_mask(
            report,
            current_position=current_position,
            n_actions=env.action_space.n,
            cfg=self.gate_cfg,
        )
        logits_fn = getattr(self.base_policy, "logits", None)
        if callable(logits_fn):
            logits = logits_fn(obs)
            if not isinstance(logits, torch.Tensor):
                logits = torch.as_tensor(logits, dtype=torch.float32)
            masked = apply_regime_action_mask(
                logits,
                report,
                current_position=current_position,
                cfg=self.gate_cfg,
            )
            raw_action = int(torch.as_tensor(logits).reshape(-1, logits.shape[-1]).argmax(dim=-1)[0].item())
            action = int(masked.reshape(-1, masked.shape[-1]).argmax(dim=-1)[0].item())
        else:
            raw_action = int(self.base_policy(obs))
            action = raw_action if 0 <= raw_action < mask.size and bool(mask[raw_action]) else self._valid_fallback(mask, current_position)

        if action != raw_action:
            self.masked_count += 1
        self.reports.append(report)
        self.plans.append(plan)
        self.actions_raw.append(raw_action)
        self.actions_final.append(action)
        return action

    def summary(self) -> dict[str, Any]:
        regimes: dict[str, int] = {}
        risk_modes: dict[str, int] = {}
        transition_risks: list[float] = []
        tradeability: list[float] = []
        plan_status: dict[str, int] = {}
        recommended_playbooks: dict[str, int] = {}
        execution_order_types: dict[str, int] = {}
        execution_urgency: dict[str, int] = {}
        position_intents: dict[str, int] = {}
        reduce_only_count = 0
        no_market_count = 0
        for report in self.reports:
            regimes[report.primary_regime] = regimes.get(report.primary_regime, 0) + 1
            risk_modes[report.risk_mode] = risk_modes.get(report.risk_mode, 0) + 1
            transition_risks.append(float(report.transition_risk))
            tradeability.append(float(report.tradeability_score))
        for plan in self.plans:
            plan_status[plan.status] = plan_status.get(plan.status, 0) + 1
            recommended_playbooks[plan.recommended_playbook] = recommended_playbooks.get(plan.recommended_playbook, 0) + 1
            execution_order_types[plan.execution.order_type] = execution_order_types.get(plan.execution.order_type, 0) + 1
            execution_urgency[plan.execution.urgency] = execution_urgency.get(plan.execution.urgency, 0) + 1
            position_intents[plan.position_management.intent] = position_intents.get(plan.position_management.intent, 0) + 1
            reduce_only_count += int(plan.execution.reduce_only)
            no_market_count += int(not plan.execution.allow_market)
        return {
            "n_steps": len(self.reports),
            "n_masked_actions": int(self.masked_count),
            "masked_action_rate": float(self.masked_count / max(1, len(self.reports))),
            "primary_regimes": regimes,
            "risk_modes": risk_modes,
            "mean_transition_risk": float(np.mean(transition_risks)) if transition_risks else 0.0,
            "mean_tradeability": float(np.mean(tradeability)) if tradeability else 0.0,
            "plan_status": plan_status,
            "recommended_playbooks": recommended_playbooks,
            "execution_order_types": execution_order_types,
            "execution_urgency": execution_urgency,
            "position_intents": position_intents,
            "reduce_only_rate": float(reduce_only_count / max(1, len(self.plans))),
            "no_market_rate": float(no_market_count / max(1, len(self.plans))),
        }


def run_regime_ab_backtest(
    df: pd.DataFrame,
    base_policy: PolicyFn,
    *,
    env_cfg: Optional[EnvConfig] = None,
    cfg: Optional[RegimeABConfig] = None,
    seed: int = 0,
) -> RegimeABResult:
    """Run baseline and regime-gated variants on the same market path."""
    cfg = cfg or RegimeABConfig()
    env_cfg = env_cfg or EnvConfig(seed=seed)
    analyzer_cfg = _resolve_analyzer_config(cfg)
    baseline = run_backtest(df, base_policy, cfg=env_cfg, seed=seed)
    if cfg.adaptive_controller:
        gated_policy = RegimeAdaptiveController(
            base_policy,
            df,
            cfg=cfg.feedback,
            analyzer=RegimeIntelligence(analyzer_cfg),
        )
    else:
        gated_policy = RegimeGatedPolicy(
            base_policy,
            df,
            analyzer=RegimeIntelligence(analyzer_cfg),
            gate_cfg=cfg.gate,
            fallback_action=cfg.fallback_action,
            symbol=cfg.symbol,
        )
    gated = run_backtest(df, gated_policy, cfg=env_cfg, seed=seed)
    summary = gated_policy.summary()
    comparison = {
        "baseline": baseline.metrics.to_dict(),
        "gated": gated.metrics.to_dict(),
        "delta": _metric_delta(baseline, gated),
        "regime_summary": summary,
    }
    return RegimeABResult(
        baseline=RegimeVariantResult(cfg.baseline_name, baseline),
        gated=RegimeVariantResult(cfg.gated_name, gated, summary),
        comparison=comparison,
    )


def _resolve_analyzer_config(cfg: RegimeABConfig) -> RegimeIntelligenceConfig:
    if cfg.analyzer_profile:
        return build_regime_profile_config(
            cfg.analyzer_profile,
            benchmark_symbol=cfg.benchmark_symbol or None,
        )
    if cfg.symbol or cfg.asset_class or cfg.venue:
        profile = resolve_regime_profile(symbol=cfg.symbol, asset_class=cfg.asset_class, venue=cfg.venue)
        return profile.with_overrides(benchmark_symbol=cfg.benchmark_symbol or None)
    return cfg.analyzer


__all__ = [
    "RegimeABConfig",
    "RegimeABResult",
    "RegimeGatedPolicy",
    "RegimeVariantResult",
    "run_regime_ab_backtest",
]
