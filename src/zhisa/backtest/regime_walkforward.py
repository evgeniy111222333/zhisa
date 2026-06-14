"""Walk-forward A/B evaluation for regime-aware backtests."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional

import numpy as np
import pandas as pd

from zhisa.backtest.engine import PolicyFn
from zhisa.backtest.regime_ab import RegimeABConfig, RegimeABResult, run_regime_ab_backtest
from zhisa.backtest.splitter import Fold, SplitSpec, walk_forward_splits
from zhisa.env.trading_env import EnvConfig
from zhisa.regime.profiles import build_regime_profile_config


@dataclass(frozen=True)
class RegimeWalkForwardConfig:
    split: SplitSpec
    regime_ab: RegimeABConfig = field(default_factory=RegimeABConfig)
    min_test_bars: int = 64


@dataclass(frozen=True)
class RegimeWalkForwardResult:
    folds: list[Fold]
    fold_results: list[RegimeABResult]
    summary: dict[str, Any]


@dataclass(frozen=True)
class RegimeProfileSelectionConfig:
    split: SplitSpec
    profiles: tuple[str, ...] = ("default", "crypto_perp", "high_beta_alt")
    regime_ab: RegimeABConfig = field(default_factory=RegimeABConfig)
    min_test_bars: int = 64
    selection_metric: str = "delta_sharpe"
    higher_is_better: bool = True


@dataclass(frozen=True)
class RegimeProfileWalkForwardResult:
    profile_results: dict[str, RegimeWalkForwardResult]
    profile_scores: dict[str, float]
    best_profile: str
    summary: dict[str, Any]
    calibration_report: dict[str, Any] = field(default_factory=dict)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(np.std(values, ddof=0)) if values else 0.0


def _merge_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        nested = row.get(key, {})
        if not isinstance(nested, dict):
            continue
        for name, value in nested.items():
            out[str(name)] = out.get(str(name), 0) + int(value)
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def _rate_counts(counts: dict[str, int]) -> dict[str, float]:
    total = max(sum(counts.values()), 1)
    return {k: float(v / total) for k, v in counts.items()}


def _summarize(results: list[RegimeABResult]) -> dict[str, Any]:
    deltas = [r.comparison["delta"] for r in results]
    regime_summaries = [r.comparison["regime_summary"] for r in results]
    keys = sorted({k for d in deltas for k in d})
    mean_delta = {k: _mean([float(d.get(k, 0.0)) for d in deltas]) for k in keys}
    std_delta = {k: _std([float(d.get(k, 0.0)) for d in deltas]) for k in keys}
    gated_wins = {
        "sharpe": _mean([float(d.get("delta_sharpe", 0.0) > 0.0) for d in deltas]),
        "sortino": _mean([float(d.get("delta_sortino", 0.0) > 0.0) for d in deltas]),
        "drawdown": _mean([float(d.get("delta_max_drawdown", 0.0) < 0.0) for d in deltas]),
        "total_return": _mean([float(d.get("delta_total_return", 0.0) > 0.0) for d in deltas]),
    }
    primary_counts = _merge_counts(regime_summaries, "primary_regimes")
    risk_counts = _merge_counts(regime_summaries, "risk_modes")
    plan_counts = _merge_counts(regime_summaries, "plan_status")
    playbook_counts = _merge_counts(regime_summaries, "recommended_playbooks")
    execution_counts = _merge_counts(regime_summaries, "execution_order_types")
    urgency_counts = _merge_counts(regime_summaries, "execution_urgency")
    intent_counts = _merge_counts(regime_summaries, "position_intents")
    return {
        "n_folds": len(results),
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "gated_win_rate": gated_wins,
        "mean_masked_action_rate": _mean([float(s.get("masked_action_rate", 0.0)) for s in regime_summaries]),
        "mean_masked_actions": _mean([float(s.get("n_masked_actions", 0.0)) for s in regime_summaries]),
        "mean_transition_risk": _mean([float(s.get("mean_transition_risk", 0.0)) for s in regime_summaries]),
        "std_transition_risk": _std([float(s.get("mean_transition_risk", 0.0)) for s in regime_summaries]),
        "mean_tradeability": _mean([float(s.get("mean_tradeability", 0.0)) for s in regime_summaries]),
        "std_tradeability": _std([float(s.get("mean_tradeability", 0.0)) for s in regime_summaries]),
        "primary_regimes": primary_counts,
        "primary_regime_rate": _rate_counts(primary_counts),
        "risk_modes": risk_counts,
        "risk_mode_rate": _rate_counts(risk_counts),
        "plan_status": plan_counts,
        "plan_status_rate": _rate_counts(plan_counts),
        "recommended_playbooks": playbook_counts,
        "recommended_playbook_rate": _rate_counts(playbook_counts),
        "execution_order_types": execution_counts,
        "execution_order_type_rate": _rate_counts(execution_counts),
        "execution_urgency": urgency_counts,
        "execution_urgency_rate": _rate_counts(urgency_counts),
        "position_intents": intent_counts,
        "position_intent_rate": _rate_counts(intent_counts),
        "mean_reduce_only_rate": _mean([float(s.get("reduce_only_rate", 0.0)) for s in regime_summaries]),
        "mean_no_market_rate": _mean([float(s.get("no_market_rate", 0.0)) for s in regime_summaries]),
    }


def run_regime_walk_forward_ab(
    df: pd.DataFrame,
    base_policy: PolicyFn,
    *,
    cfg: RegimeWalkForwardConfig,
    env_cfg: Optional[EnvConfig] = None,
    seed: int = 0,
) -> RegimeWalkForwardResult:
    """Run regime A/B on each walk-forward test fold and aggregate results."""
    folds = walk_forward_splits(len(df), cfg.split)
    fold_results: list[RegimeABResult] = []
    env_cfg = env_cfg or EnvConfig(seed=seed)
    for i, fold in enumerate(folds):
        test_start, test_end = fold.test
        fold_df = df.iloc[test_start:test_end].copy()
        if len(fold_df) < max(cfg.min_test_bars, env_cfg.window + 2):
            continue
        fold_results.append(
            run_regime_ab_backtest(
                fold_df,
                base_policy,
                env_cfg=env_cfg,
                cfg=cfg.regime_ab,
                seed=seed + i,
            )
        )
    return RegimeWalkForwardResult(
        folds=folds[: len(fold_results)],
        fold_results=fold_results,
        summary=_summarize(fold_results),
    )


def _profile_score(result: RegimeWalkForwardResult, metric: str) -> float:
    mean_delta = result.summary.get("mean_delta", {})
    if isinstance(mean_delta, dict) and metric in mean_delta:
        return float(mean_delta[metric])
    value = result.summary.get(metric, 0.0)
    return float(value) if isinstance(value, (int, float, np.number)) else 0.0


def _top_count(counts: dict[str, int], rates: dict[str, float]) -> dict[str, Any]:
    if not counts:
        return {"name": "", "count": 0, "rate": 0.0}
    name, count = next(iter(counts.items()))
    return {"name": name, "count": int(count), "rate": float(rates.get(name, 0.0))}


def _profile_calibration_report(
    profile_results: dict[str, RegimeWalkForwardResult],
    profile_scores: dict[str, float],
    *,
    selection_metric: str,
    higher_is_better: bool,
    best_profile: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for profile, result in profile_results.items():
        summary = result.summary
        mean_delta = summary.get("mean_delta", {})
        std_delta = summary.get("std_delta", {})
        mean_delta = mean_delta if isinstance(mean_delta, dict) else {}
        std_delta = std_delta if isinstance(std_delta, dict) else {}
        primary = summary.get("primary_regimes", {})
        primary_rate = summary.get("primary_regime_rate", {})
        risk = summary.get("risk_modes", {})
        risk_rate = summary.get("risk_mode_rate", {})
        primary = primary if isinstance(primary, dict) else {}
        primary_rate = primary_rate if isinstance(primary_rate, dict) else {}
        risk = risk if isinstance(risk, dict) else {}
        risk_rate = risk_rate if isinstance(risk_rate, dict) else {}
        rows.append({
            "profile": profile,
            "score": float(profile_scores.get(profile, 0.0)),
            "n_folds": int(summary.get("n_folds", 0)),
            "mean_delta": {str(k): float(v) for k, v in mean_delta.items()},
            "std_delta": {str(k): float(v) for k, v in std_delta.items()},
            "masked_action_rate": float(summary.get("mean_masked_action_rate", 0.0)),
            "masked_actions": float(summary.get("mean_masked_actions", 0.0)),
            "mean_transition_risk": float(summary.get("mean_transition_risk", 0.0)),
            "mean_tradeability": float(summary.get("mean_tradeability", 0.0)),
            "reduce_only_rate": float(summary.get("mean_reduce_only_rate", 0.0)),
            "no_market_rate": float(summary.get("mean_no_market_rate", 0.0)),
            "dominant_primary_regime": _top_count(primary, primary_rate),
            "dominant_risk_mode": _top_count(risk, risk_rate),
            "primary_regime_rate": {str(k): float(v) for k, v in primary_rate.items()},
            "risk_mode_rate": {str(k): float(v) for k, v in risk_rate.items()},
        })

    rows.sort(key=lambda row: float(row["score"]), reverse=higher_is_better)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    best_row = next((row for row in rows if row["profile"] == best_profile), rows[0] if rows else {})
    default_row = next((row for row in rows if row["profile"] == "default"), None)
    runner_up = rows[1] if len(rows) > 1 else None
    score_gap = 0.0
    if runner_up is not None and best_row:
        score_gap = float(best_row["score"]) - float(runner_up["score"])

    default_comparison: dict[str, float] = {}
    if default_row is not None and best_row:
        default_comparison = {
            "score_delta": float(best_row["score"]) - float(default_row["score"]),
            "tradeability_delta": float(best_row["mean_tradeability"]) - float(default_row["mean_tradeability"]),
            "transition_risk_delta": float(best_row["mean_transition_risk"]) - float(default_row["mean_transition_risk"]),
            "masked_action_rate_delta": float(best_row["masked_action_rate"]) - float(default_row["masked_action_rate"]),
            "drawdown_delta": float(best_row["mean_delta"].get("delta_max_drawdown", 0.0))
            - float(default_row["mean_delta"].get("delta_max_drawdown", 0.0)),
        }

    reasons: list[str] = []
    if best_row:
        direction = "highest" if higher_is_better else "lowest"
        reasons.append(f"{best_profile} has the {direction} {selection_metric} score")
        if runner_up is not None:
            reasons.append(f"score gap to runner-up {runner_up['profile']} is {score_gap:.4f}")
        dominant_risk = best_row.get("dominant_risk_mode", {}).get("name", "")
        dominant_regime = best_row.get("dominant_primary_regime", {}).get("name", "")
        if dominant_regime:
            reasons.append(f"dominant regime is {dominant_regime}")
        if dominant_risk:
            reasons.append(f"dominant risk mode is {dominant_risk}")

    return {
        "selection_metric": selection_metric,
        "higher_is_better": higher_is_better,
        "best_profile": best_profile,
        "profile_rank": rows,
        "score_gap_to_runner_up": score_gap,
        "best_vs_default": default_comparison,
        "selection_reasons": reasons,
    }


def run_regime_profile_walk_forward_ab(
    df: pd.DataFrame,
    base_policy: PolicyFn,
    *,
    cfg: RegimeProfileSelectionConfig,
    env_cfg: Optional[EnvConfig] = None,
    seed: int = 0,
) -> RegimeProfileWalkForwardResult:
    """Evaluate several named regime profiles and select the best by walk-forward score."""
    profile_results: dict[str, RegimeWalkForwardResult] = {}
    profile_scores: dict[str, float] = {}
    for i, profile in enumerate(cfg.profiles):
        analyzer = build_regime_profile_config(profile)
        wf_cfg = RegimeWalkForwardConfig(
            split=cfg.split,
            regime_ab=replace(cfg.regime_ab, analyzer=analyzer, gated_name=f"regime_gated:{profile}"),
            min_test_bars=cfg.min_test_bars,
        )
        result = run_regime_walk_forward_ab(
            df,
            base_policy,
            cfg=wf_cfg,
            env_cfg=env_cfg,
            seed=seed + i * 1000,
        )
        profile_results[profile] = result
        profile_scores[profile] = _profile_score(result, cfg.selection_metric)

    if not profile_scores:
        return RegimeProfileWalkForwardResult(
            profile_results={},
            profile_scores={},
            best_profile="",
            summary={
                "selection_metric": cfg.selection_metric,
                "higher_is_better": cfg.higher_is_better,
                "n_profiles": 0,
                "calibration_report": {},
            },
            calibration_report={},
        )
    best_profile = (
        max(profile_scores, key=profile_scores.get)
        if cfg.higher_is_better
        else min(profile_scores, key=profile_scores.get)
    )
    calibration_report = _profile_calibration_report(
        profile_results,
        profile_scores,
        selection_metric=cfg.selection_metric,
        higher_is_better=cfg.higher_is_better,
        best_profile=str(best_profile),
    )
    summary = {
        "selection_metric": cfg.selection_metric,
        "higher_is_better": cfg.higher_is_better,
        "n_profiles": len(profile_results),
        "best_profile": str(best_profile),
        "profile_scores": profile_scores,
        "best_summary": profile_results[str(best_profile)].summary,
        "calibration_report": calibration_report,
    }
    return RegimeProfileWalkForwardResult(
        profile_results=profile_results,
        profile_scores=profile_scores,
        best_profile=str(best_profile),
        summary=summary,
        calibration_report=calibration_report,
    )


__all__ = [
    "RegimeProfileSelectionConfig",
    "RegimeProfileWalkForwardResult",
    "RegimeWalkForwardConfig",
    "RegimeWalkForwardResult",
    "run_regime_profile_walk_forward_ab",
    "run_regime_walk_forward_ab",
]
