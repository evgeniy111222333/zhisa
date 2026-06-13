"""Walk-forward A/B evaluation for regime-aware backtests."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from zhisa.backtest.engine import PolicyFn
from zhisa.backtest.regime_ab import RegimeABConfig, RegimeABResult, run_regime_ab_backtest
from zhisa.backtest.splitter import Fold, SplitSpec, walk_forward_splits
from zhisa.env.trading_env import EnvConfig


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


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _summarize(results: list[RegimeABResult]) -> dict[str, Any]:
    deltas = [r.comparison["delta"] for r in results]
    regime_summaries = [r.comparison["regime_summary"] for r in results]
    keys = sorted({k for d in deltas for k in d})
    mean_delta = {k: _mean([float(d.get(k, 0.0)) for d in deltas]) for k in keys}
    gated_wins = {
        "sharpe": _mean([float(d.get("delta_sharpe", 0.0) > 0.0) for d in deltas]),
        "sortino": _mean([float(d.get("delta_sortino", 0.0) > 0.0) for d in deltas]),
        "drawdown": _mean([float(d.get("delta_max_drawdown", 0.0) < 0.0) for d in deltas]),
        "total_return": _mean([float(d.get("delta_total_return", 0.0) > 0.0) for d in deltas]),
    }
    return {
        "n_folds": len(results),
        "mean_delta": mean_delta,
        "gated_win_rate": gated_wins,
        "mean_masked_action_rate": _mean([float(s.get("masked_action_rate", 0.0)) for s in regime_summaries]),
        "mean_masked_actions": _mean([float(s.get("n_masked_actions", 0.0)) for s in regime_summaries]),
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


__all__ = [
    "RegimeWalkForwardConfig",
    "RegimeWalkForwardResult",
    "run_regime_walk_forward_ab",
]
