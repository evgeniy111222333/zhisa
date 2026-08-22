"""Compare S2 checkpoints using stored full-validation trainer metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


PRIMARY_KEYS = (
    "total",
    "s2_composite_score",
    "s2_guarded_score",
    "direction_accuracy",
    "direction_balanced_accuracy",
    "direction_macro_f1",
    "direction_flat_recall",
    "direction_flat_f1",
    "direction_flat_pred_target_ratio",
    "direction_max_prediction_share",
    "direction_lift_vs_persistence_balanced",
    "return_corr",
    "value_corr",
    "volatility_corr",
    "risk_corr",
    "s2_worst_segment_direction_balanced_accuracy",
    "s2_worst_segment_flat_recall",
    "s2_worst_segment_flat_f1",
    "s2_worst_segment_volatility_corr",
    "s2_worst_segment_return_corr",
    "s2_worst_segment_persistence_lift",
    "s2_worst_segment_max_prediction_share",
    "s2_worst_segment_flat_prediction_share",
    "s2_worst_segment_flat_pred_target_ratio",
)


def _latest_val(payload: dict[str, Any]) -> dict[str, Any]:
    history = payload.get("trainer_state", {}).get("history", [])
    if not history:
        return {}
    return dict(history[-1].get("val") or {})


def _checkpoint_summary(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = dict(payload.get("trainer_state", {}) or {})
    val = _latest_val(payload)
    segment_metrics = val.get("segment_metrics", {}) or {}
    weak_segments = []
    for name, metrics in segment_metrics.items():
        weak_segments.append({
            "name": name,
            "direction_balanced_accuracy": metrics.get("direction_balanced_accuracy"),
            "direction_flat_recall": metrics.get("direction_flat_recall"),
            "direction_flat_f1": metrics.get("direction_flat_f1"),
            "direction_lift_vs_persistence_balanced": metrics.get("direction_lift_vs_persistence_balanced"),
            "return_corr": metrics.get("return_corr"),
            "volatility_corr": metrics.get("volatility_corr"),
            "risk_corr": metrics.get("risk_corr"),
            "direction_flat_pred_target_ratio": metrics.get("direction_flat_pred_target_ratio"),
            "direction_max_prediction_share": metrics.get("direction_max_prediction_share"),
            "s2_composite_score": metrics.get("s2_composite_score"),
        })
    weak_segments.sort(key=lambda item: (
        float(item.get("direction_balanced_accuracy") or 0.0),
        float(item.get("direction_flat_f1") or 0.0),
        float(item.get("return_corr") or -999.0),
    ))
    return {
        "path": str(path),
        "trainer_state": {
            "completed_epochs": state.get("completed_epochs"),
            "step": state.get("step"),
            "best_val_metric": state.get("best_val_metric"),
            "best_val_total": state.get("best_val_total"),
            "early_stopping_bad_epochs": state.get("early_stopping_bad_epochs"),
            "history_len": len(state.get("history", []) or []),
            "latest_history_epoch": (state.get("history", []) or [{}])[-1].get("epoch"),
        },
        "metrics": {key: val.get(key) for key in PRIMARY_KEYS},
        "segments": len(segment_metrics),
        "weakest_segments": weak_segments[:12],
    }


def _delta(a: Any, b: Any) -> float | None:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(b) - float(a)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best", required=True)
    parser.add_argument("--last", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    best = _checkpoint_summary(Path(args.best))
    last = _checkpoint_summary(Path(args.last))
    deltas = {
        key: _delta(best["metrics"].get(key), last["metrics"].get(key))
        for key in PRIMARY_KEYS
    }
    guarded_best = best["metrics"].get("s2_guarded_score")
    guarded_last = last["metrics"].get("s2_guarded_score")
    if isinstance(guarded_best, (int, float)) and isinstance(guarded_last, (int, float)):
        preferred = "last" if guarded_last > guarded_best else "best"
    else:
        preferred = "unknown"
    report = {
        "preferred_by_guarded_score": preferred,
        "best": best,
        "last": last,
        "last_minus_best": deltas,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
