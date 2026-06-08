"""Run v2 interpretability on a trained policy and dump a JSON report.

Usage::

    python -m zhisa.scripts.explain \\
        --checkpoint artifacts/s2/model.pt --config configs/s2_supervised.yaml \\
        --n-samples 5 --out artifacts/explain/report.json

The script:

1. Loads a :class:`PolicyNetwork` checkpoint (S1/S2/S2b/S4 output).
2. Generates (or loads) a synthetic market.
3. Pulls ``n_samples`` observations and runs
   :func:`per_dataset_summary` over them.
4. Writes a JSON-friendly report including the per-sample
   explanations and the dataset-level aggregate.

The action names use :class:`DiscreteAction` when available; for
models trained with custom action spaces the script falls back to
``"action_<idx>"`` labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from zhisa.config import load_config
from zhisa.data.dataset import MarketDataset, SampleSpec
from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.models.policy import PolicyNetwork, build_default_policy
from zhisa.training.interpretability import per_dataset_summary
from zhisa.utils.logging import get_logger
from zhisa.utils.seeding import set_seed


_LOG = get_logger(__name__)


def _default_device() -> str:
    import os
    import torch
    pref = os.environ.get("ZHISA_TEST_DEVICE", "auto").lower()
    if pref in {"cpu", "cuda"}:
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(checkpoint: str | None, cfg: dict[str, Any], device: str) -> PolicyNetwork:
    """Build a policy from a config dict and (optionally) load weights."""
    spec = SampleSpec(
        chart_window=int(cfg.get("chart_window", 32)),
        feature_window=int(cfg.get("chart_window", 32)),
        image_size=int(cfg.get("image_size", 32)),
        n_regime_states=int(cfg.get("n_regime_states", 4)),
    )
    df = generate_market(MarketConfig(n_bars=int(cfg.get("bars", 600))))
    ds = MarketDataset(df, spec=spec)
    n_feat = ds._features.shape[1] + ds._time_features.shape[1]
    n_ctx = ds._time_features.shape[1]
    model = build_default_policy(
        in_numeric_features=n_feat, in_context_features=n_ctx,
        window=spec.chart_window, image_size=spec.image_size,
        n_actions=int(cfg.get("n_actions", 9)),
        n_regime_classes=spec.n_regime_states,
    )
    if checkpoint and Path(checkpoint).exists():
        payload = __import__("torch").load(checkpoint, map_location=device, weights_only=False)
        if "model" in payload:
            try:
                model.load_state_dict(payload["model"], strict=False)
            except Exception as exc:
                _LOG.warning("could not load checkpoint strictly: %s", exc)
    model.eval()
    return model


def _jsonable(x: Any) -> Any:
    """Convert numpy/torch arrays to JSON-friendly Python objects."""
    import numpy as np
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    return str(x)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run v2 interpretability on a trained policy and dump JSON.",
    )
    p.add_argument("--config", type=str, default="configs/explain_default.yaml")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Optional path to a model checkpoint to load.")
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--n-steps", type=int, default=8,
                   help="Integrated-gradients Riemann steps (smaller = faster).")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default="artifacts/explain/report.json")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    import numpy as np

    args = _build_arg_parser().parse_args(argv)
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path).to_dict() if cfg_path.exists() else {}
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    device = args.device or str(cfg.get("device", _default_device()))

    model = _load_model(args.checkpoint, cfg, device)
    n_bars = int(cfg.get("bars", 600))
    spec = SampleSpec(
        chart_window=int(cfg.get("chart_window", 32)),
        feature_window=int(cfg.get("chart_window", 32)),
        image_size=int(cfg.get("image_size", 32)),
        n_regime_states=int(cfg.get("n_regime_states", 4)),
    )
    df = generate_market(MarketConfig(n_bars=n_bars, seed=seed))
    ds = MarketDataset(df, spec=spec)
    n = min(int(args.n_samples), len(ds))
    samples = []
    for i in range(n):
        s = ds[i]
        samples.append({
            "chart": s["chart"], "numeric": s["numeric"], "context": s["context"],
        })
    _LOG.info("running interpretability on n=%d samples (device=%s)", n, device)
    summary = per_dataset_summary(
        model, samples, n_samples=n, n_steps=int(args.n_steps),
        top_k_numeric=int(args.top_k), device=device,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, indent=2)
    _LOG.info("done. n=%d, top_actions=%s, report=%s",
              summary["n_samples"], summary["action_distribution"], str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
