"""Train a model through the S3 synthetic curriculum.

By default this runs the S1 (SSL) inner trainer across each stage, but
you can switch to S2 (supervised multi-task) via ``--inner s2``. The
final checkpoint is saved to ``--checkpoint``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from zhisa.config import load_config
from zhisa.data.dataset import MarketDataset, SampleSpec
from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.models.policy import build_default_policy
from zhisa.training.losses import LossWeights, MultiTaskLoss
from zhisa.training.optim import OptimConfig
from zhisa.training.s1_ssl import SSLPretrainer, SSLConfig
from zhisa.training.s2_supervised import SupervisedTrainer, TrainConfig
from zhisa.training.s3_curriculum import (
    CurriculumStage,
    CurriculumTrainer,
)
from zhisa.utils.seeding import set_seed


def _default_device() -> str:
    """Resolve a sensible default device from env (GPU when available)."""
    import os
    import torch
    pref = os.environ.get("ZHISA_TEST_DEVICE", "auto").lower()
    if pref in {"cpu", "cuda"}:
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"




def _make_inner_factory(cfg, inner_kind: str):
    """Build a factory that returns a fresh S1 or S2 trainer for the model."""
    bs = int(cfg.get("inner_batch_size", 32)) if cfg else 32
    lr = float(cfg.get("inner_lr", 3e-4)) if cfg else 3e-4
    log_every = int(cfg.get("inner_log_every", 50)) if cfg else 50
    dev = _default_device()

    if inner_kind == "s1":
        def factory(model):
            return SSLPretrainer(model, SSLConfig(
                epochs=1, batch_size=bs, lr=lr, log_every=log_every, device=dev,
            ))
        return factory

    if inner_kind == "s2":
        def factory(model):
            loss = MultiTaskLoss(LossWeights())
            return SupervisedTrainer(model, loss, TrainConfig(
                epochs=1, batch_size=bs, log_every=log_every, device=dev,
                optim=OptimConfig(lr=lr),
            ))
        return factory

    print(f"Unknown inner trainer {inner_kind!r}; choose 's1' or 's2'.",
          file=sys.stderr)
    raise SystemExit(2)


def _build_stages(cfg) -> list[CurriculumStage]:
    """Convert the ``stages`` list from YAML into :class:`CurriculumStage`s."""
    raw = (cfg.get("stages", []) if cfg else []) or []
    out: list[CurriculumStage] = []
    for entry in raw:
        out.append(CurriculumStage(
            name=str(entry["name"]),
            n_bars=int(entry.get("n_bars", 2000)),
            base_vol=float(entry.get("base_vol", 0.5)),
            shock_prob=float(entry.get("shock_prob", 0.0)),
            student_t_df=float(entry.get("student_t_df", 8.0)),
            epochs=int(entry.get("epochs", 1)),
            mix_with_previous=float(entry.get("mix_with_previous", 0.0)),
        ))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train S3 curriculum.")
    parser.add_argument("--config", type=str, default="configs/s3_curriculum.yaml")
    parser.add_argument("--inner", type=str, default=None,
                        choices=("s1", "s2"),
                        help="Inner trainer kind (overrides config).")
    parser.add_argument("--checkpoint", type=str, default="artifacts/s3/model.pt")
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path) if cfg_path.exists() else None

    set_seed(int(cfg.get("seed", 0)) if cfg else 0)

    # Build sample spec to derive feature dimensionality.
    chart_window = int(cfg.get("chart_window", 32)) if cfg else 32
    image_size = int(cfg.get("image_size", 32)) if cfg else 32
    spec = SampleSpec(chart_window=chart_window, feature_window=chart_window,
                      image_size=image_size)

    # Probe to know the model's required input dim.
    probe_df = generate_market(MarketConfig(n_bars=300, seed=0))
    probe_ds = MarketDataset(probe_df, spec=spec)
    n_feat = probe_ds._features.shape[1] + probe_ds._time_features.shape[1]
    n_ctx = probe_ds._time_features.shape[1]

    model = build_default_policy(
        in_numeric_features=n_feat, in_context_features=n_ctx,
        window=spec.chart_window, image_size=spec.image_size,
        n_actions=9, n_regime_classes=spec.n_regime_states,
    )

    inner = args.inner or (str(cfg.get("inner", "s1")) if cfg else "s1")
    factory = _make_inner_factory(cfg, inner)
    stages = _build_stages(cfg) or None

    ct = CurriculumTrainer(
        factory, model, stages=stages,
        sample_spec=spec,
        base_seed=int(cfg.get("seed", 0)) if cfg else 0,
        checkpoint=args.checkpoint,
    )
    result = ct.fit()
    print("S3 training complete.")
    print(result.as_frame().to_string(index=False))
    print(f"Final loss: {result.final_loss:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
