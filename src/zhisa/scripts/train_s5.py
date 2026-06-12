"""Run the S5 online continual-learning loop on a synthetic stream.

Usage::

    python -m zhisa.scripts.train_s5 --config configs/s5_continual.yaml

The trainer generates a fresh market on every iteration (or consumes
an explicit stream if you pass ``--market-stream``), runs the inner
trainer for one epoch, samples a replay batch, and records the
episode reward into the drift detector. When drift is signalled the
EWC weight is bumped up so the next consolidation is strict.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional

import torch.nn as nn

from zhisa.config import load_config
from zhisa.data.dataset import MarketDataset, SampleSpec
from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.models.policy import build_default_policy
from zhisa.training.losses import LossWeights, MultiTaskLoss
from zhisa.training.optim import OptimConfig
from zhisa.training.s1_ssl import SSLConfig, SSLPretrainer
from zhisa.training.s2_supervised import SupervisedTrainer, TrainConfig
from zhisa.training.s4_rl import PPOConfig, PPOTrainer
from zhisa.training.s5_continual import ContinualConfig, OnlineContinualTrainer
from zhisa.utils.seeding import set_seed


def _default_device() -> str:
    """Resolve a sensible default device from env (GPU when available)."""
    import os
    import torch
    pref = os.environ.get("ZHISA_TEST_DEVICE", "auto").lower()
    if pref in {"cpu", "cuda"}:
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"




def _build_inner_factory(
    cfg, inner_kind: str, spec: Optional[SampleSpec] = None,
) -> Callable[[nn.Module], object]:
    """Return a factory that builds a fresh inner trainer per iteration.

    ``spec`` is required for ``s1`` (SSL) and ``s2`` (supervised) inner
    trainers so the factory can wrap a :class:`pandas.DataFrame` into a
    :class:`MarketDataset` before calling ``fit``.
    """
    bs = int(cfg.get("inner_batch_size", 8)) if cfg else 8
    lr = float(cfg.get("inner_lr", 3e-4)) if cfg else 3e-4
    epochs = int(cfg.get("inner_epochs", 1)) if cfg else 1

    if inner_kind == "s1":
        if spec is None:
            raise ValueError("S1 SSL inner trainer needs a SampleSpec")
        s1_spec = spec
        def factory(model):
            trainer = SSLPretrainer(model, SSLConfig(
                epochs=epochs, batch_size=bs, lr=lr, log_every=10_000,
                device="cpu",
            ))
            original_fit = trainer.fit

            def wrapped_fit(data):
                ds = data if isinstance(data, MarketDataset) \
                    else MarketDataset(data, spec=s1_spec)
                return original_fit(ds)

            trainer.fit = wrapped_fit  # type: ignore[assignment]
            return trainer
        return factory

    if inner_kind == "s2":
        s2_spec = spec
        def factory(model):
            return SupervisedTrainer(
                model, MultiTaskLoss(LossWeights()),
                TrainConfig(epochs=epochs, batch_size=bs, log_every=10_000,
                            device="cpu", optim=OptimConfig(lr=lr)),
            )
        if s2_spec is None:
            raise ValueError("S2 supervised inner trainer needs a SampleSpec")
        return factory

    if inner_kind == "s4":
        def factory(model):
            return PPOTrainer(model, PPOConfig(
                n_episodes=1, max_steps_per_episode=32,
                n_epochs=1, minibatch_size=4,
                device="cpu", optim=OptimConfig(lr=lr),
                log_every=0, seed=0,
            ))
        return factory

    print(f"Unknown inner trainer {inner_kind!r}; choose s1/s2/s4.",
          file=sys.stderr)
    raise SystemExit(2)


def _build_continual_cfg(cfg, args) -> ContinualConfig:
    """Resolve config + CLI overrides into a :class:`ContinualConfig`."""
    def opt(key, default):
        return cfg.get(key, default) if cfg else default
    return ContinualConfig(
        n_iterations=int(args.n_iterations if args.n_iterations is not None
                         else opt("n_iterations", 3)),
        bars_per_iter=int(opt("bars_per_iter", 400)),
        replay_capacity=int(opt("replay_capacity", 64)),
        replay_batch_size=int(opt("replay_batch_size", 8)),
        ewc_lambda=float(opt("ewc_lambda", 1.0)),
        ewc_lambda_on_drift=float(opt("ewc_lambda_on_drift", 10.0)),
        drift_threshold=float(opt("drift_threshold", 5.0)),
        drift_alpha=float(opt("drift_alpha", 0.01)),
        drift_warmup=int(opt("drift_warmup", 3)),
        inner_epochs=int(opt("inner_epochs", 1)),
        inner_batch_size=int(opt("inner_batch_size", 8)),
        inner_lr=float(opt("inner_lr", 3e-4)),
        seed=int(opt("seed", 0)),
        device=str(opt("device", _default_device())),
        checkpoint=str(args.checkpoint) if args.checkpoint else None,
        log_every=int(opt("log_every", 1)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train S5 online continual.")
    parser.add_argument("--config", type=str, default="configs/s5_continual.yaml")
    parser.add_argument("--n-iterations", type=int, default=None)
    parser.add_argument("--inner", type=str, default=None,
                        choices=("s1", "s2", "s4"),
                        help="Inner trainer kind (overrides YAML).")
    parser.add_argument("--checkpoint", type=str, default="artifacts/s5/policy.pt")
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path) if cfg_path.exists() else None
    seed = int(cfg.get("seed", 0)) if cfg else 0
    set_seed(seed)

    chart_window = int(cfg.get("chart_window", 16)) if cfg else 16
    image_size = int(cfg.get("image_size", 16)) if cfg else 16
    spec = SampleSpec(chart_window=chart_window, feature_window=chart_window,
                      image_size=image_size)

    # Probe feature dims with a tiny market.
    probe_df = generate_market(MarketConfig(n_bars=200, seed=seed))
    probe_ds = MarketDataset(probe_df, spec=spec)
    n_feat = probe_ds._features.shape[1]
    n_ctx = probe_ds._time_features.shape[1]

    model = build_default_policy(
        in_numeric_features=n_feat, in_context_features=n_ctx,
        window=spec.chart_window, image_size=spec.image_size,
        n_actions=9, n_regime_classes=spec.n_regime_states,
    )

    inner = args.inner or (str(cfg.get("inner_kind", "s1")) if cfg else "s1")
    factory = _build_inner_factory(cfg, inner, spec=spec)
    continual_cfg = _build_continual_cfg(cfg, args)

    trainer = OnlineContinualTrainer(model, continual_cfg, factory, spec=spec)
    result = trainer.fit()

    print("S5 online continual training complete.")
    print(result.as_frame().to_string(index=False))
    print(f"drift events: {result.total_drift_events}")
    print(f"final loss: {result.final_loss:.5f}")
    if continual_cfg.checkpoint:
        print(f"checkpoint saved to: {continual_cfg.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
