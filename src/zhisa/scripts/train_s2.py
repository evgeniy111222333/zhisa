"""Train the S2 supervised multi-task policy on a market dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from zhisa.config import load_config
from zhisa.data.dataset import MarketDataset, SampleSpec
from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.models.policy import build_default_policy
from zhisa.training.losses import LossWeights, MultiTaskLoss
from zhisa.training.optim import OptimConfig
from zhisa.training.s2_supervised import SupervisedTrainer, TrainConfig
from zhisa.utils.seeding import set_seed


def _default_device() -> str:
    """Resolve a sensible default device from env (GPU when available)."""
    import os
    import torch
    pref = os.environ.get("ZHISA_TEST_DEVICE", "auto").lower()
    if pref in {"cpu", "cuda"}:
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"




def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train S2 supervised policy.")
    parser.add_argument("--config", type=str, default="configs/s2_supervised.yaml")
    parser.add_argument("--bars", type=int, default=8000)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="artifacts/s2/model.pt")
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path) if cfg_path.exists() else None

    set_seed(int(cfg.get("seed", 0)) if cfg else 0)

    # Data
    synth_cfg = MarketConfig(n_bars=args.bars)
    df = generate_market(synth_cfg)
    spec = SampleSpec(
        chart_window=int(cfg.get("chart_window", 32)) if cfg else 32,
        feature_window=int(cfg.get("chart_window", 32)) if cfg else 32,
        image_size=int(cfg.get("image_size", 32)) if cfg else 32,
    )
    ds = MarketDataset(df, spec=spec)

    # Model
    n_feat = ds._features.shape[1] + ds._time_features.shape[1]
    model = build_default_policy(
        in_numeric_features=n_feat,
        in_context_features=ds._time_features.shape[1],
        window=spec.chart_window,
        image_size=spec.image_size,
        n_actions=9,
        n_regime_classes=spec.n_regime_states,
    )

    # Training config
    epochs = args.epochs or (int(cfg.get("epochs", 2)) if cfg else 2)
    bs = args.batch_size or (int(cfg.get("batch_size", 32)) if cfg else 32)
    device = args.device or (str(cfg.get("device", _default_device())) if cfg else _default_device())
    optim_cfg = OptimConfig(
        lr=float((cfg.get("optim", {}) or {}).get("lr", 3e-4)) if cfg else 3e-4,
    )
    loss = MultiTaskLoss(LossWeights())
    trainer = SupervisedTrainer(
        model, loss, TrainConfig(epochs=epochs, batch_size=bs, device=device, optim=optim_cfg,
                                 checkpoint=args.checkpoint),
    )
    history = trainer.fit(ds)
    print("Training complete. Final loss:", history["history"][-1]["loss"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
