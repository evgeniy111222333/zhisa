"""Train the S1 self-supervised policy on a market dataset.

Produces a checkpoint at ``--checkpoint`` (default ``artifacts/s1/model.pt``)
that the S2 supervised trainer can resume from.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from zhisa.config import load_config
from zhisa.data.dataset import MarketDataset, SampleSpec
from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.models.policy import build_default_policy
from zhisa.training.s1_ssl import SSLPretrainer, SSLConfig
from zhisa.utils.seeding import set_seed


def _default_device() -> str:
    """Resolve a sensible default device from env (GPU when available)."""
    import os
    import torch
    pref = os.environ.get("ZHISA_TEST_DEVICE", "auto").lower()
    if pref in {"cpu", "cuda"}:
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"




def _ssl_config_from(cfg) -> SSLConfig:
    """Build an :class:`SSLConfig` from the merged YAML config."""
    s = (cfg.get("ssl", {}) or {}) if cfg is not None else {}
    return SSLConfig(
        projection_dim=int(s.get("projection_dim", 64)),
        hidden_dim=int(s.get("hidden_dim", 128)),
        temperature=float(s.get("temperature", 0.1)),
        mask_ratio=float(s.get("mask_ratio", 0.4)),
        ema_decay=float(s.get("ema_decay", 0.996)),
        weight_temporal=float(s.get("weight_temporal", 1.0)),
        weight_masked=float(s.get("weight_masked", 1.0)),
        weight_alignment=float(s.get("weight_alignment", 0.5)),
        grad_clip=float(s.get("grad_clip", 1.0)),
        log_every=int(s.get("log_every", 50)),
        use_ema_teacher=bool(s.get("use_ema_teacher", True)),
        use_masked_modeling=bool(s.get("use_masked_modeling", True)),
        use_temporal_contrast=bool(s.get("use_temporal_contrast", True)),
        use_cross_modal=bool(s.get("use_cross_modal", True)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train S1 self-supervised policy.")
    parser.add_argument("--config", type=str, default="configs/s1_ssl.yaml")
    parser.add_argument("--bars", type=int, default=8000)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="artifacts/s1/model.pt")
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path) if cfg_path.exists() else None

    set_seed(int(cfg.get("seed", 0)) if cfg else 0)

    # Data
    df = generate_market(MarketConfig(n_bars=args.bars))
    chart_window = int(cfg.get("chart_window", 32)) if cfg else 32
    image_size = int(cfg.get("image_size", 32)) if cfg else 32
    spec = SampleSpec(
        chart_window=chart_window,
        feature_window=chart_window,
        image_size=image_size,
    )
    ds = MarketDataset(df, spec=spec)

    # Model
    n_feat = ds._features.shape[1]
    model = build_default_policy(
        in_numeric_features=n_feat,
        in_context_features=ds._time_features.shape[1],
        window=spec.chart_window,
        image_size=spec.image_size,
        n_actions=9,
        n_regime_classes=spec.n_regime_states,
    )

    # SSL config
    ssl_cfg = _ssl_config_from(cfg)
    epochs = args.epochs or (int(cfg.get("epochs", 2)) if cfg else 2)
    bs = args.batch_size or (int(cfg.get("batch_size", 32)) if cfg else 32)
    device = args.device or (str(cfg.get("device", _default_device())) if cfg else _default_device())
    ssl_cfg.epochs = epochs
    ssl_cfg.batch_size = bs
    ssl_cfg.device = device
    ssl_cfg.checkpoint = args.checkpoint

    tr = SSLPretrainer(model, ssl_cfg)
    history = tr.fit(ds)
    final = history["history"][-1]
    print(
        "S1 training complete. "
        f"final epoch: total={final['total']:.4f} "
        f"temporal={final.get('temporal', 0.0):.4f} "
        f"masked={final.get('masked', 0.0):.4f} "
        f"alignment={final.get('alignment', 0.0):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
