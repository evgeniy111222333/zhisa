"""Train the S1 self-supervised policy on a market dataset.

Produces a checkpoint at ``--checkpoint`` (default ``artifacts/s1/model.pt``)
that the S2 supervised trainer can resume from.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset

from zhisa.config import load_config
from zhisa.data.dataset import MacroContextConfig, MarketDataset, MarketTargetConfig, SampleSpec
from zhisa.data.labeling import TripleBarrierConfig
from zhisa.data.preparation import load_prepared_split
from zhisa.models.policy import build_default_policy
from zhisa.scripts._real_data import add_market_data_args, load_market_dataframe
from zhisa.training.s1_ssl import SSLPretrainer, SSLConfig
from zhisa.utils.seeding import set_seed
from zhisa.storage.schema import Timeframe


def _default_device() -> str:
    """Resolve a sensible default device from env (GPU when available)."""
    pref = os.environ.get("ZHISA_TEST_DEVICE", "auto").lower()
    if pref in {"cpu", "cuda"}:
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"




def _ssl_config_from(cfg) -> SSLConfig:
    """Build an :class:`SSLConfig` from the merged YAML config."""
    s = (cfg.get("ssl", {}) or {}) if cfg is not None else {}
    optim = (cfg.get("optim", {}) or {}) if cfg is not None else {}
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
        lr=float(optim.get("lr", 3e-4)),
        weight_decay=float(optim.get("weight_decay", 1e-4)),
        warmup_steps=int(optim.get("warmup_steps", s.get("warmup_steps", 100))),
        temporal_horizon=int(s.get("temporal_horizon", 1)),
        val_max_batches=int(s.get("val_max_batches", 32)),
        checkpoint_every_steps=int(s.get("checkpoint_every_steps", 500)),
        use_ema_teacher=bool(s.get("use_ema_teacher", True)),
        use_masked_modeling=bool(s.get("use_masked_modeling", True)),
        use_temporal_contrast=bool(s.get("use_temporal_contrast", True)),
        use_cross_modal=bool(s.get("use_cross_modal", True)),
    )


def _market_datasets_from_frame(
    frame: pd.DataFrame,
    *,
    spec: SampleSpec,
    cache_charts: bool,
    chart_cache_size: int,
    max_bars_per_symbol: int | None = None,
    timeframe: str | None = None,
    compute_targets: bool = False,
    target_cfg: MarketTargetConfig | None = None,
    triple_barrier_cfg: TripleBarrierConfig | None = None,
    macro_cfg: MacroContextConfig | None = None,
    macro_frames_by_symbol: dict[str, pd.DataFrame] | None = None,
) -> list[MarketDataset]:
    """Build datasets per symbol and contiguous time segment."""
    if "symbol" not in frame.columns:
        raise ValueError("prepared split must contain a 'symbol' column")
    datasets: list[MarketDataset] = []
    feature_dims: set[tuple[int, int]] = set()
    expected_delta = (
        pd.Timedelta(minutes=Timeframe.from_str(timeframe).minutes)
        if timeframe
        else None
    )
    for symbol, symbol_frame in frame.groupby("symbol", sort=True):
        market = symbol_frame.drop(columns=["symbol"]).sort_index()
        macro_frame = None
        if macro_frames_by_symbol is not None:
            if symbol not in macro_frames_by_symbol:
                raise ValueError(f"missing macro prepared data for symbol {symbol!r}")
            macro_frame = macro_frames_by_symbol[symbol]
        if max_bars_per_symbol is not None:
            market = market.iloc[:max_bars_per_symbol]
        if expected_delta is None:
            segments = [(0, market)]
        else:
            segment_ids = market.index.to_series().diff().ne(expected_delta).cumsum()
            segments = list(market.groupby(segment_ids, sort=False))
        min_segment_bars = spec.chart_window + max(spec.horizons, default=0) + 2
        for segment_id, segment in segments:
            if len(segment) < min_segment_bars:
                continue
            segment = segment.copy()
            segment.name = f"{symbol}#segment-{segment_id}"
            ds = MarketDataset(
                segment,
                spec=spec,
                triple_barrier_cfg=triple_barrier_cfg,
                target_cfg=target_cfg,
                cache_charts=cache_charts,
                chart_cache_size=chart_cache_size,
                compute_targets=compute_targets,
                macro_cfg=macro_cfg,
                macro_df=macro_frame,
            )
            feature_dims.add(
                (ds._features_df.shape[1], ds._time_features_df.shape[1])
            )
            datasets.append(ds)
    if not datasets:
        raise ValueError("prepared split contains no usable symbols")
    if len(feature_dims) != 1:
        raise ValueError(f"inconsistent prepared feature dimensions: {feature_dims}")
    return datasets


def _concat(datasets: list[Dataset]) -> Dataset:
    return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train S1 self-supervised policy.")
    parser.add_argument("--config", type=str, default="configs/s1_ssl.yaml")
    parser.add_argument("--bars", type=int, default=8000)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="artifacts/s1/model.pt")
    parser.add_argument("--best-checkpoint", type=str, default=None)
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Explicit S1 checkpoint to resume or warm-start from",
    )
    parser.add_argument(
        "--reset-best-on-resume",
        action="store_true",
        help="Reset best validation score when starting a new data phase",
    )
    parser.add_argument(
        "--prepared-root",
        type=str,
        default=None,
        help="Prepared dataset root containing splits/train.parquet and val.parquet",
    )
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument(
        "--prepared-max-bars-per-symbol",
        type=int,
        default=None,
        help="Debug/smoke limit applied independently to every symbol",
    )
    parser.add_argument("--fast-render", action="store_true", help="Use pure-numpy renderer")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader num_workers")
    parser.add_argument("--cache-charts", action="store_true", help="Cache rendered charts in RAM")
    parser.add_argument(
        "--chart-cache-size",
        type=int,
        default=-1,
        help="Lazy chart LRU size; -1 disables it (recommended for large S1 data)",
    )
    add_market_data_args(parser)
    args = parser.parse_args(argv)

    if args.fast_render:
        os.environ["ZHISA_FAST_RENDER"] = "1"
    if args.workers is not None:
        os.environ["ZHISA_SSL_WORKERS"] = str(args.workers)

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path) if cfg_path.exists() else None

    seed = int(cfg.get("seed", 0)) if cfg else 0
    set_seed(seed)

    # Data
    chart_window = int(cfg.get("chart_window", 32)) if cfg else 32
    image_size = int(cfg.get("image_size", 32)) if cfg else 32
    spec = SampleSpec(
        chart_window=chart_window,
        feature_window=chart_window,
        image_size=image_size,
    )

    val_ds: Dataset | None = None
    manifest: dict | None = None
    if args.prepared_root:
        prepared_root = Path(args.prepared_root)
        manifest_path = prepared_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"prepared manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        prepared_timeframe = str(manifest["timeframe"])
        train_frame = load_prepared_split(prepared_root, args.train_split)
        train_rows = len(train_frame)
        train_markets = int(train_frame["symbol"].nunique())
        datasets = _market_datasets_from_frame(
            train_frame,
            spec=spec,
            cache_charts=args.cache_charts,
            chart_cache_size=args.chart_cache_size,
            max_bars_per_symbol=args.prepared_max_bars_per_symbol,
            timeframe=prepared_timeframe,
        )
        del train_frame
        if not args.no_validation:
            val_frame = load_prepared_split(prepared_root, args.val_split)
            val_datasets = _market_datasets_from_frame(
                val_frame,
                spec=spec,
                cache_charts=args.cache_charts,
                chart_cache_size=args.chart_cache_size,
                max_bars_per_symbol=args.prepared_max_bars_per_symbol,
                timeframe=prepared_timeframe,
            )
            del val_frame
            val_ds = _concat(val_datasets)
        print(
            f"Prepared S1 data: train={train_rows:,} rows, "
            f"markets={train_markets}, segments={len(datasets)}, "
            f"validation={'on' if val_ds is not None else 'off'}"
        )
    else:
        symbols = str(getattr(args, "symbol", "BTC/USDT")).split(",")
        datasets = []
        for sym in symbols:
            sym_args = copy.copy(args)
            sym_args.symbol = sym.strip()
            print(f"Loading data for {sym_args.symbol}...")
            try:
                df = load_market_dataframe(sym_args, seed=seed, default_bars=args.bars)
                datasets.append(
                    MarketDataset(
                        df,
                        spec=spec,
                        cache_charts=args.cache_charts,
                        chart_cache_size=args.chart_cache_size,
                        compute_targets=False,
                    )
                )
            except Exception as exc:
                print(f"Skipping {sym_args.symbol}: {exc}")
        if not datasets:
            raise ValueError("No valid datasets loaded. Check your data source.")

    ds = _concat(datasets)
    first_ds = datasets[0]

    # Model
    n_feat = first_ds._features_df.shape[1]
    model = build_default_policy(
        in_numeric_features=n_feat,
        in_context_features=first_ds._time_features_df.shape[1],
        window=spec.chart_window,
        image_size=spec.image_size,
        n_actions=9,
        n_regime_classes=spec.n_regime_states,
    )

    # SSL config
    ssl_cfg = _ssl_config_from(cfg)
    epochs = args.epochs if args.epochs is not None else (int(cfg.get("epochs", 2)) if cfg else 2)
    bs = args.batch_size if args.batch_size is not None else (int(cfg.get("batch_size", 32)) if cfg else 32)
    device = args.device or (str(cfg.get("device", _default_device())) if cfg else _default_device())
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    ssl_cfg.epochs = epochs
    ssl_cfg.batch_size = bs
    ssl_cfg.device = device
    ssl_cfg.checkpoint = args.checkpoint
    if manifest is not None:
        ssl_cfg.dataset_root = str(Path(args.prepared_root).resolve())
        ssl_cfg.dataset_timeframe = str(manifest["timeframe"])
        ssl_cfg.dataset_manifest_checksum = str(manifest["output_checksum"])
    if val_ds is not None:
        checkpoint = Path(args.checkpoint)
        ssl_cfg.best_checkpoint = args.best_checkpoint or str(
            checkpoint.with_name(f"{checkpoint.stem}_best{checkpoint.suffix}")
        )

    tr = SSLPretrainer(model, ssl_cfg)

    if args.resume_from:
        if not Path(args.resume_from).is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {args.resume_from}")
        status = tr.load(args.resume_from)
        if args.reset_best_on_resume:
            tr._best_val_total = float("inf")
        mode = status["resume_mode"].replace("_", " ")
        print(f"Loaded {args.resume_from} ({mode}): {status}")

    history = tr.fit(ds, val_ds=val_ds)
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
