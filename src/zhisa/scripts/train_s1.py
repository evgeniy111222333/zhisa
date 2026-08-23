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
from zhisa.data.render_job import materialize_parallel
from zhisa.data.render_contract import (
    enforce_parent_render_contract,
    load_checkpoint,
    resolve_render_contract,
)
from zhisa.features.channel_dropout import ChannelDropoutSpec
from zhisa.features.normalization import NormalizationSpec
from zhisa.models.policy import build_default_policy
from zhisa.rendering.spec import RenderSpec
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
        temporal_bank_size=int(s.get("temporal_bank_size", 0)),
        temporal_bank_warmup=int(s.get("temporal_bank_warmup", 128)),
        temporal_hard_offsets=tuple(
            int(x) for x in (s.get("temporal_hard_offsets", []) or [])
        ),
        lr_schedule=str(s.get("lr_schedule", "constant")),
        cosine_min_scale=float(s.get("cosine_min_scale", 0.003)),
        total_steps=int(s.get("total_steps", 0)),
        weight_trunk_align=float(s.get("weight_trunk_align", 0.0)),
        trunk_align_momentum=float(s.get("trunk_align_momentum", 0.0)),
        instrument_contrast_w=float(s.get("instrument_contrast_w", 0.0)),
        recon_depth=int(s.get("recon_depth", 1)),
        recon_use_norm=bool(s.get("recon_use_norm", True)),
        recon_use_gain=bool(s.get("recon_use_gain", False)),
        masked_target_norm=bool(s.get("masked_target_norm", False)),
        vision_grad_scale=float(s.get("vision_grad_scale", 1.0)),
        instrument_z_contrast_w=float(s.get("instrument_z_contrast_w", 0.0)),
        augment_transforms=tuple(
            str(x) for x in (s.get("augment_transforms", []) or [])
        ),
        augment_strength=float(s.get("augment_strength", 0.05)),
        augment_crop_frac=float(s.get("augment_crop_frac", 0.85)),
        augment_noise_std=float(s.get("augment_noise_std", 0.01)),
    )



def _policy_kwargs_from(cfg, *, n_instruments: int, spec) -> dict:
    """Merge optional YAML `model:` block into build_default_policy kwargs."""
    m = (cfg.get("model", {}) or {}) if cfg else {}
    kwargs = dict(
        window=spec.chart_window,
        image_size=spec.image_size,
        n_actions=9,
        n_regime_classes=spec.n_regime_states,
        n_instruments=int(n_instruments),
    )
    for key, value in m.items():
        if key in ("in_numeric_features", "in_context_features"):
            continue
        if key == "vision_channels":
            value = tuple(int(x) for x in value)
        kwargs[key] = value
    return kwargs


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
    charts_cache_dir: str | None = None,
    render_spec: RenderSpec | None = None,
    render_workers: int = 0,
    render_chunk: int = 5_000,
    normalization: NormalizationSpec | None = None,
    render_engine: str = "cpu",
    instruments: list[str] | None = None,
    channel_dropout: Optional[ChannelDropoutSpec] = None,
) -> list[MarketDataset]:
    """Build datasets per symbol and contiguous time segment.

    When ``charts_cache_dir`` is provided the ideal compute-free path is used:
    charts are compiled in advance into a :class:`CompiledChartStore` (memmap,
    content-addressed, reused across runs) and handed to the dataset, so the
    training DataLoader performs zero rasterisation.
    """
    if "symbol" not in frame.columns:
        raise ValueError("prepared split must contain a 'symbol' column")
    datasets: list[MarketDataset] = []
    render_metas: list[dict] = []
    feature_dims: set[tuple[int, int]] = set()
    expected_delta = (
        pd.Timedelta(minutes=Timeframe.from_str(timeframe).minutes)
        if timeframe
        else None
    )
    if render_spec is None:
        render_spec = RenderSpec(size=spec.image_size)
    instrument_id_by_symbol: dict = {}
    if instruments:
        instrument_id_by_symbol = {sym: i for i, sym in enumerate(instruments)}
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
        # Lineage reuse guard (cheap, once at the first symbol): probe this
        # root's segments against the existing chart store BEFORE any render
        # work. Screams when a ~zero-reuse full render is about to start
        # (ZHISA_FORCE_RENDER=1 overrides; ZHISA_LINEAGE_GUARD=0 disables).
        if (
            charts_cache_dir
            and os.environ.get("ZHISA_LINEAGE_GUARD", "1") != "0"
            and symbol == next(iter(frame.groupby("symbol", sort=True)))[0]
        ):
            from zhisa.data.lineage import guard_reuse
            seg_df = [s for _, s in segments if len(s) >= min_segment_bars][:3]
            guard_reuse(
                Path(charts_cache_dir), seg_df, spec.chart_window,
                render_spec or RenderSpec(size=spec.image_size),
                min_reuse_ratio=0.34,
                render_hint=f"full render for {symbol} (n={len(market)})",
                trim=spec.chart_window + max(spec.horizons, default=0) + 1,
            )
        for segment_id, segment in segments:
            if len(segment) < min_segment_bars:
                continue
            segment = segment.copy()
            segment.name = f"{symbol}#segment-{segment_id}"
            chart_source = None
            if charts_cache_dir:
                seg_len = max(0, len(segment) - spec.chart_window - max(spec.horizons, default=0) - 1)
                store, _ = materialize_parallel(
                    segment,
                    window=spec.chart_window,
                    spec=render_spec,
                    n=seg_len,
                    out_root=charts_cache_dir,
                    workers=render_workers,
                    chunk_size=render_chunk,
                    engine=render_engine,
                )
                chart_source = store
                render_metas.append(store.render_meta)
            _instr_id = instrument_id_by_symbol.get(symbol, 0)
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
                chart_source=chart_source,
                normalization=normalization,
                instrument_id=_instr_id,
                channel_dropout=channel_dropout,
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
    parser.add_argument(
        "--charts-cache-dir",
        type=str,
        default=None,
        help="Compile charts into a content-addressed CompiledChartStore (memmap) "
        "under this dir and serve the trainer from disk (zero in-loop rendering). "
        "Built once per dataset identity, reuses prefixes incrementally on data "
        "growth, and is reused on later runs.",
    )
    parser.add_argument(
        "--render-workers",
        type=int,
        default=0,
        help="Parallel chart-compile workers (> 0 enables multiprocessing; 0 = serial)",
    )
    parser.add_argument(
        "--render-chunk",
        type=int,
        default=5_000,
        help="Rows per worker task during compiled chart materialisation",
    )
    parser.add_argument("--normalize-mode", type=str, default=None,
                        choices=["rolling_z", "robust_z"],
                        help="Feature normalization mode (default: rolling_z)")
    parser.add_argument("--normalize-lookback", type=int, default=None)
    parser.add_argument("--render-engine", type=str, default="cpu",
                        choices=["cpu", "gpu"],
                        help="Chart compile engine (gpu requires CUDA; parity-gated)")
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
    def _norm_spec_from(args, cfg):
        _nm = (cfg.get("normalize", {}) or {}) if cfg else {}
        return NormalizationSpec(
            mode=str(args.normalize_mode or _nm.get("mode", "rolling_z")),
            lookback=int(args.normalize_lookback or _nm.get("lookback", 256)),
        )

    norm_spec = _norm_spec_from(args, cfg)

    # Optional numeric-stream Channel Dropout (train-only, keyed/seed-
    # deterministic): mirrors the chart-side KeyedAugmentor. Enabled by a
    # ``features.channel_dropout`` block in the YAML config; off by default.
    _cd_block = ((cfg.get("features", {}) or {}) if cfg else {}).get("channel_dropout") or {}
    channel_dropout = (
        ChannelDropoutSpec(
            p=float(_cd_block.get("p", 0.15)),
            max_channels=int(_cd_block.get("max_channels", 3)),
            pair_bucket=int(_cd_block.get("pair_bucket", 16)),
        )
        if _cd_block
        else None
    )

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
        train_symbols = sorted(train_frame["symbol"].unique())
        datasets = _market_datasets_from_frame(
            train_frame,
            spec=spec,
            cache_charts=args.cache_charts,
            chart_cache_size=args.chart_cache_size,
            max_bars_per_symbol=args.prepared_max_bars_per_symbol,
            timeframe=prepared_timeframe,
            charts_cache_dir=args.charts_cache_dir,
            render_spec=RenderSpec(size=spec.image_size),
            render_workers=args.render_workers,
            render_chunk=args.render_chunk,
            render_engine=args.render_engine,
            normalization=norm_spec,
            channel_dropout=channel_dropout,
            instruments=train_symbols,
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
                charts_cache_dir=args.charts_cache_dir,
                render_spec=RenderSpec(size=spec.image_size),
                render_workers=args.render_workers,
                render_chunk=args.render_chunk,
                render_engine=args.render_engine,
                normalization=norm_spec,
                channel_dropout=channel_dropout,
                instruments=train_symbols,
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

    _nm = (cfg.get("normalize", {}) or {}) if cfg else {}
    norm_spec = NormalizationSpec(
        mode=str(args.normalize_mode or _nm.get("mode", "rolling_z")),
        lookback=int(args.normalize_lookback or _nm.get("lookback", 256)),
    )

    ds = _concat(datasets)
    first_ds = datasets[0]

    # Model
    n_feat = first_ds._features_df.shape[1]
    n_instruments = len(train_symbols) if args.prepared_root else 1
    model = build_default_policy(
        in_numeric_features=n_feat,
        in_context_features=first_ds._time_features_df.shape[1],
        **_policy_kwargs_from(cfg, n_instruments=n_instruments, spec=spec),
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
    # Render provenance / byte-equivalence contract from the compiled store.
    render_sources = [
        ds._chart_source
        for ds in datasets
        if getattr(ds, "_chart_source", None) is not None
    ]
    if render_sources:
        fingerprints = {s.render_meta["fingerprint"] for s in render_sources}
        if len(fingerprints) != 1:
            raise RuntimeError(
                "compiled chart stores use inconsistent render identities across "
                f"segments: {sorted(fingerprints)[:4]}"
            )
        m = render_sources[0].render_meta
        ssl_cfg.renderer_version = str(m.get("renderer"))
        ssl_cfg.render_spec_hash = str(m.get("spec_hash"))
        ssl_cfg.render_fingerprint = str(m.get("fingerprint"))
        ssl_cfg.render_store_checksum = str(render_sources[0].render_checksum())
        print(
            "Render contract: "
            f"renderer={ssl_cfg.renderer_version} "
            f"spec={ssl_cfg.render_spec_hash[:12]} "
            f"fp={ssl_cfg.render_fingerprint[:12]} "
            f"checksum={ssl_cfg.render_store_checksum[:12]}"
        )
    render_contract_actual = resolve_render_contract(datasets, spec.image_size)
    if args.resume_from:
        _rp = load_checkpoint(args.resume_from)
        enforce_parent_render_contract(
            render_contract_actual, _rp, stage_label="S1-resume"
        )
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
