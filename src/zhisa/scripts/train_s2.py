"""Train the S2 supervised market heads on labelled market data."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from zhisa.config import load_config
from zhisa.data.dataset import MacroContextConfig, MarketDataset, MarketTargetConfig, SampleSpec
from zhisa.data.labeling import TripleBarrierConfig
from zhisa.data.preparation import load_prepared_split, load_prepared_symbol
from zhisa.models.policy import PolicyConfig, PolicyNetwork, build_default_policy
from zhisa.scripts._real_data import add_market_data_args, load_market_dataframe
from zhisa.scripts.train_s1 import _concat, _market_datasets_from_frame
from zhisa.training.losses import LossWeights, MultiTaskLoss
from zhisa.training.optim import OptimConfig
from zhisa.training.s1_ssl import _filter_matching_state_dict
from zhisa.training.s2_supervised import SupervisedTrainer, TrainConfig
from zhisa.utils.seeding import set_seed


WARM_START_OPTIONAL_MISSING_PREFIXES = (
    "heads.policy",
    "macro_numeric.",
    "timeframe_embed.",
    "macro_gate.",
    "macro_proj.",
    "macro_norm.",
)


def _default_device() -> str:
    pref = os.environ.get("ZHISA_TEST_DEVICE", "auto").lower()
    if pref in {"cpu", "cuda"}:
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"


def _optim_config_from(cfg) -> OptimConfig:
    raw = (cfg.get("optim", {}) or {}) if cfg else {}
    return OptimConfig(
        lr=float(raw.get("lr", 3e-4)),
        weight_decay=float(raw.get("weight_decay", 1e-4)),
        betas=tuple(raw.get("betas", (0.9, 0.95))),
        scheduler=str(raw.get("scheduler", "cosine")),
        warmup_steps=int(raw.get("warmup_steps", 200)),
        step_size=int(raw.get("step_size", 1000)),
        step_gamma=float(raw.get("step_gamma", 0.5)),
        t_max=int(raw.get("t_max", 10_000)),
    )


def _loss_weights_from(cfg) -> LossWeights:
    raw = (cfg.get("loss_weights", {}) or {}) if cfg else {}
    defaults = LossWeights()
    values = {}
    for item in fields(LossWeights):
        default = getattr(defaults, item.name)
        value = raw.get(item.name, default)
        values[item.name] = int(value) if isinstance(default, int) else float(value)
    return LossWeights(**values)


def _sample_spec(cfg, s1_payload: dict | None) -> SampleSpec:
    chart_window = int(cfg.get("chart_window", 32)) if cfg else 32
    image_size = int(cfg.get("image_size", 32)) if cfg else 32
    model_cfg = (s1_payload or {}).get("model_config") or (s1_payload or {}).get("config")
    if model_cfg:
        chart_window = int(model_cfg.get("window", chart_window))
        image_size = int(model_cfg.get("image_size", image_size))
    horizons = tuple(int(x) for x in (cfg.get("horizons", (4, 16, 64)) if cfg else (4, 16, 64)))
    return SampleSpec(
        chart_window=chart_window,
        feature_window=chart_window,
        horizons=horizons,
        image_size=image_size,
        n_regime_states=int(cfg.get("n_regime_states", 4)) if cfg else 4,
    )


def _target_config_from(cfg) -> tuple[MarketTargetConfig, TripleBarrierConfig]:
    raw = (cfg.get("targets", {}) or {}) if cfg else {}
    direction_mode = str(raw.get("direction_mode", "forward_return"))
    horizon_overrides = _target_horizon_overrides_from(raw)
    target_cfg = MarketTargetConfig(
        direction_mode=direction_mode,
        flat_return_bps=float(raw.get("flat_return_bps", 1.0)),
        flat_volatility_mult=float(raw.get("flat_volatility_mult", 0.0)),
        flat_min_bps=float(raw.get("flat_min_bps", 0.0)),
        flat_max_bps=float(raw.get("flat_max_bps", 0.0)),
        use_log_return=bool(raw.get("use_log_return", False)),
        horizon_overrides=horizon_overrides or None,
    )
    tb_raw = raw.get("triple_barrier", {}) or {}
    tb_cfg = TripleBarrierConfig(
        tp_atr_mult=float(tb_raw.get("tp_atr_mult", 2.0)),
        sl_atr_mult=float(tb_raw.get("sl_atr_mult", 2.0)),
        max_holding=int(tb_raw.get("max_holding", 32)),
        atr_window=int(tb_raw.get("atr_window", 14)),
    )
    allow_asym = bool(raw.get("allow_asymmetric_triple_barrier_direction", False))
    if (
        direction_mode == "triple_barrier"
        and not allow_asym
        and abs(tb_cfg.tp_atr_mult - tb_cfg.sl_atr_mult) > 1e-12
    ):
        raise ValueError(
            "S2 direction_mode='triple_barrier' requires symmetric TP/SL by default; "
            "set targets.allow_asymmetric_triple_barrier_direction=true only for an explicit legacy experiment"
        )
    return target_cfg, tb_cfg


def _target_horizon_overrides_from(raw: dict) -> dict[int, dict[str, float | bool]]:
    allowed = {
        "flat_return_bps",
        "flat_volatility_mult",
        "flat_min_bps",
        "flat_max_bps",
        "use_log_return",
    }
    source = raw.get("horizon_overrides", {}) or {}
    if not isinstance(source, dict):
        raise ValueError("targets.horizon_overrides must be a mapping")
    overrides: dict[int, dict[str, float | bool]] = {}
    for key, values in source.items():
        try:
            horizon = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid horizon override key: {key!r}") from exc
        if horizon <= 0:
            raise ValueError("horizon override keys must be positive integers")
        if not isinstance(values, dict):
            raise ValueError(f"targets.horizon_overrides.{key} must be a mapping")
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(
                f"unknown target horizon override fields for {horizon}: {sorted(unknown)}"
            )
        item: dict[str, float | bool] = {}
        for name, value in values.items():
            item[name] = bool(value) if name == "use_log_return" else float(value)
        if (
            "flat_min_bps" in item
            and "flat_max_bps" in item
            and float(item["flat_max_bps"]) > 0.0
            and float(item["flat_max_bps"]) < float(item["flat_min_bps"])
        ):
            raise ValueError(
                f"targets.horizon_overrides.{horizon}.flat_max_bps must be >= flat_min_bps"
            )
        overrides[horizon] = item
    return overrides


def _critical_warm_start_missing_keys(missing: list[str]) -> list[str]:
    return [
        key for key in missing
        if not key.startswith(WARM_START_OPTIONAL_MISSING_PREFIXES)
    ]


def _macro_config_from(cfg) -> MacroContextConfig:
    raw = (cfg.get("macro_context", {}) or {}) if cfg else {}
    return MacroContextConfig(
        enabled=bool(raw.get("enabled", False)),
        window=int(raw.get("window", 64)),
        resample_rule=str(raw.get("resample_rule", "1h")),
        source=str(raw.get("source", "resample")),
    )


def _macro_prepared_root_from(cfg, cli_value: str | None) -> str | None:
    if cli_value:
        return cli_value
    raw = (cfg.get("macro_context", {}) or {}) if cfg else {}
    value = raw.get("prepared_root")
    return str(value) if value else None


def _load_macro_prepared_frames(
    root: Path,
    *,
    primary_manifest: dict,
    expected_timeframe: str,
) -> tuple[dict[str, pd.DataFrame], dict]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"macro prepared manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    timeframe = str(manifest.get("timeframe", ""))
    if timeframe != expected_timeframe:
        raise ValueError(
            f"macro prepared timeframe must be {expected_timeframe!r}, got {timeframe!r}"
        )
    primary_symbols = set(primary_manifest.get("symbols") or [])
    macro_symbols = set(manifest.get("symbols") or [])
    missing = sorted(primary_symbols - macro_symbols)
    if missing:
        raise ValueError(f"macro prepared data is missing symbols: {missing}")
    frames: dict[str, pd.DataFrame] = {}
    for symbol in sorted(primary_symbols):
        frame = load_prepared_symbol(root, symbol)
        if "symbol" in frame.columns:
            frame = frame.drop(columns=["symbol"])
        frames[symbol] = frame.sort_index()
    return frames, manifest


def _build_policy(
    first_ds: MarketDataset,
    spec: SampleSpec,
    s1_payload: dict | None,
    cfg=None,
) -> PolicyNetwork:
    n_feat = first_ds._features_df.shape[1]
    n_macro_feat = (
        first_ds._macro_features_df.shape[1]
        if first_ds._macro_features_df is not None
        else n_feat
    )
    n_ctx = first_ds._time_features_df.shape[1]
    macro_cfg = _macro_config_from(cfg)
    raw = (s1_payload or {}).get("model_config") or (s1_payload or {}).get("config")
    if not raw:
        return build_default_policy(
            in_numeric_features=n_feat,
            in_macro_features=n_macro_feat,
            in_context_features=n_ctx,
            window=spec.chart_window,
            macro_window=macro_cfg.window,
            image_size=spec.image_size,
            n_actions=9,
            n_regime_classes=spec.n_regime_states,
            market_horizons=tuple(int(x) for x in spec.horizons),
            use_macro_context=macro_cfg.enabled,
        )

    model_cfg = dict(raw)
    if isinstance(model_cfg.get("vision_channels"), list):
        model_cfg["vision_channels"] = tuple(model_cfg["vision_channels"])
    if isinstance(model_cfg.get("market_horizons"), list):
        model_cfg["market_horizons"] = tuple(int(x) for x in model_cfg["market_horizons"])
    allowed = {item.name for item in fields(PolicyConfig)}
    model_cfg = {key: value for key, value in model_cfg.items() if key in allowed}
    expected = {
        "in_numeric_features": n_feat,
        "in_macro_features": n_macro_feat,
        "in_context_features": n_ctx,
        "window": spec.chart_window,
        "macro_window": macro_cfg.window,
        "image_size": spec.image_size,
        "n_regime_classes": spec.n_regime_states,
    }
    mismatches = {
        key: (model_cfg.get(key), value)
        for key, value in expected.items()
        if key in model_cfg and int(model_cfg[key]) != int(value)
    }
    if mismatches:
        raise ValueError(f"S1 checkpoint is incompatible with the S2 dataset: {mismatches}")
    model_cfg.update(expected)
    model_cfg["market_horizons"] = tuple(int(x) for x in spec.horizons)
    model_cfg["use_macro_context"] = macro_cfg.enabled
    return PolicyNetwork(PolicyConfig(**model_cfg))


def _load_s1_representation(model: PolicyNetwork, payload: dict) -> int:
    source = payload.get("model", payload)
    filtered = _filter_matching_state_dict(
        source,
        model,
        excluded_prefixes=("heads.", "memory."),
    )
    if not filtered:
        raise ValueError("S1 checkpoint contains no compatible representation weights")
    model.load_state_dict(filtered, strict=False)
    return len(filtered)


def _target_counts(
    datasets: list[MarketDataset],
    *,
    kind: str,
    n_classes: int,
) -> np.ndarray:
    counts = np.zeros(n_classes, dtype=np.int64)
    for ds in datasets:
        start = ds.spec.chart_window - 1
        stop = start + len(ds)
        if kind == "direction":
            values = ds._tb_label_arr[start:stop] + 1
        elif kind == "regime":
            values = ds._regime_arr[start:stop]
        else:
            raise ValueError(kind)
        counts += np.bincount(values, minlength=n_classes)[:n_classes]
    return counts


def _direction_sample_weights(
    datasets: list[MarketDataset],
    *,
    mode: str,
    power: float = 0.75,
    max_weight: float = 8.0,
) -> torch.Tensor | None:
    """Build sample weights that balance direction labels inside each segment.

    Global class weights can still hide a rare per-symbol FLAT class. This
    sampler gives every segment its own class-frequency correction, then
    normalises each segment back to mean weight 1 so market coverage remains
    balanced.
    """
    mode = str(mode or "none").lower()
    if mode == "none":
        return None
    if mode not in {"per_symbol_direction", "per_segment_direction"}:
        raise ValueError(f"unknown sample_balance mode: {mode!r}")
    if power < 0.0:
        raise ValueError("sample_balance_power must be >= 0")
    if max_weight <= 0.0:
        raise ValueError("sample_balance_max_weight must be positive")

    all_weights: list[np.ndarray] = []
    audits: list[str] = []
    for ds in datasets:
        start = ds.spec.chart_window - 1
        stop = start + len(ds)
        labels = ds._tb_label_arr[start:stop].astype(np.int64) + 1
        counts = np.bincount(labels, minlength=3)[:3].astype(np.float64)
        active = counts > 0
        raw = np.zeros(3, dtype=np.float64)
        if np.any(active):
            raw[active] = counts[active].sum() / (
                active.sum() * np.maximum(counts[active], 1.0)
            )
            raw[active] = np.power(raw[active], float(power))
            raw[active] = np.minimum(raw[active], float(max_weight))
            raw[active] /= max(raw[active].mean(), 1e-12)
        weights = raw[labels]
        weights /= max(float(weights.mean()), 1e-12)
        # Keep the user-facing cap true for the final sampler weights. In
        # extremely imbalanced tiny segments this can lower that segment's
        # total draw mass slightly, which is preferable to unbounded rare-class
        # oversampling.
        weights = np.minimum(weights, float(max_weight))
        all_weights.append(weights.astype(np.float32, copy=False))
        name = str(getattr(ds.df, "name", f"segment-{len(audits)}"))
        audits.append(
            f"{name}:counts={counts.astype(int).tolist()} weights={raw.round(4).tolist()}"
        )
    result = torch.from_numpy(np.concatenate(all_weights).astype(np.float32))
    print(
        "Sample balance: "
        f"mode={mode} power={power:.3f} max_weight={max_weight:.3f} "
        f"mean={float(result.mean()):.4f} min={float(result.min()):.4f} max={float(result.max()):.4f}"
    )
    for line in audits[:16]:
        print(f"  {line}")
    if len(audits) > 16:
        print(f"  ... {len(audits) - 16} more segments")
    return result


def _sqrt_inverse_weights(counts: np.ndarray) -> torch.Tensor:
    active = counts > 0
    if not np.any(active):
        raise ValueError(f"cannot balance classes with no targets: {counts.tolist()}")
    weights = np.zeros_like(counts, dtype=np.float64)
    weights[active] = np.sqrt(counts[active].sum() / (active.sum() * counts[active].astype(np.float64)))
    weights[active] /= weights[active].mean()
    return torch.tensor(weights, dtype=torch.float32)


def _inverse_weights(counts: np.ndarray) -> torch.Tensor:
    active = counts > 0
    if not np.any(active):
        raise ValueError(f"cannot balance classes with no targets: {counts.tolist()}")
    weights = np.zeros_like(counts, dtype=np.float64)
    weights[active] = counts[active].sum() / (active.sum() * counts[active].astype(np.float64))
    weights[active] /= weights[active].mean()
    return torch.tensor(weights, dtype=torch.float32)


def _effective_number_weights(counts: np.ndarray, beta: float = 0.9999) -> torch.Tensor:
    active = counts > 0
    if not np.any(active):
        raise ValueError(f"cannot balance classes with no targets: {counts.tolist()}")
    if not 0.0 <= beta < 1.0:
        raise ValueError("effective-number beta must be in [0, 1)")
    weights = np.zeros_like(counts, dtype=np.float64)
    effective = 1.0 - np.power(beta, counts[active].astype(np.float64))
    weights[active] = (1.0 - beta) / np.maximum(effective, 1e-12)
    weights[active] /= weights[active].mean()
    return torch.tensor(weights, dtype=torch.float32)


def _cap_class_weights(
    weights: torch.Tensor,
    *,
    max_weight: float = 0.0,
    flat_max_weight: float = 0.0,
) -> torch.Tensor:
    """Cap class weights while preserving the active-class mean scale.

    Direction labels have a rare but semantically delicate FLAT class. We
    want it represented, not treated as a high-confidence answer whenever the
    model is unsure. A cap keeps class balancing from overpowering sampler
    balance and label smoothing.
    """
    capped = weights.clone().float()
    active = capped > 0
    if max_weight > 0.0:
        capped[active] = capped[active].clamp(max=float(max_weight))
    if flat_max_weight > 0.0 and capped.numel() >= 2 and bool(active[1]):
        capped[1] = min(float(capped[1]), float(flat_max_weight))
    if bool(active.any()):
        capped[active] = capped[active] / capped[active].mean().clamp_min(1e-12)
    if max_weight > 0.0:
        capped[active] = capped[active].clamp(max=float(max_weight))
    if flat_max_weight > 0.0 and capped.numel() >= 2 and bool(active[1]):
        capped[1] = min(float(capped[1]), float(flat_max_weight))
    return capped


def _class_weights(
    counts: np.ndarray,
    mode: str,
    *,
    beta: float = 0.9999,
    max_weight: float = 0.0,
    flat_max_weight: float = 0.0,
) -> torch.Tensor:
    if mode == "sqrt_inverse":
        weights = _sqrt_inverse_weights(counts)
    elif mode == "inverse":
        weights = _inverse_weights(counts)
    elif mode in {"effective", "effective_number"}:
        weights = _effective_number_weights(counts, beta=beta)
    else:
        raise ValueError(f"unknown class_balance mode: {mode!r}")
    return _cap_class_weights(
        weights,
        max_weight=max_weight,
        flat_max_weight=flat_max_weight,
    )


def _target_contract_dict(
    target_cfg: MarketTargetConfig,
    tb_cfg: TripleBarrierConfig,
) -> dict:
    return {
        "direction_mode": target_cfg.direction_mode,
        "flat_return_bps": target_cfg.flat_return_bps,
        "flat_volatility_mult": target_cfg.flat_volatility_mult,
        "flat_min_bps": target_cfg.flat_min_bps,
        "flat_max_bps": target_cfg.flat_max_bps,
        "use_log_return": target_cfg.use_log_return,
        "horizon_overrides": target_cfg.horizon_overrides or {},
        "triple_barrier": {
            "tp_atr_mult": tb_cfg.tp_atr_mult,
            "sl_atr_mult": tb_cfg.sl_atr_mult,
            "max_holding": tb_cfg.max_holding,
            "atr_window": tb_cfg.atr_window,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train S2 supervised market heads.")
    parser.add_argument("--config", type=str, default="configs/s2_supervised.yaml")
    parser.add_argument("--bars", type=int, default=8000)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="artifacts/s2/model.pt")
    parser.add_argument("--best-checkpoint", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--s1-checkpoint", type=str, default=None)
    parser.add_argument(
        "--warm-start-checkpoint",
        type=str,
        default=None,
        help="Initialise the full S2 model from a checkpoint but reset optimizer/scheduler/trainer state.",
    )
    parser.add_argument("--prepared-root", type=str, default=None)
    parser.add_argument(
        "--macro-prepared-root",
        type=str,
        default=None,
        help="Prepared higher-timeframe dataset root used when macro_context.source='prepared'.",
    )
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--prepared-max-bars-per-symbol", type=int, default=None)
    parser.add_argument("--fast-render", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--cache-charts", action="store_true")
    parser.add_argument("--chart-cache-size", type=int, default=-1)
    add_market_data_args(parser)
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path) if cfg_path.exists() else None
    seed = int(cfg.get("seed", 0)) if cfg else 0
    set_seed(seed)
    if args.fast_render:
        os.environ["ZHISA_FAST_RENDER"] = "1"

    s1_payload = None
    if args.s1_checkpoint:
        s1_path = Path(args.s1_checkpoint)
        if not s1_path.is_file():
            raise FileNotFoundError(f"S1 checkpoint not found: {s1_path}")
        s1_payload = torch.load(s1_path, map_location="cpu", weights_only=False)
        stage = (s1_payload.get("checkpoint_meta") or {}).get("stage")
        if stage not in (None, "s1_ssl"):
            raise ValueError(f"expected an S1 checkpoint, got stage={stage!r}")

    warm_start_payload = None
    if args.warm_start_checkpoint:
        warm_path = Path(args.warm_start_checkpoint)
        if not warm_path.is_file():
            raise FileNotFoundError(f"warm-start checkpoint not found: {warm_path}")
        warm_start_payload = torch.load(warm_path, map_location="cpu", weights_only=False)
        warm_stage = (warm_start_payload.get("checkpoint_meta") or {}).get("stage")
        if warm_stage not in (None, "s2_supervised"):
            raise ValueError(
                "S2 warm start expects an S2 checkpoint; "
                f"got stage={warm_stage!r}"
            )

    spec = _sample_spec(cfg, warm_start_payload or s1_payload)
    target_cfg, tb_cfg = _target_config_from(cfg)
    macro_cfg = _macro_config_from(cfg)
    manifest = None
    macro_manifest = None
    val_ds: Dataset | None = None
    if args.prepared_root:
        prepared_root = Path(args.prepared_root)
        manifest_path = prepared_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"prepared manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        timeframe = str(manifest["timeframe"])
        macro_frames_by_symbol = None
        macro_root_value = _macro_prepared_root_from(cfg, args.macro_prepared_root)
        if macro_cfg.enabled and macro_cfg.source == "prepared":
            if not macro_root_value:
                raise ValueError(
                    "macro_context.source='prepared' requires --macro-prepared-root "
                    "or macro_context.prepared_root"
                )
            macro_frames_by_symbol, macro_manifest = _load_macro_prepared_frames(
                Path(macro_root_value),
                primary_manifest=manifest,
                expected_timeframe=macro_cfg.resample_rule,
            )
            print(
                "Prepared macro context: "
                f"root={macro_root_value} timeframe={macro_manifest['timeframe']} "
                f"symbols={len(macro_frames_by_symbol)}"
            )
        train_frame = load_prepared_split(prepared_root, args.train_split)
        train_rows = len(train_frame)
        train_datasets = _market_datasets_from_frame(
            train_frame,
            spec=spec,
            cache_charts=args.cache_charts,
            chart_cache_size=args.chart_cache_size,
            max_bars_per_symbol=args.prepared_max_bars_per_symbol,
            timeframe=timeframe,
            compute_targets=True,
            target_cfg=target_cfg,
            triple_barrier_cfg=tb_cfg,
            macro_cfg=macro_cfg,
            macro_frames_by_symbol=macro_frames_by_symbol,
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
                timeframe=timeframe,
                compute_targets=True,
                target_cfg=target_cfg,
                triple_barrier_cfg=tb_cfg,
                macro_cfg=macro_cfg,
                macro_frames_by_symbol=macro_frames_by_symbol,
            )
            del val_frame
            val_ds = _concat(val_datasets)
        train_ds = _concat(train_datasets)
        print(
            f"Prepared S2 data: train={train_rows:,} rows, "
            f"segments={len(train_datasets)}, validation={'on' if val_ds is not None else 'off'}"
        )
    else:
        df = load_market_dataframe(args, seed=seed, default_bars=args.bars)
        only_ds = MarketDataset(
            df,
            spec=spec,
            triple_barrier_cfg=tb_cfg,
            target_cfg=target_cfg,
            cache_charts=args.cache_charts,
            chart_cache_size=args.chart_cache_size,
            macro_cfg=macro_cfg,
        )
        train_datasets = [only_ds]
        train_ds = only_ds

    first_ds = train_datasets[0]
    model = _build_policy(first_ds, spec, warm_start_payload or s1_payload, cfg)
    if warm_start_payload is not None:
        missing, unexpected = model.load_state_dict(warm_start_payload["model"], strict=False)
        critical_missing = _critical_warm_start_missing_keys(list(missing))
        if unexpected or critical_missing:
            raise RuntimeError(
                "warm-start checkpoint is incompatible: "
                f"unexpected={unexpected[:20]} missing={critical_missing[:20]}"
            )
        print(
            "Warm-started full S2 model from "
            f"{args.warm_start_checkpoint}; optimizer/scheduler state reset for fine-tune."
        )
    elif s1_payload is not None:
        loaded = _load_s1_representation(model, s1_payload)
        print(
            f"Loaded {loaded} S1 representation tensors; S2 market heads are freshly initialized. "
            f"Input contract: window={spec.chart_window}, image={spec.image_size}."
        )

    balance = str(cfg.get("class_balance", "none") if cfg else "none").lower()
    direction_weights = None
    regime_weights = None
    if balance in {"sqrt_inverse", "inverse", "effective", "effective_number"}:
        direction_counts = _target_counts(train_datasets, kind="direction", n_classes=3)
        regime_counts = _target_counts(
            train_datasets,
            kind="regime",
            n_classes=spec.n_regime_states,
        )
        beta = float(cfg.get("class_balance_beta", 0.9999) if cfg else 0.9999)
        direction_weights = _class_weights(
            direction_counts,
            balance,
            beta=beta,
            max_weight=float(cfg.get("direction_class_weight_max", 0.0) if cfg else 0.0),
            flat_max_weight=float(cfg.get("direction_flat_class_weight_max", 0.0) if cfg else 0.0),
        )
        regime_weights = _class_weights(regime_counts, balance, beta=beta)
        print(
            f"Class balance: mode={balance} direction={direction_counts.tolist()} weights={direction_weights.tolist()} "
            f"regime={regime_counts.tolist()} weights={regime_weights.tolist()}"
        )
        if target_cfg.direction_mode == "forward_return":
            down, flat, up = direction_counts.tolist()
            directional = down + up
            if directional > 0:
                up_share = up / directional
                print(
                    "Direction target audit: "
                    f"mode=forward_return down={down} flat={flat} up={up} "
                    f"up_share_ex_flat={up_share:.4f}"
                )
    elif balance != "none":
        raise ValueError(f"unknown class_balance mode: {balance!r}")

    loss_weights = _loss_weights_from(cfg)
    loss = MultiTaskLoss(
        loss_weights,
        label_smoothing=float(cfg.get("label_smoothing", 0.05)) if cfg else 0.05,
        direction_class_weights=direction_weights,
        regime_class_weights=regime_weights,
        return_direction_weight=float(cfg.get("return_direction_weight", 0.0)) if cfg else 0.0,
        return_corr_weight=float(cfg.get("return_corr_weight", 0.0)) if cfg else 0.0,
        return_target_scale=float((cfg.get("target_scales", {}) or {}).get("return", 1.0)) if cfg else 1.0,
        value_target_scale=float((cfg.get("target_scales", {}) or {}).get("value", 1.0)) if cfg else 1.0,
        volatility_target_scale=float((cfg.get("target_scales", {}) or {}).get("volatility", 1.0)) if cfg else 1.0,
        risk_target_scale=float((cfg.get("target_scales", {}) or {}).get("risk", 1.0)) if cfg else 1.0,
        volatility_log_weight=float(cfg.get("volatility_log_weight", 0.0)) if cfg else 0.0,
        volatility_corr_weight=float(cfg.get("volatility_corr_weight", 0.0)) if cfg else 0.0,
        direction_multi_horizon_weights=torch.tensor(
            cfg.get("direction_multi_horizon_weights", []),
            dtype=torch.float32,
        ) if cfg and cfg.get("direction_multi_horizon_weights") else None,
        return_multi_horizon_weights=torch.tensor(
            cfg.get("return_multi_horizon_weights", []),
            dtype=torch.float32,
        ) if cfg and cfg.get("return_multi_horizon_weights") else None,
    )
    train_sample_weights = _direction_sample_weights(
        train_datasets,
        mode=str(cfg.get("sample_balance", "none") if cfg else "none"),
        power=float(cfg.get("sample_balance_power", 0.75) if cfg else 0.75),
        max_weight=float(cfg.get("sample_balance_max_weight", 8.0) if cfg else 8.0),
    )
    epochs = args.epochs if args.epochs is not None else int(cfg.get("epochs", 2) if cfg else 2)
    batch_size = args.batch_size if args.batch_size is not None else int(cfg.get("batch_size", 32) if cfg else 32)
    device = args.device or str(cfg.get("device", _default_device()) if cfg else _default_device())
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    workers = args.workers if args.workers is not None else int(cfg.get("workers", 0) if cfg else 0)
    best_checkpoint = args.best_checkpoint
    if val_ds is not None and not best_checkpoint:
        checkpoint = Path(args.checkpoint)
        best_checkpoint = str(checkpoint.with_name(f"{checkpoint.stem}_best{checkpoint.suffix}"))
    train_cfg = TrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        grad_clip=float(cfg.get("grad_clip", 1.0)) if cfg else 1.0,
        num_workers=workers,
        log_every=int(cfg.get("log_every", 50)) if cfg else 50,
        eval_every=int(cfg.get("eval_every", 1)) if cfg else 1,
        val_max_batches=int(cfg.get("val_max_batches", 0)) if cfg else 0,
        checkpoint=args.checkpoint,
        best_checkpoint=best_checkpoint,
        checkpoint_every_steps=int(cfg.get("checkpoint_every_steps", 0)) if cfg else 0,
        freeze_encoder_epochs=int(cfg.get("freeze_encoder_epochs", 0)) if cfg else 0,
        encoder_lr_scale=float(cfg.get("encoder_lr_scale", 1.0)) if cfg else 1.0,
        early_stopping_patience=int(cfg.get("early_stopping_patience", 0)) if cfg else 0,
        early_stopping_min_delta=float(cfg.get("early_stopping_min_delta", 0.0)) if cfg else 0.0,
        early_stopping_min_epochs=int(cfg.get("early_stopping_min_epochs", 0)) if cfg else 0,
        early_stopping_trend_window=int(cfg.get("early_stopping_trend_window", 0)) if cfg else 0,
        early_stopping_trend_min_delta=float(cfg.get("early_stopping_trend_min_delta", 0.0)) if cfg else 0.0,
        device=device,
        seed=seed,
        dataset_root=str(Path(args.prepared_root).resolve()) if args.prepared_root else None,
        dataset_timeframe=str(manifest["timeframe"]) if manifest else None,
        dataset_manifest_checksum=str(manifest["output_checksum"]) if manifest else None,
        target_config={
            **_target_contract_dict(target_cfg, tb_cfg),
            "horizons": [int(x) for x in spec.horizons],
            "macro_context": {
                "enabled": macro_cfg.enabled,
                "source": macro_cfg.source,
                "window": macro_cfg.window,
                "resample_rule": macro_cfg.resample_rule,
                "prepared_root": str(Path(_macro_prepared_root_from(cfg, args.macro_prepared_root)).resolve())
                if _macro_prepared_root_from(cfg, args.macro_prepared_root)
                else None,
                "prepared_timeframe": str(macro_manifest["timeframe"]) if macro_manifest else None,
                "prepared_manifest_checksum": str(macro_manifest["output_checksum"]) if macro_manifest else None,
            },
        },
        champion_metric=str(cfg.get("champion_metric", "s2_composite_score") if cfg else "s2_composite_score"),
        champion_mode=str(cfg.get("champion_mode", "max") if cfg else "max"),
        segment_validation=bool(cfg.get("segment_validation", False) if cfg else False),
        guard_min_direction_balanced=float(cfg.get("guard_min_direction_balanced", 0.0) if cfg else 0.0),
        guard_min_flat_recall=float(cfg.get("guard_min_flat_recall", 0.0) if cfg else 0.0),
        guard_min_flat_f1=float(cfg.get("guard_min_flat_f1", 0.0) if cfg else 0.0),
        guard_min_volatility_corr=float(cfg.get("guard_min_volatility_corr", -1.0) if cfg else -1.0),
        guard_min_return_corr=float(cfg.get("guard_min_return_corr", -1.0) if cfg else -1.0),
        guard_min_persistence_lift=float(cfg.get("guard_min_persistence_lift", -1.0) if cfg else -1.0),
        guard_max_prediction_share=float(cfg.get("guard_max_prediction_share", 1.0) if cfg else 1.0),
        guard_max_flat_prediction_share=float(cfg.get("guard_max_flat_prediction_share", 1.0) if cfg else 1.0),
        guard_min_flat_pred_target_ratio=float(cfg.get("guard_min_flat_pred_target_ratio", 0.0) if cfg else 0.0),
        guard_max_flat_pred_target_ratio=float(cfg.get("guard_max_flat_pred_target_ratio", 10.0) if cfg else 10.0),
        guard_penalty_scale=float(cfg.get("guard_penalty_scale", 0.0) if cfg else 0.0),
        optim=_optim_config_from(cfg),
    )
    trainer = SupervisedTrainer(model, loss, train_cfg, train_sample_weights=train_sample_weights)
    if args.resume_from:
        if not Path(args.resume_from).is_file():
            raise FileNotFoundError(f"S2 resume checkpoint not found: {args.resume_from}")
        print(f"Resumed S2: {trainer.load(args.resume_from)}")
    result = trainer.fit(train_ds, val_ds=val_ds)
    final = result["history"][-1]
    message = f"S2 training complete. final_loss={final['loss']:.6f}"
    if "val" in final:
        message += (
            f" val_total={final['val']['total']:.6f}"
            f" direction_acc={final['val']['direction_accuracy']:.4f}"
            f" direction_bal_acc={final['val']['direction_balanced_accuracy']:.4f}"
            f" return_corr={final['val']['return_corr']:.4f}"
            f" s2_score={final['val']['s2_composite_score']:.4f}"
            f" guarded_score={final['val'].get('s2_guarded_score', final['val']['s2_composite_score']):.4f}"
            f" regime_acc={final['val']['regime_accuracy']:.4f}"
        )
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
