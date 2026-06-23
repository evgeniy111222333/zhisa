"""Frozen-probe and data-destruction audit for an S1 checkpoint."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

os.environ.setdefault("ZHISA_FAST_RENDER", "1")

from zhisa.data.dataset import MarketDataset, SampleSpec, multimodal_collate
from zhisa.models.policy import PolicyConfig, PolicyNetwork


def _longest_contiguous(frame: pd.DataFrame, timeframe: str = "15min") -> pd.DataFrame:
    expected = pd.Timedelta(timeframe)
    ids = frame.index.to_series().diff().ne(expected).cumsum()
    segments = [part for _, part in frame.groupby(ids, sort=False)]
    return max(segments, key=len)


def _load_probe_frames(data_dir: Path, symbol: str, max_bars: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = data_dir.parent if data_dir.name == "symbols" else data_dir
    splits = root / "splits"
    if splits.is_dir():
        frames = []
        for split in ("train", "test"):
            frame = pd.read_parquet(splits / f"{split}.parquet")
            if "symbol" not in frame.columns:
                raise ValueError(f"prepared {split} split has no symbol column")
            frame = frame.loc[frame["symbol"] == symbol].drop(columns=["symbol"])
            frame = _longest_contiguous(frame.sort_index())
            frames.append(frame.iloc[-max_bars:])
        return frames[0], frames[1]

    symbol_path = data_dir / f"{symbol.replace('/', '_')}.parquet"
    frame = _longest_contiguous(pd.read_parquet(symbol_path).sort_index())
    if len(frame) < max_bars * 2:
        raise ValueError(f"{symbol_path} needs at least {max_bars * 2} contiguous bars")
    return frame.iloc[-2 * max_bars : -max_bars], frame.iloc[-max_bars:]


def _derangement(batch_size: int, device: torch.device) -> torch.Tensor:
    if batch_size < 2:
        return torch.arange(batch_size, device=device)
    return torch.roll(torch.arange(batch_size, device=device), shifts=batch_size // 2)


@torch.no_grad()
def extract_data(
    ds: MarketDataset,
    model: PolicyNetwork,
    *,
    batch_size: int,
    device: torch.device,
    destruction_mode: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=multimodal_collate,
        shuffle=False,
    )
    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        chart = batch.chart.to(device)
        numeric = batch.numeric.to(device)
        context = batch.context.to(device)
        permutation = _derangement(len(chart), device)
        if destruction_mode == "shuffle_numbers":
            numeric = numeric[permutation]
            context = context[permutation]
        elif destruction_mode == "shuffle_all":
            chart = chart[permutation]
            numeric = numeric[permutation]
            context = context[permutation]
        elif destruction_mode is not None:
            raise ValueError(f"unknown destruction mode: {destruction_mode}")

        # S1 pretrains PolicyNetwork.encode(), before working memory and heads.
        embeddings.append(model.encode(chart, numeric, context).cpu().numpy())
        labels.append(batch.label_dir.cpu().numpy())
    return np.concatenate(embeddings), np.concatenate(labels)


def _directional_only(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Timeout/no-barrier events are neutral. The directional probe is SL (-1)
    # versus TP (+1), mapped to binary targets.
    mask = y != 0
    return x[mask], (y[mask] > 0).astype(np.int64)


def _metrics(y: np.ndarray, pred: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "roc_auc": float(roc_auc_score(y, probability)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--max-bars", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cache-charts", action="store_true")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    raw_config = dict(checkpoint["model_config"])
    if isinstance(raw_config.get("vision_channels"), list):
        raw_config["vision_channels"] = tuple(raw_config["vision_channels"])
    model = PolicyNetwork(PolicyConfig(**raw_config))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(device)

    spec = SampleSpec(
        chart_window=int(raw_config["window"]),
        feature_window=int(raw_config["window"]),
        image_size=int(raw_config["image_size"]),
        n_regime_states=int(raw_config["n_regime_classes"]),
    )
    train_frame, test_frame = _load_probe_frames(
        Path(args.data_dir), args.symbol, args.max_bars
    )
    train_ds = MarketDataset(
        train_frame,
        spec=spec,
        cache_charts=args.cache_charts,
        chart_cache_size=-1,
        compute_targets=True,
    )
    test_ds = MarketDataset(
        test_frame,
        spec=spec,
        cache_charts=args.cache_charts,
        chart_cache_size=-1,
        compute_targets=True,
    )

    print("Extracting clean train/test embeddings...")
    train_x, train_y = extract_data(
        train_ds, model, batch_size=args.batch_size, device=device
    )
    test_x, test_y = extract_data(
        test_ds, model, batch_size=args.batch_size, device=device
    )
    train_x, train_y = _directional_only(train_x, train_y)
    test_x, test_y = _directional_only(test_x, test_y)

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=0,
    )
    classifier.fit(train_x, train_y)

    def evaluate(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        return _metrics(y, classifier.predict(x), classifier.predict_proba(x)[:, 1])

    train_metrics = evaluate(train_x, train_y)
    clean_metrics = evaluate(test_x, test_y)
    destroyed = {}
    for mode in ("shuffle_numbers", "shuffle_all"):
        print(f"Extracting test embeddings: {mode}...")
        x, y = extract_data(
            test_ds,
            model,
            batch_size=args.batch_size,
            device=device,
            destruction_mode=mode,
        )
        x, y = _directional_only(x, y)
        destroyed[mode] = evaluate(x, y)

    counts = np.bincount(test_y, minlength=2)
    majority_accuracy = float(counts.max() / counts.sum())
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "symbol": args.symbol,
        "device": str(device),
        "train_samples": int(len(train_y)),
        "test_samples": int(len(test_y)),
        "test_class_counts": {"down": int(counts[0]), "up": int(counts[1])},
        "majority_accuracy": majority_accuracy,
        "train": train_metrics,
        "clean_test": clean_metrics,
        **destroyed,
        "generalization_gap_balanced_accuracy": float(
            train_metrics["balanced_accuracy"] - clean_metrics["balanced_accuracy"]
        ),
        "destruction_drop_balanced_accuracy": float(
            clean_metrics["balanced_accuracy"]
            - destroyed["shuffle_all"]["balanced_accuracy"]
        ),
    }
    report_path = out_dir / "overfitting_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 64)
    print(f"Majority accuracy:              {majority_accuracy:.2%}")
    print(f"Train balanced accuracy:        {train_metrics['balanced_accuracy']:.2%}")
    print(f"Clean test balanced accuracy:   {clean_metrics['balanced_accuracy']:.2%}")
    print(f"Shuffled numbers balanced acc:  {destroyed['shuffle_numbers']['balanced_accuracy']:.2%}")
    print(f"Shuffled all balanced accuracy: {destroyed['shuffle_all']['balanced_accuracy']:.2%}")
    print(
        "Generalization gap:             "
        f"{report['generalization_gap_balanced_accuracy']:.2%}"
    )
    print(
        "Full-destruction drop:          "
        f"{report['destruction_drop_balanced_accuracy']:.2%}"
    )
    print(f"Report: {report_path}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
