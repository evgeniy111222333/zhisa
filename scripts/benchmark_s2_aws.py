"""Short end-to-end S2 throughput benchmark for an AWS GPU instance."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch

os.environ.setdefault("ZHISA_FAST_RENDER", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from zhisa.data.dataset import SampleSpec, multimodal_collate
from zhisa.data.preparation import load_prepared_split
from zhisa.scripts.train_s1 import _concat, _market_datasets_from_frame
from zhisa.scripts.train_s2 import _build_policy, _load_s1_representation
from zhisa.training.dataloader_factory import build_dataloader
from zhisa.training.losses import MultiTaskLoss
from zhisa.training.optim import OptimConfig, build_optimizer


def _move(batch, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "chart": batch.chart.to(device, non_blocking=True),
        "numeric": batch.numeric.to(device, non_blocking=True),
        "context": batch.context.to(device, non_blocking=True),
        "label_dir": batch.label_dir.to(device, non_blocking=True),
        "label_vol": batch.label_vol.to(device, non_blocking=True),
        "label_risk": batch.label_risk.to(device, non_blocking=True),
        "label_regime": batch.label_regime.to(device, non_blocking=True),
        "label_ret": batch.label_ret.to(device, non_blocking=True),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--s1-checkpoint", required=True)
    parser.add_argument("--batches", default="128,256,512,768,1024")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bars-per-symbol", type=int, default=5000)
    parser.add_argument("--timed-batches", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    device = torch.device("cuda")
    payload = torch.load(args.s1_checkpoint, map_location="cpu", weights_only=False)
    model_cfg = payload.get("model_config") or payload["config"]
    spec = SampleSpec(
        chart_window=int(model_cfg["window"]),
        feature_window=int(model_cfg["window"]),
        image_size=int(model_cfg["image_size"]),
    )
    root = Path(args.prepared_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    frame = load_prepared_split(root, "train")
    datasets = _market_datasets_from_frame(
        frame,
        spec=spec,
        cache_charts=False,
        chart_cache_size=-1,
        max_bars_per_symbol=args.bars_per_symbol,
        timeframe=str(manifest["timeframe"]),
        compute_targets=True,
    )
    del frame
    dataset = _concat(datasets)
    first = datasets[0]
    print(f"benchmark_samples={len(dataset):,} segments={len(datasets)}")

    results = []
    for batch_size in [int(x) for x in args.batches.split(",") if x.strip()]:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = _build_policy(first, spec, payload).to(device)
        _load_s1_representation(model, payload)
        model.train()
        loss_fn = MultiTaskLoss().to(device)
        optimizer = build_optimizer(
            list(model.parameters()) + list(loss_fn.parameters()),
            OptimConfig(lr=1e-4, scheduler="none"),
        )
        loader = build_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.workers,
            collate_fn=multimodal_collate,
            drop_last=True,
        )
        iterator = iter(loader)
        timed = 0
        started = None
        status = "ok"
        try:
            for index in range(args.timed_batches + 2):
                batch = next(iterator)
                moved = _move(batch, device)
                output = model(
                    chart=moved["chart"],
                    numeric=moved["numeric"],
                    context=moved["context"],
                )
                losses = loss_fn(output, moved)
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if index == 1:
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                elif index > 1:
                    timed += 1
        except StopIteration:
            status = "dataset_exhausted"
        except torch.cuda.OutOfMemoryError:
            status = "oom"
        finally:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started if started is not None else 0.0
        samples = timed * batch_size
        result = {
            "batch_size": batch_size,
            "workers": args.workers,
            "status": status,
            "timed_batches": timed,
            "seconds": round(elapsed, 3),
            "samples_per_second": round(samples / elapsed, 2) if elapsed else 0.0,
            "steps_per_second": round(timed / elapsed, 4) if elapsed else 0.0,
            "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        }
        results.append(result)
        print(json.dumps(result), flush=True)
        del iterator, loader, optimizer, loss_fn, model
        gc.collect()
        torch.cuda.empty_cache()

    print("RESULTS=" + json.dumps(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
