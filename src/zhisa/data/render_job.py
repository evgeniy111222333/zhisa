"""Compiled render job: parallel + incremental chart materialisation.

This module turns chart materialisation into a proper *job* rather than a
serial loop inside the data constructor:

- **Parallel**: a :class:`ProcessPoolExecutor` renders disjoint row-slices of a
  symbol into the same memmap; every row is produced by the pure canonical
  renderer, so the result is byte-identical regardless of worker count.
- **Incremental**: when a frame grows (new bars appended upstream), only the
  windows that now see new data are re-rendered. The unchanged *prefix* of an
  older artefact is reused after verifying a prefix-stable covered-hash, and
  its bytes are copied into the new artefact instead of recomputed.
- **Atomic**: a build is written to a temporary directory and renamed only on
  success, so a crash never leaves a partially-usable artefact behind.

Rendered bytes are identical to ``CompiledChartStore.build`` (serial); this
module only changes *how fast* and *how much* is recomputed.
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from zhisa.data.chart_store import (
    CompiledChartStore,
    content_key,
    covered_prefix_hash,
    frame_checksum,
)
from zhisa.rendering.chart_renderer import (
    _render_ohlcv_canonical,
    render_fingerprint,
)
from zhisa.rendering.spec import RenderSpec


@dataclass
class RenderJobStats:
    """What the job did (for logs / tests)."""

    n_rows: int
    workers: int
    chunk_size: int
    reused_artifact: bool = False
    reused_prefix_rows: int = 0
    rendered_rows: int = 0


def _render_rows_task(ohlcv_arr, window, spec_dict, start, end, bin_path, image_size, total_rows) -> int:
    """Render ``[start, end)`` rows into a shared memmap (top-level worker fn).

    Each row depends only on its own window + spec, so disjoint slices can be
    computed in any order and the resulting file is byte-identical. Returns the
    number of rows rendered.
    """
    spec = RenderSpec.from_meta(spec_dict)
    values = np.ascontiguousarray(ohlcv_arr, dtype=np.float64)
    arr = np.memmap(
        bin_path, dtype="float32", mode="r+",
        shape=(int(total_rows), 3, int(image_size), int(image_size)),
    )
    try:
        for row in range(int(start), int(end)):
            img = _render_ohlcv_canonical(values[row:row + window], spec).transpose(2, 0, 1)
            arr[int(row)] = img.astype(np.float32, copy=False)
        arr.flush()
    finally:
        del arr
    return int(end) - int(start)


def materialize_parallel(
    df: pd.DataFrame,
    window: int,
    spec: RenderSpec,
    *,
    n: Optional[int] = None,
    out_root: Optional[Path | str] = None,
    workers: int = 0,
    chunk_size: int = 5_000,
    progress_every: int = 50_000,
    engine: str = "cpu",
) -> tuple[CompiledChartStore, RenderJobStats]:
    """Compile charts for the contiguous samples ``0..n-1``.

    1. exact content-addressed cache hit  -> reuse unchanged;
    2. incremental reuse of a shorter artefact with the same render identity;
    3. else full parallel build.
    """
    window = int(window)
    n_ = len(df) if n is None else int(n)
    if out_root is None:
        store = CompiledChartStore.build(df, window=window, spec=spec, indices=range(n_), out_root=None)
        return store, RenderJobStats(n_rows=n_, workers=0, chunk_size=0)

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    full_frame_hash = frame_checksum(df)
    key = content_key(spec, window, full_frame_hash, range(n_), n_)
    artifact_dir = out_root / key

    if (artifact_dir / "charts.bin").exists() and (artifact_dir / "meta.json").exists():
        return CompiledChartStore.open(artifact_dir), RenderJobStats(
            n_rows=n_, workers=workers, chunk_size=chunk_size, reused_artifact=True
        )

    reuse = _find_incremental_reuse(out_root, df, window, spec, n_)
    if reuse is not None:
        return _build_with_reuse(df, window, spec, n_, artifact_dir, reuse, workers, chunk_size, full_frame_hash, key, progress_every, engine)

    return _build_full(df, window, spec, n_, artifact_dir, workers, chunk_size, full_frame_hash, key, progress_every, engine)


def _find_incremental_reuse(
    out_root: Path, df: pd.DataFrame, window: int, spec: RenderSpec, n_new: int
) -> Optional[tuple[Path, int]]:
    """Return ``(artifact_dir, reused_rows)`` for the best prefix-compatible artefact.

    A candidate must:
    - share the same render identity (fingerprint) and window,
    - be a contiguous ``0..last`` artefact shorter than the new build,
    - and its covered prefix must be hash-verified against the new frame (so the
      shared rows really are identical inputs).
    """
    want = render_fingerprint(spec)
    best: Optional[tuple[int, Path]] = None
    for meta_path in out_root.glob("*/meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if str(meta.get("fingerprint", "")) != want:
            continue
        if int(meta.get("chart_window_bars", -1)) != window:
            continue
        if int(meta.get("indices_start", 0)) != 0:
            continue
        old_last = int(meta.get("indices_last", -1))
        old_hash = str(meta.get("covered_prefix_hash", ""))
        if old_last < 0 or old_last >= n_new - 1 or not old_hash:
            continue
        if old_last + window > len(df):
            continue
        if covered_prefix_hash(df, window, old_last) != old_hash:
            continue
        if best is None or old_last > best[0]:
            best = (old_last, meta_path.parent)
    if best is None:
        return None
    return best[1], best[0] + 1  # reused_rows = old_last + 1


def _build_full(df, window, spec, n_, artifact_dir, workers, chunk_size, full_frame_hash, key, progress_every, engine="cpu"):
    engine = _resolve_engine(df, window, spec, n_, engine)
    tmp = _tmp_dir(artifact_dir.parent)
    try:
        total = _render_into(tmp, df, window, spec, 0, n_, workers, chunk_size, progress_every, engine)
        _finalize(tmp, artifact_dir, df, window, spec, n_, full_frame_hash, key, engine, workers)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return CompiledChartStore.open(artifact_dir), RenderJobStats(
        n_rows=n_, workers=workers, chunk_size=chunk_size, rendered_rows=total
    )


def _build_with_reuse(df, window, spec, n_, artifact_dir, reuse, workers, chunk_size, full_frame_hash, key, progress_every, engine="cpu"):
    old_dir, reused_rows = reuse
    tmp = _tmp_dir(artifact_dir.parent)
    try:
        try:
            old_store = CompiledChartStore.open(old_dir)
        except Exception:
            old_store = None
        if old_store is None or len(old_store) < reused_rows:
            shutil.rmtree(tmp, ignore_errors=True)
            return _build_full(df, window, spec, n_, artifact_dir, workers, chunk_size, full_frame_hash, key, progress_every, engine)
        # Copy the verified-identical prefix bytes (no re-render), then render the tail.
        bin_path = tmp / "charts.bin"
        image_size = int(spec.size)
        arr = np.memmap(bin_path, dtype="float32", mode="w+",
                        shape=(n_, 3, image_size, image_size))
        try:
            arr[:reused_rows] = old_store.arr[:reused_rows]
            arr.flush()
        finally:
            del arr
        tail = _render_into(tmp, df, window, spec, reused_rows, n_, workers, chunk_size, progress_every, engine, create=False)
        _finalize(tmp, artifact_dir, df, window, spec, n_, full_frame_hash, key, engine, workers)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return CompiledChartStore.open(artifact_dir), RenderJobStats(
        n_rows=n_, workers=workers, chunk_size=chunk_size,
        reused_prefix_rows=reused_rows, rendered_rows=tail,
    )


def _render_into(tmp, df, window, spec, start, end, workers, chunk_size, progress_every, engine="cpu", create: bool = True) -> int:
    if engine == "gpu":
        return _render_into_gpu(tmp, df, window, spec, start, end, chunk_size)
    bin_path = str(tmp / "charts.bin")
    image_size = int(spec.size)
    total = int(end)
    ohlcv = df[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64)
    if create:
        # Pre-allocate the file at its final shape so workers can open it 'r+'.
        mem = np.memmap(bin_path, dtype="float32", mode="w+",
                        shape=(total, 3, image_size, image_size))
        del mem

    chunks = list(_chunk_ranges(start, end, chunk_size))
    spec_dict = spec.to_meta()
    if workers <= 1:
        done = 0
        for cs, ce in chunks:
            done += _render_rows_task(ohlcv, window, spec_dict, cs, ce, bin_path, image_size, total)
        return done
    n_workers = min(int(workers), len(chunks))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [
            ex.submit(_render_rows_task, ohlcv, window, spec_dict, cs, ce, bin_path, image_size, total)
            for cs, ce in chunks
        ]
        return sum(f.result() for f in futs)


def _chunk_ranges(start: int, end: int, chunk: int):
    s = int(start)
    while s < int(end):
        e = min(s + max(int(chunk), 1), int(end))
        yield s, e
        s = e


def _tmp_dir(out_root: Path) -> Path:
    d = out_root / f".job_tmp_{uuid.uuid4().hex}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def _finalize(tmp, artifact_dir, df, window, spec, n_, full_frame_hash, key, engine="cpu", workers=0):
    idx = range(n_)
    last_index = n_ - 1
    covered_hash = covered_prefix_hash(df, window, last_index) if n_ > 0 else ""
    meta = CompiledChartStore._make_meta(
        full_frame_hash, key, spec, n_, int(spec.size), int(window), idx, covered_hash, engine=engine
    )
    meta["render_machine"] = None  # intentional: keeps meta schema identical to serial builds
    from zhisa.data.chart_store import checksummed_meta
    meta = checksummed_meta(meta, tmp / "charts.bin")
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir, ignore_errors=True)
    os.replace(tmp, artifact_dir)

def _machine_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return f"{torch.cuda.get_device_name(0)}"
    except Exception:
        pass
    return "cpu"


def _resolve_engine(df, window, spec, n_, engine: str) -> str:
    """Gate the GPU engine against the CPU canonical on a corpus; fallback on fail."""
    if engine != "gpu":
        return engine
    from zhisa.rendering.gpu import gpu_device
    if gpu_device() is None:
        print("render engine 'gpu' requested but no CUDA device ? falling back to cpu")
        return "cpu"
    n_gate = max(1, min(int(n_), 12))
    step = max(1, int(n_) // n_gate)
    ohlcv = df[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64)
    corpus = np.stack(
        [ohlcv[max(0, i): max(0, i) + int(window)] for i in range(0, int(n_), step)]
    )
    from zhisa.rendering.gpu import validate_gpu_against_cpu
    v = validate_gpu_against_cpu(corpus, spec, atol=1e-4)
    print(
        f"render engine parity gate: ok={v.ok} maxdiff={v.max_abs_diff:.2e} "
        f"n_diff={v.n_diff_pixels} device={v.device}"
    )
    if not v.ok:
        print("GPU parity gate FAILED ? falling back to cpu")
        return "cpu"
    return "gpu"


def _render_into_gpu(tmp, df, window, spec, start, end, chunk_size) -> int:
    """GPU batch materialisation: render rows [start, end) into the memmap."""
    from zhisa.rendering.gpu import gpu_device, render_batch_gpu
    bin_path = str(tmp / "charts.bin")
    image_size = int(spec.size)
    total = int(end)
    ohlcv = df[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64)
    mem = np.memmap(bin_path, dtype="float32", mode="w+" if start == 0 else "r+",
                    shape=(total, 3, image_size, image_size))
    del mem
    dev = gpu_device()
    S = int(spec.size) * max(int(spec.supersample), 1)
    budget = max(1, int(1.5e9 // max(S * S * 3 * 8, 1)))
    gpu_batch = max(1, min(max(int(chunk_size), 1), budget))
    arr = np.memmap(bin_path, dtype="float32", mode="r+",
                    shape=(total, 3, image_size, image_size))
    try:
        r = int(start)
        while r < int(end):
            hi = min(r + gpu_batch, int(end))
            wins = np.stack([ohlcv[i:i + int(window)].copy() for i in range(r, hi)])
            out = render_batch_gpu(wins, spec=spec, device=dev)
            arr[r:hi] = out
            r = hi
        arr.flush()
    finally:
        del arr
    return int(end) - int(start)
