"""Tests for the parity-checked GPU chart rasterizer and engine integration."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from zhisa.rendering.gpu import (
    GPU_ENGINE_NAME,
    gpu_device,
    render_batch_gpu,
    render_ohlcv_gpu,
    validate_gpu_against_cpu,
)
from zhisa.rendering.goldens import golden_fixture
from zhisa.rendering.spec import RenderSpec
from zhisa.data.render_job import materialize_parallel

_HAS_CUDA = torch.cuda.is_available()


pytestmark = [
    pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required"),
    pytest.mark.gpu,
]


def _corpus(B: int = 4, N: int = 128):
    return np.stack([golden_fixture(N) for _ in range(B)], axis=0)


def test_render_batch_shape_and_range():
    spec = RenderSpec(size=64, supersample=2)
    img = render_batch_gpu(_corpus(), spec=spec)
    assert img.shape == (4, 3, 64, 64)
    assert img.dtype == np.float32
    assert (img >= 0.0).all() and (img <= 1.0).all()


def test_deterministic_same_gpu():
    corpus = _corpus()
    for ss in (1, 2, 4):
        spec = RenderSpec(size=64, supersample=ss)
        a = render_batch_gpu(corpus, spec=spec)
        b = render_batch_gpu(corpus, spec=spec)
        assert np.array_equal(a, b), ss


def test_single_equals_batch():
    corpus = _corpus(3, 96)
    spec = RenderSpec(size=48, supersample=2)
    batch = render_batch_gpu(corpus, spec=spec)
    for i in range(3):
        single = render_ohlcv_gpu(corpus[i], spec=spec)
        assert np.array_equal(single, batch[i])


@pytest.mark.parametrize("ss", [1, 2, 4])
@pytest.mark.parametrize("size", [32, 64])
def test_parity_with_cpu_canonical(ss, size):
    corpus = _corpus(3, 128)
    spec = RenderSpec(size=size, supersample=ss)
    v = validate_gpu_against_cpu(corpus, spec, atol=1e-5)
    assert v.ok, f"maxdiff={v.max_abs_diff:.3e} n_diff={v.n_diff_pixels}"
    assert v.n_diff_pixels == 0


def test_parity_variants(include_volume=True, include_overlays=True):
    spec = RenderSpec(
        size=64, supersample=4,
        include_volume=include_volume, include_overlays=include_overlays,
    )
    v = validate_gpu_against_cpu(_corpus(3, 128), spec, atol=1e-5)
    assert v.ok


def test_non_finite_rejected():
    bad = _corpus(2, 64)
    bad[1, 10, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        render_batch_gpu(bad, spec=RenderSpec(size=32))


def test_validate_report_fields():
    v = validate_gpu_against_cpu(_corpus(2, 96), RenderSpec(size=48))
    assert v.ok
    assert v.max_abs_diff <= 1e-5
    assert v.total_pixels > 0
    assert v.device.startswith("cuda")


def test_render_job_engine_gpu_materializes(tmp_path):
    fx = golden_fixture(240)
    df = pd.DataFrame(
        {
            "open": fx[:, 0], "high": fx[:, 1], "low": fx[:, 2],
            "close": fx[:, 3], "volume": fx[:, 4],
            "timestamp": pd.date_range("2024-01-01", periods=240, freq="5min"),
        }
    ).set_index("timestamp")
    spec = RenderSpec(size=32, supersample=2)
    store_gpu, st_gpu = materialize_parallel(
        df, window=32, spec=spec, n=150, out_root=tmp_path / "gpu",
        workers=0, chunk_size=64, engine="gpu",
    )
    assert store_gpu.render_meta["engine"] == "gpu"
    assert st_gpu.rendered_rows == 150
    # deterministic across two gpu builds -> identical bytes
    store_gpu2, _ = materialize_parallel(
        df, window=32, spec=spec, n=150, out_root=tmp_path / "gpu",
        workers=0, chunk_size=64, engine="gpu",
    )
    assert store_gpu.render_checksum(full=True) == store_gpu2.render_checksum(full=True)
    # parity vs cpu build on sampled rows
    cpu = materialize_parallel(
        df, window=32, spec=spec, n=150, out_root=tmp_path / "cpu",
        workers=0, chunk_size=64, engine="cpu",
    )[0]
    for i in (0, 50, 149):
        d = np.abs(store_gpu[i].astype(float) - cpu[i].astype(float)).max()
        assert d <= 1e-4, (i, d)
    assert cpu.render_meta["engine"] == "cpu"


def test_engine_gate_falls_back_without_cuda(tmp_path, monkeypatch):
    import zhisa.rendering.gpu as gmod
    monkeypatch.setattr(gmod, "gpu_device", lambda *a, **k: None)
    fx = golden_fixture(160)
    df = pd.DataFrame(
        {
            "open": fx[:, 0], "high": fx[:, 1], "low": fx[:, 2],
            "close": fx[:, 3], "volume": fx[:, 4],
            "timestamp": pd.date_range("2024-01-01", periods=160, freq="5min"),
        }
    ).set_index("timestamp")
    spec = RenderSpec(size=32)
    store, st = materialize_parallel(
        df, window=32, spec=spec, n=100, out_root=tmp_path, engine="gpu"
    )
    assert store.render_meta["engine"] == "cpu"  # resolved to cpu fallback
    assert GPU_ENGINE_NAME  # import sanity


def test_gpu_engine_name_constants():
    assert isinstance(GPU_ENGINE_NAME, str) and GPU_ENGINE_NAME