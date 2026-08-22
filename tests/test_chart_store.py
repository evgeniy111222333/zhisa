"""Tests for the content-addressed compiled chart store."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from zhisa.data.chart_store import (
    CompiledChartStore,
    content_key,
    frame_checksum,
    render_checksum,
)
from zhisa.rendering.chart_renderer import render_fingerprint
from zhisa.rendering.goldens import golden_fixture
from zhisa.rendering.spec import RenderSpec


def _frame(n_bars: int = 256) -> pd.DataFrame:
    fx = golden_fixture(n_bars)
    return pd.DataFrame(
        {
            "open": fx[:, 0],
            "high": fx[:, 1],
            "low": fx[:, 2],
            "close": fx[:, 3],
            "volume": fx[:, 4],
            "timestamp": pd.date_range("2024-01-01", periods=n_bars, freq="5min"),
        }
    ).set_index("timestamp")


def test_frame_checksum_is_stable():
    df = _frame()
    assert frame_checksum(df) == frame_checksum(df)
    df2 = df.copy()
    df2.loc[df2.index[0], "close"] += 1.0
    assert frame_checksum(df) != frame_checksum(df2)


def test_content_key_depends_on_all_identity_parts():
    df = _frame()
    spec = RenderSpec(size=32)
    k1 = content_key(spec, 32, frame_checksum(df), None, 200)
    assert k1 == content_key(spec, 32, frame_checksum(df), None, 200)
    assert k1 != content_key(RenderSpec(size=64), 32, frame_checksum(df), None, 200)
    assert k1 != content_key(spec, 64, frame_checksum(df), None, 200)
    assert k1 != content_key(spec, 32, frame_checksum(df), None, 100)


def test_build_memmap_and_read(tmp_path):
    df = _frame()
    spec = RenderSpec(size=32)
    store = CompiledChartStore.build(df, window=32, spec=spec, indices=range(120), out_root=tmp_path)
    assert len(store) == 120
    assert store[0].shape == (3, 32, 32)
    assert store[0].dtype == np.float32
    assert store.render_meta["n_images"] == 120
    assert store.render_meta["spec_hash"] == spec.content_hash()
    assert store.verify_input(df, 32, spec, range(120))


def test_cache_reuse(tmp_path):
    df = _frame()
    spec = RenderSpec(size=32)
    a = CompiledChartStore.build(df, window=32, spec=spec, indices=range(100), out_root=tmp_path)
    b = CompiledChartStore.build(df, window=32, spec=spec, indices=range(100), out_root=tmp_path)
    assert a.meta["content_key"] == b.meta["content_key"]
    assert len(list(tmp_path.glob("*/charts.bin"))) == 1  # only one artefact


def test_different_spec_is_different_artefact(tmp_path):
    df = _frame()
    a = CompiledChartStore.build(df, window=32, spec=RenderSpec(size=32), indices=range(50), out_root=tmp_path)
    b = CompiledChartStore.build(df, window=32, spec=RenderSpec(size=64), indices=range(50), out_root=tmp_path)
    assert a.meta["content_key"] != b.meta["content_key"]
    assert len(list(tmp_path.glob("*/charts.bin"))) == 2


def test_byte_equivalence_disk_vs_memory():
    df = _frame()
    spec = RenderSpec(size=32)
    disk = CompiledChartStore.build(df, window=32, spec=spec, indices=range(80), out_root=None)
    mem = CompiledChartStore.build(df, window=32, spec=spec, indices=range(80), out_root=None)
    assert disk.render_checksum() == mem.render_checksum()
    assert disk.is_byte_equivalent(mem)
    # same spec + same frame + same indices -> identical pixels
    for i in (0, 40, 79):
        assert np.array_equal(disk[i], mem[i])


def test_render_matches_canonical_renderer_directly():
    df = _frame()
    spec = RenderSpec(size=32)
    store = CompiledChartStore.build(df, window=32, spec=spec, indices=range(10), out_root=None)
    from zhisa.rendering.chart_renderer import render_ohlcv
    ohlcv = df[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64)
    for t in (0, 5, 9):
        expected = render_ohlcv(ohlcv[t:t + 32], spec=spec)
        got = store.to_tensor(t)
        assert torch_allclose(expected, got)


def test_out_of_range_index_raises(tmp_path):
    df = _frame()
    store = CompiledChartStore.build(df, window=32, spec=RenderSpec(size=32), indices=range(5), out_root=tmp_path)
    with pytest.raises(IndexError):
        _ = store[5]
    with pytest.raises(IndexError):
        _ = store[-6]


def test_verify_input_rejects_different_frame_or_window(tmp_path):
    df = _frame()
    spec = RenderSpec(size=32)
    store = CompiledChartStore.build(df, window=32, spec=spec, indices=range(30), out_root=tmp_path)
    assert store.verify_input(df, 32, spec, range(30)) is True
    assert store.verify_input(df, 64, spec, range(30)) is False
    df2 = _frame(300)
    assert store.verify_input(df2, 32, spec, range(30)) is False


def test_open_roundtrip(tmp_path):
    df = _frame()
    spec = RenderSpec(size=32)
    CompiledChartStore.build(df, window=32, spec=spec, indices=range(40), out_root=tmp_path)
    artifact_dir = next(tmp_path.glob("*/charts.bin")).parent
    opened = CompiledChartStore.open(artifact_dir)
    rebuilt = CompiledChartStore.build(df, window=32, spec=spec, indices=range(40), out_root=tmp_path)
    assert opened.is_byte_equivalent(rebuilt)
    # Full-contract checksum is the strictest guarantee.
    assert opened.render_checksum(full=True) == rebuilt.render_checksum(full=True)


def test_render_checksum_sample_vs_full_consistent_for_identical(tmp_path):
    df = _frame()
    spec = RenderSpec(size=32)
    a = CompiledChartStore.build(df, window=32, spec=spec, indices=range(40), out_root=tmp_path)
    b = CompiledChartStore.build(df, window=32, spec=spec, indices=range(40), out_root=tmp_path)
    assert a.render_checksum(full=False) == b.render_checksum(full=False)
    assert a.render_checksum(full=True) == b.render_checksum(full=True)


def torch_allclose(a, b):
    import torch
    return torch.equal(torch.as_tensor(a), torch.as_tensor(b))


def test_open_detects_corrupted_bytes(tmp_path):
    df = _frame()
    spec = RenderSpec(size=32)
    CompiledChartStore.build(df, window=32, spec=spec, indices=range(40), out_root=tmp_path)
    artifact = next(tmp_path.glob("*/charts.bin")).parent
    meta = json.loads((artifact / "meta.json").read_text(encoding="utf-8"))
    raw = np.fromfile(artifact / "charts.bin", dtype=np.float32)
    shape = (int(meta["n_images"]), 3, int(meta["image_size"]), int(meta["image_size"]))
    arr = raw.reshape(shape)

    ok = CompiledChartStore(path=None, arr=arr, meta=meta)
    assert ok.verify_integrity(full=True) is True
    assert ok.verify_integrity(full=False) is True

    corrupted = arr.copy()
    corrupted[0, 0, 0, 0] += 1.0  # flip a sampled row's pixel
    bad = CompiledChartStore(path=None, arr=corrupted, meta=meta)
    assert bad.verify_integrity(full=True) is False
    assert bad.verify_integrity(full=False) is False

    # Meta tampering (fingerprint changed) also invalidates the contract.
    tampered_meta = dict(meta)
    tampered_meta["fingerprint"] = "0" * 64
    bad2 = CompiledChartStore(path=None, arr=arr, meta=tampered_meta)
    assert bad2.verify_integrity(full=False) is False


def test_open_detects_truncated_or_appended_file(tmp_path):
    df = _frame()
    spec = RenderSpec(size=32)
    CompiledChartStore.build(df, window=32, spec=spec, indices=range(40), out_root=tmp_path)
    artifact = next(tmp_path.glob("*/charts.bin")).parent
    bin_path = artifact / "charts.bin"
    with open(bin_path, "ab") as f:   # append junk -> size mismatch
        f.write(b"\x00" * 64)
    store = CompiledChartStore.open(artifact, verify=False)
    assert store.verify_integrity() is False
    with pytest.raises(RuntimeError):
        CompiledChartStore.open(artifact, verify=True)