"""Tests for the compiled render job (parallel + incremental materialisation)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zhisa.data.chart_store import CompiledChartStore
from zhisa.data.render_job import materialize_parallel
from zhisa.rendering.goldens import golden_fixture
from zhisa.rendering.spec import RenderSpec


def _frame(n_bars: int = 400) -> pd.DataFrame:
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


def _sizes():
    return 280, 32, 32  # n, window, image


def test_parallel_matches_serial_build():
    df = _frame(400)
    n, window, img = _sizes()
    spec = RenderSpec(size=img, supersample=2)
    job_store, _ = materialize_parallel(df, window=window, spec=spec, n=n, workers=2, chunk_size=40)
    serial = CompiledChartStore.build(df, window=window, spec=spec, indices=range(n))
    assert job_store.render_checksum(full=True) == serial.render_checksum(full=True)
    assert job_store.is_byte_equivalent(serial, full=True)


def test_workers_dont_change_bytes():
    df = _frame(400)
    n, window, img = _sizes()
    spec = RenderSpec(size=img)
    a, _ = materialize_parallel(df, window=window, spec=spec, n=n, workers=1, chunk_size=50)
    b, _ = materialize_parallel(df, window=window, spec=spec, n=n, workers=3, chunk_size=30)
    assert a.render_checksum(full=True) == b.render_checksum(full=True)


def test_exact_cache_reuse(tmp_path):
    df = _frame(400)
    n, window, img = _sizes()
    spec = RenderSpec(size=img)
    s1, st1 = materialize_parallel(df, window=window, spec=spec, n=n, out_root=tmp_path, workers=2, chunk_size=40)
    s2, st2 = materialize_parallel(df, window=window, spec=spec, n=n, out_root=tmp_path, workers=2, chunk_size=40)
    assert st1.rendered_rows == n
    assert st1.reused_artifact is False
    assert st2.reused_artifact is True
    assert st2.rendered_rows == 0
    assert s1.render_checksum(full=True) == s2.render_checksum(full=True)


def test_incremental_reuse_after_data_growth(tmp_path):
    df = _frame(400)          # initial frame
    n0, window, img = 200, 32, 32
    spec = RenderSpec(size=img)
    old, st_old = materialize_parallel(df, window=window, spec=spec, n=n0, out_root=tmp_path, workers=2, chunk_size=60)
    assert st_old.rendered_rows == n0

    # Frame grows (new bars appended): simulate by a longer frame whose prefix
    # is character-identical.
    grown = _frame(400)
    grown2 = grown.copy()
    n1 = 300
    new, st_new = materialize_parallel(grown2, window=window, spec=spec, n=n1, out_root=tmp_path, workers=2, chunk_size=60)

    # The prefix rows must be reused, only the tail re-rendered.
    assert st_new.reused_prefix_rows == n0
    assert st_new.rendered_rows == (n1 - n0)
    assert st_new.reused_artifact is False

    # And the final artefact is byte-identical to what a fresh full build gives.
    fresh = CompiledChartStore.build(grown2, window=window, spec=spec, indices=range(n1))
    assert new.render_checksum(full=True) == fresh.render_checksum(full=True)
    # Prefix manually preserved.
    assert np.array_equal(new[:n0].copy(), old[:n0].copy())


def test_incremental_refuses_reuse_when_identity_differs(tmp_path):
    df = _frame(400)
    n0, window, img = 150, 32, 32
    materialize_parallel(df, window=window, spec=RenderSpec(size=img), n=n0, out_root=tmp_path)
    # Different window size -> must NOT reuse.
    _, st = materialize_parallel(
        _frame(400), window=64, spec=RenderSpec(size=img), n=220, out_root=tmp_path, workers=2
    )
    assert st.reused_prefix_rows == 0
    assert st.rendered_rows == 220


def test_incremental_rejects_corrupted_prefix_hash(tmp_path):
    """A frame whose prefix differs from the stored covered-hash must not reuse."""
    df = _frame(300)
    n0, window, img = 120, 16, 16
    materialize_parallel(df, window=window, spec=RenderSpec(size=img), n=n0, out_root=tmp_path)
    mu = df.copy()
    mu.iloc[5, 2] = mu.iloc[5, 2] * 1.5  # tamper inside the covered prefix
    _, st = materialize_parallel(mu, window=window, spec=RenderSpec(size=img), n=200, out_root=tmp_path, workers=2)
    assert st.reused_prefix_rows == 0  # must fall back to a full build
    assert st.rendered_rows == 200


def test_atomic_partial_build_never_visible(tmp_path):
    """A crashed job leaves only the temp dir; a later run rebuilds cleanly."""
    df = _frame(400)
    n, window, img = 250, 32, 32
    spec = RenderSpec(size=img)
    s, _ = materialize_parallel(df, window=window, spec=spec, n=n, out_root=tmp_path, workers=1, chunk_size=500)
    # Simulate an interrupted leftover: junk temp dir + a partial bin under a wrong key.
    (tmp_path / ".job_tmp_deadbeef").mkdir()
    (tmp_path / ".job_tmp_deadbeef" / "charts.bin").write_bytes(b"\x00" * 16)
    s2, st2 = materialize_parallel(df, window=window, spec=spec, n=n, out_root=tmp_path, workers=1)
    assert st2.reused_artifact is True  # the exact artefact still exists and wins
    assert s.render_checksum(full=True) == s2.render_checksum(full=True)


def test_in_memory_fallback(tmp_path):
    df = _frame(200)
    n, window, img = 120, 32, 32
    spec = RenderSpec(size=img)
    store, st = materialize_parallel(df, window=window, spec=spec, n=n, out_root=None)
    assert st.reused_artifact is False
    assert len(store) == n
    assert store[0].shape == (3, img, img)