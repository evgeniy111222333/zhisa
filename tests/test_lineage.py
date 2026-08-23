"""Tests for the prepared-data lineage guards (they scream, loudly)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zhisa.data.lineage import (
    LineageError,
    assert_prepared_consistent,
    fingerprint_tsdb,
    guard_reuse,
    probe_reuse,
    scan_prepared,
    write_lineage,
)
from zhisa.rendering.spec import RenderSpec
from zhisa.data.chart_store import frame_checksum


def _frame(n=300, seed=0, gap_at=None):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
    df = pd.DataFrame({"open": close, "high": close, "low": close,
                       "close": close, "volume": rng.uniform(1, 2, n)}, index=idx)
    if gap_at is not None:
        df = df.drop(index=idx[gap_at:gap_at + 3])
    return df


def _make_root(path: Path, frames: dict[str, pd.DataFrame], manifest_checksum="abc123"):
    (path / "symbols").mkdir(parents=True)
    for sym, df in frames.items():
        df.to_parquet(path / "symbols" / f"{sym}.parquet")
    (path / "manifest.json").write_text(json.dumps({
        "output_checksum": manifest_checksum,
        "gap_policy": {"max_ffill_bars": 4, "drop_long_gaps": True,
                       "require_monotonic": True, "repair_version": "repair-v1"},
    }))


def test_scan_prepared_invariants_and_lineage_roundtrip(tmp_path):
    frames = {"A": _frame(seed=1), "B": _frame(seed=2, gap_at=50)}
    root = tmp_path / "root"
    _make_root(root, frames)
    scan = scan_prepared(root)
    assert scan.rows_total == len(frames["A"]) + len(frames["B"])  # 300 + 297
    assert scan.per_symbol["A"]["rows"] == 300
    assert scan.per_symbol["B"]["gap_total_missing_bars"] == 3
    assert scan.repair_identity and "repair-v1" in scan.repair_identity
    lp = write_lineage(root, scan)
    assert lp.is_file()
    again = assert_prepared_consistent(root)
    assert again.rows_total == scan.rows_total


def test_lineage_screams_on_row_tamper(tmp_path):
    frames = {"A": _frame(seed=1)}
    root = tmp_path / "root"
    _make_root(root, frames)
    scan = write_lineage(root, scan_prepared(root))
    # tamper: change one volume value -> checksum change
    df = pd.read_parquet(root / "symbols" / "A.parquet")
    df.iloc[10, -1] = df.iloc[10, -1] + 1.0
    df.to_parquet(root / "symbols" / "A.parquet")
    with pytest.raises(LineageError, match="ohlcv_checksum"):
        assert_prepared_consistent(root)


def test_lineage_screams_on_row_count_change(tmp_path):
    frames = {"A": _frame(seed=1)}
    root = tmp_path / "root"
    _make_root(root, frames)
    write_lineage(root, scan_prepared(root))
    df = pd.read_parquet(root / "symbols" / "A.parquet")
    new_idx = pd.date_range(df.index[-1] + pd.Timedelta(hours=1), periods=5,
                            freq="1h", tz="UTC")
    extra = _frame(seed=2, n=50).iloc[:5].copy()
    extra.index = new_idx
    pd.concat([df, extra]).to_parquet(root / "symbols" / "A.parquet")
    with pytest.raises(LineageError, match="rows"):
        assert_prepared_consistent(root)


def test_scan_screams_on_non_monotonic_and_nan(tmp_path):
    root = tmp_path / "root"
    (root / "symbols").mkdir(parents=True)
    df = _frame(seed=1).sample(frac=1.0, random_state=0)  # rows shuffled w/ index
    df.to_parquet(root / "symbols" / "A.parquet")
    with pytest.raises(LineageError, match="monotonic"):
        scan_prepared(root)
    (root / "symbols" / "A.parquet").unlink()
    df2 = _frame(seed=1)
    df2.iloc[5, -1] = np.nan
    df2.to_parquet(root / "symbols" / "A.parquet")
    with pytest.raises(LineageError, match="non-finite"):
        scan_prepared(root)


def test_tsdb_fingerprint_stable_and_drift_sensitive(tmp_path):
    tsdb = tmp_path / "tsdb"
    (tsdb / "A" / "1h").mkdir(parents=True)
    (tsdb / "B" / "1h").mkdir(parents=True)
    _frame(seed=1).to_parquet(tsdb / "A" / "1h" / "data.parquet")
    _frame(seed=2).to_parquet(tsdb / "B" / "1h" / "data.parquet")
    f1 = fingerprint_tsdb(tsdb, ["A", "B"])
    assert f1 == fingerprint_tsdb(tsdb, ["A", "B"])
    _frame(seed=1, n=301).to_parquet(tsdb / "A" / "1h" / "data.parquet")
    assert f1 != fingerprint_tsdb(tsdb, ["A", "B"])


def test_reuse_probe_and_guard_scream(tmp_path):
    charts = tmp_path / "charts"
    charts.mkdir()
    df = _frame(seed=1)
    spec = RenderSpec(size=32)
    # empty store -> no hits -> guard screams
    probe = probe_reuse(charts, [df], 32, spec)
    assert probe["hit_rate"] == 0.0
    with pytest.raises(LineageError, match="chart-reuse guard"):
        guard_reuse(charts, [df], 32, spec, min_reuse_ratio=0.34)
    # force env overrides
    os.environ["ZHISA_FORCE_RENDER"] = "1"
    try:
        res = guard_reuse(charts, [df], 32, spec, min_reuse_ratio=0.34)
        assert res["forced"] is True
    finally:
        del os.environ["ZHISA_FORCE_RENDER"]
    # min_ratio=0 disables
    res = guard_reuse(charts, [df], 32, spec, min_reuse_ratio=0.0)
    assert res["allowed"] is True
    # seed a real store dir with the expected key -> hit
    from zhisa.data.chart_store import content_key
    key = content_key(spec, 32, frame_checksum(df), range(len(df)), len(df))
    (charts / key).mkdir()
    probe2 = probe_reuse(charts, [df], 32, spec)
    assert probe2["hit_rate"] == 1.0


def test_repair_identity_changes_on_semantic_bump():
    from zhisa.data.feature_specs import GapPolicy
    a = GapPolicy().identity()
    b = GapPolicy(repair_version="repair-v2").identity()
    assert a != b
    assert "repair-v1" in a and "repair-v2" in b


def _base_root(path: Path) -> Path:
    """A minimal but realistic base prepared root (OHLCV + symbol splits)."""
    root = path / "base"
    (root / "symbols").mkdir(parents=True)
    (root / "splits").mkdir(parents=True)
    frames = {}
    for sym in ("A", "B"):
        df = _frame(seed=1 if sym == "A" else 2)
        df.to_parquet(root / "symbols" / f"{sym}.parquet")
        frames[sym] = df
    train = pd.concat(
        [frames["A"].iloc[:200].assign(symbol="A"),
         frames["B"].iloc[:200].assign(symbol="B")]
    ).sort_index()
    train.to_parquet(root / "splits" / "train.parquet")
    val = pd.concat(
        [frames["A"].iloc[200:250].assign(symbol="A"),
         frames["B"].iloc[200:250].assign(symbol="B")]
    ).sort_index()
    val.to_parquet(root / "splits" / "val.parquet")
    (root / "manifest.json").write_text(json.dumps({
        "version": "v1", "timeframe": "1h",
        "symbols": ["A", "B"],
        "rows_total": 600,
        "rows_per_symbol": {"A": 300, "B": 300},
        "feature_columns": ["open", "high", "low", "close", "volume"],
        "output_checksum": "base123",
    }))
    return root


def test_enrich_from_preserves_ohlcv_and_adds_columns(tmp_path):
    from zhisa.data.preparation import enrich_prepared_root
    base = _base_root(tmp_path)
    out = tmp_path / "v3_1"
    summary = enrich_prepared_root(base, out, with_regime_betas=True)
    assert summary["rows_total"] == 600
    assert summary["base_checksum"] == "base123"
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert m["derived_from"]["base_root"] == str(base)
    assert m["derived_from"]["ohclv_byte_identical"] is True
    for c in ("beta_up_64", "corr_stress_256", "breadth_64", "dispersion_256"):
        assert c in m["feature_columns"], c
    # OHLCV byte-identity per symbol
    ohlcv = ["open", "high", "low", "close", "volume"]
    for sym in ("A", "B"):
        a = pd.read_parquet(base / "symbols" / f"{sym}.parquet")[ohlcv].to_numpy(np.float64)
        b = pd.read_parquet(out / "symbols" / f"{sym}.parquet")[ohlcv].to_numpy(np.float64)
        assert np.array_equal(a, b)
        assert pd.read_parquet(out / "symbols" / f"{sym}.parquet").shape[1] > 5
    # splits rebuilt with new columns and full symbol coverage
    tr = pd.read_parquet(out / "splits" / "train.parquet")
    assert set(tr["symbol"].unique()) == {"A", "B"}
    assert "beta_256" in tr.columns
    assert len(tr) == 400
    # lineage file present and consistent
    assert_prepared_consistent(out)
    lp = json.loads((out / "lineage.json").read_text(encoding="utf-8"))
    assert lp["rows_total"] == 600


def test_enrich_from_screams_on_row_layout_change(tmp_path):
    """If the base root is mutated after its lineage was committed, the
    consistency check must scream before anything consumes it."""
    from zhisa.data.preparation import enrich_prepared_root
    base = _base_root(tmp_path)
    write_lineage(base, scan_prepared(base))
    df = pd.read_parquet(base / "symbols" / "A.parquet")
    df.iloc[10, -1] += 1.0
    df.to_parquet(base / "symbols" / "A.parquet")
    with pytest.raises(LineageError):
        assert_prepared_consistent(base)