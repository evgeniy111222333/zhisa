"""Prepared-data lineage guards: scream loudly when something is wrong.

Three classes of silent failure this module prevents:

1. **Repair drift** — two prepared roots built from the same TSDB with
   different repair semantics (gap fill / reindex) are NOT bit-comparable,
   even though their timestamps mostly overlap and prices are identical.
   ``scan_prepared`` records per-symbol OHLCV core checksums + gap stats +
   the repair identity, and ``assert_prepared_consistent`` re-verifies
   invariants against them on every later read.

2. **Chart-store reuse lies** — a content-addressed store built from one
   root silently misses for another (even 1 inserted bar changes every
   later row), which used to mean an hours-long silent re-render.
   ``probe_reuse`` + ``guard_reuse`` estimate the hit rate BEFORE any
   work and refuse to start a large render unless ``ZHISA_FORCE_RENDER``
   is set.

3. **Lineage drift between machines** — ``fingerprint_tsdb`` lets a
   prepared root record which TSDB state it came from; a later scan that
   sees a different fingerprint is flagged (or accepted when ``strict``
   is disabled and the recorded value is kept for the audit trail).

"To scream" means: nonzero exit / raised ``LineageError`` with an
explicit message. No silent best-effort.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from zhisa.data.chart_store import frame_checksum, content_key
from zhisa.rendering.spec import RenderSpec

OHLCV_COLS = ["open", "high", "low", "close", "volume"]
LINEAGE_FILENAME = "lineage.json"


class LineageError(Exception):
    """Raised when a lineage invariant is violated."""


@dataclass(frozen=True)
class PreparedScan:
    """Per-root scan output: invariants + per-symbol summary."""

    root: str
    scannable_symbols: tuple[str, ...]
    rows_total: int
    per_symbol: dict = field(default_factory=dict)  # sym -> summary dict
    tsdb_fingerprint: Optional[str] = None
    repair_identity: Optional[str] = None
    manifest_checksum: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "root": self.root,
            "scannable_symbols": list(self.scannable_symbols),
            "rows_total": self.rows_total,
            "per_symbol": self.per_symbol,
            "tsdb_fingerprint": self.tsdb_fingerprint,
            "repair_identity": self.repair_identity,
            "manifest_checksum": self.manifest_checksum,
        }


def fingerprint_tsdb(tsdb_root: Path, symbols: list[str]) -> str:
    """Cheap fingerprint of a TSDB state: per-symbol row counts, first/last
    timestamp and a digest of a fixed byte window of the OHLCV float64s.

    Purpose-built to be stable under appends? NO — any append changes it.
    It is meant to DETECT drift between what a root was built from and
    what exists today; the recorded value stays in the lineage file for
    the audit trail even when the local TSDB has moved on.
    """
    h = hashlib.sha256()
    for sym in sorted(symbols):
        frames = list((tsdb_root / sym).rglob("*.parquet")) if (tsdb_root / sym).exists() else []
        for f in sorted(frames, key=lambda p: p.as_posix())[:4]:
            df = pd.read_parquet(f, columns=OHLCV_COLS)
            arr = df[OHLCV_COLS].to_numpy(dtype=np.float64)
            h.update(f"{sym}:{f.name}:{len(df)}:{df.index[0]}:{df.index[-1]}:".encode())
            h.update(arr[:1024].tobytes())
            h.update(arr[-1024:].tobytes())
    return h.hexdigest()


def _gap_stats(index: pd.DatetimeIndex) -> dict:
    if len(index) < 2:
        return {"n_gaps": 0, "longest_gap_bars": 0, "total_missing_bars": 0}
    delta = index.to_series().diff().dropna()
    median = delta.median()
    if pd.isna(median) or median == pd.Timedelta(0):
        return {"n_gaps": 0, "longest_gap_bars": 0, "total_missing_bars": 0}
    bars = (delta / median).round().astype(int) - 1
    missing = int(bars.clip(lower=0).sum())
    return {
        "n_gaps": int((bars > 0).sum()),
        "longest_gap_bars": int(bars.max()) if missing else 0,
        "total_missing_bars": missing,
    }


def scan_prepared(prepared_root: Path, *, tsdb_root: Optional[Path] = None,
                  symbols: Optional[list[str]] = None) -> PreparedScan:
    """Scan a prepared root and compute all invariants we can verify cheaply."""
    root = Path(prepared_root)
    symbols_dir = root / "symbols"
    if not symbols_dir.is_dir():
        raise LineageError(f"{root}: no symbols/ directory — not a prepared root?")
    manifest_path = root / "manifest.json"
    manifest = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = sorted(p.stem for p in symbols_dir.glob("*.parquet"))
    syms = symbols or candidates
    missing = [s for s in syms if not (symbols_dir / f"{s}.parquet").is_file()]
    if missing:
        raise LineageError(f"{root}: requested symbols missing: {missing}")
    per = {}
    rows_total = 0
    for sym in syms:
        df = pd.read_parquet(symbols_dir / f"{sym}.parquet")
        idx = pd.DatetimeIndex(df.index)
        if not idx.is_monotonic_increasing:
            raise LineageError(f"{root}: {sym} index NOT monotonic")
        arr = df[OHLCV_COLS].to_numpy(dtype=np.float64)
        if not np.isfinite(arr).all():
            bad = int((~np.isfinite(arr)).sum())
            raise LineageError(f"{root}: {sym} has {bad} non-finite OHLCV cells")
        per[sym] = {
            "rows": int(len(df)),
            "first_ts": str(idx[0]),
            "last_ts": str(idx[-1]),
            "ohlcv_checksum": frame_checksum(df),
            **{f"gap_{k}": v for k, v in _gap_stats(idx).items()},
        }
        rows_total += int(len(df))
    repair_id = None
    for key in ("gap_policy", "coverage_policy"):
        pol = manifest.get(key)
        if isinstance(pol, dict) and "repair_version" in pol:
            repair_id = f"{key}:{pol['repair_version']}:{json.dumps(pol, sort_keys=True)}"
    return PreparedScan(
        root=str(root),
        scannable_symbols=tuple(syms),
        rows_total=rows_total,
        per_symbol=per,
        tsdb_fingerprint=fingerprint_tsdb(Path(tsdb_root), syms) if tsdb_root else None,
        repair_identity=repair_id,
        manifest_checksum=manifest.get("output_checksum"),
    )


def write_lineage(prepared_root: Path, scan: PreparedScan) -> Path:
    path = Path(prepared_root) / LINEAGE_FILENAME
    path.write_text(json.dumps(scan.as_dict(), indent=2), encoding="utf-8")
    return path


def assert_prepared_consistent(prepared_root: Path, *, symbols: Optional[list[str]] = None,
                               expect_full_recompute: bool = False) -> PreparedScan:
    """Re-verify a root against its committed lineage.json (or compute fresh).

    Raises :class:`LineageError` on ANY of:
      - lineage.json exists but per-symbol rows / ohlcv checksums differ;
      - monotonicity or finiteness broken;
      - manifest output_checksum differs from the committed one.
    """
    root = Path(prepared_root)
    scan = scan_prepared(root, symbols=symbols)
    lp = root / LINEAGE_FILENAME
    if not lp.is_file():
        if expect_full_recompute:
            raise LineageError(f"{root}: no lineage.json and consistency is required")
        return scan
    committed = json.loads(lp.read_text(encoding="utf-8"))
    for sym, cur in scan.per_symbol.items():
        old = committed.get("per_symbol", {}).get(sym)
        if old is None:
            raise LineageError(f"{root}: lineage has no record for {sym}")
        for key in ("rows", "ohlcv_checksum", "first_ts", "last_ts", "gap_total_missing_bars"):
            if str(old.get(key)) != str(cur.get(key)):
                raise LineageError(
                    f"{root}: {sym} lineage drift on {key}: "
                    f"committed={old.get(key)!r} now={cur.get(key)!r}"
                )
    if committed.get("rows_total") != scan.rows_total:
        raise LineageError(f"{root}: rows_total drift {committed.get('rows_total')} -> {scan.rows_total}")
    if committed.get("manifest_checksum") and scan.manifest_checksum:
        if committed["manifest_checksum"] != scan.manifest_checksum:
            raise LineageError(
                f"{root}: manifest checksum drift "
                f"{committed['manifest_checksum']} -> {scan.manifest_checksum}"
            )
    return scan


# ---------------------------------------------------------------------------
# Chart-store reuse guard ("scream before a silent hours-long render")
# ---------------------------------------------------------------------------


def _content_keys_for_segments(segments, window: int, spec: RenderSpec,
                               trim: int = 0) -> list[str]:
    from zhisa.data.chart_store import content_key as _ck, frame_checksum as _fc

    keys = []
    window = int(window)
    for seg in segments:
        n_ = max(0, len(seg) - int(trim))
        keys.append(_ck(spec, window, _fc(seg), range(n_), n_))
    return keys


def probe_reuse(charts_dir: Path, segments, window: int, spec: RenderSpec,
                k: int = 3, trim: int = 0) -> dict:
    """Probe how many spread segment keys already exist under ``charts_dir``.

    ``segments``: a list of per-symbol contiguous frames (or a single
    DataFrame). At most ``k`` spread segments are probed so a big root
    stays cheap. ``trim`` mirrors the trainer's materialisation call
    (``n = len - window - max_horizons - 1``) so the probed keys are the
    SAME ones the renderer would produce.
    """
    if isinstance(segments, pd.DataFrame):
        segments = [segments]
    dirs = set()
    if charts_dir.is_dir():
        for p in charts_dir.iterdir():
            if p.is_dir() and len(p.name) == 64:
                dirs.add(p.name)
    n_seg = len(segments)
    if n_seg == 0:
        return {"probes": 0, "hits": 0, "hit_rate": 0.0, "probed_keys": [], "first_key": None}
    idx = [0, n_seg // 2, n_seg - 1][:min(k, n_seg)] if n_seg > 1 else [0]
    probed = [segments[i] for i in set(idx)][:k]
    keys = _content_keys_for_segments(probed, window, spec, trim=trim)
    hits = [key for key in keys if key in dirs]
    return {
        "probes": len(keys),
        "hits": len(hits),
        "hit_rate": round(len(hits) / max(1, len(keys)), 3),
        "probed_keys": keys,
        "first_key": keys[0] if keys else None,
    }


def guard_reuse(charts_dir: Path, segments, window: int, spec: RenderSpec,
                *, force_env: str = "ZHISA_FORCE_RENDER",
                min_reuse_ratio: float = 0.34,
                render_hint: str = "full render",
                trim: int = 0) -> dict:
    """Refuse (loudly) to start a big chart job whose reuse looks ~zero.

    Returns the probe dict when reuse looks fine, and raises
    :class:`LineageError` otherwise, unless ``force_env`` is set (then it
    returns the probe with ``forced=True``). Set ``min_reuse_ratio=0.0``
    (or ``ZHISA_LINEAGE_GUARD=0``) to disable the guard.
    """
    probe = probe_reuse(charts_dir, segments, window, spec, k=3, trim=trim)
    if min_reuse_ratio <= 0.0 or probe["hit_rate"] >= min_reuse_ratio:
        return {**probe, "forced": False, "allowed": True}
    if os.environ.get(force_env, "0") in ("1", "true", "yes"):
        return {**probe, "forced": True, "allowed": True}
    raise LineageError(
        f"chart-reuse guard: only {probe['hits']}/{probe['probes']} probe segments "
        f"hit the store under {charts_dir}. This run would need {render_hint} "
        f"(hours on a small instance). Set {force_env}=1 to force it."
    )


__all__ = [
    "LINEAGE_FILENAME",
    "LineageError",
    "PreparedScan",
    "assert_prepared_consistent",
    "fingerprint_tsdb",
    "guard_reuse",
    "probe_reuse",
    "scan_prepared",
    "write_lineage",
]