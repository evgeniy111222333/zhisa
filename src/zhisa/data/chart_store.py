"""CompiledChartStore: content-addressed, memory-mapped chart artefacts.

The ideal split:

- **Build time** (one-off, offline): every chart window is rendered once with
  the canonical renderer and persisted to disk as a raw ``float32`` memmap
  plus a machine-readable ``meta.json``. This is the *compiled* dataset.
- **Train/inference time**: the DataLoader only *reads* the memmap — zero
  rendering, zero normalization, zero module imports for charts. The GPU is
  never starved by CPU rasterisation.

The store is **content-addressed**: the on-disk key is derived from
``(renderer version, RenderSpec hash, chart window, input frame checksum,
index range)``. Building the same content twice finds the existing artefact
and reuses it; changing any part of the identity creates a new artefact.
Two stores describe visually identical images if and only if their
fingerprint matches, and the *render contract* (see
:meth:`CompiledChartStore.render_checksum`) lets downstream stages assert
byte-equivalence before they are allowed to start.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from zhisa.rendering.chart_renderer import (
    CANONICAL_RENDERER_VERSION,
    _render_ohlcv_canonical,
    render_fingerprint,
)
from zhisa.rendering.spec import RenderSpec


def _sha256_binary(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def frame_checksum(df: pd.DataFrame) -> str:
    """Deterministic SHA-256 of the OHLCV float64 payload of a frame."""
    cols = ["open", "high", "low", "close", "volume"]
    arr = np.ascontiguousarray(df[cols].to_numpy(dtype=np.float64))
    return _sha256_binary(arr.tobytes(order="C"))


def covered_prefix_hash(df: pd.DataFrame, window: int, last_index: int, block: int = 4096) -> str:
    """Append-stable digest of the OHLCV rows a ``0..last_index`` window set sees.

    Instead of hashing the whole prefix linearly (O(rows) per refresh), the
    prefix is folded **block-by-block** (``block`` rows per SHA-256, chained
    with the previous digest — a Merkle-lite chain). Because blocks are
    fixed-size from row 0, appending data adds new blocks without changing the
    digests of existing ones, so:
      * verifying that an older artefact is still a valid prefix of a grown
        frame costs O(covered / block) hashes, not O(covered);
      * a freshly grown frame reproduces the *old* prefix digest exactly.
    """
    cols = ["open", "high", "low", "close", "volume"]
    block = max(int(block), 1)
    last_index = int(last_index)
    end = min(int(len(df)), last_index + int(window))
    if end <= 0:
        return _sha256_binary(b"")

    arr = np.ascontiguousarray(df.iloc[:end][cols].to_numpy(dtype=np.float64))
    n = arr.shape[0]
    chain = b""
    pos = 0
    while pos < n:
        chunk = arr[pos:pos + block]
        chain = hashlib.sha256(chunk.tobytes(order="C") + chain).digest()
        pos += block
    return chain.hex()


def _indices_checksum(indices: Optional[Sequence[int]], n: int) -> str:
    if indices is None:
        return f"0:{int(n)}"
    arr = np.asarray([int(i) for i in indices], dtype=np.int64)
    return _sha256_binary(np.ascontiguousarray(arr).tobytes(order="C"))


def content_key(spec: RenderSpec, window: int, frame_hash: str, indices: Optional[Sequence[int]], n: int) -> str:
    """Stable content identity of a compiled chart artefact."""
    payload = "|".join(
        [
            render_fingerprint(spec),
            str(int(window)),
            frame_hash,
            _indices_checksum(indices, n),
        ]
    )
    return _sha256_binary(payload.encode("utf-8"))


def render_checksum(
    arr: np.ndarray,
    meta: dict,
    *,
    sample_only: bool = True,
    exclude_keys: tuple[str, ...] = ("render_checksum", "render_checksum_full"),
) -> str:
    """Run-contract checksum for byte-equivalence guarantees.

    When ``sample_only`` (default) the digest covers the metadata plus a
    deterministic sample of rows (head, quarters, tail) — cheap and stable.
    Set ``sample_only=False`` for a full-file digest when closing out a
    release artefact.

    ``exclude_keys`` omits fields whose *value* depends on the digest itself
    (the persisted ``render_checksum``), avoiding a self-referential meta.
    """
    h = hashlib.sha256()
    canonical_source = {k: v for k, v in meta.items() if k not in exclude_keys}
    canonical = json.dumps(canonical_source, sort_keys=True, separators=(",", ":")).encode("utf-8")
    h.update(canonical)
    n = int(meta["n_images"])
    if sample_only:
        idx = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
    else:
        idx = list(range(n))
    for i in idx:
        row = np.ascontiguousarray(arr[i]).tobytes(order="C")
        h.update(row)
    return h.hexdigest()


def checksummed_meta(meta: dict, arr_or_binpath) -> dict:
    """Return ``meta`` with stable ``render_checksum`` (sampled) and
    ``render_checksum_full`` (all rows) fields attached.

    Both are computed over the meta *without themselves* so re-reading the
    artefact and recomputing reproduces them. The full digest costs one full
    file pass and is written once at compile time.
    """
    meta = dict(meta)
    n = int(meta["n_images"])
    excl = ("render_checksum", "render_checksum_full")
    h = hashlib.sha256()
    h.update(json.dumps({k: v for k, v in meta.items() if k not in excl},
                        sort_keys=True, separators=(",", ":")).encode("utf-8"))
    h_full = hashlib.sha256()
    h_full.update(json.dumps({k: v for k, v in meta.items() if k not in excl},
                             sort_keys=True, separators=(",", ":")).encode("utf-8"))

    def rows() -> Iterable[np.ndarray]:
        if isinstance(arr_or_binpath, (str, Path)):
            shape = (
                n,
                int(meta["image_channels"]),
                int(meta["image_size"]),
                int(meta["image_size"]),
            )
            mm = np.memmap(arr_or_binpath, dtype="float32", mode="r", shape=shape)
            for i in range(n):
                yield np.ascontiguousarray(mm[i])
            del mm
        else:
            for i in range(n):
                yield np.ascontiguousarray(arr_or_binpath[i])

    for i, row in enumerate(rows()):
        h_full.update(row.tobytes(order="C"))
        if i in sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1}):
            h.update(row.tobytes(order="C"))
    meta["render_checksum"] = h.hexdigest()
    meta["render_checksum_full"] = h_full.hexdigest()
    return meta


@dataclass
class CompiledChartStore:
    """Read a compiled (N, 3, H, W) chart array from a memmap / array."""

    path: Optional[Path]
    arr: np.ndarray
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        df: pd.DataFrame,
        window: int,
        spec: RenderSpec,
        *,
        indices: Optional[Sequence[int]] = None,
        out_root: Optional[Path | str] = None,
        progress_log_every: int = 50_000,
    ) -> "CompiledChartStore":
        """Render every window once and persist (or reuse) the artefact.

        If ``out_root`` is given and a content-addressed artefact already
        exists, it is opened instead of re-rendered. Returns an in-memory
        store otherwise.
        """
        frame_hash = frame_checksum(df)
        n = len(df) if indices is None else len(list(indices))
        if indices is None:
            indices = tuple(range(n))
        key = content_key(spec, int(window), frame_hash, indices, n)
        out_root = Path(out_root) if out_root is not None else None

        if out_root is not None:
            out_root.mkdir(parents=True, exist_ok=True)
            artifact_dir = out_root / key
            bin_path = artifact_dir / "charts.bin"
            meta_path = artifact_dir / "meta.json"
            if bin_path.exists() and meta_path.exists():
                return cls.open(artifact_dir)

            artifact_dir.mkdir(parents=True, exist_ok=True)
            return cls._materialize_to_disk(df, window, spec, indices, bin_path, meta_path, frame_hash, key, progress_log_every)

        # Pure in-memory build (tests, small datasets): still via canonical
        # renderer so pixels are identical to the on-disk path.
        image_size = int(spec.size)
        arr = np.empty((n, 3, image_size, image_size), dtype=np.float32)
        _fill_array(arr, df, window, spec, indices, progress_log_every)
        last_index = max((int(i) for i in indices), default=-1)
        covered_hash = covered_prefix_hash(df, int(window), last_index) if last_index >= 0 else ""
        meta = cls._make_meta(frame_hash, key, spec, n, image_size, window, indices, covered_hash)
        meta = checksummed_meta(meta, arr)
        return cls(path=None, arr=arr, meta=meta)

    @classmethod
    def _materialize_to_disk(cls, df, window, spec, indices, bin_path, meta_path, frame_hash, key, progress_log_every):
        n = len(list(indices))
        image_size = int(spec.size)
        last_index = max((int(i) for i in indices), default=-1)
        covered_hash = covered_prefix_hash(df, int(window), last_index) if last_index >= 0 else ""
        arr = np.memmap(bin_path, dtype="float32", mode="w+", shape=(n, 3, image_size, image_size))
        try:
            _fill_array(arr, df, window, spec, indices, progress_log_every)
            arr.flush()
        finally:
            del arr
        meta = cls._make_meta(frame_hash, key, spec, n, image_size, window, indices, covered_hash)
        meta = checksummed_meta(meta, bin_path)
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        return cls.open(bin_path.parent)

    @staticmethod
    def _make_meta(frame_hash, key, spec, n, image_size, window, indices, covered_hash, engine: str = "cpu") -> dict:
        idx = [int(i) for i in (indices if indices is not None else range(int(n)))]
        last_index = max(idx) if idx else -1
        return {
            "format": "zhisa_compiled_charts",
            "version": 1,
            "content_key": key,
            "renderer": CANONICAL_RENDERER_VERSION,
            "spec": spec.to_meta(),
            "spec_hash": spec.content_hash(),
            "fingerprint": render_fingerprint(spec),
            "engine": str(engine),
            "render_machine": None,
            "n_images": int(n),
            "image_channels": 3,
            "image_size": int(image_size),
            "chart_window_bars": int(window),
            "input_frame_checksum": frame_hash,
            "covered_prefix_hash": covered_hash,
            "indices_start": int(min(idx)) if idx else 0,
            "indices_last": last_index,
            "dtype": "float32",
            "layout": "nchw",
        }

    @classmethod
    def open(cls, path: Path | str, *, verify: bool = True) -> "CompiledChartStore":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        bin_path = path / "charts.bin"
        if not bin_path.exists():
            raise FileNotFoundError(bin_path)
        shape = (
            int(meta["n_images"]),
            int(meta["image_channels"]),
            int(meta["image_size"]),
            int(meta["image_size"]),
        )
        arr = np.memmap(bin_path, dtype="float32", mode="r", shape=shape)
        store = cls(path=path, arr=arr, meta=meta)
        if verify and not store.verify_integrity(full=False):
            raise RuntimeError(
                f"compiled chart store failed integrity verification: {path}"
            )
        return store

    def verify_integrity(self, *, full: bool = False) -> bool:
        """Cheap on-open integrity: schema, file size, persisted checksum.

        ``full=False`` verifies the deterministic sample checksum stored at
        build time; ``full=True`` recomputes over every row (slow, for release
        audits). Returns False on any inconsistency instead of raising.
        """
        meta = self.meta
        try:
            if meta.get("format") != "zhisa_compiled_charts":
                return False
            if meta.get("dtype") != "float32" or meta.get("layout") != "nchw":
                return False
            n = int(meta["n_images"])
            c = int(meta["image_channels"])
            s = int(meta["image_size"])
            if (self.arr.shape) != (n, c, s, s):
                return False
            if self.path is not None:
                expected_bytes = n * c * s * s * 4
                if (self.path / "charts.bin").stat().st_size != expected_bytes:
                    return False
            if "render_checksum" in meta:
                fresh = render_checksum(
                    self.arr, meta, sample_only=True,
                    exclude_keys=("render_checksum", "render_checksum_full"),
                )
                if fresh != str(meta["render_checksum"]):
                    return False
            if full and "render_checksum_full" in meta:
                fresh_full = render_checksum(
                    self.arr, meta, sample_only=False,
                    exclude_keys=("render_checksum", "render_checksum_full"),
                )
                if fresh_full != str(meta["render_checksum_full"]):
                    return False
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return int(self.meta["n_images"])

    def __getitem__(self, i):
        if isinstance(i, slice):
            idx = np.arange(len(self))[i]
            out = np.empty((len(idx),) + self.arr.shape[1:], dtype=self.arr.dtype)
            for row, j in enumerate(idx):
                out[row] = self.arr[j]
            return out
        i = int(i)
        if i < 0:
            i += len(self)
        if i < 0 or i >= len(self):
            raise IndexError(i)
        return self.arr[i]

    def to_tensor(self, i: int):
        import torch
        return torch.from_numpy(np.ascontiguousarray(self.arr[i]))

    # ------------------------------------------------------------------
    # Contract
    # ------------------------------------------------------------------

    def render_checksum(self, *, full: bool = False) -> str:
        return render_checksum(
            self.arr, self.meta, sample_only=not full, exclude_keys=("render_checksum",)
        )

    @property
    def render_meta(self) -> dict:
        """Provenance block (content key, spec hash, renderer) for audit/contract."""
        return self.meta

    # ------------------------------------------------------------------
    # Freshness / update cycle (append-stable covered-prefix hashing)
    # ------------------------------------------------------------------

    def is_contiguous(self) -> bool:
        return int(self.meta.get("indices_start", 0)) == 0 and int(self.meta.get("n_images", 0)) >= 1

    def stale_for(self, df: pd.DataFrame, window: int, spec: RenderSpec, n: Optional[int] = None) -> bool:
        """Is this artefact stale w.r.t. the *current* frame?

        True when the covered prefix no longer matches (data changed in the
        captured range) or when a desired sample count ``n`` differs from what
        the artefact holds. Only contiguous artefacts (built for ``0..n-1``)
        are comparable; sparse artefacts are never provably fresh.
        """
        if not self.is_contiguous():
            raise NotImplementedError("freshness is only defined for contiguous 0..n-1 artefacts")
        stored_last = int(self.meta.get("indices_last", -1))
        if int(stored_last) < 0 or int(stored_last) + int(window) > len(df):
            return True
        current = covered_prefix_hash(df, int(window), stored_last)
        if current != str(self.meta.get("covered_prefix_hash", "")):
            return True
        if n is not None and int(n) != int(self.meta["n_images"]):
            return True
        return False

    def assert_fresh(self, df: pd.DataFrame, window: int, spec: RenderSpec, n: Optional[int] = None) -> None:
        if self.stale_for(df, window, spec, n=n):
            raise RuntimeError(
                "compiled chart store is stale for the current frame: "
                f"content_key={self.meta.get('content_key', '')[:12]} "
                f"n_images={self.meta.get('n_images')} "
                f"indices_last={self.meta.get('indices_last')}. "
                "Refresh via zhisa.data.data_cycle.update_prepared_charts."
            )

    def refresh(
        self,
        df: pd.DataFrame,
        window: int,
        spec: RenderSpec,
        *,
        n: int,
        out_root: Path | str,
        workers: int = 0,
        chunk: int = 5_000,
    ):
        """Refresh this artefact against the current frame (incremental when possible)."""
        from zhisa.data.render_job import materialize_parallel
        store, stats = materialize_parallel(
            df, window=window, spec=spec, n=int(n),
            out_root=out_root, workers=workers, chunk_size=chunk,
        )
        store.assert_fresh(df, window, spec, n=int(n))
        return store, stats

    def is_byte_equivalent(self, other: "CompiledChartStore", *, full: bool = False) -> bool:
        if self.meta.get("fingerprint") != other.meta.get("fingerprint"):
            return False
        if self.meta.get("n_images") != other.meta.get("n_images"):
            return False
        if self.meta.get("image_size") != other.meta.get("image_size"):
            return False
        return self.render_checksum(full=full) == other.render_checksum(full=full)

    def verify_input(self, df: pd.DataFrame, window: int, spec: RenderSpec, indices: Optional[Sequence[int]] = None) -> bool:
        """Check that this artefact really was built from ``(df, window, spec)``."""
        n = len(df) if indices is None else len(list(indices))
        key = content_key(spec, int(window), frame_checksum(df), indices, n)
        return key == self.meta.get("content_key")


def _fill_array(arr: np.ndarray, df: pd.DataFrame, window: int, spec: RenderSpec, indices, progress_log_every: int) -> None:
    """Render windows into ``arr`` (memmap or ndarray) in place."""
    ohlcv = df[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64)
    n = arr.shape[0]
    for row, t in enumerate(indices):
        t = int(t)
        if t < 0 or t + window > len(ohlcv):
            raise IndexError(f"window [{t}:{t + window}] out of frame (len={len(ohlcv)})")
        img = _render_ohlcv_canonical(ohlcv[t:t + window], spec).transpose(2, 0, 1)
        arr[row] = img.astype(np.float32, copy=False)
        if progress_log_every and row > 0 and row % progress_log_every == 0:
            pass  # avoid import-time logging; caller may pass hooks later
    return None