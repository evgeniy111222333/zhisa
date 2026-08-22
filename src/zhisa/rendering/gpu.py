"""GPU chart rasterizer: deterministic, functional, parity-checked.

Why not a naive scatter renderer? A plain ``scatter_(..., fill)`` on overlapping
candles is a **write race** — non-deterministic across threads/runs, which would
break the project's byte-equivalence contract. Instead this rasterizer is
*functional*:

- a pixel's value is the palette colour of the **bar with the max bar-index**
  that covers it (which is exactly what the CPU canonical fancy-assignment
  produces, since numpy applies later-indexed bars last);
- that "max over covering bars" is computed with ``scatter_reduce("amax")``,
  which is **order-independent** (float max is exact, commutative, associative)
  → deterministic regardless of thread scheduling or device;
- overlays (SMA runs) resolve by max *period* rank, matching last-period-wins.

Parity with the CPU canonical renderer is *not* guaranteed bit-for-bit in
general (fp64/fp32 truncation boundaries, numpy ``interp`` duplicate-xp
semantics, FMA contraction). Therefore GPU output is **always validated** against
the CPU canonical on a corpus before being trusted (see
:func:`validate_gpu_against_cpu`), and the engine used is recorded in the
artefact metadata. Non-finite input is rejected identically in both engines.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from zhisa.rendering.chart_renderer import _render_ohlcv_canonical
from zhisa.rendering.spec import RenderSpec

GPU_ENGINE_NAME = "gpu_canonical"
GPU_ENGINE_VERSION = "1.0.0"


@dataclass(frozen=True)
class GPUValidation:
    ok: bool
    max_abs_diff: float
    n_diff_pixels: int
    total_pixels: int
    atol: float
    device: str


def gpu_device(prefer: Optional[str] = None) -> Optional[torch.device]:
    if not torch.cuda.is_available():
        return None
    idx = 0
    if prefer is not None and prefer.startswith("cuda") and ":" in prefer:
        idx = int(prefer.split(":")[1])
    if idx >= torch.cuda.device_count():
        return None
    return torch.device(f"cuda:{idx}")


def _check_finite(ohlcv: torch.Tensor) -> None:
    if not torch.isfinite(ohlcv).all():
        raise ValueError("ohlcv contains non-finite values; rendering is undefined for NaN/Inf")


def _geometry(ohlcv: torch.Tensor, spec: RenderSpec) -> dict:
    """Per-window geometry in fp64 (closest parity to the CPU numpy path)."""
    B, N, _ = ohlcv.shape
    S = int(spec.size) * max(int(spec.supersample), 1)
    price_h = int(S * spec.price_frac)
    eps = float(spec.min_range)

    lo = ohlcv[:, :, 2].min(dim=1).values.unsqueeze(-1)   # low
    hi = ohlcv[:, :, 1].max(dim=1).values.unsqueeze(-1)   # high
    rng = torch.clamp(hi - lo, min=eps)

    denom = float(max(N - 1, 1))
    xs = torch.floor(torch.arange(N, dtype=ohlcv.dtype, device=ohlcv.device) * (S - 1) / denom)
    xs = xs.to(torch.int64)

    def price_y(values: torch.Tensor) -> torch.Tensor:
        scaled = (values - lo) / rng * (price_h - 2)
        return (price_h - torch.floor(scaled)).clamp(0, price_h - 1).to(torch.int64)

    return {
        "S": S,
        "price_h": price_h,
        "xs": xs,
        "price_y": price_y,
        "yo": price_y(ohlcv[:, :, 0]),
        "yc": price_y(ohlcv[:, :, 3]),
        "yh": price_y(ohlcv[:, :, 1]),
        "yl": price_y(ohlcv[:, :, 2]),
        "lo": lo, "hi": hi, "rng": rng,
    }


def _scatter_amax(rows: int, cols: int, col_idx: torch.Tensor, src_val: torch.Tensor) -> torch.Tensor:
    """Functional 'max over covering contributors' raster.

    Every contributor lives at (batch, row, bar): ``src_val`` (B, rows, N)
    holds its value or -1 for 'no cover'; ``col_idx`` (N,) maps a bar to a
    pixel column. Returns (B, rows, cols) with -1 where nothing covers
    (order-independent amax).
    """
    B, _, N = src_val.shape
    idx_col = col_idx.view(1, 1, -1).expand(B, rows, N)
    out = torch.full((B, rows, cols), -1.0, dtype=src_val.dtype, device=src_val.device)
    out = out.scatter_reduce(2, idx_col, src_val, reduce="amax", include_self=False)
    return out


def _candle_raster(geom: dict, ohlcv: torch.Tensor) -> torch.Tensor:
    """(B, price_h, S) max covering bar index (-1 = background)."""
    B = ohlcv.shape[0]
    price_h = geom["price_h"]
    y_grid = torch.arange(price_h, dtype=torch.int64, device=ohlcv.device).unsqueeze(0)
    min_y = torch.minimum(geom["yh"], geom["yl"])
    max_y = torch.maximum(geom["yh"], geom["yl"])
    cover = (y_grid.unsqueeze(-1) >= min_y.unsqueeze(1)) & (y_grid.unsqueeze(-1) <= max_y.unsqueeze(1))
    bar_ids = torch.arange(ohlcv.shape[1], dtype=torch.float64, device=ohlcv.device)
    src_val = torch.where(cover, bar_ids.view(1, 1, -1).expand(B, -1, -1), -1.0)
    return _scatter_amax(price_h, geom["S"], geom["xs"], src_val)


def _volume_raster(geom: dict, ohlcv: torch.Tensor) -> torch.Tensor:
    """(B, S-price_h, S) max covering volume bar index (-1 = background)."""
    B = ohlcv.shape[0]
    S, price_h = geom["S"], geom["price_h"]
    v = ohlcv[:, :, 4]
    vmax = torch.clamp(v.max(dim=1).values.unsqueeze(-1), min=1e-9)
    bar_heights = torch.floor(v / vmax * (S - price_h - 1)).to(torch.int64)
    y_grid = torch.arange(price_h, S, dtype=torch.int64, device=ohlcv.device)  # (rows,)
    start = S - bar_heights
    cover = y_grid.unsqueeze(0).unsqueeze(-1) >= start.unsqueeze(1)
    bar_ids = torch.arange(ohlcv.shape[1], dtype=torch.float64, device=ohlcv.device)
    src_val = torch.where(cover, bar_ids.view(1, 1, -1).expand(B, -1, -1), -1.0)
    return _scatter_amax(S - price_h, S, geom["xs"], src_val)


def _running_mean(close: torch.Tensor, period: int) -> torch.Tensor:
    """Match CPU ``_running_mean`` (min_periods=1, prefix-expanding head)."""
    n = close.size(-1)
    if n < period:
        one = torch.arange(1, n + 1, dtype=close.dtype, device=close.device)
        return torch.cumsum(close, dim=-1) / one
    kernel = torch.full((period,), 1.0 / period, dtype=close.dtype, device=close.device)
    sma = torch.nn.functional.conv1d(close.unsqueeze(1), kernel.view(1, 1, -1)).squeeze(1)
    prefix = torch.cumsum(close[..., : period - 1], dim=-1) / torch.arange(
        1, period, dtype=close.dtype, device=close.device
    )
    return torch.cat([prefix, sma], dim=-1)


def _interp_interp(xp: torch.Tensor, fp: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Replicate ``np.interp(x, xp, fp)`` for a non-decreasing xp with duplicates.

    numpy.interp: exact hits return the LAST fp among duplicate xp; between two
    distinct xp it returns linear interpolation (clamped at the edges). xp is
    the (non-decreasing, integer) column map; x is the full pixel grid.
    Returns (B, S) fp64.
    """
    N = xp.size(0)
    x = x.to(torch.float64)
    xp_t = xp.to(torch.float64)
    fp_t = fp.to(torch.float64)  # (B, N)

    i = torch.searchsorted(xp_t, x, right=True).clamp(0, N - 1) - 1
    i = i.clamp(min=0)
    same = xp_t[i] == x

    # last duplicate of an exact hit
    j = i
    for _ in range(N):
        nxt = (j + 1).clamp(max=N - 1)
        m = same & (xp_t[nxt] == x)
        if not bool(m.any()):
            break
        j = torch.where(m, nxt, j)
    hit = fp_t.gather(1, j.view(1, -1).expand(fp_t.size(0), -1))

    # linear interpolation between i and i+1 for non-exact positions
    i_next = (i + 1).clamp(max=N - 1)
    x0 = xp_t[i]
    x1 = xp_t[i_next]
    y0 = fp_t.gather(1, i.view(1, -1).expand(fp_t.size(0), -1))
    y1 = fp_t.gather(1, i_next.view(1, -1).expand(fp_t.size(0), -1))
    span = (x1 - x0).clamp(min=1e-300)
    t = (x - x0) / span
    lin = y0 + (y1 - y0) * t

    out = torch.where(same, hit, lin)
    # clamp x outside [xp0, xpN-1] to edge ys (safety; x grid is inside by design)
    out = out.clamp(min=fp_t.min(dim=-1).values.view(-1, 1), max=fp_t.max(dim=-1).values.view(-1, 1))
    return out


def render_batch_gpu(
    ohlcv: np.ndarray,
    spec: Optional[RenderSpec] = None,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Render ``(B, N, >=5)`` windows to ``(B, 3, size, size)`` float32 (numpy).

    Deterministic, functional rasterisation (scatter-reduce-amax), fp64
    geometry, clamps identical to the CPU canonical path.
    """
    spec = spec or RenderSpec()
    device = device or gpu_device()
    if device is None:
        raise RuntimeError("GPU rendering requested but no CUDA device available")
    ohlcv = np.ascontiguousarray(ohlcv, dtype=np.float64)
    B, N, _ = ohlcv.shape
    t = torch.from_numpy(ohlcv).to(device)
    _check_finite(t)

    size, ss = int(spec.size), max(int(spec.supersample), 1)
    S = size * ss
    price_h = int(S * spec.price_frac)

    geom = _geometry(t, spec)

    candles = _candle_raster(geom, t)          # (B, price_h, S)
    volumes = _volume_raster(geom, t)          # (B, S-price_h, S)

    # overlays: per-pixel max period rank (-1 none)
    lines = None
    periods_sorted = sorted(spec.overlays, key=lambda ov: int(ov[0]))
    if spec.include_overlays and periods_sorted:
        ranks = torch.full((B, price_h, S), -1.0, dtype=torch.float64, device=device)
        rr = torch.arange(B, device=device).view(B, 1).expand(B, S)
        cc = torch.arange(S, device=device).view(1, S).expand(B, S)
        for rank, (period, _col) in enumerate(periods_sorted):
            if N < int(period):
                continue
            sma = _running_mean(t[:, :, 3], int(period))
            ys = geom["price_y"](sma)
            x_grid = torch.arange(S, dtype=torch.float64, device=device)
            interp_y = _interp_interp(geom["xs"].to(torch.float64), ys, x_grid)
            line_y = torch.round(interp_y).clamp(0, price_h - 1).to(torch.int64)
            ranks[rr, line_y, cc] = float(rank)
        lines = ranks

    # palette
    green = torch.tensor(spec.green, dtype=torch.float64, device=device)
    red = torch.tensor(spec.red, dtype=torch.float64, device=device)
    bg = torch.tensor(spec.background, dtype=torch.float64, device=device)
    is_up = (t[:, :, 3] >= t[:, :, 0]).to(torch.float64).unsqueeze(-1)
    colors = torch.where(is_up > 0.5, green.view(1, 1, 3), red.view(1, 1, 3))  # (B,N,3)

    def _gathered(color_buffer, price_rows):
        idx = color_buffer.to(torch.int64)
        valid = idx >= 0
        safe = idx.clamp(min=0)
        bb = torch.arange(B, device=device)
        pick = colors[bb.view(B, 1, 1).expand(B, price_rows, S), safe]  # (B, price_rows, S, 3)
        return torch.where(valid.unsqueeze(-1), pick, bg.view(1, 1, 1, 3).expand(B, price_rows, S, 3))

    out = torch.full(
        (B, S, S, 3), 0.0, dtype=torch.float64, device=device
    )
    out[:, :, :, :] = bg.view(1, 1, 1, 3)
    out[:, :price_h] = _gathered(candles, price_h)
    if spec.include_volume:
        out[:, price_h:] = _gathered(volumes, S - price_h)

    if lines is not None:
        overlay_cols = torch.stack([torch.tensor(c, dtype=torch.float64, device=device) for _p, c in periods_sorted], dim=0)  # (P,3)
        rank_idx = lines.to(torch.int64).clamp(min=0)
        line_color = overlay_cols[rank_idx]                       # (B, price_h, S, 3)
        overlayed = torch.where((lines >= 0).unsqueeze(-1), line_color, out[:, :price_h])
        out[:, :price_h] = overlayed

    if ss > 1:
        out = out.view(B, size, ss, size, ss, 3).mean(dim=(2, 4))
    return out.detach().cpu().to(torch.float32).numpy().transpose(0, 3, 1, 2)


def render_ohlcv_gpu(ohlcv: np.ndarray, spec: Optional[RenderSpec] = None, device=None) -> np.ndarray:
    """Render a single ``(N, >=5)`` window to ``(3, size, size)``."""
    return render_batch_gpu(np.asarray(ohlcv)[None, ...], spec=spec, device=device)[0]


def validate_gpu_against_cpu(
    ohlcv_corpus: np.ndarray,
    spec: RenderSpec,
    *,
    atol: float = 1e-5,
    device=None,
) -> GPUValidation:
    """Gate: fresh GPU render vs CPU canonical on a corpus (B, N, 5)."""
    ohlcv_corpus = np.ascontiguousarray(ohlcv_corpus, dtype=np.float64)
    B = ohlcv_corpus.shape[0]
    cpu = np.stack(
        [_render_ohlcv_canonical(ohlcv_corpus[i], spec).transpose(2, 0, 1) for i in range(B)]
    ).astype(np.float32)
    gpu = render_batch_gpu(ohlcv_corpus, spec=spec, device=device).astype(np.float32)
    diff = np.abs(cpu.astype(np.float64) - gpu.astype(np.float64))
    maxd = float(diff.max())
    ndiff = int((diff > atol).sum())
    return GPUValidation(
        ok=(maxd <= atol),
        max_abs_diff=maxd,
        n_diff_pixels=ndiff,
        total_pixels=int(cpu.size),
        atol=atol,
        device=str(device or gpu_device()),
    )