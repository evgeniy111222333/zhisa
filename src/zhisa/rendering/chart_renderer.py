"""Chart image renderer: OHLCV DataFrame window -> RGB tensor.

The renderer is deliberately lightweight — it draws a candlestick chart
with a small set of overlays and returns a ``torch.Tensor`` of shape
``(3, H, W)`` in [0, 1]. It is used both as a model input channel and
as a debugging / interpretability tool.
"""
from __future__ import annotations

import os
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    from matplotlib.patches import Rectangle  # noqa: E402
    _HAS_MPL = True
except Exception:  # pragma: no cover
    _HAS_MPL = False


_DEFAULT_BG = (0.05, 0.06, 0.09)
_DEFAULT_FG = (0.85, 0.88, 0.92)
_GREEN = (0.20, 0.80, 0.40)
_RED = (0.95, 0.35, 0.30)
_GREY = (0.45, 0.48, 0.55)


def _draw_candles(ax, df: pd.DataFrame) -> None:
    if not _HAS_MPL:
        return
    x = np.arange(len(df))
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    width = 0.6
    for i in range(len(df)):
        color = _GREEN if c[i] >= o[i] else _RED
        # Wick
        ax.plot([i, i], [l[i], h[i]], color=color, linewidth=0.7, solid_capstyle="butt")
        # Body
        body_low = min(o[i], c[i])
        body_height = max(abs(o[i] - c[i]), 1e-9 * (h[i] - l[i] + 1e-12))
        rect = Rectangle((i - width / 2, body_low), width, body_height,
                         facecolor=color, edgecolor=color, linewidth=0.4)
        ax.add_patch(rect)
    ax.set_xlim(-0.5, len(df) - 0.5)


def _draw_overlay(ax, df: pd.DataFrame, kind: str, period: int, color: str) -> None:
    if not _HAS_MPL:
        return
    s = df["close"].rolling(period, min_periods=1).mean()
    ax.plot(np.arange(len(df)), s.to_numpy(), color=color, linewidth=1.0, label=f"{kind}{period}")


def _draw_volume(ax, df: pd.DataFrame) -> None:
    if not _HAS_MPL:
        return
    x = np.arange(len(df))
    o = df["open"].to_numpy()
    c = df["close"].to_numpy()
    v = df["volume"].to_numpy()
    vmax = v.max() if v.max() > 0 else 1.0
    for i in range(len(df)):
        color = _GREEN if c[i] >= o[i] else _RED
        ax.plot([i, i], [0, v[i] / vmax], color=color, linewidth=1.0, alpha=0.55)


def _df_to_image(df: pd.DataFrame, size: int) -> np.ndarray:
    if not _HAS_MPL:
        # Last-resort: a tiny placeholder image (so the rest of the pipeline
        # can still be exercised in environments without matplotlib).
        return np.full((size, size, 3), 0.5, dtype=np.float32)
    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
    gs = fig.add_gridspec(4, 1, hspace=0.0)
    ax_price = fig.add_subplot(gs[0:3, 0])
    ax_vol = fig.add_subplot(gs[3, 0], sharex=ax_price)
    for ax in (ax_price, ax_vol):
        ax.set_facecolor(_DEFAULT_BG)
        for spine in ax.spines.values():
            spine.set_color(_GREY)
        ax.tick_params(colors=_DEFAULT_FG, labelsize=4, length=0)
    fig.patch.set_facecolor(_DEFAULT_BG)

    _draw_candles(ax_price, df)
    _draw_overlay(ax_price, df, "SMA", 10, "#3aa0ff")
    _draw_overlay(ax_price, df, "SMA", 30, "#ffaa33")
    _draw_volume(ax_vol, df)
    ax_price.set_ylim(df["low"].min(), df["high"].max())
    ax_vol.set_ylim(0, 1)
    ax_vol.set_yticks([])
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    try:
        w, h = fig.canvas.get_width_height()
        buf = fig.canvas.tostring_argb()
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        rgb = arr[:, :, 1:4]  # ARGB -> RGB
    except Exception:
        # Fallback via savefig to a buffer (works on more backends)
        bio = BytesIO()
        fig.savefig(bio, format="png", facecolor=fig.get_facecolor())
        bio.seek(0)
        from PIL import Image
        rgb = np.asarray(Image.open(bio).convert("RGB"))
    plt.close(fig)
    return rgb.astype(np.float32) / 255.0


def render_chart(df: pd.DataFrame, size: int = 64) -> torch.Tensor:
    """Render a small candlestick + volume chart from an OHLCV window.

    Returns a ``torch.FloatTensor`` of shape ``(3, H, W)`` in [0, 1].
    Falls back to a uniform grey image if matplotlib is unavailable.

    Set the environment variable ``ZHISA_FAST_RENDER=1`` to use the
    pure-numpy renderer (much faster, used by default in tests).
    """
    if df is None or len(df) == 0:
        return torch.full((3, size, size), 0.5, dtype=torch.float32)
    use_fast = os.environ.get("ZHISA_FAST_RENDER", "0") == "1"
    if use_fast:
        rgb = _fast_render(df, size)
    else:
        rgb = _df_to_image(df, size)
    # Resize to (size, size) if needed (defensive)
    if rgb.shape[0] != size or rgb.shape[1] != size:
        try:
            from PIL import Image
            img = Image.fromarray((rgb * 255).astype(np.uint8)).resize((size, size))
            rgb = np.asarray(img).astype(np.float32) / 255.0
        except Exception:
            rgb = np.full((size, size, 3), 0.5, dtype=np.float32)
    t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    return t.float()


def _fast_render(df: pd.DataFrame, size: int) -> np.ndarray:
    """A pure-numpy candlestick renderer; an order of magnitude faster
    than matplotlib and good enough for feature extraction."""
    values = df[["open", "high", "low", "close", "volume"]].to_numpy(
        dtype=np.float64, copy=False
    )
    return _fast_render_ohlcv(values, size)


def render_chart_array(ohlcv: np.ndarray, size: int = 64) -> torch.Tensor:
    """Render a contiguous ``(N, 5)`` OHLCV array without pandas overhead."""
    rgb = _fast_render_ohlcv(ohlcv, size)
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float()


def _fast_render_ohlcv(ohlcv: np.ndarray, size: int) -> np.ndarray:
    if len(ohlcv) < 2:
        return np.full((size, size, 3), 0.5, dtype=np.float32)
    values = np.asarray(ohlcv, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 5:
        raise ValueError("ohlcv must have shape (N, >=5)")
    n = len(values)
    rgb = np.full((size, size, 3), 0.05, dtype=np.float32)
    o_arr, h_arr, l_arr, c_arr, v_arr = values[:, :5].T

    lo = float(l_arr.min())
    hi = float(h_arr.max())
    rng = max(hi - lo, 1e-9)
    xs = (np.arange(n) * (size - 1) / max(n - 1, 1)).astype(int)
    price_h = int(size * 0.75)

    def price_y(values_: np.ndarray) -> np.ndarray:
        y = price_h - ((values_ - lo) / rng * (price_h - 2)).astype(int)
        return np.clip(y, 0, price_h - 1)

    y_o, y_c = price_y(o_arr), price_y(c_arr)
    y_h, y_l = price_y(h_arr), price_y(l_arr)
    colors = np.where(
        (c_arr >= o_arr)[:, None],
        np.asarray(_GREEN, dtype=np.float32),
        np.asarray(_RED, dtype=np.float32),
    )
    y_grid = np.arange(price_h)[:, None]
    wick = (y_grid >= np.minimum(y_h, y_l)) & (y_grid <= np.maximum(y_h, y_l))
    body = (y_grid >= np.minimum(y_o, y_c)) & (y_grid <= np.maximum(y_o, y_c))
    py, bars = np.nonzero(wick | body)
    rgb[py, xs[bars]] = colors[bars]

    vmax = max(v_arr.max(), 1e-9)
    bar_heights = (v_arr / vmax * (size - price_h - 1)).astype(int)
    volume_y = np.arange(price_h, size)[:, None]
    volume_mask = volume_y >= (size - bar_heights)[None, :]
    vy, bars = np.nonzero(volume_mask)
    rgb[vy + price_h, xs[bars]] = colors[bars]

    for p, col in ((10, np.array([0.23, 0.63, 1.0], dtype=np.float32)),
                   (30, np.array([1.0, 0.67, 0.20], dtype=np.float32))):
        if n < p:
            continue
        kernel = np.ones(p) / p
        sma = np.convolve(c_arr, kernel, mode='valid')
        pad_len = n - len(sma)
        if pad_len > 0:
            pad = np.cumsum(c_arr[:pad_len]) / np.arange(1, pad_len + 1)
            sma = np.concatenate((pad, sma))
        ys = price_y(sma)
        x_grid = np.arange(size)
        line_y = np.rint(np.interp(x_grid, xs, ys)).astype(int)
        rgb[np.clip(line_y, 0, price_h - 1), x_grid] = col
    return rgb


def _draw_line(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: np.ndarray) -> None:
    """Bresenham-style line drawing onto an HxWx3 image (clipped)."""
    H, W = img.shape[:2]
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        if 0 <= x < W and 0 <= y < H:
            img[y, x] = color
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def render_chart_batch(
    windows: list[pd.DataFrame],
    size: int = 64,
) -> torch.Tensor:
    """Batched wrapper around :func:`render_chart`."""
    return torch.stack([render_chart(w, size=size) for w in windows], dim=0)
