"""Multimodal PyTorch Dataset for trading.

Each sample at index ``t`` contains:
    - chart image of the last ``chart_window`` bars
    - numeric feature tensor over the same window
    - context vector (time, instrument, position)
    - labels (triple-barrier, realized vol, regime) for supervision

The dataset supports the typical ``torch.utils.data.Dataset`` protocol
and exposes a ``collate`` function that pads/batches multimodal samples
into ``MultimodalBatch`` namedtuples.

Performance design (revised):
    * All pandas tables are converted to C-contiguous numpy arrays in
      ``__init__`` (one-shot cost). Hot-path ``__getitem__`` uses zero-copy
      numpy slicing and ``torch.from_numpy``.
    * If ``cache_charts`` is true (the default in the ``precompute`` mode),
      every chart image is rendered once during ``__init__`` and stored in
      a single preallocated ``(N, 3, H, W)`` numpy array. This eliminates
      matplotlib from the training hot path entirely.
    * When ``cache_charts`` is false, charts are rendered lazily and memoized
      in an LRU-style dict bounded by ``chart_cache_size``.
    * The fast path is bit-exactly equivalent to the old pandas-based
      ``__getitem__`` (verified by ``tests/test_dataset_perf.py``).
"""
from __future__ import annotations

import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from zhisa.data.labeling import (
    ForwardReturnConfig,
    TripleBarrierConfig,
    forward_return_targets,
    hmm_regime_labels,
    realized_volatility,
    triple_barrier,
)
from zhisa.features.ohlcv import compute_ohlcv_features, normalize_feature_window
from zhisa.features.time import compute_time_features
from zhisa.rendering.chart_renderer import render_chart, render_chart_array

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


@dataclass
class SampleSpec:
    """What every sample must contain."""

    chart_window: int = 64
    feature_window: int = 64
    horizons: Sequence[int] = (4, 16, 64)  # for triple-barrier labels
    image_size: int = 64
    include_volume: bool = True
    include_indicators: bool = True
    n_regime_states: int = 4


@dataclass
class MarketTargetConfig:
    """Label contract for supervised market-head training."""

    direction_mode: str = "triple_barrier"
    flat_return_bps: float = 1.0
    flat_volatility_mult: float = 0.0
    flat_min_bps: float = 0.0
    flat_max_bps: float = 0.0
    use_log_return: bool = False
    horizon_overrides: dict[int, dict[str, float | bool]] | None = None


@dataclass
class MacroContextConfig:
    """Higher-timeframe context contract for a primary market sample."""

    enabled: bool = False
    window: int = 64
    resample_rule: str = "1h"
    source: str = "resample"


@dataclass
class MultimodalBatch:
    """A batched collection of multimodal samples."""

    chart: torch.Tensor         # (B, 3, H, W)
    numeric: torch.Tensor       # (B, T, F)
    context: torch.Tensor       # (B, C)
    label_dir: torch.Tensor     # (B,) long  -- direction label in {-1, 0, +1}
    label_vol: torch.Tensor     # (B,) float -- forward realised vol
    label_regime: torch.Tensor  # (B,) long
    label_ret: torch.Tensor     # (B,) float -- forward return
    label_risk: torch.Tensor    # (B,) float -- downside return + realised vol
    mask: torch.Tensor          # (B, T) bool -- which time steps are valid
    meta: list                  # per-sample metadata (symbol, ts, ...)
    label_dir_multi: torch.Tensor | None = None  # (B, H) long
    label_ret_multi: torch.Tensor | None = None  # (B, H) float
    label_dir_persistence: torch.Tensor | None = None  # (B,) long, causal t-horizon baseline
    label_dir_multi_persistence: torch.Tensor | None = None  # (B, H) long
    macro_numeric: torch.Tensor | None = None  # (B, M, F) float


class MarketDataset(Dataset):
    """Multimodal trading dataset over a single OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV dataframe with a ``DatetimeIndex``.
    spec : SampleSpec, optional
        Description of the sample shape.
    triple_barrier_cfg : TripleBarrierConfig, optional
        Configuration for explicit ``direction_mode="triple_barrier"``
        legacy labelling and for callers that still inspect ``_tb_cfg_primary``.
    cache_charts : bool, default ``True``
        When true, all chart images are rendered once in ``__init__`` and
        stored in a single ``(N, 3, H, W)`` float32 array. This trades
        ~``len(ds) * 3 * H * W * 4`` bytes of RAM (≈ 50-150 MB on a typical
        8k-bar run with H=W=64) for a much faster hot path. The first
        epoch is then as fast as every subsequent one.
    precompute : bool, default ``True``
        When true (the default), all pandas tables are converted to numpy
        arrays in ``__init__`` so that ``__getitem__`` does not touch
        pandas at all. This is the recommended setting; it is exposed
        only for debugging/legacy-compat.
    chart_cache_size : int, default ``0``
        When ``cache_charts`` is false, this bounds the lazy chart cache
        to the last N entries (LRU). ``0`` means unbounded (legacy), while
        a negative value disables the lazy cache entirely.

    Notes
    -----
    When ``cache_charts=True`` and ``precompute=True`` (the default), the
    class sets the marker attribute ``__fast_getitem__ = True`` so that
    :func:`zhisa.training.dataloader_factory.build_dataloader` will pick
    ``num_workers=0`` by default. With precomputed numpy views, the IPC
    overhead of multiple workers is larger than the work per item.
    """

    # Set by __init__ when the fast path is active. Used by the
    # DataLoader factory to pick num_workers=0 by default.
    __fast_getitem__: bool = False

    def __init__(
        self,
        df: pd.DataFrame,
        spec: Optional[SampleSpec] = None,
        triple_barrier_cfg: Optional[TripleBarrierConfig] = None,
        target_cfg: Optional[MarketTargetConfig] = None,
        cache_charts: bool = True,
        precompute: bool = True,
        chart_cache_size: int = 0,
        compute_targets: bool = True,
        macro_cfg: Optional[MacroContextConfig] = None,
        macro_df: Optional[pd.DataFrame] = None,
    ) -> None:
        spec = spec or SampleSpec()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("df must have a DatetimeIndex")
        if len(df) < spec.chart_window + 2:
            raise ValueError("df is too short for the requested window")
        self.df = df
        self._ohlcv_arr = np.ascontiguousarray(
            df[["open", "high", "low", "close", "volume"]].to_numpy(
                dtype=np.float64
            )
        )
        self._fast_render = os.environ.get("ZHISA_FAST_RENDER", "0") == "1"
        self.spec = spec
        self.tb_cfg = triple_barrier_cfg or TripleBarrierConfig()
        self.target_cfg = target_cfg or MarketTargetConfig()
        if self.target_cfg.direction_mode not in {"forward_return", "triple_barrier"}:
            raise ValueError(
                "direction_mode must be 'forward_return' or 'triple_barrier', "
                f"got {self.target_cfg.direction_mode!r}"
            )
        self._cache_charts = bool(cache_charts)
        self._precompute = bool(precompute)
        self._chart_cache_size = int(chart_cache_size)
        self._compute_targets = bool(compute_targets)
        self.macro_cfg = macro_cfg or MacroContextConfig()
        self._macro_source_df = macro_df
        if self.macro_cfg.enabled and self.macro_cfg.window <= 0:
            raise ValueError("macro_cfg.window must be positive when macro context is enabled")
        if self.macro_cfg.enabled and self.macro_cfg.source not in {"resample", "prepared"}:
            raise ValueError("macro_cfg.source must be 'resample' or 'prepared'")
        if self.macro_cfg.enabled and self.macro_cfg.source == "prepared" and macro_df is None:
            raise ValueError("macro_cfg.source='prepared' requires macro_df")

        logger.info(f"MarketDataset Init: Processing {len(df)} bars. Step 1/5: Computing OHLCV features...")
        self._features_df = compute_ohlcv_features(
            df, include_volume=spec.include_volume, include_indicators=spec.include_indicators
        )

        logger.info("MarketDataset Init: Step 2/5: Computing Time embeddings...")
        self._time_features_df = compute_time_features(df)
        # Backward-compatible aliases used by existing training scripts and probes.
        self._features = self._features_df
        self._time_features = self._time_features_df
        self._macro_features_df: pd.DataFrame | None = None
        self._macro_features_arr: np.ndarray | None = None
        self._macro_primary_indices: np.ndarray | None = None
        if self.macro_cfg.enabled:
            logger.info(
                "MarketDataset Init: Computing causal macro context: source=%s rule=%s window=%d",
                self.macro_cfg.source,
                self.macro_cfg.resample_rule,
                self.macro_cfg.window,
            )
            self._macro_features_df, self._macro_primary_indices = self._build_macro_context()

        # Primary triple-barrier at smallest horizon; use horizon 16 by default
        primary = spec.horizons[len(spec.horizons) // 2] if spec.horizons else 16
        self._tb_cfg_primary = TripleBarrierConfig(
            tp_atr_mult=self.tb_cfg.tp_atr_mult,
            sl_atr_mult=self.tb_cfg.sl_atr_mult,
            max_holding=primary,
            atr_window=self.tb_cfg.atr_window,
        )

        def forward_cfg_for(horizon: int) -> ForwardReturnConfig:
            values = {
                "flat_return_bps": self.target_cfg.flat_return_bps,
                "flat_volatility_mult": self.target_cfg.flat_volatility_mult,
                "flat_min_bps": self.target_cfg.flat_min_bps,
                "flat_max_bps": self.target_cfg.flat_max_bps,
                "use_log_return": self.target_cfg.use_log_return,
            }
            overrides = self.target_cfg.horizon_overrides or {}
            override = overrides.get(int(horizon), {})
            values.update(override)
            return ForwardReturnConfig(
                horizon=int(horizon),
                flat_return_bps=float(values["flat_return_bps"]),
                flat_volatility_mult=float(values["flat_volatility_mult"]),
                flat_min_bps=float(values["flat_min_bps"]),
                flat_max_bps=float(values["flat_max_bps"]),
                log_return=bool(values["use_log_return"]),
            )

        if self._compute_targets:
            if self.target_cfg.direction_mode == "forward_return":
                logger.info("MarketDataset Init: Step 3/5: Computing Forward Return Labels (Returns/Directions)...")
                self._tb_df = forward_return_targets(df, forward_cfg_for(int(primary)))
            else:
                logger.info("MarketDataset Init: Step 3/5: Computing Triple Barrier Labels (Returns/Directions)...")
                self._tb_df = triple_barrier(df, self._tb_cfg_primary)

            self._tb_multi_dfs = []
            for horizon in self.spec.horizons:
                horizon = int(horizon)
                if horizon == int(primary):
                    self._tb_multi_dfs.append(self._tb_df)
                elif self.target_cfg.direction_mode == "forward_return":
                    self._tb_multi_dfs.append(
                        forward_return_targets(df, forward_cfg_for(horizon))
                    )
                else:
                    self._tb_multi_dfs.append(
                        triple_barrier(
                            df,
                            TripleBarrierConfig(
                                tp_atr_mult=self.tb_cfg.tp_atr_mult,
                                sl_atr_mult=self.tb_cfg.sl_atr_mult,
                                max_holding=horizon,
                                atr_window=self.tb_cfg.atr_window,
                            ),
                        )
                    )

            logger.info("MarketDataset Init: Step 4/5: Computing Realized Volatility...")
            self._vol_series = realized_volatility(df, horizon=primary)

            logger.info("MarketDataset Init: Step 5/5: Computing HMM Regime Labels (Macro States)...")
            self._regime_series = hmm_regime_labels(
                df, n_states=spec.n_regime_states, lookback=256, prefer_sklearn=True
            )
        else:
            logger.info("MarketDataset Init: Steps 3-5 skipped (compute_targets=False). Using dummy labels.")
            self._tb_df = pd.DataFrame({"label": np.zeros(len(df), dtype=np.int64), "ret": np.zeros(len(df), dtype=np.float32)}, index=df.index)
            self._tb_multi_dfs = [self._tb_df for _ in self.spec.horizons]
            self._vol_series = pd.Series(np.zeros(len(df), dtype=np.float32), index=df.index)
            self._regime_series = pd.Series(np.zeros(len(df), dtype=np.int64), index=df.index)

        # ---- Effective length (mirrors the original __len__ logic) ----
        self._horizon_max = max(self.spec.horizons) if self.spec.horizons else 0
        self._length = max(0, len(self.df) - self.spec.chart_window - self._horizon_max - 1)

        # ---- Hot-path numpy views ----
        if self._precompute:
            self._features_arr = np.ascontiguousarray(
                self._features_df.to_numpy(dtype=np.float32)
            )
            self._time_features_arr = np.ascontiguousarray(
                self._time_features_df.to_numpy(dtype=np.float32)
            )
            if self._macro_features_df is not None:
                self._macro_features_arr = np.ascontiguousarray(
                    self._macro_features_df.to_numpy(dtype=np.float32)
                )
            self._tb_label_arr = np.ascontiguousarray(
                self._tb_df["label"].to_numpy(dtype=np.int64)
            )
            self._tb_ret_arr = np.ascontiguousarray(
                self._tb_df["ret"].to_numpy(dtype=np.float32)
            )
            self._tb_multi_label_arr = np.ascontiguousarray(
                np.stack(
                    [
                        item["label"].to_numpy(dtype=np.int64)
                        for item in self._tb_multi_dfs
                    ],
                    axis=1,
                )
            )
            self._tb_multi_ret_arr = np.ascontiguousarray(
                np.stack(
                    [
                        item["ret"].to_numpy(dtype=np.float32)
                        for item in self._tb_multi_dfs
                    ],
                    axis=1,
                )
            )
            self._vol_arr = np.ascontiguousarray(
                self._vol_series.to_numpy(dtype=np.float32)
            )
            self._regime_arr = np.ascontiguousarray(
                self._regime_series.to_numpy(dtype=np.int64)
            )
        else:
            # Legacy views: still keep DataFrame references for debug/tests.
            self._features_arr = None
            self._time_features_arr = None
            self._macro_features_arr = None
            self._tb_label_arr = None
            self._tb_ret_arr = None
            self._tb_multi_label_arr = None
            self._tb_multi_ret_arr = None
            self._vol_arr = None
            self._regime_arr = None

        # ---- Chart cache (preallocated when cache_charts is true) ----
        if self._cache_charts:
            N = self._length
            H = W = self.spec.image_size
            self._chart_arr = np.empty((N, 3, H, W), dtype=np.float32)
            if N > 0:
                logger.info(
                    f"MarketDataset Init: Pre-rendering {N} chart windows "
                    f"({3*H*W*N/1024/1024:.1f} MB)..."
                )
                self._precompute_charts()
        else:
            self._chart_arr = None
            if self._chart_cache_size > 0:
                self._chart_cache: "OrderedDict[int, torch.Tensor]" = OrderedDict()
            else:
                self._chart_cache = {}

        # Advertise the fast path to the dataloader factory.
        self.__fast_getitem__ = bool(self._cache_charts and self._precompute)

        logger.info("MarketDataset Init: All tables processed and ready for DataLoader!")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _build_macro_context(self) -> tuple[pd.DataFrame, np.ndarray]:
        """Build closed higher-timeframe features and causal row mapping.

        The prepared market data is open-time indexed. A 1h candle stamped
        12:00 is not fully known while processing 15m bars inside 12:00-12:59,
        so a 15m sample at time ``t`` may only use ``floor(t, 1h) - 1h`` or
        earlier macro candles.
        """
        if self.macro_cfg.source == "prepared":
            if self._macro_source_df is None:
                raise ValueError("prepared macro context requires macro_df")
            if not isinstance(self._macro_source_df.index, pd.DatetimeIndex):
                raise ValueError("macro_df must have a DatetimeIndex")
            macro = self._macro_source_df[["open", "high", "low", "close", "volume"]].sort_index()
        else:
            ohlcv = self.df[["open", "high", "low", "close", "volume"]]
            macro = ohlcv.resample(
                self.macro_cfg.resample_rule,
                label="left",
                closed="left",
            ).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            })
        macro = macro.dropna(subset=["open", "high", "low", "close"])
        if len(macro) == 0:
            raise ValueError("macro context produced no closed candles")
        macro_features = compute_ohlcv_features(
            macro,
            include_volume=self.spec.include_volume,
            include_indicators=self.spec.include_indicators,
        )
        try:
            rule_delta = pd.Timedelta(self.macro_cfg.resample_rule)
        except ValueError as exc:
            raise ValueError(
                "macro_cfg.resample_rule must be a fixed pandas Timedelta, "
                f"got {self.macro_cfg.resample_rule!r}"
            ) from exc
        primary_index = pd.DatetimeIndex(self.df.index)
        allowed = primary_index.floor(self.macro_cfg.resample_rule) - rule_delta
        macro_ns = macro_features.index.view("int64")
        allowed_ns = allowed.view("int64")
        mapped = np.searchsorted(macro_ns, allowed_ns, side="right") - 1
        return macro_features, mapped.astype(np.int64, copy=False)

    def _precompute_charts(self) -> None:
        """Render every unique chart window once into ``self._chart_arr``."""
        N = self._length
        W = self.spec.chart_window
        df = self.df
        import time
        start_time = time.time()
        # Reuse the existing render_chart entry point for bit-exactness.
        # The cost is O(N * W) matplotlib calls; we do it once at init.
        for t in range(N):
            if t > 0 and t % 50000 == 0:
                elapsed = time.time() - start_time
                logger.info(f"MarketDataset Init: Rendered {t}/{N} charts ({t/N*100:.1f}%) in {elapsed:.1f}s...")
            if self._fast_render:
                img = render_chart_array(
                    self._ohlcv_arr[t : t + W], size=self.spec.image_size
                )
            else:
                window_df = df.iloc[t : t + W]
                img = render_chart(window_df, size=self.spec.image_size)
            # img is a (3, H, W) float32 tensor
            if isinstance(img, torch.Tensor):
                arr = img.detach().cpu().numpy()
            else:
                arr = np.asarray(img, dtype=np.float32)
            # Defensive: ensure correct shape/dtype
            if arr.shape != (3, self.spec.image_size, self.spec.image_size):
                arr = arr.reshape(3, self.spec.image_size, self.spec.image_size)
            self._chart_arr[t] = arr.astype(np.float32, copy=False)
        
        total_time = time.time() - start_time
        logger.info(f"MarketDataset Init: Successfully rendered all {N} charts in {total_time:.1f}s!")

    def _get_chart(self, t: int) -> torch.Tensor:
        """Return the chart tensor for sample ``t`` (precomputed or cached)."""
        if self._chart_arr is not None:
            return torch.from_numpy(self._chart_arr[t])
        # Lazy / LRU path
        cache = self._chart_cache
        if self._chart_cache_size >= 0 and t in cache:
            if self._chart_cache_size > 0:
                cache.move_to_end(t)
            return cache[t]
        start = t
        end = t + self.spec.chart_window
        if self._fast_render:
            img = render_chart_array(
                self._ohlcv_arr[start:end], size=self.spec.image_size
            )
        else:
            window_df = self.df.iloc[start:end]
            img = render_chart(window_df, size=self.spec.image_size)
        if self._chart_cache_size < 0:
            return img
        if self._chart_cache_size > 0:
            cache[t] = img
            cache.move_to_end(t)
            if len(cache) > self._chart_cache_size:
                cache.popitem(last=False)
        else:
            cache[t] = img
        return img

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self._length

    def _features_window(self, start: int, end: int) -> np.ndarray:
        if self._features_arr is not None:
            return self._features_arr[start:end]
        return self._features_df.iloc[start:end].to_numpy(dtype=np.float32)

    def _history_window(self, t: int, end: int) -> np.ndarray:
        hist_start = max(0, t - 256)
        if self._features_arr is not None:
            return self._features_arr[hist_start:end]
        return self._features_df.iloc[hist_start:end].to_numpy(dtype=np.float32)

    def _ctx_row(self, primary_idx: int) -> np.ndarray:
        if self._time_features_arr is not None:
            return self._time_features_arr[primary_idx]
        return self._time_features_df.iloc[primary_idx].to_numpy(dtype=np.float32)

    def _macro_window(self, primary_idx: int) -> np.ndarray | None:
        if not self.macro_cfg.enabled:
            return None
        if self._macro_primary_indices is None or self._macro_features_df is None:
            raise RuntimeError("macro context is enabled but macro tables are missing")
        idx = int(self._macro_primary_indices[primary_idx])
        window = int(self.macro_cfg.window)
        n_feat = int(self._macro_features_df.shape[1])
        if idx < 0:
            return np.zeros((window, n_feat), dtype=np.float32)
        features = self._macro_features_arr
        if features is None:
            features = self._macro_features_df.to_numpy(dtype=np.float32)
        start = max(0, idx - window + 1)
        end = idx + 1
        raw = features[start:end]
        hist_start = max(0, idx - 256)
        history = features[hist_start:end]
        normalized = normalize_feature_window(raw, history)
        if normalized.shape[0] < window:
            pad = np.zeros((window - normalized.shape[0], n_feat), dtype=np.float32)
            normalized = np.concatenate([pad, normalized], axis=0)
        return np.ascontiguousarray(normalized[-window:], dtype=np.float32)

    def _label_dir(self, primary_idx: int) -> int:
        if self._tb_label_arr is not None:
            return int(self._tb_label_arr[primary_idx])
        return int(self._tb_df["label"].iloc[primary_idx])

    def _label_ret(self, primary_idx: int) -> float:
        if self._tb_ret_arr is not None:
            return float(self._tb_ret_arr[primary_idx])
        return float(self._tb_df["ret"].iloc[primary_idx])

    def _label_dir_multi(self, primary_idx: int) -> np.ndarray:
        if self._tb_multi_label_arr is not None:
            return self._tb_multi_label_arr[primary_idx]
        return np.asarray(
            [int(item["label"].iloc[primary_idx]) for item in self._tb_multi_dfs],
            dtype=np.int64,
        )

    def _label_dir_multi_persistence(self, primary_idx: int) -> np.ndarray:
        labels = []
        for h_idx, horizon in enumerate(self.spec.horizons):
            horizon = int(horizon)
            if primary_idx < horizon:
                labels.append(0)
                continue
            past_idx = primary_idx - horizon
            if self._tb_multi_label_arr is not None:
                labels.append(int(self._tb_multi_label_arr[past_idx, h_idx]))
            else:
                labels.append(int(self._tb_multi_dfs[h_idx]["label"].iloc[past_idx]))
        return np.asarray(labels, dtype=np.int64)

    def _label_ret_multi(self, primary_idx: int) -> np.ndarray:
        if self._tb_multi_ret_arr is not None:
            return self._tb_multi_ret_arr[primary_idx]
        return np.asarray(
            [float(item["ret"].iloc[primary_idx]) for item in self._tb_multi_dfs],
            dtype=np.float32,
        )

    def _label_vol(self, primary_idx: int) -> float:
        if self._vol_arr is not None:
            v = float(self._vol_arr[primary_idx])
        else:
            v = float(self._vol_series.iloc[primary_idx])
        if np.isnan(v):
            return 0.0
        return v

    def _label_regime(self, primary_idx: int) -> int:
        if self._regime_arr is not None:
            return int(self._regime_arr[primary_idx])
        return int(self._regime_series.iloc[primary_idx])

    def __getitem__(self, t: int) -> dict:
        spec = self.spec
        start = t
        end = t + spec.chart_window
        primary_idx = end - 1

        feature_window = self._features_window(start, end)
        history_window = self._history_window(t, end)
        # Numeric features only (NaN -> 0, robust fill). The cyclic time
        # embeddings are placed in ``context`` (last bar) so the dataset's
        # ``numeric`` shape matches the env's contract — and the policy's
        # ``in_numeric_features`` default (32) is wire-compatible.
        num = normalize_feature_window(feature_window, history_window)

        chart = self._get_chart(t)
        ctx = self._ctx_row(primary_idx)
        macro_numeric = self._macro_window(primary_idx)
        lbl_dir = self._label_dir(primary_idx)
        lbl_ret = self._label_ret(primary_idx)
        lbl_dir_multi = self._label_dir_multi(primary_idx)
        lbl_dir_multi_persistence = self._label_dir_multi_persistence(primary_idx)
        primary_horizon_idx = len(spec.horizons) // 2 if spec.horizons else 0
        lbl_dir_persistence = int(lbl_dir_multi_persistence[primary_horizon_idx])
        lbl_ret_multi = self._label_ret_multi(primary_idx)
        lbl_vol = self._label_vol(primary_idx)
        lbl_risk = max(-lbl_ret, 0.0) + max(lbl_vol, 0.0)
        lbl_regime = self._label_regime(primary_idx)

        mask = np.ones(spec.chart_window, dtype=bool)
        sample = {
            "chart": chart,
            "numeric": torch.from_numpy(num),
            "context": torch.from_numpy(ctx),
            "label_dir": torch.tensor(lbl_dir, dtype=torch.long),
            "label_dir_persistence": torch.tensor(lbl_dir_persistence, dtype=torch.long),
            "label_ret": torch.tensor(lbl_ret, dtype=torch.float32),
            "label_dir_multi": torch.from_numpy(lbl_dir_multi.astype(np.int64, copy=False)),
            "label_dir_multi_persistence": torch.from_numpy(
                lbl_dir_multi_persistence.astype(np.int64, copy=False)
            ),
            "label_ret_multi": torch.from_numpy(lbl_ret_multi.astype(np.float32, copy=False)),
            "label_vol": torch.tensor(lbl_vol, dtype=torch.float32),
            "label_risk": torch.tensor(lbl_risk, dtype=torch.float32),
            "label_regime": torch.tensor(lbl_regime, dtype=torch.long),
            "mask": torch.from_numpy(mask),
            "meta": {
                "ts": str(self.df.index[primary_idx]),
                "t": int(primary_idx),
                "instrument": getattr(self.df, "name", "unknown"),
            },
        }
        if macro_numeric is not None:
            sample["macro_numeric"] = torch.from_numpy(macro_numeric)
        return sample


def multimodal_collate(batch: Sequence[dict]) -> MultimodalBatch:
    """Default collate: stack all tensors, list metas."""
    keys_tensor = ("chart", "numeric", "context", "label_dir", "label_dir_persistence", "label_ret",
                   "label_vol", "label_risk", "label_regime", "mask",
                   "label_dir_multi", "label_dir_multi_persistence", "label_ret_multi")
    out: dict = {}
    for k in keys_tensor:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    if "macro_numeric" in batch[0]:
        out["macro_numeric"] = torch.stack([b["macro_numeric"] for b in batch], dim=0)
    out["meta"] = [b["meta"] for b in batch]
    return MultimodalBatch(**out)
