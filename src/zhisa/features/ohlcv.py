"""OHLCV-derived numeric features plus optional market-context features."""
from __future__ import annotations

import re
from typing import List, Optional

import numpy as np
import pandas as pd

from zhisa.features.indicators import (
    atr,
    bollinger,
    donchian,
    ema,
    rsi,
    sma,
    vwap_session,
)


_OHLCV_COLUMNS = {"open", "high", "low", "close", "volume"}
_PRICE_CONTEXT_SUFFIXES = (
    "_open",
    "_high",
    "_low",
    "_close",
    "_price",
    "_mark_price",
)
_DIRECT_CONTEXT_COLUMNS = {
    "funding_rate",
    "premium_index",
    "global_long_account",
    "global_short_account",
    "top_account_long_account",
    "top_account_short_account",
    "top_position_long_account",
    "top_position_short_account",
}
_RATIO_CONTEXT_COLUMNS = {
    "global_long_short_ratio",
    "top_account_long_short_ratio",
    "top_position_long_short_ratio",
    "long_short_ratio",
    "top_trader_long_short_ratio",
    "taker_buy_sell_ratio",
}
_POSITIVE_SIZE_CONTEXT_COLUMNS = {
    "open_interest",
    "open_interest_value",
    "cmc_circulating_supply",
    "futures_volume",
    "futures_quote_volume",
    "trades",
    "kline_taker_buy_volume",
    "kline_taker_sell_volume",
    "kline_taker_buy_quote_volume",
    "kline_taker_sell_quote_volume",
    "taker_buy_volume",
    "taker_sell_volume",
}


def _safe_log(x: pd.Series) -> pd.Series:
    return np.log(x.replace(0, np.nan))


def _clean_context_feature_name(name: str) -> str:
    clean = re.sub(r"[^0-9a-zA-Z_]+", "_", str(name).strip().lower())
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean or "unknown"


def _ctx_unprefixed(name: str) -> str:
    """Canonical context feature name, independent of the ``ctx_`` prefix.

    The prepared-data merger stores context columns as ``ctx_funding_rate``,
    ``ctx_open_interest`` … while the ad-hoc live join exposes them without a
    prefix. Matching on the *cleaned* name (prefix stripped) makes both paths
    behave identically for the known funding/OI/ratio indicators.
    """
    return re.sub(r"^ctx_", "", _clean_context_feature_name(name)) or "unknown"


def _seven_day_window(index) -> int:
    """Rolling window for the '7d' context z-scores in bars.

    The original 2016 constant is 7 days of 5m bars. For a 1h / 15m merged
    context the same 2016 bars would silently mean 84 / 21 days, i.e. the
    z-score would stop being a 7-day metric. Derive it from the index
    frequency instead (5m stays 2016 -> identical behaviour).
    """
    if len(index) < 2:
        return 2016
    delta = (pd.DatetimeIndex(index)[1:] - pd.DatetimeIndex(index)[:-1]).median()
    if not isinstance(delta, pd.Timedelta) or delta <= pd.Timedelta(0):
        return 2016
    return max(1, int(pd.Timedelta(days=7) / delta))


def _numeric_context_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        name = str(col)
        if name in _OHLCV_COLUMNS or name == "timestamp":
            continue
        cols.append(name)
    return cols


def _log1p_signed(series: pd.Series) -> pd.Series:
    return np.sign(series) * np.log1p(series.abs())


def _safe_log_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    return np.log((numer.clip(lower=0.0) + 1e-12) / (denom.clip(lower=0.0) + 1e-12))


def _add_market_context_features(
    out: pd.DataFrame,
    df: pd.DataFrame,
    close: pd.Series,
    volume: pd.Series | None,
) -> None:
    """Append causal futures/market-context features when extra columns exist."""
    context_cols = _numeric_context_columns(df)
    if not context_cols:
        return

    # Map the canonical (prefix-stripped) name -> raw column so BOTH an
    # unprefixed live join and the ctx_-prefixed prepared merge are handled:
    #   ctx_funding_rate, funding_rate  ->  funding_rate
    by_clean: dict[str, str] = {}
    for col in context_cols:
        key = _ctx_unprefixed(col)
        by_clean.setdefault(key, col)
    cleaned = list(by_clean.keys())

    handled: set[str] = set()
    availability = pd.Series(0.0, index=df.index, dtype=float)
    for col in context_cols:
        availability += pd.to_numeric(df[col], errors="coerce").notna().astype(float)
    out["ctx_available_frac"] = availability / max(len(context_cols), 1)

    def series(col: str) -> pd.Series:
        return pd.to_numeric(df[col], errors="coerce")

    def add(name: str, values: pd.Series) -> None:
        out[f"ctx_{_clean_context_feature_name(name)}"] = values

    price_cols = [
        key for key in cleaned
        if key.endswith(_PRICE_CONTEXT_SUFFIXES)
        or key in {"mark_price", "index_price", "funding_mark_price"}
    ]
    for key in price_cols:
        raw = by_clean[key]
        values = series(raw)
        add(f"{key}_basis", (values - close) / (close + 1e-12))
        handled.add(raw)

    for key in cleaned:
        raw = by_clean[key]
        values = series(raw)
        if key in _DIRECT_CONTEXT_COLUMNS:
            add(key, values)
            handled.add(raw)
        elif key in _RATIO_CONTEXT_COLUMNS:
            add(f"{key}_log", np.log(values.clip(lower=1e-12)))
            handled.add(raw)
        elif key in _POSITIVE_SIZE_CONTEXT_COLUMNS:
            positive = values.clip(lower=0.0)
            add(f"{key}_log1p", np.log1p(positive))
            add(f"{key}_logret_1", _safe_log(positive).diff())
            handled.add(raw)

    # Specific computed features for known critical indicators. The rolling
    # window is a real 7 days at the bar frequency of the frame.
    win = _seven_day_window(df.index)
    mp = 12
    if "funding_rate" in by_clean:
        funding_series = series(by_clean["funding_rate"])
        fr_mean = funding_series.rolling(win, min_periods=mp).mean()
        fr_std = funding_series.rolling(win, min_periods=mp).std()
        out["ctx_funding_zscore_7d"] = (funding_series - fr_mean) / (fr_std + 1e-12)

    if "open_interest" in by_clean:
        oi_series = series(by_clean["open_interest"])
        oi_mean = oi_series.rolling(win, min_periods=mp).mean()
        oi_std = oi_series.rolling(win, min_periods=mp).std()
        out["ctx_oi_zscore_7d"] = (oi_series - oi_mean) / (oi_std + 1e-12)

    if "top_trader_long_short_ratio" in by_clean:
        ls_series = series(by_clean["top_trader_long_short_ratio"])
        ls_mean = ls_series.rolling(win, min_periods=mp).mean()
        ls_std = ls_series.rolling(win, min_periods=mp).std()
        out["ctx_ls_zscore_7d"] = (ls_series - ls_mean) / (ls_std + 1e-12)

    if {"taker_buy_volume", "taker_sell_volume"}.issubset(by_clean):
        buy = series(by_clean["taker_buy_volume"])
        sell = series(by_clean["taker_sell_volume"])
        total = buy + sell
        add("taker_imbalance", (buy - sell) / (total + 1e-12))
        add("taker_buy_sell_log_ratio", _safe_log_ratio(buy, sell))
        handled.update({by_clean["taker_buy_volume"], by_clean["taker_sell_volume"]})

    if {"kline_taker_buy_volume", "kline_taker_sell_volume"}.issubset(by_clean):
        buy = series(by_clean["kline_taker_buy_volume"])
        sell = series(by_clean["kline_taker_sell_volume"])
        total = buy + sell
        add("kline_taker_imbalance", (buy - sell) / (total + 1e-12))
        add("kline_taker_buy_sell_log_ratio", _safe_log_ratio(buy, sell))
        handled.update({by_clean["kline_taker_buy_volume"], by_clean["kline_taker_sell_volume"]})

    if "volume_delta" in by_clean:
        delta = series(by_clean["volume_delta"])
        if "taker_buy_volume" in by_clean and "taker_sell_volume" in by_clean:
            denom = series(by_clean["taker_buy_volume"]) + series(by_clean["taker_sell_volume"])
        elif volume is not None:
            denom = volume
        else:
            denom = delta.abs()
        add("volume_delta_imbalance", delta / (denom.abs() + 1e-12))
        handled.add(by_clean["volume_delta"])

    if "open_interest" in by_clean and volume is not None:
        oi = series(by_clean["open_interest"]).clip(lower=0.0)
        add("open_interest_over_volume_log", np.log1p(oi / (volume.clip(lower=0.0) + 1e-12)))
        handled.add(by_clean["open_interest"])

    # Keep unknown numeric context columns visible, but transform them into
    # scale-tolerant signed log features so one large raw value cannot dominate.
    for col in context_cols:
        if col in handled:
            continue
        name = _ctx_unprefixed(col)
        values = series(col)
        add(f"{name}_signed_log1p", _log1p_signed(values))


def compute_ohlcv_features(
    df: pd.DataFrame,
    *,
    include_volume: bool = True,
    include_indicators: bool = True,
    ema_periods: Optional[List[int]] = None,
    sma_periods: Optional[List[int]] = None,
    atr_period: int = 14,
    rsi_period: int = 14,
    bb_period: int = 20,
    donchian_period: int = 20,
    include_market_context: bool = True,
) -> pd.DataFrame:
    """Build a feature matrix aligned to ``df.index``.

    Features include log-returns at several lags, range, body/wick ratios,
    volume z-score, and a configurable set of moving-average / oscillator
    indicators. All values are returned as a numeric DataFrame with no
    look-ahead: indicators use only past data within the same frame.
    """
    ema_periods = ema_periods or [8, 21, 55]
    sma_periods = sma_periods or [10, 20, 50]
    out = pd.DataFrame(index=df.index)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    vol = df.get("volume", None)

    # Returns
    log_close = _safe_log(close)
    for lag in (1, 2, 4, 8, 16):
        out[f"logret_{lag}"] = log_close.diff(lag)

    # Range, body, wick ratios
    rng = (high - low).replace(0, np.nan)
    out["body_over_range"] = (close - open_).abs() / rng
    out["upper_wick_over_range"] = (high - np.maximum(close, open_)) / rng
    out["lower_wick_over_range"] = (np.minimum(close, open_) - low) / rng
    out["close_over_open"] = (close - open_) / (open_ + 1e-12)
    out["hl_over_close"] = (high - low) / close

    # Volatility
    log_ret1 = log_close.diff()
    for w in (8, 16, 32):
        out[f"rv_{w}"] = log_ret1.rolling(w, min_periods=2).std()
    if include_indicators:
        a = atr(df, period=atr_period)
        out[f"atr_{atr_period}"] = a
        out[f"atr_pct_{atr_period}"] = a / close

    # Volume
    if include_volume and vol is not None:
        for w in (16, 64):
            mv = vol.rolling(w, min_periods=2).mean()
            sv = vol.rolling(w, min_periods=2).std()
            out[f"vol_z_{w}"] = (vol - mv) / (sv + 1e-12)
        out["vol_over_avg_16"] = vol / (vol.rolling(16, min_periods=1).mean() + 1e-12)

    # Moving averages: distance to MA, slope, cross
    if include_indicators:
        for p in sma_periods:
            s = sma(close, p)
            out[f"sma_dist_{p}"] = (close - s) / (close + 1e-12)
            out[f"sma_slope_{p}"] = (s - s.shift(p // 2 or 1)) / (close + 1e-12)
        for p in ema_periods:
            e = ema(close, p)
            out[f"ema_dist_{p}"] = (close - e) / (close + 1e-12)
        bb = bollinger(close, period=bb_period)
        out[f"bb_pct_b_{bb_period}"] = bb["pct_b"]
        out[f"bb_bw_{bb_period}"] = bb["bandwidth"]
        out["rsi_" + str(rsi_period)] = rsi(close, rsi_period) / 100.0
        don = donchian(high, low, period=donchian_period)
        out[f"don_pos_{donchian_period}"] = (close - don["low"]) / (don["high"] - don["low"] + 1e-12)
        # VWAP distance
        vwap = vwap_session(df)
        out["vwap_dist"] = (close - vwap) / (close + 1e-12)

    if include_market_context:
        _add_market_context_features(out, df, close, vol)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out.astype(np.float64)


def normalize_feature_window(
    feature_window: np.ndarray,
    history_window: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Normalize a feature window using the mean and std of a trailing history window.

    Robustly handles NaNs/Infs and returns a clean float32 array.
    """
    hist_clean = np.nan_to_num(history_window, nan=0.0, posinf=0.0, neginf=0.0)
    feat_clean = np.nan_to_num(feature_window, nan=0.0, posinf=0.0, neginf=0.0)

    mu = hist_clean.mean(axis=0)
    sd = hist_clean.std(axis=0) + eps

    normed = (feat_clean - mu) / sd
    return np.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
