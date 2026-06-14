"""Ingest public Binance USD-M futures context data.

This command is data-only. It uses public market-data endpoints, does not
require API keys, and has no order-placement code path.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from zhisa.scripts._real_data import parse_utc_timestamp
from zhisa.storage.schema import Timeframe


BASE_URL = "https://fapi.binance.com"

KLINE_LIMIT = 1500
STATS_LIMIT = 500
FUNDING_LIMIT = 1000


JsonFetcher = Callable[[str, dict[str, Any], str, float], Any]


@dataclass(frozen=True)
class EndpointSpec:
    name: str
    path: str
    parser: str
    params: dict[str, Any]
    limit: int


def _symbol_to_binance(symbol: str) -> str:
    """Convert common CCXT-ish symbols to Binance USD-M symbols."""
    clean = str(symbol).strip().upper()
    if ":" in clean:
        clean = clean.split(":", 1)[0]
    return clean.replace("/", "").replace("-", "").replace("_", "")


def _symbol_slug(symbol: str) -> str:
    return _symbol_to_binance(symbol)


def _timestamp_ms(value: str | None) -> int | None:
    ts = parse_utc_timestamp(value)
    if ts is None:
        return None
    return int(ts.timestamp() * 1000)


def _timeframe_ms(timeframe: str) -> int:
    return int(Timeframe.from_str(timeframe).minutes * 60_000)


def _fetch_json(
    path: str,
    params: dict[str, Any],
    base_url: str = BASE_URL,
    timeout: float = 30.0,
) -> Any:
    query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    url = f"{base_url}{path}?{query}" if query else f"{base_url}{path}"
    request = urllib.request.Request(url, headers={"User-Agent": "zhisa-futures-context/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw)


def _fetch_forward(
    spec: EndpointSpec,
    *,
    start_ms: int,
    end_ms: int,
    interval_ms: int,
    fetcher: JsonFetcher = _fetch_json,
    base_url: str = BASE_URL,
    timeout: float = 30.0,
    sleep_s: float = 0.05,
) -> list[Any]:
    rows: list[Any] = []
    cursor = int(start_ms)
    while cursor <= end_ms:
        params = {
            **spec.params,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": spec.limit,
        }
        data = fetcher(spec.path, params, base_url, timeout)
        if not isinstance(data, list):
            raise RuntimeError(f"{spec.name} returned non-list payload: {type(data).__name__}")
        if not data:
            break
        rows.extend(data)
        last_ts = _row_timestamp_ms(data[-1], spec.parser)
        if last_ts is None or last_ts < cursor:
            break
        cursor = int(last_ts) + max(1, interval_ms)
        if len(data) < spec.limit:
            break
        if sleep_s > 0:
            time.sleep(sleep_s)
    return rows


def _fetch_windowed_stats(
    spec: EndpointSpec,
    *,
    start_ms: int,
    end_ms: int,
    interval_ms: int,
    fetcher: JsonFetcher = _fetch_json,
    base_url: str = BASE_URL,
    timeout: float = 30.0,
    sleep_s: float = 0.05,
) -> list[Any]:
    """Fetch futures/data metrics in bounded windows.

    Binance's futures/data endpoints can behave like "latest rows before
    endTime" when a very large window and small limit are supplied. Keeping
    each request <= limit bars makes the returned timestamps unambiguous.
    """
    rows: list[Any] = []
    cursor = int(start_ms)
    window_ms = max(interval_ms, (spec.limit - 1) * interval_ms)
    while cursor <= end_ms:
        chunk_end = min(end_ms, cursor + window_ms)
        params = {
            **spec.params,
            "startTime": cursor,
            "endTime": chunk_end,
            "limit": spec.limit,
        }
        data = fetcher(spec.path, params, base_url, timeout)
        if not isinstance(data, list):
            raise RuntimeError(f"{spec.name} returned non-list payload: {type(data).__name__}")
        rows.extend(data)
        cursor = chunk_end + interval_ms
        if sleep_s > 0:
            time.sleep(sleep_s)
    return rows


def _row_timestamp_ms(row: Any, parser: str) -> int | None:
    if parser in {"klines", "mark_klines", "index_klines", "premium_klines"}:
        return int(row[0]) if isinstance(row, list) and row else None
    if isinstance(row, dict):
        if parser == "funding":
            return int(row["fundingTime"])
        return int(row["timestamp"])
    return None


def _dedupe_rows(rows: Iterable[Any], parser: str) -> list[Any]:
    by_ts: dict[int, Any] = {}
    for row in rows:
        ts = _row_timestamp_ms(row, parser)
        if ts is not None:
            by_ts[int(ts)] = row
    return [by_ts[k] for k in sorted(by_ts)]


def _to_index(values: list[int]) -> pd.DatetimeIndex:
    return pd.to_datetime(values, unit="ms", utc=True)


def _numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _parse_futures_klines(rows: list[Any]) -> pd.DataFrame:
    rows = _dedupe_rows(rows, "klines")
    index = _to_index([int(row[0]) for row in rows])
    frame = pd.DataFrame(
        {
            "futures_open": [_numeric(row[1]) for row in rows],
            "futures_high": [_numeric(row[2]) for row in rows],
            "futures_low": [_numeric(row[3]) for row in rows],
            "futures_close": [_numeric(row[4]) for row in rows],
            "futures_volume": [_numeric(row[5]) for row in rows],
            "futures_quote_volume": [_numeric(row[7]) for row in rows],
            "trades": [int(row[8]) for row in rows],
            "kline_taker_buy_volume": [_numeric(row[9]) for row in rows],
            "kline_taker_buy_quote_volume": [_numeric(row[10]) for row in rows],
        },
        index=index,
    )
    frame["kline_taker_sell_volume"] = frame["futures_volume"] - frame["kline_taker_buy_volume"]
    frame["kline_taker_sell_quote_volume"] = frame["futures_quote_volume"] - frame["kline_taker_buy_quote_volume"]
    return frame


def _parse_price_klines(rows: list[Any], prefix: str) -> pd.DataFrame:
    parser = {
        "mark": "mark_klines",
        "index": "index_klines",
        "premium": "premium_klines",
    }[prefix]
    rows = _dedupe_rows(rows, parser)
    index = _to_index([int(row[0]) for row in rows])
    frame = pd.DataFrame(
        {
            f"{prefix}_open": [_numeric(row[1]) for row in rows],
            f"{prefix}_high": [_numeric(row[2]) for row in rows],
            f"{prefix}_low": [_numeric(row[3]) for row in rows],
            f"{prefix}_close": [_numeric(row[4]) for row in rows],
        },
        index=index,
    )
    if prefix == "mark":
        frame["mark_price"] = frame["mark_close"]
    elif prefix == "index":
        frame["index_price"] = frame["index_close"]
    elif prefix == "premium":
        frame["premium_index"] = frame["premium_close"]
    return frame


def _parse_funding(rows: list[Any], interval_ms: int) -> pd.DataFrame:
    rows = _dedupe_rows(rows, "funding")
    if not rows:
        return pd.DataFrame()
    raw_index = pd.to_datetime([int(row["fundingTime"]) for row in rows], unit="ms", utc=True)
    floored = raw_index.floor(f"{interval_ms // 60_000}min")
    frame = pd.DataFrame(
        {
            "funding_rate": [_numeric(row.get("fundingRate")) for row in rows],
            "funding_mark_price": [_numeric(row.get("markPrice")) for row in rows],
        },
        index=floored,
    )
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def _parse_open_interest(rows: list[Any]) -> pd.DataFrame:
    rows = _dedupe_rows(rows, "stats")
    index = _to_index([int(row["timestamp"]) for row in rows])
    return pd.DataFrame(
        {
            "open_interest": [_numeric(row.get("sumOpenInterest")) for row in rows],
            "open_interest_value": [_numeric(row.get("sumOpenInterestValue")) for row in rows],
            "cmc_circulating_supply": [_numeric(row.get("CMCCirculatingSupply")) for row in rows],
        },
        index=index,
    )


def _parse_long_short(rows: list[Any], prefix: str) -> pd.DataFrame:
    rows = _dedupe_rows(rows, "stats")
    index = _to_index([int(row["timestamp"]) for row in rows])
    return pd.DataFrame(
        {
            f"{prefix}_long_account": [_numeric(row.get("longAccount")) for row in rows],
            f"{prefix}_short_account": [_numeric(row.get("shortAccount")) for row in rows],
            f"{prefix}_long_short_ratio": [_numeric(row.get("longShortRatio")) for row in rows],
        },
        index=index,
    )


def _parse_taker_ratio(rows: list[Any]) -> pd.DataFrame:
    rows = _dedupe_rows(rows, "stats")
    index = _to_index([int(row["timestamp"]) for row in rows])
    frame = pd.DataFrame(
        {
            "taker_buy_volume": [_numeric(row.get("buyVol")) for row in rows],
            "taker_sell_volume": [_numeric(row.get("sellVol")) for row in rows],
            "taker_buy_sell_ratio": [_numeric(row.get("buySellRatio")) for row in rows],
        },
        index=index,
    )
    frame["volume_delta"] = frame["taker_buy_volume"] - frame["taker_sell_volume"]
    return frame


def _summary_frame(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(df)),
        "start": str(df.index[0]) if len(df) else None,
        "end": str(df.index[-1]) if len(df) else None,
        "columns": list(df.columns),
    }


def _quality_audit(df: pd.DataFrame, timeframe: str) -> dict[str, Any]:
    if df.empty:
        return {
            "rows": 0,
            "start": None,
            "end": None,
            "expected_rows": 0,
            "missing_rows": 0,
            "coverage_pct": 0.0,
            "duplicate_index": 0,
            "monotonic_index": True,
            "non_null_by_column": {},
            "null_by_column": {},
            "first_non_null_by_column": {},
            "last_non_null_by_column": {},
        }
    freq = Timeframe.from_str(timeframe).pandas_freq
    expected = pd.date_range(df.index[0], df.index[-1], freq=freq, tz="UTC")
    missing = expected.difference(df.index)
    non_null = {col: int(df[col].notna().sum()) for col in df.columns}
    nulls = {col: int(df[col].isna().sum()) for col in df.columns}
    first_non_null = {}
    last_non_null = {}
    for col in df.columns:
        valid = df.index[df[col].notna()]
        first_non_null[col] = str(valid[0]) if len(valid) else None
        last_non_null[col] = str(valid[-1]) if len(valid) else None
    return {
        "rows": int(len(df)),
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
        "expected_rows": int(len(expected)),
        "missing_rows": int(len(missing)),
        "missing_preview": [str(x) for x in missing[:10]],
        "coverage_pct": float(100.0 * (1.0 - len(missing) / max(len(expected), 1))),
        "duplicate_index": int(df.index.duplicated().sum()),
        "monotonic_index": bool(df.index.is_monotonic_increasing),
        "non_null_by_column": non_null,
        "null_by_column": nulls,
        "first_non_null_by_column": first_non_null,
        "last_non_null_by_column": last_non_null,
    }


def _endpoint_specs(binance_symbol: str, timeframe: str) -> list[EndpointSpec]:
    return [
        EndpointSpec(
            name="futures_klines",
            path="/fapi/v1/klines",
            parser="klines",
            params={"symbol": binance_symbol, "interval": timeframe},
            limit=KLINE_LIMIT,
        ),
        EndpointSpec(
            name="mark_price_klines",
            path="/fapi/v1/markPriceKlines",
            parser="mark_klines",
            params={"symbol": binance_symbol, "interval": timeframe},
            limit=KLINE_LIMIT,
        ),
        EndpointSpec(
            name="index_price_klines",
            path="/fapi/v1/indexPriceKlines",
            parser="index_klines",
            params={"pair": binance_symbol, "interval": timeframe},
            limit=KLINE_LIMIT,
        ),
        EndpointSpec(
            name="premium_index_klines",
            path="/fapi/v1/premiumIndexKlines",
            parser="premium_klines",
            params={"symbol": binance_symbol, "interval": timeframe},
            limit=KLINE_LIMIT,
        ),
        EndpointSpec(
            name="funding_rate",
            path="/fapi/v1/fundingRate",
            parser="funding",
            params={"symbol": binance_symbol},
            limit=FUNDING_LIMIT,
        ),
        EndpointSpec(
            name="open_interest_hist",
            path="/futures/data/openInterestHist",
            parser="stats",
            params={"symbol": binance_symbol, "period": timeframe},
            limit=STATS_LIMIT,
        ),
        EndpointSpec(
            name="global_long_short",
            path="/futures/data/globalLongShortAccountRatio",
            parser="stats",
            params={"symbol": binance_symbol, "period": timeframe},
            limit=STATS_LIMIT,
        ),
        EndpointSpec(
            name="top_account_long_short",
            path="/futures/data/topLongShortAccountRatio",
            parser="stats",
            params={"symbol": binance_symbol, "period": timeframe},
            limit=STATS_LIMIT,
        ),
        EndpointSpec(
            name="top_position_long_short",
            path="/futures/data/topLongShortPositionRatio",
            parser="stats",
            params={"symbol": binance_symbol, "period": timeframe},
            limit=STATS_LIMIT,
        ),
        EndpointSpec(
            name="taker_long_short",
            path="/futures/data/takerlongshortRatio",
            parser="stats",
            params={"symbol": binance_symbol, "period": timeframe},
            limit=STATS_LIMIT,
        ),
    ]


def _parse_component(name: str, rows: list[Any], interval_ms: int) -> pd.DataFrame:
    if name == "futures_klines":
        return _parse_futures_klines(rows)
    if name == "mark_price_klines":
        return _parse_price_klines(rows, "mark")
    if name == "index_price_klines":
        return _parse_price_klines(rows, "index")
    if name == "premium_index_klines":
        return _parse_price_klines(rows, "premium")
    if name == "funding_rate":
        return _parse_funding(rows, interval_ms)
    if name == "open_interest_hist":
        return _parse_open_interest(rows)
    if name == "global_long_short":
        return _parse_long_short(rows, "global")
    if name == "top_account_long_short":
        return _parse_long_short(rows, "top_account")
    if name == "top_position_long_short":
        return _parse_long_short(rows, "top_position")
    if name == "taker_long_short":
        return _parse_taker_ratio(rows)
    raise ValueError(f"Unknown component: {name}")


def _probe_force_orders(
    binance_symbol: str,
    *,
    fetcher: JsonFetcher = _fetch_json,
    base_url: str = BASE_URL,
    timeout: float = 30.0,
) -> dict[str, Any]:
    params = {"symbol": binance_symbol, "limit": 1}
    try:
        data = fetcher("/fapi/v1/allForceOrders", params, base_url, timeout)
        rows = len(data) if isinstance(data, list) else 1
        return {"endpoint": "allForceOrders", "available": True, "rows": int(rows)}
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return {
            "endpoint": "allForceOrders",
            "available": False,
            "http_status": int(exc.code),
            "message": body[:500],
        }
    except Exception as exc:
        return {
            "endpoint": "allForceOrders",
            "available": False,
            "error_type": type(exc).__name__,
            "message": str(exc)[:500],
        }


def download_symbol_context(
    symbol: str,
    *,
    timeframe: str,
    start_ms: int,
    end_ms: int,
    fetcher: JsonFetcher = _fetch_json,
    base_url: str = BASE_URL,
    timeout: float = 30.0,
    sleep_s: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    binance_symbol = _symbol_to_binance(symbol)
    interval_ms = _timeframe_ms(timeframe)
    components: dict[str, pd.DataFrame] = {}
    endpoint_status: dict[str, Any] = {}

    for spec in _endpoint_specs(binance_symbol, timeframe):
        try:
            if spec.parser == "stats":
                rows = _fetch_windowed_stats(
                    spec,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    interval_ms=interval_ms,
                    fetcher=fetcher,
                    base_url=base_url,
                    timeout=timeout,
                    sleep_s=sleep_s,
                )
            else:
                step_ms = interval_ms if spec.parser != "funding" else 8 * 60 * 60_000
                rows = _fetch_forward(
                    spec,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    interval_ms=step_ms,
                    fetcher=fetcher,
                    base_url=base_url,
                    timeout=timeout,
                    sleep_s=sleep_s,
                )
            frame = _parse_component(spec.name, rows, interval_ms)
            if not frame.empty:
                start_ts = pd.to_datetime(start_ms, unit="ms", utc=True)
                end_ts = pd.to_datetime(end_ms, unit="ms", utc=True)
                frame = frame[(frame.index >= start_ts) & (frame.index <= end_ts)]
            components[spec.name] = frame
            endpoint_status[spec.name] = {"ok": True, **_summary_frame(frame)}
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            components[spec.name] = pd.DataFrame()
            endpoint_status[spec.name] = {
                "ok": False,
                "http_status": int(exc.code),
                "message": body[:500],
            }
        except Exception as exc:
            components[spec.name] = pd.DataFrame()
            endpoint_status[spec.name] = {
                "ok": False,
                "error_type": type(exc).__name__,
                "message": str(exc)[:500],
            }

    base = components.get("futures_klines", pd.DataFrame()).copy()
    if base.empty:
        non_empty = [frame for frame in components.values() if not frame.empty]
        if non_empty:
            full_index = non_empty[0].index
            for frame in non_empty[1:]:
                full_index = full_index.union(frame.index)
            base = pd.DataFrame(index=full_index.sort_values())
        else:
            base = pd.DataFrame()

    out = base.sort_index()
    for name, frame in components.items():
        if name == "futures_klines" or frame.empty:
            continue
        if name == "funding_rate":
            aligned = frame.reindex(out.index.union(frame.index)).sort_index().ffill().reindex(out.index)
        else:
            aligned = frame.reindex(out.index)
        out = out.join(aligned, how="left")

    if "global_long_short_ratio" in out.columns and "long_short_ratio" not in out.columns:
        out["long_short_ratio"] = out["global_long_short_ratio"]
    if "top_account_long_short_ratio" in out.columns and "top_trader_long_short_ratio" not in out.columns:
        out["top_trader_long_short_ratio"] = out["top_account_long_short_ratio"]

    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.index.name = "timestamp"

    audit = {
        "symbol": symbol,
        "binance_symbol": binance_symbol,
        "timeframe": timeframe,
        "requested_start": str(pd.to_datetime(start_ms, unit="ms", utc=True)),
        "requested_end": str(pd.to_datetime(end_ms, unit="ms", utc=True)),
        "safety": {
            "real_orders_enabled": False,
            "api_keys_required": False,
            "source": "public_binance_usdm_market_data",
        },
        "quality": _quality_audit(out, timeframe),
        "endpoints": endpoint_status,
        "unavailable_or_partial": {
            "liquidations_rest": _probe_force_orders(
                binance_symbol,
                fetcher=fetcher,
                base_url=base_url,
                timeout=timeout,
            )
        },
    }
    return out, audit


def _parse_symbols(raw: list[str]) -> list[str]:
    symbols: list[str] = []
    for part in raw:
        symbols.extend([chunk.strip() for chunk in part.split(",") if chunk.strip()])
    return symbols


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest public Binance USD-M futures context data.")
    parser.add_argument("--symbols", nargs="+", default=["BTC/USDT"], help="Symbols, comma-separated or space-separated.")
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--start", type=str, default=None, help="Inclusive UTC start timestamp.")
    parser.add_argument("--end", type=str, default=None, help="Inclusive UTC end timestamp.")
    parser.add_argument("--out-root", type=str, default="data/futures_context/binance_usdm")
    parser.add_argument("--audit-out", type=str, default="artifacts/real_data/binance_futures_context_audit.json")
    parser.add_argument("--base-url", type=str, default=BASE_URL)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--sleep", type=float, default=0.05, help="Polite pause between paginated public requests.")
    args = parser.parse_args(argv)

    end_ts = parse_utc_timestamp(args.end) or pd.Timestamp.utcnow()
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    start_ts = parse_utc_timestamp(args.start)
    if start_ts is None:
        start_ts = end_ts - pd.Timedelta(days=7)

    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)
    if start_ms >= end_ms:
        raise ValueError("--start must be earlier than --end")

    out_root = Path(args.out_root)
    summaries: list[dict[str, Any]] = []
    for symbol in _parse_symbols(args.symbols):
        df, audit = download_symbol_context(
            symbol,
            timeframe=args.timeframe,
            start_ms=start_ms,
            end_ms=end_ms,
            base_url=args.base_url,
            timeout=float(args.timeout),
            sleep_s=float(args.sleep),
        )
        symbol_dir = out_root / _symbol_slug(symbol) / args.timeframe
        symbol_dir.mkdir(parents=True, exist_ok=True)
        data_path = symbol_dir / "context.parquet"
        meta_path = symbol_dir / "meta.json"
        df.to_parquet(data_path, engine="pyarrow", index=True)
        audit["path"] = str(data_path)
        audit["meta_path"] = str(meta_path)
        meta_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        summaries.append(audit)

    payload = {
        "mode": "public_binance_usdm_context_ingest_no_orders",
        "symbols": summaries,
        "out_root": str(out_root),
    }
    audit_out = Path(args.audit_out)
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    audit_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
