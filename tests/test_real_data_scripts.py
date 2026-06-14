"""Tests for the real-data sandbox CLIs without network access."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from zhisa.backtest.metrics import compute_metrics
from zhisa.data.crypto_loader import CCXTCryptoLoader
from zhisa.scripts import backtest as backtest_script
from zhisa.scripts import ingest_binance_futures_context
from zhisa.scripts import ingest_real_data, paper_run
from zhisa.scripts import monitor_real_data
from zhisa.scripts._real_data import load_market_dataframe
from zhisa.storage.schema import SeriesKey, Timeframe
from zhisa.storage.tsdb import TimeSeriesDB


def _realish_ohlcv(n: int = 160) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    trend = np.linspace(100.0, 104.0, n)
    wave = np.sin(np.linspace(0.0, 8.0, n)) * 0.8
    close = trend + wave
    open_ = np.r_[close[0], close[:-1]]
    spread = np.maximum(np.abs(close - open_) * 0.6, 0.05)
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": np.linspace(900.0, 1200.0, n),
        },
        index=idx,
    )


def test_shared_loader_reads_csv_and_trims_latest(tmp_path: Path):
    df = _realish_ohlcv(40)
    csv_path = tmp_path / "ohlcv.csv"
    export = df.copy()
    export.index.name = "timestamp"
    export.to_csv(csv_path)

    args = SimpleNamespace(
        data_source="csv",
        csv=str(csv_path),
        timestamp_column="timestamp",
        start=None,
        end=None,
        latest_bars=12,
        bars=0,
        tsdb_root=str(tmp_path / "tsdb"),
        symbol="BTC/USDT",
        timeframe="5m",
    )

    loaded = load_market_dataframe(args)
    assert len(loaded) == 12
    assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]
    assert loaded.index.tz is not None


def test_ingest_real_data_uses_public_ohlcv_loader(tmp_path: Path, monkeypatch):
    fetched = _realish_ohlcv(30)

    def fake_fetch(self, symbol, timeframe="5m", since_ms=None, limit=1000, max_bars=None):
        assert self.exchange_id == "binance"
        assert symbol == "BTC/USDT"
        assert timeframe == "5m"
        assert max_bars == 30
        return fetched

    monkeypatch.setattr(CCXTCryptoLoader, "fetch_ohlcv", fake_fetch)
    db_root = tmp_path / "tsdb"

    rc = ingest_real_data.main(
        [
            "--exchange",
            "binance",
            "--symbol",
            "BTC/USDT",
            "--timeframe",
            "5m",
            "--max-bars",
            "30",
            "--db-root",
            str(db_root),
        ]
    )

    assert rc == 0
    db = TimeSeriesDB(db_root)
    key = SeriesKey("BTC/USDT", Timeframe.M5)
    assert db.has_series(key)
    assert db.get_meta(key).row_count == len(fetched)


def test_backtest_script_can_read_tsdb_data(tmp_path: Path, monkeypatch):
    db_root = tmp_path / "tsdb"
    key = SeriesKey("BTC/USDT", Timeframe.M5)
    TimeSeriesDB(db_root).ingest(key, _realish_ohlcv(80))
    captured = {}

    def fake_run_backtest(df, policy, cfg, *, seed=0):
        captured["df"] = df
        captured["cfg"] = cfg
        return SimpleNamespace(metrics=compute_metrics(np.array([1.0, 1.01, 1.02])))

    monkeypatch.setattr(backtest_script, "run_backtest", fake_run_backtest)

    rc = backtest_script.main(
        [
            "--data-source",
            "tsdb",
            "--tsdb-root",
            str(db_root),
            "--symbol",
            "BTC/USDT",
            "--timeframe",
            "5m",
            "--bars",
            "25",
            "--out",
            "",
        ]
    )

    assert rc == 0
    assert len(captured["df"]) == 25
    assert captured["cfg"].seed == 0


def test_paper_run_writes_no_money_artifacts(tmp_path: Path):
    db_root = tmp_path / "tsdb"
    key = SeriesKey("BTC/USDT", Timeframe.M5)
    TimeSeriesDB(db_root).ingest(key, _realish_ohlcv(100))
    out_dir = tmp_path / "paper"

    rc = paper_run.main(
        [
            "--data-source",
            "tsdb",
            "--tsdb-root",
            str(db_root),
            "--symbol",
            "BTC/USDT",
            "--timeframe",
            "5m",
            "--bars",
            "80",
            "--window",
            "8",
            "--image-size",
            "8",
            "--episode-length",
            "10",
            "--out",
            str(out_dir),
        ]
    )

    assert rc == 0
    assert (out_dir / "paper_run_summary.json").exists()
    assert (out_dir / "paper_policy_equity.csv").exists()
    assert (out_dir / "paper_policy_decisions.csv").exists()
    summary = json.loads((out_dir / "paper_run_summary.json").read_text(encoding="utf-8"))
    assert summary["safety"]["real_orders_enabled"] is False
    assert summary["data"]["rows"] == 80


def test_monitor_real_data_writes_regime_opportunity_artifacts(tmp_path: Path):
    db_root = tmp_path / "tsdb"
    key = SeriesKey("BTC/USDT", Timeframe.M5)
    TimeSeriesDB(db_root).ingest(key, _realish_ohlcv(180))
    out_dir = tmp_path / "monitor"

    rc = monitor_real_data.main(
        [
            "--tsdb-root",
            str(db_root),
            "--symbols",
            "BTC/USDT",
            "--timeframe",
            "5m",
            "--bars",
            "160",
            "--scan-bars",
            "80",
            "--stride",
            "20",
            "--horizon",
            "4",
            "--out",
            str(out_dir),
        ]
    )

    assert rc == 0
    summary_path = out_dir / "market_monitor_summary.json"
    latest_path = out_dir / "market_monitor_latest.csv"
    scan_path = out_dir / "market_monitor_scan.csv"
    assert summary_path.exists()
    assert latest_path.exists()
    assert scan_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["safety"]["real_orders_enabled"] is False
    assert summary["symbols"][0]["symbol"] == "BTC/USDT"
    assert "opportunity_quality" in summary["symbols"][0]["scan"]
    latest = pd.read_csv(latest_path)
    scan = pd.read_csv(scan_path)
    assert len(latest) == 1
    assert len(scan) > 0
    assert "opportunity" in latest.columns


def test_monitor_directional_stats_score_shorts_with_inverse_return():
    df = _realish_ohlcv(20)
    # Force a clean +10% forward move from t=0. That is good for a long
    # and bad for a short, so the directional metric must flip sign.
    df["close"] = np.linspace(100.0, 110.0, len(df))
    short_stats = monitor_real_data._future_directional_stats(df, t=0, horizon=10, direction=-1)
    long_stats = monitor_real_data._future_directional_stats(df, t=0, horizon=10, direction=1)

    assert long_stats["directional_forward_return"] > 0.0
    assert short_stats["directional_forward_return"] < 0.0


def _fake_binance_futures_fetch(path, params, base_url, timeout):
    start = int(params.get("startTime", pd.Timestamp("2026-01-01", tz="UTC").timestamp() * 1000))
    end = int(params.get("endTime", start + 60 * 60 * 1000))
    step = 5 * 60 * 1000
    stamps = list(range(start - (start % step), end + 1, step))
    stamps = [ts for ts in stamps if start <= ts <= end]

    if path == "/fapi/v1/allForceOrders":
        return []
    if path.endswith("fundingRate"):
        return [
            {
                "symbol": params["symbol"],
                "fundingTime": start + 4,
                "fundingRate": "0.0001",
                "markPrice": "100.0",
            }
        ]
    if path.endswith("openInterestHist"):
        return [
            {
                "symbol": params["symbol"],
                "sumOpenInterest": str(1000 + i),
                "sumOpenInterestValue": str(100000 + i),
                "CMCCirculatingSupply": "20000000",
                "timestamp": ts,
            }
            for i, ts in enumerate(stamps)
        ]
    if path.endswith("globalLongShortAccountRatio") or path.endswith("topLongShortAccountRatio") or path.endswith("topLongShortPositionRatio"):
        return [
            {
                "symbol": params["symbol"],
                "longAccount": "0.6",
                "shortAccount": "0.4",
                "longShortRatio": "1.5",
                "timestamp": ts,
            }
            for ts in stamps
        ]
    if path.endswith("takerlongshortRatio"):
        return [
            {
                "buyVol": str(80 + i),
                "sellVol": str(40 + i),
                "buySellRatio": "2.0",
                "timestamp": ts,
            }
            for i, ts in enumerate(stamps)
        ]

    rows = []
    for i, ts in enumerate(stamps):
        open_ = 100.0 + i
        close = open_ + 0.5
        rows.append(
            [
                ts,
                str(open_),
                str(open_ + 1.0),
                str(open_ - 1.0),
                str(close),
                "10",
                ts + step - 1,
                "1000",
                20,
                "6",
                "600",
                "0",
            ]
        )
    return rows


def test_binance_futures_context_download_aligns_public_metrics():
    start = int(pd.Timestamp("2026-05-16T00:00:00Z").timestamp() * 1000)
    end = int(pd.Timestamp("2026-05-16T01:00:00Z").timestamp() * 1000)

    df, audit = ingest_binance_futures_context.download_symbol_context(
        "BTC/USDT",
        timeframe="5m",
        start_ms=start,
        end_ms=end,
        fetcher=_fake_binance_futures_fetch,
        sleep_s=0.0,
    )

    assert len(df) == 13
    assert audit["safety"]["real_orders_enabled"] is False
    assert audit["quality"]["coverage_pct"] == 100.0
    assert df["funding_rate"].notna().all()
    assert {"open_interest", "global_long_short_ratio", "taker_buy_volume", "taker_sell_volume"}.issubset(df.columns)
    assert df["volume_delta"].iloc[0] == 40.0
    assert df["taker_buy_sell_ratio"].iloc[0] == 2.0
    assert df["long_short_ratio"].iloc[0] == 1.5


def test_binance_futures_context_cli_writes_parquet_and_audit(tmp_path: Path, monkeypatch):
    original_download = ingest_binance_futures_context.download_symbol_context

    def fake_download(symbol, *, timeframe, start_ms, end_ms, fetcher=ingest_binance_futures_context._fetch_json, base_url=ingest_binance_futures_context.BASE_URL, timeout=30.0, sleep_s=0.05):
        df, audit = original_download(
            symbol,
            timeframe=timeframe,
            start_ms=start_ms,
            end_ms=end_ms,
            fetcher=_fake_binance_futures_fetch,
            base_url=base_url,
            timeout=timeout,
            sleep_s=0.0,
        )
        return df, audit

    monkeypatch.setattr(ingest_binance_futures_context, "download_symbol_context", fake_download)
    out_root = tmp_path / "ctx"
    audit_out = tmp_path / "audit.json"

    rc = ingest_binance_futures_context.main(
        [
            "--symbols",
            "BTC/USDT,ETH/USDT",
            "--timeframe",
            "5m",
            "--start",
            "2026-05-16T00:00:00Z",
            "--end",
            "2026-05-16T01:00:00Z",
            "--out-root",
            str(out_root),
            "--audit-out",
            str(audit_out),
            "--sleep",
            "0",
        ]
    )

    assert rc == 0
    assert (out_root / "BTCUSDT" / "5m" / "context.parquet").exists()
    assert (out_root / "ETHUSDT" / "5m" / "context.parquet").exists()
    payload = json.loads(audit_out.read_text(encoding="utf-8"))
    assert payload["mode"] == "public_binance_usdm_context_ingest_no_orders"
    assert len(payload["symbols"]) == 2
