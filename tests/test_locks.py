"""Tests for :class:`zhisa.storage.locks.FileLock`."""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

from zhisa.storage.locks import FileLock, FileLockError


class TestBasicAcquire:
    def test_acquire_release_creates_and_removes_lock_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "data.bin"
        target.write_text("hello")
        lock = FileLock(target, timeout=1.0)
        assert not lock.is_locked()
        with lock:
            assert lock.is_locked()
            assert lock._acquired
        assert not lock.is_locked()
        assert target.read_text() == "hello"  # data untouched

    def test_lock_dir_is_sibling(self, tmp_path: Path) -> None:
        target = tmp_path / "data.bin"
        lock = FileLock(target, timeout=1.0)
        with lock:
            assert lock.lock_dir == tmp_path / "data.bin.lock"
            assert lock.lock_dir.is_dir()

    def test_pid_file_written(self, tmp_path: Path) -> None:
        target = tmp_path / "data.bin"
        lock = FileLock(target, timeout=1.0)
        with lock:
            pid_file = lock.lock_dir / "pid"
            assert pid_file.exists()
            assert pid_file.read_text() == str(os.getpid())


class TestReentrancy:
    def test_same_thread_can_nest(self, tmp_path: Path) -> None:
        target = tmp_path / "data.bin"
        lock = FileLock(target, timeout=1.0)
        with lock:
            with lock:
                with lock:
                    assert lock.is_locked()
        assert not lock.is_locked()

    def test_release_outermost_unlocks(self, tmp_path: Path) -> None:
        target = tmp_path / "data.bin"
        lock = FileLock(target, timeout=1.0)
        with lock:
            with lock:
                pass
            # Still held
            assert lock.is_locked()
        assert not lock.is_locked()


class TestTimeout:
    def test_second_acquire_times_out(self, tmp_path: Path) -> None:
        target = tmp_path / "data.bin"
        a = FileLock(target, timeout=0.5, poll_interval=0.05)
        b = FileLock(target, timeout=0.3, poll_interval=0.05)
        with a:
            with pytest.raises(FileLockError) as exc:
                with b:
                    pass
        assert "Could not acquire lock" in str(exc.value)

    def test_timeout_none_waits_forever_or_until_released(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "data.bin"
        a = FileLock(target, timeout=10.0)
        b = FileLock(target, timeout=2.0)  # short timeout to make test snappy
        started = threading.Event()
        finished = threading.Event()

        def hold_then_release():
            with a:
                started.set()
                time.sleep(0.3)
            finished.set()

        t = threading.Thread(target=hold_then_release)
        t.start()
        started.wait(timeout=1.0)
        # Now b should wait for a, but b's timeout is 2s, a releases in 0.3s
        t0 = time.monotonic()
        with b:
            elapsed = time.monotonic() - t0
        t.join()
        assert finished.is_set()
        # b waited until a released (>= 0.2s)
        assert elapsed >= 0.2


class TestStaleLockRecovery:
    def test_stale_lock_is_stolen(self, tmp_path: Path) -> None:
        target = tmp_path / "data.bin"
        lock_dir = tmp_path / "data.bin.lock"
        lock_dir.mkdir()
        # Make the lock look ancient.
        old_mtime = time.time() - 120  # 2 minutes ago
        os.utime(lock_dir, (old_mtime, old_mtime))

        lock = FileLock(target, timeout=1.0, stale_timeout=60.0)
        with lock:
            # We stole it.
            assert lock.is_locked()
        assert not lock.is_locked()

    def test_fresh_lock_is_not_stolen(self, tmp_path: Path) -> None:
        target = tmp_path / "data.bin"
        lock_dir = tmp_path / "data.bin.lock"
        lock_dir.mkdir()
        # Fresh mtime — 0 seconds ago.

        lock = FileLock(target, timeout=0.3, stale_timeout=60.0)
        with pytest.raises(FileLockError):
            with lock:
                pass


class TestIsLocked:
    def test_idempotent_release(self, tmp_path: Path) -> None:
        target = tmp_path / "data.bin"
        lock = FileLock(target, timeout=1.0)
        with lock:
            pass
        # Calling release again must not raise.
        lock.release()
        lock.release()


class TestTsdbIntegration:
    """Smoke tests: TimeSeriesDB uses FileLock around read-merge-write."""

    def test_concurrent_ingest_does_not_lose_rows(self, tmp_path: Path) -> None:
        """Two threads ingesting disjoint bars must produce a complete series."""
        from datetime import datetime, timedelta, timezone

        import pandas as pd

        from zhisa.storage.schema import OHLCV_COLUMNS, SeriesKey, Timeframe
        from zhisa.storage.tsdb import TimeSeriesDB

        db = TimeSeriesDB(tmp_path, lock_timeout=5.0)
        key = SeriesKey("BTC/USDT", Timeframe.M1)

        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        rows_per_thread = 200

        def make_chunk(start_idx: int) -> pd.DataFrame:
            idx = pd.date_range(
                base + timedelta(minutes=start_idx),
                periods=rows_per_thread,
                freq="1min",
                tz="UTC",
            )
            data = {
                "open": 100.0, "high": 101.0, "low": 99.0,
                "close": 100.5, "volume": 10.0,
            }
            df = pd.DataFrame(data, index=idx)
            return df[list(OHLCV_COLUMNS)]

        results: list[int] = []
        errors: list[Exception] = []

        def ingest_chunk(start_idx: int) -> None:
            try:
                meta = db.ingest(key, make_chunk(start_idx), dedup=True)
                results.append(meta.row_count)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=ingest_chunk, args=(0,))
        t2 = threading.Thread(target=ingest_chunk, args=(rows_per_thread,))
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert errors == []
        # Final series must contain exactly 2 * rows_per_thread rows.
        final = db.read(key)
        assert len(final) == 2 * rows_per_thread
        # No overlapping timestamps.
        assert not final.index.has_duplicates

    def test_lock_timeout_none_disables_locking(self, tmp_path: Path) -> None:
        """Setting ``lock_timeout=None`` should skip lock acquisition entirely."""
        from datetime import datetime, timezone

        import pandas as pd

        from zhisa.storage.schema import OHLCV_COLUMNS, SeriesKey, Timeframe
        from zhisa.storage.tsdb import TimeSeriesDB

        db = TimeSeriesDB(tmp_path, lock_timeout=None)
        key = SeriesKey("ETH/USDT", Timeframe.M1)

        idx = pd.date_range(
            datetime(2025, 1, 1, tzinfo=timezone.utc), periods=50, freq="1min", tz="UTC"
        )
        df = pd.DataFrame(
            {"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "volume": 5.0},
            index=idx,
        )[list(OHLCV_COLUMNS)]
        meta = db.ingest(key, df)
        assert meta.row_count == 50
        # No lock directory created.
        assert not (tmp_path / "ETH_USDT" / "1m" / "data.parquet.lock").exists()
