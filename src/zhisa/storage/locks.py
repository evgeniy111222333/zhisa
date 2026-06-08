"""Cross-platform advisory file lock.

Implements inter-process and inter-thread locking using an atomic
``mkdir``-based token.  The lock is **advisory** (cooperating processes
must call :class:`FileLock`); it does not block uncooperative processes
that touch the file directly.

Why ``mkdir``?
    ``os.mkdir`` is atomic on POSIX and Windows: it either succeeds (the
    lock is acquired) or raises :class:`FileExistsError` (already taken).
    No dependency on ``fcntl`` / ``msvcrt``, so the same code works on
    Windows, Linux, macOS, and inside Docker / WSL.

Stale-lock recovery
    A crashed process may leave a lock directory behind.  After
    ``stale_timeout`` seconds the lock is considered abandoned and a
    new acquirer may forcibly remove it.  Stale detection uses
    ``os.path.getmtime`` on the lock directory, so the clock matters;
    on NFS-mounted filesystems ensure clocks are roughly in sync.

Usage::

    from zhisa.storage.locks import FileLock

    with FileLock("/data/tsdb/BTC_USDT/1h/data.parquet"):
        # exclusive critical section
        df = pd.read_parquet(path)
        df = process(df)
        df.to_parquet(path)

The lock is also reentrant *per thread* (a single thread may acquire the
same lock multiple times without deadlocking) and supports an optional
acquisition timeout.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Optional, Union


__all__ = ["FileLock", "FileLockError"]


class FileLockError(Exception):
    """Raised when a :class:`FileLock` operation fails."""


# Per-(path, tid) reentrancy counter.  We use a thread-local counter
# because ``mkdir`` is atomic across processes, so the *first* acquisition
# on this thread creates the lock directory; nested acquisitions on the
# same thread just bump the counter and return immediately.
_reentrant_depth: threading.local = threading.local()


class FileLock:
    """Cross-platform advisory lock with stale-lock recovery.

    Args:
        path: The file (or directory) to lock.  A sibling lock
            directory ``{path}.lock`` is created next to it.
        timeout: Maximum seconds to wait for acquisition
            (``None`` = block forever).
        stale_timeout: Seconds after which an unreleased lock is
            considered abandoned and may be forcibly stolen.
        poll_interval: Seconds between acquisition retries.
    """

    def __init__(
        self,
        path: Union[str, Path],
        *,
        timeout: Optional[float] = 10.0,
        stale_timeout: float = 60.0,
        poll_interval: float = 0.05,
    ) -> None:
        self.path = Path(path)
        self.lock_dir = Path(str(self.path) + ".lock")
        self.timeout = timeout
        self.stale_timeout = stale_timeout
        self.poll_interval = poll_interval
        self._acquired: bool = False
        self._lock_tid: Optional[int] = None

    # ────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────

    def acquire(self) -> None:
        """Acquire the lock, blocking up to ``timeout`` seconds."""
        # Reentrancy: if the same thread tries to acquire the *same
        # FileLock instance* it's a nested ``with`` and we just bump
        # the depth.  Two different FileLock instances on the same
        # path are *different* locks conceptually — the second one
        # must block until the first releases.
        tid = threading.get_ident()
        depth = getattr(_reentrant_depth, "depths", None)
        if depth is not None:
            existing = depth.get(id(self))
            if existing and existing["tid"] == tid:
                existing["count"] += 1
                return

        deadline = None if self.timeout is None else time.monotonic() + self.timeout
        while True:
            if self._try_acquire(tid):
                if depth is None:
                    depth = {}
                    _reentrant_depth.depths = depth
                depth[id(self)] = {"tid": tid, "count": 1}
                return

            if deadline is not None and time.monotonic() >= deadline:
                raise FileLockError(
                    f"Could not acquire lock on {self.path} within {self.timeout}s "
                    f"(held by pid={self._read_holder_pid() or '?'})"
                )

            time.sleep(self.poll_interval)

    def release(self) -> None:
        """Release the lock.  Safe to call multiple times."""
        if not self._acquired:
            return

        # Handle reentrancy: only release on outermost unlock.
        tid = threading.get_ident()
        depth = getattr(_reentrant_depth, "depths", None)
        if depth is not None:
            entry = depth.get(id(self))
            if entry and entry["tid"] == tid:
                if entry["count"] > 1:
                    entry["count"] -= 1
                    return
                # outermost — fall through to actual release
                del depth[id(self)]

        try:
            # Remove pid file first, then directory.  Removing the
            # directory is the actual release token.
            pid_file = self.lock_dir / "pid"
            if pid_file.exists():
                try:
                    pid_file.unlink()
                except OSError:
                    pass
            self.lock_dir.rmdir()
        except FileNotFoundError:
            # Already gone (e.g. forcibly removed as stale).  That's OK.
            pass
        except OSError as exc:
            # Directory not empty for some reason — try recursive cleanup
            # of just our pid file, then warn.
            raise FileLockError(f"Failed to release lock {self.lock_dir}: {exc}") from exc
        finally:
            self._acquired = False
            self._lock_tid = None

    def is_locked(self) -> bool:
        """Return True if the lock is currently held (by anyone)."""
        return self.lock_dir.exists()

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    # ────────────────────────────────────────────────────────
    # Context-manager nesting
    # ────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        state = "held" if self._acquired else "free"
        return f"FileLock({self.path!s}, {state})"

    # ────────────────────────────────────────────────────────
    # Internals
    # ────────────────────────────────────────────────────────

    def _try_acquire(self, tid: int) -> bool:
        """Single attempt: return True if we got the lock now."""
        try:
            # ``exist_ok=False`` is the atomic primitive: succeeds iff
            # the directory did not exist before this call.
            self.lock_dir.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            # Already held by someone else (possibly a stale lock from
            # a crashed process).  Try to steal if stale.
            if not self._is_stale():
                return False
            if not self._force_steal():
                return False
            # Stale lock was just removed — re-attempt the mkdir to
            # create a fresh lock directory.
            try:
                self.lock_dir.mkdir(parents=False, exist_ok=False)
            except FileExistsError:
                # Another process stole it between our steal and our
                # mkdir.  Let the outer loop retry.
                return False

        # We hold the directory.  Drop the pid for diagnostics; if this
        # write fails (e.g. read-only fs), we still consider the lock
        # acquired because the directory itself is the real token.
        try:
            (self.lock_dir / "pid").write_text(str(os.getpid()), encoding="utf-8")
        except OSError:
            pass

        self._acquired = True
        self._lock_tid = tid
        return True

    def _is_stale(self) -> bool:
        """Return True if the held lock is older than ``stale_timeout``."""
        try:
            mtime = self.lock_dir.stat().st_mtime
        except FileNotFoundError:
            return False  # disappeared between check and use
        return (time.time() - mtime) > self.stale_timeout

    def _force_steal(self) -> bool:
        """Best-effort removal of a stale lock.  Returns True on success."""
        try:
            # Only remove contents that look like our pid file — anything
            # else is suspicious and we leave it alone.
            for child in self.lock_dir.iterdir():
                if child.is_file() and child.name == "pid":
                    child.unlink(missing_ok=True)
            self.lock_dir.rmdir()
            return True
        except (FileNotFoundError, OSError):
            return False

    def _read_holder_pid(self) -> Optional[str]:
        try:
            return (self.lock_dir / "pid").read_text(encoding="utf-8").strip()
        except (FileNotFoundError, OSError):
            return None
