"""Watchdog — kill stale fido processes and restart them."""

from __future__ import annotations

import fcntl
import logging
import os
import subprocess
import time
from pathlib import Path

log = logging.getLogger(__name__)

_STALE_MINUTES = 10
_KILL_WAIT = 2.0


class Watchdog:
    """Check whether fido is running and restart it when stale.

    Accepts ``work_dir`` via the constructor.  All side-effecting helpers are
    ordinary methods so tests can patch them on the instance.
    """

    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir

    def resolve_git_dir(self) -> Path:
        """Return the absolute .git directory for self.work_dir."""
        result = subprocess.run(
            ["git", "rev-parse", "--absolute-git-dir"],
            cwd=self.work_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip())

    def is_lock_free(self, lock_path: Path) -> bool:
        """Return True if no process holds the fido lock."""
        if not lock_path.exists():
            return True
        with open(lock_path) as fd:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                return False

    def is_stale(self, log_path: Path) -> bool:
        """Return True if the log is older than _STALE_MINUTES minutes."""
        if not log_path.exists():
            return False
        age = time.time() - log_path.stat().st_mtime
        return age > _STALE_MINUTES * 60

    def get_lock_pids(self, lock_path: Path) -> list[int]:
        """Return PIDs of all processes with the lock file open."""
        result = subprocess.run(
            ["lsof", str(lock_path)],
            capture_output=True,
            text=True,
        )
        pids: set[int] = set()
        for line in result.stdout.splitlines()[1:]:  # skip lsof header row
            parts = line.split()
            if len(parts) > 1:
                try:
                    pids.add(int(parts[1]))
                except ValueError:
                    pass
        return sorted(pids)

    def kill_pids(self, pids: list[int]) -> None:
        """SIGKILL each PID, ignoring processes that have already exited."""
        for pid in pids:
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass

    def restart_worker(self, log_path: Path) -> None:
        """Launch a new fido worker in the background, appending to log_path."""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as log_fd:
            subprocess.Popen(
                ["uv", "run", "kennel", "worker", str(self.work_dir)],
                stdout=log_fd,
                stderr=log_fd,
                start_new_session=True,
            )

    def run(self) -> int:
        """Run one watchdog iteration. Returns 0."""
        git_dir = self.resolve_git_dir()
        fido_dir = git_dir / "fido"
        lock_path = fido_dir / "lock"
        log_path = Path.home() / "log" / "fido.log"

        if self.is_lock_free(lock_path):
            return 0

        if self.is_stale(log_path):
            log.info("fido stale (log untouched %d+ min) — killing", _STALE_MINUTES)
            self.kill_pids(self.get_lock_pids(lock_path))
            time.sleep(_KILL_WAIT)
            log.info("restarting")
            self.restart_worker(log_path)

        return 0


def run(work_dir: Path) -> int:
    """Module-level entry point: create a Watchdog and run it."""
    return Watchdog(work_dir).run()
