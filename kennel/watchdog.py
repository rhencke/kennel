"""Watchdog — check WorkerThread health and restart dead or stuck threads."""

from __future__ import annotations

import logging
import threading
import time

from kennel.config import RepoConfig
from kennel.registry import WorkerRegistry

log = logging.getLogger(__name__)

_WATCHDOG_INTERVAL: float = 30.0
_STALE_THRESHOLD: float = (
    600.0  # seconds of no progress before a worker is considered stuck
)
_MAX_STALE_COUNT: int = 2  # consecutive stale detections before forcing a restart


class Watchdog:
    """Check WorkerThread health and restart dead or stuck threads.

    Accepts *registry* and *repos* via the constructor so tests can inject
    mock objects without patching module-level names.

    Dead threads (is_alive returns False) are restarted immediately.

    Stale threads (alive but no progress for *_stale_threshold* seconds) are
    counted across consecutive checks.  Once the count reaches
    *_max_stale_count* the thread is stopped, joined, and restarted.  The
    count resets whenever a healthy check is observed so transient slowness
    (a single long Opus call) does not accumulate toward a forced restart.
    """

    def __init__(
        self,
        registry: WorkerRegistry,
        repos: dict[str, RepoConfig],
        *,
        _stale_threshold: float = _STALE_THRESHOLD,
        _max_stale_count: int = _MAX_STALE_COUNT,
    ) -> None:
        self.registry = registry
        self.repos = repos
        self._stale_threshold = _stale_threshold
        self._max_stale_count = _max_stale_count
        self._stale_counts: dict[str, int] = {}

    def run(self) -> int:
        """Run one watchdog iteration. Returns 0."""
        for repo_name, repo_cfg in self.repos.items():
            if not self.registry.is_alive(repo_name):
                error = self.registry.get_thread_crash_error(repo_name)
                if error is not None:
                    self.registry.record_crash(repo_name, error)
                log.info("thread for %s is not alive — restarting", repo_name)
                self._stale_counts.pop(repo_name, None)
                self.registry.start(repo_cfg)
            elif self.registry.is_stale(repo_name, self._stale_threshold):
                count = self._stale_counts.get(repo_name, 0) + 1
                self._stale_counts[repo_name] = count
                log.warning(
                    "thread for %s is alive but stale (%d/%d)",
                    repo_name,
                    count,
                    self._max_stale_count,
                )
                if count >= self._max_stale_count:
                    log.error(
                        "thread for %s stuck after %d stale checks — forcing restart",
                        repo_name,
                        count,
                    )
                    self.registry.record_crash(
                        repo_name,
                        f"stuck: no progress for {self._stale_threshold:.0f}s",
                    )
                    self._stale_counts.pop(repo_name, None)
                    self.registry.stop_and_join(repo_name)
                    self.registry.start(repo_cfg)
            else:
                self._stale_counts.pop(repo_name, None)
        return 0

    def start_thread(
        self, *, _interval: float = _WATCHDOG_INTERVAL
    ) -> threading.Thread:
        """Start a single daemon thread that periodically checks all repos."""

        def _loop() -> None:
            while True:
                time.sleep(_interval)
                self.run()

        t = threading.Thread(target=_loop, daemon=True, name="watchdog")
        t.start()
        return t


def run(registry: WorkerRegistry, repos: dict[str, RepoConfig]) -> int:
    """Module-level entry point: create a Watchdog and run it."""
    return Watchdog(registry, repos).run()
