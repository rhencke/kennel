"""Watchdog — check WorkerThread health and restart dead threads."""

from __future__ import annotations

import logging
import threading
import time

from kennel.config import RepoConfig
from kennel.registry import WorkerRegistry

log = logging.getLogger(__name__)

_WATCHDOG_INTERVAL: float = 30.0
# Display-only: /status endpoint flags workers with no activity for this
# long as "stuck" in the UI.  Not used for restart decisions (see class
# docstring).
_STALE_THRESHOLD: float = 600.0


class Watchdog:
    """Check WorkerThread health and restart dead threads.

    Accepts *registry* and *repos* via the constructor so tests can inject
    mock objects without patching module-level names.

    Dead threads (is_alive returns False) are restarted immediately.

    Stale threads (alive but no activity reported) are left alone.  The
    claude subprocess has its own idle timeout, so a worker that looks
    stuck is either waiting for claude to finish or will bubble up a
    timeout error on its own.  A forced restart of a live thread races on
    the fido lockfile (the old thread may not exit before the new one
    starts, and Python threads can't be cleanly killed), which caused a
    restart loop in the field — so we no longer do it.
    """

    def __init__(
        self,
        registry: WorkerRegistry,
        repos: dict[str, RepoConfig],
    ) -> None:
        self.registry = registry
        self.repos = repos

    def run(self) -> int:
        """Run one watchdog iteration. Returns 0."""
        for repo_name, repo_cfg in self.repos.items():
            if not self.registry.is_alive(repo_name):
                error = self.registry.get_thread_crash_error(repo_name)
                if error is not None:
                    self.registry.record_crash(repo_name, error)
                log.info("thread for %s is not alive — restarting", repo_name)
                self.registry.start(repo_cfg)
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
