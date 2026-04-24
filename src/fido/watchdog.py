"""Watchdog — check WorkerThread health and restart dead threads."""

import logging
import threading
import time
from datetime import datetime, timezone

from fido.config import RepoConfig
from fido.github import GitHub
from fido.registry import WorkerRegistry

log = logging.getLogger(__name__)

_WATCHDOG_INTERVAL: float = 30.0
# Display-only: /status endpoint flags workers with no activity for this
# long as "stuck" in the UI.  Not used for restart decisions (see class
# docstring).
_STALE_THRESHOLD: float = 600.0
# Hourly reconcile cadence (closes #812) — webhooks keep the cache fresh
# in steady state; this catches drift from any lost events.
_RECONCILE_INTERVAL: float = 3600.0


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


class ReconcileWatchdog:
    """Hourly cache reconcile to heal drift from missed webhooks (#812).

    For each repo, fetches a fresh ``find_all_open_issues`` snapshot and
    calls :meth:`~fido.issue_cache.IssueTreeCache.reconcile_with_inventory`
    to apply any divergence.  Skips repos whose cache hasn't been
    bootstrapped yet — the worker thread does that on its first
    ``find_next_issue`` iteration.

    A failed ``find_all_open_issues`` (rate limit, transient network
    blip) logs the exception and continues to the next repo — the next
    hourly tick will retry.
    """

    def __init__(
        self,
        registry: WorkerRegistry,
        repos: dict[str, RepoConfig],
        gh: GitHub,
    ) -> None:
        self.registry = registry
        self.repos = repos
        self.gh = gh

    def run(self) -> int:
        """Run one reconcile pass across every repo with a loaded cache.

        Returns 0 (parallels :class:`Watchdog.run` for symmetry).
        """
        for repo_name in self.repos:
            cache = self.registry.get_issue_cache(repo_name)
            if not cache.is_loaded:
                log.info(
                    "reconcile-watchdog[%s]: cache not yet loaded — skipping",
                    repo_name,
                )
                continue
            owner, name = repo_name.split("/", 1)
            snapshot_started_at = datetime.now(tz=timezone.utc)
            try:
                inventory = self.gh.find_all_open_issues(owner, name)
            except Exception:
                log.exception(
                    "reconcile-watchdog[%s]: find_all_open_issues failed — "
                    "next hourly tick will retry",
                    repo_name,
                )
                continue
            drift = cache.reconcile_with_inventory(
                inventory, snapshot_started_at=snapshot_started_at
            )
            log.info("reconcile-watchdog[%s]: applied %d corrections", repo_name, drift)
        return 0

    def start_thread(
        self, *, _interval: float = _RECONCILE_INTERVAL
    ) -> threading.Thread:
        """Start a single daemon thread that reconciles every *_interval* seconds."""

        def _loop() -> None:
            while True:
                time.sleep(_interval)
                self.run()

        t = threading.Thread(target=_loop, daemon=True, name="reconcile-watchdog")
        t.start()
        return t
