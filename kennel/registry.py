"""WorkerRegistry — per-repo WorkerThread lifecycle management."""

from __future__ import annotations

import logging
from collections.abc import Callable

from kennel.config import RepoConfig
from kennel.github import GitHub
from kennel.worker import WorkerThread

log = logging.getLogger(__name__)


class WorkerRegistry:
    """Owns and manages one :class:`~kennel.worker.WorkerThread` per repo.

    Threads are created via the injected *thread_factory* so tests can
    supply mock threads without patching module-level names.

    Usage::

        registry = WorkerRegistry(my_factory)
        registry.start(repo_cfg)   # create + start thread
        registry.wake("owner/repo")  # nudge thread to check for work
        registry.stop_all()          # clean shutdown
    """

    def __init__(self, thread_factory: Callable[[RepoConfig], WorkerThread]) -> None:
        self._threads: dict[str, WorkerThread] = {}
        self._factory = thread_factory

    def start(self, repo_cfg: RepoConfig) -> None:
        """Create and start a WorkerThread for *repo_cfg*."""
        thread = self._factory(repo_cfg)
        self._threads[repo_cfg.name] = thread
        thread.start()
        log.info("started WorkerThread for %s", repo_cfg.name)

    def wake(self, repo_name: str) -> None:
        """Wake the thread for *repo_name* so it checks for work immediately.

        No-op if no thread is registered for that repo.
        """
        thread = self._threads.get(repo_name)
        if thread:
            thread.wake()

    def stop_all(self) -> None:
        """Request every managed thread to stop after its current iteration."""
        for thread in self._threads.values():
            thread.stop()

    def stop_and_join(self, repo_name: str, timeout: float = 30.0) -> None:
        """Stop the thread for *repo_name* and wait up to *timeout* seconds for it to exit.

        No-op if no thread is registered for that repo.
        """
        thread = self._threads.get(repo_name)
        if thread:
            thread.stop()
            thread.join(timeout=timeout)

    def is_alive(self, repo_name: str) -> bool:
        """Return True if the thread for *repo_name* is currently alive."""
        thread = self._threads.get(repo_name)
        return thread is not None and thread.is_alive()


def _make_thread(repo_cfg: RepoConfig) -> WorkerThread:
    """Default factory: create a WorkerThread with a live GitHub client."""
    return WorkerThread(repo_cfg.work_dir, GitHub())


def make_registry(repos: dict[str, RepoConfig]) -> WorkerRegistry:
    """Create a :class:`WorkerRegistry` and start threads for all repos.

    Uses :func:`_make_thread` as the factory so each thread gets its own
    live :class:`~kennel.github.GitHub` client.  Pass a custom registry
    directly (with a mock factory) in tests instead of calling this.
    """
    registry = WorkerRegistry(_make_thread)
    for repo_cfg in repos.values():
        registry.start(repo_cfg)
    return registry
