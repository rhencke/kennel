"""WorkerRegistry — per-repo WorkerThread lifecycle management."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

from kennel.config import RepoConfig
from kennel.github import GitHub
from kennel.worker import WorkerThread

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(tz=timezone.utc)


@dataclass
class WorkerActivity:
    """Snapshot of what one worker is currently doing."""

    repo_name: str
    what: str
    busy: bool
    last_progress_at: datetime


@dataclass
class WorkerCrash:
    """Running record of unexpected worker deaths for one repo."""

    death_count: int
    last_error: str
    last_crash_time: datetime


@dataclass
class WebhookActivity:
    """One in-flight webhook handler running alongside the worker.

    Created when ``_process_action`` starts handling a webhook action; removed
    when that handler returns (success or failure).  Surfaced in ``kennel
    status`` as a sub-bullet under the repo so we can see what's being
    handled beyond the worker's own task.
    """

    handle_id: int
    description: str
    started_at: datetime


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
        self._activities: dict[str, WorkerActivity] = {}
        self._activity_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._crashes: dict[str, WorkerCrash] = {}
        self._crash_lock = threading.Lock()
        self._started_at: dict[str, datetime] = {}
        self._started_at_lock = threading.Lock()
        self._webhook_activities: dict[str, list[WebhookActivity]] = {}
        self._webhook_lock = threading.Lock()

    def start(self, repo_cfg: RepoConfig) -> None:
        """Create and start a WorkerThread for *repo_cfg*."""
        thread = self._factory(repo_cfg)
        self._threads[repo_cfg.name] = thread
        with self._started_at_lock:
            self._started_at[repo_cfg.name] = _utcnow()
        thread.start()
        log.info("started WorkerThread for %s", repo_cfg.name)

    def thread_started_at(self, repo_name: str) -> datetime | None:
        """Return when the worker thread for *repo_name* was started, or None."""
        with self._started_at_lock:
            return self._started_at.get(repo_name)

    def wake(self, repo_name: str) -> None:
        """Wake the thread for *repo_name* so it checks for work immediately.

        No-op if no thread is registered for that repo.
        """
        thread = self._threads.get(repo_name)
        if thread:
            thread.wake()

    def abort_task(self, repo_name: str) -> None:
        """Signal the worker for *repo_name* to abort its current task.

        No-op if no thread is registered for that repo.
        """
        thread = self._threads.get(repo_name)
        if thread:
            thread.abort_task()

    def report_activity(
        self,
        repo_name: str,
        what: str,
        busy: bool,
        *,
        _now: Callable[[], datetime] = _utcnow,
    ) -> None:
        """Record what *repo_name*'s worker is currently doing."""
        with self._activity_lock:
            self._activities[repo_name] = WorkerActivity(
                repo_name=repo_name, what=what, busy=busy, last_progress_at=_now()
            )

    def is_stale(
        self,
        repo_name: str,
        threshold: float,
        *,
        _now: Callable[[], datetime] = _utcnow,
    ) -> bool:
        """Return True if *repo_name*'s last progress is older than *threshold* seconds.

        Returns False when no activity has been recorded for the repo (e.g. it
        has never reported in) — the caller can treat that as a fresh start
        rather than a stall.
        """
        with self._activity_lock:
            activity = self._activities.get(repo_name)
        if activity is None:
            return False
        return (_now() - activity.last_progress_at).total_seconds() > threshold

    def get_all_activities(self) -> list[WorkerActivity]:
        """Return a snapshot of all registered workers' current activities."""
        with self._activity_lock:
            return list(self._activities.values())

    def record_crash(self, repo_name: str, error: str) -> None:
        """Record an unexpected worker death for *repo_name*.

        Increments the death count and stores the error message and time of
        the most recent crash.  Safe to call from any thread.
        """
        with self._crash_lock:
            existing = self._crashes.get(repo_name)
            self._crashes[repo_name] = WorkerCrash(
                death_count=(existing.death_count + 1 if existing else 1),
                last_error=error,
                last_crash_time=datetime.now(),
            )

    def get_crash_info(self, repo_name: str) -> WorkerCrash | None:
        """Return crash history for *repo_name*, or None if it has never crashed."""
        with self._crash_lock:
            return self._crashes.get(repo_name)

    @contextmanager
    def webhook_activity(
        self,
        repo_name: str,
        description: str,
        *,
        _now: Callable[[], datetime] = _utcnow,
    ) -> Generator[None, None, None]:
        """Register an in-flight webhook handler for the duration of the block.

        Usage::

            with registry.webhook_activity("owner/repo", "triaging comment"):
                ...do the work...

        Appears in ``kennel status`` as a sub-bullet under the repo.  Entries
        self-unregister on block exit (both success and exception paths).
        """
        activity = WebhookActivity(
            handle_id=id(object()),  # cheap unique id per call
            description=description,
            started_at=_now(),
        )
        with self._webhook_lock:
            self._webhook_activities.setdefault(repo_name, []).append(activity)
        try:
            yield
        finally:
            with self._webhook_lock:
                items = self._webhook_activities.get(repo_name)
                if items is not None:
                    self._webhook_activities[repo_name] = [
                        a for a in items if a.handle_id != activity.handle_id
                    ]

    def get_webhook_activities(self, repo_name: str) -> list[WebhookActivity]:
        """Return a snapshot of in-flight webhook activities for *repo_name*."""
        with self._webhook_lock:
            return list(self._webhook_activities.get(repo_name, []))

    @contextmanager
    def status_update(self) -> Generator[None, None, None]:
        """Context manager that serializes GitHub status updates across workers.

        Only one worker may generate and publish a status at a time.  Callers
        should hold this for the entire report-activity → generate-status →
        set-user-status flow so that an idle worker cannot overwrite a busy
        worker's status.
        """
        with self._status_lock:
            yield

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

    def get_thread_crash_error(self, repo_name: str) -> str | None:
        """Return the crash_error stored on the thread for *repo_name*, or None."""
        thread = self._threads.get(repo_name)
        return thread.crash_error if thread is not None else None


def _make_thread(
    repo_cfg: RepoConfig,
    registry: WorkerRegistry,
    *,
    _GitHub=GitHub,
    _WorkerThread=WorkerThread,
) -> WorkerThread:
    """Default factory: create a WorkerThread with a live GitHub client."""
    return _WorkerThread(
        repo_cfg.work_dir,
        repo_cfg.name,
        _GitHub(),
        registry,
        repo_cfg.membership,
    )


def make_registry(
    repos: dict[str, RepoConfig],
    *,
    _thread_factory=_make_thread,
) -> WorkerRegistry:
    """Create a :class:`WorkerRegistry` and start threads for all repos.

    Uses :func:`_make_thread` as the factory so each thread gets its own
    live :class:`~kennel.github.GitHub` client.  Pass a custom registry
    directly (with a mock factory) in tests instead of calling this.
    """

    def factory(cfg: RepoConfig) -> WorkerThread:
        return _thread_factory(cfg, registry)

    registry = WorkerRegistry(factory)
    for repo_cfg in repos.values():
        registry.start(repo_cfg)
    return registry
