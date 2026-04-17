"""WorkerRegistry — per-repo WorkerThread lifecycle management."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

from kennel.config import Config, RepoConfig
from kennel.github import GitHub
from kennel.provider import PromptSession, Provider
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

    *thread_id* is :func:`threading.get_ident` captured at context entry so
    status display can match this webhook to the active
    :class:`~kennel.claude.ClaudeTalker` (whose ``thread_id`` field is from
    the same call) — letting the CLI attach claude stats to the specific
    webhook line that's driving claude.
    """

    handle_id: int
    description: str
    started_at: datetime
    thread_id: int


@dataclass(frozen=True)
class WebhookActivityHandle:
    """Opaque handle for updating one in-flight webhook activity safely."""

    repo_name: str
    handle_id: int
    registry: "WorkerRegistry"

    def set_description(self, description: str) -> None:
        """Update the displayed description for this webhook activity."""
        self.registry.set_webhook_description(
            self.repo_name, self.handle_id, description
        )


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

    def __init__(self, thread_factory: Callable[..., WorkerThread]) -> None:
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
        self._rescoping: dict[str, bool] = {}
        self._rescoping_lock = threading.Lock()

    def start(self, repo_cfg: RepoConfig) -> None:
        """Create and start a WorkerThread for *repo_cfg*.

        If a previous thread for this repo crashed (dead but not stopped
        orderly), its live session is rescued and handed to the replacement so
        the persistent :class:`~kennel.claude.ClaudeSession` survives the crash.
        """
        provider = None
        session_issue = None
        old_thread = self._threads.get(repo_cfg.name)
        if (
            old_thread is not None
            and not old_thread.is_alive()
            and not old_thread._stop  # pyright: ignore[reportPrivateUsage]
        ):
            # Crashed thread — rescue the live provider before replacing it
            provider = old_thread.detach_provider()
            session_issue, old_thread._session_issue = old_thread._session_issue, None  # pyright: ignore[reportPrivateUsage]
        thread = self._factory(repo_cfg, provider=provider, session_issue=session_issue)
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
    ) -> Generator[WebhookActivityHandle, None, None]:
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
            thread_id=threading.get_ident(),
        )
        handle = WebhookActivityHandle(repo_name, activity.handle_id, self)
        with self._webhook_lock:
            self._webhook_activities.setdefault(repo_name, []).append(activity)
        try:
            yield handle
        finally:
            with self._webhook_lock:
                items = self._webhook_activities.get(repo_name)
                if items is not None:
                    self._webhook_activities[repo_name] = [
                        a for a in items if a.handle_id != activity.handle_id
                    ]

    def set_webhook_description(
        self, repo_name: str, handle_id: int, description: str
    ) -> None:
        """Replace one webhook activity entry with an updated description."""
        with self._webhook_lock:
            items = self._webhook_activities.get(repo_name)
            if items is None:
                return
            for i, activity in enumerate(items):
                if activity.handle_id != handle_id:
                    continue
                items[i] = WebhookActivity(
                    handle_id=activity.handle_id,
                    description=description,
                    started_at=activity.started_at,
                    thread_id=activity.thread_id,
                )
                return

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

    def get_session_owner(self, repo_name: str) -> str | None:
        """Return the name of the thread currently holding the ClaudeSession lock.

        Delegates to :attr:`~kennel.worker.WorkerThread.session_owner` on the
        registered thread.  Returns ``None`` when no thread is registered for
        the repo, no session exists, or the lock is currently free.
        """
        thread = self._threads.get(repo_name)
        return thread.session_owner if thread is not None else None

    def get_session_alive(self, repo_name: str) -> bool:
        """Return True if the persistent ClaudeSession subprocess is alive.

        Distinct from :meth:`get_session_owner` — an idle session that nobody
        currently holds still reports ``session_alive=True`` so status display
        can distinguish "session exists, idle" from "no session".
        """
        thread = self._threads.get(repo_name)
        return thread.session_alive if thread is not None else False

    def get_session_pid(self, repo_name: str) -> int | None:
        """Return the PID of the persistent ClaudeSession subprocess, or None.

        Read authoritatively from the kennel-tracked session rather than
        pgrep — the system prompt file path changed in #456 so the pgrep
        heuristic in :mod:`kennel.status` can no longer locate it.
        """
        thread = self._threads.get(repo_name)
        return thread.session_pid if thread is not None else None

    def get_session_dropped_count(self, repo_name: str) -> int:
        """Return how many stale persistent session ids were dropped for *repo_name*."""
        thread = self._threads.get(repo_name)
        return thread.session_dropped_count if thread is not None else 0

    def set_rescoping(self, repo_name: str, active: bool) -> None:
        """Set the rescoping-active flag for *repo_name*.

        Called by the background reorder thread when it starts (``active=True``)
        and when it finishes (``active=False``), so the status display can show
        uncertain task counts while the task list is being rewritten by Opus.
        """
        with self._rescoping_lock:
            self._rescoping[repo_name] = active

    def is_rescoping(self, repo_name: str) -> bool:
        """Return True if a background rescope is currently in flight for *repo_name*.

        Returns False for unknown repos (no rescope has ever been registered).
        """
        with self._rescoping_lock:
            return self._rescoping.get(repo_name, False)

    def get_session(self, repo_name: str) -> PromptSession | None:
        """Return the live persistent session for *repo_name*.

        Used by :func:`kennel.claude.set_session_resolver` so webhook-handler
        prompt calls can route through the per-repo persistent session
        instead of spawning extra one-shot subprocesses.  Returns ``None``
        when no worker thread is registered for the repo or the thread has
        not yet created its session.
        """
        thread = self._threads.get(repo_name)
        return thread._session if thread is not None else None  # pyright: ignore[reportPrivateUsage]


def _make_thread(
    repo_cfg: RepoConfig,
    registry: WorkerRegistry,
    *,
    gh: GitHub,
    provider: Provider | None = None,
    session_issue: int | None = None,
    config: Config | None = None,
    _WorkerThread: type[WorkerThread] = WorkerThread,
) -> WorkerThread:
    """Default factory: create a WorkerThread with the provided GitHub client."""
    return _WorkerThread(
        repo_cfg.work_dir,
        repo_cfg.name,
        gh,
        registry,
        repo_cfg.membership,
        provider=provider,
        session_issue=session_issue,
        config=config,
        repo_cfg=repo_cfg,
    )


def make_registry(
    repos: dict[str, RepoConfig],
    gh: GitHub,
    config: Config | None = None,
    *,
    _thread_factory: Callable[..., WorkerThread] = _make_thread,
) -> WorkerRegistry:
    """Create a :class:`WorkerRegistry` and start threads for all repos.

    Uses :func:`_make_thread` as the factory; all threads share the provided
    :class:`~kennel.github.GitHub` client.  Pass a custom registry directly
    (with a mock factory) in tests instead of calling this.
    """

    def factory(
        cfg: RepoConfig,
        *,
        provider: Provider | None = None,
        session_issue: int | None = None,
    ) -> WorkerThread:
        return _thread_factory(
            cfg,
            registry,
            gh=gh,
            provider=provider,
            session_issue=session_issue,
            config=config,
        )

    registry = WorkerRegistry(factory)
    for repo_cfg in repos.values():
        registry.start(repo_cfg)
    return registry
