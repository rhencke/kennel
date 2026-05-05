"""WorkerRegistry — per-repo WorkerThread lifecycle management."""

import logging
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from frozendict import frozendict

from fido.atomic import AtomicReference
from fido.config import Config, RepoConfig
from fido.github import GitHub
from fido.issue_cache import IssueTreeCache
from fido.provider import PromptSession, Provider
from fido.rocq import handler_preemption as preemption_fsm
from fido.rocq import worker_registry_crash as registry_fsm
from fido.worker import WorkerThread

if TYPE_CHECKING:
    from fido.events import Dispatcher

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(tz=timezone.utc)


@dataclass(frozen=True)
class WorkerActivity:
    """Snapshot of what one worker is currently doing.

    Frozen so instances can be stored inside frozen :class:`RepoState`
    without breaking the immutability guarantee of the atomic snapshot.
    """

    repo_name: str
    what: str
    busy: bool
    last_progress_at: datetime


@dataclass(frozen=True)
class WorkerCrash:
    """Running record of unexpected worker deaths for one repo.

    Frozen so instances can be stored inside frozen :class:`RepoState`
    without breaking the immutability guarantee of the atomic snapshot.
    """

    death_count: int
    last_error: str
    last_crash_time: datetime


@dataclass(frozen=True)
class RepoState:
    """Per-repo sub-snapshot within :class:`FidoState`.

    *key* is the repo slug (e.g. ``"rhencke/confusio"``), matching the key
    under which this record is stored in :attr:`FidoState.repos`.

    *started_at* is the UTC timestamp when the most recent
    :class:`~fido.worker.WorkerThread` for this repo was started.

    *activity* is the current :class:`WorkerActivity` for this repo, or
    ``None`` if the worker has not yet reported any activity.  Migrated from
    ``WorkerRegistry._activities`` / ``_activity_lock`` in PR 2/6.

    *crash_record* is the accumulated :class:`WorkerCrash` history for this
    repo, or ``None`` if the worker has never crashed.  Migrated from
    ``WorkerRegistry._crashes`` / ``_crash_lock`` in PR 2/6.

    As subsequent lock-free PRs migrate fields out of the per-lock dicts in
    :class:`WorkerRegistry`, those fields grow here (e.g. ``rescoping``).
    Each migration removes the corresponding lock and dict from
    ``WorkerRegistry.__init__``.
    """

    key: str
    started_at: datetime
    activity: WorkerActivity | None = None
    crash_record: WorkerCrash | None = None


@dataclass(frozen=True)
class FidoState:
    """Atomically-swapped coordination snapshot owned by :class:`WorkerRegistry`.

    ``repos`` maps each repo slug to its :class:`RepoState` snapshot.  It is
    a :class:`frozendict` so stale readers that hold a reference to an old
    snapshot can never accidentally mutate the mapping — the immutability
    guarantee holds even on the free-threaded (no-GIL) build.

    **Convergence target**: ``FidoState`` is intended to grow to cover all
    coordination fields currently scattered across the per-lock dicts in
    :class:`WorkerRegistry` (worker activities, crash records, webhook
    activities, rescoping flags, …).  As each field migrates here, the
    corresponding lock disappears.

    **Replaces FidoStatus long-term**: :class:`~fido.status.FidoStatus` in
    ``status.py`` is a display-oriented dataclass shaped by the old
    ``/status.json`` schema.  The end-state is to serialise ``FidoState``
    directly to JSON — even where that changes the wire format — and retire
    ``FidoStatus``.  The ``./fido status`` CLI output serves as the living
    requirement: whatever ``FidoState`` carries must be sufficient to render
    everything that ``./fido status`` currently shows.
    """

    repos: frozendict[str, RepoState]


_EMPTY_FIDO_STATE = FidoState(repos=frozendict())


@dataclass
class WebhookActivity:
    """One in-flight webhook handler running alongside the worker.

    Created when ``_process_action`` starts handling a webhook action; removed
    when that handler returns (success or failure).  Surfaced in ``fido
    status`` as a sub-bullet under the repo so we can see what's being
    handled beyond the worker's own task.

    *thread_id* is :func:`threading.get_ident` captured at context entry so
    status display can match this webhook to the active
    :class:`~fido.provider.SessionTalker` (whose ``thread_id`` field is from
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
    """Owns and manages one :class:`~fido.worker.WorkerThread` per repo.

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
        # _threads_lock guards _threads: written by start() on the watchdog
        # thread, read from HTTP handler threads (wake, abort_task, get_session,
        # etc.).  Python 3.14t has no GIL — dict reads and writes are not atomic.
        self._threads_lock = threading.Lock()
        self._factory = thread_factory
        self._activities: dict[str, WorkerActivity] = {}
        self._activity_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._crashes: dict[str, WorkerCrash] = {}
        self._crash_lock = threading.Lock()
        # _state holds the atomically-swapped FidoState snapshot.  Writers
        # call AtomicReference.update() under the single internal lock; readers
        # call AtomicReference.get() without any lock (observationally lock-free).
        self._state: AtomicReference[FidoState] = AtomicReference(_EMPTY_FIDO_STATE)
        self._webhook_activities: dict[str, list[WebhookActivity]] = {}
        self._webhook_lock = threading.Lock()
        self._rescoping: dict[str, bool] = {}
        self._rescoping_lock = threading.Lock()
        # Per-repo untriaged-webhook inbox (#1067).  Counts model-needing
        # webhook handlers that have arrived but not yet finished processing.
        # Protected by _untriaged_lock; _untriaged_drained events are set when
        # the count hits 0 so the worker can wait efficiently at turn boundaries.
        self._untriaged: dict[str, int] = {}
        self._untriaged_drained: dict[str, threading.Event] = {}
        self._untriaged_lock = threading.Lock()
        # Per-repo handler-preemption FSM state from handler_preemption.v.
        # Protected by _untriaged_lock (same lock as the inbox counter).
        self._preemption_fsm_states: dict[str, preemption_fsm.State] = {}
        # Per-repo issue tree caches shared between worker + webhook
        # threads (closes #812).  Lazily created on first lookup so tests
        # that don't exercise the cache path don't pay setup cost.
        self._issue_caches: dict[str, IssueTreeCache] = {}
        self._issue_cache_lock = threading.Lock()
        # Per-repo FSM state from worker_registry_crash.v.  Only written by
        # start(), which is called sequentially during startup and from the
        # single watchdog daemon thread during crash recovery — no lock needed.
        self._registry_fsm_states: dict[str, registry_fsm.State] = {}

    def _registry_fsm_transition(
        self, repo_name: str, event: registry_fsm.Event
    ) -> registry_fsm.State:
        """Fire *event* for *repo_name*, raising ``AssertionError`` if rejected.

        Single oracle for every FSM transition in :meth:`start`, so a
        coordination bug surfaces as a crash rather than silent drift.
        Initialises the repo's FSM state to ``Absent`` on first access.
        """
        prev = self._registry_fsm_states.get(repo_name, registry_fsm.Absent())
        new_state = registry_fsm.transition(prev, event)
        if new_state is None:
            raise AssertionError(
                f"worker_registry_crash FSM: {type(event).__name__} rejected in "
                f"state {type(prev).__name__} for repo {repo_name!r}"
            )
        self._registry_fsm_states[repo_name] = new_state
        log.debug(
            "registry[%s]: FSM %s →%s via %s",
            repo_name,
            type(prev).__name__,
            type(new_state).__name__,
            type(event).__name__,
        )
        return new_state

    def start(self, repo_cfg: RepoConfig) -> None:
        """Create and start a WorkerThread for *repo_cfg*.

        If a previous thread for this repo crashed (dead but not stopped
        orderly), its live session is rescued and handed to the replacement so
        the persistent :class:`~fido.claude.ClaudeSession` survives the crash.

        FSM oracle — the :mod:`fido.rocq.worker_registry_crash` extracted
        model is asserted on every call so a coordination violation surfaces
        as an immediate crash rather than silent state divergence:

        - Fresh start (no prior thread):  ``Absent → Launch → Active``
        - Crash recovery with rescue:     ``Active → ThreadDies → Crashed →
                                            Rescue → Active``
        - Re-enable after orderly stop:   ``Active → ThreadStops → Stopped →
                                            Launch → Active``
        - Live predecessor (bug):         ``Launch`` rejected from ``Active``
                                          → ``AssertionError``
                                          (``no_start_while_active`` invariant)

        E1 flip point: when the E-band work lands, :meth:`start` can be
        split at the boundary — a thin entry point feeds the right event
        (``Launch`` vs ``Rescue``) into the extracted ``registry_fsm.transition``
        and dispatches on the returned state, replacing the hand-written
        ``old_thread is not None and not old_thread.is_alive() and not
        old_thread.was_stopped`` guard entirely.
        """
        provider = None
        session_issue = None
        with self._threads_lock:
            old_thread = self._threads.get(repo_cfg.name)
        if old_thread is None:
            # No predecessor — initial start: Absent → Active.
            self._registry_fsm_transition(repo_cfg.name, registry_fsm.Launch())
        elif not old_thread.is_alive() and not old_thread.was_stopped:
            # Crashed predecessor — rescue the live provider:
            # Active → Crashed (ThreadDies) → Active (Rescue).
            self._registry_fsm_transition(repo_cfg.name, registry_fsm.ThreadDies())
            self._registry_fsm_transition(repo_cfg.name, registry_fsm.Rescue())
            provider = old_thread.detach_provider()
            session_issue, old_thread._session_issue = old_thread._session_issue, None  # pyright: ignore[reportPrivateUsage]
            # Heal the rescued session before the new worker runs.  A worker
            # crash mid-turn can leave the ClaudeSession FSM in a non-Idle state
            # (e.g. Sending after BrokenPipe on stdin write); without this
            # respawn the next worker's first send() hits "Send rejected in
            # state Sending" and the watchdog rescues into a permanent crash
            # loop.  recover() respawns the subprocess with --resume so
            # conversation context is preserved.
            if provider is not None:
                provider.agent.recover_session()
        elif not old_thread.is_alive():
            # Orderly-stopped predecessor (was_stopped is True):
            # Active → Stopped (ThreadStops) → Active (Launch).
            self._registry_fsm_transition(repo_cfg.name, registry_fsm.ThreadStops())
            self._registry_fsm_transition(repo_cfg.name, registry_fsm.Launch())
        else:
            # Alive predecessor — the FSM rejects Launch from Active, surfacing
            # the no_start_while_active violation as an immediate AssertionError.
            self._registry_fsm_transition(repo_cfg.name, registry_fsm.Launch())
        thread = self._factory(repo_cfg, provider=provider, session_issue=session_issue)
        with self._threads_lock:
            self._threads[repo_cfg.name] = thread
        _name = repo_cfg.name
        _now = _utcnow()
        self._state.lens_update(
            lambda root: root.repos[_name],
            RepoState(key=_name, started_at=_now),
        )
        thread.start()
        log.info("started WorkerThread for %s", repo_cfg.name)

    def thread_started_at(self, repo_name: str) -> datetime | None:
        """Return when the worker thread for *repo_name* was started, or None.

        Lock-free: reads from the current :class:`FidoState` snapshot without
        acquiring any lock.
        """
        repo_state = self._state.get().repos.get(repo_name)
        return repo_state.started_at if repo_state is not None else None

    def wake(self, repo_name: str) -> None:
        """Wake the thread for *repo_name* so it checks for work immediately.

        No-op if no thread is registered for that repo.
        """
        with self._threads_lock:
            thread = self._threads.get(repo_name)
        if thread:
            thread.wake()

    def abort_task(self, repo_name: str, *, task_id: str) -> None:
        """Signal the worker for *repo_name* to abort *task_id*.

        Callers must pass the id of the task they intend to abort so a
        leaked signal cannot clobber a different task on the next loop
        iteration (closes #1193).

        No-op if no thread is registered for that repo.
        """
        with self._threads_lock:
            thread = self._threads.get(repo_name)
        if thread:
            thread.abort_task(task_id=task_id)

    def recover_provider(self, repo_name: str) -> bool:
        """Recover the attached provider session for *repo_name*, if present."""
        with self._threads_lock:
            thread = self._threads.get(repo_name)
        if thread is None:
            return False
        return thread.recover_provider()

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
                last_crash_time=_utcnow(),
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

        Appears in ``fido status`` as a sub-bullet under the repo.  Entries
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
        with self._threads_lock:
            threads = list(self._threads.values())
        for thread in threads:
            thread.stop()

    def stop_and_join(self, repo_name: str, timeout: float = 30.0) -> None:
        """Stop the thread for *repo_name* and wait up to *timeout* seconds for it to exit.

        No-op if no thread is registered for that repo.
        """
        with self._threads_lock:
            thread = self._threads.get(repo_name)
        if thread:
            thread.stop()
            thread.join(timeout=timeout)

    def is_alive(self, repo_name: str) -> bool:
        """Return True if the thread for *repo_name* is currently alive."""
        with self._threads_lock:
            thread = self._threads.get(repo_name)
        return thread is not None and thread.is_alive()

    def get_thread_crash_error(self, repo_name: str) -> str | None:
        """Return the crash_error stored on the thread for *repo_name*, or None."""
        with self._threads_lock:
            thread = self._threads.get(repo_name)
        return thread.crash_error if thread is not None else None

    def get_session_owner(self, repo_name: str) -> str | None:
        """Return the name of the thread currently holding the ClaudeSession lock.

        Delegates to :attr:`~fido.worker.WorkerThread.session_owner` on the
        registered thread.  Returns ``None`` when no thread is registered for
        the repo, no session exists, or the lock is currently free.
        """
        with self._threads_lock:
            thread = self._threads.get(repo_name)
        return thread.session_owner if thread is not None else None

    def get_session_alive(self, repo_name: str) -> bool:
        """Return True if the persistent ClaudeSession subprocess is alive.

        Distinct from :meth:`get_session_owner` — an idle session that nobody
        currently holds still reports ``session_alive=True`` so status display
        can distinguish "session exists, idle" from "no session".
        """
        with self._threads_lock:
            thread = self._threads.get(repo_name)
        return thread.session_alive if thread is not None else False

    def get_session_pid(self, repo_name: str) -> int | None:
        """Return the PID of the persistent ClaudeSession subprocess, or None.

        Read authoritatively from the fido-tracked session rather than
        pgrep — the system prompt file path changed in #456 so the pgrep
        heuristic in :mod:`fido.status` can no longer locate it.
        """
        with self._threads_lock:
            thread = self._threads.get(repo_name)
        return thread.session_pid if thread is not None else None

    def get_session_dropped_count(self, repo_name: str) -> int:
        """Return how many stale persistent session ids were dropped for *repo_name*."""
        with self._threads_lock:
            thread = self._threads.get(repo_name)
        return thread.session_dropped_count if thread is not None else 0

    def get_session_sent_count(self, repo_name: str) -> int:
        """Return the number of messages sent to the current session subprocess for *repo_name*."""
        with self._threads_lock:
            thread = self._threads.get(repo_name)
        return thread.session_sent_count if thread is not None else 0

    def get_session_received_count(self, repo_name: str) -> int:
        """Return the number of events received from the current session subprocess for *repo_name*."""
        with self._threads_lock:
            thread = self._threads.get(repo_name)
        return thread.session_received_count if thread is not None else 0

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

    # ── untriaged-webhook inbox (#1067) ──────────────────────────────────

    def _preemption_state(self, repo_name: str) -> preemption_fsm.State:
        """Return the repo's handler-preemption oracle state."""
        return self._preemption_fsm_states.get(repo_name, preemption_fsm.empty_state)

    def _preemption_state_name(self, state: preemption_fsm.State) -> str:
        """Return a compact product-state label for debug logging."""
        return (
            f"legacy={type(state.legacy_demand).__name__},"
            f"durable={type(state.durable_demand).__name__},"
            f"interrupt={type(state.provider_interrupt).__name__}"
        )

    def _preemption_has_durable_demand(self, state: preemption_fsm.State) -> bool:
        """Return True when the oracle state has durable queued demand."""
        return isinstance(state.durable_demand, preemption_fsm.DurableNonEmpty)

    def _preemption_fsm_transition(
        self, repo_name: str, event: preemption_fsm.Event
    ) -> preemption_fsm.State:
        """Fire *event* for *repo_name*'s handler-preemption FSM.

        Raises ``AssertionError`` if the transition is rejected — same pattern
        as :meth:`_registry_fsm_transition`.  Must be called under
        ``_untriaged_lock``.

        Initialises the repo's FSM state to ``empty_state`` on first access.
        """
        prev = self._preemption_state(repo_name)
        new_state = preemption_fsm.transition(prev, event)
        if new_state is None:
            raise AssertionError(
                f"handler_preemption FSM: {type(event).__name__} rejected in "
                f"state {self._preemption_state_name(prev)} for repo {repo_name!r}"
            )
        self._preemption_fsm_states[repo_name] = new_state
        log.debug(
            "preemption[%s]: FSM %s →%s via %s",
            repo_name,
            self._preemption_state_name(prev),
            self._preemption_state_name(new_state),
            type(event).__name__,
        )
        return new_state

    def enter_untriaged(self, repo_name: str) -> None:
        """Record that one model-needing webhook handler has started for *repo_name*.

        Called synchronously on the HTTP handler thread when a webhook that
        requires model processing is dispatched — before the background worker
        thread is spawned.  Increments the per-repo untriaged count and clears
        the drained event so any waiting worker knows the inbox is non-empty.
        """
        with self._untriaged_lock:
            old = self._untriaged.get(repo_name, 0)
            self._untriaged[repo_name] = old + 1
            if old == 0:
                # Boolean abstraction in handler_preemption.v: legacy demand
                # is NonEmpty iff *any* handler is in flight.  Fire the FSM
                # transition only on the 0→1 edge so the FSM agrees with the
                # Python count by construction.
                self._preemption_fsm_transition(
                    repo_name, preemption_fsm.WebhookArrives()
                )
            ev = self._untriaged_drained.get(repo_name)
            if ev is None:
                ev = threading.Event()
                self._untriaged_drained[repo_name] = ev
            ev.clear()
        log.debug("untriaged inbox[%s]: +1 → %d", repo_name, old + 1)

    def exit_untriaged(self, repo_name: str) -> None:
        """Record that one model-needing webhook handler has finished for *repo_name*.

        Called at the end of ``_process_action``.  Decrements the count; when
        it reaches zero, sets the drained event so a waiting worker can resume.
        Logs a warning (but does not raise) if called when the count is already
        zero — the accounting should balance, but a stray call is safer to log
        than to crash.
        """
        with self._untriaged_lock:
            old = self._untriaged.get(repo_name, 0)
            if old <= 0:
                log.warning(
                    "untriaged inbox[%s]: exit_untriaged called but count is already %d",
                    repo_name,
                    old,
                )
                return
            new = old - 1
            self._untriaged[repo_name] = new
            if new == 0:
                # Boolean abstraction: fire HandlerDone only on the 1→0 edge.
                # The FSM agrees with the Python count by construction, so no
                # post-transition with_legacy override is needed.
                self._preemption_fsm_transition(repo_name, preemption_fsm.HandlerDone())
                ev = self._untriaged_drained.get(repo_name)
                if ev is not None:
                    ev.set()
        log.debug("untriaged inbox[%s]: -1 → %d", repo_name, new)

    def has_untriaged(self, repo_name: str) -> bool:
        """Return True if any model-needing webhooks are waiting to be processed.

        Used by the worker at turn boundaries to decide whether to yield before
        starting the next provider turn.
        """
        with self._untriaged_lock:
            return self._untriaged.get(repo_name, 0) > 0

    def note_durable_demand(self, repo_name: str) -> None:
        """Record that durable webhook demand is queued for *repo_name*."""
        with self._untriaged_lock:
            current = self._preemption_state(repo_name)
            if self._preemption_has_durable_demand(current):
                return
            self._preemption_fsm_transition(
                repo_name, preemption_fsm.DurableDemandRecorded()
            )

    def note_provider_interrupt_requested(self, repo_name: str) -> None:
        """Record that a queued durable demand requested provider interrupt."""
        with self._untriaged_lock:
            self._preemption_fsm_transition(
                repo_name, preemption_fsm.InterruptRequested()
            )

    def note_durable_demand_drained(self, repo_name: str) -> None:
        """Record that durable webhook demand for *repo_name* has drained."""
        with self._untriaged_lock:
            current = self._preemption_state(repo_name)
            if not self._preemption_has_durable_demand(current):
                return
            self._preemption_fsm_transition(
                repo_name, preemption_fsm.DurableDemandDrained()
            )

    def wait_for_inbox_drain(
        self, repo_name: str, timeout: float | None = None
    ) -> bool:
        """Block until the untriaged inbox for *repo_name* empties, or *timeout* elapses.

        Returns ``True`` when the inbox is empty (drained), ``False`` on
        timeout.  Safe to call even if ``enter_untriaged`` has never been
        called for the repo — returns ``True`` immediately.

        The caller acquires the event reference under the lock and then waits
        outside it so the lock is not held during the blocking wait.
        """
        with self._untriaged_lock:
            if self._untriaged.get(repo_name, 0) == 0:
                return True
            # ``enter_untriaged`` always creates the drained event before
            # bumping the counter, so a non-zero count guarantees the
            # event exists.
            ev = self._untriaged_drained[repo_name]
        return ev.wait(timeout=timeout)

    def force_clear_untriaged(self, repo_name: str) -> int:
        """Reset the untriaged inbox to zero, log loud, and signal drained.

        Backstop for the case where some producer ``enter_untriaged`` call
        leaks (no matching ``exit_untriaged`` ever fires) and the worker is
        otherwise stuck waiting forever (#1280).  Returns the count that was
        cleared.  No-op when the count is already zero.
        """
        with self._untriaged_lock:
            cleared = self._untriaged.get(repo_name, 0)
            if cleared <= 0:
                return 0
            self._untriaged[repo_name] = 0
            # The FSM is on the boolean abstraction: legacy is NonEmpty iff
            # any handler is in flight.  When count > 0, FSM legacy is
            # NonEmpty by construction (enter_untriaged fired WebhookArrives
            # on the 0→1 edge), so HandlerDone is the valid drain event.
            # Without this transition, the FSM stays at LegacyNonEmpty after
            # force-clear and the next assert_worker_turn_ok crashes (#1330).
            self._preemption_fsm_transition(repo_name, preemption_fsm.HandlerDone())
            ev = self._untriaged_drained.get(repo_name)
            if ev is not None:
                ev.set()
        log.warning(
            "untriaged inbox[%s]: force-cleared %d leaked hold(s) — "
            "some enter_untriaged call had no matching exit_untriaged",
            repo_name,
            cleared,
        )
        return cleared

    def assert_worker_turn_ok(self, repo_name: str) -> None:
        """Assert that the worker may start a provider turn for *repo_name*.

        Fires ``WorkerTurnStart`` through the handler-preemption FSM oracle.
        If the inbox or durable queue is non-empty, the transition is rejected
        and an ``AssertionError`` surfaces the coordination violation.

        Called by the worker before each ``provider_run()`` — after the
        yield-for-untriaged wait has drained the inbox.
        """
        with self._untriaged_lock:
            self._preemption_fsm_transition(repo_name, preemption_fsm.WorkerTurnStart())

    def get_session(self, repo_name: str) -> PromptSession | None:
        """Return the live persistent session for *repo_name*.

        Used by :func:`fido.provider.set_session_resolver` so webhook-handler
        prompt calls can route through the per-repo persistent session
        instead of spawning extra one-shot subprocesses.  Returns ``None``
        when no worker thread is registered for the repo or the thread has
        not yet created its session.
        """
        with self._threads_lock:
            thread = self._threads.get(repo_name)
        if thread is None:
            return None
        provider = thread.current_provider()
        return provider.agent.session if provider is not None else None

    def get_issue_cache(self, repo_name: str) -> IssueTreeCache:
        """Return (lazily creating) the per-repo issue tree cache (#812).

        Cache is shared between the worker thread (picker reads) and the
        webhook handler thread (event mutations).  Lifetime is tied to
        the registry, not to any one worker thread — so a worker thread
        crash + restart inherits the same cache.
        """
        with self._issue_cache_lock:
            cache = self._issue_caches.get(repo_name)
            if cache is None:
                cache = IssueTreeCache(repo_name)
                self._issue_caches[repo_name] = cache
            return cache

    def all_issue_caches(self) -> list[IssueTreeCache]:
        """Snapshot list of every issue cache that has been created.

        Used by ``fido status`` to surface per-repo cache metrics.
        """
        with self._issue_cache_lock:
            return list(self._issue_caches.values())


def _make_thread(
    repo_cfg: RepoConfig,
    registry: WorkerRegistry,
    *,
    gh: GitHub,
    provider: Provider | None = None,
    session_issue: int | None = None,
    config: Config | None = None,
    dispatchers: "dict[str, Dispatcher]",
    _WorkerThread: type[WorkerThread] = WorkerThread,
) -> WorkerThread:
    """Default factory: create a WorkerThread with the provided GitHub client.

    Hands the per-repo :class:`IssueTreeCache` (created lazily by the
    registry) into the worker so the picker reads from the cache and the
    webhook handler can patch it (closes #812).
    """
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
        dispatcher=dispatchers[repo_cfg.name],
        issue_cache=registry.get_issue_cache(repo_cfg.name),
    )


def make_registry(
    repos: dict[str, RepoConfig],
    gh: GitHub,
    config: Config | None = None,
    *,
    dispatchers: "dict[str, Dispatcher]",
    _thread_factory: Callable[..., WorkerThread] = _make_thread,
) -> WorkerRegistry:
    """Create a :class:`WorkerRegistry` and start threads for all repos.

    Uses :func:`_make_thread` as the factory; all threads share the provided
    :class:`~fido.github.GitHub` client.  Pass a custom registry directly
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
            dispatchers=dispatchers,
        )

    registry = WorkerRegistry(factory)
    for repo_cfg in repos.values():
        registry.start(repo_cfg)
    return registry
