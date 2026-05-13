"""WorkerRegistry — per-repo WorkerThread lifecycle management."""

import logging
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from fido import provider as provider_module
from fido.appstate import (
    _EPOCH,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _ZERO_CRASH,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _ZERO_TALKER,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    FidoState,
    IssueCacheSnapshot,
    ProviderSnapshot,
    TalkerSnapshot,
    ThreadSnapshot,
    WebhookActivity,
    WorkerActivity,
    WorkerCrash,
    zero_repo_state,
)
from fido.atomic import AtomicUpdater
from fido.config import Config, RepoConfig
from fido.github import GitHub
from fido.issue_cache import CacheMetrics, IssueTreeCache
from fido.provider import PromptSession, Provider
from fido.repo import Repo
from fido.rocq import handler_preemption as preemption_fsm
from fido.rocq import worker_registry_crash as registry_fsm
from fido.state import (
    State,
    _resolve_git_dir,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
)
from fido.tasks import Tasks
from fido.worker import WorkerThread

if TYPE_CHECKING:
    from fido.events import Dispatcher

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(tz=timezone.utc)


@dataclass(frozen=True, slots=True)
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

    Write-only relative to :class:`FidoState` — the registry holds only the
    :class:`~fido.atomic.AtomicUpdater` face of the atomic cell.  The
    :class:`~fido.atomic.AtomicReader` face lives in the composition root and
    is passed directly to the status serialisation path so only the one
    collaborator that actually reads holds the read face.

    Usage::

        state_reader, state_updater = create_atomic(FidoState(repos=frozendict(), github_limits=GitHubLimit()))
        registry = WorkerRegistry(my_factory, state_updater)
        registry.start(repo_cfg)   # create + start thread
        registry.wake("owner/repo")  # nudge thread to check for work
        registry.stop_all()          # clean shutdown
    """

    def __init__(
        self,
        thread_factory: Callable[..., WorkerThread],
        state_updater: "AtomicUpdater[FidoState]",
    ) -> None:
        # _threads is single-writer: only start() (called from the watchdog
        # daemon thread or from the startup sequence) ever writes to it.
        # HTTP handler threads (wake, abort_task, get_session, …) only read.
        # CPython 3.14t's per-object dict lock makes individual dict.get() /
        # dict.__setitem__ calls safe without an additional application-level
        # lock, and the single-writer discipline means readers never observe a
        # half-installed entry.  No _threads_lock needed.
        self._threads: dict[str, WorkerThread] = {}
        self._factory = thread_factory
        self._status_lock = threading.Lock()
        # _state_updater is the write-only face of the atomic FidoState cell.
        # Writers call _state_updater.update(selector, value) to CAS-install a
        # value at a Lens path.  The read face (AtomicReader) lives in the
        # composition root; this class never reads the snapshot.
        self._state_updater: AtomicUpdater[FidoState] = state_updater
        # Owner-side crash records: the watchdog increments death_count here,
        # then publishes the result into FidoState via a pure lens write.
        # Only the watchdog thread writes; start() reads during crash recovery
        # (also on the watchdog thread after startup).  No lock needed —
        # single-writer per repo.
        self._crash_records: dict[str, WorkerCrash] = {}
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
        # Per-repo in-flight webhook activities.  _webhook_activities is the
        # authoritative store; mutations update it under _webhook_lock and then
        # publish the frozen tuple to FidoState via _publish_webhook_activities
        # so the snapshot is always up to date.
        self._webhook_activities: dict[str, list[WebhookActivity]] = {}
        self._webhook_lock = threading.Lock()
        # One :class:`~fido.repo.Repo` instance per managed repository,
        # holding the per-repo collaborators (Tasks, State, …) wired
        # with the registry + repo_name so their on_mutate hooks
        # auto-publish IssueSnapshot updates on every disk write
        # (#1696).  Webhook handlers, the worker, and reorder_tasks
        # all reach those collaborators via :meth:`repo_for` — single
        # source of truth, no scattered ``Tasks(work_dir)`` /
        # ``State(fido_dir)`` constructions that would bypass the
        # snapshot publisher.
        #
        # Single-writer + free-threaded dict safety: ``_repos`` follows
        # the same discipline as ``_threads`` — only :meth:`start`
        # writes (called sequentially during startup or from the
        # single watchdog daemon thread on crash recovery), and
        # :meth:`repo_for` / :meth:`tasks_for` / :meth:`state_for`
        # only read.  CPython 3.14t's per-object dict lock makes
        # individual ``dict.get`` / ``dict.__setitem__`` calls safe
        # without an additional application-level lock, and the
        # single-writer discipline means readers never observe a
        # half-installed entry.
        self._repos: dict[str, Repo] = {}

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
        # Construct the per-repo Repo with publishing-aware Tasks and
        # State so every disk write auto-fires the IssueSnapshot
        # publisher via on_mutate (#1696).  Created here, single
        # instance per repo, so every webhook handler / worker /
        # reorder_tasks gets the same collaborators via :meth:`repo_for`.
        # Resolve the canonical git_dir once via ``git rev-parse
        # --absolute-git-dir`` and store it on the Repo — linked
        # worktrees / submodules have ``work_dir/.git`` as a *file*
        # pointing to the actual git directory elsewhere, and every
        # consumer (worker, webhook handlers, status path) needs the
        # same resolved path (#1696 codex P1 round 5).  Failing
        # loudly here surfaces a misconfigured workspace at startup
        # rather than as a confusing flock-on-a-file error later.
        git_dir = _resolve_git_dir(repo_cfg.work_dir)  # pyright: ignore[reportPrivateUsage]
        fido_dir = git_dir / "fido"
        repo = Repo(
            name=repo_cfg.name,
            work_dir=repo_cfg.work_dir,
            git_dir=git_dir,
            tasks=Tasks(
                repo_cfg.work_dir,
                state_updater=self._state_updater,
                repo_name=repo_cfg.name,
                fido_dir=fido_dir,
            ),
            state=State(
                fido_dir,
                state_updater=self._state_updater,
                repo_name=repo_cfg.name,
            ),
        )
        self._repos[repo_cfg.name] = repo
        thread = self._factory(repo_cfg, provider=provider, session_issue=session_issue)
        self._threads[repo_cfg.name] = thread
        _name = repo_cfg.name
        _now = _utcnow()
        # Prepopulate the full RepoState with zero values.  Crash history
        # comes from the class-owned _crash_records (not from FidoState),
        # so this is a pure write — no read-modify-write CAS.
        crash_record = self._crash_records.get(_name, _ZERO_CRASH)
        new_repo = dc_replace(
            zero_repo_state(_name, started_at=_now),
            crash_record=crash_record,
        )
        self._state_updater.update(lambda root: root.repos[_name], new_repo)
        # Seed the initial IssueSnapshot + TaskListSnapshot leaves
        # from existing on-disk state.json / tasks.json now that the
        # FidoState entry exists.  Calling on_mutate with the loaded
        # data shares the publish path — no separate seed code (#1696).
        repo.state.on_mutate(repo.state.load())
        repo.tasks.on_mutate(repo.tasks.list())
        # Wire the per-repo talker on_change callback BEFORE launching
        # the worker thread so an immediate ``register_talker`` from
        # early provider activity has somewhere to publish (#1696
        # codex).  Idempotent — re-wiring on start-after-crash
        # overwrites the same key.
        provider_module.set_talker_change_callback(
            repo_cfg.name, self._make_talker_publisher(repo_cfg.name)
        )
        # On crash recovery the existing IssueTreeCache instance is
        # preserved in ``_issue_caches`` but ``zero_repo_state(...)``
        # above reset its snapshot to ``loaded=false``.  Republish the
        # cache's current metrics so /status.json doesn't regress to
        # an empty snapshot until the next cache mutation (#1696
        # codex).
        with self._issue_cache_lock:
            existing_cache = self._issue_caches.get(repo_cfg.name)
        if existing_cache is not None:
            existing_cache._notify_change()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        thread.start()
        self._publish_thread_snapshot(repo_cfg.name)
        self._publish_provider_snapshot(repo_cfg.name)
        log.info("started WorkerThread for %s", repo_cfg.name)

    def _make_talker_publisher(
        self, repo_name: str
    ) -> "Callable[[provider_module.SessionTalker | None], None]":
        """Return a per-repo on_change callback that publishes a fresh
        :class:`TalkerSnapshot` (or ``None`` for unregister) into
        FidoState every time the talker for *repo_name* changes."""

        def publish(talker: "provider_module.SessionTalker | None") -> None:
            snap = (
                _ZERO_TALKER
                if talker is None
                else TalkerSnapshot(
                    thread_id=talker.thread_id,
                    kind=talker.kind,
                    description=talker.description,
                    subprocess_pid=talker.subprocess_pid or 0,
                    started_at=talker.started_at,
                )
            )
            self._state_updater.update(lambda root: root.repos[repo_name].talker, snap)

        return publish

    def wake(self, repo_name: str) -> None:
        """Wake the thread for *repo_name* so it checks for work immediately.

        No-op if no thread is registered for that repo.
        """
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
        thread = self._threads.get(repo_name)
        if thread:
            thread.abort_task(task_id=task_id)

    def recover_provider(self, repo_name: str) -> bool:
        """Recover the attached provider session for *repo_name*, if present.

        Publishes a fresh :class:`ThreadSnapshot` and :class:`ProviderSnapshot`
        after recovery so the SCADA display reflects the post-recovery session
        state (e.g. a previously detached provider is now reattached).
        """
        thread = self._threads.get(repo_name)
        if thread is None:
            return False
        result = thread.recover_provider()
        self._publish_thread_snapshot(repo_name)
        self._publish_provider_snapshot(repo_name)
        return result

    def report_activity(
        self,
        repo_name: str,
        what: str,
        busy: bool,
        *,
        _now: Callable[[], datetime] = _utcnow,
    ) -> None:
        """Record what *repo_name*'s worker is currently doing.

        Lock-free pure write: installs the new :class:`WorkerActivity` on the
        repo's :class:`RepoState` via lens-update on the :class:`FidoState`
        snapshot.  The repo must have been :meth:`start`-ed first (the
        ``repos[name]`` key is prepopulated by :meth:`start`).
        """
        activity = WorkerActivity(
            repo_name=repo_name, what=what, busy=busy, last_progress_at=_now()
        )
        _name = repo_name
        self._state_updater.update(lambda root: root.repos[_name].activity, activity)

    def record_crash(self, repo_name: str, error: str) -> None:
        """Record an unexpected worker death for *repo_name*.

        Increments the death count from the class-owned ``_crash_records``
        store, then publishes the result into :class:`FidoState` via a pure
        lens write.  Called from the watchdog thread only.

        The repo must have been :meth:`start`-ed first (the ``repos[name]``
        key is prepopulated by :meth:`start`).
        """
        existing = self._crash_records.get(repo_name, _ZERO_CRASH)
        new_crash = WorkerCrash(
            death_count=existing.death_count + 1,
            last_error=error,
            last_crash_time=_utcnow(),
        )
        self._crash_records[repo_name] = new_crash
        _name = repo_name
        self._state_updater.update(
            lambda root: root.repos[_name].crash_record, new_crash
        )
        self._publish_thread_snapshot(repo_name)

    def _publish_thread_snapshot(self, repo_name: str) -> None:
        """Publish an immutable :class:`ThreadSnapshot` for *repo_name* to :class:`FidoState`.

        Reads lifecycle state values from the thread in ``_threads`` and
        installs a fresh :class:`ThreadSnapshot` at
        ``repos[repo_name].thread`` via a single lens write.  No mutable
        thread reference is stored in the snapshot.

        Must be called only after the thread has been inserted into
        ``_threads`` and the repo's :class:`RepoState` has been written to
        ``FidoState`` (i.e. at the end of :meth:`start`).
        """
        thread = self._threads[repo_name]
        snapshot = ThreadSnapshot(
            is_alive=thread.is_alive(),
            was_stopped=thread.was_stopped,
            crash_error=thread.crash_error or "",
        )
        _name = repo_name
        self._state_updater.update(lambda root: root.repos[_name].thread, snapshot)

    def _publish_provider_snapshot(self, repo_name: str) -> None:
        """Publish an immutable :class:`ProviderSnapshot` for *repo_name* to :class:`FidoState`.

        Reads provider session state from the thread in ``_threads`` and
        installs a fresh :class:`ProviderSnapshot` at
        ``repos[repo_name].provider`` via a single lens write.  No mutable
        thread or session reference is stored in the snapshot.

        Must be called only after the thread has been inserted into
        ``_threads`` and the repo's :class:`RepoState` has been written to
        ``FidoState`` (i.e. at the end of :meth:`start`).
        """
        thread = self._threads[repo_name]
        snapshot = ProviderSnapshot(
            session_owner=thread.session_owner or "",
            session_alive=thread.session_alive,
            session_pid=thread.session_pid or 0,
            session_dropped_count=thread.session_dropped_count,
            session_sent_count=thread.session_sent_count,
            session_received_count=thread.session_received_count,
        )
        _name = repo_name
        self._state_updater.update(lambda root: root.repos[_name].provider, snapshot)

    def publish_provider_snapshot(self, repo_name: str) -> None:
        """Publish a fresh :class:`ProviderSnapshot` for *repo_name* to :class:`FidoState`.

        Public wrapper around :meth:`_publish_provider_snapshot`, satisfying the
        :class:`~fido.worker.ActivityReporter` protocol so workers can trigger
        snapshot publication at loop-iteration boundaries — after session attach
        and after each :class:`~fido.worker.Worker` turn.
        """
        self._publish_provider_snapshot(repo_name)

    def _publish_webhook_activities(self, repo_name: str) -> None:
        """Publish the webhook-activity tuple for *repo_name* to :class:`FidoState`.

        Reads the current list from ``_webhook_activities``, builds the frozen
        tuple, and installs it at ``repos[name].webhook_activities`` via a
        single lens write.  The repo must have been :meth:`start`-ed first —
        crashes (``KeyError``) if it is not yet in the snapshot.

        Must be called under ``_webhook_lock``.
        """
        acts = tuple(self._webhook_activities.get(repo_name, []))
        _name = repo_name
        self._state_updater.update(
            lambda root: root.repos[_name].webhook_activities, acts
        )

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

        The authoritative activity list is kept in ``_webhook_activities``
        (protected by ``_webhook_lock``); each mutation publishes a fresh
        frozen tuple to :class:`FidoState` via :meth:`_publish_webhook_activities`.
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
            self._publish_webhook_activities(repo_name)
        try:
            yield handle
        finally:
            with self._webhook_lock:
                acts = self._webhook_activities.get(repo_name, [])
                self._webhook_activities[repo_name] = [
                    a for a in acts if a.handle_id != activity.handle_id
                ]
                self._publish_webhook_activities(repo_name)

    def set_webhook_description(
        self, repo_name: str, handle_id: int, description: str
    ) -> None:
        """Replace one webhook activity entry with an updated description.

        Updates the authoritative ``_webhook_activities`` dict and publishes the
        new frozen tuple to :class:`FidoState`.  No-op if *handle_id* is not
        present for *repo_name*.
        """
        with self._webhook_lock:
            acts = self._webhook_activities.get(repo_name, [])
            if not any(a.handle_id == handle_id for a in acts):
                return
            self._webhook_activities[repo_name] = [
                dc_replace(a, description=description)
                if a.handle_id == handle_id
                else a
                for a in acts
            ]
            self._publish_webhook_activities(repo_name)

    def get_webhook_activities(self, repo_name: str) -> list[WebhookActivity]:
        """Return a snapshot of in-flight webhook activities for *repo_name*.

        Reads from the authoritative ``_webhook_activities`` dict.  Returns an
        empty list if *repo_name* is not registered.
        """
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
        threads = list(self._threads.values())
        for thread in threads:
            thread.stop()

    def stop_and_join(self, repo_name: str, timeout: float = 30.0) -> None:
        """Stop the thread for *repo_name* and wait up to *timeout* seconds for it to exit.

        Reads the thread reference from ``_threads`` outside any lock (single-writer
        dict, so the read is safe), then calls ``stop()`` and ``join()`` outside any
        lock — eliminating the original lock-across-join contention that caused
        ``/status.json`` to stall for up to 30 seconds (#1342).

        No-op if no thread is registered for that repo.
        """
        thread = self._threads.get(repo_name)
        if thread:
            thread.stop()
            thread.join(timeout=timeout)
            self._publish_thread_snapshot(repo_name)

    def is_alive(self, repo_name: str) -> bool:
        """Return True if the thread for *repo_name* is currently alive."""
        thread = self._threads.get(repo_name)
        return thread is not None and thread.is_alive()

    def repo_for(self, repo_name: str) -> Repo:
        """Return the registry-owned :class:`~fido.repo.Repo` for *repo_name*.

        One :class:`Repo` per managed repository, holding the
        publishing-aware :class:`~fido.tasks.Tasks` and
        :class:`~fido.state.State` (their on_mutate fires the SCADA
        snapshot publisher on every disk write, #1696).  Webhook
        handlers, the worker, and ``reorder_tasks`` MUST reach the
        per-repo collaborators via this accessor — bare
        ``Tasks(work_dir)`` / ``State(fido_dir)`` constructions would
        silently bypass the publish hook.
        """
        return self._repos[repo_name]

    def tasks_for(self, repo_name: str) -> Tasks:
        """Convenience accessor for ``self.repo_for(name).tasks``."""
        return self._repos[repo_name].tasks

    def state_for(self, repo_name: str) -> State:
        """Convenience accessor for ``self.repo_for(name).state``."""
        return self._repos[repo_name].state

    def get_thread_crash_error(self, repo_name: str) -> str | None:
        """Return the crash_error stored on the thread for *repo_name*, or None."""
        return self._threads[repo_name].crash_error

    def set_rescoping(self, repo_name: str, active: bool) -> None:
        """Set the rescoping-active flag for *repo_name*.

        Called by the background reorder thread when it starts (``active=True``)
        and when it finishes (``active=False``), so the status display can show
        uncertain task counts while the task list is being rewritten by Opus.

        Single source of truth: a per-repo bool on :class:`RepoState`,
        published via the atomic lens.  No internal dict / lock — the
        snapshot is the state (#1696 parity).
        """
        self._state_updater.update(lambda root: root.repos[repo_name].rescoping, active)

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

        New caches are wired with an ``on_change`` callback that
        publishes a fresh :class:`IssueCacheSnapshot` into the per-repo
        SCADA leaf after every mutation (#1696 parity).  Repos must be
        ``start()``-ed first so the lens has a target.
        """
        with self._issue_cache_lock:
            cache = self._issue_caches.get(repo_name)
            if cache is None:
                cache = IssueTreeCache(
                    repo_name,
                    on_change=self._make_issue_cache_publisher(repo_name),
                )
                self._issue_caches[repo_name] = cache
            return cache

    def _make_issue_cache_publisher(
        self, repo_name: str
    ) -> "Callable[[CacheMetrics], None]":
        """Return a per-repo on_change callback that publishes a fresh
        :class:`IssueCacheSnapshot` into FidoState every time the cache
        mutates.

        Pulled out as a helper so the closure binds *repo_name* (not the
        loop variable) and tests can verify the wiring without touching
        the snapshot internals."""

        def publish(metrics: "CacheMetrics") -> None:
            snap = IssueCacheSnapshot(
                loaded=metrics.inventory_loaded_at is not None,
                open_issues=metrics.open_issue_count,
                events_applied=metrics.events_applied,
                events_dropped_stale=metrics.events_dropped_stale,
                last_event_at=metrics.last_event_at or _EPOCH,
                last_reconcile_at=metrics.last_reconcile_at or _EPOCH,
                last_reconcile_drift=metrics.last_reconcile_drift,
            )
            self._state_updater.update(
                lambda root: root.repos[repo_name].issue_cache, snap
            )

        return publish

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
    state_updater: AtomicUpdater[FidoState] | None = None,
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
        state_updater=state_updater,
    )


def make_registry(
    repos: dict[str, RepoConfig],
    gh: GitHub,
    config: Config | None = None,
    *,
    dispatchers: "dict[str, Dispatcher]",
    state_updater: "AtomicUpdater[FidoState]",
    _thread_factory: Callable[..., WorkerThread] = _make_thread,
) -> WorkerRegistry:
    """Create a :class:`WorkerRegistry` and start threads for all repos.

    Uses :func:`_make_thread` as the factory; all threads share the provided
    :class:`~fido.github.GitHub` client.  The caller (composition root) is
    responsible for creating the atomic cell via :func:`~fido.atomic.create_atomic`
    and passing the updater face here.  Pass a custom registry directly
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
            state_updater=state_updater,
        )

    registry = WorkerRegistry(factory, state_updater)
    for repo_cfg in repos.values():
        registry.start(repo_cfg)
    return registry
