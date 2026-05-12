"""Coordination snapshot types for FidoState.

All frozen dataclasses that compose :class:`FidoState` — the atomically-swapped
coordination snapshot shared between :class:`~fido.registry.WorkerRegistry`,
the composition root, and the status serialisation path.  Extracted from
``registry.py`` so provider and session modules can import :class:`FidoState`
directly from this leaf module, with no back-reference to the registry.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from frozendict import frozendict

from fido.rate_limit import GitHubLimit


@dataclass(frozen=True, slots=True)
class WorkerActivity:
    """Snapshot of what one worker is currently doing.

    Frozen so instances can be stored inside frozen :class:`RepoState`
    without breaking the immutability guarantee of the atomic snapshot.
    """

    repo_name: str
    what: str
    busy: bool
    last_progress_at: datetime


@dataclass(frozen=True, slots=True)
class WorkerCrash:
    """Running record of unexpected worker deaths for one repo.

    Frozen so instances can be stored inside frozen :class:`RepoState`
    without breaking the immutability guarantee of the atomic snapshot.

    Readers check ``death_count == 0`` to detect the "never crashed" zero
    sentinel (defined as ``_ZERO_CRASH`` in :mod:`fido.registry`).
    """

    death_count: int
    last_error: str
    last_crash_time: datetime


@dataclass(frozen=True, slots=True)
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

    Frozen so instances can be stored inside frozen :class:`RepoState`
    without breaking the immutability guarantee of the atomic snapshot.
    """

    handle_id: int
    description: str
    started_at: datetime
    thread_id: int


@dataclass(frozen=True, slots=True)
class ProviderSnapshot:
    """Immutable snapshot of one repo's provider (ClaudeSession) state.

    Captures provider session metadata as primitive values at the moment of
    each provider lifecycle event (session start, recovery).  Stored on
    :class:`RepoState` so the SCADA display path reads stale-free values
    without holding the provider lock.

    Fields mirror the public session properties on
    :class:`~fido.worker.WorkerThread` that are read by the status display
    path.  Published by :meth:`~fido.registry.WorkerRegistry._publish_provider_snapshot`
    at lifecycle boundaries where session state actually changes.
    """

    session_owner: str | None
    session_alive: bool
    session_pid: int | None
    session_dropped_count: int
    session_sent_count: int
    session_received_count: int


@dataclass(frozen=True, slots=True)
class ThreadSnapshot:
    """Immutable snapshot of one :class:`~fido.worker.WorkerThread`'s lifecycle state.

    Captures thread lifecycle metadata as primitive values — no handles to the
    mutable :class:`~fido.worker.WorkerThread` object itself.  Stored inside
    the frozen :class:`FidoState` so the SCADA display invariant is preserved:
    a snapshot reader can never reach through to a live thread and observe
    partial mutations.

    Provider session state (session_owner, session_alive, session_pid, and
    the message counters) lives on :class:`ProviderSnapshot` instead so it can
    be published independently at provider lifecycle boundaries.
    """

    is_alive: bool
    was_stopped: bool
    crash_error: str | None


@dataclass(frozen=True, slots=True)
class IssueSnapshot:
    """Immutable snapshot of one repo's current issue/PR/task state.

    Captures the disk-resident state.json + tasks.json values that the
    SCADA display path renders for ``./fido status`` and ``/status.json``.
    Published by :meth:`~fido.registry.WorkerRegistry._publish_issue_snapshot`
    at worker iteration boundaries so the status path never has to read
    state.json or tasks.json from disk on the request thread (#1690).

    ``current_task`` is the title of the in-progress task, falling back to
    the first pending task when nothing is in progress.  ``task_number``
    and ``task_total`` count non-completed entries — same semantics as the
    original ``_collect_fido_state``.

    Frozen so instances can be stored inside the frozen :class:`RepoState`
    without breaking the immutability guarantee of the atomic snapshot.
    """

    issue: int | None
    issue_title: str | None
    issue_started_at: str | None
    pr_number: int | None
    pr_title: str | None
    pending_task_count: int
    completed_task_count: int
    current_task: str | None
    task_number: int | None
    task_total: int | None


def make_issue_snapshot(
    state: Mapping[str, Any], task_list: Iterable[Mapping[str, Any]]
) -> "IssueSnapshot":
    """Build an :class:`IssueSnapshot` from a parsed state.json + tasks list.

    Pure data transform: takes the raw state dict and the task records and
    computes the same fields the legacy disk-reading
    ``_collect_fido_state`` produced.  Workers call this once per
    iteration and publish the result to :class:`FidoState` via
    :meth:`~fido.registry.WorkerRegistry.publish_issue_snapshot`; the
    SCADA status path then reads the snapshot directly instead of
    re-reading state.json and tasks.json on every request (#1690).
    """
    tasks = list(task_list)
    pending = sum(1 for t in tasks if t.get("status") == "pending")
    completed = sum(1 for t in tasks if t.get("status") == "completed")

    current_task: str | None = None
    for t in tasks:
        if t.get("status") == "in_progress":
            current_task = t.get("title")
            break
    if current_task is None:
        for t in tasks:
            if t.get("status") == "pending":
                current_task = t.get("title")
                break

    non_completed = [t for t in tasks if t.get("status") != "completed"]
    task_number: int | None = None
    task_total: int | None = None
    if non_completed:
        task_total = len(non_completed)
        for idx, t in enumerate(non_completed, start=1):
            if t.get("status") == "in_progress":
                task_number = idx
                break
        if task_number is None:
            task_number = 1

    return IssueSnapshot(
        issue=state.get("issue"),
        issue_title=state.get("issue_title"),
        issue_started_at=state.get("issue_started_at"),
        pr_number=state.get("pr_number"),
        pr_title=state.get("pr_title"),
        pending_task_count=pending,
        completed_task_count=completed,
        current_task=current_task,
        task_number=task_number,
        task_total=task_total,
    )


@dataclass(frozen=True, slots=True)
class RepoState:
    """Per-repo sub-snapshot within :class:`FidoState`.

    *key* is the repo slug (e.g. ``"rhencke/confusio"``), matching the key
    under which this record is stored in :attr:`FidoState.repos`.

    *started_at* is the UTC timestamp when the most recent
    :class:`~fido.worker.WorkerThread` for this repo was started.

    *activity* is the current :class:`WorkerActivity` for this repo.
    Initialised to the zero sentinel (``what=""``) by :meth:`~fido.registry.WorkerRegistry.start`; the
    worker replaces it on its first :meth:`~fido.registry.WorkerRegistry.report_activity` call.

    *crash_record* is the accumulated :class:`WorkerCrash` history for this
    repo.  Initialised to ``_ZERO_CRASH`` (``death_count=0``) by
    :meth:`~fido.registry.WorkerRegistry.start`; the watchdog replaces it on each crash.

    *webhook_activities* is the tuple of in-flight webhook handlers for this
    repo.  Published from the authoritative ``_webhook_activities`` dict on
    :class:`~fido.registry.WorkerRegistry` via
    :meth:`~fido.registry.WorkerRegistry._publish_webhook_activities`; starts empty.

    *thread* is the immutable :class:`ThreadSnapshot` for the repo's
    :class:`~fido.worker.WorkerThread`.  ``None`` until the first
    :meth:`~fido.registry.WorkerRegistry.start` call; refreshed by
    :meth:`~fido.registry.WorkerRegistry._publish_thread_snapshot` immediately after the
    thread is started.  No mutable thread reference is stored here — the
    snapshot captures primitive values only.

    *provider* is the immutable :class:`ProviderSnapshot` for the repo's
    provider (ClaudeSession).  ``None`` until the first
    :meth:`~fido.registry.WorkerRegistry.start` call; refreshed by
    :meth:`~fido.registry.WorkerRegistry._publish_provider_snapshot` at provider lifecycle
    events (session start, recovery).

    As subsequent lock-free PRs migrate fields out of the per-lock dicts in
    :class:`~fido.registry.WorkerRegistry`, those fields grow here (e.g. ``rescoping``).
    Each migration removes the corresponding lock and dict from
    ``WorkerRegistry.__init__``.
    """

    key: str
    started_at: datetime
    activity: WorkerActivity
    crash_record: WorkerCrash
    webhook_activities: tuple[WebhookActivity, ...]
    thread: "ThreadSnapshot | None" = None
    provider: "ProviderSnapshot | None" = None
    issue: IssueSnapshot | None = None


@dataclass(frozen=True, slots=True)
class FidoState:
    """Atomically-swapped coordination snapshot.

    ``repos`` maps each repo slug to its :class:`RepoState` snapshot.  It is
    a :class:`frozendict` so stale readers that hold a reference to an old
    snapshot can never accidentally mutate the mapping — the immutability
    guarantee holds even on the free-threaded (no-GIL) build.  Each
    :class:`RepoState` carries a :class:`ThreadSnapshot` at its ``thread``
    field — an immutable copy of the thread's observable state at the last
    :meth:`~fido.registry.WorkerRegistry.start` call.

    The atomic cell is owned by the composition root (``run()`` in
    ``server.py``), not by :class:`~fido.registry.WorkerRegistry`.  The root
    creates both faces via :func:`~fido.atomic.create_atomic`, passes the
    :class:`~fido.atomic.AtomicUpdater` to
    :class:`~fido.registry.WorkerRegistry` and
    :class:`~fido.rate_limit.RateLimitMonitor`, and passes the
    :class:`~fido.atomic.AtomicReader` to the status serialisation path.

    **Convergence target**: ``FidoState`` is intended to grow to cover all
    coordination fields currently scattered across the per-lock dicts in
    :class:`~fido.registry.WorkerRegistry` (worker activities, crash records,
    webhook activities, rescoping flags, …).  As each field migrates here,
    the corresponding lock disappears.

    **Replaces FidoStatus long-term**: :class:`~fido.status.FidoStatus` in
    ``status.py`` is a display-oriented dataclass shaped by the old
    ``/status.json`` schema.  The end-state is to serialise ``FidoState``
    directly to JSON — even where that changes the wire format — and retire
    ``FidoStatus``.  The ``./fido status`` CLI output serves as the living
    requirement: whatever ``FidoState`` carries must be sufficient to render
    everything that ``./fido status`` currently shows.
    """

    repos: frozendict[str, RepoState]
    github_limits: GitHubLimit
