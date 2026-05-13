"""Coordination snapshot types for FidoState.

Leaf module: only frozen dataclass definitions and their pure-data helpers.
Everyone else imports from here — :mod:`fido.appstate` itself imports
nothing from the rest of the project, so adding a snapshot field never
risks an import cycle.

**No ``None``, no defaults.**  Every constructor parameter on every
class here is required, and no field is typed ``T | None``.  Absent /
not-yet-polled / not-yet-started states are represented by
*precalculated zero constants* (``_ZERO_*``) defined alongside their
type.  This keeps :class:`~fido.atomic.AtomicUpdater` simple — every
lens path always points at a real value, so writes are always pure
replacements with no "create the missing branch first" special case
— and SCADA readers never have to ``if foo is not None``.

Sentinel conventions per type are documented on each class.  In
general: ``""`` for absent strings, ``0`` for absent counts and IDs,
:data:`_EPOCH` for absent timestamps, and a class-specific
``_ZERO_*`` constant for absent composite values.
"""

from dataclasses import dataclass
from datetime import datetime, timezone

from frozendict import frozendict

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class ProviderLimitWindow:
    """One provider-specific limit window normalized for shared policy/UI use.

    Sentinels: ``limit == 0`` marks an *unpolled* window — no successful
    poll has populated the values yet.  Real provider windows always
    report a positive ``limit``, so the zero unambiguously means "no
    data".  ``used == 0`` with ``limit > 0`` is a perfectly legitimate
    "fully fresh quota" state.
    """

    name: str
    used: int
    limit: int
    resets_at: datetime
    unit: str

    @property
    def pressure(self) -> float:
        """Return ``used / limit``, or ``0.0`` for an unpolled window."""
        if self.limit <= 0:
            return 0.0
        return self.used / self.limit


_ZERO_WINDOW_REST = ProviderLimitWindow(
    name="rest", used=0, limit=0, resets_at=_EPOCH, unit=""
)
_ZERO_WINDOW_GRAPHQL = ProviderLimitWindow(
    name="graphql", used=0, limit=0, resets_at=_EPOCH, unit=""
)


@dataclass(frozen=True)
class GitHubLimit:
    """Normalized GitHub platform rate-limit state (REST + GraphQL windows).

    The zero value (:data:`_ZERO_GITHUB_LIMITS`) is the initial sentinel
    — both windows have ``limit=0``, meaning the monitor has not yet
    completed a successful poll.  After the first successful refresh
    both windows carry positive limits.

    Stored at :attr:`FidoState.github_limits`; updated atomically via
    :class:`~fido.atomic.AtomicUpdater`.
    """

    rest: ProviderLimitWindow
    graphql: ProviderLimitWindow


_ZERO_GITHUB_LIMITS = GitHubLimit(rest=_ZERO_WINDOW_REST, graphql=_ZERO_WINDOW_GRAPHQL)


@dataclass(frozen=True, slots=True)
class WorkerActivity:
    """Snapshot of what one worker is currently doing.

    Sentinel: ``what == ""`` marks the unpopulated zero — the worker
    has been registered but has not yet reported its first activity.
    The per-repo zero is built by :func:`zero_activity` because
    ``repo_name`` is part of the value.
    """

    repo_name: str
    what: str
    busy: bool
    last_progress_at: datetime


def zero_activity(repo_name: str) -> WorkerActivity:
    """Return the zero :class:`WorkerActivity` for *repo_name*."""
    return WorkerActivity(
        repo_name=repo_name, what="", busy=False, last_progress_at=_EPOCH
    )


@dataclass(frozen=True, slots=True)
class WorkerCrash:
    """Running record of unexpected worker deaths for one repo.

    Sentinel: ``death_count == 0`` marks "never crashed".  See
    :data:`_ZERO_CRASH`.
    """

    death_count: int
    last_error: str
    last_crash_time: datetime


_ZERO_CRASH = WorkerCrash(death_count=0, last_error="", last_crash_time=_EPOCH)


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

    No precalculated zero — webhook activities are *added* to the
    :attr:`RepoState.webhook_activities` tuple, never represented by a
    zero entry.  Absence is the empty tuple ``()``.
    """

    handle_id: int
    description: str
    started_at: datetime
    thread_id: int


@dataclass(frozen=True, slots=True)
class ProviderSnapshot:
    """Immutable snapshot of one repo's provider (ClaudeSession) state.

    Sentinel: ``session_owner == ""`` marks "no session has been
    created yet".  After a session starts, ``session_owner`` is the
    thread label (e.g. ``"worker-home"``).
    """

    session_owner: str
    session_alive: bool
    session_pid: int
    session_dropped_count: int
    session_sent_count: int
    session_received_count: int


_ZERO_PROVIDER = ProviderSnapshot(
    session_owner="",
    session_alive=False,
    session_pid=0,
    session_dropped_count=0,
    session_sent_count=0,
    session_received_count=0,
)


@dataclass(frozen=True, slots=True)
class ThreadSnapshot:
    """Immutable snapshot of one :class:`~fido.worker.WorkerThread`'s lifecycle state.

    Captures thread lifecycle metadata as primitive values — no handles to the
    mutable :class:`~fido.worker.WorkerThread` object itself.  Stored inside
    the frozen :class:`FidoState` so the SCADA display invariant is preserved:
    a snapshot reader can never reach through to a live thread and observe
    partial mutations.

    Sentinel: the zero (``is_alive=False``, ``was_stopped=False``,
    ``crash_error=""``) is what gets published for a freshly-registered
    repo before its thread has started.
    """

    is_alive: bool
    was_stopped: bool
    crash_error: str


_ZERO_THREAD = ThreadSnapshot(is_alive=False, was_stopped=False, crash_error="")


@dataclass(frozen=True, slots=True)
class IssueSnapshot:
    """Immutable snapshot of one repo's current issue / PR fields.

    Captures the disk-resident state.json values (issue id/title/started,
    pr id/title) that the SCADA display path renders for ``./fido status``
    and ``/status.json``.  Published by :class:`~fido.state.State` at every
    state.json write — independent leaf, no cross-source reads (#1696).

    Sentinels: ``issue == 0`` and ``pr_number == 0`` mark "no issue
    assigned" / "no PR yet" (real GitHub issue/PR numbers are positive).
    Strings default to ``""``.
    """

    issue: int
    issue_title: str
    issue_started_at: str
    pr_number: int
    pr_title: str


_ZERO_ISSUE = IssueSnapshot(
    issue=0, issue_title="", issue_started_at="", pr_number=0, pr_title=""
)


@dataclass(frozen=True, slots=True)
class TaskListSnapshot:
    """Immutable snapshot of one repo's current task-list-derived fields.

    Computed from tasks.json contents at every Tasks mutation;
    independent leaf in :class:`RepoState`, sibling of
    :class:`IssueSnapshot`.

    Sentinels: ``current_task == ""`` for "no task selected",
    ``task_number == 0`` and ``task_total == 0`` for "no tasks at all"
    or "all completed".
    """

    pending_task_count: int
    completed_task_count: int
    current_task: str
    task_number: int
    task_total: int


_ZERO_TASK_LIST = TaskListSnapshot(
    pending_task_count=0,
    completed_task_count=0,
    current_task="",
    task_number=0,
    task_total=0,
)


@dataclass(frozen=True, slots=True)
class IssueCacheSnapshot:
    """Immutable snapshot of one repo's :class:`~fido.issue_cache.IssueTreeCache`
    health for the SCADA display path.

    Mirrors the public fields of :class:`~fido.issue_cache.CacheMetrics`
    as JSON-friendly primitives.  Published by
    :class:`~fido.issue_cache.IssueTreeCache` after every mutation
    (load_inventory / apply_event / reconcile_with_inventory) via the
    ``on_change`` callback supplied at construction (#1696 parity).

    Sentinel: ``loaded=False`` means inventory has not been loaded;
    timestamp fields default to :data:`_EPOCH`.
    """

    loaded: bool
    open_issues: int
    events_applied: int
    events_dropped_stale: int
    last_event_at: datetime
    last_reconcile_at: datetime
    last_reconcile_drift: int


_ZERO_ISSUE_CACHE = IssueCacheSnapshot(
    loaded=False,
    open_issues=0,
    events_applied=0,
    events_dropped_stale=0,
    last_event_at=_EPOCH,
    last_reconcile_at=_EPOCH,
    last_reconcile_drift=0,
)


@dataclass(frozen=True, slots=True)
class TalkerSnapshot:
    """Immutable snapshot of one repo's currently-driving
    :class:`~fido.provider.SessionTalker` for the SCADA display path.

    Mirrors the public :class:`~fido.provider.SessionTalker` fields as
    JSON-friendly primitives.  Published by the provider module's
    register/unregister hooks via the per-repo on_change callback wired
    by :class:`~fido.registry.WorkerRegistry` at startup (#1696 parity).

    Sentinel: ``thread_id == 0`` and ``kind == ""`` mark "no thread is
    currently driving the provider for this repo" — the provider is
    idle.  Real ``threading.get_ident()`` results are positive.
    """

    thread_id: int
    kind: str  # "worker", "webhook", or "" for idle
    description: str
    subprocess_pid: int
    started_at: datetime


_ZERO_TALKER = TalkerSnapshot(
    thread_id=0, kind="", description="", subprocess_pid=0, started_at=_EPOCH
)


@dataclass(frozen=True, slots=True)
class ProviderPressureSnapshot:
    """Immutable snapshot of one repo's provider-pressure summary.

    Mirrors :class:`~fido.provider.ProviderPressureStatus` as
    JSON-friendly primitives (the level / warning / paused booleans are
    pre-computed so the JSON wire format need not re-derive them).
    Published by :class:`~fido.provider_pressure.ProviderPressureMonitor`
    to ``RepoState.provider_pressure`` once per polling cycle (#1696
    parity).

    Sentinel: ``level == "unknown"`` (matched by the ``provider == ""``
    zero) marks "no successful poll yet".
    """

    provider: str  # ProviderID enum value, e.g. "claude-code"
    window_name: str
    pressure: float
    percent_used: int
    resets_at: datetime
    unavailable_reason: str
    level: str  # "ok" / "warning" / "paused" / "unavailable" / "unknown"
    warning: bool
    paused: bool


_ZERO_PROVIDER_PRESSURE = ProviderPressureSnapshot(
    provider="",
    window_name="",
    pressure=0.0,
    percent_used=0,
    resets_at=_EPOCH,
    unavailable_reason="",
    level="unknown",
    warning=False,
    paused=False,
)


@dataclass(frozen=True, slots=True)
class RepoState:
    """Per-repo sub-snapshot within :class:`FidoState`.

    *key* is the repo slug (e.g. ``"rhencke/confusio"``), matching the key
    under which this record is stored in :attr:`FidoState.repos`.

    Every field is required and always carries a real (zero or non-zero)
    value — there is no ``None``, no defaults, no "absent" representation
    other than the precalculated zero constants on each leaf type.  This
    keeps :class:`~fido.atomic.AtomicUpdater` simple: every lens path is
    always valid, every write is just a replacement of an existing value.

    The per-repo zero is built by :func:`zero_repo_state` because ``key``
    and ``activity`` (which embeds ``repo_name``) carry the repo slug.
    """

    key: str
    started_at: datetime
    activity: WorkerActivity
    crash_record: WorkerCrash
    webhook_activities: tuple[WebhookActivity, ...]
    thread: ThreadSnapshot
    provider: ProviderSnapshot
    issue: IssueSnapshot
    task_list: TaskListSnapshot
    issue_cache: IssueCacheSnapshot
    talker: TalkerSnapshot
    provider_pressure: ProviderPressureSnapshot
    rescoping: bool


def zero_repo_state(repo_name: str, started_at: datetime = _EPOCH) -> RepoState:
    """Return a fresh zero :class:`RepoState` for *repo_name*.

    Used by :meth:`~fido.registry.WorkerRegistry.start` to prepopulate the
    snapshot when a repo first appears, and by tests that need a baseline
    state without enumerating every field.
    """
    return RepoState(
        key=repo_name,
        started_at=started_at,
        activity=zero_activity(repo_name),
        crash_record=_ZERO_CRASH,
        webhook_activities=(),
        thread=_ZERO_THREAD,
        provider=_ZERO_PROVIDER,
        issue=_ZERO_ISSUE,
        task_list=_ZERO_TASK_LIST,
        issue_cache=_ZERO_ISSUE_CACHE,
        talker=_ZERO_TALKER,
        provider_pressure=_ZERO_PROVIDER_PRESSURE,
        rescoping=False,
    )


@dataclass(frozen=True, slots=True)
class FidoState:
    """Atomically-swapped coordination snapshot.

    ``repos`` maps each repo slug to its :class:`RepoState` snapshot.  It is
    a :class:`frozendict` so stale readers that hold a reference to an old
    snapshot can never accidentally mutate the mapping — the immutability
    guarantee holds even on the free-threaded (no-GIL) build.

    The atomic cell is owned by the composition root (``run()`` in
    ``server.py``), not by :class:`~fido.registry.WorkerRegistry`.  The root
    creates both faces via :func:`~fido.atomic.create_atomic`, passes the
    :class:`~fido.atomic.AtomicUpdater` to
    :class:`~fido.registry.WorkerRegistry` and
    :class:`~fido.rate_limit.RateLimitMonitor`, and passes the
    :class:`~fido.atomic.AtomicReader` to the status serialisation path.

    Sentinel: ``process_started_at == _EPOCH`` marks "fido has not
    started" (used in tests and pre-startup state).  The composition
    root replaces it with ``datetime.now(...)`` once the server begins
    serving requests.
    """

    repos: frozendict[str, RepoState]
    github_limits: GitHubLimit
    process_started_at: datetime


_ZERO_FIDO_STATE = FidoState(
    repos=frozendict(),
    github_limits=_ZERO_GITHUB_LIMITS,
    process_started_at=_EPOCH,
)
