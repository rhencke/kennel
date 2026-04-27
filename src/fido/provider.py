"""Provider contracts and normalized provider-limit state.

Also home to provider-neutral session coordination primitives:
talker registry, thread-kind / thread-repo plumbing, preempt decision
gate, and :class:`OwnedSession` — the base class both
:class:`~fido.claude.ClaudeSession` and
:class:`~fido.copilotcli.CopilotCLISession` inherit from.  These all
used to live in :mod:`fido.claude` under ``Claude``-prefixed names
from before the copilot provider existed; they were moved here once both
providers needed them so the naming stopped lying about scope.
"""

import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol, TypeAlias

import fido.rocq.transition as fsm

log = logging.getLogger(__name__)


class ProviderID(StrEnum):
    """Supported LLM providers for fido."""

    CLAUDE_CODE = "claude-code"
    COPILOT_CLI = "copilot-cli"
    CODEX = "codex"


@dataclass(frozen=True)
class ProviderPalette:
    """ANSI truecolor palette for a provider's status rendering.

    ``dim_bg`` tints a repo's section block so all that repo's lines share
    a provider-identifying background; ``bright_fg`` colors the provider's
    token inside the global ``limits:`` line.  Both values are RGB triples
    in the 0–255 range.

    Guidelines for adding a new provider:

    * ``dim_bg`` should be noticeably tinted but dark enough that bright
      foreground colors (white, cyan, yellow, magenta) remain readable
      against it.  Aim for ≥4.5:1 contrast against white.
    * ``bright_fg`` should be saturated and mid-to-high lightness so it
      remains legible on a typical dark terminal background.  Light
      terminals will lose contrast — users can opt out with ``NO_COLOR``.
    """

    dim_bg: tuple[int, int, int]
    bright_fg: tuple[int, int, int]


# Provider-specific color palettes.  Kept in one table so the contrast
# audit test can iterate every provider and assert WCAG AA thresholds;
# :mod:`fido.status` looks colors up by ``ProviderID`` at render time.
PROVIDER_PALETTES: dict[ProviderID, ProviderPalette] = {
    ProviderID.CLAUDE_CODE: ProviderPalette(
        dim_bg=(60, 30, 5),  # noticeable burnt orange tint, still dark
        bright_fg=(255, 160, 60),  # Claude-orange, legible on dark terminals
    ),
    ProviderID.COPILOT_CLI: ProviderPalette(
        dim_bg=(40, 20, 60),  # obvious plum tint, still dark
        bright_fg=(180, 130, 255),  # Copilot-purple, legible on dark terminals
    ),
}


def palette_for(provider: ProviderID) -> ProviderPalette | None:
    """Return the :class:`ProviderPalette` for *provider*, or ``None``.

    Returns ``None`` for providers without a registered palette (e.g.
    ``CODEX`` today).  Callers treat ``None`` as "render without
    provider-specific color", not as an error.
    """
    return PROVIDER_PALETTES.get(provider)


class TurnSessionMode(StrEnum):
    """How a provider turn should treat existing conversation state."""

    REUSE = "reuse"
    FRESH = "fresh"


ReasoningEffort: TypeAlias = Literal["low", "medium", "high"]
ReasoningEffortSpec: TypeAlias = ReasoningEffort | tuple[ReasoningEffort, ...] | None


@dataclass(frozen=True, eq=False)
class ProviderModel:
    """Explicit provider model selection, including reasoning effort when supported."""

    model: str
    effort: ReasoningEffortSpec = None

    @property
    def efforts(self) -> tuple[ReasoningEffort, ...]:
        """Return configured effort levels in preference order."""
        if self.effort is None:
            return ()
        if isinstance(self.effort, tuple):
            return self.effort
        return (self.effort,)

    def __str__(self) -> str:
        return self.model

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ProviderModel):
            return (self.model, self.effort) == (other.model, other.effort)
        if isinstance(other, str):
            return self.model == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.model, self.effort))


def coerce_provider_model(model: ProviderModel | str) -> ProviderModel:
    """Return *model* as a :class:`ProviderModel`."""
    if isinstance(model, ProviderModel):
        return model
    return ProviderModel(model)


def model_name(model: ProviderModel | str) -> str:
    """Return the provider-native model id for *model*."""
    return coerce_provider_model(model).model


@dataclass(frozen=True)
class ProviderLimitWindow:
    """One provider-specific limit window normalized for shared policy/UI use."""

    name: str
    used: int | None = None
    limit: int | None = None
    resets_at: datetime | None = None
    unit: str | None = None

    @property
    def pressure(self) -> float | None:
        """Return ``used / limit`` when both sides are known and limit is positive."""
        if self.used is None or self.limit is None or self.limit <= 0:
            return None
        return self.used / self.limit


@dataclass(frozen=True)
class ProviderLimitSnapshot:
    """Normalized limit state for one provider at one point in time."""

    provider: ProviderID
    windows: tuple[ProviderLimitWindow, ...] = ()
    unavailable_reason: str | None = None

    def closest_to_exhaustion(self) -> ProviderLimitWindow | None:
        """Return the highest-pressure window, or the first window when pressure is unknown."""
        pressured = [window for window in self.windows if window.pressure is not None]
        if pressured:
            return max(pressured, key=lambda window: window.pressure or 0.0)
        return self.windows[0] if self.windows else None


PROVIDER_PRESSURE_WARN_THRESHOLD = 0.90
PROVIDER_PRESSURE_PAUSE_THRESHOLD = 0.95


@dataclass(frozen=True)
class ProviderPressureStatus:
    """Normalized provider-pressure summary used by status UI and pause policy."""

    provider: ProviderID
    window_name: str | None = None
    pressure: float | None = None
    resets_at: datetime | None = None
    unavailable_reason: str | None = None

    @classmethod
    def from_snapshot(cls, snapshot: ProviderLimitSnapshot) -> "ProviderPressureStatus":
        """Summarize the closest-to-hit window from *snapshot*."""
        closest = snapshot.closest_to_exhaustion()
        return cls(
            provider=snapshot.provider,
            window_name=closest.name if closest is not None else None,
            pressure=closest.pressure if closest is not None else None,
            resets_at=closest.resets_at if closest is not None else None,
            unavailable_reason=snapshot.unavailable_reason,
        )

    @property
    def percent_used(self) -> int | None:
        """Return the nearest whole-percent pressure, or ``None`` when unknown."""
        if self.pressure is None:
            return None
        return round(self.pressure * 100)

    @property
    def level(self) -> Literal["ok", "warning", "paused", "unavailable", "unknown"]:
        """Return the normalized pressure level for display/policy."""
        if self.unavailable_reason is not None:
            return "unavailable"
        if self.pressure is None:
            return "unknown"
        if self.pressure >= PROVIDER_PRESSURE_PAUSE_THRESHOLD:
            return "paused"
        if self.pressure >= PROVIDER_PRESSURE_WARN_THRESHOLD:
            return "warning"
        return "ok"

    @property
    def warning(self) -> bool:
        """Return whether the provider has crossed the warning threshold."""
        return self.level == "warning"

    @property
    def paused(self) -> bool:
        """Return whether the provider has crossed the pause threshold."""
        return self.level == "paused"


class PromptSession(Protocol):
    """Persistent prompt/session collaborator owned by a repo worker thread."""

    @property
    def owner(self) -> str | None:
        """Return the logical owner label for the session, if any."""
        ...

    @property
    def pid(self) -> int | None:
        """Return the subprocess PID backing the session, if any."""
        ...

    @property
    def session_id(self) -> str | None:
        """Return the provider-native persistent session identifier, if any."""
        ...

    @property
    def dropped_session_count(self) -> int:
        """Return how many stale persistent session ids were dropped, if tracked."""
        ...

    @property
    def sent_count(self) -> int:
        """Return cumulative number of user turns sent to the provider since boot.

        Accumulates across subprocess respawns — model switches and recoveries
        do not reset the count.
        """
        ...

    @property
    def received_count(self) -> int:
        """Return cumulative number of responses/events received from the provider since boot.

        Accumulates across subprocess respawns — model switches and recoveries
        do not reset the count.
        """
        ...

    def prompt(
        self,
        content: str,
        model: ProviderModel | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Run a single prompt turn and return the final assistant text."""
        ...

    def send(self, content: str) -> None:
        """Send a turn to the session without waiting for the final result."""
        ...

    def consume_until_result(self) -> str:
        """Block until the current in-flight turn produces its final text result."""
        ...

    @property
    def last_turn_cancelled(self) -> bool:
        """Report whether the most recent turn was cancelled by preemption."""
        ...

    def switch_model(self, model: ProviderModel) -> None:
        """Switch the live session to *model* in-place without kill, respawn,
        or session-state loss."""
        ...

    def switch_tools(self, tools: str | None) -> None:
        """Restrict or restore available tools for the continued session.

        *tools* is passed as ``--allowedTools`` to the subprocess on respawn
        while preserving ``--resume`` so conversation context carries across
        the mode boundary.  ``None`` = no restriction (worker mode); a
        non-empty string = the triage allowlist (handler mode, typically
        :data:`~fido.claude.HANDLER_ALLOWED_TOOLS`, enforcing #1042).
        No-op if unchanged.  Called automatically by
        :meth:`OwnedSession.hold_for_handler`.
        """
        ...

    def recover(self) -> None:
        """Reattach or revive the underlying session after an interruption."""
        ...

    def reset(self, model: ProviderModel | None = None) -> None:
        """Reset the session, optionally selecting a fresh initial *model*."""
        ...

    def is_alive(self) -> bool:
        """Return whether the underlying session transport is still alive."""
        ...

    def stop(self) -> None:
        """Shut down the session and release any underlying resources."""
        ...

    def __enter__(self) -> "PromptSession":
        """Enter the session's turn-serialization context."""
        ...

    def __exit__(self, *args: object) -> None:
        """Exit the session's turn-serialization context."""
        ...

    def hold_for_handler(
        self, *, preempt_worker: bool = False
    ) -> "contextmanager[PromptSession]":  # type: ignore[type-arg]
        """Hold the session lock across multiple prompt calls for webhook
        handlers.  Implemented by :class:`OwnedSession` (fix for
        #658)."""
        ...

    def preempt_worker(self) -> bool:
        """Fire the cancel signal synchronously if a worker currently holds
        the session.  Safe to call from any thread — does not require the
        caller's thread kind to be set.  Returns True if a worker was
        preempted, False if the session was idle or held by a webhook.
        Implemented by :class:`OwnedSession` (fix for #955)."""
        ...


# ── Provider-neutral session coordination ────────────────────────────────────
# At most one thread per repo may be "talking to" a provider subprocess at any
# moment — either via the persistent session's lock or via a one-shot
# provider invocation.  Concurrent registration for the same repo means
# something is leaking a sub-provider and we halt loud rather than silently
# proliferate processes.


class SessionLeakError(RuntimeError):
    """Raised when a second thread tries to talk to a provider for a repo
    that already has an active talker.  Fatal — fido halts rather than
    let provider subprocesses multiply silently."""


@dataclass(frozen=True)
class SessionTalker:
    """Snapshot of the thread currently driving a provider subprocess.

    *thread_id* is :func:`threading.get_ident` — a globally-unique integer
    identifier for the live thread, used to match this talker entry to a
    specific thread (e.g. a webhook handler's ``WebhookActivity``) when
    rendering status.  The human-readable thread name is looked up at
    display time rather than cached here.

    *kind* distinguishes between the persistent session path (``"worker"`` —
    the worker thread is inside a ``with session:`` block) and webhook
    handler invocations (``"webhook"``).  *description* is a short human
    label for status display.
    """

    repo_name: str
    thread_id: int
    kind: Literal["worker", "webhook"]
    description: str
    claude_pid: int
    started_at: datetime


_talkers: dict[str, SessionTalker] = {}
_talkers_lock = threading.Lock()

# Thread-local coordination state so downstream helpers (events, prompts)
# know which repo they belong to and whether the caller is a worker or a
# webhook.  Set at :func:`fido.server.WebhookHandler._process_action`
# entry and at :meth:`fido.worker.WorkerThread.run` entry, cleared on
# exit.  Reads fall back to sensible defaults for tests that don't set it.
_thread_local: threading.local = threading.local()


def set_thread_repo(repo_name: str | None) -> None:
    """Set (or clear, with ``None``) the repo_name for this thread."""
    if repo_name is None:
        if hasattr(_thread_local, "repo_name"):
            del _thread_local.repo_name
    else:
        _thread_local.repo_name = repo_name


def current_repo() -> str | None:
    """Return the repo_name set by :func:`set_thread_repo` on this thread."""
    return getattr(_thread_local, "repo_name", None)


def set_thread_kind(kind: Literal["worker", "webhook"] | None) -> None:
    """Set (or clear, with ``None``) the caller kind for this thread.

    Workers call this with ``"worker"`` at :meth:`WorkerThread.run` entry; the
    webhook handler calls it with ``"webhook"`` at ``_process_action`` entry.
    :meth:`ClaudeSession.prompt` and :meth:`CopilotCLISession.__enter__`
    consult it to decide whether their preempt signal should fire —
    webhooks preempt workers; workers and webhooks never preempt a running
    webhook (fix for #637).
    """
    if kind is None:
        if hasattr(_thread_local, "kind"):
            del _thread_local.kind
    else:
        _thread_local.kind = kind


def current_thread_kind() -> Literal["worker", "webhook"]:
    """Return the caller kind for this thread.  Defaults to ``"worker"``
    when not set (non-entry code paths and tests)."""
    return getattr(_thread_local, "kind", "worker")


def try_preempt_worker(
    repo_name: str | None, cancel_fn: Callable[[], None]
) -> tuple[bool, Literal["worker", "webhook"] | None]:
    """Invoke *cancel_fn* iff the calling thread is a webhook AND the
    session's current lock holder is a worker.  Otherwise do nothing.

    Returns ``(preempted, current_kind)`` — *preempted* is ``True`` only when
    *cancel_fn* was invoked; *current_kind* is the current holder's kind
    (``"worker"``, ``"webhook"``, or ``None`` when the session is idle).
    The caller uses that to log the outcome.

    Provider-neutral decision gate for #637.  The *mechanism* of cancelling
    a running turn differs across providers (stream-json ``control_request``
    for claude, ACP ``cancel(session_id)`` for copilot), but the *decision*
    is identical and lives here.  Worker callers never preempt anyone;
    webhooks queue behind other webhooks (FIFO on the session lock) instead
    of cancelling each other.
    """
    caller_kind = current_thread_kind()
    current = get_talker(repo_name) if repo_name is not None else None
    current_kind = current.kind if current is not None else None
    if caller_kind == "webhook" and current_kind == "worker":
        cancel_fn()
        return True, current_kind
    return False, current_kind


def register_talker(talker: SessionTalker) -> None:
    """Register *talker* as the active provider driver for its repo.

    Raises :class:`SessionLeakError` if a talker for the same repo is
    already registered — the guarantee is one provider session per repo
    at a time.
    """
    with _talkers_lock:
        existing = _talkers.get(talker.repo_name)
        if existing is not None:
            raise SessionLeakError(
                f"provider session leak for repo {talker.repo_name}: "
                f"tid={existing.thread_id} ({existing.kind}, "
                f"{existing.description}, pid={existing.claude_pid}) "
                f"still active when tid={talker.thread_id} ({talker.kind}, "
                f"{talker.description}, pid={talker.claude_pid}) tried to start"
            )
        _talkers[talker.repo_name] = talker


def unregister_talker(repo_name: str, thread_id: int) -> None:
    """Remove the talker entry for *repo_name* if it belongs to *thread_id*.

    Idempotent — safe to call from cleanup paths that may race the registry.
    Non-matching ``thread_id`` is a no-op (defensive against cross-thread
    cleanup bugs).
    """
    with _talkers_lock:
        existing = _talkers.get(repo_name)
        if existing is not None and existing.thread_id == thread_id:
            del _talkers[repo_name]


def get_talker(repo_name: str) -> SessionTalker | None:
    """Return the active talker for *repo_name*, or ``None`` if idle."""
    with _talkers_lock:
        return _talkers.get(repo_name)


def talker_now() -> datetime:
    """Seam for tests — patch this to freeze time in talker timestamps."""
    return datetime.now(tz=timezone.utc)


_session_resolver: Callable[[str], PromptSession | None] | None = None
"""Callback the event/webhook layer uses to find its repo's persistent
:class:`PromptSession` — installed once by :mod:`fido.server` at startup.

Every in-process prompt call goes through the persistent session, so this
is a required piece of wiring.  Callers fail loud if it's missing — the
only time that should happen is a forgotten resolver install, not a real
production path."""


def set_session_resolver(
    resolver: Callable[[str], PromptSession | None] | None,
) -> None:
    """Install (or clear) the session resolver callback."""
    global _session_resolver
    _session_resolver = resolver


def current_repo_session() -> PromptSession:
    """Return the live :class:`PromptSession` driving the current thread's
    repo.  Raises when either the thread-local repo_name or the session
    resolver is missing — those are wiring bugs, not conditions callers
    should paper over.
    """
    repo = current_repo()
    if repo is None:
        raise RuntimeError(
            "current_repo_session called without a thread-local repo_name"
            " — server.WebhookHandler._process_action and WorkerThread.run"
            " both set it; this caller is missing the install."
        )
    if _session_resolver is None:
        raise RuntimeError(
            "current_repo_session called before set_session_resolver — "
            "server._run() installs it at startup; nothing should run before."
        )
    session = _session_resolver(repo)
    if session is None:
        raise RuntimeError(
            f"no provider session registered for repo {repo} — worker thread "
            "has not yet created its session"
        )
    if not session.is_alive():
        raise RuntimeError(
            f"provider session for repo {repo} is not alive — watchdog should "
            "have restarted the worker thread"
        )
    return session


class OwnedSession:
    """Base class for :class:`~fido.claude.ClaudeSession` and
    :class:`~fido.copilotcli.CopilotCLISession` providing the shared
    reentrance counter, FSM-driven lock coordination, and the
    :meth:`hold_for_handler` context manager (fix for #658).

    The FSM extracted from ``models/session_lock.v`` is the **authoritative**
    lock coordinator.  :attr:`_fsm_state` is the single source of truth for
    which role currently holds the session.  :meth:`_fsm_acquire_worker`,
    :meth:`_fsm_acquire_handler`, and :meth:`_fsm_release` are the
    provider-neutral acquire/release primitives; subclasses call them at the
    outermost entry/exit boundary (depth 0 → 1 and 1 → 0).

    Handler threads have priority over the worker: :meth:`_fsm_acquire_worker`
    yields to any handler registered in :attr:`_handler_queue`, even when the
    state is :class:`~fido.rocq.transition.Free`.  Handler-on-handler ordering
    is FIFO — the queue preserves the order in which handlers called
    :meth:`_fsm_acquire_handler`.

    Subclasses must set:

    - ``self._repo_name`` — ``str | None`` (fed to
      :func:`try_preempt_worker` for the preempt decision)
    - ``self._fire_worker_cancel()`` — method that aborts whatever turn the
      current lock-holder's provider subprocess is running

    """

    _reentry_tls: threading.local
    _repo_name: str | None
    _fsm_lock: threading.Lock
    _fsm_cond: threading.Condition
    _fsm_state: fsm.State
    _handler_queue: list[threading.Event]
    # Allowed-tools value passed to :meth:`switch_tools` on handler entry.
    # ``None`` means no restriction (default for CopilotCLISession whose
    # switch_tools is a no-op; ClaudeSession sets this to
    # :data:`~fido.claude.HANDLER_ALLOWED_TOOLS` in its constructor).
    _handler_tools: str | None

    def _init_handler_reentry(self) -> None:
        """Subclasses call this from their ``__init__`` to set up the
        thread-local reentrance counter and the FSM-driven lock coordinator.
        Separate method (not ``__init__``) so the base doesn't compete
        with the subclass constructor signature."""
        self._reentry_tls = threading.local()
        # Authoritative FSM state — which role holds the session.
        # Protected by _fsm_lock; threads wait on _fsm_cond for state changes.
        self._fsm_lock = threading.Lock()
        self._fsm_cond = threading.Condition(self._fsm_lock)
        self._fsm_state: fsm.State = fsm.Free()
        # FIFO queue of threading.Event waiters for handler threads that could
        # not acquire immediately.  _fsm_release pops from the front and sets
        # the event to hand ownership to the next handler.
        self._handler_queue: list[threading.Event] = []
        # Tool restriction applied on handler entry via switch_tools().
        # ClaudeSession overrides to HANDLER_ALLOWED_TOOLS in its constructor.
        self._handler_tools: str | None = None

    def _bump_entry_depth(self) -> int:
        """Increment and return the new per-thread entry depth (1 at
        outermost entry, 2 at first nested, etc.)."""
        depth = getattr(self._reentry_tls, "depth", 0) + 1
        self._reentry_tls.depth = depth
        return depth

    def _drop_entry_depth(self) -> int:
        """Decrement and return the new per-thread entry depth (0 at
        outermost exit)."""
        depth = self._reentry_tls.depth - 1
        self._reentry_tls.depth = depth
        return depth

    def _fsm_acquire_worker(self) -> None:
        """Block until the worker can acquire the session.

        Waits on :attr:`_fsm_cond` until :attr:`_fsm_state` is
        :class:`~fido.rocq.transition.Free` **and** :attr:`_handler_queue`
        is empty.  Handlers have priority: if any handler is registered in
        the queue, the worker yields even when the state is ``Free``.

        Fires the ``WorkerAcquire`` FSM transition and updates
        :attr:`_fsm_state` atomically under :attr:`_fsm_lock`.
        """
        tid = threading.get_ident()
        with self._fsm_cond:
            waited = False
            while True:
                if isinstance(self._fsm_state, fsm.Free) and not self._handler_queue:
                    new = fsm.transition(self._fsm_state, fsm.WorkerAcquire())
                    assert new is not None  # Free + WorkerAcquire always succeeds
                    self._fsm_state = new  # → OwnedByWorker
                    log.debug(
                        "fsm[%s]: WorkerAcquire (tid=%d, waited=%s, queue=%d)",
                        self._repo_name or "?",
                        tid,
                        "yes" if waited else "no",
                        len(self._handler_queue),
                    )
                    return
                log.debug(
                    "fsm[%s]: WorkerAcquire blocked — state=%s, queue=%d (tid=%d)",
                    self._repo_name or "?",
                    type(self._fsm_state).__name__,
                    len(self._handler_queue),
                    tid,
                )
                waited = True
                self._fsm_cond.wait()

    def _fsm_acquire_handler(self) -> None:
        """Block until the handler can acquire the session.

        If :attr:`_fsm_state` is :class:`~fido.rocq.transition.Free`, fires
        ``HandlerAcquire`` and returns immediately.  Otherwise appends a
        :class:`threading.Event` waiter to the back of :attr:`_handler_queue`
        and blocks until :meth:`_fsm_release` transfers ownership here.

        Handler-on-handler ordering is FIFO: the queue preserves the order in
        which handlers registered.  When the worker is the current holder,
        the caller must have already fired :meth:`_fire_worker_cancel` so the
        worker drains its turn and calls :meth:`_fsm_release`.
        """
        tid = threading.get_ident()
        waiter: threading.Event | None = None
        with self._fsm_cond:
            new = fsm.transition(self._fsm_state, fsm.HandlerAcquire())
            if new is not None:
                # Free → OwnedByHandler: immediate acquisition.
                self._fsm_state = new
                log.debug(
                    "fsm[%s]: HandlerAcquire immediate (tid=%d)",
                    self._repo_name or "?",
                    tid,
                )
                return
            # Occupied; register in the FIFO handler queue and wait.
            waiter = threading.Event()
            self._handler_queue.append(waiter)
            log.debug(
                "fsm[%s]: HandlerAcquire queued — state=%s, position=%d (tid=%d)",
                self._repo_name or "?",
                type(self._fsm_state).__name__,
                len(self._handler_queue),
                tid,
            )
        # Wait outside the Condition so _fsm_release can acquire _fsm_lock.
        assert waiter is not None
        waiter.wait()
        # _fsm_release set _fsm_state = OwnedByHandler and signalled us.
        log.debug(
            "fsm[%s]: HandlerAcquire dequeued (tid=%d)",
            self._repo_name or "?",
            tid,
        )

    def _fsm_release(self) -> None:
        """Release the FSM lock.

        Fires the appropriate release event for the current holder
        (``WorkerRelease`` from
        :class:`~fido.rocq.transition.OwnedByWorker`,
        ``HandlerRelease`` from
        :class:`~fido.rocq.transition.OwnedByHandler`).

        If :attr:`_handler_queue` is non-empty, ownership transfers directly
        to the next queued handler: the handler's :class:`threading.Event` is
        set and :attr:`_fsm_state` becomes
        :class:`~fido.rocq.transition.OwnedByHandler`.  Otherwise
        :attr:`_fsm_state` transitions to :class:`~fido.rocq.transition.Free`
        and worker threads waiting on :attr:`_fsm_cond` are notified.

        Raises :class:`RuntimeError` if the current state is
        :class:`~fido.rocq.transition.Free` (``release_only_by_owner``
        invariant).
        """
        tid = threading.get_ident()
        with self._fsm_cond:
            ev = (
                fsm.WorkerRelease()
                if isinstance(self._fsm_state, fsm.OwnedByWorker)
                else fsm.HandlerRelease()
            )
            new_state = fsm.transition(self._fsm_state, ev)
            if new_state is None:
                raise RuntimeError(
                    f"session-lock FSM: release_only_by_owner violated — "
                    f"{type(ev).__name__} rejected in state "
                    f"{type(self._fsm_state).__name__}"
                )
            if self._handler_queue:
                # Hand ownership directly to the next waiting handler.
                waiter = self._handler_queue.pop(0)
                self._fsm_state = fsm.OwnedByHandler()
                waiter.set()
                log.debug(
                    "fsm[%s]: %s → OwnedByHandler (tid=%d, queue=%d remaining)",
                    self._repo_name or "?",
                    type(ev).__name__,
                    tid,
                    len(self._handler_queue),
                )
            else:
                # No handlers waiting; transition to Free and wake workers.
                self._fsm_state = new_state  # → Free
                self._fsm_cond.notify_all()
                log.debug(
                    "fsm[%s]: %s → Free (tid=%d)",
                    self._repo_name or "?",
                    type(ev).__name__,
                    tid,
                )

    def _fire_worker_cancel(self) -> None:
        """Abort the current lock-holder's turn.  Subclasses override
        with their provider-specific cancel mechanism."""
        raise NotImplementedError  # pragma: no cover — abstract hook

    def switch_tools(self, tools: str | None) -> None:
        """Restrict or restore the subprocess tool allowlist.

        Subclasses that wrap a persistent Claude Code subprocess (e.g.
        :class:`~fido.claude.ClaudeSession`) override this to respawn with
        ``--allowedTools <value>`` so handler turns get a triage tool set
        and worker turns get the full set back.  Subclasses that do not
        support tool switching (e.g. CopilotCLISession) override with a no-op.

        Called automatically by :meth:`hold_for_handler` on entry
        (``tools = self._handler_tools``) and on exit (``tools = None``).
        """
        raise NotImplementedError  # pragma: no cover — abstract hook

    def preempt_worker(self) -> bool:
        """Fire the cancel signal synchronously if a worker currently holds
        the session.

        Intended to be called from the HTTP handler thread in
        :meth:`~fido.server.WebhookHandler._do_post_inner`, **before** the
        background handler thread is spawned, so the cancel fires at
        webhook-arrival time rather than at background-thread-schedule time
        (fix for #955).

        Does not require the caller's thread kind to be ``"webhook"`` —
        the caller *is* the HTTP handler, so we know it's a webhook context
        without checking :func:`current_thread_kind`.

        Returns ``True`` if a worker was preempted, ``False`` if the session
        was idle or already held by another webhook.
        """
        current = get_talker(self._repo_name) if self._repo_name else None
        current_kind = current.kind if current is not None else None
        if current_kind == "worker":
            self._fire_worker_cancel()
            log.info(
                "webhook: preempting worker synchronously (repo=%s, tid=%d)",
                self._repo_name,
                threading.get_ident(),
            )
            return True
        log.info(
            "webhook: no worker to preempt — holder is %s (repo=%s, tid=%d)",
            current_kind or "none",
            self._repo_name,
            threading.get_ident(),
        )
        return False

    @contextmanager
    def hold_for_handler(
        self, *, preempt_worker: bool = False
    ) -> Iterator["OwnedSession"]:
        """Hold the session lock across multiple prompt calls.

        Webhook handlers wrap their entire body in this so the worker
        can't acquire the lock between individual turns (triage → reply
        → reaction) and stall the reply behind a long worker turn (#658).
        Inner ``with session:`` / ``session.prompt`` calls re-enter via the
        reentrance counter and skip the FSM acquire.

        When *preempt_worker* is true, fires the caller's
        :meth:`_fire_worker_cancel` once upfront (via
        :func:`try_preempt_worker`) so a currently-running worker turn
        aborts immediately rather than being waited out.  Logs the
        outcome — preempted vs. queuing — so webhook latency spikes
        are visible in the log without needing to trace ``session.prompt``
        calls (fixes the missing diagnostic from #955).

        Switches the session to triage handler mode (via
        :meth:`switch_tools` with :attr:`_handler_tools`) on entry, then
        restores the unrestricted worker tool set on exit (#1042).
        """
        if preempt_worker:
            preempted, current_kind = try_preempt_worker(
                self._repo_name, self._fire_worker_cancel
            )
            if preempted:
                log.info(
                    "hold_for_handler: preempting worker (tid=%d, repo=%s)",
                    threading.get_ident(),
                    self._repo_name,
                )
            else:
                log.info(
                    "hold_for_handler: queuing behind %s holder (tid=%d, repo=%s)",
                    current_kind or "none",
                    threading.get_ident(),
                    self._repo_name,
                )
        with self:  # type: ignore[attr-defined]
            self.switch_tools(self._handler_tools)
            try:
                yield self
            finally:
                self.switch_tools(None)


class ProviderAgent(Protocol):
    """Runtime/session boundary for a provider's interactive agent."""

    @property
    def provider_id(self) -> ProviderID:
        """Return the stable provider identifier for this agent."""
        ...

    @property
    def session(self) -> PromptSession | None:
        """Return the currently attached persistent session, if any."""
        ...

    @property
    def session_owner(self) -> str | None:
        """Return the attached session owner label, if available."""
        ...

    @property
    def session_alive(self) -> bool:
        """Return whether the attached session transport is currently alive."""
        ...

    @property
    def session_pid(self) -> int | None:
        """Return the attached session PID, if the provider exposes one."""
        ...

    @property
    def session_id(self) -> str | None:
        """Return the provider-native persistent session identifier, if any."""
        ...

    @property
    def session_dropped_count(self) -> int:
        """Return the attached session's dropped-session count, if available."""
        ...

    @property
    def session_sent_count(self) -> int:
        """Return the number of messages sent to the current session subprocess since spawn."""
        ...

    @property
    def session_received_count(self) -> int:
        """Return the number of stream-json events received from the current session subprocess since spawn."""
        ...

    voice_model: ProviderModel
    work_model: ProviderModel
    brief_model: ProviderModel

    def attach_session(self, session: PromptSession | None) -> None:
        """Attach *session* as the agent's persistent session."""
        ...

    def detach_session(self) -> PromptSession | None:
        """Detach and return the current persistent session, if any."""
        ...

    def ensure_session(
        self,
        model: ProviderModel | None = None,
        *,
        session_id: str | None = None,
    ) -> None:
        """Ensure that a persistent session exists, optionally seeded with
        *model* and resumed from *session_id* (for providers that support
        durable conversation ids — claude ``--resume``, Copilot ACP
        ``load_session``)."""
        ...

    def stop_session(self) -> None:
        """Stop and detach any currently attached persistent session."""
        ...

    def run_turn(
        self,
        content: str,
        *,
        model: ProviderModel | None = None,
        system_prompt: str | None = None,
        retry_on_preempt: bool = False,
        session_mode: TurnSessionMode = TurnSessionMode.REUSE,
    ) -> str:
        """Run one interactive turn through the persistent session and return text."""
        ...

    def print_prompt_from_file(
        self,
        system_file: Path,
        prompt_file: Path,
        model: ProviderModel,
        timeout: int = 30,
        idle_timeout: float = 1800.0,
        cwd: Path | str = ".",
    ) -> str:
        """Run a one-shot prompt sourced from files and return raw provider output."""
        ...

    def resume_session(
        self,
        session_id: str,
        prompt_file: Path,
        model: ProviderModel,
        timeout: int = 300,
        idle_timeout: float = 1800.0,
        cwd: Path | str = ".",
    ) -> str:
        """Resume a prior one-shot provider session and return raw output."""
        ...

    def generate_reply(
        self,
        prompt: str,
        model: ProviderModel | None = None,
        timeout: int = 30,
    ) -> str:
        """Generate a short natural-language reply for a GitHub comment flow."""
        ...

    def generate_branch_name(
        self,
        prompt: str,
        model: ProviderModel | None = None,
        timeout: int = 15,
    ) -> str:
        """Generate a git branch-name slug from *prompt*."""
        ...

    def generate_status(
        self,
        prompt: str,
        system_prompt: str,
        model: ProviderModel | None = None,
    ) -> str:
        """Generate a two-line GitHub status message."""
        ...

    def generate_status_emoji(
        self,
        prompt: str,
        system_prompt: str,
        model: ProviderModel | None = None,
    ) -> str:
        """Generate a single emoji suitable for a GitHub status."""
        ...

    def generate_status_with_session(
        self,
        prompt: str,
        system_prompt: str,
        model: ProviderModel | None = None,
        timeout: int = 15,
    ) -> tuple[str, str]:
        """Generate a status and return ``(status_text, resumable_session_id)``."""
        ...

    def resume_status(
        self,
        session_id: str,
        prompt: str,
        model: ProviderModel | None = None,
        timeout: int = 15,
    ) -> str:
        """Resume a prior status-generation session and return the refined text."""
        ...

    def extract_session_id(self, output: str) -> str:
        """Extract a session id from provider-specific raw one-shot output."""
        ...


def safe_voice_turn(
    agent: ProviderAgent,
    content: str,
    *,
    model: ProviderModel | None = None,
    system_prompt: str | None = None,
    log_prefix: str = "safe_voice_turn",
) -> str:
    """Run a voice turn with ``retry_on_preempt=True``; raise on empty.

    Centralises the ``retry_on_preempt=True`` flag so every voice-turn call
    site automatically retries when a session preemption returns an empty
    result.  If the result is still empty after retries, raises
    ``ValueError`` — the session reconnect layer handles recovery.
    """
    result = agent.run_turn(
        content,
        model=model,
        system_prompt=system_prompt,
        retry_on_preempt=True,
    )
    if not result:
        raise ValueError(f"{log_prefix}: run_turn returned empty after retries")
    return result


class ProviderAPI(Protocol):
    """Read-only account/service API surface for a provider."""

    @property
    def provider_id(self) -> ProviderID:
        """Return the stable provider identifier for this API client."""
        ...

    def get_limit_snapshot(self) -> ProviderLimitSnapshot:
        """Return the provider's current normalized rate-limit snapshot."""
        ...


class Provider(Protocol):
    """Composite provider with separate API and agent collaborators."""

    @property
    def provider_id(self) -> ProviderID:
        """Return the stable provider identifier for this composite provider."""
        ...

    @property
    def api(self) -> ProviderAPI:
        """Return the provider's read-only API collaborator."""
        ...

    @property
    def agent(self) -> ProviderAgent:
        """Return the provider's interactive runtime collaborator."""
        ...
