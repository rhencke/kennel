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

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol, TypeAlias, cast

from fido.rocq.transition import Free as _FsmFree
from fido.rocq.transition import HandlerAcquire as _FsmHandlerAcquire
from fido.rocq.transition import HandlerRelease as _FsmHandlerRelease
from fido.rocq.transition import OwnedByWorker as _FsmOwnedByWorker
from fido.rocq.transition import State as _FsmState
from fido.rocq.transition import WorkerAcquire as _FsmWorkerAcquire
from fido.rocq.transition import WorkerRelease as _FsmWorkerRelease
from fido.rocq.transition import transition as _fsm_transition


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

    def wait_for_pending_preempt(self, timeout: float = 30.0) -> bool:
        """Wait for any queued preemption work to drain before retrying a turn."""
        ...

    def switch_model(self, model: ProviderModel) -> None:
        """Switch the live session to *model* without resetting session state."""
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
    reentrance counter and :meth:`hold_for_handler` context manager
    (fix for #658).

    Subclasses own their own lock (must be a :class:`threading.RLock` so
    nested acquires by the same thread are free) and implement the
    session-specific first-enter / last-exit work in their own ``__enter__``
    and ``__exit__``.  This base only handles the per-thread depth
    bookkeeping and the shared ``hold_for_handler`` wrapper so the two
    providers don't drift apart.

    Required subclass attributes:

    - ``self._lock`` — :class:`threading.RLock`
    - ``self._repo_name`` — ``str | None`` (fed to
      :func:`try_preempt_worker` for the preempt decision)
    - ``self._fire_worker_cancel()`` — method that aborts whatever turn the
      current lock-holder's provider subprocess is running
    """

    _reentry_tls: threading.local
    _repo_name: str | None
    _oracle_state: _FsmState

    def _init_handler_reentry(self) -> None:
        """Subclasses call this from their ``__init__`` to set up the
        thread-local reentrance counter and the session-lock FSM oracle.
        Separate method (not ``__init__``) so the base doesn't compete
        with the subclass constructor signature."""
        self._reentry_tls = threading.local()
        # Oracle initial state: nobody holds the session lock.
        self._oracle_state: _FsmState = _FsmFree()

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

    def _oracle_on_acquire(self, kind: str) -> None:
        """Assert the session-lock FSM oracle accepts an outermost acquire.

        Called by subclasses from ``__enter__`` when the entry depth
        transitions 0 → 1.  *kind* is ``"worker"`` for the background
        worker thread or ``"webhook"`` for a webhook handler holding the
        session via :meth:`hold_for_handler`.

        Crashes with theorem name ``no_dual_ownership`` if the model
        rejects the event — i.e. the session is already owned and a
        second acquire arrived without an intervening release.

        The session RLock is held by the caller, so the oracle state
        update is implicitly serialized.
        """
        ev = _FsmWorkerAcquire() if kind == "worker" else _FsmHandlerAcquire()
        new_state = _fsm_transition(self._oracle_state, ev)
        if new_state is None:
            raise RuntimeError(
                f"session-lock FSM oracle: no_dual_ownership violated — "
                f"{type(ev).__name__} rejected in state "
                f"{type(self._oracle_state).__name__}"
            )
        self._oracle_state = cast(_FsmState, new_state)

    def _oracle_on_release(self) -> None:
        """Assert the session-lock FSM oracle accepts the outermost release.

        Called by subclasses from ``__exit__`` when the entry depth
        transitions 1 → 0.  Derives the release event from the current
        FSM state so the caller does not need to track the holder kind
        separately: :class:`_FsmOwnedByWorker` → ``WorkerRelease``,
        anything else → ``HandlerRelease``.

        Crashes with theorem name ``release_only_by_owner`` if the
        model rejects the event — i.e. the session is not owned by the
        releasing role (cross-release or spurious release from Free).

        The session RLock is held by the caller, so the oracle state
        update is implicitly serialized.
        """
        ev = (
            _FsmWorkerRelease()
            if isinstance(self._oracle_state, _FsmOwnedByWorker)
            else _FsmHandlerRelease()
        )
        new_state = _fsm_transition(self._oracle_state, ev)
        if new_state is None:
            raise RuntimeError(
                f"session-lock FSM oracle: release_only_by_owner violated — "
                f"{type(ev).__name__} rejected in state "
                f"{type(self._oracle_state).__name__}"
            )
        self._oracle_state = cast(_FsmState, new_state)

    def _fire_worker_cancel(self) -> None:
        """Abort the current lock-holder's turn.  Subclasses override
        with their provider-specific cancel mechanism."""
        raise NotImplementedError  # pragma: no cover — abstract hook

    @contextmanager
    def hold_for_handler(
        self, *, preempt_worker: bool = False
    ) -> Iterator["OwnedSession"]:
        """Hold the session lock across multiple prompt calls.

        Webhook handlers wrap their entire body in this so the worker
        can't acquire the lock between individual turns (triage → reply
        → reaction) and stall the reply behind a long worker turn (#658).
        Inner ``with session:`` / ``session.prompt`` calls re-acquire the
        RLock and skip the first-enter setup via the reentrance counter.

        When *preempt_worker* is true, fires the caller's
        :meth:`_fire_worker_cancel` once upfront (via
        :func:`try_preempt_worker`) so a currently-running worker turn
        aborts immediately rather than being waited out.
        """
        if preempt_worker:
            try_preempt_worker(self._repo_name, self._fire_worker_cancel)
        with self:  # type: ignore[attr-defined]
            yield self


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
