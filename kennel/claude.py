"""Claude CLI wrappers — all claude subprocess calls in one place."""

from __future__ import annotations

import json
import logging
import os
import re
import select
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import requests as _requests

from kennel.provider import (
    PromptSession,
    Provider,
    ProviderAgent,
    ProviderAPI,
    ProviderID,
    ProviderLimitSnapshot,
    ProviderLimitWindow,
    ProviderModel,
    TurnSessionMode,
    model_name,
)
from kennel.session_agent import SessionBackedAgent

log = logging.getLogger(__name__)

# Backstop timeout for select.select.  The wakeup pipe (_wakeup_r/_wakeup_w)
# normally kicks select awake immediately on cancel; this value only matters
# if the pipe write is lost.  Keep it short enough to bound worst-case
# preempt latency without busy-looping.
_SELECT_POLL_INTERVAL = 1.0

# Maximum number of characters included when logging a raw line from the
# claude subprocess, to keep log records readable.
_LOG_LINE_TRUNCATE = 200
_CLAUDE_API_TIMEOUT = 20
_CLAUDE_USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
_CLAUDE_USAGE_BETA = "oauth-2025-04-20"
_CLAUDE_USAGE_USER_AGENT = "claude-code/2.1.110"
_CLAUDE_USAGE_CACHE_SECONDS = 300.0


class _Trunc:
    """Lazy-truncating wrapper for log arguments.

    Pass an instance as a ``%s`` or ``%r`` log argument instead of slicing
    inline.  The truncation is deferred to format time, so it is free when
    the log level is disabled.

    Usage::

        log.debug("event: %s", _Trunc(line))
        log.warning("stderr=%r", _Trunc(result.stderr))
    """

    __slots__ = ("_s",)

    def __init__(self, s: str) -> None:
        self._s = s

    def __str__(self) -> str:
        return self._s[:_LOG_LINE_TRUNCATE]

    def __repr__(self) -> str:
        return repr(self._s[:_LOG_LINE_TRUNCATE])


# Tracked long-running claude subprocesses (the streaming ones), so kennel
# can clean them up on shutdown.  Short-lived ``subprocess.run`` calls cap
# at 30s and aren't tracked.  Use ``kill_active_children`` from a signal
# handler to terminate everything before exiting.
_active_children: set[subprocess.Popen[str]] = set()
_active_children_lock = threading.Lock()


def _register_child(proc: subprocess.Popen[str]) -> None:
    with _active_children_lock:
        _active_children.add(proc)


def _unregister_child(proc: subprocess.Popen[str]) -> None:
    with _active_children_lock:
        _active_children.discard(proc)


# ── Claude-talker registry ────────────────────────────────────────────────────
# At most one thread per repo may be "talking to" a claude subprocess at any
# moment — either via the persistent :class:`ClaudeSession` lock or via a
# one-shot ``claude --print`` (streaming or batch).  Concurrent registration
# for the same repo means something is leaking a sub-claude and we halt loud
# rather than silently proliferate processes.


class ClaudeLeakError(RuntimeError):
    """Raised when a second thread tries to talk to claude for a repo that
    already has an active talker.  Fatal — kennel halts rather than let
    sub-claudes multiply silently."""


@dataclass(frozen=True)
class ClaudeTalker:
    """Snapshot of the thread currently driving a claude subprocess.

    *thread_id* is :func:`threading.get_ident` — a globally-unique integer
    identifier for the live thread, used to match this talker entry to a
    specific thread (e.g. a webhook handler's :class:`WebhookActivity`) when
    rendering status.  The human-readable thread name is looked up at
    display time rather than cached here.

    *kind* distinguishes between the persistent session path (``"worker"`` —
    the worker thread is inside a ``with session:`` block) and one-shot
    ``claude --print`` invocations from a webhook handler (``"webhook"``).
    *description* is a short human label for status display.
    """

    repo_name: str
    thread_id: int
    kind: Literal["worker", "webhook"]
    description: str
    claude_pid: int
    started_at: datetime


_talkers: dict[str, ClaudeTalker] = {}
_talkers_lock = threading.Lock()

# Thread-local repo_name so downstream one-shot claude calls know which
# repo they belong to without plumbing repo_name through every helper in
# :mod:`kennel.events`.  Set at :func:`kennel.server.WebhookHandler._process_action`
# entry and at :meth:`kennel.worker.WorkerThread.run` entry, cleared on
# exit.  Reads fall back to ``None`` in tests and tools that do not set it.
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


def register_talker(talker: ClaudeTalker) -> None:
    """Register *talker* as the active claude driver for its repo.

    Raises :class:`ClaudeLeakError` if a talker for the same repo is already
    registered — the guarantee is one claude per repo at a time.
    """
    with _talkers_lock:
        existing = _talkers.get(talker.repo_name)
        if existing is not None:
            raise ClaudeLeakError(
                f"claude leak for repo {talker.repo_name}: "
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


def get_talker(repo_name: str) -> ClaudeTalker | None:
    """Return the active talker for *repo_name*, or ``None`` if idle."""
    with _talkers_lock:
        return _talkers.get(repo_name)


def _talker_now() -> datetime:
    """Seam for tests — override this module attribute to freeze time."""
    return datetime.now(tz=timezone.utc)


_session_resolver: Callable[[str], PromptSession | None] | None = None
"""Callback the event/webhook layer uses to find its repo's persistent
:class:`ClaudeSession` — installed once by :mod:`kennel.server` at startup.

Every in-process prompt call goes through the persistent session, so
this is a required piece of wiring.  Callers (:meth:`ClaudeClient.run_turn`) fail
loud if it's missing — the only time that should happen is a forgotten
resolver install, not a real production path."""


def set_session_resolver(
    resolver: Callable[[str], PromptSession | None] | None,
) -> None:
    """Install (or clear) the session resolver callback."""
    global _session_resolver
    _session_resolver = resolver


def current_repo_session() -> PromptSession:
    """Return the live :class:`ClaudeSession` driving the current thread's repo.

    Production always has both a thread-local ``repo_name`` (set by
    :func:`set_thread_repo` in the worker thread and webhook handler
    entrypoints) and an installed :func:`set_session_resolver` callback,
    and every worker reaches this code after :meth:`Worker.create_session`
    has populated the session.  So this raises rather than falling back —
    a missing session is a wiring bug, not a condition callers should
    paper over.
    """
    repo = current_repo()
    if repo is None:
        raise RuntimeError(
            "ClaudeClient.run_turn called without a thread-local repo_name"
            " — server.WebhookHandler._process_action and WorkerThread.run"
            " both set it; this caller is missing the install."
        )
    if _session_resolver is None:
        raise RuntimeError(
            "ClaudeClient.run_turn called before set_session_resolver — "
            "server._run() installs it at startup; nothing should run before."
        )
    session = _session_resolver(repo)
    if session is None:
        raise RuntimeError(
            f"no ClaudeSession registered for repo {repo} — worker thread "
            "has not yet created its session"
        )
    if not session.is_alive():
        raise RuntimeError(
            f"ClaudeSession for repo {repo} is not alive — watchdog should "
            "have restarted the worker thread"
        )
    return session


def _thread_name_for_id(thread_id: int) -> str | None:
    """Return the human-readable name of the live thread with *thread_id*,
    or ``None`` if that thread has exited.

    Used by status display to render a :class:`ClaudeTalker`'s thread
    without caching the name in the registry — a dead thread's entry is
    already being cleaned up and the name would be stale.
    """
    for t in threading.enumerate():
        if t.ident == thread_id:
            return t.name
    return None


def kill_active_children(grace_seconds: float = 2.0) -> None:
    """Send SIGTERM to every tracked claude subprocess, then SIGKILL stragglers.

    Called from kennel's shutdown signal handler so children don't outlive
    the parent and reparent to PID 1.  Safe to call multiple times.
    """
    with _active_children_lock:
        children = list(_active_children)
    if not children:
        return
    log.info("kennel shutdown: terminating %d claude child(ren)", len(children))
    for proc in children:
        try:
            proc.terminate()
        except OSError, ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_seconds
    for proc in children:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
                proc.wait(timeout=1.0)
            except OSError, ProcessLookupError, subprocess.TimeoutExpired:
                pass
        except OSError, ProcessLookupError:
            pass


def _claude(
    *args: str,
    prompt: str | None = None,
    timeout: int = 30,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    """Run the claude CLI with the given args, optionally piping prompt to stdin.

    Under the free-threaded (3.14t) runtime we observed
    ``subprocess.run(..., timeout=...)`` failing to fire on long-hung
    claude children — the worker sat on a futex for minutes past the
    requested timeout (closes #489).  When the default runner is used,
    drive the subprocess with explicit ``Popen`` + ``communicate`` so
    the child is guaranteed to be killed and reaped when the budget
    elapses.  Test overrides via *runner* still flow through whatever
    mock the test supplies.  Logs entry and exit so a stalled status
    call is localisable in the kennel log.
    """
    cmd = ["claude", *args]
    log.debug("_claude: running (timeout=%ds) %s", timeout, cmd[:3])
    if runner is not subprocess.run:
        return runner(
            cmd, input=prompt, capture_output=True, text=True, timeout=timeout
        )
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if prompt is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout = ""
    stderr = ""
    try:
        stdout, stderr = proc.communicate(input=prompt, timeout=timeout)
    except subprocess.TimeoutExpired:
        log.warning(
            "_claude: subprocess exceeded %ds — killing and re-raising", timeout
        )
        proc.kill()
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        raise subprocess.TimeoutExpired(
            proc.args, timeout, output=stdout, stderr=stderr
        )
    log.debug("_claude: returned rc=%d", proc.returncode)
    return subprocess.CompletedProcess(
        args=proc.args,
        returncode=proc.returncode,
        stdout=stdout,
        stderr=stderr,
    )


_TEST_EXPORTS = (_Trunc, _claude)


_RETURNCODE_IDLE_TIMEOUT = -1
"""Sentinel returncode used in :class:`ClaudeStreamError` when the process is
killed due to an idle timeout rather than exiting with a real non-zero code."""


class ClaudeStreamError(Exception):
    """Raised by _run_streaming when the subprocess exits with a non-zero code or times out."""

    def __init__(self, returncode: int) -> None:
        self.returncode = returncode
        super().__init__(f"claude exited with code {returncode}")


class ClaudeProviderError(RuntimeError):
    """Raised when Claude reports an API/provider failure for a turn."""

    def __init__(
        self,
        *,
        message: str,
        status_code: int | None = None,
        request_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.status_code = status_code
        self.request_id = request_id
        self.payload = payload or {}
        super().__init__(message)


# ── Stream-JSON helpers ───────────────────────────────────────────────────────


def extract_session_id(output: str) -> str:
    """Extract the session_id from stream-json output.

    Scans each line for a JSON object with ``"type": "result"`` and a
    non-empty ``"session_id"`` field.  Returns the last such value found
    (matching bash ``| tail -1``), or an empty string if none is present.
    """
    result = ""
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") == "result" and obj.get("session_id"):
            result = str(obj["session_id"])
    return result


def extract_result_text(output: str) -> str:
    """Extract the result text from stream-json output.

    Scans each line for a JSON object with ``"type": "result"`` and a
    non-empty string ``"result"`` field.  Returns the last such value found,
    or an empty string if none is present.
    """
    result = ""
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") == "result" and isinstance(obj.get("result"), str):
            text = obj["result"]
            if text:
                result = text
    return result


def _provider_error_from_result_text(text: str) -> ClaudeProviderError | None:
    """Parse a Claude CLI API error string from ``type=result`` text."""
    match = re.match(r"^API Error:\s*(\d+)\s+(.*)$", text.strip(), re.DOTALL)
    if not match:
        return None
    status_code = int(match.group(1))
    tail = match.group(2).strip()
    payload: dict[str, Any] = {}
    message = tail
    request_id = None
    try:
        parsed = json.loads(tail)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        payload = parsed
        error = parsed.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or tail)
        request_id_value = parsed.get("request_id")
        if isinstance(request_id_value, str) and request_id_value:
            request_id = request_id_value
    return ClaudeProviderError(
        message=f"Claude API error {status_code}: {message}",
        status_code=status_code,
        request_id=request_id,
        payload=payload,
    )


def _provider_error_from_event(obj: dict[str, Any]) -> ClaudeProviderError | None:
    """Return a provider error for a stream-json event when it encodes one."""
    event_type = obj.get("type")
    if event_type == "result" and isinstance(obj.get("result"), str):
        return _provider_error_from_result_text(obj["result"])
    if event_type != "error":
        return None
    error_obj = obj.get("error")
    if isinstance(error_obj, dict):
        message = str(error_obj.get("message") or error_obj)
    else:
        message = str(error_obj or obj)
    request_id = obj.get("request_id")
    return ClaudeProviderError(
        message=f"Claude API error: {message}",
        request_id=request_id if isinstance(request_id, str) else None,
        payload=obj,
    )


def raise_for_provider_error_output(output: str) -> None:
    """Raise the first provider error encoded in stream-json *output*."""
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            provider_error = _provider_error_from_result_text(stripped)
            if provider_error is not None:
                raise provider_error
            continue
        provider_error = _provider_error_from_event(obj)
        if provider_error is not None:
            raise provider_error


_EMPTY_RETRY_COUNT = 2
_EMPTY_RETRY_DELAY = 1.0


def _run_streaming(
    cmd: list[str],
    stdin_file: Path,
    idle_timeout: float = 1800.0,
    cwd: Path | str | None = None,
    popen: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    selector: Callable[..., tuple[list[Any], list[Any], list[Any]]] = select.select,
    clock: Callable[[], float] = time.monotonic,
) -> Iterator[str]:
    """Run a command, streaming stdout with idle-timeout detection.

    Yields each line of stdout as it arrives.  If no new output arrives for
    *idle_timeout* seconds, the process is killed and
    ``ClaudeStreamError(_RETURNCODE_IDLE_TIMEOUT)`` is raised.  If the process
    exits with a non-zero code,
    ``ClaudeStreamError(returncode)`` is raised.  ``FileNotFoundError``
    propagates naturally if the command is not found.
    """
    proc = popen(
        cmd,
        stdin=stdin_file.open(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
    )
    _register_child(proc)
    assert proc.stdout is not None  # guaranteed by stdout=PIPE
    repo_name = current_repo()
    thread_id = threading.get_ident()
    talker_registered = False
    if repo_name is not None:
        register_talker(
            ClaudeTalker(
                repo_name=repo_name,
                thread_id=thread_id,
                kind="webhook",
                description=f"one-shot claude --print (pid {proc.pid})",
                claude_pid=proc.pid,
                started_at=_talker_now(),
            )
        )
        talker_registered = True

    try:
        last_activity = clock()

        while True:
            ready, _, _ = selector([proc.stdout], [], [], _SELECT_POLL_INTERVAL)
            if ready:
                line = proc.stdout.readline()
                if not line:
                    break  # EOF
                yield line
                log.debug(line.rstrip())
                last_activity = clock()
            elif proc.poll() is not None:
                break  # process exited
            elif clock() - last_activity > idle_timeout:
                log.warning("claude idle for %.0fs — killing", idle_timeout)
                proc.kill()
                proc.wait()
                raise ClaudeStreamError(_RETURNCODE_IDLE_TIMEOUT)

        proc.wait()
        # Drain any remaining output
        remaining = proc.stdout.read()
        if remaining:
            yield remaining
        if proc.returncode != 0:
            raise ClaudeStreamError(proc.returncode)
    finally:
        if talker_registered and repo_name is not None:
            unregister_talker(repo_name, thread_id)
        _unregister_child(proc)


# ── Persistent bidirectional session ─────────────────────────────────────────


class ClaudeSession:
    """A long-lived claude process driven via bidirectional stream-json.

    Spawns ``claude --input-format stream-json --output-format stream-json``
    and keeps it running across multiple turns.  Each :meth:`send` writes one
    JSON user message to stdin; :meth:`iter_events` reads structured events
    from stdout until the turn completes (``type=result`` or ``type=error``).

    **Lifetime / persistence model**

    The session outlives individual :class:`~kennel.worker.Worker` crashes:
    :class:`~kennel.worker.WorkerThread` holds the session in
    ``_session`` across iterations and passes it into each new ``Worker``
    instance, so an unexpected exception in ``Worker.run()`` does not tear the
    session down.  The watchdog restarts the thread and the next Worker
    inherits the same session.

    The session does *not* survive a kennel/home restart.  When kennel
    replaces itself via ``os.execvp`` (e.g. after a self-update), all
    in-memory state is lost, including the ``WorkerThread`` and its session.
    The new process starts with ``_session = None`` and creates a fresh session
    on the first iteration.

    Pass *popen* and *selector* to override subprocess creation and
    ``select.select`` in tests; these default to the real implementations.
    """

    def __init__(
        self,
        system_file: Path,
        work_dir: Path | str | None = None,
        model: ProviderModel | None = None,
        idle_timeout: float = 1800.0,
        popen: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
        selector: Callable[..., tuple[list[Any], list[Any], list[Any]]] = select.select,
        repo_name: str | None = None,
    ) -> None:
        self._idle_timeout = idle_timeout
        self._selector = selector
        self._system_file = system_file
        self._work_dir = work_dir
        self._popen_fn = popen
        # Reentrant so :meth:`switch_model` and :meth:`restart` can
        # reacquire while called from inside a ``with session:`` block
        # (e.g. ``prompt()`` acquires the lock, then calls switch_model
        # which also needs to serialize with other sessions' access —
        # a plain threading.Lock self-deadlocks).
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._repo_name = repo_name
        self._model = model_name(
            ProviderModel("claude-opus-4-6") if model is None else model
        )
        # Latest session_id seen in a stream-json event.  Updated inside
        # :meth:`iter_events` so a subsequent :meth:`switch_model` can
        # restart with ``--resume <sid>`` and keep conversation context
        # across the model swap.  Empty until the first claude event with
        # a session_id arrives.
        self._session_id = ""
        # True when the most recent :meth:`iter_events` call exited early
        # because :attr:`_cancel` was set (i.e. another thread preempted the
        # turn via :meth:`prompt`).  Cleared at the start of each turn.
        # Callers use this to distinguish "turn completed with empty result"
        # from "turn was interrupted and should be retried once the lock is
        # free again".
        self._last_turn_cancelled = False
        # True when a :meth:`send` has been issued whose ``type=result``
        # boundary has not yet been consumed.  Cleared when
        # :meth:`iter_events` sees ``type=result``/``type=error``/EOF.  When
        # still True at the start of the next :meth:`send`, the prior turn
        # was cancelled without draining — its result (and any tail events)
        # is still on stdout and would be read by the next caller's
        # :meth:`consume_until_result` as its own.  That's the stream-leak
        # root cause in #499: without this flag we can't tell the stream
        # is dirty.  :meth:`send` drains to the boundary before writing new
        # content so every turn starts on a clean slate.
        self._in_turn = False
        # Set by :meth:`prompt` right after :attr:`_cancel` and before it
        # blocks on :attr:`_lock`.  Cleared inside :meth:`__enter__` once the
        # preempter actually acquires the lock.  Workers check this in their
        # retry loop to yield fairly: without it, a freshly-released worker
        # thread can re-acquire :attr:`_lock` before the waiting webhook
        # gets scheduled, and the next :meth:`iter_events` clears
        # :attr:`_cancel` — starving the preempter for a full worker turn.
        # See yield-starvation discussion in #499 comments.
        self._preempt_pending = threading.Event()
        # Wakeup pipe: writing a byte to _wakeup_w kicks select() out of its
        # blocking wait in iter_events() so the cancel signal is noticed
        # immediately instead of waiting up to _SELECT_POLL_INTERVAL.
        self._wakeup_r, self._wakeup_w = os.pipe()
        os.set_blocking(self._wakeup_r, False)
        os.set_blocking(self._wakeup_w, False)
        self._proc = self._spawn()
        _register_child(self._proc)

    def wait_for_pending_preempt(self, timeout: float = 30.0) -> bool:
        """Block for up to *timeout* seconds while a preempter holds the
        lock queue.  Returns ``True`` if the preemption completed within the
        window, ``False`` on timeout (no preempter pending in the first place
        returns immediately with ``False``).

        Workers call this after their cancelled-turn exit to let the
        preempter actually acquire :attr:`_lock` before the worker retries —
        Python's :class:`threading.RLock` isn't FIFO-fair under contention
        so the worker can otherwise race ahead and starve the preempter for
        a full turn.
        """
        if not self._preempt_pending.is_set():
            return False
        # Wait for the preempter's __enter__ to clear the event, meaning they
        # hold the lock now.  If they don't manage within the deadline, bail.
        # is_set() -> wait for clear: threading.Event has no "wait-for-clear",
        # so poll with short sleeps.
        import time as _time

        started = _time.monotonic()
        deadline = started + timeout
        log.info(
            "session: worker ceding lock to pending preempter (tid=%d)",
            threading.get_ident(),
        )
        while self._preempt_pending.is_set():
            if _time.monotonic() >= deadline:
                log.warning(
                    "session: preempter still pending after %.2fs — worker "
                    "proceeding anyway",
                    timeout,
                )
                return False
            _time.sleep(0.01)
        log.info(
            "session: preempter acquired lock after %.3fs yield",
            _time.monotonic() - started,
        )
        return True

    def _wake(self) -> None:
        """Write a byte to the wakeup pipe to kick select() awake."""
        try:
            os.write(self._wakeup_w, b"\x00")
        except OSError:
            pass  # pipe full or closed — cancel event is the authority

    def _drain_wakeup(self) -> None:
        """Drain any pending bytes from the wakeup pipe."""
        try:
            while os.read(self._wakeup_r, 1024):
                pass
        except OSError:
            pass  # EAGAIN (empty) or closed — either way, drained

    @property
    def repo_name(self) -> str | None:
        """Repo this session belongs to, for :class:`ClaudeTalker` registration."""
        return self._repo_name

    @property
    def last_turn_cancelled(self) -> bool:
        """``True`` when the most recent :meth:`iter_events` call exited
        early because another thread set the cancel event (preempted the
        turn via :meth:`prompt` or :meth:`interrupt`).

        Callers that want resumption semantics can check this after a turn
        and re-send the same content once the session lock is free again —
        effectively 'hand the session back to the worker and ask it to
        resume what it was doing'.
        """
        return self._last_turn_cancelled

    def _spawn(self) -> subprocess.Popen[str]:
        """Spawn the claude subprocess with bidirectional stream-json I/O.

        Model is set via ``--model`` at spawn time — the runtime ``/model``
        slash command isn't honored in stream-json mode (claude echoes
        "Unknown command: /model" and hangs without producing a turn
        boundary).  When :attr:`_session_id` is non-empty the new process
        resumes the prior conversation via ``--resume`` so context
        survives a model swap.
        """
        cmd = [
            "claude",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
            "--model",
            self._model,
            "--system-prompt-file",
            str(self._system_file),
        ]
        if self._session_id:
            cmd += ["--resume", self._session_id]
        return self._popen_fn(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self._work_dir,
        )

    def is_alive(self) -> bool:
        """Return True if the claude subprocess is still running."""
        return self._proc.poll() is None

    @property
    def pid(self) -> int:
        """PID of the live claude subprocess.

        Read directly off the tracked ``Popen`` — callers should use this
        rather than pgrep, since :class:`ClaudeSession` uses
        ``sub/persona.md`` (outside any ``fido_dir``) as its system prompt
        and the pgrep-based heuristic in :mod:`kennel.status` can't find it.
        """
        return self._proc.pid

    @property
    def session_id(self) -> str | None:
        """Durable Claude conversation id, if one has been established."""
        return self._session_id or None

    @property
    def dropped_session_count(self) -> int:
        """Claude stream-json sessions do not track dropped resume tokens."""
        return 0

    def _respawn(self, *, clear_session_id: bool, reason: str) -> None:
        """Stop the current subprocess and spawn a replacement."""
        log.info("ClaudeSession: %s (model=%s)", reason, self._model)
        with self._lock:
            _unregister_child(self._proc)
            if self._proc.poll() is None:
                try:
                    self._proc.kill()
                    self._proc.wait(timeout=1.0)
                except (OSError, ProcessLookupError, subprocess.TimeoutExpired) as exc:
                    log.warning("ClaudeSession._respawn: kill/wait failed: %s", exc)
                    raise
            if clear_session_id:
                self._session_id = ""
            # Fresh subprocess has no in-flight turn, so next send() skips
            # the drain path.
            self._in_turn = False
            self._proc = self._spawn()
            _register_child(self._proc)
        log.info("ClaudeSession: respawn complete, new pid %d", self._proc.pid)

    def recover(self) -> None:
        """Respawn the subprocess while preserving the durable conversation id."""
        self._respawn(clear_session_id=False, reason="recovering session")

    def reset(self, model: ProviderModel | None = None) -> None:
        """Respawn the subprocess with a fresh conversation."""
        if model is not None:
            self._model = model_name(model)
        self._respawn(clear_session_id=True, reason="resetting conversation")

    @property
    def owner(self) -> str | None:
        """Name of the thread currently holding the session lock, or ``None``.

        Derived from the global :class:`ClaudeTalker` registry so reads are
        always serialized through ``_talkers_lock`` — correct under the
        free-threaded (3.14t) runtime without relying on the GIL.  Returns
        ``None`` for sessions without a ``repo_name`` (unit-test fixtures)
        and when the active talker is a webhook rather than the worker.
        """
        if self._repo_name is None:
            return None
        talker = get_talker(self._repo_name)
        if talker is None or talker.kind != "worker":
            return None
        return _thread_name_for_id(talker.thread_id)

    def __enter__(self) -> "ClaudeSession":
        """Acquire the session lock, serializing send/receive across threads.

        Registers a ``worker``-kind :class:`ClaudeTalker` so status surfaces
        this thread as the one driving claude.  Does *not* clear the cancel
        event — that is deferred to :meth:`iter_events` so a signal that
        lands between one holder's :meth:`__exit__` and the next holder's
        :meth:`iter_events` is not silently dropped.

        Raises :class:`ClaudeLeakError` if another thread is already
        registered as the talker for this repo — indicates a sub-claude is
        leaking.  On leak, the session lock is released before raising so the
        holder we would have taken over from does not deadlock.
        """
        self._lock.acquire()
        # We hold the lock now; any preempter waiting on this
        # (wait_for_pending_preempt) can wake.
        self._preempt_pending.clear()
        if self._repo_name is not None:
            try:
                register_talker(
                    ClaudeTalker(
                        repo_name=self._repo_name,
                        thread_id=threading.get_ident(),
                        kind="worker",
                        description="persistent session turn",
                        claude_pid=self._proc.pid,
                        started_at=_talker_now(),
                    )
                )
            except ClaudeLeakError:
                self._lock.release()
                raise
        return self

    def __exit__(self, *args: object) -> None:
        """Release the session lock.  Unregisters the :class:`ClaudeTalker`
        before releasing the lock so no other thread can race in and see our
        stale talker entry.
        """
        if self._repo_name is not None:
            unregister_talker(self._repo_name, threading.get_ident())
        self._lock.release()

    def send(self, content: str) -> None:
        """Write a user message to the session stdin, flushing immediately.

        If the prior turn was cancelled without draining (:attr:`_in_turn`
        still True), abort it via ``control_request`` and read events until
        the turn boundary first — otherwise the next
        :meth:`consume_until_result` would return that prior turn's
        ``type=result`` and the caller would receive stale content as its
        own (the stream-leak in #499).
        """
        if self._in_turn:
            self._drain_to_boundary()
        msg = json.dumps(
            {"type": "user", "message": {"role": "user", "content": content}}
        )
        assert self._proc.stdin is not None
        self._proc.stdin.write(msg + "\n")
        self._proc.stdin.flush()
        self._in_turn = True

    def _drain_to_boundary(self, deadline: float = 10.0) -> None:
        """Abort the in-flight turn and read events until ``type=result`` /
        ``type=error`` / EOF, discarding them.  Used by :meth:`send` before
        writing a new user message when a prior turn was cancelled.

        Sends a ``control_request`` interrupt so claude closes the turn
        quickly rather than running it to completion — typical drain is
        well under a second.  Falls back to :meth:`recover` if the
        deadline elapses without a boundary, so a wedged subprocess can't
        stall the caller indefinitely.
        """
        assert self._proc.stdout is not None
        if self._proc.poll() is not None:
            self._in_turn = False
            return
        try:
            self._send_control_interrupt()
        except (BrokenPipeError, OSError) as exc:
            log.warning(
                "ClaudeSession._drain_to_boundary: control_request failed: %s", exc
            )
            self._in_turn = False
            return
        end_time = time.monotonic() + deadline
        while time.monotonic() < end_time:
            ready, _, _ = self._selector(
                [self._proc.stdout, self._wakeup_r], [], [], _SELECT_POLL_INTERVAL
            )
            self._drain_wakeup()
            if self._proc.stdout in ready:
                line = self._proc.stdout.readline()
                if not line:
                    break  # EOF
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                sid = obj.get("session_id")
                if isinstance(sid, str) and sid:
                    self._session_id = sid
                # Humanify drained events at INFO too so a cancelled turn's
                # tail is visible in the log rather than silently discarded.
                self._log_event(obj)
                if obj.get("type") in ("result", "error"):
                    log.info(
                        "ClaudeSession: drained stale %s event",
                        obj.get("type"),
                    )
                    self._in_turn = False
                    return
            elif self._proc.poll() is not None:
                break
        log.warning(
            "ClaudeSession._drain_to_boundary: no boundary after %.1fs — restarting",
            deadline,
        )
        self._in_turn = False
        self.recover()

    def _send_control_interrupt(self) -> None:
        """Write a stream-json ``control_request`` interrupt to subprocess stdin.

        Tells the Claude subprocess to abort the current turn at the protocol
        level.  The subprocess responds with a ``control_response`` on stdout
        and then emits a ``type=result`` to close the turn.  Call this while
        holding the session lock so it does not race with other stdin writes.
        """
        msg = json.dumps(
            {
                "type": "control_request",
                "request_id": str(uuid.uuid4()),
                "request": {"subtype": "interrupt"},
            }
        )
        assert self._proc.stdin is not None
        self._proc.stdin.write(msg + "\n")
        self._proc.stdin.flush()

    def interrupt(self, content: str) -> None:
        """Interrupt the in-flight turn at the protocol level, then send *content*.

        One owner-controlled sequence while holding the lock:

        1. Sets the cancel event so any concurrent :meth:`iter_events` caller
           exits on its next poll cycle and releases the lock.
        2. Acquires the lock (blocks until the holder exits).
        3. Sends a stream-json ``control_request`` interrupt so the Claude
           subprocess aborts the current turn (not just our local reading of it).
        4. Drains events until the turn boundary (``type=result`` /
           ``type=error`` / EOF) so the stream is clean for the next caller.
        5. Sends *content* as the follow-up user message.

        This guarantees no unread old-turn output is left on stdout for the
        next :meth:`iter_events` caller to inherit.
        """
        self._cancel.set()
        self._wake()
        with self._lock:
            self._send_control_interrupt()
            self.consume_until_result()
            self.send(content)

    def prompt(
        self,
        content: str,
        model: ProviderModel | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Steal the session, send *content* as a user message, return the result.

        Intended as the session-aware replacement for one-shot
        client turn helpers on webhook-
        handler threads.  Runs one full turn on the persistent session:

        1. Signal cancel to wake any current holder out of
           :meth:`iter_events`.  They exit their turn early, their context
           manager releases the lock and unregisters their talker.
        2. Acquire ``self`` as a context manager — blocks while the previous
           holder is winding down.  Registers a fresh :class:`ClaudeTalker`
           so status display attributes claude to this thread.
        3. Switch model if *model* is provided and differs from the session
           default.
        4. Send *content* (optionally prefixed with *system_prompt*) and
           consume until the result boundary, returning the text.

        Does **not** send a ``control_request`` interrupt to the subprocess
        before the user message.  On a fresh subprocess with no in-flight
        turn, claude ignores the control_request and never emits a
        ``type=result``, so ``consume_until_result`` would hang
        indefinitely.  The :attr:`_cancel` signal + lock hand-off already
        ensures the previous holder's :meth:`iter_events` has exited, so
        there is nothing in-flight to interrupt.
        """
        self._cancel.set()
        self._wake()
        # Mark that a preempter is queued so the current lock holder (a
        # worker) can wait_for_pending_preempt and cede the lock fairly
        # instead of racing to re-acquire after it yields.
        self._preempt_pending.set()
        log.info(
            "session.prompt: preempt requested (tid=%d, model=%s)",
            threading.get_ident(),
            self._model if model is None else model_name(model),
        )
        try:
            with self:
                if model is not None:
                    self.switch_model(model)
                if system_prompt:
                    body = f"{system_prompt}\n\n---\n\n{content}"
                else:
                    body = content
                self.send(body)
                return self.consume_until_result()
        finally:
            # If an exception blew us out of `with self:` before __enter__
            # could clear the event, do it here so a stuck event doesn't
            # trap the worker forever.
            self._preempt_pending.clear()

    def switch_model(self, model: ProviderModel | str) -> None:
        """Switch the active model.  Restart-based — stream-json does not
        accept ``/model`` or any slash command (claude echoes "Unknown
        command" and never emits a turn boundary, hanging the reader).

        Holds :attr:`_lock` for the full swap so callers waiting on
        :meth:`__enter__` block gracefully until the new subprocess is
        listening.  When a prior ``session_id`` is known we pass
        ``--resume`` to the new subprocess so the conversation transcript
        carries over across the swap — no context loss when phase
        transitions flip opus → sonnet or vice versa.

        No-op when *model* equals the current model.
        """
        target_model = model_name(model)
        if target_model == self._model:
            return
        log.info(
            "switch_model: %s → %s (restart-based, resume=%s)",
            self._model,
            target_model,
            self._session_id or "—",
        )
        with self._lock:
            _unregister_child(self._proc)
            if self._proc.poll() is None:
                try:
                    self._proc.kill()
                    self._proc.wait(timeout=1.0)
                except (OSError, ProcessLookupError, subprocess.TimeoutExpired) as exc:
                    log.warning(
                        "switch_model: kill/wait of old subprocess failed: %s", exc
                    )
                    raise
            self._model = target_model
            # Fresh subprocess — any in-flight turn on the old one died
            # with it, so next send() has nothing to drain.
            self._in_turn = False
            self._proc = self._spawn()
            _register_child(self._proc)
        log.info(
            "switch_model: new pid %d ready (model=%s)",
            self._proc.pid,
            target_model,
        )

    def _log_event(self, obj: dict[str, Any]) -> None:
        """Emit a human-readable INFO log line for a stream-json *obj*.

        Makes stalls pinpointable to a specific tool call or turn rather
        than leaving a silent gap in the kennel log between "preempt
        requested" and "preempter acquired".  Closes #493.
        """
        t = obj.get("type")
        if t == "assistant":
            message = obj.get("message", {})
            for c in message.get("content") or []:
                if not isinstance(c, dict):
                    continue
                ct = c.get("type")
                if ct == "text":
                    text = str(c.get("text") or "").replace("\n", " ")[:200]
                    log.info("claude> %s", text)
                elif ct == "tool_use":
                    name = c.get("name") or "?"
                    args = c.get("input") or {}
                    preview = (
                        args.get("command")
                        or args.get("file_path")
                        or (args.get("pattern") or "")
                    )
                    if not preview and args:
                        preview = str(next(iter(args.values())))
                    log.info("claude tool: %s %s", name, str(preview)[:120])
        elif t == "user":
            message = obj.get("message", {})
            content = message.get("content")
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "tool_result":
                        tr = c.get("content", "")
                        size = len(str(tr))
                        log.info("claude tool result (%d chars)", size)
        elif t == "system":
            log.info("claude system: %s", obj.get("subtype") or "?")
        elif t == "result":
            result = str(obj.get("result") or "").replace("\n", " ")[:200]
            log.info("claude result: %s", result)
        elif t == "error":
            log.warning("claude error: %s", obj.get("error") or obj)

    def iter_events(self) -> Iterator[dict[str, Any]]:
        """Yield parsed stream-json events for the current turn.

        Clears the cancel event at the start so any signal that arrived during
        the lock-handoff window (between the previous holder's :meth:`__exit__`
        and this call) is consumed here rather than immediately aborting the
        new turn.  After that, checks the event on each poll cycle so a
        concurrent :meth:`interrupt` can abort mid-turn.

        Reads lines from stdout, parsing each as JSON.  Stops (and returns)
        when a ``type=result`` or ``type=error`` event is yielded, when the
        process exits (EOF), or when no output arrives for *idle_timeout*
        seconds (raises :class:`ClaudeStreamError` with
        :data:`_RETURNCODE_IDLE_TIMEOUT` in that case).

        Raises ``json.JSONDecodeError`` if a non-empty stdout line cannot be
        parsed — this is a protocol violation from the claude subprocess and
        should not be silently swallowed.
        """
        assert self._proc.stdout is not None
        self._cancel.clear()
        self._last_turn_cancelled = False
        last_activity = time.monotonic()

        while True:
            if self._cancel.is_set():
                log.debug("ClaudeSession: cancelled — exiting turn early")
                self._last_turn_cancelled = True
                # Intentionally leave _in_turn = True: the caller who set
                # _cancel will have the next send() drain the boundary
                # we're abandoning here.
                break
            ready, _, _ = self._selector(
                [self._proc.stdout, self._wakeup_r], [], [], _SELECT_POLL_INTERVAL
            )
            self._drain_wakeup()
            if self._proc.stdout in ready:
                line = self._proc.stdout.readline()
                if not line:
                    self._in_turn = False
                    break  # EOF
                line = line.strip()
                if not line:
                    last_activity = time.monotonic()
                    continue
                obj = json.loads(line)
                self._log_event(obj)
                last_activity = time.monotonic()
                # Track the latest session_id so :meth:`switch_model` can
                # restart with ``--resume <sid>`` and keep conversation
                # context across the swap.
                sid = obj.get("session_id")
                if isinstance(sid, str) and sid:
                    self._session_id = sid
                yield obj
                if obj.get("type") in ("result", "error"):
                    self._in_turn = False
                    break
            elif self._proc.poll() is not None:
                self._in_turn = False
                break  # process exited
            elif time.monotonic() - last_activity > self._idle_timeout:
                log.warning(
                    "ClaudeSession: idle for %.0fs — killing", self._idle_timeout
                )
                self._proc.kill()
                self._proc.wait()
                self.recover()
                raise ClaudeStreamError(_RETURNCODE_IDLE_TIMEOUT)

    def consume_until_result(self) -> str:
        """Drain events for the current turn and return the result text.

        Exhausts :meth:`iter_events` and returns the ``result`` field from
        the ``type=result`` event, or an empty string if the turn ends
        without one (EOF, ``type=error``, or idle-timeout kill).
        """
        result_text = ""
        for event in self.iter_events():
            provider_error = _provider_error_from_event(event)
            if provider_error is not None:
                log.error(
                    "ClaudeSession: provider failure during turn: %s"
                    " (status=%s, request_id=%s)",
                    provider_error,
                    provider_error.status_code,
                    provider_error.request_id or "—",
                )
                self.recover()
                raise provider_error
            if event.get("type") == "result" and isinstance(event.get("result"), str):
                result_text = event["result"]
        return result_text

    def stop(self, grace_seconds: float = 2.0) -> None:
        """Shut down the session: close stdin, wait for exit, kill if needed.

        Always unregisters the process from ``_active_children``, even if the
        process has already exited before :meth:`stop` is called.
        """
        try:
            try:
                if self._proc.stdin and not self._proc.stdin.closed:
                    self._proc.stdin.close()
            except OSError as exc:
                log.debug("ClaudeSession.stop: stdin close failed: %s", exc)
                raise
            try:
                self._proc.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                try:
                    self._proc.kill()
                    self._proc.wait(timeout=1.0)
                except (OSError, ProcessLookupError, subprocess.TimeoutExpired) as exc:
                    log.debug("ClaudeSession.stop: kill/wait failed: %s", exc)
                    raise
            except (OSError, ProcessLookupError) as exc:
                log.debug("ClaudeSession.stop: wait failed: %s", exc)
        finally:
            _unregister_child(self._proc)


# ── Claude provider collaborators ────────────────────────────────────────────


@dataclass(frozen=True)
class _ClaudeOAuthState:
    access_token: str


def _default_claude_credentials_path() -> Path:
    return Path.home() / ".claude" / ".credentials.json"


def _load_claude_oauth_state(
    credentials_path: Path | None = None,
) -> _ClaudeOAuthState | None:
    path = (
        _default_claude_credentials_path()
        if credentials_path is None
        else credentials_path
    )
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    oauth = raw["claudeAiOauth"]
    access_token = oauth["accessToken"]
    if not isinstance(access_token, str) or not access_token:
        raise ValueError("Claude OAuth credentials missing accessToken")
    return _ClaudeOAuthState(access_token=access_token)


def _parse_usage_reset(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"Claude usage reset time must be a string or null, got {value!r}"
        )
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _usage_window(name: str, value: object) -> ProviderLimitWindow | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"Claude usage window {name} must be an object or null")
    utilization = value.get("utilization")
    if utilization is None:
        return None
    if not isinstance(utilization, int | float):
        raise ValueError(
            f"Claude usage window {name} utilization must be numeric or null"
        )
    return ProviderLimitWindow(
        name=name,
        used=int(round(float(utilization))),
        limit=100,
        resets_at=_parse_usage_reset(value.get("resets_at")),
        unit="%",
    )


class ClaudeAPI(ProviderAPI):
    """Read-only account API for Claude Code usage and limits."""

    def __init__(
        self,
        *,
        session: _requests.Session | None = None,
        oauth_state_fn: Callable[
            [], _ClaudeOAuthState | None
        ] = _load_claude_oauth_state,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._session = session if session is not None else _requests.Session()
        self._oauth_state_fn = oauth_state_fn
        self._monotonic = monotonic
        self._limit_snapshot_lock = threading.Lock()
        self._limit_snapshot_cached_at: float | None = None
        self._limit_snapshot_cache: ProviderLimitSnapshot | None = None

    @property
    def provider_id(self) -> ProviderID:
        return ProviderID.CLAUDE_CODE

    def get_limit_snapshot(self) -> ProviderLimitSnapshot:
        with self._limit_snapshot_lock:
            if (
                self._limit_snapshot_cache is not None
                and self._limit_snapshot_cached_at is not None
                and self._monotonic() - self._limit_snapshot_cached_at
                < _CLAUDE_USAGE_CACHE_SECONDS
            ):
                return self._limit_snapshot_cache
            oauth_state = self._oauth_state_fn()
            if oauth_state is None:
                snapshot = ProviderLimitSnapshot(
                    provider=self.provider_id,
                    unavailable_reason="Claude Code is not logged in.",
                )
            else:
                try:
                    response = self._session.get(
                        _CLAUDE_USAGE_URL,
                        headers={
                            "Authorization": f"Bearer {oauth_state.access_token}",
                            "anthropic-beta": _CLAUDE_USAGE_BETA,
                            "Content-Type": "application/json",
                            "User-Agent": _CLAUDE_USAGE_USER_AGENT,
                        },
                        timeout=_CLAUDE_API_TIMEOUT,
                    )
                    response.raise_for_status()
                    payload = response.json()
                    if not isinstance(payload, dict):
                        raise ValueError("Claude usage response must be a JSON object")
                    windows = tuple(
                        window
                        for window in (
                            _usage_window("five_hour", payload.get("five_hour")),
                            _usage_window("seven_day", payload.get("seven_day")),
                            _usage_window(
                                "seven_day_sonnet", payload.get("seven_day_sonnet")
                            ),
                        )
                        if window is not None
                    )
                    if windows:
                        snapshot = ProviderLimitSnapshot(
                            provider=self.provider_id, windows=windows
                        )
                    else:
                        snapshot = ProviderLimitSnapshot(
                            provider=self.provider_id,
                            unavailable_reason=(
                                "Claude usage is only available for subscription plans."
                            ),
                        )
                except Exception as exc:
                    log.exception("ClaudeAPI: failed to fetch usage snapshot")
                    snapshot = ProviderLimitSnapshot(
                        provider=self.provider_id,
                        unavailable_reason=f"Claude usage unavailable: {exc}",
                    )
            self._limit_snapshot_cache = snapshot
            self._limit_snapshot_cached_at = self._monotonic()
            return snapshot


class ClaudeClient(SessionBackedAgent, ProviderAgent):
    """Injectable collaborator for one-shot Claude CLI interactions.

    Wraps subprocess-based, session-based, and streaming Claude helpers
    so callers can depend on an explicit boundary instead of module-level
    functions.  Constructor-injected dependencies replace the per-call
    ``runner=`` / ``streaming_runner=`` overrides used by the free functions.

    The class does not own the persistent :class:`ClaudeSession` — it
    receives a *session_fn* callback that resolves the session for the
    calling context (thread-local repo in production, a test fake in
    tests).
    """

    voice_model = ProviderModel("claude-opus-4-6")
    work_model = ProviderModel("claude-sonnet-4-6")
    brief_model = ProviderModel("claude-haiku-4-5-20251001")

    def __init__(
        self,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        session_fn: Callable[[], PromptSession] = current_repo_session,
        streaming_runner: Callable[..., Iterator[str]] = _run_streaming,
        sleep_fn: Callable[[float], None] = time.sleep,
        session_factory: Callable[..., PromptSession] | None = None,
        session_system_file: Path | None = None,
        work_dir: Path | str | None = None,
        repo_name: str | None = None,
        session: PromptSession | None = None,
    ) -> None:
        self._runner = runner
        self._streaming_runner = streaming_runner
        self._sleep_fn = sleep_fn
        self._session_factory = (
            ClaudeSession if session_factory is None else session_factory
        )
        super().__init__(
            session_fn=session_fn,
            session_system_file=session_system_file,
            work_dir=work_dir,
            repo_name=repo_name,
            session=session,
        )

    @property
    def provider_id(self) -> ProviderID:
        return ProviderID.CLAUDE_CODE

    def _spawn_owned_session(self, model: ProviderModel | None = None) -> PromptSession:
        system_file = self._session_system_file
        work_dir = self._work_dir
        assert system_file is not None
        assert work_dir is not None
        return self._session_factory(
            system_file,
            work_dir=work_dir,
            repo_name=self._repo_name,
            model=model,
        )

    def ensure_session(self, model: ProviderModel | None = None) -> None:
        created = False
        with self._session_lock:
            session = self._session
            if session is None:
                if self._session_system_file is None or self._work_dir is None:
                    raise ValueError(
                        "ClaudeClient.ensure_session requires session_system_file and work_dir"
                    )
                if model is None:
                    raise ValueError(
                        "ClaudeClient.ensure_session requires model when creating a session"
                    )
                session = self._spawn_owned_session(model)
                self._session = session
                created = True
        if model is None or (created and model == self.voice_model):
            return
        session.switch_model(model)

    def _prompt_failure_is_passthrough(self, exc: Exception) -> bool:
        return isinstance(exc, ClaudeProviderError)

    def _should_retry_prompt_failure(
        self,
        exc: Exception,
        session: PromptSession,
    ) -> bool:
        return isinstance(exc, (ClaudeStreamError, BrokenPipeError, OSError)) or (
            self._session_is_dead(session)
        )

    def _dead_prompt_error_message(self) -> str:
        return "Claude session died during prompt"

    def _json_parse_candidates(self, raw: str) -> tuple[str, ...]:
        return (raw, *(m.group() for m in re.finditer(r"\{.*?\}", raw, re.DOTALL)))

    def run_turn(
        self,
        content: str,
        *,
        model: ProviderModel | None = None,
        system_prompt: str | None = None,
        retry_on_preempt: bool = False,
        session_mode: TurnSessionMode = TurnSessionMode.REUSE,
    ) -> str:
        session = self._resolve_turn_session(
            model=model,
            session_mode=session_mode,
        )
        attempt = 0
        while True:
            result = self._prompt_with_recovery(
                session,
                content,
                model=model,
                system_prompt=system_prompt,
            )
            if (
                not retry_on_preempt
                or getattr(session, "last_turn_cancelled", False) is not True
            ):
                return result
            session.wait_for_pending_preempt()
            attempt += 1
            log.info("ClaudeClient.run_turn: preempted mid-flight — retry %d", attempt)

    def generate_status(
        self,
        prompt: str,
        system_prompt: str,
        model: ProviderModel | None = None,
    ) -> str:
        """Ask claude to generate a GitHub status (two lines: emoji + text)."""
        return self.run_turn(
            prompt,
            model=self.voice_model if model is None else model,
            system_prompt=system_prompt,
        )

    def generate_status_emoji(
        self,
        prompt: str,
        system_prompt: str,
        model: ProviderModel | None = None,
    ) -> str:
        """Ask claude to choose a single emoji for a GitHub status."""
        return self._run_turn_json_value(
            prompt,
            "emoji",
            self.voice_model if model is None else model,
            system_prompt=system_prompt,
        )

    # ── Streaming file-based helpers ─────────────────────────────────────

    def print_prompt_from_file(
        self,
        system_file: Path,
        prompt_file: Path,
        model: ProviderModel,
        timeout: int = 30,
        idle_timeout: float = 1800.0,
        cwd: Path | str = ".",
    ) -> str:
        """Run claude --print reading system prompt and user prompt from files."""
        cmd = [
            "claude",
            "--model",
            model_name(model),
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
            "--system-prompt-file",
            str(system_file),
            "--print",
        ]
        output = "".join(
            self._streaming_runner(cmd, prompt_file, idle_timeout, cwd=cwd)
        ).strip()
        raise_for_provider_error_output(output)
        return output

    def resume_session(
        self,
        session_id: str,
        prompt_file: Path,
        model: ProviderModel,
        timeout: int = 300,
        idle_timeout: float = 1800.0,
        cwd: Path | str = ".",
    ) -> str:
        """Continue an existing claude session by ID, feeding prompt_file on stdin."""
        cmd = [
            "claude",
            "--model",
            model_name(model),
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
            "--resume",
            session_id,
            "--print",
        ]
        output = "".join(
            self._streaming_runner(cmd, prompt_file, idle_timeout, cwd=cwd)
        ).strip()
        raise_for_provider_error_output(output)
        return output

    def extract_session_id(self, output: str) -> str:
        return extract_session_id(output)


class ClaudeCode(Provider):
    """Composite Claude provider with separate account API and runtime agent."""

    def __init__(
        self,
        *,
        api: ProviderAPI | None = None,
        agent: ProviderAgent | None = None,
        session: PromptSession | None = None,
    ) -> None:
        if agent is None:
            agent = ClaudeClient(session=session)
        elif session is not None:
            agent.attach_session(session)
        self._api = ClaudeAPI() if api is None else api
        self._agent = agent

    @property
    def provider_id(self) -> ProviderID:
        return ProviderID.CLAUDE_CODE

    @property
    def api(self) -> ProviderAPI:
        return self._api

    @property
    def agent(self) -> ProviderAgent:
        return self._agent
