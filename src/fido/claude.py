"""Claude CLI wrappers — all claude subprocess calls in one place."""

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
from datetime import datetime
from pathlib import Path
from typing import Any

import requests as _requests

from fido import provider
from fido.idle_timeout import IdleDeadline
from fido.provider import (
    GLOBAL_DISALLOWED_TOOLS,
    READ_ONLY_ALLOWED_TOOLS,
    OwnedSession,
    PromptSession,
    Provider,
    ProviderAgent,
    ProviderAPI,
    ProviderID,
    ProviderLimitSnapshot,
    ProviderLimitWindow,
    ProviderModel,
    model_name,
)
from fido.rocq import claude_session as stream_fsm
from fido.session_agent import SessionBackedAgent

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


# Tracked long-running claude subprocesses (the streaming ones), so fido
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


def _thread_name_for_id(thread_id: int) -> str | None:
    """Return the human-readable name of the live thread with *thread_id*,
    or ``None`` if that thread has exited.

    Used by status display to render a :class:`provider.SessionTalker`'s thread
    without caching the name in the registry — a dead thread's entry is
    already being cleaned up and the name would be stale.
    """
    for t in threading.enumerate():
        if t.ident == thread_id:
            return t.name
    return None


def kill_active_children(grace_seconds: float = 2.0) -> None:
    """Send SIGTERM to every tracked claude subprocess, then SIGKILL stragglers.

    Called from fido's shutdown signal handler so children don't outlive
    the parent and reparent to PID 1.  Safe to call multiple times.
    """
    with _active_children_lock:
        children = list(_active_children)
    if not children:
        return
    log.info("fido shutdown: terminating %d claude child(ren)", len(children))
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
    call is localisable in the fido log.
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

_RETURNCODE_CANCEL_DRAIN_TIMEOUT = -2
# How long ``iter_events`` waits, after the cancel signal arrives, for
# claude to emit the ``type=result`` event that closes the cancelled turn.
# Claude normally responds to ``control_request interrupt`` in <1s, so 5s
# is generous; if it elapses the subprocess is wedged and must be recovered.
_CANCEL_DRAIN_TIMEOUT = 5.0


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
    repo_name = provider.current_repo()
    thread_id = threading.get_ident()
    talker_registered = False
    if repo_name is not None:
        provider.register_talker(
            provider.SessionTalker(
                repo_name=repo_name,
                thread_id=thread_id,
                kind="webhook",
                description=f"one-shot claude --print (pid {proc.pid})",
                claude_pid=proc.pid,
                started_at=provider.talker_now(),
            )
        )
        talker_registered = True

    try:
        idle_deadline = IdleDeadline(
            idle_timeout,
            poll_interval=_SELECT_POLL_INTERVAL,
            clock=clock,
        )

        while True:
            poll_timeout = idle_deadline.poll_timeout_or_expired()
            if poll_timeout is None:
                log.warning("claude idle for %.0fs — killing", idle_timeout)
                proc.kill()
                proc.wait()
                raise ClaudeStreamError(_RETURNCODE_IDLE_TIMEOUT)
            ready, _, _ = selector([proc.stdout], [], [], poll_timeout)
            if ready:
                line = proc.stdout.readline()
                if not line:
                    break  # EOF
                yield line
                log.debug(line.rstrip())
                idle_deadline.reset()
            elif proc.poll() is not None:
                break  # process exited

        proc.wait()
        # Drain any remaining output
        remaining = proc.stdout.read()
        if remaining:
            yield remaining
        if proc.returncode != 0:
            raise ClaudeStreamError(proc.returncode)
    finally:
        if talker_registered and repo_name is not None:
            provider.unregister_talker(repo_name, thread_id)
        _unregister_child(proc)


# ── Persistent bidirectional session ─────────────────────────────────────────


class ClaudeSession(OwnedSession):
    """A long-lived claude process driven via bidirectional stream-json.

    Spawns ``claude --input-format stream-json --output-format stream-json``
    and keeps it running across multiple turns.  Each :meth:`send` writes one
    JSON user message to stdin; :meth:`iter_events` reads structured events
    from stdout until the turn completes (``type=result`` or ``type=error``).

    **Lifetime / persistence model**

    The session outlives individual :class:`~fido.worker.Worker` crashes:
    :class:`~fido.worker.WorkerThread` holds the session in
    ``_session`` across iterations and passes it into each new ``Worker``
    instance, so an unexpected exception in ``Worker.run()`` does not tear the
    session down.  The watchdog restarts the thread and the next Worker
    inherits the same session.

    The session does *not* survive a full fido restart.  When fido
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
        session_id: str | None = None,
        tools: str | None = None,
    ) -> None:
        self._idle_timeout = idle_timeout
        self._selector = selector
        self._system_file = system_file
        self._work_dir = work_dir
        self._popen_fn = popen
        self._cancel = threading.Event()
        self._repo_name = repo_name
        self._model = model_name(
            ProviderModel("claude-opus-4-6") if model is None else model
        )
        # Allowed-tools restriction passed as ``--allowedTools`` to the
        # subprocess.  ``None`` = no restriction (worker mode, default);
        # any string = triage allowlist (handler mode, typically
        # :data:`READ_ONLY_ALLOWED_TOOLS`).  Changed via :meth:`switch_tools`,
        # which respawns the subprocess while keeping ``--resume`` so
        # conversation context survives the mode transition (#1042).
        self._tools: str | None = tools
        # Latest session_id seen in a stream-json event.  Updated inside
        # :meth:`iter_events` so :meth:`recover`, :meth:`reset`, and
        # :meth:`switch_model` can pass ``--resume <sid>``
        # to :meth:`_spawn` and keep conversation context across a
        # subprocess restart.  Seeded from the *session_id* constructor
        # kwarg so the first :meth:`_spawn` can ``--resume`` a durable
        # claude conversation persisted across fido restarts (#649).
        # Empty until the first claude event with a session_id arrives.
        self._session_id = session_id or ""
        # Stream-protocol FSM state — replaces the ad-hoc ``_in_turn`` and
        # ``_last_turn_cancelled`` flags.  See ``models/claude_session.v``
        # for the formal transition table.  In short: ``Idle`` between
        # turns, ``Sending`` after :meth:`send` until the first reply
        # event, ``AwaitingReply`` while events flow, ``Draining`` after a
        # cancel until the boundary, and ``Cancelled`` once a cancelled
        # turn has closed cleanly (consumed by the next :meth:`send`).
        self._stream_lock = threading.Lock()
        self._stream_state: stream_fsm.State = stream_fsm.Idle()
        # Per-thread reentrance counter for the ``with self:`` context so
        # :meth:`hold_for_handler` can nest inner :meth:`prompt` calls
        # without double-registering the talker (fix for #658).
        self._init_handler_reentry()
        # Wakeup pipe: writing a byte to _wakeup_w kicks select() out of its
        # blocking wait in iter_events() so the cancel signal is noticed
        # immediately instead of waiting up to _SELECT_POLL_INTERVAL.
        self._wakeup_r, self._wakeup_w = os.pipe()
        os.set_blocking(self._wakeup_r, False)
        os.set_blocking(self._wakeup_w, False)
        # Cumulative message counters — they accumulate since boot and are NOT
        # reset on :meth:`_respawn`, so per-subprocess resets from model
        # switches or recoveries don't erase the history.
        # _metrics_lock guards both counters: they are written by the worker
        # thread (send / iter_events) and read from other threads (status,
        # registry).  Python 3.14t has no GIL, so += is not atomic.
        self._metrics_lock = threading.Lock()
        self._sent_count: int = 0
        self._received_count: int = 0
        self._proc = self._spawn()
        _register_child(self._proc)

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
        """Repo this session belongs to, for :class:`provider.SessionTalker` registration."""
        return self._repo_name

    def _stream_transition(self, event: stream_fsm.Event) -> stream_fsm.State:
        """Fire *event* on the stream-protocol FSM, raising ``AssertionError``
        if the transition is rejected by the formal model.

        Single oracle for every FSM transition in :class:`ClaudeSession`,
        so a coordination bug surfaces as a crash rather than as silent
        protocol drift.
        """
        with self._stream_lock:
            prev = self._stream_state
            new_state = stream_fsm.transition(prev, event)
            if new_state is None:
                raise AssertionError(
                    f"claude_session FSM: {type(event).__name__} rejected in "
                    f"state {type(prev).__name__}"
                )
            self._stream_state = new_state
            log.debug(
                "ClaudeSession[%s]: stream %s →%s via %s",
                self._repo_name or "?",
                type(prev).__name__,
                type(new_state).__name__,
                type(event).__name__,
            )
            return new_state

    def _stream_reset(self) -> None:
        """Reset the stream FSM directly to ``Idle``.

        Only legal on crash/kill/respawn paths — the formal model has no
        edge that transitions arbitrary states back to ``Idle`` because
        that would mask protocol bugs.  The respawn itself is the
        invariant-restoring event: the subprocess is gone, so any prior
        in-flight turn is moot.
        """
        with self._stream_lock:
            self._stream_state = stream_fsm.Idle()

    @property
    def last_turn_cancelled(self) -> bool:
        """``True`` when the most recent :meth:`iter_events` call exited
        early because another thread set the cancel event (preempted the
        turn via :meth:`prompt` or :meth:`interrupt`).

        Backed by the stream FSM: a turn that observed ``CancelFire`` and
        reached its ``TurnReturn`` ends in ``Cancelled``; a normal turn
        ends in ``Idle``.  The next :meth:`send` consumes the ``Cancelled``
        state by firing ``TurnReturn`` to return to ``Idle``.

        Callers that want resumption semantics can check this after a turn
        and re-send the same content once the session lock is free again —
        effectively 'hand the session back to the worker and ask it to
        resume what it was doing'.
        """
        with self._stream_lock:
            return isinstance(self._stream_state, stream_fsm.Cancelled)

    def _spawn(self) -> subprocess.Popen[str]:
        """Spawn the claude subprocess with bidirectional stream-json I/O.

        Model is set via ``--model`` at spawn time.  When
        :attr:`_session_id` is non-empty the new process resumes the
        prior conversation via ``--resume`` so context survives a
        subprocess restart (e.g. :meth:`recover` or :meth:`reset`).
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
        if self._tools is not None:
            cmd += ["--allowedTools", self._tools]
        # GLOBAL_DISALLOWED_TOOLS applies to every spawn regardless of
        # which phase's allowlist is active — these are tools no phase can
        # legitimately use (harness owns commit/push, bypass-prone Agent /
        # Skill / Task* family always denied).
        cmd += ["--disallowedTools", GLOBAL_DISALLOWED_TOOLS]
        if self._session_id:
            cmd += ["--resume", self._session_id]
        proc = self._popen_fn(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self._work_dir,
        )
        self._start_stderr_pump(proc)
        return proc

    def _start_stderr_pump(self, proc: subprocess.Popen[str]) -> None:
        """Drain the subprocess's stderr into the logger.

        Without this drain, ``stderr=subprocess.PIPE`` leaves the pipe buffer
        unread.  Once the buffer fills, the claude subprocess blocks on its
        next stderr write and eventually deadlocks — symptom: the first
        ``set_status`` prompt finds the session dead with no diagnostic.

        Runs as a daemon thread so it terminates naturally when the
        subprocess exits (stderr EOF) or when the process shuts down.
        """
        pid = proc.pid
        stderr = proc.stderr
        if stderr is None:  # pragma: no cover — Popen with PIPE always sets this
            return

        def pump() -> None:
            try:
                for raw in stderr:
                    line = raw.rstrip()
                    if line:
                        log.info("ClaudeSession[pid=%d] stderr: %s", pid, line)
            except OSError, ValueError:
                # ValueError on closed file; OSError on broken pipe.
                # Both mean the subprocess is gone — stop pumping.
                pass

        threading.Thread(
            target=pump,
            name=f"claude-stderr-pump-{pid}",
            daemon=True,
        ).start()

    def is_alive(self) -> bool:
        """Return True if the claude subprocess is still running."""
        return self._proc.poll() is None

    @property
    def pid(self) -> int:
        """PID of the live claude subprocess.

        Read directly off the tracked ``Popen`` — callers should use this
        rather than pgrep, since :class:`ClaudeSession` uses
        ``sub/persona.md`` (outside any ``fido_dir``) as its system prompt
        and the pgrep-based heuristic in :mod:`fido.status` can't find it.
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

    @property
    def sent_count(self) -> int:
        """Cumulative number of user messages sent to claude since boot.

        Accumulates across subprocess respawns — model switches and recoveries
        do not reset the count.
        """
        with self._metrics_lock:
            return self._sent_count

    @property
    def received_count(self) -> int:
        """Cumulative number of stream-json events received from claude since boot.

        Accumulates across subprocess respawns — model switches and recoveries
        do not reset the count.
        """
        with self._metrics_lock:
            return self._received_count

    def _respawn(self, *, clear_session_id: bool, reason: str) -> None:
        """Stop the current subprocess and spawn a replacement."""
        log.info("ClaudeSession: %s (model=%s)", reason, self._model)
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
        # Fresh subprocess has no in-flight turn — reset the stream FSM
        # directly to ``Idle`` so the next send() starts a new turn from a
        # clean slate.
        self._stream_reset()
        # Message counters are cumulative since boot — do NOT reset on
        # respawn.  Per-subprocess counts would bounce to zero on every
        # model switch or recovery, making wedge detection meaningless.
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

        Derived from the global :class:`provider.SessionTalker` registry so reads are
        always serialized through ``_talkers_lock`` — correct under the
        free-threaded (3.14t) runtime without relying on the GIL.  Returns
        ``None`` for sessions without a ``repo_name`` (unit-test fixtures)
        and when the active talker is a webhook rather than the worker.
        """
        if self._repo_name is None:
            return None
        talker = provider.get_talker(self._repo_name)
        if talker is None or talker.kind != "worker":
            return None
        return _thread_name_for_id(talker.thread_id)

    def __enter__(self) -> "ClaudeSession":
        """Acquire the session lock, serializing send/receive across threads.

        On the outermost entry (via the :class:`OwnedSession`
        reentrance counter), delegates to :meth:`_fsm_acquire_worker` or
        :meth:`_fsm_acquire_handler` based on the calling thread's kind
        (read from :func:`provider.current_thread_kind` — never via any
        session-shared attribute).  Handler acquires queue behind any
        current holder and are served FIFO; worker acquires yield to any
        queued handler.

        Registers a :class:`provider.SessionTalker` after acquiring.
        Nested entries (from :meth:`hold_for_handler`) re-enter the
        reentrance counter and skip the FSM acquire.

        Does *not* clear the cancel event — that is deferred to
        :meth:`iter_events` so a signal that lands between one holder's
        :meth:`__exit__` and the next holder's :meth:`iter_events` is not
        silently dropped.

        Raises :class:`provider.SessionLeakError` on the outermost entry if another
        thread is already registered as the talker for this repo.  The
        FSM lock is released before raising so the prior holder isn't
        deadlocked.
        """
        depth = getattr(self._reentry_tls, "depth", 0)
        if depth > 0:
            self._bump_entry_depth()
            return self
        kind = provider.current_thread_kind()
        if kind == "worker":
            self._fsm_acquire_worker()
        else:
            # preempt-always: a webhook entering while a worker holds the
            # session fires the cancel before queueing on the FSM, so the
            # worker's turn aborts and releases promptly rather than being
            # waited out (#637).  Webhook-on-webhook still queues FIFO with
            # no cancel.
            provider.try_preempt_worker(self._repo_name, self._fire_worker_cancel)
            self._fsm_acquire_handler()
        self._bump_entry_depth()
        if self._repo_name is not None:
            try:
                provider.register_talker(
                    provider.SessionTalker(
                        repo_name=self._repo_name,
                        thread_id=threading.get_ident(),
                        kind=kind,
                        description="persistent session turn",
                        claude_pid=self._proc.pid,
                        started_at=provider.talker_now(),
                    )
                )
            except provider.SessionLeakError:
                self._drop_entry_depth()
                self._fsm_release()
                raise
        return self

    def __exit__(self, *args: object) -> None:
        """Release the session lock.  Unregisters the :class:`provider.SessionTalker`
        before releasing so no other thread can race in and see a stale talker entry.
        """
        depth = self._drop_entry_depth()
        if depth == 0:
            if self._repo_name is not None:
                provider.unregister_talker(self._repo_name, threading.get_ident())
            self._fsm_release()

    def send(self, content: str) -> None:
        """Write a user message to the session stdin, flushing immediately.

        Any prior cancelled turn is drained to its ``type=result`` boundary
        inside :meth:`iter_events` itself (cancel no longer breaks early —
        see #979 / #955 cascade).  By the time :meth:`send` is called the
        previous turn is closed and the stream is clean, so this method
        only needs to atomically place a single user message on stdin.

        FSM: if the prior turn ended ``Cancelled``, fire ``TurnReturn`` to
        return to ``Idle`` (acknowledging the cancellation), then fire
        ``Send`` to enter ``Sending`` for the new turn.
        """
        # Acknowledge a prior cancelled turn: Cancelled → Idle.
        with self._stream_lock:
            if isinstance(self._stream_state, stream_fsm.Cancelled):
                new_state = stream_fsm.transition(
                    self._stream_state, stream_fsm.TurnReturn()
                )
                assert new_state is not None
                self._stream_state = new_state
        # Idle → Sending.
        self._stream_transition(stream_fsm.Send())
        msg = json.dumps(
            {"type": "user", "message": {"role": "user", "content": content}}
        )
        assert self._proc.stdin is not None
        self._proc.stdin.write(msg + "\n")
        self._proc.stdin.flush()
        with self._metrics_lock:
            self._sent_count += 1

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
            self._stream_reset()
            return
        try:
            self._send_control_interrupt()
        except (BrokenPipeError, OSError) as exc:
            log.warning(
                "ClaudeSession._drain_to_boundary: control_request failed: %s", exc
            )
            self._stream_reset()
            return
        # Calling _drain_to_boundary means "we're cancelling the current
        # turn".  Fire CancelFire on the FSM only when the current state
        # accepts it (Sending or AwaitingReply); other states (e.g. Idle
        # in test scaffolding) leave the FSM untouched.
        with self._stream_lock:
            if isinstance(
                self._stream_state, stream_fsm.Sending | stream_fsm.AwaitingReply
            ):
                new_state = stream_fsm.transition(
                    self._stream_state, stream_fsm.CancelFire()
                )
                assert new_state is not None
                self._stream_state = new_state
        end_time = time.monotonic() + deadline
        while time.monotonic() < end_time:
            ready, _, _ = self._selector(
                [self._proc.stdout, self._wakeup_r], [], [], _SELECT_POLL_INTERVAL
            )
            self._drain_wakeup()
            if self._cancel.is_set():
                # Preempt fired — exit early without changing FSM state so
                # send() skips writing and iter_events() breaks on the cancel.
                log.debug(
                    "ClaudeSession._drain_to_boundary: cancel set — aborting drain early"
                )
                return
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
                    self._stream_reset()
                    return
            elif self._proc.poll() is not None:
                break
        log.warning(
            "ClaudeSession._drain_to_boundary: no boundary after %.1fs — restarting",
            deadline,
        )
        self._stream_reset()
        self.recover()

    def _fire_worker_cancel(self) -> None:
        """Signal the worker thread to stop its in-flight turn — but do NOT
        write to claude's stdin from this (webhook) thread.

        Single-owner invariant: only the lock-holder writes to claude's
        stdin.  The webhook thread is firing the cancel from outside the
        lock; if it wrote the ``control_request interrupt`` directly, the
        bytes could interleave with the worker's mid-turn writes (#979),
        and — more importantly — claude's protocol state would diverge
        from fido's because the side-channel write isn't part of the
        worker's turn-handshake.

        Instead, set :attr:`_cancel` and wake the worker via the wakeup
        pipe.  The worker, inside :meth:`iter_events`, observes the
        cancel signal, sends ``control_request interrupt`` itself
        (writing under its own ownership of the lock), drains stdout
        until the ``type=result`` boundary that closes the cancelled
        turn, then exits the turn with ``_in_turn=False``.  Only then
        does the worker release the lock — handing the webhook a fully
        settled session.
        """
        self._cancel.set()
        self._wake()

    def _on_force_release(self, *, reason: str) -> None:
        """Knock a wedged holder out of its parked IO call by killing the
        subprocess.

        :meth:`OwnedSession.force_release` has already fired the
        ``ForceRelease`` FSM event and (if any handler is queued)
        transferred ownership.  All that is left is to escape the
        wedged thread from its blocking call inside
        :meth:`consume_until_result`: closing the subprocess's stdout
        makes the parked ``select()`` return ready with EOF, the
        ``iter_events`` loop hits the EOF branch and returns,
        :meth:`prompt`'s ``with self:`` runs ``__exit__``, and the
        ``_evicted_tids`` guard in :meth:`OwnedSession._fsm_release`
        skips the now-incorrect ``WorkerRelease`` / ``HandlerRelease``.

        The kill is best-effort: if the subprocess has already exited
        (race with idle timeout, normal shutdown, or another recovery
        path), :meth:`subprocess.Popen.kill` raises
        :class:`ProcessLookupError` which is caught and ignored.
        """
        log.warning(
            "ClaudeSession[%s]: force_release killing pid=%s (reason=%r)",
            self._repo_name or "?",
            self._proc.pid,
            reason,
        )
        try:
            self._proc.kill()
        except (ProcessLookupError, OSError) as exc:
            log.debug(
                "ClaudeSession[%s]: force_release kill: %s",
                self._repo_name or "?",
                exc,
            )
        # Reset the stream FSM so the next acquire's first turn does not
        # inherit a stale Sending/AwaitingReply state from the killed
        # subprocess.  Mirrors the cleanup in :meth:`_drain_to_boundary`
        # after a cancel-drain timeout.
        self._stream_reset()

    def _send_control_interrupt(self) -> str:
        """Write a stream-json ``control_request`` interrupt to subprocess
        stdin and return the request_id so the caller can assert the
        matching ``control_response`` arrives.

        Single-owner invariant: only the lock-holder calls this.  Webhook
        threads firing a preempt go through :meth:`_fire_worker_cancel`,
        which sets a flag and wakes the worker — the worker (the actual
        lock-holder) is the one that actually writes the interrupt to
        stdin from inside :meth:`iter_events`.  This keeps claude's
        protocol state consistent with fido's and avoids cross-thread
        stdin writes entirely (#979).
        """
        request_id = str(uuid.uuid4())
        msg = json.dumps(
            {
                "type": "control_request",
                "request_id": request_id,
                "request": {"subtype": "interrupt"},
            }
        )
        assert self._proc.stdin is not None
        self._proc.stdin.write(msg + "\n")
        self._proc.stdin.flush()
        return request_id

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
        with self:
            self._send_control_interrupt()
            self.consume_until_result()
            self.send(content)

    def prompt(
        self,
        content: str,
        *,
        model: ProviderModel | None = None,
        allowed_tools: str | None = READ_ONLY_ALLOWED_TOOLS,
        system_prompt: str | None = None,
    ) -> str:
        """Send *content* as a user message on the persistent session and
        return the result.

        Acquires the session lock and runs one turn: optional
        :meth:`switch_model`, :meth:`switch_tools`, :meth:`send` (which
        drains any lingering boundary events from a prior aborted turn), and
        :meth:`consume_until_result`.

        ``allowed_tools`` defaults to :data:`READ_ONLY_ALLOWED_TOOLS` (closes
        #1413) — the safe shape for synthesis / rescope / setup / status /
        voice rewrite / reply drafting.  Task implementation passes ``None``
        explicitly to permit edits and arbitrary Bash.
        :data:`GLOBAL_DISALLOWED_TOOLS` is applied unconditionally on top.

        Preemption (cancelling a running worker turn so a webhook handler can
        acquire the lock promptly) is handled upstream: the HTTP handler fires
        :meth:`~fido.provider.OwnedSession.preempt_worker` synchronously
        (#955), and :meth:`~fido.provider.OwnedSession.hold_for_handler` fires
        a second preemption as a safety net before acquiring the lock (#658).
        There is no preemption logic here.
        """
        tid = threading.get_ident()
        t_start = time.monotonic()
        with self:
            log.info(
                "session.prompt: lock acquired (tid=%d, waited=%.2fs)",
                tid,
                time.monotonic() - t_start,
            )
            if model is not None:
                self.switch_model(model)
            self.switch_tools(allowed_tools)
            if system_prompt:
                body = f"{system_prompt}\n\n---\n\n{content}"
            else:
                body = content
            self.send(body)
            result = self.consume_until_result()
            log.info(
                "session.prompt: turn complete (tid=%d, total=%.2fs, "
                "result_len=%d, cancelled=%s)",
                tid,
                time.monotonic() - t_start,
                len(result or ""),
                self.last_turn_cancelled,
            )
            return result

    def switch_model(self, model: ProviderModel | str) -> None:
        """Switch the active model by respawning the claude subprocess
        with ``--model <new> --resume <session_id>``.

        The in-place ``control_request`` ``set_model`` path was removed
        because claude-code 2.1.114 wedges its set_model handler after
        the first turn: any second real model switch on the same
        subprocess never receives its ``control_response`` and hangs
        until the idle timeout.  Probes against the raw claude binary
        confirmed the wedge is independent of fido's protocol handling.

        Respawn-with-resume preserves conversation context (claude
        re-reads the session transcript via ``--resume``) at the cost of
        one claude boot per switch (~1.4s).  Must be called between
        turns (the stream FSM must be ``Idle`` or ``Cancelled``);
        :meth:`_respawn` resets the FSM defensively.

        No-op when *model* equals the current model.
        """
        target_model = model_name(model)
        if target_model == self._model:
            return
        log.info(
            "switch_model: %s → %s (respawn-with-resume)",
            self._model,
            target_model,
        )
        self._model = target_model
        self._respawn(
            clear_session_id=False,
            reason=f"switching model to {target_model}",
        )
        log.info("switch_model: now on model=%s", target_model)

    def switch_tools(self, tools: str | None) -> None:
        """Restrict or restore available tools by respawning with a different
        ``--allowedTools`` value.

        When *tools* is a non-None string (typically
        :data:`READ_ONLY_ALLOWED_TOOLS`), the subprocess is spawned with
        ``--allowedTools <value>`` so only triage tools are available —
        this is the handler mode that enforces the invariant from #1042:
        webhook handlers may inspect the codebase and perform triage actions
        but must not perform implementation work.  When *tools* is ``None``, the subprocess uses
        the full tool set (no ``--allowedTools`` flag) — this is the normal
        worker mode.

        A change triggers a subprocess respawn with ``--resume`` so
        conversation context is preserved across the mode boundary.  No-op
        when *tools* already matches the current value.
        """
        if tools == self._tools:
            return
        log.info(
            "switch_tools: %r → %r (respawn-with-resume)",
            self._tools,
            tools,
        )
        self._tools = tools
        self._respawn(
            clear_session_id=False,
            reason=f"switching tools to {tools!r}",
        )
        log.info("switch_tools: complete, tools=%r", self._tools)

    def _log_event(self, obj: dict[str, Any]) -> None:
        """Emit a human-readable INFO log line for a stream-json *obj*.

        Makes stalls pinpointable to a specific tool call or turn rather
        than leaving a silent gap in the fido log between "preempt
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
        # Clear the cancel event at the start of each turn.  Any cancel signal
        # that was set before this holder acquired the lock was meant for the
        # previous holder — the FSM's FIFO handler queue ensures the next
        # holder's turn starts clean without needing a separate pending flag.
        self._cancel.clear()
        idle_deadline = IdleDeadline(
            self._idle_timeout,
            poll_interval=_SELECT_POLL_INTERVAL,
        )
        cancelled_at: float | None = None
        cancel_request_id: str | None = None
        cancel_ack_seen = False

        while True:
            # When the cancel signal arrives (set by the webhook thread via
            # _fire_worker_cancel + _wake), THIS thread — the lock-holder,
            # the only thread allowed to write to claude's stdin — sends
            # the ``control_request interrupt`` immediately so claude
            # aborts the running tool right now (real preemption, not
            # next-turn-boundary).  We then keep reading stdout so the
            # corresponding ``control_response`` and ``type=result`` for
            # the cancelled turn are consumed inside this iter_events call,
            # leaving claude's protocol state and fido's state aligned and
            # the next holder a clean session.  This single-owner invariant
            # is what fixes the cross-thread write race (#979) without
            # needing a separate stdin lock.
            if self._cancel.is_set() and cancelled_at is None:
                log.debug(
                    "ClaudeSession: cancel signal seen — sending interrupt and "
                    "draining to boundary"
                )
                cancelled_at = time.monotonic()
                # Fire CancelFire on the FSM only when the current state
                # accepts it (Sending or AwaitingReply, i.e. an actual
                # in-flight turn).  When iter_events is invoked outside a
                # send-driven turn (test scaffolding, recovery paths) the
                # FSM stays put.
                with self._stream_lock:
                    in_turn = isinstance(
                        self._stream_state,
                        stream_fsm.Sending | stream_fsm.AwaitingReply,
                    )
                    if in_turn:
                        new_state = stream_fsm.transition(
                            self._stream_state, stream_fsm.CancelFire()
                        )
                        assert new_state is not None
                        self._stream_state = new_state
                if in_turn:
                    try:
                        cancel_request_id = self._send_control_interrupt()
                    except (BrokenPipeError, OSError) as exc:
                        log.warning(
                            "ClaudeSession: cancel interrupt write failed: %s", exc
                        )
                        cancel_request_id = None  # unable to assert ack
            if (
                cancelled_at is not None
                and time.monotonic() - cancelled_at > _CANCEL_DRAIN_TIMEOUT
            ):
                log.warning(
                    "ClaudeSession: no type=result %.1fs after cancel — recovering",
                    _CANCEL_DRAIN_TIMEOUT,
                )
                self._proc.kill()
                self._proc.wait()
                self._stream_reset()
                self.recover()
                raise ClaudeStreamError(_RETURNCODE_CANCEL_DRAIN_TIMEOUT)
            poll_timeout = idle_deadline.poll_timeout_or_expired()
            if poll_timeout is None:
                log.warning(
                    "ClaudeSession: idle for %.0fs — killing", self._idle_timeout
                )
                self._proc.kill()
                self._proc.wait()
                self.recover()
                raise ClaudeStreamError(_RETURNCODE_IDLE_TIMEOUT)
            ready, _, _ = self._selector(
                [self._proc.stdout, self._wakeup_r], [], [], poll_timeout
            )
            self._drain_wakeup()
            if self._proc.stdout in ready:
                line = self._proc.stdout.readline()
                if not line:
                    self._stream_reset()
                    break  # EOF
                line = line.strip()
                if not line:
                    idle_deadline.reset()
                    continue
                obj = json.loads(line)
                self._log_event(obj)
                with self._metrics_lock:
                    self._received_count += 1
                idle_deadline.reset()
                # First non-empty event after Send transitions Sending →
                # AwaitingReply.  Other states (AwaitingReply, Draining,
                # Idle scaffolding) leave the FSM untouched.
                with self._stream_lock:
                    if isinstance(self._stream_state, stream_fsm.Sending):
                        new_state = stream_fsm.transition(
                            self._stream_state, stream_fsm.ReplyChunk()
                        )
                        assert new_state is not None
                        self._stream_state = new_state
                # Track the latest session_id so :meth:`recover` and
                # :meth:`reset` can resume via ``--resume <sid>`` on the
                # next :meth:`_spawn`.
                sid = obj.get("session_id")
                if isinstance(sid, str) and sid:
                    self._session_id = sid
                # Track the cancel-interrupt's control_response ack so we
                # can ASSERT a clean drain before yielding the lock to
                # the next holder.  Without the ack we don't know claude
                # actually accepted the interrupt; exiting prematurely
                # leaves protocol state divergent and risks wedging
                # subsequent control_requests.
                if (
                    cancel_request_id is not None
                    and obj.get("type") == "control_response"
                    and (obj.get("response") or {}).get("request_id")
                    == cancel_request_id
                ):
                    cancel_ack_seen = True
                yield obj
                if obj.get("type") in ("result", "error"):
                    # Fire TurnReturn on the FSM where the model accepts it:
                    # AwaitingReply → Idle for normal turns, Draining →
                    # Cancelled for cancelled turns.  Other states (Idle,
                    # Cancelled, Sending) leave the FSM untouched — the test
                    # scaffolding can call iter_events without a prior send,
                    # and Sending here would mean we never observed a
                    # streaming chunk before the boundary, which the FSM
                    # treats as an invalid path.
                    with self._stream_lock:
                        if isinstance(
                            self._stream_state,
                            stream_fsm.AwaitingReply | stream_fsm.Draining,
                        ):
                            new_state = stream_fsm.transition(
                                self._stream_state, stream_fsm.TurnReturn()
                            )
                            assert new_state is not None
                            self._stream_state = new_state
                    if cancelled_at is not None and cancel_request_id is not None:
                        # The cancelled turn must have closed cleanly: we
                        # sent the interrupt, claude acked it, and now
                        # we've reached the boundary.  If the ack never
                        # arrived, the drain wasn't clean — kill + recover
                        # rather than hand a half-cancelled session to the
                        # next holder.
                        if not cancel_ack_seen:
                            log.warning(
                                "ClaudeSession: cancel boundary reached without "
                                "control_response ack — recovering"
                            )
                            self._proc.kill()
                            self._proc.wait()
                            self.recover()
                            raise ClaudeStreamError(_RETURNCODE_CANCEL_DRAIN_TIMEOUT)
                    break
            elif self._proc.poll() is not None:
                self._stream_reset()
                break  # process exited

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
                except _requests.RequestException as exc:
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
        session_fn: Callable[[], PromptSession] = provider.current_repo_session,
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

    @property
    def supports_no_commit_reset(self) -> bool:
        return True

    def _spawn_owned_session(
        self,
        model: ProviderModel | None = None,
        *,
        session_id: str | None = None,
    ) -> PromptSession:
        system_file = self._session_system_file
        work_dir = self._work_dir
        assert system_file is not None
        assert work_dir is not None
        return self._session_factory(
            system_file,
            work_dir=work_dir,
            repo_name=self._repo_name,
            model=model,
            session_id=session_id,
        )

    def ensure_session(
        self,
        model: ProviderModel | None = None,
        *,
        session_id: str | None = None,
    ) -> None:
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
                session = self._spawn_owned_session(model, session_id=session_id)
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

    def generate_status(
        self,
        prompt: str,
        system_prompt: str,
        model: ProviderModel | None = None,
        *,
        allowed_tools: str | None = READ_ONLY_ALLOWED_TOOLS,
    ) -> str:
        """Ask claude to generate a GitHub status (two lines: emoji + text)."""
        return self.run_turn(
            prompt,
            model=self.voice_model if model is None else model,
            allowed_tools=allowed_tools,
            system_prompt=system_prompt,
        )

    def generate_status_emoji(
        self,
        prompt: str,
        system_prompt: str,
        model: ProviderModel | None = None,
        *,
        allowed_tools: str | None = READ_ONLY_ALLOWED_TOOLS,
    ) -> str:
        """Ask claude to choose a single emoji for a GitHub status."""
        return self._run_turn_json_value(
            prompt,
            "emoji",
            self.voice_model if model is None else model,
            allowed_tools=allowed_tools,
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
        *,
        allowed_tools: str | None = READ_ONLY_ALLOWED_TOOLS,
    ) -> str:
        """Run claude --print reading system prompt and user prompt from files."""
        # `timeout` is on the protocol because codex/copilot use a wall-clock
        # cap; the persistent-session path here applies idle_timeout via
        # _streaming_runner instead, so the wall-clock cap is unused.
        del timeout
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
        if allowed_tools is not None:
            cmd += ["--allowedTools", allowed_tools]
        cmd += ["--disallowedTools", GLOBAL_DISALLOWED_TOOLS]
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
        *,
        allowed_tools: str | None = READ_ONLY_ALLOWED_TOOLS,
    ) -> str:
        """Continue an existing claude session by ID, feeding prompt_file on stdin."""
        # See note on print_prompt_from_file: claude uses idle_timeout, not timeout.
        del timeout
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
        if allowed_tools is not None:
            cmd += ["--allowedTools", allowed_tools]
        cmd += ["--disallowedTools", GLOBAL_DISALLOWED_TOOLS]
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
