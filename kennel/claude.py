"""Claude CLI wrappers — all claude subprocess calls in one place."""

from __future__ import annotations

import json
import logging
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
from typing import Literal

log = logging.getLogger(__name__)

# How many seconds select.select waits for stdout before checking for
# process exit or idle-timeout.  Short enough to react quickly, long enough
# not to busy-loop.
_SELECT_POLL_INTERVAL = 10.0

# Maximum number of characters included when logging a raw line from the
# claude subprocess, to keep log records readable.
_LOG_LINE_TRUNCATE = 200


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
_active_children: set[subprocess.Popen] = set()
_active_children_lock = threading.Lock()


def _register_child(proc: subprocess.Popen) -> None:
    with _active_children_lock:
        _active_children.add(proc)


def _unregister_child(proc: subprocess.Popen) -> None:
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


_session_resolver: Callable[[str], "ClaudeSession | None"] | None = None
"""Callback the event/webhook layer uses to find its repo's persistent
:class:`ClaudeSession` — installed once by :mod:`kennel.server` at startup.

Every in-process prompt call goes through the persistent session, so
this is a required piece of wiring.  Callers (:func:`print_prompt`) fail
loud if it's missing — the only time that should happen is a forgotten
resolver install, not a real production path."""


def set_session_resolver(
    resolver: Callable[[str], "ClaudeSession | None"] | None,
) -> None:
    """Install (or clear) the session resolver callback."""
    global _session_resolver
    _session_resolver = resolver


def _session_for_current_repo() -> "ClaudeSession":
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
            "claude.print_prompt called without a thread-local repo_name — "
            "server.WebhookHandler._process_action and WorkerThread.run both "
            "set it; this caller is missing the install."
        )
    if _session_resolver is None:
        raise RuntimeError(
            "claude.print_prompt called before set_session_resolver — "
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
    """Run the claude CLI with the given args, optionally piping prompt to stdin."""
    return runner(
        ["claude", *args],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


_RETURNCODE_IDLE_TIMEOUT = -1
"""Sentinel returncode used in :class:`ClaudeStreamError` when the process is
killed due to an idle timeout rather than exiting with a real non-zero code."""


class ClaudeStreamError(Exception):
    """Raised by _run_streaming when the subprocess exits with a non-zero code or times out."""

    def __init__(self, returncode: int) -> None:
        self.returncode = returncode
        super().__init__(f"claude exited with code {returncode}")


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


# ── Simple print calls (no tool use) ─────────────────────────────────────────


_EMPTY_RETRY_COUNT = 2
_EMPTY_RETRY_DELAY = 1.0


def print_prompt(
    prompt: str,
    model: str,
    system_prompt: str | None = None,
) -> str:
    """Run a prompt turn on this thread's persistent :class:`ClaudeSession`,
    returning the result text.

    Always routes through the session resolved via :func:`set_session_resolver`
    — production has one claude subprocess per repo, and every prompt call
    shares it.  Errors propagate (we fail open, not silently masked).
    """
    session = _session_for_current_repo()
    return session.prompt(prompt, model=model, system_prompt=system_prompt)


def print_prompt_json(
    prompt: str,
    key: str,
    model: str,
    system_prompt: str | None = None,
) -> str:
    """Run a prompt turn and parse JSON from the response at *key*.

    Appends a JSON-format instruction to *system_prompt* so Claude outputs
    ``{"key": "..."}``.  Scans the raw response for a JSON object so
    preamble or trailing text from Opus does not corrupt the result.
    Returns ``""`` if the JSON parse fails (but claude errors propagate
    from :func:`print_prompt` — we fail open on infrastructure failure).
    """
    json_instruction = (
        f'Respond with ONLY a JSON object in the form {{"{key}": "your answer"}}.'
        " No other text before or after the JSON."
    )
    full_system = (
        f"{system_prompt}\n\n{json_instruction}" if system_prompt else json_instruction
    )
    raw = print_prompt(prompt, model, system_prompt=full_system)
    if not raw:
        return ""
    # Try parsing whole output, then scan for JSON objects (handles preamble).
    candidates = [raw] + [m.group() for m in re.finditer(r"\{.*?\}", raw, re.DOTALL)]
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj.get(key), str):
                return obj[key]
        except json.JSONDecodeError, AttributeError:
            continue
    return ""


def _run_streaming(
    cmd: list[str],
    stdin_file: Path,
    idle_timeout: float = 1800.0,
    cwd: Path | str | None = None,
    popen: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    selector: Callable[..., tuple[list, list, list]] = select.select,
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


def print_prompt_from_file(
    system_file: Path,
    prompt_file: Path,
    model: str,
    timeout: int = 30,
    idle_timeout: float = 1800.0,
    cwd: Path | str | None = None,
    streaming_runner: Callable[..., Iterator[str]] = _run_streaming,
) -> str:
    """Run claude --print reading system prompt and user prompt from files.

    Returns the full stdout on success.  Kills the process if no output
    is produced for *idle_timeout* seconds (default 30 min).

    Raises ``ClaudeStreamError`` on nonzero exit or idle timeout.
    ``FileNotFoundError`` propagates if the claude CLI is not installed.
    Both are authoritative failures — callers should not silently ignore them.
    """
    cmd = [
        "claude",
        "--model",
        model,
        "--output-format",
        "stream-json",
        "--verbose",
        "--dangerously-skip-permissions",
        "--system-prompt-file",
        str(system_file),
        "--print",
    ]
    return "".join(streaming_runner(cmd, prompt_file, idle_timeout, cwd=cwd)).strip()


def resume_session(
    session_id: str,
    prompt_file: Path,
    model: str,
    timeout: int = 300,
    idle_timeout: float = 1800.0,
    cwd: Path | str | None = None,
    streaming_runner: Callable[..., Iterator[str]] = _run_streaming,
) -> str:
    """Continue an existing claude session by ID, feeding prompt_file on stdin.

    Returns the full stdout on success.  Kills the process if no output
    is produced for *idle_timeout* seconds (default 30 min).

    Raises ``ClaudeStreamError`` on nonzero exit or idle timeout.
    ``FileNotFoundError`` propagates if the claude CLI is not installed.
    Both are authoritative failures — callers should not silently ignore them.
    """
    cmd = [
        "claude",
        "--model",
        model,
        "--output-format",
        "stream-json",
        "--verbose",
        "--dangerously-skip-permissions",
        "--resume",
        session_id,
        "--print",
    ]
    return "".join(streaming_runner(cmd, prompt_file, idle_timeout, cwd=cwd)).strip()


# ── Specialised wrappers used by events.py ───────────────────────────────────


def triage_comment(
    prompt: str,
    model: str = "claude-opus-4-6",
    timeout: int = 15,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    """Ask claude to triage a PR comment. Returns the raw first line of output.

    Best-effort enrichment: returns ``""`` on nonzero exit, timeout, or missing
    CLI — callers must treat an empty result as "unable to triage".
    """
    try:
        result = _claude(
            "--model", model, "--print", "-p", prompt, timeout=timeout, runner=runner
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0]
        return ""
    except subprocess.TimeoutExpired, FileNotFoundError:
        return ""


def generate_reply(
    prompt: str,
    model: str = "claude-opus-4-6",
    timeout: int = 30,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    """Ask claude to generate a short reply. Returns stripped output or empty string.

    Best-effort enrichment: returns ``""`` on nonzero exit, timeout, or missing
    CLI — callers must treat an empty result as "no reply generated".
    """
    try:
        result = _claude(
            "--model", model, "--print", "-p", prompt, timeout=timeout, runner=runner
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except subprocess.TimeoutExpired, FileNotFoundError:
        return ""


def generate_branch_name(
    prompt: str,
    model: str = "claude-haiku-4-5-20251001",
    timeout: int = 15,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    """Ask claude to generate a git branch name slug. Returns first line of output.

    Best-effort enrichment: returns ``""`` on nonzero exit, timeout, or missing
    CLI — callers must treat an empty result as "use a fallback branch name".
    """
    try:
        result = _claude(
            "--model", model, "--print", "-p", prompt, timeout=timeout, runner=runner
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0]
        return ""
    except subprocess.TimeoutExpired, FileNotFoundError:
        return ""


def generate_status(
    prompt: str,
    system_prompt: str,
    model: str = "claude-opus-4-6",
) -> str:
    """Ask claude to generate a GitHub status (two lines: emoji + text)."""
    return print_prompt(
        prompt=prompt,
        model=model,
        system_prompt=system_prompt,
    )


def generate_status_emoji(
    prompt: str,
    system_prompt: str,
    model: str = "claude-opus-4-6",
) -> str:
    """Ask claude to choose a single emoji for a GitHub status."""
    return print_prompt(
        prompt=prompt,
        model=model,
        system_prompt=system_prompt,
    )


def generate_status_with_session(
    prompt: str,
    system_prompt: str,
    model: str = "claude-opus-4-6",
    timeout: int = 15,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    _sleep: Callable[[float], None] = time.sleep,
) -> tuple[str, str]:
    """Generate a GitHub status, returning (status_text, session_id).

    Best-effort enrichment: returns ``("", "")`` on any failure — callers must
    treat an all-empty tuple as "status unavailable".  Uses stream-json output
    format to capture the session_id alongside the response text, enabling
    follow-up calls (e.g., ``resume_status`` to shorten a long response).
    Retries up to ``_EMPTY_RETRY_COUNT`` times when Claude exits 0 but
    produces no output.
    """
    for attempt in range(_EMPTY_RETRY_COUNT + 1):
        try:
            result = _claude(
                "--model",
                model,
                "--output-format",
                "stream-json",
                "--verbose",
                "--dangerously-skip-permissions",
                "--system-prompt",
                system_prompt,
                "--print",
                "-p",
                prompt,
                timeout=timeout,
                runner=runner,
            )
            if result.returncode != 0:
                log.warning(
                    "generate_status_with_session: claude exited %d", result.returncode
                )
                return "", ""
            raw = result.stdout.strip()
            text = extract_result_text(raw)
            sid = extract_session_id(raw)
            if text:
                return text, sid
            if result.stderr:
                log.warning(
                    "generate_status_with_session: stderr=%r", _Trunc(result.stderr)
                )
            log.debug("generate_status_with_session: stdout=%r", _Trunc(result.stdout))
            if attempt < _EMPTY_RETRY_COUNT:
                log.warning(
                    "generate_status_with_session: empty output on attempt %d — retrying",
                    attempt + 1,
                )
                _sleep(_EMPTY_RETRY_DELAY)
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            log.warning("generate_status_with_session: %s", exc)
            return "", ""
    log.warning(
        "generate_status_with_session: empty output after %d attempts",
        _EMPTY_RETRY_COUNT + 1,
    )
    return "", ""


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
        idle_timeout: float = 1800.0,
        popen: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
        selector: Callable[..., tuple[list, list, list]] = select.select,
        repo_name: str | None = None,
    ) -> None:
        self._idle_timeout = idle_timeout
        self._selector = selector
        self._system_file = system_file
        self._work_dir = work_dir
        self._popen_fn = popen
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._repo_name = repo_name
        # True when the most recent :meth:`iter_events` call exited early
        # because :attr:`_cancel` was set (i.e. another thread preempted the
        # turn via :meth:`prompt`).  Cleared at the start of each turn.
        # Callers use this to distinguish "turn completed with empty result"
        # from "turn was interrupted and should be retried once the lock is
        # free again".
        self._last_turn_cancelled = False
        self._proc = self._spawn()
        _register_child(self._proc)

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
        """Spawn the claude subprocess with bidirectional stream-json I/O."""
        cmd = [
            "claude",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
            "--system-prompt-file",
            str(self._system_file),
        ]
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

    def restart(self) -> None:
        """Stop the current subprocess and start a fresh one.

        Unregisters the dead process from ``_active_children``, kills it if
        still running, then spawns a replacement and registers it.  The
        conversation transcript is lost — callers are responsible for
        re-sending any context the new process needs.
        """
        log.warning("ClaudeSession: restarting after unexpected process death")
        _unregister_child(self._proc)
        if self._proc.poll() is None:
            try:
                self._proc.kill()
                self._proc.wait(timeout=1.0)
            except (OSError, ProcessLookupError, subprocess.TimeoutExpired) as exc:
                log.warning("ClaudeSession.restart: kill/wait failed: %s", exc)
                raise
        self._proc = self._spawn()
        _register_child(self._proc)

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
        """Write a user message to the session stdin, flushing immediately."""
        msg = json.dumps(
            {"type": "user", "message": {"role": "user", "content": content}}
        )
        assert self._proc.stdin is not None
        self._proc.stdin.write(msg + "\n")
        self._proc.stdin.flush()

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
        with self._lock:
            self._send_control_interrupt()
            self.consume_until_result()
            self.send(content)

    def prompt(
        self,
        content: str,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Steal the session, send *content* as a user message, return the result.

        Intended as the session-aware replacement for one-shot
        :func:`print_prompt` / :func:`print_prompt_json` calls on webhook-
        handler threads.  Runs one full turn on the persistent session:

        1. Signal cancel to wake any current holder out of
           :meth:`iter_events`.  They exit their turn early, their context
           manager releases the lock and unregisters their talker.
        2. Acquire ``self`` as a context manager — blocks while the previous
           holder is winding down.  Registers a fresh :class:`ClaudeTalker`
           so status display attributes claude to this thread.
        3. Send a stream-json ``control_request`` interrupt + drain stale
           events from the cancelled turn so stdout is clean.
        4. Switch model if *model* is provided and differs from the session
           default.
        5. Send *content* (optionally prefixed with *system_prompt*) and
           consume until the result boundary, returning the text.
        """
        self._cancel.set()
        with self:
            self._send_control_interrupt()
            self.consume_until_result()
            if model is not None:
                self.switch_model(model)
            if system_prompt:
                body = f"{system_prompt}\n\n---\n\n{content}"
            else:
                body = content
            self.send(body)
            return self.consume_until_result()

    def switch_model(self, model: str) -> None:
        """Switch the active model by sending a /model slash command.

        Sends ``/model <model>`` as a user message and drains any response
        events so the turn boundary is clean before the next call to
        :meth:`send` + :meth:`iter_events`.
        """
        self.send(f"/model {model}")
        for _ in self.iter_events():
            pass

    def iter_events(self) -> Iterator[dict]:
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
                break
            ready, _, _ = self._selector(
                [self._proc.stdout], [], [], _SELECT_POLL_INTERVAL
            )
            if ready:
                line = self._proc.stdout.readline()
                if not line:
                    break  # EOF
                line = line.strip()
                if not line:
                    last_activity = time.monotonic()
                    continue
                obj = json.loads(line)
                log.debug("ClaudeSession event: %s", _Trunc(line))
                last_activity = time.monotonic()
                yield obj
                if obj.get("type") in ("result", "error"):
                    break
            elif self._proc.poll() is not None:
                break  # process exited
            elif time.monotonic() - last_activity > self._idle_timeout:
                log.warning(
                    "ClaudeSession: idle for %.0fs — killing", self._idle_timeout
                )
                self._proc.kill()
                self._proc.wait()
                self.restart()
                raise ClaudeStreamError(_RETURNCODE_IDLE_TIMEOUT)

    def consume_until_result(self) -> str:
        """Drain events for the current turn and return the result text.

        Exhausts :meth:`iter_events` and returns the ``result`` field from
        the ``type=result`` event, or an empty string if the turn ends
        without one (EOF, ``type=error``, or idle-timeout kill).
        """
        result_text = ""
        for event in self.iter_events():
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


def resume_status(
    session_id: str,
    prompt: str,
    model: str = "claude-opus-4-6",
    timeout: int = 15,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    """Resume an existing claude session to refine a status response.

    Best-effort enrichment: returns ``""`` on nonzero exit, timeout, or missing
    CLI — callers must treat an empty result as "refinement unavailable".
    Passes *prompt* as a positional argument (``-p``), so no file is needed.
    """
    try:
        result = _claude(
            "--model",
            model,
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
            "--resume",
            session_id,
            "--print",
            "-p",
            prompt,
            timeout=timeout,
            runner=runner,
        )
        if result.returncode != 0:
            return ""
        return extract_result_text(result.stdout.strip())
    except subprocess.TimeoutExpired, FileNotFoundError:
        return ""
