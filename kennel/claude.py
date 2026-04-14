"""Claude CLI wrappers — all claude subprocess calls in one place."""

from __future__ import annotations

import json
import logging
import re
import select
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from kennel.state import PreemptQueue

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
    timeout: int = 30,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    _sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Run claude --print with a prompt, return stdout (empty string on failure).

    Best-effort enrichment: nonzero exit, timeout, and missing CLI all return
    ``""`` — callers must treat an empty result as "unavailable" rather than
    as an error.  Uses -p to pass the prompt as a positional argument (no tool
    access).  Retries up to ``_EMPTY_RETRY_COUNT`` times (with a short delay)
    when Claude exits 0 but produces no output, to handle transient empty
    responses.
    """
    args: list[str] = ["--model", model, "--print"]
    if system_prompt is not None:
        args += ["--system-prompt", system_prompt]
    args += ["-p", prompt]
    for attempt in range(_EMPTY_RETRY_COUNT + 1):
        try:
            result = _claude(*args, timeout=timeout, runner=runner)
            if result.returncode != 0:
                log.warning("print_prompt: claude exited %d", result.returncode)
                return ""
            text = result.stdout.strip()
            if text:
                return text
            if result.stderr:
                log.warning("print_prompt: stderr=%r", _Trunc(result.stderr))
            log.debug("print_prompt: stdout=%r", _Trunc(result.stdout))
            if attempt < _EMPTY_RETRY_COUNT:
                log.warning(
                    "print_prompt: empty output on attempt %d — retrying",
                    attempt + 1,
                )
                _sleep(_EMPTY_RETRY_DELAY)
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            log.warning("print_prompt: %s", exc)
            return ""
    log.warning("print_prompt: empty output after %d attempts", _EMPTY_RETRY_COUNT + 1)
    return ""


def print_prompt_json(
    prompt: str,
    key: str,
    model: str,
    system_prompt: str | None = None,
    timeout: int = 30,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    """Run claude --print, parse JSON from output, return the string at *key*.

    Best-effort enrichment: returns ``""`` on any subprocess failure or JSON
    parse error — callers must treat an empty result as "unavailable".
    Appends a JSON-format instruction to *system_prompt* so Claude outputs
    {"key": "..."}.  Scans the raw response for a JSON object, so preamble
    or trailing text from Opus does not corrupt the result.
    """
    json_instruction = (
        f'Respond with ONLY a JSON object in the form {{"{key}": "your answer"}}.'
        " No other text before or after the JSON."
    )
    full_system = (
        f"{system_prompt}\n\n{json_instruction}" if system_prompt else json_instruction
    )
    raw = print_prompt(
        prompt, model, system_prompt=full_system, timeout=timeout, runner=runner
    )
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

    try:
        last_activity = time.monotonic()

        while True:
            ready, _, _ = selector([proc.stdout], [], [], _SELECT_POLL_INTERVAL)
            if ready:
                line = proc.stdout.readline()
                if not line:
                    break  # EOF
                yield line
                log.debug(line.rstrip())
                last_activity = time.monotonic()
            elif proc.poll() is not None:
                break  # process exited
            elif time.monotonic() - last_activity > idle_timeout:
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
    timeout: int = 15,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    """Ask claude to generate a GitHub status (two lines: emoji + text).

    Best-effort enrichment: returns ``""`` on any failure — callers must treat
    an empty result as "status unavailable".
    """
    return print_prompt(
        prompt=prompt,
        model=model,
        system_prompt=system_prompt,
        timeout=timeout,
        runner=runner,
    )


def generate_status_emoji(
    prompt: str,
    system_prompt: str,
    model: str = "claude-opus-4-6",
    timeout: int = 15,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    """Ask claude to choose a single emoji for a GitHub status.

    Best-effort enrichment: returns ``""`` on any failure — callers must treat
    an empty result as "emoji unavailable".
    """
    return print_prompt(
        prompt=prompt,
        model=model,
        system_prompt=system_prompt,
        timeout=timeout,
        runner=runner,
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
        preempt_queue: PreemptQueue | None = None,
    ) -> None:
        self._idle_timeout = idle_timeout
        self._selector = selector
        self._system_file = system_file
        self._work_dir = work_dir
        self._popen_fn = popen
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._preempt_queue = preempt_queue
        self._owner: str | None = None
        self._proc = self._spawn()
        _register_child(self._proc)

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

        Set to :func:`threading.current_thread().name <threading.current_thread>`
        immediately after the lock is acquired in :meth:`__enter__` and
        cleared before it is released in :meth:`__exit__`.  Read without
        holding the lock — safe to poll from status-display threads for a
        best-effort snapshot of who is actively driving the session.
        """
        return self._owner

    def __enter__(self) -> "ClaudeSession":
        """Acquire the session lock, serializing send/receive across threads.

        Also clears the cancel event so a turn that follows a preempt or
        interrupt starts with a clean slate.  Records the current thread name
        in :attr:`owner` so status display can show who holds the session.
        """
        self._lock.acquire()
        self._cancel.clear()
        self._owner = threading.current_thread().name
        return self

    def __exit__(self, *args: object) -> None:
        """Release the session lock.  Clears :attr:`owner` before releasing."""
        self._owner = None
        self._lock.release()

    def send(self, content: str) -> None:
        """Write a user message to the session stdin, flushing immediately."""
        msg = json.dumps(
            {"type": "user", "message": {"role": "user", "content": content}}
        )
        assert self._proc.stdin is not None
        self._proc.stdin.write(msg + "\n")
        self._proc.stdin.flush()

    def interrupt(self, content: str) -> None:
        """Signal the in-flight turn to stop, then send *content* under the lock.

        Sets the cancel event so :meth:`iter_events` exits on its next poll
        cycle.  The lock holder's ``with session:`` block then unwinds and
        releases the lock.  This method acquires the lock (blocking until the
        holder releases it), sends *content* as a normal user message, and
        releases the lock.  No stdin write ever races with the holder's writes.

        For preempt/interrupt use cases (rescope abort, CI-failure cancel)
        where the in-flight turn must be cancelled and a follow-up sent.
        """
        self._cancel.set()
        with self._lock:
            self.send(content)

    def preempt(self, content: str) -> None:
        """Signal the in-flight turn to stop and queue a follow-up user turn.

        Sets the cancel event so :meth:`iter_events` exits on its next poll
        cycle.  The lock holder's ``with session:`` block then unwinds and
        releases the lock.  This method acquires the lock (blocking until the
        holder releases it), appends *content* to the durable FIFO queue, and
        releases the lock.  Multiple consecutive preempts accumulate — no
        message is lost even if preempt is called again before the first
        queued turn is consumed.  The queued content is retrieved one item at
        a time via :meth:`take_queued_content` by the next caller that holds
        the lock.

        Only for preempt paths (rescope abort, CI-failure cancel).  Normal
        turns must go through the context-manager lock.
        """
        self._cancel.set()
        with self._lock:
            assert self._preempt_queue is not None, (
                "preempt_queue required for preempt()"
            )
            self._preempt_queue.push(content)

    def take_queued_content(self) -> str | None:
        """Pop and return the oldest queued preempt content, or ``None``.

        Returns ``None`` if the queue is empty.  Call this while holding the
        session lock (via the context manager) so that reading and removing
        from the queue is serialized with respect to a concurrent
        :meth:`preempt` appending to it.
        """
        assert self._preempt_queue is not None, (
            "preempt_queue required for take_queued_content()"
        )
        return self._preempt_queue.pop()

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
        last_activity = time.monotonic()

        while True:
            if self._cancel.is_set():
                log.debug("ClaudeSession: cancelled — exiting turn early")
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
