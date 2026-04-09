"""Claude CLI wrappers — all claude subprocess calls in one place."""

from __future__ import annotations

import json
import logging
import re
import select
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path

log = logging.getLogger(__name__)


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


def print_prompt(
    prompt: str,
    model: str,
    system_prompt: str | None = None,
    timeout: int = 30,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    """Run claude --print with a prompt, return stdout (empty string on failure).

    Uses -p to pass the prompt as a positional argument (no tool access).
    """
    args: list[str] = ["--model", model, "--print"]
    if system_prompt is not None:
        args += ["--system-prompt", system_prompt]
    args += ["-p", prompt]
    try:
        result = _claude(*args, timeout=timeout, runner=runner)
        return result.stdout.strip() if result.returncode == 0 else ""
    except subprocess.TimeoutExpired, FileNotFoundError:
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

    Appends a JSON-format instruction to *system_prompt* so Claude outputs
    {"key": "..."}.  Scans the raw response for a JSON object, so preamble
    or trailing text from Opus does not corrupt the result.

    Returns the string value at *key*, or empty string on failure.
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
    *idle_timeout* seconds, the process is killed and ``ClaudeStreamError(-1)``
    is raised.  If the process exits with a non-zero code,
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

    last_activity = time.monotonic()

    while True:
        ready, _, _ = selector([proc.stdout], [], [], 10.0)
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
            raise ClaudeStreamError(-1)

    proc.wait()
    # Drain any remaining output
    remaining = proc.stdout.read()
    if remaining:
        yield remaining
    if proc.returncode != 0:
        raise ClaudeStreamError(proc.returncode)


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

    Returns stdout on success, empty string on failure.  Kills the process
    if no output is produced for *idle_timeout* seconds (default 30 min).
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
    try:
        return "".join(
            streaming_runner(cmd, prompt_file, idle_timeout, cwd=cwd)
        ).strip()
    except ClaudeStreamError, FileNotFoundError:
        return ""


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

    Returns stdout on success, empty string on failure.  Kills the process
    if no output is produced for *idle_timeout* seconds (default 30 min).
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
    try:
        return "".join(
            streaming_runner(cmd, prompt_file, idle_timeout, cwd=cwd)
        ).strip()
    except ClaudeStreamError, FileNotFoundError:
        return ""


# ── Specialised wrappers used by events.py ───────────────────────────────────


def triage_comment(
    prompt: str,
    model: str = "claude-opus-4-6",
    timeout: int = 15,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    """Ask claude to triage a PR comment. Returns the raw first line of output.

    Returns empty string on timeout or if the CLI is not found.
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
    """Ask claude to generate a short reply. Returns stripped output or empty string."""
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
    """Ask claude to generate a git branch name slug. Returns first line of output."""
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
    """Ask claude to generate a GitHub status (two lines: emoji + text)."""
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

    Returns the stripped response, or an empty string on failure.
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
) -> tuple[str, str]:
    """Generate a GitHub status, returning (status_text, session_id).

    Uses stream-json output format to capture the session_id alongside the
    response text, enabling follow-up calls (e.g., ``resume_status`` to
    shorten a long response).  Returns ``("", "")`` on failure.
    """
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
            return "", ""
        raw = result.stdout.strip()
        return extract_result_text(raw), extract_session_id(raw)
    except subprocess.TimeoutExpired, FileNotFoundError:
        return "", ""


def resume_status(
    session_id: str,
    prompt: str,
    model: str = "claude-opus-4-6",
    timeout: int = 15,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    """Resume an existing claude session to refine a status response.

    Passes *prompt* as a positional argument (``-p``), so no file is needed.
    Returns the response text extracted from stream-json output, or an empty
    string on failure.
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
