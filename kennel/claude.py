"""Claude CLI wrappers — all claude subprocess calls in one place."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _claude(
    *args: str,
    prompt: str | None = None,
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    """Run the claude CLI with the given args, optionally piping prompt to stdin."""
    return subprocess.run(
        ["claude", *args],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ── Simple print calls (no tool use) ─────────────────────────────────────────


def print_prompt(
    prompt: str,
    model: str,
    system_prompt: str | None = None,
    timeout: int = 30,
) -> str:
    """Run claude --print with a prompt, return stdout (empty string on failure).

    Uses -p to pass the prompt as a positional argument (no tool access).
    """
    args: list[str] = ["--model", model, "--print"]
    if system_prompt is not None:
        args += ["--system-prompt", system_prompt]
    args += ["-p", prompt]
    try:
        result = _claude(*args, timeout=timeout)
        return result.stdout.strip() if result.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def print_prompt_from_file(
    system_file: Path,
    prompt_file: Path,
    model: str,
    timeout: int = 30,
) -> str:
    """Run claude --print reading system prompt and user prompt from files.

    Returns the session_id from the result JSON (last non-empty line of stdout).
    Used for sub-Claude sessions started fresh (not resumed).
    """
    try:
        result = subprocess.run(
            [
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
            ],
            stdin=prompt_file.open(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def resume_session(
    session_id: str,
    prompt_file: Path,
    model: str,
    timeout: int = 300,
) -> str:
    """Continue an existing claude session by ID, feeding prompt_file on stdin.

    Returns stdout (progress output), empty string on failure.
    """
    try:
        result = subprocess.run(
            [
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
            ],
            stdin=prompt_file.open(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


# ── Specialised wrappers used by events.py ───────────────────────────────────


def triage_comment(
    prompt: str,
    model: str = "claude-opus-4-6",
    timeout: int = 15,
) -> str:
    """Ask claude to triage a PR comment. Returns the raw first line of output.

    Returns empty string on timeout or if the CLI is not found.
    """
    try:
        result = _claude("--model", model, "--print", "-p", prompt, timeout=timeout)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0]
        return ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def generate_reply(
    prompt: str,
    model: str = "claude-opus-4-6",
    timeout: int = 30,
) -> str:
    """Ask claude to generate a short reply. Returns stripped output or empty string."""
    try:
        result = _claude("--model", model, "--print", "-p", prompt, timeout=timeout)
        return result.stdout.strip() if result.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def generate_branch_name(
    prompt: str,
    model: str = "claude-haiku-4-5-20251001",
    timeout: int = 15,
) -> str:
    """Ask claude to generate a git branch name slug. Returns first line of output."""
    try:
        result = _claude("--model", model, "--print", "-p", prompt, timeout=timeout)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0]
        return ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def generate_status(
    prompt: str,
    system_prompt: str,
    model: str = "claude-opus-4-6",
    timeout: int = 15,
) -> str:
    """Ask claude to generate a GitHub status (two lines: emoji + text)."""
    return print_prompt(
        prompt=prompt,
        model=model,
        system_prompt=system_prompt,
        timeout=timeout,
    )
