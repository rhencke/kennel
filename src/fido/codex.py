"""Codex CLI helpers.

This module intentionally covers only the non-persistent ``codex exec`` path.
Persistent session transport and provider factory wiring land in later Codex
epic issues.
"""

import json
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from fido.provider import ProviderModel, model_name


class CodexCLIError(RuntimeError):
    """Raised when the Codex CLI process exits unsuccessfully."""

    def __init__(self, returncode: int, stderr: str) -> None:
        self.returncode = returncode
        self.stderr = stderr
        message = stderr.strip() or f"codex exited with code {returncode}"
        super().__init__(message)


class CodexProviderError(RuntimeError):
    """Raised when Codex reports a provider/auth/quota failure in JSONL output."""

    def __init__(
        self,
        *,
        message: str,
        kind: str = "provider",
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.kind = kind
        self.payload = payload or {}
        super().__init__(message)


def _codex(
    *args: str,
    prompt: str | None = None,
    timeout: int = 30,
    cwd: Path | str | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    """Run the Codex CLI with *args*, optionally piping *prompt* to stdin."""
    return runner(
        ["codex", *args],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
    )


def _iter_jsonl(output: str) -> Iterable[dict[str, Any]]:
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


def extract_session_id(output: str) -> str:
    """Extract the final Codex thread id from ``codex exec --json`` output."""
    result = ""
    for obj in _iter_jsonl(output):
        if obj.get("type") != "thread.started":
            continue
        thread_id = obj.get("thread_id")
        if isinstance(thread_id, str) and thread_id:
            result = thread_id
    return result


def extract_result_text(output: str) -> str:
    """Extract the last completed agent message from Codex exec JSONL."""
    result = ""
    for obj in _iter_jsonl(output):
        if obj.get("type") != "item.completed":
            continue
        item = obj.get("item")
        if not isinstance(item, dict) or item.get("type") != "agent_message":
            continue
        text = item.get("text")
        if isinstance(text, str) and text:
            result = text
    return result


def _classify_provider_error(message: str) -> str:
    lowered = message.lower()
    if "rate limit" in lowered or "rate_limit" in lowered or "quota" in lowered:
        return "rate_limit"
    if "auth" in lowered or "login" in lowered or "unauthorized" in lowered:
        return "auth"
    if "cancel" in lowered or "interrupt" in lowered:
        return "cancelled"
    return "provider"


def _provider_error_from_event(obj: dict[str, Any]) -> CodexProviderError | None:
    event_type = obj.get("type")
    message: str | None = None
    payload: dict[str, Any] = obj
    if event_type == "error":
        raw = obj.get("message")
        message = raw if isinstance(raw, str) else str(obj)
    elif event_type == "turn.failed":
        error = obj.get("error")
        if isinstance(error, dict):
            raw = error.get("message")
            message = raw if isinstance(raw, str) else str(error)
        else:
            message = str(error or obj)
    if message is None:
        return None
    return CodexProviderError(
        message=message,
        kind=_classify_provider_error(message),
        payload=payload,
    )


def raise_for_provider_error_output(output: str) -> None:
    """Raise the first provider failure encoded in Codex exec JSONL output."""
    for obj in _iter_jsonl(output):
        provider_error = _provider_error_from_event(obj)
        if provider_error is not None:
            raise provider_error


def run_codex_exec(
    prompt: str,
    *,
    model: ProviderModel | str,
    timeout: int = 30,
    cwd: Path | str = ".",
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    """Run one non-persistent Codex exec turn and return raw JSONL output."""
    work_dir = Path(cwd).resolve()
    completed = _codex(
        "exec",
        "--json",
        "--model",
        model_name(model),
        "--sandbox",
        "danger-full-access",
        "--ask-for-approval",
        "never",
        "--skip-git-repo-check",
        "-C",
        str(work_dir),
        "-",
        prompt=prompt,
        timeout=timeout,
        cwd=work_dir,
        runner=runner,
    )
    if completed.returncode != 0:
        raise CodexCLIError(completed.returncode, completed.stderr)
    raise_for_provider_error_output(completed.stdout)
    return completed.stdout
