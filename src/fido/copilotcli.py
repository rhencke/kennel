"""Copilot CLI provider implementation."""

import asyncio
import json
import logging
import os
import re
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Coroutine, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TypeVar

import acp
from acp.exceptions import RequestError
from acp.schema import (
    AllowedOutcome,
    ClientCapabilities,
    DeniedOutcome,
    EnvVariable,
    FileSystemCapabilities,
    Implementation,
    PermissionOption,
    TerminalExitStatus,
    ToolCallUpdate,
)

from fido import provider
from fido.provider import (
    OwnedSession,
    PromptSession,
    Provider,
    ProviderAgent,
    ProviderAPI,
    ProviderID,
    ProviderLimitSnapshot,
    ProviderLimitWindow,
    ProviderModel,
    ReasoningEffort,
    coerce_provider_model,
)
from fido.session_agent import SessionBackedAgent

log = logging.getLogger(__name__)

_T = TypeVar("_T")

_COPILOT_COMMAND = ("copilot", "--acp", "--allow-all")
_COPILOT_JSON_BASE_ARGS = (
    "--output-format",
    "json",
    "--stream",
    "off",
    "--allow-all",
    "-s",
)
_ACP_STREAM_LIMIT = 8 * 1024 * 1024


def _is_missing_session_error(exc: RequestError) -> bool:
    """Return True when ACP reports that a session id no longer exists."""
    return "Resource not found: Session " in str(exc) and " not found" in str(exc)


def _is_line_limit_overrun_error(exc: Exception) -> bool:
    """Return True when ACP aborted on an oversized newline-delimited frame."""
    return "Separator is found, but chunk is longer than limit" in str(exc)


# #666: Copilot CLI emits this exact string as its final assistant content
# when a turn is cancelled mid-flight.  We translate it to "no result" at
# the talker boundary so consumers can't accidentally post it as a reply
# body or a PR description.
_COPILOT_CANCEL_SENTINEL = "Info: Operation cancelled by user"


def _is_cancel_sentinel(text: str) -> bool:
    """Return True when *text* is Copilot's cancellation sentinel.

    Matches both the bare sentinel and cases where it trails a block of
    real narration (seen in logs where Copilot streams progress and then
    appends the cancel notice as its final line).
    """
    if not text:
        return False
    stripped = text.rstrip()
    return stripped == _COPILOT_CANCEL_SENTINEL or stripped.endswith(
        "\n" + _COPILOT_CANCEL_SENTINEL
    )


def _iter_jsonl(output: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            result.append(obj)
    return result


def extract_result_text(output: str) -> str:
    """Extract the last assistant message content from Copilot JSONL output."""
    result = ""
    for obj in _iter_jsonl(output):
        if obj.get("type") != "assistant.message":
            continue
        data = obj.get("data")
        if not isinstance(data, dict):
            continue
        content = data.get("content")
        if isinstance(content, str):
            result = content
    return result


def extract_session_id(output: str) -> str:
    """Extract the final Copilot session id from JSONL output."""
    result = ""
    for obj in _iter_jsonl(output):
        if obj.get("type") != "result":
            continue
        session_id = obj.get("sessionId")
        if isinstance(session_id, str):
            result = session_id
    return result


# Models Copilot CLI currently accepts.  Surfaces from the CLI as the
# error message when an unknown name is requested ("Supported values:
# auto, gpt-5-mini, gpt-4.1, claude-haiku-4.5").  Update this set when
# the CLI's supported list changes (closes #1205).
_COPILOT_SUPPORTED_MODELS: frozenset[str] = frozenset(
    {"auto", "gpt-5-mini", "gpt-4.1", "claude-haiku-4.5"}
)

# How long to treat a recorded quota error as an active pause.  Copilot
# does not expose a reset timestamp, so we use a conservative one-hour
# window.  Update when real quota reset semantics are confirmed.
_COPILOT_QUOTA_PAUSE_SECONDS = 3600.0

# Substrings that identify a Copilot error as quota / rate-limit related.
# Grounded in the actual strings emitted by @github/copilot (sdk/index.js):
#
#   "You've reached your weekly rate limit."    ← user_weekly_rate_limited
#   "You've hit the rate limit for this model." ← user_model_rate_limited
#                                                   / integration_rate_limited
#   "You've hit your global rate limit."        ← user_global_rate_limited
#                                                   / rate_limited (fallback)
#   "Rate limit reached, waiting 1 minute before retrying..."
#   "Rate limit exceeded"
#   "No remaining quota for premium requests"
#   "Quota is insufficient to finish this session."
#
# All of the above fold to either "rate limit", "rate_limit", or "quota"
# when lowercased and substring-matched.
_COPILOT_QUOTA_PATTERNS: tuple[str, ...] = (
    "rate limit",  # covers all "You've … rate limit …" and "Rate limit …" messages
    "rate_limit",  # covers error codes in ACP exceptions (e.g. user_weekly_rate_limited)
    "quota",  # covers "No remaining quota" and "Quota is insufficient"
)


def _is_copilot_quota_error(exc: Exception) -> bool:
    """Return True when *exc* looks like a Copilot quota or rate-limit error.

    Matches against :data:`_COPILOT_QUOTA_PATTERNS`, which are grounded in the
    actual error messages emitted by ``@github/copilot`` (sdk/index.js).
    """
    lowered = str(exc).lower()
    return any(pat in lowered for pat in _COPILOT_QUOTA_PATTERNS)


def _normalize_model(model: ProviderModel | str | None) -> ProviderModel | None:
    if model is None:
        return None
    normalized = coerce_provider_model(model)
    lowered = normalized.model.lower()
    if lowered.startswith("claude-haiku"):
        return ProviderModel("claude-haiku-4.5", normalized.effort)
    if lowered.startswith(("claude-opus", "claude-sonnet")):
        return ProviderModel("gpt-4.1", normalized.effort)
    return normalized


def _combine_prompt(
    content: str, *, base_system_prompt: str = "", system_prompt: str | None = None
) -> str:
    parts = [
        part.strip()
        for part in (base_system_prompt, system_prompt, content)
        if part is not None and part.strip()
    ]
    return "\n\n---\n\n".join(parts)


def _unexpected_new_session_prompt(content: str) -> str:
    """Prepend a recovery nudge after ACP unexpectedly drops a stale session."""
    return (
        "IMPORTANT: Your previous persistent Copilot session was unexpectedly lost, "
        "so you are now continuing in a brand new session.\n\n"
        "Re-orient from the current repository state before acting. Do not rely on "
        "memory from the lost session. Treat the prompt below as the authoritative "
        "current task context, verify the live repo/branch/task state as needed, "
        "and continue the work without duplicating already-completed steps.\n\n"
        "---\n\n"
        f"{content}"
    )


def _copilot(
    *args: str,
    timeout: int = 30,
    cwd: Path | str | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    return runner(
        ["copilot", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
    )


def _repo_log_extra(repo_name: str | None) -> dict[str, str]:
    """Return a logging extra dict that routes records to the repo log."""
    if not repo_name:
        return {}
    return {"repo_name": repo_name.rsplit("/", 1)[-1]}


def _log_for_repo(
    level: int,
    repo_name: str | None,
    message: str,
    *args: object,
) -> None:
    extra = _repo_log_extra(repo_name)
    if extra:
        log.log(level, message, *args, extra=extra)
        return
    log.log(level, message, *args)


def _transcript_block(label: str, content: str) -> str:
    """Render one multiline transcript block for the log."""
    return f"{label} >>>\n{content}\n<<< {label}"


def _preview_log_value(value: object, limit: int = 200) -> str:
    """Return a compact one-line preview suitable for human logs."""
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, sort_keys=True)
        except TypeError:
            text = str(value)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _tool_input_preview(raw_input: object) -> str:
    if not isinstance(raw_input, dict):
        return _preview_log_value(raw_input)
    preview = raw_input.get("command") or raw_input.get("path") or raw_input.get("url")
    if not preview:
        preview = raw_input.get("query") or raw_input.get("prompt")
    if not preview and raw_input:
        preview = next(iter(raw_input.values()))
    return _preview_log_value(preview)


@dataclass
class _TerminalRecord:
    process: subprocess.Popen[str]
    output_limit: int | None
    chunks: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    readers: tuple[threading.Thread, ...] = ()

    def append(self, chunk: str) -> None:
        with self.lock:
            self.chunks.append(chunk)

    def output(self) -> tuple[str, bool, int | None, str | None]:
        with self.lock:
            text = "".join(self.chunks)
        truncated = False
        if self.output_limit is not None and len(text) > self.output_limit:
            text = text[-self.output_limit :]
            truncated = True
        returncode = self.process.poll()
        if returncode is None:
            return text, truncated, None, None
        if returncode < 0:
            return text, truncated, None, signal.Signals(-returncode).name
        return text, truncated, returncode, None


class _TerminalManager:
    def __init__(
        self,
        *,
        popen: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    ) -> None:
        self._popen = popen
        self._terminals: dict[str, _TerminalRecord] = {}
        self._lock = threading.Lock()

    def create(
        self,
        command: str,
        *,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: list[EnvVariable] | None = None,
        output_byte_limit: int | None = None,
    ) -> str:
        merged_env = os.environ.copy()
        for var in env or []:
            merged_env[var.name] = var.value
        process = self._popen(
            [command, *(args or [])],
            cwd=cwd,
            env=merged_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        terminal_id = str(uuid.uuid4())
        record = _TerminalRecord(process=process, output_limit=output_byte_limit)
        readers = tuple(
            threading.Thread(
                target=self._read_stream,
                args=(record, stream),
                daemon=True,
            )
            for stream in (process.stdout, process.stderr)
            if stream is not None
        )
        record.readers = readers
        with self._lock:
            self._terminals[terminal_id] = record
        for reader in readers:
            reader.start()
        return terminal_id

    def _read_stream(self, record: _TerminalRecord, stream: Any) -> None:  # noqa: ANN401  # asyncio stream protocol
        try:
            for chunk in iter(lambda: stream.read(4096), ""):
                if not chunk:
                    break
                record.append(chunk)
        finally:
            stream.close()

    def output(self, terminal_id: str) -> tuple[str, bool, int | None, str | None]:
        return self._terminal(terminal_id).output()

    def wait(self, terminal_id: str) -> tuple[int | None, str | None]:
        process = self._terminal(terminal_id).process
        returncode = process.wait()
        if returncode < 0:
            return None, signal.Signals(-returncode).name
        return returncode, None

    def kill(self, terminal_id: str) -> None:
        process = self._terminal(terminal_id).process
        if process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def release(self, terminal_id: str) -> None:
        with self._lock:
            record = self._terminals.pop(terminal_id)
        if record.process.poll() is None:
            record.process.kill()
            try:
                record.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                record.process.kill()
                record.process.wait()

    def _terminal(self, terminal_id: str) -> _TerminalRecord:
        with self._lock:
            return self._terminals[terminal_id]


class _CopilotACPClient:
    def __init__(
        self,
        runtime: "CopilotACPRuntime",
        *,
        terminals: _TerminalManager | None = None,
    ) -> None:
        self._runtime = runtime
        self._terminals = _TerminalManager() if terminals is None else terminals

    def on_connect(self, conn: acp.Agent) -> None:
        del conn
        self._runtime.log_info("copilot system: connected")

    async def read_text_file(
        self,
        path: str,
        session_id: str,
        limit: int | None = None,
        line: int | None = None,
        **kwargs: Any,  # noqa: ANN401  # session-update JSON pass-through,
    ) -> acp.ReadTextFileResponse:
        del session_id, kwargs
        text = Path(path).read_text()
        if line is not None:
            text = "".join(text.splitlines(keepends=True)[max(line - 1, 0) :])
        if limit is not None:
            text = text[:limit]
        return acp.ReadTextFileResponse(content=text)

    async def write_text_file(
        self,
        content: str,
        path: str,
        session_id: str,
        **kwargs: Any,  # noqa: ANN401  # session-update JSON pass-through,
    ) -> acp.WriteTextFileResponse:
        del session_id, kwargs
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return acp.WriteTextFileResponse()

    async def create_terminal(
        self,
        command: str,
        session_id: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: list[EnvVariable] | None = None,
        output_byte_limit: int | None = None,
        **kwargs: Any,  # noqa: ANN401  # session-update JSON pass-through,
    ) -> acp.CreateTerminalResponse:
        del session_id, kwargs
        terminal_id = self._terminals.create(
            command,
            args=args,
            cwd=cwd,
            env=env,
            output_byte_limit=output_byte_limit,
        )
        return acp.CreateTerminalResponse(terminal_id=terminal_id)

    async def terminal_output(
        self,
        session_id: str,
        terminal_id: str,
        **kwargs: Any,  # noqa: ANN401  # session-update JSON pass-through,
    ) -> acp.TerminalOutputResponse:
        del session_id, kwargs
        output, truncated, exit_code, signal_name = self._terminals.output(terminal_id)
        exit_status = None
        if exit_code is not None or signal_name is not None:
            exit_status = TerminalExitStatus(
                exit_code=exit_code,
                signal=signal_name,
            )
        return acp.TerminalOutputResponse(
            output=output,
            truncated=truncated,
            exit_status=exit_status,
        )

    async def wait_for_terminal_exit(
        self,
        session_id: str,
        terminal_id: str,
        **kwargs: Any,  # noqa: ANN401  # session-update JSON pass-through,
    ) -> acp.WaitForTerminalExitResponse:
        del session_id, kwargs
        exit_code, signal_name = await asyncio.to_thread(
            self._terminals.wait, terminal_id
        )
        return acp.WaitForTerminalExitResponse(
            exit_code=exit_code,
            signal=signal_name,
        )

    async def kill_terminal(
        self,
        session_id: str,
        terminal_id: str,
        **kwargs: Any,  # noqa: ANN401  # session-update JSON pass-through,
    ) -> acp.KillTerminalResponse:
        del session_id, kwargs
        await asyncio.to_thread(self._terminals.kill, terminal_id)
        return acp.KillTerminalResponse()

    async def release_terminal(
        self,
        session_id: str,
        terminal_id: str,
        **kwargs: Any,  # noqa: ANN401  # session-update JSON pass-through,
    ) -> acp.ReleaseTerminalResponse:
        del session_id, kwargs
        await asyncio.to_thread(self._terminals.release, terminal_id)
        return acp.ReleaseTerminalResponse()

    async def request_permission(
        self,
        options: list[PermissionOption],
        session_id: str,
        tool_call: ToolCallUpdate,
        **kwargs: Any,  # noqa: ANN401  # session-update JSON pass-through,
    ) -> acp.RequestPermissionResponse:
        del session_id, tool_call, kwargs
        for option in options:
            if option.kind.startswith("allow"):
                return acp.RequestPermissionResponse(
                    outcome=AllowedOutcome(
                        option_id=option.option_id, outcome="selected"
                    )
                )
        return acp.RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    async def session_update(
        self,
        session_id: str,
        update: dict[str, Any],
        **kwargs: Any,  # noqa: ANN401  # session-update JSON pass-through,
    ) -> None:
        del kwargs
        self._runtime.record_session_update(session_id, update)


class CopilotACPRuntime:
    """Own the long-lived Copilot ACP subprocess and connection."""

    def __init__(
        self,
        *,
        work_dir: Path,
        repo_name: str | None = None,
        command: Sequence[str] = _COPILOT_COMMAND,
        spawn_agent_process: Callable[..., AbstractAsyncContextManager[Any]] = (
            acp.spawn_agent_process
        ),
        client_factory: Callable[["CopilotACPRuntime"], _CopilotACPClient]
        | None = None,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
    ) -> None:
        self._work_dir = work_dir
        self._repo_name = repo_name
        self._command_base = tuple(command)
        self._spawn_agent_process = spawn_agent_process
        self._client_factory = (
            self._default_client_factory if client_factory is None else client_factory
        )
        self._client_capabilities = (
            ClientCapabilities(
                fs=FileSystemCapabilities(read_text_file=True, write_text_file=True),
                terminal=True,
            )
            if client_capabilities is None
            else client_capabilities
        )
        self._client_info = (
            Implementation(name="fido", title="fido", version="0.1.0")
            if client_info is None
            else client_info
        )
        self._loop = asyncio.new_event_loop()
        self._loop_ready = threading.Event()
        self._stopped = False
        self._agent_cm: AbstractAsyncContextManager[Any] | None = None
        self._client: _CopilotACPClient | None = None
        self._connection: Any = None
        self._process: Any = None
        self._initialize_response: Any = None
        self._attached_session_id: str | None = None
        self._current_effort: ReasoningEffort | None = None
        self._active_prompt_session_id: str | None = None
        self._prompt_chunks: list[str] = []
        self._tool_starts_logged: set[str] = set()
        self._tool_results_logged: set[str] = set()
        self._metrics_lock = threading.Lock()
        self._dropped_session_count = 0
        self._needs_session_recovery_nudge = False
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"copilot-acp-{work_dir.name}",
            daemon=True,
        )
        self._thread.start()
        self._loop_ready.wait()

    def _default_client_factory(
        self, runtime: "CopilotACPRuntime"
    ) -> _CopilotACPClient:
        return _CopilotACPClient(runtime)

    def log_info(self, message: str, *args: object) -> None:
        _log_for_repo(logging.INFO, self._repo_name, message, *args)

    def log_warning(self, message: str, *args: object) -> None:
        _log_for_repo(logging.WARNING, self._repo_name, message, *args)

    def _command_for_effort(self, effort: ReasoningEffort | None) -> tuple[str, ...]:
        if effort is None:
            return self._command_base
        return (*self._command_base, "--effort", effort)

    def _preferred_efforts(
        self, model: ProviderModel | None
    ) -> tuple[ReasoningEffort | None, ...]:
        if model is None or not model.efforts:
            return (None,)
        return tuple(model.efforts)

    @property
    def pid(self) -> int | None:
        process = self._process
        return process.pid if process is not None else None

    def is_alive(self) -> bool:
        process = self._process
        return process is not None and process.returncode is None

    @property
    def dropped_session_count(self) -> int:
        with self._metrics_lock:
            return self._dropped_session_count

    def ensure_session(
        self, session_id: str | None, model: ProviderModel | str | None
    ) -> str:
        return self._run_async(self._ensure_session_async(session_id, model))

    def recover_session(
        self, session_id: str | None, model: ProviderModel | str | None
    ) -> str:
        return self._run_async(self._recover_session_async(session_id, model))

    def reset_session(self, model: ProviderModel | str | None) -> str:
        return self._run_async(self._reset_session_async(model))

    def prompt(
        self, session_id: str, content: str, model: ProviderModel | str | None
    ) -> tuple[str, str, str]:
        return self._run_async(self._prompt_async(session_id, content, model))

    def cancel(self, session_id: str) -> None:
        self._run_async(self._cancel_async(session_id))

    def stop(self) -> None:
        if self._stopped:
            return
        self._run_async(self._close_connection_async())
        self._stopped = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)

    def record_session_update(self, session_id: str, update: dict[str, Any]) -> None:
        if session_id != self._active_prompt_session_id:
            return
        update_type = getattr(update, "session_update", "")
        if update_type == "agent_message_chunk":
            content = getattr(update, "content", None)
            text = getattr(content, "text", None)
            if isinstance(text, str):
                self._prompt_chunks.append(text)
            return
        if update_type == "tool_call":
            self._log_tool_call(update)
            return
        if update_type == "tool_call_update":
            self._log_tool_result(update)

    def _log_tool_call(self, update: dict[str, Any]) -> None:
        tool_call_id = getattr(update, "tool_call_id", None)
        if (
            not isinstance(tool_call_id, str)
            or tool_call_id in self._tool_starts_logged
        ):
            return
        self._tool_starts_logged.add(tool_call_id)
        title = str(
            getattr(update, "title", None) or getattr(update, "kind", None) or "?"
        )
        preview = _tool_input_preview(getattr(update, "raw_input", None))
        if preview:
            self.log_info("copilot tool: %s — %s", title, preview)
            return
        self.log_info("copilot tool: %s", title)

    def _log_tool_result(self, update: dict[str, Any]) -> None:
        tool_call_id = getattr(update, "tool_call_id", None)
        if (
            not isinstance(tool_call_id, str)
            or tool_call_id in self._tool_results_logged
        ):
            return
        status = getattr(update, "status", None)
        raw_output = getattr(update, "raw_output", None)
        if raw_output is None and status not in {"completed", "failed"}:
            return
        self._tool_results_logged.add(tool_call_id)
        title = str(
            getattr(update, "title", None) or getattr(update, "kind", None) or "?"
        )
        preview = _preview_log_value(raw_output)
        if status == "failed":
            if preview:
                self.log_warning("copilot tool failed: %s — %s", title, preview)
                return
            self.log_warning("copilot tool failed: %s", title)
            return
        if preview:
            self.log_info("copilot tool result: %s — %s", title, preview)
            return
        self.log_info("copilot tool result: %s", title)

    async def _ensure_session_async(
        self, session_id: str | None, model: ProviderModel | str | None
    ) -> str:
        normalized = _normalize_model(model)
        await self._ensure_connection_async(normalized)
        target_session_id = session_id
        if target_session_id is None or self._attached_session_id != target_session_id:
            target_session_id = await self._attach_session_async(target_session_id)
        await self._set_model_async(target_session_id, normalized)
        return target_session_id

    async def _recover_session_async(
        self, session_id: str | None, model: ProviderModel | str | None
    ) -> str:
        await self._close_connection_async()
        return await self._ensure_session_async(session_id, model)

    async def _reset_session_async(self, model: ProviderModel | str | None) -> str:
        normalized = _normalize_model(model)
        await self._ensure_connection_async(normalized)
        self._needs_session_recovery_nudge = False
        target_session_id = await self._attach_session_async(None)
        await self._set_model_async(target_session_id, normalized)
        return target_session_id

    async def _prompt_async(
        self, session_id: str, content: str, model: ProviderModel | str | None
    ) -> tuple[str, str, str]:
        target_session_id = await self._ensure_session_async(session_id, model)
        connection = self._connection
        if connection is None:
            raise RuntimeError("Copilot ACP connection is not available")
        prompt_content = (
            _unexpected_new_session_prompt(content)
            if self._needs_session_recovery_nudge
            else content
        )
        self._needs_session_recovery_nudge = False
        self._active_prompt_session_id = target_session_id
        self._prompt_chunks = []
        self._tool_starts_logged = set()
        self._tool_results_logged = set()
        response = await connection.prompt(
            prompt=[acp.text_block(prompt_content)],
            session_id=target_session_id,
        )
        text = "".join(self._prompt_chunks)
        self._active_prompt_session_id = None
        self._prompt_chunks = []
        self._tool_starts_logged = set()
        self._tool_results_logged = set()
        return text, response.stop_reason, target_session_id

    async def _cancel_async(self, session_id: str) -> None:
        connection = self._connection
        if connection is None:
            return
        await connection.cancel(session_id=session_id)

    async def _ensure_connection_async(self, model: ProviderModel | None) -> None:
        allowed_efforts = self._preferred_efforts(model)
        if (
            self.is_alive()
            and self._connection is not None
            and self._current_effort in allowed_efforts
        ):
            return
        last_error: Exception | None = None
        for effort in allowed_efforts:
            await self._close_connection_async()
            client = self._client_factory(self)
            command = self._command_for_effort(effort)
            agent_cm = self._spawn_agent_process(
                client,
                command[0],
                *command[1:],
                cwd=self._work_dir,
                transport_kwargs={
                    "stderr": subprocess.DEVNULL,
                    "limit": _ACP_STREAM_LIMIT,
                },
            )
            try:
                connection, process = await agent_cm.__aenter__()
                initialize_response = await connection.initialize(
                    protocol_version=acp.PROTOCOL_VERSION,
                    client_capabilities=self._client_capabilities,
                    client_info=self._client_info,
                )
            except Exception as exc:
                await agent_cm.__aexit__(None, None, None)
                last_error = exc
                continue
            self._agent_cm = agent_cm
            self._client = client
            self._connection = connection
            self._process = process
            self._initialize_response = initialize_response
            self._attached_session_id = None
            self._current_effort = effort
            return
        if last_error is not None:
            raise last_error

    async def _attach_session_async(self, session_id: str | None) -> str:
        connection = self._connection
        if connection is None:
            raise RuntimeError("Copilot ACP connection is not available")
        if session_id is None:
            response = await connection.new_session(cwd=str(self._work_dir))
            self._attached_session_id = response.session_id
            return response.session_id
        try:
            if self._supports_load_session():
                await connection.load_session(
                    cwd=str(self._work_dir), session_id=session_id
                )
            else:
                await connection.resume_session(
                    cwd=str(self._work_dir), session_id=session_id
                )
        except RequestError as exc:
            if not _is_missing_session_error(exc):
                raise
            self.log_warning(
                "copilot session dropped: %s not found — starting fresh",
                session_id,
            )
            with self._metrics_lock:
                self._dropped_session_count += 1
            self._needs_session_recovery_nudge = True
            response = await connection.new_session(cwd=str(self._work_dir))
            self._attached_session_id = response.session_id
            return response.session_id
        self._attached_session_id = session_id
        return session_id

    async def _set_model_async(
        self, session_id: str, model: ProviderModel | str | None
    ) -> None:
        normalized = _normalize_model(model)
        if normalized is None:
            return
        connection = self._connection
        if connection is None:
            raise RuntimeError("Copilot ACP connection is not available")
        await connection.set_session_model(
            model_id=normalized.model, session_id=session_id
        )

    async def _close_connection_async(self) -> None:
        agent_cm = self._agent_cm
        self._agent_cm = None
        self._client = None
        self._connection = None
        self._process = None
        self._initialize_response = None
        self._attached_session_id = None
        self._current_effort = None
        self._active_prompt_session_id = None
        self._prompt_chunks = []
        self._tool_starts_logged = set()
        self._tool_results_logged = set()
        if agent_cm is not None:
            await agent_cm.__aexit__(None, None, None)

    def _supports_load_session(self) -> bool:
        response = self._initialize_response
        if response is None:
            return False
        capabilities = getattr(response, "agent_capabilities", None)
        if capabilities is None:
            return False
        return bool(getattr(capabilities, "load_session", False))

    def _run_async(self, coro: Coroutine[object, object, _T]) -> _T:
        if self._stopped:
            raise RuntimeError("Copilot ACP runtime is stopped")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop_ready.set()
        self._loop.run_forever()
        pending = asyncio.all_tasks(self._loop)
        for task in pending:
            task.cancel()
        if pending:
            self._loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
        self._loop.close()


class CopilotCLISession(OwnedSession):
    """Persistent Copilot CLI ACP session."""

    def __init__(
        self,
        system_file: Path,
        *,
        work_dir: Path | str,
        model: ProviderModel | str,
        repo_name: str | None = None,
        runtime: CopilotACPRuntime | None = None,
        runtime_factory: Callable[..., CopilotACPRuntime] | None = None,
        session_id: str | None = None,
    ) -> None:
        self._work_dir = Path(work_dir)
        self._repo_name = repo_name
        try:
            self._base_system_prompt = system_file.read_text()
        except OSError:
            self._base_system_prompt = ""
        if runtime is not None:
            self._runtime = runtime
        else:
            factory = CopilotACPRuntime if runtime_factory is None else runtime_factory
            self._runtime = factory(work_dir=self._work_dir, repo_name=repo_name)
        self._init_handler_reentry()
        self._pending_content: str | None = None
        self._last_turn_cancelled = False
        self._model = coerce_provider_model(model)
        # _metrics_lock guards both counters: they are written by the worker
        # thread (send / prompt) and read from other threads (status,
        # registry).  Python 3.14t has no GIL, so += is not atomic.
        self._metrics_lock = threading.Lock()
        self._sent_count: int = 0
        self._received_count: int = 0
        # Resume an existing Copilot ACP session when *session_id* is given
        # (fix for #649 — persisted across fido restarts in state.json).
        # When the id is no longer known to Copilot, ensure_session falls
        # back to creating a fresh session automatically.
        self._session_id: str | None = self._runtime.ensure_session(session_id, model)

    @property
    def owner(self) -> str | None:
        if self._repo_name is None:
            return None
        talker = provider.get_talker(self._repo_name)
        if talker is None or talker.kind != "worker":
            return None
        for t in threading.enumerate():
            if t.ident == talker.thread_id:
                return t.name
        return None

    @property
    def pid(self) -> int | None:
        return self._runtime.pid

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def dropped_session_count(self) -> int:
        return self._runtime.dropped_session_count

    @property
    def sent_count(self) -> int:
        """Cumulative number of user turns sent to Copilot since boot.

        Accumulates across session switches and recoveries — model changes and
        resets do not reset the count.
        """
        with self._metrics_lock:
            return self._sent_count

    @property
    def received_count(self) -> int:
        """Cumulative number of responses received from Copilot since boot.

        Accumulates across session switches and recoveries — model changes and
        resets do not reset the count.
        """
        with self._metrics_lock:
            return self._received_count

    @property
    def last_turn_cancelled(self) -> bool:
        return self._last_turn_cancelled

    def prompt(
        self,
        content: str,
        *,
        model: ProviderModel | None = None,
        allowed_tools: str | None = None,  # see comment below
        system_prompt: str | None = None,
    ) -> str:
        # ``allowed_tools`` is part of the ``PromptSession`` protocol (closes
        # #1413), but Copilot CLI's ACP runtime has no equivalent of
        # ``--allowedTools`` so the kwarg is informational here.  Default
        # differs from the protocol's READ_ONLY default because there's
        # nothing to enforce.
        del allowed_tools
        with self:
            return self._prompt_locked(
                content, model=model, system_prompt=system_prompt
            )

    def send(self, content: str) -> None:
        self._pending_content = content

    def consume_until_result(self) -> str:
        content = self._pending_content
        self._pending_content = None
        if content is None:
            return ""
        return self._prompt_locked(content, model=self._model, system_prompt=None)

    def switch_model(self, model: ProviderModel | str) -> None:
        normalized = coerce_provider_model(model)
        self._session_id = self._runtime.ensure_session(self._session_id, normalized)
        self._model = normalized

    def recover(self) -> None:
        self._session_id = self._runtime.recover_session(self._session_id, self._model)

    def reset(self, model: ProviderModel | None = None) -> None:
        effective_model = self._model if model is None else coerce_provider_model(model)
        self._session_id = self._runtime.reset_session(effective_model)
        if model is not None:
            self._model = coerce_provider_model(model)
        self._pending_content = None
        self._last_turn_cancelled = False

    def is_alive(self) -> bool:
        return self._runtime.is_alive()

    def stop(self) -> None:
        self._runtime.stop()

    def _fire_worker_cancel(self) -> None:
        """Provider-specific cancel mechanism used by
        :meth:`~fido.provider.OwnedSession.preempt_worker` and
        :meth:`~fido.provider.OwnedSession.hold_for_handler`.  Asks the Copilot
        ACP runtime to cancel the currently-active prompt so the worker's turn
        returns with ``stop_reason="cancelled"`` and releases the session lock.
        No-op if there is no active session id or the runtime is already down.
        """
        session_id = self._session_id
        if session_id is not None and self._runtime.is_alive():
            self._runtime.cancel(session_id)

    def __enter__(self) -> "CopilotCLISession":
        """Acquire the session lock, serializing prompt calls across threads.

        On the outermost entry (via the :class:`OwnedSession` reentrance
        counter), delegates to :meth:`_fsm_acquire_worker` or
        :meth:`_fsm_acquire_handler` based on the calling thread's kind
        (read from :func:`provider.current_thread_kind`).  Handler acquires
        queue behind any current holder and are served FIFO; worker acquires
        yield to any queued handler.

        Registers a :class:`provider.SessionTalker` after acquiring.
        Nested entries (from :meth:`hold_for_handler`) re-enter the
        reentrance counter and skip the FSM acquire.

        Raises :class:`provider.SessionLeakError` on the outermost entry if
        another thread is already registered as the talker for this repo.  The
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
            # session fires the runtime cancel before queueing on the FSM,
            # so the worker's turn aborts and releases promptly rather than
            # being waited out (#637).  Webhook-on-webhook still queues
            # FIFO with no cancel.
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
                        description="copilot-cli session turn",
                        claude_pid=0,  # no claude subprocess — ACP runtime
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

    def _prompt_locked(
        self,
        content: str,
        *,
        model: ProviderModel | str | None,
        system_prompt: str | None,
    ) -> str:
        effective_model = self._model if model is None else coerce_provider_model(model)
        prompt = _combine_prompt(
            content,
            base_system_prompt=self._base_system_prompt,
            system_prompt=system_prompt,
        )
        _log_for_repo(
            logging.INFO,
            self._repo_name,
            "%s",
            _transcript_block("copilot prompt", prompt),
        )
        with self._metrics_lock:
            self._sent_count += 1
        result, stop_reason, session_id = self._runtime.prompt(
            self._session_id or "",
            prompt,
            effective_model,
        )
        with self._metrics_lock:
            self._received_count += 1
        self._session_id = session_id
        if model is not None:
            self._model = coerce_provider_model(model)
        cancelled = stop_reason == "cancelled" or _is_cancel_sentinel(result)
        self._last_turn_cancelled = cancelled
        _log_for_repo(
            logging.INFO,
            self._repo_name,
            "%s",
            _transcript_block("copilot result", result),
        )
        # #666: Copilot CLI emits "Info: Operation cancelled by user" as its
        # final assistant content when a turn is cancelled mid-flight.  If a
        # caller (e.g. events.post reply paths) forwards that string as a
        # generated body, it ends up posted on a PR as if it were real text.
        # Normalize to empty at the talker boundary so every consumer's
        # existing ``if not body`` guard treats it correctly.
        if cancelled:
            return ""
        return result


class CopilotCLIAPI(ProviderAPI):
    """Read-only account API for Copilot CLI.

    Copilot CLI exposes no native quota endpoint, so limit state is derived
    from observed errors: when a caller records a quota-shaped error via
    :meth:`record_quota_error`, :meth:`get_limit_snapshot` returns a 100%-
    usage window that expires after :data:`_COPILOT_QUOTA_PAUSE_SECONDS`.
    Once the pause window expires the snapshot reverts to "unknown".
    """

    def __init__(
        self,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        pause_seconds: float = _COPILOT_QUOTA_PAUSE_SECONDS,
    ) -> None:
        self._monotonic = monotonic
        self._now = now
        self._pause_seconds = pause_seconds
        self._lock = threading.Lock()
        self._quota_error_at_monotonic: float | None = None
        self._quota_error_at_wall: datetime | None = None

    @property
    def provider_id(self) -> ProviderID:
        return ProviderID.COPILOT_CLI

    def record_quota_error(self, exc: Exception) -> bool:
        """Record *exc* as a quota error if it looks quota-shaped.

        Returns ``True`` when the error was classified as quota-related and
        the pause window was (re)started, ``False`` otherwise.
        """
        if not _is_copilot_quota_error(exc):
            return False
        with self._lock:
            self._quota_error_at_monotonic = self._monotonic()
            self._quota_error_at_wall = self._now()
        log.warning(
            "copilot-cli quota error recorded — pausing for %.0fs", self._pause_seconds
        )
        return True

    def get_limit_snapshot(self) -> ProviderLimitSnapshot:
        with self._lock:
            recorded_at = self._quota_error_at_monotonic
            wall = self._quota_error_at_wall
        if recorded_at is None or wall is None:
            return ProviderLimitSnapshot(provider=self.provider_id)
        elapsed = self._monotonic() - recorded_at
        if elapsed >= self._pause_seconds:
            return ProviderLimitSnapshot(provider=self.provider_id)
        resets_at = wall + timedelta(seconds=self._pause_seconds)
        return ProviderLimitSnapshot(
            provider=self.provider_id,
            windows=(
                ProviderLimitWindow(
                    name="quota",
                    used=100,
                    limit=100,
                    resets_at=resets_at,
                    unit="%",
                ),
            ),
        )


class CopilotCLIClient(SessionBackedAgent, ProviderAgent):
    """Injectable collaborator for Copilot CLI interactions."""

    voice_model = ProviderModel("gpt-4.1", "high")
    work_model = ProviderModel("gpt-4.1", "medium")
    brief_model = ProviderModel("gpt-5-mini", "low")

    def __init__(
        self,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        session_fn: Callable[[], PromptSession] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        session_factory: Callable[..., PromptSession] | None = None,
        session_system_file: Path | None = None,
        work_dir: Path | str | None = None,
        repo_name: str | None = None,
        session: PromptSession | None = None,
        api: CopilotCLIAPI | None = None,
    ) -> None:
        self._runner = runner
        self._sleep_fn = sleep_fn
        self._quota_api = api
        self._session_factory = (
            CopilotCLISession if session_factory is None else session_factory
        )
        super().__init__(
            session_fn=provider.current_repo_session
            if session_fn is None
            else session_fn,
            session_system_file=session_system_file,
            work_dir=work_dir,
            repo_name=repo_name,
            session=session,
        )

    @property
    def provider_id(self) -> ProviderID:
        return ProviderID.COPILOT_CLI

    @property
    def supports_no_commit_reset(self) -> bool:
        return False

    def _spawn_owned_session(
        self, model: ProviderModel, *, session_id: str | None = None
    ) -> PromptSession:
        system_file = self._session_system_file
        work_dir = self._work_dir
        assert system_file is not None
        assert work_dir is not None
        return self._session_factory(
            system_file,
            work_dir=work_dir,
            model=model,
            repo_name=self._repo_name,
            session_id=session_id,
        )

    def _should_retry_prompt_failure(
        self,
        exc: Exception,
        session: PromptSession,
    ) -> bool:
        # Quota errors are not recoverable via session restart — record and halt.
        if self._quota_api is not None and self._quota_api.record_quota_error(exc):
            return False
        message = str(exc)
        return (
            isinstance(exc, (BrokenPipeError, OSError))
            or _is_line_limit_overrun_error(exc)
            or message == "Copilot ACP connection is not available"
            or (
                message != "Copilot ACP runtime is stopped"
                and self._session_is_dead(session)
            )
        )

    def _dead_prompt_error_message(self) -> str:
        return "Copilot CLI session died during prompt"

    def print_prompt_from_file(
        self,
        system_file: Path,
        prompt_file: Path,
        model: ProviderModel,
        timeout: int = 30,
        idle_timeout: float = 1800.0,
        cwd: Path | str = ".",
        *,
        allowed_tools: str | None = None,  # informational; Copilot has no equivalent
    ) -> str:
        del idle_timeout, allowed_tools
        prompt = _combine_prompt(
            prompt_file.read_text(),
            base_system_prompt=system_file.read_text(),
        )
        return self._run_cli_prompt(prompt, model=model, timeout=timeout, cwd=cwd)

    def resume_session(
        self,
        session_id: str,
        prompt_file: Path,
        model: ProviderModel,
        timeout: int = 300,
        idle_timeout: float = 1800.0,
        cwd: Path | str = ".",
        *,
        allowed_tools: str | None = None,  # informational; Copilot has no equivalent
    ) -> str:
        del idle_timeout, allowed_tools
        return self._run_cli_prompt(
            prompt_file.read_text(),
            model=model,
            timeout=timeout,
            cwd=cwd,
            session_id=session_id,
        )

    def extract_session_id(self, output: str) -> str:
        return extract_session_id(output)

    def _run_cli_prompt(
        self,
        prompt: str,
        *,
        model: ProviderModel | str,
        timeout: int,
        cwd: Path | str | None = None,
        session_id: str | None = None,
    ) -> str:
        normalized = _normalize_model(model)
        assert normalized is not None
        _log_for_repo(
            logging.INFO,
            self._repo_name,
            "%s",
            _transcript_block("copilot prompt", prompt),
        )
        efforts = normalized.efforts or (None,)
        for effort in efforts:
            cmd = ["--model", normalized.model]
            if effort is not None:
                cmd += ["--effort", effort]
            cmd += [*_COPILOT_JSON_BASE_ARGS]
            if session_id is not None:
                cmd.append(f"--resume={session_id}")
            cmd += ["-p", prompt]
            result = _copilot(
                *cmd,
                timeout=timeout,
                cwd=self._work_dir if cwd is None else cwd,
                runner=self._runner,
            )
            if result.returncode == 0:
                text = extract_result_text(result.stdout.strip())
                _log_for_repo(
                    logging.INFO,
                    self._repo_name,
                    "%s",
                    _transcript_block("copilot result", text),
                )
                return result.stdout.strip()
        return ""


class CopilotCLI(Provider):
    """Composite Copilot provider with separate API and runtime agent."""

    def __init__(
        self,
        *,
        api: CopilotCLIAPI | None = None,
        agent: ProviderAgent | None = None,
        session: PromptSession | None = None,
    ) -> None:
        api_instance = api if api is not None else CopilotCLIAPI()
        if agent is None:
            agent = CopilotCLIClient(session=session, api=api_instance)
        elif session is not None:
            agent.attach_session(session)
        self._api = api_instance
        self._agent = agent

    @property
    def provider_id(self) -> ProviderID:
        return ProviderID.COPILOT_CLI

    @property
    def api(self) -> ProviderAPI:
        return self._api

    @property
    def agent(self) -> ProviderAgent:
        return self._agent
