"""Copilot CLI provider implementation."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import acp
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

from kennel import claude
from kennel.provider import (
    PromptSession,
    Provider,
    ProviderAgent,
    ProviderAPI,
    ProviderID,
    ProviderLimitSnapshot,
    ProviderModel,
    ReasoningEffort,
    coerce_provider_model,
)
from kennel.session_agent import SessionBackedAgent

log = logging.getLogger(__name__)

_COPILOT_COMMAND = ("copilot", "--acp", "--allow-all")
_COPILOT_JSON_BASE_ARGS = (
    "--output-format",
    "json",
    "--stream",
    "off",
    "--allow-all",
    "-s",
)
_COPILOT_LIMITS_UNAVAILABLE = "Copilot CLI does not expose local usage limits."


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


def _normalize_model(model: ProviderModel | str | None) -> ProviderModel | None:
    if model is None:
        return None
    normalized = coerce_provider_model(model)
    lowered = normalized.model.lower()
    if lowered.startswith("claude-opus"):
        return ProviderModel("gpt-5.4", normalized.effort)
    if lowered.startswith("claude-sonnet"):
        return ProviderModel("gpt-5.4", normalized.effort)
    if lowered.startswith("claude-haiku"):
        return ProviderModel("gpt-5.4", normalized.effort)
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

    def _read_stream(self, record: _TerminalRecord, stream: Any) -> None:
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

    async def on_connect(self, conn: acp.Agent) -> None:
        return None

    async def read_text_file(
        self,
        path: str,
        session_id: str,
        limit: int | None = None,
        line: int | None = None,
        **kwargs: Any,
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
        **kwargs: Any,
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
        **kwargs: Any,
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
        **kwargs: Any,
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
        **kwargs: Any,
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
        **kwargs: Any,
    ) -> acp.KillTerminalResponse:
        del session_id, kwargs
        await asyncio.to_thread(self._terminals.kill, terminal_id)
        return acp.KillTerminalResponse()

    async def release_terminal(
        self,
        session_id: str,
        terminal_id: str,
        **kwargs: Any,
    ) -> acp.ReleaseTerminalResponse:
        del session_id, kwargs
        await asyncio.to_thread(self._terminals.release, terminal_id)
        return acp.ReleaseTerminalResponse()

    async def request_permission(
        self,
        options: list[PermissionOption],
        session_id: str,
        tool_call: ToolCallUpdate,
        **kwargs: Any,
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
        update: Any,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self._runtime.record_session_update(session_id, update)


class CopilotACPRuntime:
    """Own the long-lived Copilot ACP subprocess and connection."""

    def __init__(
        self,
        *,
        work_dir: Path,
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
            Implementation(name="kennel", title="kennel", version="0.1.0")
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

    def record_session_update(self, session_id: str, update: Any) -> None:
        if session_id != self._active_prompt_session_id:
            return
        if getattr(update, "session_update", "") != "agent_message_chunk":
            return
        content = getattr(update, "content", None)
        text = getattr(content, "text", None)
        if isinstance(text, str):
            self._prompt_chunks.append(text)

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
        self._active_prompt_session_id = target_session_id
        self._prompt_chunks = []
        response = await connection.prompt(
            prompt=[acp.text_block(content)],
            session_id=target_session_id,
        )
        text = "".join(self._prompt_chunks)
        self._active_prompt_session_id = None
        self._prompt_chunks = []
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
                transport_kwargs={"stderr": subprocess.DEVNULL},
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
        if self._supports_load_session():
            await connection.load_session(
                cwd=str(self._work_dir), session_id=session_id
            )
        else:
            await connection.resume_session(
                cwd=str(self._work_dir), session_id=session_id
            )
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

    def _run_async(self, coro: Any) -> Any:
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


class CopilotCLISession:
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
            self._runtime = factory(work_dir=self._work_dir)
        self._lock = threading.Lock()
        self._owner_lock = threading.Lock()
        self._owner: str | None = None
        self._pending_preempts = 0
        self._preempt_condition = threading.Condition()
        self._thread_state = threading.local()
        self._pending_content: str | None = None
        self._last_turn_cancelled = False
        self._model = coerce_provider_model(model)
        self._session_id: str | None = self._runtime.ensure_session(None, model)

    @property
    def owner(self) -> str | None:
        with self._owner_lock:
            return self._owner

    @property
    def pid(self) -> int | None:
        return self._runtime.pid

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def last_turn_cancelled(self) -> bool:
        return self._last_turn_cancelled

    def wait_for_pending_preempt(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._preempt_condition:
            while self._pending_preempts > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._preempt_condition.wait(timeout=remaining)
        return True

    def prompt(
        self,
        content: str,
        model: ProviderModel | None = None,
        system_prompt: str | None = None,
    ) -> str:
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

    def __enter__(self) -> "CopilotCLISession":
        waited = not self._lock.acquire(blocking=False)
        if waited:
            with self._preempt_condition:
                self._pending_preempts += 1
            session_id = self._session_id
            if session_id is not None and self._runtime.is_alive():
                self._runtime.cancel(session_id)
            self._lock.acquire()
        self._thread_state.waited = waited
        with self._owner_lock:
            self._owner = threading.current_thread().name
        return self

    def __exit__(self, *args: object) -> None:
        with self._owner_lock:
            self._owner = None
        self._lock.release()
        if getattr(self._thread_state, "waited", False):
            with self._preempt_condition:
                self._pending_preempts -= 1
                if self._pending_preempts == 0:
                    self._preempt_condition.notify_all()
            self._thread_state.waited = False

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
        result, stop_reason, session_id = self._runtime.prompt(
            self._session_id or "",
            prompt,
            effective_model,
        )
        self._session_id = session_id
        if model is not None:
            self._model = coerce_provider_model(model)
        self._last_turn_cancelled = stop_reason == "cancelled"
        return result


class CopilotCLIAPI(ProviderAPI):
    """Read-only account API for Copilot CLI."""

    @property
    def provider_id(self) -> ProviderID:
        return ProviderID.COPILOT_CLI

    def get_limit_snapshot(self) -> ProviderLimitSnapshot:
        return ProviderLimitSnapshot(
            provider=self.provider_id,
            unavailable_reason=_COPILOT_LIMITS_UNAVAILABLE,
        )


class CopilotCLIClient(SessionBackedAgent, ProviderAgent):
    """Injectable collaborator for Copilot CLI interactions."""

    voice_model = ProviderModel("gpt-5.4", "high")
    work_model = ProviderModel("gpt-5.4", "medium")
    brief_model = ProviderModel("gpt-5.4", "low")

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
    ) -> None:
        self._runner = runner
        self._sleep_fn = sleep_fn
        self._session_factory = (
            CopilotCLISession if session_factory is None else session_factory
        )
        super().__init__(
            session_fn=claude.current_repo_session
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

    def _spawn_owned_session(self, model: ProviderModel) -> PromptSession:
        system_file = self._session_system_file
        work_dir = self._work_dir
        assert system_file is not None
        assert work_dir is not None
        return self._session_factory(
            system_file,
            work_dir=work_dir,
            model=model,
            repo_name=self._repo_name,
        )

    def _session_is_dead(self, session: PromptSession) -> bool:
        return session.is_alive() is False

    def _recover_prompt_session(self, session: PromptSession) -> bool:
        recover = getattr(session, "recover", None)
        if not callable(recover):
            return False
        recover()
        return True

    def _prompt_with_recovery(
        self,
        session: PromptSession,
        content: str,
        *,
        model: ProviderModel | None,
        system_prompt: str | None,
    ) -> str:
        recovered = False
        while True:
            try:
                result = session.prompt(
                    content, model=model, system_prompt=system_prompt
                )
            except Exception as exc:
                message = str(exc)
                should_retry = (
                    isinstance(exc, (BrokenPipeError, OSError))
                    or message == "Copilot ACP connection is not available"
                    or (
                        message != "Copilot ACP runtime is stopped"
                        and self._session_is_dead(session)
                    )
                )
                if (
                    recovered
                    or not should_retry
                    or not self._recover_prompt_session(session)
                ):
                    raise
                recovered = True
                log.warning(
                    "CopilotCLIClient: recovered session after prompt failure: %s", exc
                )
                continue
            if (
                result
                or getattr(session, "last_turn_cancelled", False) is True
                or not self._session_is_dead(session)
            ):
                return result
            if recovered or not self._recover_prompt_session(session):
                raise RuntimeError("Copilot CLI session died during prompt")
            recovered = True
            log.warning("CopilotCLIClient: recovered session after empty dead prompt")

    def run_turn(
        self,
        content: str,
        *,
        model: ProviderModel | None = None,
        system_prompt: str | None = None,
        retry_on_preempt: bool = False,
        fresh_session: bool = False,
    ) -> str:
        session = self._resolve_turn_session(
            model=model,
            fresh_session=fresh_session,
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
            log.info(
                "CopilotCLIClient.run_turn: preempted mid-flight — retry %d", attempt
            )

    def _run_turn_json_value(
        self,
        prompt: str,
        key: str,
        model: ProviderModel,
        system_prompt: str | None = None,
    ) -> str:
        json_instruction = (
            f'Respond with ONLY a JSON object in the form {{"{key}": "your answer"}}.'
            " No other text before or after the JSON."
        )
        full_system = (
            f"{system_prompt}\n\n{json_instruction}"
            if system_prompt
            else json_instruction
        )
        raw = self.run_turn(prompt, model=model, system_prompt=full_system)
        if not raw:
            return ""
        candidates = [raw]
        for candidate in candidates:
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and isinstance(obj.get(key), str):
                return obj[key]
        return ""

    def print_prompt_from_file(
        self,
        system_file: Path,
        prompt_file: Path,
        model: ProviderModel,
        timeout: int = 30,
        idle_timeout: float = 1800.0,
        cwd: Path | str = ".",
    ) -> str:
        del idle_timeout
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
    ) -> str:
        del idle_timeout
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
                return result.stdout.strip()
        return ""


class CopilotCLI(Provider):
    """Composite Copilot provider with separate API and runtime agent."""

    def __init__(
        self,
        *,
        api: ProviderAPI | None = None,
        agent: ProviderAgent | None = None,
        session: PromptSession | None = None,
    ) -> None:
        if agent is None:
            agent = CopilotCLIClient(session=session)
        elif session is not None:
            agent.attach_session(session)
        self._api = CopilotCLIAPI() if api is None else api
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
