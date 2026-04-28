"""Codex CLI and app-server helpers."""

import json
import logging
import queue
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, Protocol

import fido.provider as provider
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
    coerce_provider_model,
    model_name,
)
from fido.session_agent import SessionBackedAgent

log = logging.getLogger(__name__)

_CODEX_RATE_LIMIT_CACHE_SECONDS = 300.0
_CODEX_APP_SERVER_TIMEOUT = 30.0
_CODEX_APP_SERVER_MAX_LINE_BYTES = 8 * 1024 * 1024
_CODEX_CLIENT_INFO = {
    "name": "fido",
    "title": "Fido",
    "version": "0.1.0",
}


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


class CodexProtocolError(RuntimeError):
    """Raised when the Codex app-server stream violates the protocol."""


class CodexAppServerProcess(Protocol):
    """Small protocol for a text-mode ``codex app-server`` subprocess."""

    stdin: IO[str] | None
    stdout: IO[str] | None
    stderr: IO[str] | None
    pid: int

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def kill(self) -> None: ...


class CodexAppServer(Protocol):
    """JSON-RPC-ish app-server boundary used by Codex API/session code."""

    @property
    def pid(self) -> int | None: ...

    def request(
        self, method: str, params: dict[str, Any] | None = None, *, timeout: float = ...
    ) -> Any: ...

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None: ...

    def wait_notification(
        self,
        method: str,
        *,
        predicate: Callable[[dict[str, Any]], bool] | None = None,
        timeout: float = ...,
    ) -> dict[str, Any]: ...

    def is_alive(self) -> bool: ...

    def stop(self) -> None: ...


@dataclass(frozen=True)
class _Response:
    result: Any = None
    error: dict[str, Any] | None = None


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


def _spawn_app_server(
    *,
    cwd: Path | str | None = None,
) -> CodexAppServerProcess:
    return subprocess.Popen(  # noqa: S603 - command is fixed, args are not shell-expanded.
        ["codex", "app-server", "--listen", "stdio://"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        bufsize=1,
    )


class CodexAppServerClient:
    """Thread-safe owner for one ``codex app-server`` stdio connection."""

    def __init__(
        self,
        *,
        process_factory: Callable[..., CodexAppServerProcess] = _spawn_app_server,
        cwd: Path | str | None = None,
        timeout: float = _CODEX_APP_SERVER_TIMEOUT,
        max_line_bytes: int = _CODEX_APP_SERVER_MAX_LINE_BYTES,
    ) -> None:
        self._process_factory = process_factory
        self._cwd = cwd
        self._timeout = timeout
        self._max_line_bytes = max_line_bytes
        self._send_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._response_cond = threading.Condition(self._state_lock)
        self._next_id = 1
        self._responses: dict[int, _Response] = {}
        self._notifications: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stderr_lines: queue.Queue[str] = queue.Queue()
        self._stopped = False
        self._protocol_error: BaseException | None = None
        self._process = self._process_factory(cwd=self._cwd)
        self._reader = threading.Thread(
            target=self._read_stdout,
            name="codex-app-server-stdout",
            daemon=True,
        )
        self._stderr_reader = threading.Thread(
            target=self._read_stderr,
            name="codex-app-server-stderr",
            daemon=True,
        )
        self._reader.start()
        self._stderr_reader.start()
        try:
            self._initialize()
        except Exception:
            self.stop()
            raise

    @property
    def pid(self) -> int | None:
        return self._process.pid

    def _initialize(self) -> None:
        self.request(
            "initialize",
            {
                "clientInfo": _CODEX_CLIENT_INFO,
                "capabilities": {
                    "experimentalApi": True,
                    "optOutNotificationMethods": [],
                },
            },
            timeout=self._timeout,
        )
        self.notify("initialized")

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float = _CODEX_APP_SERVER_TIMEOUT,
    ) -> Any:
        request_id = self._next_request_id()
        self._write({"id": request_id, "method": method, "params": params or {}})
        deadline = time.monotonic() + timeout
        with self._response_cond:
            while request_id not in self._responses:
                self._raise_if_unavailable_locked()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Codex app-server request timed out: {method}")
                self._response_cond.wait(min(remaining, 0.25))
            response = self._responses.pop(request_id)
        if response.error is not None:
            message = response.error.get("message")
            raise CodexProviderError(
                message=message if isinstance(message, str) else str(response.error),
                kind=_classify_provider_error(str(message or response.error)),
                payload=response.error,
            )
        return response.result

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._write({"method": method, "params": params or {}})

    def wait_notification(
        self,
        method: str,
        *,
        predicate: Callable[[dict[str, Any]], bool] | None = None,
        timeout: float = _CODEX_APP_SERVER_TIMEOUT,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        deferred: list[dict[str, Any]] = []
        try:
            while True:
                with self._state_lock:
                    self._raise_if_unavailable_locked()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out waiting for Codex notification: {method}"
                    )
                try:
                    notification = self._notifications.get(timeout=min(remaining, 0.25))
                except queue.Empty:
                    continue
                if method != "*" and notification.get("method") != method:
                    deferred.append(notification)
                    continue
                params = notification.get("params")
                if not isinstance(params, dict):
                    raise CodexProtocolError(
                        f"Codex notification {method} params must be an object"
                    )
                if predicate is None or predicate(params):
                    return notification
                deferred.append(notification)
        finally:
            for notification in deferred:
                self._notifications.put(notification)

    def is_alive(self) -> bool:
        with self._state_lock:
            return (
                not self._stopped
                and self._protocol_error is None
                and self._process.poll() is None
            )

    def stop(self) -> None:
        with self._response_cond:
            self._stopped = True
            self._response_cond.notify_all()
        if self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)

    def _next_request_id(self) -> int:
        with self._state_lock:
            request_id = self._next_id
            self._next_id += 1
            return request_id

    def _write(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, separators=(",", ":")) + "\n"
        with self._send_lock:
            stdin = self._process.stdin
            if stdin is None:
                raise CodexProtocolError("Codex app-server stdin is unavailable")
            stdin.write(encoded)
            stdin.flush()

    def _read_stdout(self) -> None:
        stdout = self._process.stdout
        if stdout is None:
            self._fail_protocol(CodexProtocolError("Codex app-server stdout missing"))
            return
        try:
            while True:
                line = stdout.readline()
                if line == "":
                    self._fail_protocol(
                        CodexProtocolError("Codex app-server closed stdout")
                    )
                    return
                if len(line.encode()) > self._max_line_bytes:
                    self._fail_protocol(
                        CodexProtocolError("Codex app-server line too large")
                    )
                    return
                self._handle_line(line)
        except BaseException as exc:
            self._fail_protocol(exc)

    def _read_stderr(self) -> None:
        stderr = self._process.stderr
        if stderr is None:
            return
        for line in stderr:
            self._stderr_lines.put(line.rstrip())

    def _handle_line(self, line: str) -> None:
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CodexProtocolError(
                f"Codex app-server emitted invalid JSON: {exc}"
            ) from exc
        if not isinstance(message, dict):
            raise CodexProtocolError("Codex app-server message must be a JSON object")
        raw_id = message.get("id")
        if isinstance(raw_id, int) and ("result" in message or "error" in message):
            error = message.get("error")
            if error is not None and not isinstance(error, dict):
                raise CodexProtocolError(
                    "Codex app-server response error must be an object"
                )
            with self._response_cond:
                self._responses[raw_id] = _Response(
                    result=message.get("result"), error=error
                )
                self._response_cond.notify_all()
            return
        method = message.get("method")
        if isinstance(method, str) and "params" in message:
            self._notifications.put(message)
            return
        raise CodexProtocolError(f"Unsupported Codex app-server message: {message!r}")

    def _fail_protocol(self, exc: BaseException) -> None:
        with self._response_cond:
            if not self._stopped and self._protocol_error is None:
                self._protocol_error = exc
            self._response_cond.notify_all()

    def _raise_if_unavailable_locked(self) -> None:
        if self._protocol_error is not None:
            raise self._protocol_error
        if self._stopped:
            raise CodexProtocolError("Codex app-server connection is stopped")
        if self._process.poll() is not None:
            raise CodexProtocolError("Codex app-server process exited")


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


def _normalize_limit_name(value: object, fallback: str) -> str:
    if not isinstance(value, str) or not value:
        return fallback
    return "_".join(value.lower().replace("-", "_").split())


def _parse_rate_limit_reset(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, int | float):
        raise ValueError(f"Codex rate limit resetsAt must be numeric, got {value!r}")
    return datetime.fromtimestamp(float(value), tz=UTC)


def _rate_limit_window(
    limit_id: str, suffix: str, payload: object
) -> ProviderLimitWindow | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError(f"Codex rate limit {limit_id}.{suffix} must be an object")
    used_percent = payload.get("usedPercent")
    if used_percent is None:
        return None
    if not isinstance(used_percent, int | float):
        raise ValueError(
            f"Codex rate limit {limit_id}.{suffix}.usedPercent must be numeric"
        )
    return ProviderLimitWindow(
        name=f"{limit_id}_{suffix}",
        used=round(float(used_percent)),
        limit=100,
        resets_at=_parse_rate_limit_reset(payload.get("resetsAt")),
        unit="%",
    )


def _credits_depleted(payload: object, reached_type: object) -> bool:
    if not isinstance(payload, dict) or not isinstance(reached_type, str):
        return False
    if "credits_depleted" not in reached_type:
        return False
    has_credits = payload.get("hasCredits")
    unlimited = payload.get("unlimited")
    return has_credits is False and unlimited is False


def _reached_window_name(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if "credits_depleted" in value:
        return "credits"
    if "usage_limit_reached" in value:
        return "workspace_usage"
    if "rate_limit_reached" in value:
        return "rate_limit_reached"
    return _normalize_limit_name(value, "codex_limit_reached")


def _codex_limit_windows(payload: dict[str, Any]) -> tuple[ProviderLimitWindow, ...]:
    rate_limits_by_id = payload.get("rateLimitsByLimitId")
    raw_limits: list[object]
    if isinstance(rate_limits_by_id, dict) and rate_limits_by_id:
        raw_limits = [rate_limits_by_id[key] for key in sorted(rate_limits_by_id)]
    else:
        rate_limits = payload["rateLimits"]
        if isinstance(rate_limits, list):
            raw_limits = list(rate_limits)
        elif isinstance(rate_limits, dict):
            raw_limits = [rate_limits]
        else:
            raise ValueError("Codex rateLimits must be an object or list")

    windows: list[ProviderLimitWindow] = []
    for index, raw_limit in enumerate(raw_limits):
        if not isinstance(raw_limit, dict):
            raise ValueError("Codex rate limit snapshots must be objects")
        limit_id = _normalize_limit_name(raw_limit.get("limitId"), f"codex_{index}")
        reached_names: set[str] = set()
        for suffix in ("primary", "secondary"):
            window = _rate_limit_window(limit_id, suffix, raw_limit.get(suffix))
            if window is not None:
                windows.append(window)
                if window.pressure is not None and window.pressure >= 1.0:
                    reached_names.add(window.name)
        reached_type = raw_limit.get("rateLimitReachedType")
        if _credits_depleted(raw_limit.get("credits"), reached_type):
            credit_name = f"{limit_id}_credits"
            windows.append(ProviderLimitWindow(name=credit_name, used=100, limit=100))
            reached_names.add(credit_name)
        reached_name = _reached_window_name(reached_type)
        full_reached_name = f"{limit_id}_{reached_name}" if reached_name else None
        if full_reached_name is not None and full_reached_name not in reached_names:
            windows.append(
                ProviderLimitWindow(name=full_reached_name, used=100, limit=100)
            )
    return tuple(windows)


class CodexAPI(ProviderAPI):
    """Read-only account API for Codex quota and limits."""

    def __init__(
        self,
        *,
        client_factory: Callable[[], CodexAppServer] = CodexAppServerClient,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client_factory = client_factory
        self._monotonic = monotonic
        self._limit_snapshot_lock = threading.Lock()
        self._limit_snapshot_cached_at: float | None = None
        self._limit_snapshot_cache: ProviderLimitSnapshot | None = None

    @property
    def provider_id(self) -> ProviderID:
        return ProviderID.CODEX

    def get_limit_snapshot(self) -> ProviderLimitSnapshot:
        with self._limit_snapshot_lock:
            if (
                self._limit_snapshot_cache is not None
                and self._limit_snapshot_cached_at is not None
                and self._monotonic() - self._limit_snapshot_cached_at
                < _CODEX_RATE_LIMIT_CACHE_SECONDS
            ):
                return self._limit_snapshot_cache
            client: CodexAppServer | None = None
            try:
                client = self._client_factory()
                payload = client.request(
                    "account/rateLimits/read", timeout=_CODEX_APP_SERVER_TIMEOUT
                )
                if not isinstance(payload, dict):
                    raise ValueError("Codex rate limit response must be a JSON object")
                windows = _codex_limit_windows(payload)
                snapshot = (
                    ProviderLimitSnapshot(provider=self.provider_id, windows=windows)
                    if windows
                    else ProviderLimitSnapshot(
                        provider=self.provider_id,
                        unavailable_reason="Codex rate limits did not include usable windows.",
                    )
                )
            except Exception as exc:
                log.exception("CodexAPI: failed to fetch rate limit snapshot")
                snapshot = ProviderLimitSnapshot(
                    provider=self.provider_id,
                    unavailable_reason=f"Codex usage unavailable: {exc}",
                )
            finally:
                if client is not None:
                    client.stop()
            self._limit_snapshot_cache = snapshot
            self._limit_snapshot_cached_at = self._monotonic()
            return snapshot


class CodexSession(OwnedSession):
    """Persistent Codex app-server thread/session implementation."""

    voice_model = ProviderModel("gpt-5.5", "xhigh")
    work_model = ProviderModel("gpt-5.5", "medium")
    brief_model = ProviderModel("gpt-5.5", "low")

    def __init__(
        self,
        system_file: Path,
        *,
        work_dir: Path | str,
        model: ProviderModel | str,
        repo_name: str | None = None,
        client_factory: Callable[..., CodexAppServer] = CodexAppServerClient,
        session_id: str | None = None,
    ) -> None:
        self._work_dir = Path(work_dir).resolve()
        self._repo_name = repo_name
        self._base_system_prompt = (
            system_file.read_text() if system_file.exists() else ""
        )
        self._client_factory = client_factory
        self._client = self._client_factory(cwd=self._work_dir)
        self._model = coerce_provider_model(model)
        self._state_lock = threading.Lock()
        self._session_id: str | None = None
        self._turn_lock = threading.Lock()
        self._active_turn_id: str | None = None
        self._last_turn_cancelled = False
        self._sent_count = 0
        self._received_count = 0
        self._dropped_session_count = 0
        self._init_handler_reentry()
        self._ensure_thread(session_id=session_id)

    @property
    def owner(self) -> str | None:
        if self._repo_name is None:
            return None
        talker = provider.get_talker(self._repo_name)
        if talker is None or talker.kind != "worker":
            return None
        for thread in threading.enumerate():
            if thread.ident == talker.thread_id:
                return thread.name
        return None

    @property
    def pid(self) -> int | None:
        with self._state_lock:
            return self._client.pid

    @property
    def session_id(self) -> str | None:
        with self._state_lock:
            return self._session_id

    @property
    def dropped_session_count(self) -> int:
        with self._state_lock:
            return self._dropped_session_count

    @property
    def sent_count(self) -> int:
        with self._state_lock:
            return self._sent_count

    @property
    def received_count(self) -> int:
        with self._state_lock:
            return self._received_count

    @property
    def last_turn_cancelled(self) -> bool:
        with self._state_lock:
            return self._last_turn_cancelled

    def prompt(
        self,
        content: str,
        model: ProviderModel | None = None,
        system_prompt: str | None = None,
    ) -> str:
        with self:
            if model is not None:
                self.switch_model(model)
            self.send(_combine_prompt(content, self._base_system_prompt, system_prompt))
            return self.consume_until_result()

    def send(self, content: str) -> None:
        thread_id = self._require_thread_id()
        with self._state_lock:
            self._last_turn_cancelled = False
        result = self._client.request(
            "turn/start",
            {
                "threadId": thread_id,
                "input": [
                    {"type": "text", "text": content, "text_elements": []},
                ],
                "model": self._model.model,
                "effort": self._model.efforts[0] if self._model.efforts else None,
                "cwd": str(self._work_dir),
                "approvalPolicy": "never",
                "sandboxPolicy": {"type": "dangerFullAccess"},
            },
        )
        turn = result.get("turn") if isinstance(result, dict) else None
        turn_id = turn.get("id") if isinstance(turn, dict) else None
        if not isinstance(turn_id, str) or not turn_id:
            raise CodexProtocolError("Codex turn/start response missing turn.id")
        with self._turn_lock:
            self._active_turn_id = turn_id
        with self._state_lock:
            self._sent_count += 1

    def consume_until_result(self) -> str:
        with self._turn_lock:
            active_turn_id = self._active_turn_id
        if active_turn_id is None:
            return ""
        final_text = ""
        thread_id = self._require_thread_id()
        while True:
            notification = self._client.wait_notification(
                "*",
                predicate=lambda params: _notification_matches(
                    params, thread_id=thread_id, turn_id=active_turn_id
                ),
                timeout=_CODEX_APP_SERVER_TIMEOUT,
            )
            method = notification.get("method")
            params = notification["params"]
            if method == "error":
                message = params.get("message")
                raise CodexProviderError(
                    message=message if isinstance(message, str) else str(params),
                    kind=_classify_provider_error(str(message or params)),
                    payload=params,
                )
            item = params.get("item")
            if (
                method == "item/completed"
                and isinstance(item, dict)
                and item.get("type") in {"agent_message", "agentMessage"}
            ):
                text = item.get("text")
                if isinstance(text, str):
                    final_text = text
                    with self._state_lock:
                        self._received_count += 1
            completed = (
                _extract_completed_turn(params) if method == "turn/completed" else None
            )
            if completed is None and method == "item/completed":
                completed = self._poll_completed_turn(thread_id, active_turn_id)
            if completed is not None:
                return self._finish_turn(completed, final_text)

    def switch_model(self, model: ProviderModel | str) -> None:
        self._model = coerce_provider_model(model)

    def switch_tools(self, tools: str | None) -> None:
        del tools

    def recover(self) -> None:
        with self._state_lock:
            old_session_id = self._session_id
            old_client = self._client
        old_client.stop()
        new_client = self._client_factory(cwd=self._work_dir)
        with self._state_lock:
            self._client = new_client
        self._ensure_thread(session_id=old_session_id)

    def reset(self, model: ProviderModel | None = None) -> None:
        if model is not None:
            self._model = coerce_provider_model(model)
        with self._state_lock:
            self._session_id = None
            self._last_turn_cancelled = False
        with self._turn_lock:
            self._active_turn_id = None
        self._ensure_thread(session_id=None)

    def is_alive(self) -> bool:
        with self._state_lock:
            return self._client.is_alive()

    def stop(self) -> None:
        with self._state_lock:
            client = self._client
        client.stop()

    def interrupt_active_turn(self) -> None:
        """Interrupt the currently active turn, if one is in flight."""
        self._fire_worker_cancel()

    def _fire_worker_cancel(self) -> None:
        with self._turn_lock:
            turn_id = self._active_turn_id
        with self._state_lock:
            thread_id = self._session_id
            client = self._client
        if thread_id is None or turn_id is None or not client.is_alive():
            return
        client.request(
            "turn/interrupt", {"threadId": thread_id, "turnId": turn_id}, timeout=5
        )

    def __enter__(self) -> "CodexSession":
        depth = getattr(self._reentry_tls, "depth", 0)
        if depth > 0:
            self._bump_entry_depth()
            return self
        kind = provider.current_thread_kind()
        if kind == "worker":
            self._fsm_acquire_worker()
        else:
            self._fsm_acquire_handler()
        self._bump_entry_depth()
        if self._repo_name is not None:
            try:
                provider.register_talker(
                    provider.SessionTalker(
                        repo_name=self._repo_name,
                        thread_id=threading.get_ident(),
                        kind=kind,
                        description="codex session turn",
                        claude_pid=self.pid or 0,
                        started_at=provider.talker_now(),
                    )
                )
            except provider.SessionLeakError:
                self._drop_entry_depth()
                self._fsm_release()
                raise
        return self

    def __exit__(self, *args: object) -> None:
        depth = self._drop_entry_depth()
        if depth == 0:
            if self._repo_name is not None:
                provider.unregister_talker(self._repo_name, threading.get_ident())
            self._fsm_release()

    def _ensure_thread(self, *, session_id: str | None) -> None:
        if session_id:
            try:
                result = self._client.request(
                    "thread/resume", self._thread_params(session_id)
                )
                with self._state_lock:
                    self._session_id = _thread_id_from_result(result)
                return
            except CodexProviderError:
                log.exception("CodexSession: failed to resume session %s", session_id)
                with self._state_lock:
                    self._dropped_session_count += 1
        result = self._client.request("thread/start", self._thread_params(None))
        with self._state_lock:
            self._session_id = _thread_id_from_result(result)

    def _thread_params(self, session_id: str | None) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self._model.model,
            "cwd": str(self._work_dir),
            "approvalPolicy": "never",
            "sandbox": "danger-full-access",
            "developerInstructions": self._base_system_prompt,
        }
        if session_id is not None:
            params["threadId"] = session_id
            params["excludeTurns"] = True
        return params

    def _require_thread_id(self) -> str:
        with self._state_lock:
            session_id = self._session_id
        if session_id is None:
            raise CodexProtocolError("Codex session has no thread id")
        return session_id

    def _poll_completed_turn(
        self, thread_id: str, turn_id: str
    ) -> dict[str, Any] | None:
        try:
            notification = self._client.wait_notification(
                "turn/completed",
                predicate=lambda params: _notification_matches(
                    params, thread_id=thread_id, turn_id=turn_id
                ),
                timeout=0.01,
            )
        except TimeoutError:
            return None
        return notification["params"]

    def _finish_turn(self, params: dict[str, Any], final_text: str) -> str:
        with self._turn_lock:
            self._active_turn_id = None
        turn = params.get("turn")
        status = turn.get("status") if isinstance(turn, dict) else params.get("status")
        if isinstance(status, str) and status.lower() in {"interrupted", "cancelled"}:
            with self._state_lock:
                self._last_turn_cancelled = True
            return ""
        if isinstance(status, str) and status.lower() in {"failed", "error"}:
            error = params.get("error")
            message = error.get("message") if isinstance(error, dict) else str(params)
            raise CodexProviderError(
                message=message if isinstance(message, str) else str(error),
                kind=_classify_provider_error(str(message)),
                payload=params,
            )
        with self._state_lock:
            self._last_turn_cancelled = False
        return final_text


def _thread_id_from_result(result: Any) -> str:
    thread = result.get("thread") if isinstance(result, dict) else None
    thread_id = thread.get("id") if isinstance(thread, dict) else None
    if not isinstance(thread_id, str) or not thread_id:
        raise CodexProtocolError("Codex thread response missing thread.id")
    return thread_id


def _combine_prompt(
    content: str, base_system_prompt: str, system_prompt: str | None
) -> str:
    pieces = [piece for piece in (base_system_prompt, system_prompt, content) if piece]
    return "\n\n".join(pieces)


def _notification_matches(
    params: dict[str, Any], *, thread_id: str, turn_id: str
) -> bool:
    raw_thread_id = params.get("threadId")
    if isinstance(raw_thread_id, str) and raw_thread_id != thread_id:
        return False
    raw_turn_id = params.get("turnId")
    if isinstance(raw_turn_id, str) and raw_turn_id != turn_id:
        return False
    turn = params.get("turn")
    if isinstance(turn, dict):
        nested_turn_id = turn.get("id")
        if isinstance(nested_turn_id, str) and nested_turn_id != turn_id:
            return False
    return True


def _extract_completed_turn(params: dict[str, Any]) -> dict[str, Any] | None:
    turn = params.get("turn")
    if isinstance(turn, dict) and isinstance(turn.get("status"), str):
        return params
    return None


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


def run_codex_exec_resume(
    session_id: str,
    prompt: str,
    *,
    model: ProviderModel | str,
    timeout: int = 300,
    cwd: Path | str = ".",
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    """Run one non-persistent Codex exec resume turn and return raw JSONL output."""
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
        "resume",
        session_id,
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


class CodexClient(SessionBackedAgent, ProviderAgent):
    """Injectable collaborator for Codex CLI and app-server interactions."""

    voice_model = ProviderModel("gpt-5.5", "xhigh")
    work_model = ProviderModel("gpt-5.5", "medium")
    brief_model = ProviderModel("gpt-5.5", "low")

    def __init__(
        self,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        session_fn: Callable[[], PromptSession] = provider.current_repo_session,
        session_factory: Callable[..., PromptSession] | None = None,
        session_system_file: Path | None = None,
        work_dir: Path | str | None = None,
        repo_name: str | None = None,
        session: PromptSession | None = None,
    ) -> None:
        self._runner = runner
        self._session_factory = (
            CodexSession if session_factory is None else session_factory
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
        return ProviderID.CODEX

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

    def _prompt_failure_is_passthrough(self, exc: Exception) -> bool:
        return isinstance(exc, CodexProviderError)

    def _dead_prompt_error_message(self) -> str:
        return "Codex session died during prompt"

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
        prompt = _combine_prompt(prompt_file.read_text(), system_file.read_text(), None)
        return run_codex_exec(
            prompt,
            model=model,
            timeout=timeout,
            cwd=cwd,
            runner=self._runner,
        )

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
        return run_codex_exec_resume(
            session_id,
            prompt_file.read_text(),
            model=model,
            timeout=timeout,
            cwd=cwd,
            runner=self._runner,
        )

    def extract_session_id(self, output: str) -> str:
        return extract_session_id(output)


class Codex(Provider):
    """Composite Codex provider with separate account API and runtime agent."""

    def __init__(
        self,
        *,
        api: ProviderAPI | None = None,
        agent: ProviderAgent | None = None,
        session: PromptSession | None = None,
    ) -> None:
        if agent is None:
            agent = CodexClient(session=session)
        elif session is not None:
            agent.attach_session(session)
        self._api = CodexAPI() if api is None else api
        self._agent = agent

    @property
    def provider_id(self) -> ProviderID:
        return ProviderID.CODEX

    @property
    def api(self) -> ProviderAPI:
        return self._api

    @property
    def agent(self) -> ProviderAgent:
        return self._agent
