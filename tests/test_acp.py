"""Tests for shared ACP infrastructure."""

from __future__ import annotations

import io
import signal
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from acp.exceptions import RequestError

from kennel.acp import (
    _ACP_STREAM_LIMIT,
    ACPClient,
    ACPClientBase,
    ACPRuntime,
    ACPSession,
    _is_cancel_sentinel,
    _is_line_limit_overrun_error,
    _preview_log_value,
    _TerminalManager,
    _tool_input_preview,
    combine_prompt,
)
from kennel.provider import (
    ProviderID,
    ProviderModel,
)


class FakeProcess:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO(stderr)
        self.returncode = None if returncode == 0 else returncode
        self._wait_returncode = returncode
        self.pid = 4321
        self.kill_calls = 0

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.returncode = self._wait_returncode
        return self._wait_returncode

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -signal.SIGKILL


class FakeConnection:
    def __init__(
        self,
        *,
        load_supported: bool = True,
        fail_load_session: bool = False,
        fail_resume_session: bool = False,
    ) -> None:
        self.load_supported = load_supported
        self.fail_load_session = fail_load_session
        self.fail_resume_session = fail_resume_session
        self.initialize_calls: list[tuple[int, object, object]] = []
        self.new_session_calls: list[str] = []
        self.load_session_calls: list[tuple[str, str]] = []
        self.resume_session_calls: list[tuple[str, str]] = []
        self.set_session_model_calls: list[tuple[str, str]] = []
        self.prompt_calls: list[tuple[str, list[object]]] = []
        self.cancel_calls: list[str] = []
        self._next_session = 0
        self.client: ACPClientBase | None = None

    async def initialize(
        self, protocol_version: int, client_capabilities: object, client_info: object
    ) -> object:
        self.initialize_calls.append(
            (protocol_version, client_capabilities, client_info)
        )
        return SimpleNamespace(
            agent_capabilities=SimpleNamespace(
                load_session=self.load_supported,
                session_capabilities=SimpleNamespace(
                    resume=True if not self.load_supported else None
                ),
            )
        )

    async def new_session(self, cwd: str) -> object:
        self.new_session_calls.append(cwd)
        self._next_session += 1
        return SimpleNamespace(session_id=f"sess-{self._next_session}")

    async def load_session(self, cwd: str, session_id: str) -> object:
        self.load_session_calls.append((cwd, session_id))
        if self.fail_load_session:
            raise RequestError(
                404, f"Resource not found: Session {session_id} not found"
            )
        return SimpleNamespace()

    async def resume_session(self, cwd: str, session_id: str) -> object:
        self.resume_session_calls.append((cwd, session_id))
        if self.fail_resume_session:
            raise RequestError(
                404, f"Resource not found: Session {session_id} not found"
            )
        return SimpleNamespace()

    async def set_session_model(self, model_id: str, session_id: str) -> object:
        self.set_session_model_calls.append((model_id, session_id))
        return SimpleNamespace()

    async def prompt(self, prompt: list[object], session_id: str) -> object:
        self.prompt_calls.append((session_id, prompt))
        return SimpleNamespace(stop_reason="end_turn")

    async def cancel(self, session_id: str) -> None:
        self.cancel_calls.append(session_id)


class FakeAgentContext:
    def __init__(self, client: ACPClientBase, connection: FakeConnection) -> None:
        self.client = client
        self.connection = connection
        self.connection.client = client

    async def __aenter__(self) -> tuple[FakeConnection, FakeProcess]:
        self.client.on_connect(MagicMock())
        return self.connection, FakeProcess()

    async def __aexit__(self, *args: object) -> None:
        pass


def _spawn_factory(*connections: FakeConnection):
    it = iter(connections)

    @asynccontextmanager
    async def spawn(client: ACPClientBase, command: str, *args: str, **kwargs: object):
        del command, args, kwargs
        context = FakeAgentContext(client, next(it))
        try:
            yield await context.__aenter__()
        finally:
            await context.__aexit__(None, None, None)

    return spawn


class TestHelpers:
    def test_is_line_limit_overrun_error(self) -> None:
        assert _is_line_limit_overrun_error(
            ValueError("Separator is found, but chunk is longer than limit")
        )
        assert not _is_line_limit_overrun_error(ValueError("boom"))

    def test_is_cancel_sentinel(self) -> None:
        sentinel = "CANCELLED"
        assert _is_cancel_sentinel(sentinel, sentinel)
        assert _is_cancel_sentinel("narration\nCANCELLED", sentinel)
        assert not _is_cancel_sentinel("narration", sentinel)
        assert not _is_cancel_sentinel("", sentinel)

    def test_combine_prompt(self) -> None:
        assert (
            combine_prompt("task", base_system_prompt="base", system_prompt="sys")
            == "base\n\n---\n\nsys\n\n---\n\ntask"
        )
        assert combine_prompt("task") == "task"

    def test_preview_log_value(self) -> None:
        assert _preview_log_value(None) == ""
        assert _preview_log_value("hello") == "hello"
        assert _preview_log_value({"a": 1}) == '{"a": 1}'

        class Unserializable:
            def __str__(self) -> str:
                return "unserializable"

        assert _preview_log_value(Unserializable()) == "unserializable"

    def test_tool_input_preview(self) -> None:
        assert _tool_input_preview({"command": "ls"}) == "ls"
        assert _tool_input_preview({"path": "/etc"}) == "/etc"
        assert _tool_input_preview({"url": "http://google.com"}) == "http://google.com"
        assert _tool_input_preview({"query": "search"}) == "search"
        assert _tool_input_preview({"prompt": "hi"}) == "hi"
        assert _tool_input_preview({"other": "val"}) == "val"
        assert _tool_input_preview({}) == ""
        assert _tool_input_preview("not a dict") == "not a dict"


class TestTerminalManager:
    def test_lifecycle(self) -> None:
        process = MagicMock()
        process.stdout = io.StringIO("output")
        process.stderr = io.StringIO("error")
        process.poll.return_value = None
        process.wait.return_value = 0

        popen = MagicMock(return_value=process)
        mgr = _TerminalManager(popen=popen)

        tid = mgr.create(
            "ls", args=["-l"], cwd="/tmp", env=[SimpleNamespace(name="A", value="B")]
        )

        # Wait for threads to read
        time.sleep(0.1)

        out, truncated, code, sig = mgr.output(tid)
        assert "output" in out
        assert "error" in out
        assert not truncated
        assert code is None

        assert mgr.wait(tid) == (0, None)
        process.poll.return_value = 0
        out, _, code, _ = mgr.output(tid)
        assert code == 0

        mgr.release(tid)
        process.kill.assert_not_called()

    def test_kill_and_release_active(self) -> None:
        process = MagicMock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired(["kill"], 1), 0]

        popen = MagicMock(return_value=process)
        mgr = _TerminalManager(popen=popen)
        tid = mgr.create("long")

        mgr.kill(tid)
        assert process.kill.call_count == 2

        process.poll.return_value = None
        process.wait.side_effect = None
        process.wait.return_value = 0
        mgr.release(tid)
        assert process.kill.call_count == 3


class TestACPRuntime:
    def test_spawn_uses_large_transport_limit(self, tmp_path: Path) -> None:
        seen_kwargs: dict[str, object] = {}

        @asynccontextmanager
        async def spawn(client, command, *args, **kwargs):
            seen_kwargs.update(kwargs)
            yield FakeConnection(), FakeProcess()

        runtime = ACPRuntime(
            ProviderID.GEMINI, ["gemini"], work_dir=tmp_path, spawn_agent_process=spawn
        )
        runtime.ensure_session(None, None)
        assert seen_kwargs["transport_kwargs"]["limit"] == _ACP_STREAM_LIMIT
        runtime.stop()

    def test_ensure_prompt_and_stop(self, tmp_path: Path) -> None:
        conn = FakeConnection()
        runtime = ACPRuntime(
            ProviderID.GEMINI,
            ["gemini"],
            work_dir=tmp_path,
            spawn_agent_process=_spawn_factory(conn),
        )
        sid = runtime.ensure_session(None, None)
        assert sid == "sess-1"

        res, reason, sid2 = runtime.prompt(sid, "hi", None)
        assert sid2 == sid
        assert reason == "end_turn"

        runtime.stop()
        assert runtime._stopped

    def test_missing_session_recovery(self, tmp_path: Path) -> None:
        conn = FakeConnection(fail_load_session=True)
        runtime = ACPRuntime(
            ProviderID.GEMINI,
            ["gemini"],
            work_dir=tmp_path,
            spawn_agent_process=_spawn_factory(conn),
        )
        sid = runtime.ensure_session("missing", None)
        assert sid == "sess-1"
        assert runtime.dropped_session_count == 1
        runtime.stop()

    def test_record_session_update(self, tmp_path: Path) -> None:
        runtime = ACPRuntime(
            ProviderID.GEMINI,
            ["gemini"],
            work_dir=tmp_path,
            spawn_agent_process=_spawn_factory(FakeConnection()),
        )
        runtime._active_prompt_session_id = "s1"

        runtime.record_session_update(
            "s1",
            SimpleNamespace(
                session_update="agent_message_chunk",
                content=SimpleNamespace(text="chunk"),
            ),
        )
        assert runtime._prompt_chunks == ["chunk"]

        runtime.record_session_update(
            "s1",
            SimpleNamespace(
                session_update="tool_call",
                tool_call_id="t1",
                title="tool",
                raw_input={"cmd": "ls"},
            ),
        )
        runtime.record_session_update(
            "s1",
            SimpleNamespace(
                session_update="tool_call_update",
                tool_call_id="t1",
                title="tool",
                status="completed",
                raw_output="ok",
            ),
        )

        runtime.stop()

    def test_normalize_model_default(self, tmp_path: Path) -> None:
        runtime = ACPRuntime(ProviderID.GEMINI, ["gemini"], work_dir=tmp_path)
        assert runtime._normalize_model("m1") == ProviderModel("m1")
        runtime.stop()

    def test_normalize_model_custom(self, tmp_path: Path) -> None:
        norm = MagicMock(return_value=ProviderModel("mapped"))
        runtime = ACPRuntime(
            ProviderID.GEMINI, ["gemini"], work_dir=tmp_path, normalize_model=norm
        )
        assert runtime._normalize_model("orig") == ProviderModel("mapped")
        runtime.stop()


class TestACPSession:
    def test_lifecycle(self, tmp_path: Path) -> None:
        sys_file = tmp_path / "sys.md"
        sys_file.write_text("sys")
        runtime = MagicMock()
        runtime.provider_id = ProviderID.GEMINI
        runtime.ensure_session.return_value = "s1"
        runtime.prompt.return_value = ("out", "end_turn", "s1")
        runtime.pid = 123
        runtime.dropped_session_count = 0
        runtime.is_alive.return_value = True

        session = ACPSession(
            sys_file,
            work_dir=tmp_path,
            model="m",
            cancel_sentinel="STOP",
            runtime=runtime,
        )
        assert session.session_id == "s1"
        assert session.pid == 123
        assert session.is_alive()

        assert session.prompt("hi") == "out"
        runtime.prompt.assert_called_with("s1", "sys\n\n---\n\nhi", ProviderModel("m"))

        session.send("queued")
        assert session.consume_until_result() == "out"

        session.switch_model("m2")
        runtime.ensure_session.assert_called_with("s1", ProviderModel("m2"))

        session.recover()
        runtime.recover_session.assert_called()

        session.reset()
        runtime.reset_session.assert_called()

        session.stop()
        runtime.stop.assert_called()

    def test_preemption(self, tmp_path: Path) -> None:
        sys_file = tmp_path / "sys.md"
        sys_file.write_text("")
        runtime = MagicMock()
        runtime.provider_id = ProviderID.GEMINI
        runtime.is_alive.return_value = True

        session = ACPSession(
            sys_file,
            work_dir=tmp_path,
            model="m",
            cancel_sentinel="STOP",
            runtime=runtime,
        )

        # Test wait_for_pending_preempt timeout
        session._pending_preempts = 1
        assert not session.wait_for_pending_preempt(timeout=0.01)

    def test_init_requires_runtime(self, tmp_path: Path) -> None:
        sys_file = tmp_path / "sys.md"
        sys_file.write_text("")
        with pytest.raises(ValueError, match="runtime is required"):
            ACPSession(sys_file, work_dir=tmp_path, model="m", cancel_sentinel="S")
        sys_file = tmp_path / "sys.md"
        sys_file.write_text("")
        runtime = MagicMock()
        runtime.provider_id = ProviderID.GEMINI

        session = ACPSession(
            sys_file,
            work_dir=tmp_path,
            model="m",
            cancel_sentinel="STOP",
            runtime=runtime,
            repo_name="a/b",
        )

        with session:
            assert session.owner == threading.current_thread().name
            # Reentrant
            with session:
                pass


class TestACPClient:
    def test_retry_logic(self) -> None:
        client = ACPClient()
        session = MagicMock()

        # Retry on pipe error
        assert client._should_retry_prompt_failure(BrokenPipeError(), session)

        # No retry on stopped runtime
        assert not client._should_retry_prompt_failure(
            Exception("GEMINI ACP runtime is stopped"), session
        )

        assert client._dead_prompt_error_message() == "ACP session died during prompt"
