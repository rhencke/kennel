from __future__ import annotations

import asyncio
import io
import json
import signal
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kennel import claude
from kennel.copilotcli import (
    CopilotACPRuntime,
    CopilotCLI,
    CopilotCLIAPI,
    CopilotCLIClient,
    CopilotCLISession,
    _combine_prompt,
    _CopilotACPClient,
    _normalize_model,
    _TerminalManager,
    extract_result_text,
    extract_session_id,
)
from kennel.provider import ProviderID, ProviderLimitSnapshot, ProviderModel


def _completed(
    stdout: str = "", returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _copilot_output(text: str = "OK", session_id: str = "sess-123") -> str:
    return "\n".join(
        [
            json.dumps({"type": "assistant.message", "data": {"content": text}}),
            json.dumps(
                {
                    "type": "result",
                    "sessionId": session_id,
                    "exitCode": 0,
                }
            ),
        ]
    )


class TestNormalizeModel:
    def test_maps_claude_haiku_to_gpt54(self) -> None:
        assert _normalize_model("claude-haiku-4-5-20251001") == ProviderModel("gpt-5.4")


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


class FakeRuntime:
    def __init__(self) -> None:
        self.ensure_calls: list[tuple[str | None, str | None]] = []
        self.recover_calls: list[tuple[str | None, str | None]] = []
        self.reset_calls: list[str | None] = []
        self.prompt_calls: list[tuple[str, str, str | None]] = []
        self.cancel_calls: list[str] = []
        self.stop_called = False
        self.pid = 111
        self.alive = True
        self.next_prompt = ("done", "end_turn", "sess-next")

    def ensure_session(self, session_id: str | None, model: str | None) -> str:
        self.ensure_calls.append((session_id, model))
        return "sess-created" if session_id is None else session_id

    def recover_session(self, session_id: str | None, model: str | None) -> str:
        self.recover_calls.append((session_id, model))
        return "sess-recovered"

    def reset_session(self, model: str | None) -> str:
        self.reset_calls.append(model)
        return "sess-reset"

    def prompt(
        self, session_id: str, content: str, model: str | None
    ) -> tuple[str, str, str]:
        self.prompt_calls.append((session_id, content, model))
        return self.next_prompt

    def cancel(self, session_id: str) -> None:
        self.cancel_calls.append(session_id)

    def is_alive(self) -> bool:
        return self.alive

    def stop(self) -> None:
        self.stop_called = True


class FakeConnection:
    def __init__(self, *, load_supported: bool = True) -> None:
        self.load_supported = load_supported
        self.initialize_calls: list[tuple[int, object, object]] = []
        self.new_session_calls: list[str] = []
        self.load_session_calls: list[tuple[str, str]] = []
        self.resume_session_calls: list[tuple[str, str]] = []
        self.set_session_model_calls: list[tuple[str, str]] = []
        self.prompt_calls: list[tuple[str, list[object]]] = []
        self.cancel_calls: list[str] = []
        self._next_session = 0
        self.client: _CopilotACPClient | None = None

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
        return SimpleNamespace()

    async def resume_session(self, cwd: str, session_id: str) -> object:
        self.resume_session_calls.append((cwd, session_id))
        return SimpleNamespace()

    async def set_session_model(self, model_id: str, session_id: str) -> object:
        self.set_session_model_calls.append((model_id, session_id))
        return SimpleNamespace()

    async def prompt(self, prompt: list[object], session_id: str) -> object:
        self.prompt_calls.append((session_id, prompt))
        assert self.client is not None
        self.client._runtime.record_session_update(
            session_id,
            SimpleNamespace(
                session_update="agent_message_chunk",
                content=SimpleNamespace(text="hello from copilot"),
            ),
        )
        return SimpleNamespace(stop_reason="end_turn")

    async def cancel(self, session_id: str) -> None:
        self.cancel_calls.append(session_id)


class FakeAgentContext:
    def __init__(self, client: _CopilotACPClient, connection: FakeConnection) -> None:
        self._client = client
        self._connection = connection
        self._process = SimpleNamespace(pid=9876, returncode=None)

    async def __aenter__(self) -> tuple[FakeConnection, object]:
        self._connection.client = self._client
        return self._connection, self._process

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self._process.returncode = 0


def _spawn_factory(*connections: FakeConnection):
    available = list(connections)

    @asynccontextmanager
    async def spawn(
        client: _CopilotACPClient, command: str, *args: str, **kwargs: object
    ):
        del command, args, kwargs
        connection = available.pop(0)
        context = FakeAgentContext(client, connection)
        try:
            yield await context.__aenter__()
        finally:
            await context.__aexit__(None, None, None)

    return spawn


class TestHelpers:
    def test_extract_helpers_ignore_invalid_lines(self) -> None:
        output = "\n".join(
            [
                "",
                "not-json",
                json.dumps(["not", "a", "dict"]),
                json.dumps({"type": "assistant.message", "data": "bad"}),
                json.dumps({"type": "result", "sessionId": 123}),
            ]
        )
        assert extract_result_text(output) == ""
        assert extract_session_id(output) == ""

    def test_extract_result_text_returns_last_message(self) -> None:
        output = "\n".join(
            [
                json.dumps({"type": "assistant.message", "data": {"content": "first"}}),
                json.dumps(
                    {"type": "assistant.message", "data": {"content": "second"}}
                ),
            ]
        )
        assert extract_result_text(output) == "second"

    def test_extract_session_id_returns_last_result(self) -> None:
        output = "\n".join(
            [
                json.dumps({"type": "result", "sessionId": "one"}),
                json.dumps({"type": "result", "sessionId": "two"}),
            ]
        )
        assert extract_session_id(output) == "two"

    def test_combine_prompt_joins_sections(self) -> None:
        assert (
            _combine_prompt(
                "task", base_system_prompt="persona", system_prompt="system"
            )
            == "persona\n\n---\n\nsystem\n\n---\n\ntask"
        )

    def test_normalize_passthrough_is_used_by_cli_prompt(self, tmp_path: Path) -> None:
        runner = MagicMock(return_value=_completed(_copilot_output()))
        client = CopilotCLIClient(runner=runner, work_dir=tmp_path)
        client._run_cli_prompt("body", model="gpt-5", timeout=1)
        assert "gpt-5" in runner.call_args.args[0]


class TestTerminalManager:
    def test_read_stream_stops_on_empty_chunk(self) -> None:
        manager = _TerminalManager()

        class EmptyStream:
            def __init__(self) -> None:
                self.closed = False

            def read(self, size: int) -> str | None:
                del size
                return None

            def close(self) -> None:
                self.closed = True

        stream = EmptyStream()
        record = SimpleNamespace(append=MagicMock())
        manager._read_stream(record, stream)  # pyright: ignore[reportPrivateUsage]
        record.append.assert_not_called()
        assert stream.closed is True

    def test_terminal_record_handles_signal_exit(self) -> None:
        process = FakeProcess("", "", 0)
        process.returncode = -signal.SIGTERM
        manager = _TerminalManager(popen=lambda *args, **kwargs: process)
        terminal_id = manager.create("sleep")
        output, truncated, exit_code, signal_name = manager.output(terminal_id)
        assert output == ""
        assert truncated is False
        assert exit_code is None
        assert signal_name == "SIGTERM"

    def test_terminal_record_handles_normal_exit(self) -> None:
        process = FakeProcess("", "", 0)
        process.returncode = 7
        manager = _TerminalManager(popen=lambda *args, **kwargs: process)
        terminal_id = manager.create("sleep")
        output, truncated, exit_code, signal_name = manager.output(terminal_id)
        assert output == ""
        assert truncated is False
        assert exit_code == 7
        assert signal_name is None

    def test_create_output_wait_and_release(self) -> None:
        manager = _TerminalManager(
            popen=lambda *args, **kwargs: FakeProcess("hello", " world", 0)
        )
        terminal_id = manager.create("echo", args=["hi"], output_byte_limit=4)
        time.sleep(0.05)
        output, truncated, exit_code, signal_name = manager.output(terminal_id)
        assert output in {"orld", "ello"}
        assert truncated is True
        assert exit_code is None
        assert signal_name is None
        assert manager.wait(terminal_id) == (0, None)
        manager.release(terminal_id)

    def test_kill_kills_running_process(self) -> None:
        process = FakeProcess("x", "", 0)
        process.returncode = None
        manager = _TerminalManager(popen=lambda *args, **kwargs: process)
        terminal_id = manager.create("sleep")
        manager.kill(terminal_id)
        assert process.kill_calls == 1

    def test_wait_returns_signal_name(self) -> None:
        process = FakeProcess("", "", 0)
        process._wait_returncode = -signal.SIGTERM
        manager = _TerminalManager(popen=lambda *args, **kwargs: process)
        terminal_id = manager.create("sleep")
        assert manager.wait(terminal_id) == (None, "SIGTERM")

    def test_create_merges_env_and_timeout_branches(self) -> None:
        class TimeoutProcess(FakeProcess):
            def __init__(self) -> None:
                super().__init__("", "", 0)
                self._wait_calls = 0
                self.returncode = None
                self.last_env: dict[str, str] | None = None

            def wait(self, timeout: float | None = None) -> int:
                self._wait_calls += 1
                if self._wait_calls == 1 and timeout is not None:
                    raise subprocess.TimeoutExpired("cmd", timeout)
                self.returncode = 0
                return 0

        process = TimeoutProcess()

        def popen(*args, **kwargs):
            del args
            process.last_env = kwargs["env"]
            return process

        manager = _TerminalManager(popen=popen)
        terminal_id = manager.create(
            "echo",
            env=[SimpleNamespace(name="COPILOT_TEST", value="yes")],
        )
        assert process.last_env is not None
        assert process.last_env["COPILOT_TEST"] == "yes"
        manager.kill(terminal_id)
        process.returncode = None
        manager.release(terminal_id)

        release_process = TimeoutProcess()
        release_manager = _TerminalManager(
            popen=lambda *args, **kwargs: release_process
        )
        release_id = release_manager.create("echo")
        release_manager.release(release_id)


class TestCopilotACPClient:
    def test_file_terminal_permission_and_updates(self, tmp_path: Path) -> None:
        runtime = MagicMock()
        terminals = MagicMock()
        terminals.create.return_value = "term-1"
        terminals.output.return_value = ("out", False, 0, None)
        terminals.wait.return_value = (0, None)
        client = _CopilotACPClient(runtime, terminals=terminals)

        path = tmp_path / "file.txt"
        path.write_text("a\nb\nc")
        read = asyncio.run(client.read_text_file(str(path), "sess", limit=2, line=2))
        assert read.content == "b\n"

        written = tmp_path / "nested" / "new.txt"
        asyncio.run(client.write_text_file("body", str(written), "sess"))
        assert written.read_text() == "body"

        created = asyncio.run(client.create_terminal("bash", "sess"))
        assert created.terminal_id == "term-1"

        output = asyncio.run(client.terminal_output("sess", "term-1"))
        assert output.output == "out"
        assert output.exit_status is not None

        waited = asyncio.run(client.wait_for_terminal_exit("sess", "term-1"))
        assert waited.exit_code == 0

        asyncio.run(client.kill_terminal("sess", "term-1"))
        asyncio.run(client.release_terminal("sess", "term-1"))

        allow = asyncio.run(
            client.request_permission(
                [SimpleNamespace(kind="allow_once", option_id="yes")],
                "sess",
                MagicMock(),
            )
        )
        deny = asyncio.run(
            client.request_permission(
                [SimpleNamespace(kind="reject_once", option_id="no")],
                "sess",
                MagicMock(),
            )
        )
        assert allow.outcome.option_id == "yes"
        assert deny.outcome.outcome == "cancelled"

        update = SimpleNamespace(
            session_update="agent_message_chunk", content=SimpleNamespace(text="hello")
        )
        asyncio.run(client.session_update("sess", update))
        runtime.record_session_update.assert_called_once_with("sess", update)

    def test_on_connect_is_noop(self) -> None:
        assert (
            asyncio.run(_CopilotACPClient(MagicMock()).on_connect(MagicMock())) is None
        )


class TestCopilotACPRuntime:
    def test_command_for_none_effort_uses_base_command(self, tmp_path: Path) -> None:
        runtime = CopilotACPRuntime(
            work_dir=tmp_path,
            spawn_agent_process=_spawn_factory(FakeConnection()),
        )
        try:
            assert runtime._command_for_effort(None) == runtime._command_base
        finally:
            runtime.stop()

    def test_command_for_effort_appends_flag(self, tmp_path: Path) -> None:
        runtime = CopilotACPRuntime(
            work_dir=tmp_path,
            spawn_agent_process=_spawn_factory(FakeConnection()),
        )
        try:
            assert runtime._command_for_effort("high") == (
                *runtime._command_base,
                "--effort",
                "high",
            )
        finally:
            runtime.stop()

    def test_preferred_efforts_preserves_order(self, tmp_path: Path) -> None:
        runtime = CopilotACPRuntime(
            work_dir=tmp_path,
            spawn_agent_process=_spawn_factory(FakeConnection()),
        )
        try:
            model = ProviderModel("gpt-5.4", ("xhigh", "high"))
            assert runtime._preferred_efforts(model) == ("xhigh", "high")
        finally:
            runtime.stop()

    def test_ensure_prompt_and_stop(self, tmp_path: Path) -> None:
        connection = FakeConnection(load_supported=True)
        runtime = CopilotACPRuntime(
            work_dir=tmp_path,
            spawn_agent_process=_spawn_factory(connection),
        )
        try:
            session_id = runtime.ensure_session(None, "claude-opus-4-6")
            assert session_id == "sess-1"
            assert connection.new_session_calls == [str(tmp_path)]
            assert connection.set_session_model_calls == [("gpt-5.4", "sess-1")]
            text, stop_reason, active_session = runtime.prompt(
                session_id, "hello", None
            )
            assert text == "hello from copilot"
            assert stop_reason == "end_turn"
            assert active_session == "sess-1"
            runtime.cancel("sess-1")
            assert connection.cancel_calls == ["sess-1"]
        finally:
            runtime.stop()

    def test_resume_and_recover(self, tmp_path: Path) -> None:
        first = FakeConnection(load_supported=False)
        second = FakeConnection(load_supported=False)
        runtime = CopilotACPRuntime(
            work_dir=tmp_path,
            spawn_agent_process=_spawn_factory(first, second),
        )
        try:
            assert runtime.ensure_session("persisted", None) == "persisted"
            assert first.resume_session_calls == [(str(tmp_path), "persisted")]
            assert (
                runtime.recover_session("persisted", "claude-sonnet-4-6") == "persisted"
            )
            assert second.resume_session_calls == [(str(tmp_path), "persisted")]
            assert second.set_session_model_calls == [("gpt-5.4", "persisted")]
            assert runtime.reset_session(None) == "sess-1"
        finally:
            runtime.stop()

    def test_load_session_branch_is_used_when_supported(self, tmp_path: Path) -> None:
        connection = FakeConnection(load_supported=True)
        runtime = CopilotACPRuntime(
            work_dir=tmp_path,
            spawn_agent_process=_spawn_factory(connection),
        )
        try:
            assert runtime.ensure_session("persisted", None) == "persisted"
            assert connection.load_session_calls == [(str(tmp_path), "persisted")]
        finally:
            runtime.stop()

    def test_cancel_without_connection_is_noop(self, tmp_path: Path) -> None:
        runtime = CopilotACPRuntime(
            work_dir=tmp_path,
            spawn_agent_process=_spawn_factory(FakeConnection()),
        )
        try:
            assert runtime.pid is None
            runtime.cancel("sess")
        finally:
            runtime.stop()

    def test_internal_runtime_branches(self, tmp_path: Path) -> None:
        runtime = CopilotACPRuntime(
            work_dir=tmp_path,
            spawn_agent_process=_spawn_factory(FakeConnection()),
        )
        try:
            runtime.record_session_update(
                "wrong",
                SimpleNamespace(
                    session_update="agent_message_chunk",
                    content=SimpleNamespace(text="ignored"),
                ),
            )
            runtime._active_prompt_session_id = "sess"  # pyright: ignore[reportPrivateUsage]
            runtime.record_session_update(
                "sess",
                SimpleNamespace(
                    session_update="tool_call",
                    content=SimpleNamespace(text="ignored"),
                ),
            )
            runtime._initialize_response = None  # pyright: ignore[reportPrivateUsage]
            assert runtime._supports_load_session() is False  # pyright: ignore[reportPrivateUsage]
            runtime._initialize_response = SimpleNamespace(agent_capabilities=None)  # pyright: ignore[reportPrivateUsage]
            assert runtime._supports_load_session() is False  # pyright: ignore[reportPrivateUsage]
            runtime._connection = None  # pyright: ignore[reportPrivateUsage]
            with pytest.raises(RuntimeError, match="connection is not available"):
                asyncio.run(runtime._attach_session_async("sess"))  # pyright: ignore[reportPrivateUsage]
            with pytest.raises(RuntimeError, match="connection is not available"):
                asyncio.run(runtime._set_model_async("sess", "gpt-5"))  # pyright: ignore[reportPrivateUsage]

            async def no_session(session_id: str | None, model: str | None) -> str:
                del session_id, model
                return "sess"

            runtime._ensure_session_async = no_session  # pyright: ignore[reportPrivateUsage]
            with pytest.raises(RuntimeError, match="connection is not available"):
                asyncio.run(runtime._prompt_async("sess", "body", None))  # pyright: ignore[reportPrivateUsage]
            sleeper = asyncio.run_coroutine_threadsafe(
                asyncio.sleep(60),
                runtime._loop,  # pyright: ignore[reportPrivateUsage]
            )
            assert sleeper.done() is False
        finally:
            runtime.stop()
            runtime.stop()
        with pytest.raises(RuntimeError, match="runtime is stopped"):
            runtime._run_async(object())  # pyright: ignore[reportPrivateUsage]

    def test_connection_start_failure_closes_context(self, tmp_path: Path) -> None:
        class BrokenConnection(FakeConnection):
            async def initialize(
                self,
                protocol_version: int,
                client_capabilities: object,
                client_info: object,
            ) -> object:
                del protocol_version, client_capabilities, client_info
                raise RuntimeError("boom")

        runtime = CopilotACPRuntime(
            work_dir=tmp_path,
            spawn_agent_process=_spawn_factory(BrokenConnection()),
        )
        try:
            with pytest.raises(RuntimeError, match="boom"):
                runtime.ensure_session(None, None)
        finally:
            runtime.stop()


class TestCopilotCLISession:
    def test_prompt_and_lifecycle_delegate_to_runtime(self, tmp_path: Path) -> None:
        system_file = tmp_path / "persona.md"
        system_file.write_text("persona")
        runtime = FakeRuntime()
        runtime.next_prompt = ("result", "cancelled", "sess-2")
        session = CopilotCLISession(
            system_file,
            work_dir=tmp_path,
            model=CopilotCLIClient.work_model,
            runtime=runtime,
        )

        assert session.session_id == "sess-created"
        assert (
            session.prompt(
                "task", model=CopilotCLIClient.voice_model, system_prompt="system"
            )
            == "result"
        )
        assert runtime.prompt_calls == [
            (
                "sess-created",
                "persona\n\n---\n\nsystem\n\n---\n\ntask",
                CopilotCLIClient.voice_model,
            )
        ]
        assert session.last_turn_cancelled is True
        session.send("queued")
        runtime.next_prompt = ("queued-result", "end_turn", "sess-3")
        assert session.consume_until_result() == "queued-result"
        assert session.consume_until_result() == ""
        session.switch_model(CopilotCLIClient.brief_model)
        session.recover()
        session.reset(CopilotCLIClient.voice_model)
        assert session.session_id == "sess-reset"
        assert session.pid == 111
        assert session.is_alive() is True
        session.stop()
        assert runtime.stop_called is True

    def test_preempt_waits_for_contender(self, tmp_path: Path) -> None:
        system_file = tmp_path / "persona.md"
        system_file.write_text("")
        runtime = FakeRuntime()
        session = CopilotCLISession(
            system_file,
            work_dir=tmp_path,
            model=CopilotCLIClient.work_model,
            runtime=runtime,
        )
        acquired = threading.Event()
        release = threading.Event()

        session.__enter__()

        def contender() -> None:
            with session:
                acquired.set()
                release.wait()

        thread = threading.Thread(target=contender, daemon=True)
        thread.start()
        for _ in range(100):
            if runtime.cancel_calls:
                break
            time.sleep(0.01)
        assert runtime.cancel_calls == ["sess-created"]
        assert session.wait_for_pending_preempt(timeout=0.01) is False
        session.__exit__(None, None, None)
        acquired.wait(timeout=1.0)
        assert session.wait_for_pending_preempt(timeout=0.01) is False
        release.set()
        thread.join(timeout=1.0)
        assert session.wait_for_pending_preempt(timeout=1.0) is True

    def test_missing_system_file_and_runtime_factory_branch(
        self, tmp_path: Path
    ) -> None:
        runtime = FakeRuntime()
        session = CopilotCLISession(
            tmp_path / "missing.md",
            work_dir=tmp_path,
            model=CopilotCLIClient.work_model,
            runtime_factory=lambda **kwargs: runtime,
        )
        assert session.owner is None
        assert session.prompt("body") == "done"

    def test_no_cancel_when_runtime_dead(self, tmp_path: Path) -> None:
        runtime = FakeRuntime()
        runtime.alive = False
        session = CopilotCLISession(
            tmp_path / "missing.md",
            work_dir=tmp_path,
            model=CopilotCLIClient.work_model,
            runtime=runtime,
        )
        acquired = threading.Event()

        session.__enter__()

        def contender() -> None:
            with session:
                acquired.set()

        thread = threading.Thread(target=contender, daemon=True)
        thread.start()
        time.sleep(0.05)
        session.__exit__(None, None, None)
        acquired.wait(timeout=1.0)
        thread.join(timeout=1.0)
        assert runtime.cancel_calls == []


class TestCopilotCLIAPI:
    def test_limit_snapshot_is_unavailable(self) -> None:
        assert CopilotCLIAPI().get_limit_snapshot() == ProviderLimitSnapshot(
            provider=ProviderID.COPILOT_CLI,
            unavailable_reason="Copilot CLI does not expose local usage limits.",
        )


class TestCopilotCLIClient:
    def test_default_session_resolver_uses_current_repo(self) -> None:
        session = MagicMock()
        session.prompt.return_value = "ok"
        claude.set_session_resolver(lambda repo: session)
        claude.set_thread_repo("owner/repo")
        try:
            client = CopilotCLIClient()
            assert client.run_turn("hi", model=client.voice_model) == "ok"
        finally:
            claude.set_thread_repo(None)
            claude.set_session_resolver(None)

    def test_session_attachment_and_properties(self) -> None:
        attached = MagicMock(owner="worker-home")
        attached.is_alive.return_value = True
        attached.pid = 12
        client = CopilotCLIClient(session=attached)
        assert client.session is attached
        assert client.session_owner == "worker-home"
        assert client.session_alive is True
        assert client.session_pid == 12
        assert client.provider_id == ProviderID.COPILOT_CLI
        assert client.detach_session() is attached
        assert client.session is None

    def test_attach_session_sets_session(self) -> None:
        attached = MagicMock()
        client = CopilotCLIClient()
        client.attach_session(attached)
        assert client.session is attached

    def test_session_id_none_branches(self) -> None:
        assert CopilotCLIClient().session_id is None
        assert CopilotCLIClient(session=object()).session_id is None
        assert (
            CopilotCLIClient(session=SimpleNamespace(session_id=123)).session_id is None
        )

    def test_ensure_session_requires_factory_inputs(self) -> None:
        client = CopilotCLIClient()
        with pytest.raises(
            ValueError,
            match="CopilotCLIClient.ensure_session requires session_system_file and work_dir",
        ):
            client.ensure_session()

    def test_ensure_session_requires_model_when_creating_session(
        self, tmp_path: Path
    ) -> None:
        client = CopilotCLIClient(
            session_system_file=tmp_path / "persona.md",
            work_dir=tmp_path,
        )
        with pytest.raises(
            ValueError,
            match="CopilotCLIClient.ensure_session requires model when creating a session",
        ):
            client.ensure_session()

    def test_run_turn_requires_model_when_spawning_owned_session(
        self, tmp_path: Path
    ) -> None:
        client = CopilotCLIClient(
            session_system_file=tmp_path / "persona.md",
            work_dir=tmp_path,
        )
        with pytest.raises(
            ValueError,
            match="CopilotCLIClient.run_turn requires model when creating a session",
        ):
            client.run_turn("fetch")

    def test_fresh_session_requires_reset_method(self) -> None:
        client = CopilotCLIClient(session=object())
        with pytest.raises(
            ValueError,
            match="CopilotCLIClient.run_turn fresh_session requires resettable session",
        ):
            client.run_turn("fetch", model=client.voice_model, fresh_session=True)

    def test_ensure_fresh_stop_noop_branches(self, tmp_path: Path) -> None:
        session = MagicMock()
        session_factory = MagicMock(return_value=session)
        client = CopilotCLIClient(
            session_factory=session_factory,
            session_system_file=tmp_path / "persona.md",
            work_dir=tmp_path,
        )
        client.ensure_session(client.voice_model)
        session.prompt.return_value = "ok"
        assert (
            client.run_turn("fetch", model=client.work_model, fresh_session=True)
            == "ok"
        )
        client.stop_session()
        session_factory.assert_called_once_with(
            tmp_path / "persona.md",
            work_dir=tmp_path,
            model=client.voice_model,
            repo_name=None,
        )
        session.reset.assert_called_once_with(client.work_model)
        session.switch_model.assert_not_called()
        session.stop.assert_called_once_with()

    def test_ensure_session_switches_model_when_session_exists(self) -> None:
        session = MagicMock()
        client = CopilotCLIClient(session=session)
        client.ensure_session(client.voice_model)
        session.switch_model.assert_called_once_with(client.voice_model)

    def test_fresh_session_spawns_when_session_missing(self, tmp_path: Path) -> None:
        session = MagicMock()
        session_factory = MagicMock(return_value=session)
        client = CopilotCLIClient(
            session_factory=session_factory,
            session_system_file=tmp_path / "persona.md",
            work_dir=tmp_path,
        )
        session.prompt.return_value = "ok"
        assert (
            client.run_turn("fetch", model=client.voice_model, fresh_session=True)
            == "ok"
        )
        session_factory.assert_called_once_with(
            tmp_path / "persona.md",
            work_dir=tmp_path,
            model=client.voice_model,
            repo_name=None,
        )
        session.reset.assert_not_called()

    def test_run_turn_retries_after_preempt(self) -> None:
        session = MagicMock()
        session.last_turn_cancelled = False
        prompts = iter(["partial", "done"])

        def prompt(*args: object, **kwargs: object) -> str:
            result = next(prompts)
            session.last_turn_cancelled = result == "partial"
            return result

        session.prompt.side_effect = prompt
        client = CopilotCLIClient(session=session)
        assert client.run_turn("fetch", retry_on_preempt=True) == "done"
        session.wait_for_pending_preempt.assert_called_once_with()

    def test_run_turn_recovers_and_retries_after_connection_loss(self) -> None:
        session = MagicMock()
        session.prompt.side_effect = [
            RuntimeError("Copilot ACP connection is not available"),
            "done",
        ]
        client = CopilotCLIClient(session=session)
        assert client.run_turn("fetch", model=client.voice_model) == "done"
        assert session.prompt.call_count == 2
        session.recover.assert_called_once_with()

    def test_run_turn_recovers_after_empty_result_from_dead_session(self) -> None:
        session = MagicMock()
        session.prompt.side_effect = ["", "done"]
        session.last_turn_cancelled = False
        session.is_alive.side_effect = [False, True]
        client = CopilotCLIClient(session=session)
        assert client.run_turn("fetch", model=client.voice_model) == "done"
        session.recover.assert_called_once_with()

    def test_run_turn_runtime_stopped_still_raises(self) -> None:
        session = MagicMock()
        session.prompt.side_effect = RuntimeError("Copilot ACP runtime is stopped")
        client = CopilotCLIClient(session=session)
        with pytest.raises(RuntimeError, match="runtime is stopped"):
            client.run_turn("fetch", model=client.voice_model)
        session.recover.assert_not_called()

    def test_run_turn_connection_loss_raises_when_session_cannot_recover(
        self,
    ) -> None:
        session = SimpleNamespace(
            prompt=MagicMock(
                side_effect=RuntimeError("Copilot ACP connection is not available")
            ),
            is_alive=lambda: False,
        )
        client = CopilotCLIClient(session=session)
        with pytest.raises(RuntimeError, match="connection is not available"):
            client.run_turn("fetch", model=client.voice_model)

    def test_run_turn_dead_empty_result_raises_after_one_recovery(self) -> None:
        session = MagicMock()
        session.prompt.side_effect = ["", ""]
        session.last_turn_cancelled = False
        session.is_alive.side_effect = [False, False]
        client = CopilotCLIClient(session=session)
        with pytest.raises(RuntimeError, match="session died during prompt"):
            client.run_turn("fetch", model=client.voice_model)
        session.recover.assert_called_once_with()

    def test_json_and_one_shot_helpers(self, tmp_path: Path) -> None:
        runner = MagicMock(return_value=_completed(_copilot_output("line1\nline2")))
        system_file = tmp_path / "system"
        prompt_file = tmp_path / "prompt"
        system_file.write_text("system")
        prompt_file.write_text("prompt")
        session = MagicMock()
        session.prompt.return_value = '{"emoji": "rocket"}'
        client = CopilotCLIClient(
            runner=runner,
            session=session,
            session_system_file=system_file,
            work_dir=tmp_path,
        )

        assert client.print_prompt_json("q", "emoji", client.voice_model) == "rocket"
        assert client.print_prompt_from_file(
            system_file, prompt_file, client.work_model
        ) == _copilot_output("line1\nline2")
        assert client.resume_session("sess-1", prompt_file, client.brief_model)
        assert client.generate_reply("reply", client.voice_model) == "line1\nline2"
        assert client.generate_branch_name("branch", client.brief_model) == "line1"
        assert client.generate_status_with_session(
            "status", "system", client.voice_model
        ) == (
            "line1\nline2",
            "sess-123",
        )
        assert (
            client.resume_status("sess-1", "status", client.voice_model)
            == "line1\nline2"
        )
        cmd = runner.call_args.args[0]
        assert "--model" in cmd
        assert "gpt-5.4" in cmd

    def test_status_helpers_delegate_to_run_turn(self) -> None:
        session = MagicMock()
        session.prompt.return_value = "ok"
        client = CopilotCLIClient(session=session)
        assert client.generate_status("status", "system", client.voice_model) == "ok"
        assert (
            client.generate_status_emoji("emoji", "system", client.voice_model) == "ok"
        )

    def test_generate_reply_handles_failures(self) -> None:
        client = CopilotCLIClient(
            runner=MagicMock(side_effect=subprocess.TimeoutExpired("copilot", 1))
        )
        assert client.generate_reply("reply", client.voice_model) == ""

    def test_empty_or_malformed_json_and_other_failures(self, tmp_path: Path) -> None:
        session = MagicMock()
        session.prompt.side_effect = ["", "not json"]
        client = CopilotCLIClient(session=session)
        assert client.print_prompt_json("q", "emoji", client.voice_model) == ""
        assert client.print_prompt_json("q", "emoji", client.voice_model) == ""

        runner = MagicMock(side_effect=FileNotFoundError("missing"))
        client = CopilotCLIClient(runner=runner, work_dir=tmp_path)
        assert client.generate_branch_name("branch", client.brief_model) == ""
        assert client.generate_status_with_session(
            "status", "system", client.voice_model
        ) == ("", "")
        assert client.resume_status("sess", "status", client.voice_model) == ""

        runner = MagicMock(return_value=_completed("", returncode=1))
        client = CopilotCLIClient(runner=runner, work_dir=tmp_path)
        assert client._run_cli_prompt("body", model="claude-opus-4-6", timeout=1) == ""


class TestCopilotCLI:
    def test_default_provider_id_and_injected_components(self) -> None:
        api = MagicMock()
        agent = MagicMock()
        provider = CopilotCLI(api=api, agent=agent)
        assert provider.provider_id == ProviderID.COPILOT_CLI
        assert provider.api is api
        assert provider.agent is agent

    def test_default_agent_receives_session(self) -> None:
        session = MagicMock()
        provider = CopilotCLI(session=session)
        assert provider.agent.session is session

    def test_injected_agent_receives_session(self) -> None:
        agent = MagicMock()
        session = MagicMock()
        CopilotCLI(agent=agent, session=session)
        agent.attach_session.assert_called_once_with(session)
