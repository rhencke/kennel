import io
import queue
import subprocess
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fido.codex import (
    Codex,
    CodexAPI,
    CodexAppServerClient,
    CodexClient,
    CodexCLIError,
    CodexProtocolError,
    CodexProviderError,
    CodexSession,
    _codex,
    extract_result_text,
    extract_session_id,
    raise_for_provider_error_output,
    run_codex_exec,
    run_codex_exec_resume,
)
from fido.provider import (
    ContextOverflowError,
    ProviderID,
    ProviderInterruptTimeout,
    ProviderModel,
)

FIXTURES = Path(__file__).parent / "fixtures" / "codex"


def _completed(
    stdout: str = "", returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


class _FakeProcess:
    def __init__(self, stdout: str = "", stderr: str = "") -> None:
        self.stdin = io.StringIO()
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO(stderr)
        self.pid = 1234
        self._returncode: int | None = None
        self.terminated = False

    def poll(self) -> int | None:
        return self._returncode

    def terminate(self) -> None:
        self.terminated = True
        self._returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self._returncode = 0
        return 0

    def kill(self) -> None:
        self._returncode = -9


class _FakeAppServer:
    def __init__(self, *, cwd: Path | str | None = None) -> None:
        self.cwd = cwd
        self.pid = 456
        self.requests: list[tuple[str, dict]] = []
        self.request_timeouts: list[float] = []
        self.notification_timeouts: list[float] = []
        self.notification_timeouts_before_match = 0
        self.on_notification_wait = lambda _timeout: None
        self.notifications: list[dict] = []
        self.responses: dict[str, object | Exception] = {}
        self.stopped = False
        self.alive = True

    def request(
        self, method: str, params: dict | None = None, *, timeout: float = 30.0
    ) -> object:
        self.request_timeouts.append(timeout)
        payload = params or {}
        self.requests.append((method, payload))
        response = self.responses.get(method)
        if isinstance(response, Exception):
            raise response
        if response is not None:
            return response
        if method == "thread/start":
            return {"thread": {"id": "thread-new"}}
        if method == "thread/resume":
            return {"thread": {"id": payload["threadId"]}}
        if method == "turn/start":
            return {"turn": {"id": "turn-1"}}
        if method == "turn/interrupt":
            return {}
        return {}

    def notify(self, method: str, params: dict | None = None) -> None:
        self.requests.append((method, params or {}))

    def wait_notification(
        self,
        method: str,
        *,
        predicate: object = None,
        timeout: float = 30.0,
    ) -> dict:
        self.notification_timeouts.append(timeout)
        self.on_notification_wait(timeout)
        if self.notification_timeouts_before_match > 0:
            self.notification_timeouts_before_match -= 1
            raise TimeoutError(method)
        for index, notification in enumerate(self.notifications):
            if method != "*" and notification["method"] != method:
                continue
            params = notification["params"]
            if predicate is None or predicate(params):
                return self.notifications.pop(index)
        raise TimeoutError(method)

    def is_alive(self) -> bool:
        return self.alive

    def stop(self) -> None:
        self.stopped = True
        self.alive = False


# ---------------------------------------------------------------------------
# Typed fake collaborators for Protocol-based DI
# ---------------------------------------------------------------------------


class _FakeRunner:
    """Typed fake for :class:`fido.infra.ProcessRunner`."""

    def __init__(self, result: subprocess.CompletedProcess[str]) -> None:
        self.result = result
        self.calls: list[tuple[list[str], dict]] = []

    def run(
        self,
        cmd: list[str],
        *,
        check: bool = True,  # noqa: ARG002
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((cmd, dict(kwargs)))
        return self.result


class _FakeAppServerSpawner:
    """Typed fake for :class:`fido.codex.AppServerSpawner`."""

    def __init__(self, process: object) -> None:
        self._process = process
        self.spawned_cwd: object = None

    def spawn(self, *, cwd: object = None) -> object:
        self.spawned_cwd = cwd
        return self._process


class _FakeAppServerFactory:
    """Typed fake for :class:`fido.codex.AppServerFactory`."""

    def __init__(self, client: object) -> None:
        self._client = client
        self.created_cwd: object = None

    def create(self, *, cwd: object = None) -> object:
        self.created_cwd = cwd
        return self._client


class _PopAppServerFactory:
    """Typed fake for :class:`fido.codex.AppServerFactory` that pops from a list."""

    def __init__(self, clients: list) -> None:
        self._clients = clients

    def create(self, *, cwd: object = None) -> object:  # noqa: ARG002
        return self._clients.pop(0)


class _FakeClock:
    """Typed fake for :class:`fido.infra.Clock` with a fixed time."""

    def __init__(self, t: float = 0.0) -> None:
        self._t = t

    def sleep(self, secs: float) -> None:  # noqa: ARG002
        pass

    def monotonic(self) -> float:
        return self._t


class _AdvancingClock:
    """Typed fake for :class:`fido.infra.Clock` whose monotonic value advances
    by the duration each time :meth:`advance` is called."""

    def __init__(self) -> None:
        self._t = 0.0

    def sleep(self, secs: float) -> None:  # noqa: ARG002
        pass

    def monotonic(self) -> float:
        return self._t

    def advance(self, secs: float) -> None:
        self._t += secs


class _FakeSessionFactory:
    """Typed fake for :class:`fido.codex.SessionFactory`."""

    def __init__(self, session: object) -> None:
        self._session = session
        self.calls: list[dict] = []

    def create(self, system_file: object, **kwargs: object) -> object:
        self.calls.append({"system_file": system_file, **kwargs})
        return self._session


class _FakeSessionResolver:
    """Typed fake for :class:`fido.codex.SessionResolver`."""

    def __init__(self, session: object) -> None:
        self._session = session

    def resolve(self) -> object:
        return self._session


class TestCodexHelper:
    def test_calls_subprocess_run_with_prompt_and_cwd(self, tmp_path: Path) -> None:
        runner = _FakeRunner(_completed("out"))
        result = _codex(
            "exec",
            "--json",
            prompt="input",
            timeout=5,
            cwd=tmp_path,
            runner=runner,
        )
        assert len(runner.calls) == 1
        cmd, kwargs = runner.calls[0]
        assert cmd == ["codex", "exec", "--json"]
        assert kwargs["input"] == "input"
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["timeout"] == 5
        assert kwargs["cwd"] == tmp_path
        assert result.stdout == "out"


class TestCodexJsonlParsing:
    def test_extract_session_id_from_thread_started(self) -> None:
        assert (
            extract_session_id(_fixture("normal.jsonl"))
            == "67e55044-10b1-426f-9247-bb680e5fe0c8"
        )

    def test_extract_session_id_returns_last_thread_started(self) -> None:
        output = (
            '{"type":"thread.started","thread_id":"one"}\n'
            '{"type":"thread.started","thread_id":"two"}\n'
        )
        assert extract_session_id(output) == "two"

    def test_extract_session_id_tolerates_malformed_lines(self) -> None:
        output = 'not-json\n{"type":"thread.started","thread_id":"ok"}\n[]'
        assert extract_session_id(output) == "ok"

    def test_extract_result_text_from_last_agent_message(self) -> None:
        assert extract_result_text(_fixture("normal.jsonl")) == "final reply"

    def test_extract_result_text_ignores_non_agent_items(self) -> None:
        output = 'not-json\n{"type":"item.completed","item":{"id":"x","type":"reasoning","text":"ignore"}}\n{"type":"item.completed","item":{"id":"y","type":"agent_message","text":"ok"}}'
        assert extract_result_text(output) == "ok"


class TestCodexProviderErrors:
    def test_rate_limit_fixture_classified(self) -> None:
        with pytest.raises(CodexProviderError) as exc_info:
            raise_for_provider_error_output(_fixture("rate-limit.jsonl"))
        assert exc_info.value.kind == "rate_limit"
        assert "rate limit" in str(exc_info.value)

    def test_auth_fixture_classified(self) -> None:
        with pytest.raises(CodexProviderError) as exc_info:
            raise_for_provider_error_output(_fixture("auth-fail.jsonl"))
        assert exc_info.value.kind == "auth"
        assert "login" in str(exc_info.value)

    def test_cancelled_fixture_classified(self) -> None:
        with pytest.raises(CodexProviderError) as exc_info:
            raise_for_provider_error_output(_fixture("cancelled.jsonl"))
        assert exc_info.value.kind == "cancelled"
        assert "cancelled" in str(exc_info.value)

    def test_context_window_exceeded_raises_context_overflow_error(self) -> None:
        with pytest.raises(ContextOverflowError) as exc_info:
            raise_for_provider_error_output(_fixture("context-window-exceeded.jsonl"))
        assert "context window" in str(exc_info.value).lower()

    def test_ignores_successful_fixture(self) -> None:
        raise_for_provider_error_output(_fixture("normal.jsonl"))


class TestRunCodexExec:
    def test_builds_stable_json_exec_command(self, tmp_path: Path) -> None:
        runner = _FakeRunner(_completed(_fixture("normal.jsonl")))
        output = run_codex_exec(
            "hello",
            model=ProviderModel("gpt-5.5", "xhigh"),
            timeout=17,
            cwd=tmp_path,
            runner=runner,
        )
        assert extract_result_text(output) == "final reply"
        assert len(runner.calls) == 1
        cmd, kwargs = runner.calls[0]
        assert cmd == [
            "codex",
            "exec",
            "--json",
            "--model",
            "gpt-5.5",
            "--sandbox",
            "danger-full-access",
            "--ask-for-approval",
            "never",
            "--skip-git-repo-check",
            "-C",
            str(tmp_path),
            "-",
        ]
        assert kwargs["input"] == "hello"
        assert kwargs["timeout"] == 17
        assert kwargs["cwd"] == tmp_path.resolve()

    def test_handler_phase_uses_read_only_sandbox(self, tmp_path: Path) -> None:
        """Non-persistent exec with allowed_tools set runs read-only (#1672)."""
        from fido.provider import READ_ONLY_ALLOWED_TOOLS

        runner = _FakeRunner(_completed(_fixture("normal.jsonl")))
        run_codex_exec(
            "hello",
            model="gpt-5.5",
            cwd=tmp_path,
            runner=runner,
            allowed_tools=READ_ONLY_ALLOWED_TOOLS,
        )
        cmd, _ = runner.calls[0]
        assert cmd[cmd.index("--sandbox") + 1] == "read-only"

    def test_normalizes_relative_work_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        monkeypatch.chdir(tmp_path)
        runner = _FakeRunner(_completed(_fixture("normal.jsonl")))
        run_codex_exec("hello", model="gpt-5.5", cwd="work", runner=runner)
        cmd, kwargs = runner.calls[0]
        assert cmd[-2] == str(work_dir)
        assert kwargs["cwd"] == work_dir

    def test_raises_cli_error_on_nonzero_exit(self) -> None:
        runner = _FakeRunner(_completed(returncode=2, stderr="bad flags"))
        with pytest.raises(CodexCLIError) as exc_info:
            run_codex_exec("hello", model="gpt-5.5", runner=runner)
        assert exc_info.value.returncode == 2
        assert exc_info.value.stderr == "bad flags"

    def test_raises_provider_error_from_successful_process_output(self) -> None:
        runner = _FakeRunner(_completed(_fixture("rate-limit.jsonl")))
        with pytest.raises(CodexProviderError) as exc_info:
            run_codex_exec("hello", model="gpt-5.5", runner=runner)
        assert exc_info.value.kind == "rate_limit"

    def test_resume_builds_stable_json_exec_resume_command(
        self, tmp_path: Path
    ) -> None:
        prompt_file = tmp_path / "prompt.md"
        prompt_file.write_text("continue")
        runner = _FakeRunner(_completed(_fixture("normal.jsonl")))
        output = run_codex_exec_resume(
            "sess-1",
            prompt_file.read_text(),
            model=ProviderModel("gpt-5.5", "medium"),
            timeout=19,
            cwd=tmp_path,
            runner=runner,
        )
        assert extract_result_text(output) == "final reply"
        cmd, kwargs = runner.calls[0]
        assert cmd == [
            "codex",
            "exec",
            "--json",
            "--model",
            "gpt-5.5",
            "--sandbox",
            "danger-full-access",
            "--ask-for-approval",
            "never",
            "--skip-git-repo-check",
            "resume",
            "sess-1",
            "-",
        ]
        assert kwargs["input"] == "continue"
        assert kwargs["timeout"] == 19
        assert kwargs["cwd"] == tmp_path.resolve()


class TestCodexAppServerClient:
    def test_initializes_and_routes_response_by_id(self) -> None:
        process = _FakeProcess(
            '{"id":1,"result":{"serverInfo":{"name":"codex"}}}\n'
            '{"id":2,"result":{"ok":true}}\n'
        )
        client = CodexAppServerClient(process_factory=_FakeAppServerSpawner(process))
        assert client.request("example/method") == {"ok": True}
        written = process.stdin.getvalue().splitlines()
        assert '"method":"initialize"' in written[0]
        assert '"method":"initialized"' in written[1]
        assert '"method":"example/method"' in written[2]
        client.stop()

    def test_invalid_json_fails_loudly(self) -> None:
        process = _FakeProcess("not-json\n")
        with pytest.raises(CodexProtocolError, match="invalid JSON"):
            CodexAppServerClient(process_factory=_FakeAppServerSpawner(process))
        assert process.terminated

    def test_oversized_line_fails_loudly(self) -> None:
        process = _FakeProcess('{"id":1,"result":{}}\n')
        with pytest.raises(CodexProtocolError, match="line too large"):
            CodexAppServerClient(
                process_factory=_FakeAppServerSpawner(process),
                max_line_bytes=1,
            )
        assert process.terminated

    def test_pid_returns_process_pid(self) -> None:
        process = _FakeProcess('{"id":1,"result":{"serverInfo":{"name":"codex"}}}\n')
        client = CodexAppServerClient(process_factory=_FakeAppServerSpawner(process))
        assert client.pid == process.pid
        client.stop()

    def test_request_raises_provider_error_when_response_has_error(self) -> None:
        process = _FakeProcess(
            '{"id":1,"result":{"serverInfo":{"name":"codex"}}}\n'
            '{"id":2,"error":{"code":-32603,"message":"server is sad"}}\n'
        )
        client = CodexAppServerClient(process_factory=_FakeAppServerSpawner(process))
        with pytest.raises(Exception) as excinfo:
            client.request("explode")
        assert "server is sad" in str(excinfo.value)
        client.stop()

    def _streaming_process(
        self, prelude: str, lines: queue.Queue[str | None]
    ) -> "_FakeProcess":
        """Return a _FakeProcess whose stdout's readline pulls from
        ``lines`` instead of EOF'ing immediately.  Use this when a test
        needs to keep the reader thread alive past the initial setup
        message so the client stays in a non-protocol-error state."""

        class _StreamingStdout:
            def __init__(self) -> None:
                self._buf = io.StringIO(prelude)
                self._closed = False

            def readline(self) -> str:
                line = self._buf.readline()
                if line:
                    return line
                # Block until lines.put() pushes more content (or None
                # to signal EOF).
                next_line = lines.get()
                if next_line is None:
                    return ""
                return next_line

            def close(self) -> None:
                self._closed = True

            def __iter__(self) -> object:
                while True:
                    line = self.readline()
                    if not line:
                        return
                    yield line

        process = _FakeProcess()
        process.stdout = _StreamingStdout()  # type: ignore[assignment]
        return process

    def test_wait_notification_returns_matching_method(self) -> None:
        lines: queue.Queue[str | None] = queue.Queue()
        process = self._streaming_process(
            '{"id":1,"result":{"serverInfo":{"name":"codex"}}}\n', lines
        )
        client = CodexAppServerClient(process_factory=_FakeAppServerSpawner(process))

        def feed() -> None:
            lines.put('{"method":"unrelated/event","params":{"x":1}}\n')
            lines.put('{"method":"target/event","params":{"value":42}}\n')

        threading.Thread(target=feed, daemon=True).start()
        notif = client.wait_notification("target/event", timeout=2.0)
        assert notif["params"] == {"value": 42}
        lines.put(None)  # EOF the reader
        client.stop()

    def test_wait_notification_predicate_filters(self) -> None:
        lines: queue.Queue[str | None] = queue.Queue()
        process = self._streaming_process(
            '{"id":1,"result":{"serverInfo":{"name":"codex"}}}\n', lines
        )
        client = CodexAppServerClient(process_factory=_FakeAppServerSpawner(process))
        lines.put('{"method":"event","params":{"keep":false}}\n')
        lines.put('{"method":"event","params":{"keep":true,"value":7}}\n')
        notif = client.wait_notification(
            "event", predicate=lambda p: p.get("keep") is True, timeout=2.0
        )
        assert notif["params"]["value"] == 7
        lines.put(None)
        client.stop()

    def test_wait_notification_times_out(self) -> None:
        lines: queue.Queue[str | None] = queue.Queue()
        process = self._streaming_process(
            '{"id":1,"result":{"serverInfo":{"name":"codex"}}}\n', lines
        )
        client = CodexAppServerClient(process_factory=_FakeAppServerSpawner(process))
        with pytest.raises(TimeoutError, match="Timed out waiting for Codex"):
            client.wait_notification("never-arrives", timeout=0.1)
        lines.put(None)
        client.stop()

    def test_wait_notification_rejects_non_object_params(self) -> None:
        lines: queue.Queue[str | None] = queue.Queue()
        process = self._streaming_process(
            '{"id":1,"result":{"serverInfo":{"name":"codex"}}}\n', lines
        )
        client = CodexAppServerClient(process_factory=_FakeAppServerSpawner(process))
        lines.put('{"method":"event","params":"not-an-object"}\n')
        with pytest.raises(CodexProtocolError, match="params must be an object"):
            client.wait_notification("event", timeout=2.0)
        lines.put(None)
        client.stop()

    def test_is_alive_reflects_process_state(self) -> None:
        lines: queue.Queue[str | None] = queue.Queue()
        process = self._streaming_process(
            '{"id":1,"result":{"serverInfo":{"name":"codex"}}}\n', lines
        )
        client = CodexAppServerClient(process_factory=_FakeAppServerSpawner(process))
        assert client.is_alive()
        client.stop()
        assert not client.is_alive()
        lines.put(None)

    def test_stop_kills_process_when_terminate_times_out(self) -> None:
        class _Stubborn(_FakeProcess):
            def terminate(self) -> None:
                # Override so terminate doesn't immediately set _returncode
                # to 0; we need the subsequent wait() to time out and trigger
                # the kill+wait fallback (codex.py:303-305).
                self.terminated = True

            def wait(self, timeout: float | None = None) -> int:
                if self._returncode is None:
                    raise subprocess.TimeoutExpired(cmd=["codex"], timeout=timeout or 0)
                return self._returncode

            def kill(self) -> None:
                self._returncode = -9

        process = _Stubborn('{"id":1,"result":{"serverInfo":{"name":"codex"}}}\n')
        client = CodexAppServerClient(process_factory=_FakeAppServerSpawner(process))
        client.stop()
        assert process.terminated
        assert process._returncode == -9  # kill() ran via the timeout path


class TestCodexAPI:
    def test_maps_primary_secondary_and_reset_windows(self) -> None:
        fake = _FakeAppServer()
        fake.responses["account/rateLimits/read"] = {
            "rateLimits": [
                {
                    "limitId": "Main Limit",
                    "primary": {"usedPercent": 43, "resetsAt": 1_700_000_000},
                    "secondary": {"usedPercent": 91.2},
                }
            ]
        }
        api = CodexAPI(
            client_factory=_FakeAppServerFactory(fake), clock=_FakeClock(1.0)
        )
        snapshot = api.get_limit_snapshot()
        assert snapshot.provider == ProviderID.CODEX
        assert [window.name for window in snapshot.windows] == [
            "main_limit_primary",
            "main_limit_secondary",
        ]
        assert snapshot.windows[0].used == 43
        assert snapshot.windows[0].unit == "%"
        assert snapshot.windows[0].resets_at is not None
        assert fake.stopped

    def test_maps_credit_depletion_to_paused_window(self) -> None:
        fake = _FakeAppServer()
        fake.responses["account/rateLimits/read"] = {
            "rateLimits": [
                {
                    "limitId": "workspace",
                    "credits": {"hasCredits": False, "unlimited": False},
                    "rateLimitReachedType": "workspace_owner_credits_depleted",
                }
            ]
        }
        snapshot = CodexAPI(
            client_factory=_FakeAppServerFactory(fake)
        ).get_limit_snapshot()
        assert [window.name for window in snapshot.windows] == ["workspace_credits"]
        assert snapshot.closest_to_exhaustion().pressure == 1.0

    def test_does_not_mark_zero_credit_balance_depleted_without_reached_type(
        self,
    ) -> None:
        fake = _FakeAppServer()
        fake.responses["account/rateLimits/read"] = {
            "rateLimits": {
                "limitId": "codex",
                "primary": {"usedPercent": 1},
                "secondary": {"usedPercent": 0},
                "credits": {"balance": "0", "hasCredits": False, "unlimited": False},
                "rateLimitReachedType": None,
            }
        }
        snapshot = CodexAPI(
            client_factory=_FakeAppServerFactory(fake)
        ).get_limit_snapshot()
        assert [window.name for window in snapshot.windows] == [
            "codex_primary",
            "codex_secondary",
        ]

    def test_malformed_response_propagates_value_error(self) -> None:
        # {"rateLimits": "bad"} causes _codex_limit_windows to raise
        # ValueError — a shape regression, not a network/auth failure, so
        # it propagates loudly after the synthetic-success fix.
        fake = _FakeAppServer()
        fake.responses["account/rateLimits/read"] = {"rateLimits": "bad"}
        with pytest.raises(
            ValueError, match="Codex rateLimits must be an object or list"
        ):
            CodexAPI(client_factory=_FakeAppServerFactory(fake)).get_limit_snapshot()

    def test_client_factory_failure_is_unavailable_snapshot(self) -> None:
        # OSError is the realistic exception when the subprocess can't be spawned
        # (e.g. binary not found). It is in the narrowed catch set.
        class _FailFactory:
            def create(self, *, cwd: object = None) -> object:  # noqa: ARG002
                raise OSError("codex unavailable")

        snapshot = CodexAPI(client_factory=_FailFactory()).get_limit_snapshot()
        assert snapshot.provider == ProviderID.CODEX
        assert snapshot.unavailable_reason is not None
        assert "codex unavailable" in snapshot.unavailable_reason

    def test_caches_snapshot(self) -> None:
        fake = _FakeAppServer()
        fake.responses["account/rateLimits/read"] = {
            "rateLimits": [{"limitId": "main", "primary": {"usedPercent": 1}}]
        }
        clock = _FakeClock(1.0)
        api = CodexAPI(client_factory=_FakeAppServerFactory(fake), clock=clock)
        first = api.get_limit_snapshot()
        second = api.get_limit_snapshot()
        assert first is second
        assert [method for method, _ in fake.requests].count(
            "account/rateLimits/read"
        ) == 1


class TestCodexSession:
    def test_starts_thread_and_runs_prompt_turn(self, tmp_path: Path) -> None:
        system_file = tmp_path / "system.md"
        system_file.write_text("base")
        fake = _FakeAppServer()
        fake.notifications.extend(
            [
                {
                    "method": "turn/started",
                    "params": {
                        "threadId": "thread-new",
                        "turn": {"id": "turn-1", "status": "inProgress"},
                    },
                },
                {
                    "method": "item/completed",
                    "params": {
                        "threadId": "thread-new",
                        "turnId": "turn-1",
                        "item": {"type": "agentMessage", "text": "reply"},
                    },
                },
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": "thread-new",
                        "turn": {"id": "turn-1", "status": "completed"},
                    },
                },
            ]
        )
        session = CodexSession(
            system_file,
            work_dir=tmp_path,
            model=ProviderModel("gpt-5.5", "medium"),
            client_factory=_FakeAppServerFactory(fake),
        )
        assert session.session_id == "thread-new"
        assert session.prompt("hello") == "reply"
        methods = [method for method, _ in fake.requests]
        assert methods == ["thread/start", "turn/start"]
        turn_params = fake.requests[1][1]
        assert turn_params["model"] == "gpt-5.5"
        assert turn_params["effort"] == "medium"
        assert turn_params["sandboxPolicy"] == {"type": "dangerFullAccess"}
        assert turn_params["input"] == [
            {"type": "text", "text": "base\n\nhello", "text_elements": []}
        ]
        assert session.sent_count == 1
        assert session.received_count == 1

    def test_prompt_with_allowed_tools_uses_read_only_sandbox(
        self, tmp_path: Path
    ) -> None:
        """Handler / synthesis / rescope / voice / status phases pass an
        explicit allowlist; codex's per-turn sandbox runs read-only so
        the model cannot Edit/Write/Bash even though the codex runtime
        has no `--allowedTools` primitive (#1672)."""
        from fido.provider import READ_ONLY_ALLOWED_TOOLS

        system_file = tmp_path / "system.md"
        system_file.write_text("base")
        fake = _FakeAppServer()
        fake.notifications.extend(
            [
                {
                    "method": "turn/started",
                    "params": {
                        "threadId": "thread-new",
                        "turn": {"id": "turn-1", "status": "inProgress"},
                    },
                },
                {
                    "method": "item/completed",
                    "params": {
                        "threadId": "thread-new",
                        "turnId": "turn-1",
                        "item": {"type": "agentMessage", "text": "reply"},
                    },
                },
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": "thread-new",
                        "turn": {"id": "turn-1", "status": "completed"},
                    },
                },
            ]
        )
        session = CodexSession(
            system_file,
            work_dir=tmp_path,
            model=ProviderModel("gpt-5.5", "medium"),
            client_factory=_FakeAppServerFactory(fake),
        )
        assert session.prompt("hello", allowed_tools=READ_ONLY_ALLOWED_TOOLS) == (
            "reply"
        )
        turn_params = fake.requests[1][1]
        assert turn_params["sandboxPolicy"] == {"type": "readOnly"}

    def test_resumes_persisted_thread_id(self, tmp_path: Path) -> None:
        system_file = tmp_path / "system.md"
        system_file.write_text("")
        fake = _FakeAppServer()
        session = CodexSession(
            system_file,
            work_dir=tmp_path,
            model="gpt-5.5",
            session_id="persisted",
            client_factory=_FakeAppServerFactory(fake),
        )
        assert session.session_id == "persisted"
        assert fake.requests[0] == (
            "thread/resume",
            {
                "model": "gpt-5.5",
                "cwd": str(tmp_path.resolve()),
                "approvalPolicy": "never",
                "sandbox": "danger-full-access",
                "developerInstructions": "",
                "threadId": "persisted",
                "excludeTurns": True,
            },
        )

    def test_stale_resume_starts_new_thread_and_counts_drop(
        self, tmp_path: Path
    ) -> None:
        system_file = tmp_path / "system.md"
        system_file.write_text("")
        fake = _FakeAppServer()
        fake.responses["thread/resume"] = CodexProviderError(message="missing")
        session = CodexSession(
            system_file,
            work_dir=tmp_path,
            model="gpt-5.5",
            session_id="stale",
            client_factory=_FakeAppServerFactory(fake),
        )
        assert session.session_id == "thread-new"
        assert session.dropped_session_count == 1
        assert [method for method, _ in fake.requests] == [
            "thread/resume",
            "thread/start",
        ]

    def test_cancel_interrupts_active_turn_and_next_turn_is_clean(
        self, tmp_path: Path
    ) -> None:
        system_file = tmp_path / "system.md"
        system_file.write_text("")
        fake = _FakeAppServer()
        fake.notifications.append(
            {
                "method": "turn/completed",
                "params": {
                    "threadId": "thread-new",
                    "turn": {"id": "turn-1", "status": "interrupted"},
                },
            }
        )
        session = CodexSession(
            system_file,
            work_dir=tmp_path,
            model="gpt-5.5",
            client_factory=_FakeAppServerFactory(fake),
        )
        session.send("work")
        session._fire_worker_cancel()
        assert session.consume_until_result() == ""
        assert session.last_turn_cancelled
        assert fake.requests[-1] == (
            "turn/interrupt",
            {"threadId": "thread-new", "turnId": "turn-1"},
        )
        assert fake.request_timeouts[-1] == 5

    def test_interrupt_timeout_is_recoverable_provider_wedge(
        self, tmp_path: Path
    ) -> None:
        system_file = tmp_path / "system.md"
        system_file.write_text("")
        fake = _FakeAppServer()
        fake.responses["turn/interrupt"] = TimeoutError("turn/interrupt")
        session = CodexSession(
            system_file,
            work_dir=tmp_path,
            model="gpt-5.5",
            client_factory=_FakeAppServerFactory(fake),
        )
        session.send("work")

        with pytest.raises(ProviderInterruptTimeout) as exc_info:
            session._fire_worker_cancel()

        assert isinstance(exc_info.value.__cause__, TimeoutError)
        assert "thread-new" in str(exc_info.value)
        assert "turn-1" in str(exc_info.value)
        assert fake.request_timeouts[-1] == 5

    def test_failed_turn_raises_provider_error(self, tmp_path: Path) -> None:
        system_file = tmp_path / "system.md"
        system_file.write_text("")
        fake = _FakeAppServer()
        fake.notifications.append(
            {
                "method": "turn/completed",
                "params": {
                    "threadId": "thread-new",
                    "turn": {"id": "turn-1", "status": "failed"},
                    "error": {"message": "rate limit reached"},
                },
            }
        )
        session = CodexSession(
            system_file,
            work_dir=tmp_path,
            model="gpt-5.5",
            client_factory=_FakeAppServerFactory(fake),
        )
        session.send("work")
        with pytest.raises(CodexProviderError) as exc_info:
            session.consume_until_result()
        assert exc_info.value.kind == "rate_limit"

    def test_turn_consumption_tolerates_quiet_notification_polls(
        self, tmp_path: Path
    ) -> None:
        system_file = tmp_path / "system.md"
        system_file.write_text("")
        fake = _FakeAppServer()
        clock = _AdvancingClock()
        fake.notification_timeouts_before_match = 2
        fake.on_notification_wait = clock.advance
        fake.notifications.extend(
            [
                {
                    "method": "item/completed",
                    "params": {
                        "threadId": "thread-new",
                        "turnId": "turn-1",
                        "item": {"type": "agentMessage", "text": "reply"},
                    },
                },
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": "thread-new",
                        "turn": {"id": "turn-1", "status": "completed"},
                    },
                },
            ]
        )
        session = CodexSession(
            system_file,
            work_dir=tmp_path,
            model="gpt-5.5",
            client_factory=_FakeAppServerFactory(fake),
            turn_idle_timeout=10.0,
            clock=clock,
        )
        session.send("work")
        assert session.consume_until_result() == "reply"
        assert fake.notification_timeouts[:3] == [1.0, 1.0, 1.0]

    def test_turn_consumption_times_out_after_idle_budget(self, tmp_path: Path) -> None:
        system_file = tmp_path / "system.md"
        system_file.write_text("")
        fake = _FakeAppServer()
        replacement = _FakeAppServer()
        clients = [fake, replacement]
        clock = _AdvancingClock()
        fake.notification_timeouts_before_match = 100
        fake.on_notification_wait = clock.advance
        session = CodexSession(
            system_file,
            work_dir=tmp_path,
            model="gpt-5.5",
            client_factory=_PopAppServerFactory(clients),
            turn_idle_timeout=2.5,
            clock=clock,
        )
        session.send("work")
        with pytest.raises(TimeoutError, match="Codex turn activity"):
            session.consume_until_result()
        assert fake.notification_timeouts == [1.0, 1.0, 0.5]
        assert fake.stopped
        assert replacement.requests[0][0] == "thread/resume"
        assert replacement.requests[0][1]["threadId"] == "thread-new"

    def test_recover_clears_stale_cancelled_turn_state(self, tmp_path: Path) -> None:
        system_file = tmp_path / "system.md"
        system_file.write_text("")
        fake = _FakeAppServer()
        replacement = _FakeAppServer()
        clients = [fake, replacement]
        session = CodexSession(
            system_file,
            work_dir=tmp_path,
            model="gpt-5.5",
            client_factory=_PopAppServerFactory(clients),
        )

        session.send("work")
        with session._state_lock:
            session._last_turn_cancelled = True

        session.recover()
        session._fire_worker_cancel()

        assert fake.stopped
        assert session.last_turn_cancelled is False
        # ``thread/resume`` carries the full session context per #1077:
        # model, cwd, approvalPolicy, sandbox, developerInstructions, plus
        # threadId and excludeTurns.  Anchor on the threadId + excludeTurns
        # rather than the full dict so future protocol enrichments don't
        # break this test.
        assert len(replacement.requests) == 1
        method, params = replacement.requests[0]
        assert method == "thread/resume"
        assert params["threadId"] == "thread-new"
        assert params["excludeTurns"] is True

    def test_consume_pending_cancel_reads_and_clears(self, tmp_path: Path) -> None:
        """codex P1 on PR #1793: atomic read+clear."""
        system_file = tmp_path / "system.md"
        system_file.write_text("")
        fake = _FakeAppServer()
        session = CodexSession(
            system_file,
            work_dir=tmp_path,
            model="gpt-5.5",
            client_factory=_FakeAppServerFactory(fake),
        )
        with session._state_lock:
            session._last_turn_cancelled = True
        assert session.consume_pending_cancel() is True
        assert session.consume_pending_cancel() is False
        assert session.last_turn_cancelled is False

    def test_prompt_preserves_cancel_observed_from_inner(self, tmp_path: Path) -> None:
        """codex P1 on PR #1793: outer wrapper preserves
        ``cancel_observed`` already attached by the inner try/except."""
        system_file = tmp_path / "system.md"
        system_file.write_text("")
        fake = _FakeAppServer()
        session = CodexSession(
            system_file,
            work_dir=tmp_path,
            model="gpt-5.5",
            client_factory=_FakeAppServerFactory(fake),
        )
        boom = BrokenPipeError("codex pipe gone")
        boom.cancel_observed = True  # type: ignore[attr-defined]
        session._prompt_inner = MagicMock(  # type: ignore[method-assign]
            side_effect=boom
        )
        with pytest.raises(BrokenPipeError) as exc_info:
            session.prompt("hi")
        assert exc_info.value.cancel_observed is True

    def test_prompt_inner_captures_cancel_on_send_failure(self, tmp_path: Path) -> None:
        """codex P1 on PR #1793: inner try/except captures the sticky
        cancel bit INSIDE the OwnedSession lock and attaches to the
        raised exception."""
        system_file = tmp_path / "system.md"
        system_file.write_text("")
        fake = _FakeAppServer()
        session = CodexSession(
            system_file,
            work_dir=tmp_path,
            model="gpt-5.5",
            client_factory=_FakeAppServerFactory(fake),
        )

        # Override ``send`` to simulate a peer-fired cancel mid-send.
        def failing_send(*_a: object, **_k: object) -> None:
            with session._state_lock:
                session._last_turn_cancelled = True
            raise BrokenPipeError("codex pipe gone")

        session.send = failing_send  # type: ignore[method-assign]
        with pytest.raises(BrokenPipeError) as exc_info:
            session.prompt("hi")
        assert exc_info.value.cancel_observed is True

    def test_prompt_attaches_false_cancel_observed_when_inner_lacks_attr(
        self, tmp_path: Path
    ) -> None:
        """codex P1 on PR #1793: when ``_prompt_inner`` raises an
        exception that lacks ``cancel_observed`` (e.g. ``__enter__``
        failure before the inner try/except), the outer wrapper
        defaults ``cancel_observed=False`` — leftover sticky True from
        a previous turn must not be misread as current-attempt
        cancel."""
        system_file = tmp_path / "system.md"
        system_file.write_text("")
        fake = _FakeAppServer()
        session = CodexSession(
            system_file,
            work_dir=tmp_path,
            model="gpt-5.5",
            client_factory=_FakeAppServerFactory(fake),
        )
        # Replace _prompt_inner to raise a bare exception (no
        # ``cancel_observed`` attached) — simulates ``__enter__``
        # failure path.
        session._prompt_inner = MagicMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("enter failed")
        )
        with pytest.raises(RuntimeError, match="enter failed") as exc_info:
            session.prompt("hi")
        assert exc_info.value.cancel_observed is False

    def test_snapshot_publisher_fires_after_send(self, tmp_path: Path) -> None:
        system_file = tmp_path / "system.md"
        system_file.write_text("base")
        fake = _FakeAppServer()
        fake.notifications.extend(
            [
                {
                    "method": "item/completed",
                    "params": {
                        "threadId": "thread-new",
                        "turnId": "turn-1",
                        "item": {"type": "agentMessage", "text": "ok"},
                    },
                },
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": "thread-new",
                        "turn": {"id": "turn-1", "status": "completed"},
                    },
                },
            ]
        )
        published: list[dict[str, object]] = []

        class Recorder:
            def publish_metrics(self, **kwargs: object) -> None:
                published.append(kwargs)

        session = CodexSession(
            system_file,
            work_dir=tmp_path,
            model=ProviderModel("gpt-5.5", "medium"),
            client_factory=_FakeAppServerFactory(fake),
            snapshot_publisher=Recorder(),
        )
        assert session.prompt("hello") == "ok"
        # sent publication fires first, then received publication after the
        # item/completed event increments the counter.
        assert len(published) == 2
        assert published[0]["sent_count"] == 1
        assert published[0]["received_count"] == 0
        assert published[1]["sent_count"] == 1
        assert published[1]["received_count"] == 1

    def test_snapshot_publisher_none_is_noop(self, tmp_path: Path) -> None:
        system_file = tmp_path / "system.md"
        system_file.write_text("base")
        fake = _FakeAppServer()
        fake.notifications.extend(
            [
                {
                    "method": "item/completed",
                    "params": {
                        "threadId": "thread-new",
                        "turnId": "turn-1",
                        "item": {"type": "agentMessage", "text": "ok"},
                    },
                },
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": "thread-new",
                        "turn": {"id": "turn-1", "status": "completed"},
                    },
                },
            ]
        )
        session = CodexSession(
            system_file,
            work_dir=tmp_path,
            model=ProviderModel("gpt-5.5", "medium"),
            client_factory=_FakeAppServerFactory(fake),
        )
        # No publisher — prompt must not raise.
        assert session.prompt("hello") == "ok"
        assert session.sent_count == 1


class TestCodexClient:
    def test_model_slots_and_provider_id(self) -> None:
        client = CodexClient(session=MagicMock())
        assert client.provider_id == ProviderID.CODEX
        assert client.voice_model == ProviderModel("gpt-5.5", "xhigh")
        assert client.work_model == ProviderModel("gpt-5.5", "medium")
        assert client.brief_model == ProviderModel("gpt-5.5", "low")

    def test_supports_no_commit_reset_is_true(self) -> None:
        assert CodexClient(session=MagicMock()).supports_no_commit_reset is True

    def test_spawns_owned_session_with_resume_id(self, tmp_path: Path) -> None:
        system_file = tmp_path / "system.md"
        system_file.write_text("persona")
        session = MagicMock()
        factory = _FakeSessionFactory(session)
        client = CodexClient(
            session_system_file=system_file,
            work_dir=tmp_path,
            repo_name="owner/repo",
            session_factory=factory,
        )
        client.ensure_session(client.voice_model, session_id="thread-1")
        assert len(factory.calls) == 1
        call = factory.calls[0]
        assert call["system_file"] == system_file
        assert call["work_dir"] == tmp_path
        assert call["model"] == client.voice_model
        assert call["repo_name"] == "owner/repo"
        assert call["session_id"] == "thread-1"
        assert call["snapshot_publisher"] is client
        assert client.session is session

    def test_run_turn_through_fake_codex_session(self, tmp_path: Path) -> None:
        system_file = tmp_path / "system.md"
        system_file.write_text("base")
        fake = _FakeAppServer()
        fake.notifications.extend(
            [
                {
                    "method": "item/completed",
                    "params": {
                        "threadId": "thread-new",
                        "turnId": "turn-1",
                        "item": {"type": "agentMessage", "text": "reply"},
                    },
                },
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": "thread-new",
                        "turn": {"id": "turn-1", "status": "completed"},
                    },
                },
            ]
        )

        class _WrappingSessionFactory:
            def create(
                self,
                system_file: object,
                **kwargs: object,
            ) -> object:
                return CodexSession(
                    system_file,  # type: ignore[arg-type]
                    **kwargs,  # type: ignore[arg-type]
                    client_factory=_FakeAppServerFactory(fake),
                )

        client = CodexClient(
            session_system_file=system_file,
            work_dir=tmp_path,
            session_factory=_WrappingSessionFactory(),
        )
        assert client.run_turn("hello", model=client.work_model) == "reply"
        assert [method for method, _ in fake.requests] == ["thread/start", "turn/start"]
        assert fake.requests[1][1]["effort"] == "medium"

    def test_provider_errors_pass_through(self) -> None:
        session = MagicMock()
        session.prompt.side_effect = CodexProviderError(message="rate limit")
        client = CodexClient(session=session)
        with pytest.raises(CodexProviderError, match="rate limit"):
            client.run_turn("hello", model=client.work_model)

    def test_file_helpers_use_exec_paths(self, tmp_path: Path) -> None:
        system_file = tmp_path / "system.md"
        prompt_file = tmp_path / "prompt.md"
        system_file.write_text("system")
        prompt_file.write_text("prompt")
        runner = _FakeRunner(_completed(_fixture("normal.jsonl")))
        client = CodexClient(runner=runner)

        assert client.print_prompt_from_file(
            system_file, prompt_file, client.work_model, cwd=tmp_path
        )
        assert runner.calls[0][1]["input"] == "system\n\nprompt"

        runner.result = _completed(_fixture("normal.jsonl"))
        client.resume_session("sess-1", prompt_file, client.brief_model, cwd=tmp_path)
        assert runner.calls[1][0][-3:] == ["resume", "sess-1", "-"]

    def test_extract_session_id(self) -> None:
        client = CodexClient(session=MagicMock())
        assert (
            client.extract_session_id(
                '{"type":"thread.started","thread_id":"codex-sess"}'
            )
            == "codex-sess"
        )


class TestCodex:
    def test_default_provider_id_and_injected_components(self) -> None:
        api = MagicMock()
        agent = MagicMock()
        provider = Codex(api=api, agent=agent)
        assert provider.provider_id == ProviderID.CODEX
        assert provider.api is api
        assert provider.agent is agent

    def test_default_agent_receives_session(self) -> None:
        session = MagicMock()
        provider = Codex(session=session)
        assert provider.agent.session is session

    def test_injected_agent_receives_session(self) -> None:
        agent = MagicMock()
        session = MagicMock()
        Codex(agent=agent, session=session)
        agent.attach_session.assert_called_once_with(session)
