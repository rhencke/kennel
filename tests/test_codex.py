import io
import subprocess
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
from fido.provider import ProviderID, ProviderInterruptTimeout, ProviderModel

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
        predicate=None,
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


class TestCodexHelper:
    def test_calls_subprocess_run_with_prompt_and_cwd(self, tmp_path: Path) -> None:
        mock_run = MagicMock(return_value=_completed("out"))
        result = _codex(
            "exec",
            "--json",
            prompt="input",
            timeout=5,
            cwd=tmp_path,
            runner=mock_run,
        )
        mock_run.assert_called_once_with(
            ["codex", "exec", "--json"],
            input="input",
            capture_output=True,
            text=True,
            timeout=5,
            cwd=tmp_path,
        )
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
        output = "\n".join(
            [
                "not-json",
                '{"type":"item.completed","item":{"id":"x","type":"reasoning","text":"ignore"}}',
                '{"type":"item.completed","item":{"id":"y","type":"agent_message","text":"ok"}}',
            ]
        )
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

    def test_ignores_successful_fixture(self) -> None:
        raise_for_provider_error_output(_fixture("normal.jsonl"))


class TestRunCodexExec:
    def test_builds_stable_json_exec_command(self, tmp_path: Path) -> None:
        mock_run = MagicMock(return_value=_completed(_fixture("normal.jsonl")))
        output = run_codex_exec(
            "hello",
            model=ProviderModel("gpt-5.5", "xhigh"),
            timeout=17,
            cwd=tmp_path,
            runner=mock_run,
        )
        assert extract_result_text(output) == "final reply"
        mock_run.assert_called_once()
        cmd = mock_run.call_args.args[0]
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
        assert mock_run.call_args.kwargs["input"] == "hello"
        assert mock_run.call_args.kwargs["timeout"] == 17
        assert mock_run.call_args.kwargs["cwd"] == tmp_path.resolve()

    def test_normalizes_relative_work_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        monkeypatch.chdir(tmp_path)
        mock_run = MagicMock(return_value=_completed(_fixture("normal.jsonl")))
        run_codex_exec("hello", model="gpt-5.5", cwd="work", runner=mock_run)
        assert mock_run.call_args.args[0][-2] == str(work_dir)
        assert mock_run.call_args.kwargs["cwd"] == work_dir

    def test_raises_cli_error_on_nonzero_exit(self) -> None:
        mock_run = MagicMock(return_value=_completed(returncode=2, stderr="bad flags"))
        with pytest.raises(CodexCLIError) as exc_info:
            run_codex_exec("hello", model="gpt-5.5", runner=mock_run)
        assert exc_info.value.returncode == 2
        assert exc_info.value.stderr == "bad flags"

    def test_raises_provider_error_from_successful_process_output(self) -> None:
        mock_run = MagicMock(return_value=_completed(_fixture("rate-limit.jsonl")))
        with pytest.raises(CodexProviderError) as exc_info:
            run_codex_exec("hello", model="gpt-5.5", runner=mock_run)
        assert exc_info.value.kind == "rate_limit"

    def test_resume_builds_stable_json_exec_resume_command(
        self, tmp_path: Path
    ) -> None:
        prompt_file = tmp_path / "prompt.md"
        prompt_file.write_text("continue")
        mock_run = MagicMock(return_value=_completed(_fixture("normal.jsonl")))
        output = run_codex_exec_resume(
            "sess-1",
            prompt_file.read_text(),
            model=ProviderModel("gpt-5.5", "medium"),
            timeout=19,
            cwd=tmp_path,
            runner=mock_run,
        )
        assert extract_result_text(output) == "final reply"
        cmd = mock_run.call_args.args[0]
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
        assert mock_run.call_args.kwargs["input"] == "continue"
        assert mock_run.call_args.kwargs["timeout"] == 19
        assert mock_run.call_args.kwargs["cwd"] == tmp_path.resolve()


class TestCodexAppServerClient:
    def test_initializes_and_routes_response_by_id(self) -> None:
        process = _FakeProcess(
            '{"id":1,"result":{"serverInfo":{"name":"codex"}}}\n'
            '{"id":2,"result":{"ok":true}}\n'
        )
        client = CodexAppServerClient(process_factory=lambda **_: process)
        assert client.request("example/method") == {"ok": True}
        written = process.stdin.getvalue().splitlines()
        assert '"method":"initialize"' in written[0]
        assert '"method":"initialized"' in written[1]
        assert '"method":"example/method"' in written[2]
        client.stop()

    def test_invalid_json_fails_loudly(self) -> None:
        process = _FakeProcess("not-json\n")
        with pytest.raises(CodexProtocolError, match="invalid JSON"):
            CodexAppServerClient(process_factory=lambda **_: process)
        assert process.terminated

    def test_oversized_line_fails_loudly(self) -> None:
        process = _FakeProcess('{"id":1,"result":{}}\n')
        with pytest.raises(CodexProtocolError, match="line too large"):
            CodexAppServerClient(
                process_factory=lambda **_: process,
                max_line_bytes=1,
            )
        assert process.terminated


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
        api = CodexAPI(client_factory=lambda: fake, monotonic=lambda: 1.0)
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
        snapshot = CodexAPI(client_factory=lambda: fake).get_limit_snapshot()
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
        snapshot = CodexAPI(client_factory=lambda: fake).get_limit_snapshot()
        assert [window.name for window in snapshot.windows] == [
            "codex_primary",
            "codex_secondary",
        ]

    def test_malformed_response_is_unavailable_snapshot(self) -> None:
        fake = _FakeAppServer()
        fake.responses["account/rateLimits/read"] = {"rateLimits": "bad"}
        snapshot = CodexAPI(client_factory=lambda: fake).get_limit_snapshot()
        assert snapshot.provider == ProviderID.CODEX
        assert snapshot.unavailable_reason is not None

    def test_client_factory_failure_is_unavailable_snapshot(self) -> None:
        def fail() -> _FakeAppServer:
            raise RuntimeError("codex unavailable")

        snapshot = CodexAPI(client_factory=fail).get_limit_snapshot()
        assert snapshot.provider == ProviderID.CODEX
        assert snapshot.unavailable_reason is not None
        assert "codex unavailable" in snapshot.unavailable_reason

    def test_caches_snapshot(self) -> None:
        fake = _FakeAppServer()
        fake.responses["account/rateLimits/read"] = {
            "rateLimits": [{"limitId": "main", "primary": {"usedPercent": 1}}]
        }
        now = 1.0
        api = CodexAPI(client_factory=lambda: fake, monotonic=lambda: now)
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
            client_factory=lambda **_: fake,
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

    def test_resumes_persisted_thread_id(self, tmp_path: Path) -> None:
        system_file = tmp_path / "system.md"
        system_file.write_text("")
        fake = _FakeAppServer()
        session = CodexSession(
            system_file,
            work_dir=tmp_path,
            model="gpt-5.5",
            session_id="persisted",
            client_factory=lambda **_: fake,
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
            client_factory=lambda **_: fake,
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
            client_factory=lambda **_: fake,
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
            client_factory=lambda **_: fake,
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
            client_factory=lambda **_: fake,
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
        now = 0.0

        def clock() -> float:
            return now

        def advance(timeout: float) -> None:
            nonlocal now
            now += timeout

        fake.notification_timeouts_before_match = 2
        fake.on_notification_wait = advance
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
            client_factory=lambda **_: fake,
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
        now = 0.0

        def clock() -> float:
            return now

        def advance(timeout: float) -> None:
            nonlocal now
            now += timeout

        fake.notification_timeouts_before_match = 100
        fake.on_notification_wait = advance
        session = CodexSession(
            system_file,
            work_dir=tmp_path,
            model="gpt-5.5",
            client_factory=lambda **_: clients.pop(0),
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


class TestCodexClient:
    def test_model_slots_and_provider_id(self) -> None:
        client = CodexClient(session=MagicMock())
        assert client.provider_id == ProviderID.CODEX
        assert client.voice_model == ProviderModel("gpt-5.5", "xhigh")
        assert client.work_model == ProviderModel("gpt-5.5", "medium")
        assert client.brief_model == ProviderModel("gpt-5.5", "low")

    def test_spawns_owned_session_with_resume_id(self, tmp_path: Path) -> None:
        system_file = tmp_path / "system.md"
        system_file.write_text("persona")
        session = MagicMock()
        factory = MagicMock(return_value=session)
        client = CodexClient(
            session_system_file=system_file,
            work_dir=tmp_path,
            repo_name="owner/repo",
            session_factory=factory,
        )
        client.ensure_session(client.voice_model, session_id="thread-1")
        factory.assert_called_once_with(
            system_file,
            work_dir=tmp_path,
            model=client.voice_model,
            repo_name="owner/repo",
            session_id="thread-1",
        )
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
        client = CodexClient(
            session_system_file=system_file,
            work_dir=tmp_path,
            session_factory=lambda *args, **kwargs: CodexSession(
                *args,
                **kwargs,
                client_factory=lambda **_: fake,
            ),
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
        runner = MagicMock(return_value=_completed(_fixture("normal.jsonl")))
        client = CodexClient(runner=runner)

        assert client.print_prompt_from_file(
            system_file, prompt_file, client.work_model, cwd=tmp_path
        )
        assert runner.call_args.kwargs["input"] == "system\n\nprompt"

        client.resume_session("sess-1", prompt_file, client.brief_model, cwd=tmp_path)
        assert runner.call_args.args[0][-3:] == ["resume", "sess-1", "-"]

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
