"""Tests for Gemini CLI provider."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

from kennel.gemini import (
    Gemini,
    GeminiACPRuntime,
    GeminiAPI,
    GeminiClient,
    GeminiSession,
    _GeminiACPClient,
    extract_result_text,
    extract_session_id,
)
from kennel.provider import (
    PromptSession,
    ProviderAgent,
    ProviderID,
    ProviderLimitSnapshot,
    ProviderModel,
)


def _completed(
    stdout: str = "", returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _gemini_output(text: str = "OK", session_id: str = "sess-123") -> str:
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


def test_extract_result_text() -> None:
    output = _gemini_output("hello world")
    assert extract_result_text(output) == "hello world"
    assert extract_result_text("") == ""
    assert extract_result_text("invalid") == ""

    # Missing data or content
    assert extract_result_text(json.dumps({"type": "assistant.message"})) == ""
    assert (
        extract_result_text(json.dumps({"type": "assistant.message", "data": {}})) == ""
    )


def test_extract_session_id() -> None:
    output = _gemini_output("OK", "sess-456")
    assert extract_session_id(output) == "sess-456"
    assert extract_session_id("") == ""
    assert extract_session_id("invalid") == ""


class TestGeminiClient:
    def test_provider_id(self) -> None:
        client = GeminiClient()
        assert client.provider_id == ProviderID.GEMINI

    def test_extract_session_id(self) -> None:
        client = GeminiClient()
        assert client.extract_session_id(_gemini_output(session_id="s1")) == "s1"

    def test_run_cli_prompt(self, tmp_path: Path) -> None:
        runner = MagicMock(return_value=_completed(_gemini_output("clipped")))
        client = GeminiClient(runner=runner, work_dir=tmp_path)

        result = client._run_cli_prompt("hello", model="m1", timeout=10)
        assert result == _gemini_output("clipped")

        runner.assert_called_once()
        args, kwargs = runner.call_args
        assert args[0] == ["gemini", "-p", "hello", "--output-format", "json"]
        assert kwargs["timeout"] == 10

    def test_run_cli_prompt_with_session(self, tmp_path: Path) -> None:
        runner = MagicMock(return_value=_completed(_gemini_output()))
        client = GeminiClient(runner=runner, work_dir=tmp_path)

        client._run_cli_prompt("hello", model="m1", timeout=10, session_id="s1")
        assert runner.call_args[0][0] == [
            "gemini",
            "-p",
            "hello",
            "--resume",
            "s1",
            "--output-format",
            "json",
        ]

    def test_run_cli_prompt_failure(self, tmp_path: Path) -> None:
        runner = MagicMock(return_value=_completed("error", returncode=1))
        client = GeminiClient(runner=runner, work_dir=tmp_path)
        assert client._run_cli_prompt("hi", model="m", timeout=1) == ""

    def test_print_prompt_from_file(self, tmp_path: Path) -> None:
        runner = MagicMock(return_value=_completed(_gemini_output("file-out")))
        client = GeminiClient(runner=runner, work_dir=tmp_path)

        sys_file = tmp_path / "sys.md"
        sys_file.write_text("sys")
        prompt_file = tmp_path / "prompt.md"
        prompt_file.write_text("prompt")

        result = client.print_prompt_from_file(
            sys_file, prompt_file, ProviderModel("m")
        )
        assert result == _gemini_output("file-out")

    def test_resume_session(self, tmp_path: Path) -> None:
        runner = MagicMock(return_value=_completed(_gemini_output("resumed")))
        client = GeminiClient(runner=runner, work_dir=tmp_path)

        prompt_file = tmp_path / "prompt.md"
        prompt_file.write_text("prompt")

        result = client.resume_session("s1", prompt_file, ProviderModel("m"))
        assert result == _gemini_output("resumed")

    def test_spawn_owned_session(self, tmp_path: Path) -> None:
        session_factory = MagicMock()
        client = GeminiClient(
            session_factory=session_factory,
            session_system_file=tmp_path / "sys.md",
            work_dir=tmp_path,
        )
        client._spawn_owned_session(ProviderModel("m"), session_id="s1")
        session_factory.assert_called_once_with(
            tmp_path / "sys.md",
            work_dir=tmp_path,
            model=ProviderModel("m"),
            repo_name=None,
            session_id="s1",
        )


class TestGeminiACPRuntime:
    def test_init_and_client_factory(self, tmp_path: Path) -> None:
        mock_spawn = MagicMock()
        runtime = GeminiACPRuntime(
            work_dir=tmp_path,
            spawn_agent_process=mock_spawn,
        )
        assert runtime.provider_id == ProviderID.GEMINI
        client = runtime._default_client_factory(runtime)
        assert isinstance(client, _GeminiACPClient)
        runtime.stop()


class TestGeminiSession:
    def test_init_defaults_runtime(self, tmp_path: Path) -> None:
        sys_file = tmp_path / "sys.md"
        sys_file.write_text("")

        mock_runtime = MagicMock(spec=GeminiACPRuntime)
        mock_runtime.ensure_session.return_value = "s1"
        factory = MagicMock(return_value=mock_runtime)

        session = GeminiSession(
            sys_file,
            work_dir=tmp_path,
            model="m",
            runtime_factory=factory,
        )
        assert session._runtime is mock_runtime
        factory.assert_called_once_with(work_dir=tmp_path, repo_name=None)
        session.stop()


class TestGemini:
    def test_properties(self) -> None:
        api = GeminiAPI()
        agent = GeminiClient()
        gemini = Gemini(api=api, agent=agent)
        assert gemini.provider_id == ProviderID.GEMINI
        assert gemini.api is api
        assert gemini.agent is agent

    def test_init_with_agent_and_session(self) -> None:
        agent = MagicMock(spec=ProviderAgent)
        session = MagicMock(spec=PromptSession)
        Gemini(agent=agent, session=session)
        agent.attach_session.assert_called_once_with(session)

    def test_init_defaults(self) -> None:
        gemini = Gemini()
        assert isinstance(gemini.api, GeminiAPI)
        assert isinstance(gemini.agent, GeminiClient)
        # Ensure the getter properties are actually called for coverage
        assert gemini.api is gemini._api
        assert gemini.agent is gemini._agent


class TestGeminiAPI:
    def test_limit_snapshot(self) -> None:
        assert GeminiAPI().get_limit_snapshot() == ProviderLimitSnapshot(
            provider=ProviderID.GEMINI,
            unavailable_reason="Gemini CLI does not yet expose usage stats.",
        )
