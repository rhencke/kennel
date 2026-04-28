import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fido.codex import (
    CodexCLIError,
    CodexProviderError,
    _codex,
    extract_result_text,
    extract_session_id,
    raise_for_provider_error_output,
    run_codex_exec,
)
from fido.provider import ProviderModel

FIXTURES = Path(__file__).parent / "fixtures" / "codex"


def _completed(
    stdout: str = "", returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


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
