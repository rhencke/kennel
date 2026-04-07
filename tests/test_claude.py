from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from kennel.claude import (
    _claude,
    extract_session_id,
    generate_branch_name,
    generate_reply,
    generate_status,
    print_prompt,
    print_prompt_from_file,
    resume_session,
    triage_comment,
)


def _completed(
    stdout: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=""
    )


class TestClaudeHelper:
    def test_calls_subprocess_run_with_prompt(self) -> None:
        with patch("subprocess.run", return_value=_completed("out")) as mock:
            result = _claude("--print", "-p", "hello", prompt="input", timeout=5)
        mock.assert_called_once_with(
            ["claude", "--print", "-p", "hello"],
            input="input",
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.stdout == "out"

    def test_defaults(self) -> None:
        with patch("subprocess.run", return_value=_completed()) as mock:
            _claude("--version")
        _, kwargs = mock.call_args
        assert kwargs["timeout"] == 30
        assert kwargs["input"] is None

    def test_no_prompt(self) -> None:
        with patch("subprocess.run", return_value=_completed("x")) as mock:
            _claude("--help")
        assert mock.call_args.kwargs["input"] is None


class TestPrintPrompt:
    def test_returns_stripped_output(self) -> None:
        with patch("subprocess.run", return_value=_completed("  hello world  \n")):
            assert print_prompt("hi", "claude-opus-4-6") == "hello world"

    def test_returns_empty_on_nonzero(self) -> None:
        with patch("subprocess.run", return_value=_completed("err", returncode=1)):
            assert print_prompt("hi", "claude-opus-4-6") == ""

    def test_returns_empty_on_timeout(self) -> None:
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired("claude", 30),
        ):
            assert print_prompt("hi", "claude-opus-4-6") == ""

    def test_returns_empty_on_file_not_found(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert print_prompt("hi", "claude-opus-4-6") == ""

    def test_includes_system_prompt(self) -> None:
        with patch("subprocess.run", return_value=_completed("ok")) as mock:
            print_prompt("question", "claude-opus-4-6", system_prompt="be helpful")
        cmd = mock.call_args.args[0]
        assert "--system-prompt" in cmd
        assert "be helpful" in cmd

    def test_no_system_prompt_arg_when_none(self) -> None:
        with patch("subprocess.run", return_value=_completed("ok")) as mock:
            print_prompt("q", "claude-opus-4-6", system_prompt=None)
        cmd = mock.call_args.args[0]
        assert "--system-prompt" not in cmd

    def test_passes_model(self) -> None:
        with patch("subprocess.run", return_value=_completed("ok")) as mock:
            print_prompt("q", "claude-haiku-4-5-20251001")
        cmd = mock.call_args.args[0]
        assert "--model" in cmd
        assert "claude-haiku-4-5-20251001" in cmd

    def test_passes_timeout(self) -> None:
        with patch("subprocess.run", return_value=_completed("ok")) as mock:
            print_prompt("q", "claude-opus-4-6", timeout=7)
        assert mock.call_args.kwargs["timeout"] == 7


class TestPrintPromptFromFile:
    def test_returns_stdout_on_success(self, tmp_path: Path) -> None:
        system_file = tmp_path / "system.md"
        system_file.write_text("be a good dog")
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("hello")
        with patch("subprocess.run", return_value=_completed("session output\n")):
            result = print_prompt_from_file(
                system_file, prompt_file, "claude-sonnet-4-6"
            )
        assert result == "session output"

    def test_returns_empty_on_nonzero(self, tmp_path: Path) -> None:
        system_file = tmp_path / "system.md"
        system_file.write_text("sys")
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("p")
        with patch("subprocess.run", return_value=_completed("", returncode=1)):
            result = print_prompt_from_file(
                system_file, prompt_file, "claude-sonnet-4-6"
            )
        assert result == ""

    def test_returns_empty_on_timeout(self, tmp_path: Path) -> None:
        system_file = tmp_path / "system.md"
        system_file.write_text("sys")
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("p")
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired("claude", 30),
        ):
            result = print_prompt_from_file(
                system_file, prompt_file, "claude-sonnet-4-6"
            )
        assert result == ""

    def test_returns_empty_on_file_not_found(self, tmp_path: Path) -> None:
        system_file = tmp_path / "system.md"
        system_file.write_text("sys")
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("p")
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = print_prompt_from_file(
                system_file, prompt_file, "claude-sonnet-4-6"
            )
        assert result == ""

    def test_passes_correct_flags(self, tmp_path: Path) -> None:
        system_file = tmp_path / "system.md"
        system_file.write_text("sys")
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("p")
        with patch("subprocess.run", return_value=_completed("out")) as mock:
            print_prompt_from_file(
                system_file, prompt_file, "claude-sonnet-4-6", timeout=60
            )
        cmd = mock.call_args.args[0]
        assert "--model" in cmd
        assert "claude-sonnet-4-6" in cmd
        assert "--output-format" in cmd
        assert "stream-json" in cmd
        assert "--verbose" in cmd
        assert "--dangerously-skip-permissions" in cmd
        assert "--system-prompt-file" in cmd
        assert str(system_file) in cmd
        assert "--print" in cmd
        assert mock.call_args.kwargs["timeout"] == 60


class TestResumeSession:
    def test_returns_stdout_on_success(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("continue")
        with patch("subprocess.run", return_value=_completed("continued\n")):
            result = resume_session("sess-123", prompt_file, "claude-sonnet-4-6")
        assert result == "continued"

    def test_returns_empty_on_nonzero(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("p")
        with patch("subprocess.run", return_value=_completed("", returncode=1)):
            result = resume_session("sess-123", prompt_file, "claude-sonnet-4-6")
        assert result == ""

    def test_returns_empty_on_timeout(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("p")
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired("claude", 300),
        ):
            result = resume_session("sess-123", prompt_file, "claude-sonnet-4-6")
        assert result == ""

    def test_returns_empty_on_file_not_found(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("p")
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = resume_session("sess-123", prompt_file, "claude-sonnet-4-6")
        assert result == ""

    def test_passes_correct_flags(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("p")
        with patch("subprocess.run", return_value=_completed("out")) as mock:
            resume_session(
                "my-session-id", prompt_file, "claude-sonnet-4-6", timeout=120
            )
        cmd = mock.call_args.args[0]
        assert "--model" in cmd
        assert "claude-sonnet-4-6" in cmd
        assert "--output-format" in cmd
        assert "stream-json" in cmd
        assert "--verbose" in cmd
        assert "--dangerously-skip-permissions" in cmd
        assert "--resume" in cmd
        assert "my-session-id" in cmd
        assert "--print" in cmd
        assert mock.call_args.kwargs["timeout"] == 120


class TestTriageComment:
    def test_returns_first_line(self) -> None:
        with patch(
            "subprocess.run", return_value=_completed("ACT: fix the thing\nextra")
        ):
            assert triage_comment("triage this") == "ACT: fix the thing"

    def test_returns_empty_on_nonzero(self) -> None:
        with patch("subprocess.run", return_value=_completed("ACT", returncode=1)):
            assert triage_comment("triage this") == ""

    def test_returns_empty_on_empty_output(self) -> None:
        with patch("subprocess.run", return_value=_completed("")):
            assert triage_comment("triage this") == ""

    def test_returns_empty_on_timeout(self) -> None:
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired("claude", 15),
        ):
            assert triage_comment("triage") == ""

    def test_returns_empty_on_file_not_found(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert triage_comment("triage") == ""

    def test_default_model_and_timeout(self) -> None:
        with patch("subprocess.run", return_value=_completed("ACT: thing")) as mock:
            triage_comment("triage this")
        cmd = mock.call_args.args[0]
        assert "claude-opus-4-6" in cmd
        assert mock.call_args.kwargs["timeout"] == 15

    def test_custom_model_and_timeout(self) -> None:
        with patch("subprocess.run", return_value=_completed("DO: fix")) as mock:
            triage_comment("triage", model="claude-haiku-4-5-20251001", timeout=5)
        cmd = mock.call_args.args[0]
        assert "claude-haiku-4-5-20251001" in cmd
        assert mock.call_args.kwargs["timeout"] == 5


class TestGenerateReply:
    def test_returns_stripped_output(self) -> None:
        with patch("subprocess.run", return_value=_completed("  woof!  \n")):
            assert generate_reply("write a reply") == "woof!"

    def test_returns_empty_on_nonzero(self) -> None:
        with patch("subprocess.run", return_value=_completed("err", returncode=1)):
            assert generate_reply("write") == ""

    def test_returns_empty_on_timeout(self) -> None:
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired("claude", 30),
        ):
            assert generate_reply("write") == ""

    def test_returns_empty_on_file_not_found(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert generate_reply("write") == ""

    def test_default_model_and_timeout(self) -> None:
        with patch("subprocess.run", return_value=_completed("ok")) as mock:
            generate_reply("write a reply")
        cmd = mock.call_args.args[0]
        assert "claude-opus-4-6" in cmd
        assert mock.call_args.kwargs["timeout"] == 30

    def test_custom_model_and_timeout(self) -> None:
        with patch("subprocess.run", return_value=_completed("ok")) as mock:
            generate_reply("write", model="claude-sonnet-4-6", timeout=10)
        cmd = mock.call_args.args[0]
        assert "claude-sonnet-4-6" in cmd
        assert mock.call_args.kwargs["timeout"] == 10


class TestGenerateBranchName:
    def test_returns_first_line(self) -> None:
        with patch("subprocess.run", return_value=_completed("add-tests\nextra line")):
            assert generate_branch_name("make branch for: add tests") == "add-tests"

    def test_returns_empty_on_nonzero(self) -> None:
        with patch("subprocess.run", return_value=_completed("slug", returncode=1)):
            assert generate_branch_name("make branch") == ""

    def test_returns_empty_on_empty_output(self) -> None:
        with patch("subprocess.run", return_value=_completed("")):
            assert generate_branch_name("make branch") == ""

    def test_returns_empty_on_timeout(self) -> None:
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired("claude", 15),
        ):
            assert generate_branch_name("make branch") == ""

    def test_returns_empty_on_file_not_found(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert generate_branch_name("make branch") == ""

    def test_default_model_and_timeout(self) -> None:
        with patch("subprocess.run", return_value=_completed("fix-bug")) as mock:
            generate_branch_name("fix bug in parser")
        cmd = mock.call_args.args[0]
        assert "claude-haiku-4-5-20251001" in cmd
        assert mock.call_args.kwargs["timeout"] == 15

    def test_custom_model_and_timeout(self) -> None:
        with patch("subprocess.run", return_value=_completed("fix-bug")) as mock:
            generate_branch_name("fix bug", model="claude-opus-4-6", timeout=20)
        cmd = mock.call_args.args[0]
        assert "claude-opus-4-6" in cmd
        assert mock.call_args.kwargs["timeout"] == 20


class TestGenerateStatus:
    def test_returns_two_lines(self) -> None:
        with patch("subprocess.run", return_value=_completed("🐶\ncoding up a storm")):
            result = generate_status("working on #42", "be fido")
        assert result == "🐶\ncoding up a storm"

    def test_returns_empty_on_failure(self) -> None:
        with patch("subprocess.run", return_value=_completed("", returncode=1)):
            result = generate_status("working", "sys")
        assert result == ""

    def test_returns_empty_on_timeout(self) -> None:
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired("claude", 15),
        ):
            result = generate_status("working", "sys")
        assert result == ""

    def test_passes_system_prompt(self) -> None:
        with patch("subprocess.run", return_value=_completed("🚀\nworking")) as mock:
            generate_status("doing stuff", system_prompt="be a dog")
        cmd = mock.call_args.args[0]
        assert "--system-prompt" in cmd
        assert "be a dog" in cmd

    def test_default_model_and_timeout(self) -> None:
        with patch("subprocess.run", return_value=_completed("🐶\nwoof")) as mock:
            generate_status("working", "sys")
        cmd = mock.call_args.args[0]
        assert "claude-opus-4-6" in cmd
        assert mock.call_args.kwargs["timeout"] == 15

    def test_custom_model_and_timeout(self) -> None:
        with patch("subprocess.run", return_value=_completed("🐶\nwoof")) as mock:
            generate_status("working", "sys", model="claude-sonnet-4-6", timeout=5)
        cmd = mock.call_args.args[0]
        assert "claude-sonnet-4-6" in cmd
        assert mock.call_args.kwargs["timeout"] == 5


class TestExtractSessionId:
    def test_returns_session_id_from_result_line(self) -> None:
        output = '{"type":"result","session_id":"abc-123"}'
        assert extract_session_id(output) == "abc-123"

    def test_returns_empty_when_no_result_line(self) -> None:
        output = '{"type":"assistant","message":{"content":[]}}'
        assert extract_session_id(output) == ""

    def test_returns_empty_on_empty_input(self) -> None:
        assert extract_session_id("") == ""

    def test_returns_empty_on_blank_lines_only(self) -> None:
        assert extract_session_id("\n\n  \n") == ""

    def test_returns_empty_on_malformed_json(self) -> None:
        assert extract_session_id("not json at all") == ""

    def test_skips_malformed_lines_and_finds_valid(self) -> None:
        output = "bad json\n" + '{"type":"result","session_id":"xyz"}'
        assert extract_session_id(output) == "xyz"

    def test_returns_last_session_id_when_multiple_result_lines(self) -> None:
        output = (
            '{"type":"result","session_id":"first"}\n'
            '{"type":"result","session_id":"last"}'
        )
        assert extract_session_id(output) == "last"

    def test_returns_empty_when_session_id_missing(self) -> None:
        output = '{"type":"result"}'
        assert extract_session_id(output) == ""

    def test_returns_empty_when_session_id_null(self) -> None:
        output = '{"type":"result","session_id":null}'
        assert extract_session_id(output) == ""

    def test_returns_empty_when_session_id_empty_string(self) -> None:
        output = '{"type":"result","session_id":""}'
        assert extract_session_id(output) == ""

    def test_ignores_non_result_types(self) -> None:
        output = '{"type":"system","session_id":"should-ignore"}'
        assert extract_session_id(output) == ""

    def test_handles_mixed_output(self) -> None:
        output = (
            '{"type":"system","session_id":"ignore"}\n'
            '{"type":"assistant","message":{}}\n'
            '{"type":"result","session_id":"real-id"}\n'
        )
        assert extract_session_id(output) == "real-id"
