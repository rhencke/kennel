from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from kennel.claude import (
    _claude,
    extract_result_text,
    extract_session_id,
    generate_branch_name,
    generate_reply,
    generate_status,
    generate_status_emoji,
    generate_status_with_session,
    print_prompt,
    print_prompt_from_file,
    print_prompt_json,
    resume_session,
    resume_status,
    session_was_active,
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


class TestPrintPromptJson:
    def test_extracts_key_from_clean_json(self) -> None:
        with patch(
            "subprocess.run", return_value=_completed('{"description": "Fixes bug."}')
        ):
            assert (
                print_prompt_json("q", "description", "claude-opus-4-6") == "Fixes bug."
            )

    def test_extracts_key_when_preamble_present(self) -> None:
        with patch(
            "subprocess.run",
            return_value=_completed(
                'Sure! Here you go: {"description": "Adds feature."} Done.'
            ),
        ):
            assert (
                print_prompt_json("q", "description", "claude-opus-4-6")
                == "Adds feature."
            )

    def test_returns_empty_when_key_missing(self) -> None:
        with patch("subprocess.run", return_value=_completed('{"other": "value"}')):
            assert print_prompt_json("q", "description", "claude-opus-4-6") == ""

    def test_returns_empty_on_no_json(self) -> None:
        with patch("subprocess.run", return_value=_completed("just plain text")):
            assert print_prompt_json("q", "description", "claude-opus-4-6") == ""

    def test_returns_empty_on_empty_output(self) -> None:
        with patch("subprocess.run", return_value=_completed("")):
            assert print_prompt_json("q", "description", "claude-opus-4-6") == ""

    def test_returns_empty_on_nonzero(self) -> None:
        with patch(
            "subprocess.run",
            return_value=_completed('{"description": "x"}', returncode=1),
        ):
            assert print_prompt_json("q", "description", "claude-opus-4-6") == ""

    def test_returns_empty_on_timeout(self) -> None:
        with patch(
            "subprocess.run", side_effect=subprocess.TimeoutExpired("claude", 30)
        ):
            assert print_prompt_json("q", "description", "claude-opus-4-6") == ""

    def test_returns_empty_on_file_not_found(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert print_prompt_json("q", "description", "claude-opus-4-6") == ""

    def test_appends_json_instruction_to_system_prompt(self) -> None:
        with patch(
            "subprocess.run", return_value=_completed('{"description": "x"}')
        ) as mock:
            print_prompt_json(
                "q", "description", "claude-opus-4-6", system_prompt="be helpful"
            )
        cmd = mock.call_args.args[0]
        assert "--system-prompt" in cmd
        idx = cmd.index("--system-prompt")
        combined = cmd[idx + 1]
        assert "be helpful" in combined
        assert "description" in combined

    def test_uses_json_instruction_as_only_system_prompt_when_none_given(self) -> None:
        with patch(
            "subprocess.run", return_value=_completed('{"description": "x"}')
        ) as mock:
            print_prompt_json("q", "description", "claude-opus-4-6", system_prompt=None)
        cmd = mock.call_args.args[0]
        assert "--system-prompt" in cmd
        idx = cmd.index("--system-prompt")
        assert "description" in cmd[idx + 1]

    def test_non_string_value_is_ignored(self) -> None:
        with patch("subprocess.run", return_value=_completed('{"description": 42}')):
            assert print_prompt_json("q", "description", "claude-opus-4-6") == ""

    def test_passes_model_and_timeout(self) -> None:
        with patch("subprocess.run", return_value=_completed('{"k": "v"}')) as mock:
            print_prompt_json("q", "k", "claude-haiku-4-5-20251001", timeout=7)
        cmd = mock.call_args.args[0]
        assert "claude-haiku-4-5-20251001" in cmd
        assert mock.call_args.kwargs["timeout"] == 7


class TestRunStreaming:
    def test_returns_output_and_returncode(self, tmp_path: Path) -> None:
        from kennel.claude import _run_streaming

        stdin_file = tmp_path / "input.txt"
        stdin_file.write_text("hello")
        output, rc = _run_streaming(["echo", "hi"], stdin_file)
        assert output == "hi"
        assert rc == 0

    def test_returns_empty_on_file_not_found(self, tmp_path: Path) -> None:
        from kennel.claude import _run_streaming

        stdin_file = tmp_path / "input.txt"
        stdin_file.write_text("")
        output, rc = _run_streaming(["/nonexistent/binary"], stdin_file)
        assert output == ""
        assert rc == -1

    def test_kills_on_idle_timeout(self, tmp_path: Path) -> None:
        from kennel.claude import _run_streaming

        stdin_file = tmp_path / "input.txt"
        stdin_file.write_text("")
        # sleep 60 will be killed after 0.1s idle timeout
        output, rc = _run_streaming(["sleep", "60"], stdin_file, idle_timeout=0.1)
        assert rc == -1

    def test_handles_process_exit_without_eof(self, tmp_path: Path) -> None:
        """Cover the proc.poll() is not None branch."""
        from kennel.claude import _run_streaming

        stdin_file = tmp_path / "input.txt"
        stdin_file.write_text("")

        mock_stdout = MagicMock()
        mock_stdout.fileno.return_value = 99
        mock_stdout.read.return_value = "leftover"

        mock_proc = MagicMock()
        mock_proc.stdout = mock_stdout
        mock_proc.stderr = MagicMock()
        # poll returns 0 (exited) — triggers the break at poll() branch
        mock_proc.poll.return_value = 0
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            # select returns NOT ready — forces poll() check
            patch("select.select", return_value=([], [], [])),
        ):
            output, rc = _run_streaming(["fake"], stdin_file)
        assert "leftover" in output
        assert rc == 0

    def test_drains_remaining_output(self, tmp_path: Path) -> None:
        """Cover the remaining = proc.stdout.read() branch."""
        from io import StringIO

        from kennel.claude import _run_streaming

        stdin_file = tmp_path / "input.txt"
        stdin_file.write_text("")

        mock_stdout = MagicMock(spec=StringIO)
        # readline returns data then EOF
        mock_stdout.readline.side_effect = ["line1\n", ""]
        mock_stdout.fileno.return_value = 99
        mock_stdout.read.return_value = "extra"

        mock_proc = MagicMock()
        mock_proc.stdout = mock_stdout
        mock_proc.stderr = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch("select.select", return_value=([mock_stdout], [], [])),
        ):
            output, rc = _run_streaming(["fake"], stdin_file)
        assert "line1" in output
        assert "extra" in output


class TestPrintPromptFromFile:
    def _files(self, tmp_path: Path) -> tuple[Path, Path]:
        sys = tmp_path / "system.md"
        sys.write_text("sys")
        prompt = tmp_path / "prompt.txt"
        prompt.write_text("p")
        return sys, prompt

    def test_returns_stdout_on_success(self, tmp_path: Path) -> None:
        sys, prompt = self._files(tmp_path)
        with patch("kennel.claude._run_streaming", return_value=("session output", 0)):
            result = print_prompt_from_file(sys, prompt, "claude-sonnet-4-6")
        assert result == "session output"

    def test_returns_empty_on_nonzero(self, tmp_path: Path) -> None:
        sys, prompt = self._files(tmp_path)
        with patch("kennel.claude._run_streaming", return_value=("err", 1)):
            result = print_prompt_from_file(sys, prompt, "claude-sonnet-4-6")
        assert result == ""

    def test_returns_empty_on_idle_timeout(self, tmp_path: Path) -> None:
        sys, prompt = self._files(tmp_path)
        with patch("kennel.claude._run_streaming", return_value=("partial", -1)):
            result = print_prompt_from_file(sys, prompt, "claude-sonnet-4-6")
        assert result == ""

    def test_passes_correct_cmd(self, tmp_path: Path) -> None:
        sys, prompt = self._files(tmp_path)
        with patch("kennel.claude._run_streaming", return_value=("out", 0)) as mock:
            print_prompt_from_file(sys, prompt, "claude-sonnet-4-6")
        cmd = mock.call_args[0][0]
        assert "--model" in cmd
        assert "claude-sonnet-4-6" in cmd
        assert "--system-prompt-file" in cmd
        assert str(sys) in cmd
        assert "--print" in cmd

    def test_passes_idle_timeout(self, tmp_path: Path) -> None:
        sys, prompt = self._files(tmp_path)
        with patch("kennel.claude._run_streaming", return_value=("out", 0)) as mock:
            print_prompt_from_file(sys, prompt, "claude-sonnet-4-6", idle_timeout=600.0)
        assert mock.call_args[0][2] == 600.0


class TestResumeSession:
    def test_returns_stdout_on_success(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("continue")
        with patch("kennel.claude._run_streaming", return_value=("continued", 0)):
            result = resume_session("sess-123", prompt_file, "claude-sonnet-4-6")
        assert result == "continued"

    def test_returns_empty_on_nonzero(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("p")
        with patch("kennel.claude._run_streaming", return_value=("", 1)):
            result = resume_session("sess-123", prompt_file, "claude-sonnet-4-6")
        assert result == ""

    def test_returns_empty_on_idle_timeout(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("p")
        with patch("kennel.claude._run_streaming", return_value=("partial", -1)):
            result = resume_session("sess-123", prompt_file, "claude-sonnet-4-6")
        assert result == ""

    def test_passes_correct_cmd(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("p")
        with patch("kennel.claude._run_streaming", return_value=("out", 0)) as mock:
            resume_session("my-session-id", prompt_file, "claude-sonnet-4-6")
        cmd = mock.call_args[0][0]
        assert "--resume" in cmd
        assert "my-session-id" in cmd
        assert "--print" in cmd

    def test_passes_idle_timeout(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("p")
        with patch("kennel.claude._run_streaming", return_value=("out", 0)) as mock:
            resume_session(
                "sess-1", prompt_file, "claude-sonnet-4-6", idle_timeout=900.0
            )
        assert mock.call_args[0][2] == 900.0


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


class TestExtractResultText:
    def test_returns_text_from_result_line(self) -> None:
        output = '{"type":"result","result":"🐶\\ncoding","session_id":"abc"}'
        assert extract_result_text(output) == "🐶\ncoding"

    def test_returns_empty_when_no_result_line(self) -> None:
        output = '{"type":"assistant","message":{"content":[]}}'
        assert extract_result_text(output) == ""

    def test_returns_empty_on_empty_input(self) -> None:
        assert extract_result_text("") == ""

    def test_returns_empty_on_blank_lines_only(self) -> None:
        assert extract_result_text("\n\n  \n") == ""

    def test_returns_empty_on_malformed_json(self) -> None:
        assert extract_result_text("not json") == ""

    def test_skips_malformed_lines_and_finds_valid(self) -> None:
        output = "bad json\n" + '{"type":"result","result":"woof"}'
        assert extract_result_text(output) == "woof"

    def test_returns_last_text_when_multiple_result_lines(self) -> None:
        output = '{"type":"result","result":"first"}\n{"type":"result","result":"last"}'
        assert extract_result_text(output) == "last"

    def test_returns_empty_when_result_field_missing(self) -> None:
        output = '{"type":"result","session_id":"abc"}'
        assert extract_result_text(output) == ""

    def test_returns_empty_when_result_is_empty_string(self) -> None:
        output = '{"type":"result","result":""}'
        assert extract_result_text(output) == ""

    def test_returns_empty_when_result_is_not_string(self) -> None:
        output = '{"type":"result","result":42}'
        assert extract_result_text(output) == ""

    def test_ignores_non_result_types(self) -> None:
        output = '{"type":"system","result":"should-ignore"}'
        assert extract_result_text(output) == ""

    def test_handles_mixed_output(self) -> None:
        output = (
            '{"type":"system","result":"ignore"}\n'
            '{"type":"assistant","message":{}}\n'
            '{"type":"result","result":"🐕\\nworking","session_id":"sid"}\n'
        )
        assert extract_result_text(output) == "🐕\nworking"


class TestSessionWasActive:
    def test_returns_true_on_any_output(self) -> None:
        assert session_was_active("some output\n") is True

    def test_returns_false_on_empty_output(self) -> None:
        assert session_was_active("") is False

    def test_returns_false_on_whitespace_only(self) -> None:
        assert session_was_active("   \n  \n") is False

    def test_returns_true_on_json_output(self) -> None:
        assert session_was_active('{"type":"result"}\n') is True


class TestGenerateStatusWithSession:
    _RESULT_LINE = '{"type":"result","result":"🐶\\ncoding","session_id":"sess-42"}'

    def test_returns_text_and_session_id_on_success(self) -> None:
        with patch("subprocess.run", return_value=_completed(self._RESULT_LINE)):
            text, sid = generate_status_with_session("doing stuff", "be fido")
        assert text == "🐶\ncoding"
        assert sid == "sess-42"

    def test_returns_empty_pair_on_nonzero(self) -> None:
        with patch(
            "subprocess.run", return_value=_completed(self._RESULT_LINE, returncode=1)
        ):
            assert generate_status_with_session("doing stuff", "sys") == ("", "")

    def test_returns_empty_pair_on_timeout(self) -> None:
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired("claude", 15),
        ):
            assert generate_status_with_session("doing stuff", "sys") == ("", "")

    def test_returns_empty_pair_on_file_not_found(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert generate_status_with_session("doing stuff", "sys") == ("", "")

    def test_passes_correct_flags(self) -> None:
        with patch(
            "subprocess.run", return_value=_completed(self._RESULT_LINE)
        ) as mock:
            generate_status_with_session(
                "working on #42",
                "be a dog",
                model="claude-sonnet-4-6",
                timeout=10,
            )
        cmd = mock.call_args.args[0]
        assert "--model" in cmd
        assert "claude-sonnet-4-6" in cmd
        assert "--output-format" in cmd
        assert "stream-json" in cmd
        assert "--verbose" in cmd
        assert "--dangerously-skip-permissions" in cmd
        assert "--system-prompt" in cmd
        assert "be a dog" in cmd
        assert "--print" in cmd
        assert "-p" in cmd
        assert "working on #42" in cmd
        assert mock.call_args.kwargs["timeout"] == 10

    def test_default_model_and_timeout(self) -> None:
        with patch(
            "subprocess.run", return_value=_completed(self._RESULT_LINE)
        ) as mock:
            generate_status_with_session("working", "sys")
        cmd = mock.call_args.args[0]
        assert "claude-opus-4-6" in cmd
        assert mock.call_args.kwargs["timeout"] == 15

    def test_returns_empty_text_when_no_result_field(self) -> None:
        no_result = '{"type":"result","session_id":"sid"}'
        with patch("subprocess.run", return_value=_completed(no_result)):
            text, sid = generate_status_with_session("working", "sys")
        assert text == ""
        assert sid == "sid"

    def test_returns_empty_session_when_no_session_id(self) -> None:
        no_sid = '{"type":"result","result":"🐶\\nwoof"}'
        with patch("subprocess.run", return_value=_completed(no_sid)):
            text, sid = generate_status_with_session("working", "sys")
        assert text == "🐶\nwoof"
        assert sid == ""


class TestGenerateStatusEmoji:
    def test_returns_emoji(self) -> None:
        with patch("subprocess.run", return_value=_completed("🐕")):
            result = generate_status_emoji("pick emoji", "be fido")
        assert result == "🐕"

    def test_returns_empty_on_failure(self) -> None:
        with patch("subprocess.run", return_value=_completed("", returncode=1)):
            result = generate_status_emoji("pick emoji", "sys")
        assert result == ""

    def test_passes_correct_flags(self) -> None:
        with patch("subprocess.run", return_value=_completed("🐕")) as mock:
            generate_status_emoji(
                "pick emoji", "be fido", model="claude-sonnet-4-6", timeout=10
            )
        cmd = mock.call_args.args[0]
        assert "claude-sonnet-4-6" in cmd
        assert "be fido" in cmd
        assert mock.call_args.kwargs["timeout"] == 10


class TestResumeStatus:
    _RESULT_LINE = '{"type":"result","result":"🐕\\nfetching","session_id":"s-1"}'

    def test_returns_text_on_success(self) -> None:
        with patch("subprocess.run", return_value=_completed(self._RESULT_LINE)):
            result = resume_status("s-1", "please shorten")
        assert result == "🐕\nfetching"

    def test_returns_empty_on_nonzero(self) -> None:
        with patch(
            "subprocess.run", return_value=_completed(self._RESULT_LINE, returncode=1)
        ):
            assert resume_status("s-1", "shorten") == ""

    def test_returns_empty_on_timeout(self) -> None:
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired("claude", 15),
        ):
            assert resume_status("s-1", "shorten") == ""

    def test_returns_empty_on_file_not_found(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert resume_status("s-1", "shorten") == ""

    def test_passes_correct_flags(self) -> None:
        with patch(
            "subprocess.run", return_value=_completed(self._RESULT_LINE)
        ) as mock:
            resume_status(
                "my-session",
                "shorten to 80 chars",
                model="claude-sonnet-4-6",
                timeout=20,
            )
        cmd = mock.call_args.args[0]
        assert "--model" in cmd
        assert "claude-sonnet-4-6" in cmd
        assert "--output-format" in cmd
        assert "stream-json" in cmd
        assert "--verbose" in cmd
        assert "--dangerously-skip-permissions" in cmd
        assert "--resume" in cmd
        assert "my-session" in cmd
        assert "--print" in cmd
        assert "-p" in cmd
        assert "shorten to 80 chars" in cmd
        assert mock.call_args.kwargs["timeout"] == 20

    def test_default_model_and_timeout(self) -> None:
        with patch(
            "subprocess.run", return_value=_completed(self._RESULT_LINE)
        ) as mock:
            resume_status("s-1", "shorten")
        cmd = mock.call_args.args[0]
        assert "claude-opus-4-6" in cmd
        assert mock.call_args.kwargs["timeout"] == 15

    def test_returns_empty_when_no_result_field(self) -> None:
        no_result = '{"type":"result","session_id":"s-1"}'
        with patch("subprocess.run", return_value=_completed(no_result)):
            assert resume_status("s-1", "shorten") == ""
