from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kennel.claude import (
    ClaudeSession,
    ClaudeStreamError,
    _active_children,
    _claude,
    _register_child,
    _unregister_child,
    extract_result_text,
    extract_session_id,
    generate_branch_name,
    generate_reply,
    generate_status,
    generate_status_emoji,
    generate_status_with_session,
    kill_active_children,
    print_prompt,
    print_prompt_from_file,
    print_prompt_json,
    resume_session,
    resume_status,
    triage_comment,
)


def _completed(
    stdout: str = "", returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


class TestClaudeHelper:
    def test_calls_subprocess_run_with_prompt(self) -> None:
        mock_run = MagicMock(return_value=_completed("out"))
        result = _claude(
            "--print", "-p", "hello", prompt="input", timeout=5, runner=mock_run
        )
        mock_run.assert_called_once_with(
            ["claude", "--print", "-p", "hello"],
            input="input",
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.stdout == "out"

    def test_defaults(self) -> None:
        mock_run = MagicMock(return_value=_completed())
        _claude("--version", runner=mock_run)
        _, kwargs = mock_run.call_args
        assert kwargs["timeout"] == 30
        assert kwargs["input"] is None

    def test_no_prompt(self) -> None:
        mock_run = MagicMock(return_value=_completed("x"))
        _claude("--help", runner=mock_run)
        assert mock_run.call_args.kwargs["input"] is None


class TestPrintPrompt:
    def test_returns_stripped_output(self) -> None:
        mock_run = MagicMock(return_value=_completed("  hello world  \n"))
        assert print_prompt("hi", "claude-opus-4-6", runner=mock_run) == "hello world"

    def test_returns_empty_on_nonzero(self) -> None:
        mock_run = MagicMock(return_value=_completed("err", returncode=1))
        mock_sleep = MagicMock()
        assert (
            print_prompt("hi", "claude-opus-4-6", runner=mock_run, _sleep=mock_sleep)
            == ""
        )
        mock_run.assert_called_once()
        mock_sleep.assert_not_called()

    def test_returns_empty_on_timeout(self) -> None:
        mock_run = MagicMock(side_effect=subprocess.TimeoutExpired("claude", 30))
        mock_sleep = MagicMock()
        assert (
            print_prompt("hi", "claude-opus-4-6", runner=mock_run, _sleep=mock_sleep)
            == ""
        )
        mock_run.assert_called_once()
        mock_sleep.assert_not_called()

    def test_returns_empty_on_file_not_found(self) -> None:
        mock_run = MagicMock(side_effect=FileNotFoundError)
        mock_sleep = MagicMock()
        assert (
            print_prompt("hi", "claude-opus-4-6", runner=mock_run, _sleep=mock_sleep)
            == ""
        )
        mock_run.assert_called_once()
        mock_sleep.assert_not_called()

    def test_retries_on_empty_output_then_succeeds(self) -> None:
        mock_run = MagicMock(
            side_effect=[_completed(""), _completed(""), _completed("hello")]
        )
        mock_sleep = MagicMock()
        assert (
            print_prompt("hi", "claude-opus-4-6", runner=mock_run, _sleep=mock_sleep)
            == "hello"
        )
        assert mock_run.call_count == 3
        assert mock_sleep.call_count == 2

    def test_returns_empty_after_all_retries_exhausted(self) -> None:
        mock_run = MagicMock(return_value=_completed(""))
        mock_sleep = MagicMock()
        assert (
            print_prompt("hi", "claude-opus-4-6", runner=mock_run, _sleep=mock_sleep)
            == ""
        )
        assert mock_run.call_count == 3
        assert mock_sleep.call_count == 2

    def test_includes_system_prompt(self) -> None:
        mock_run = MagicMock(return_value=_completed("ok"))
        print_prompt(
            "question", "claude-opus-4-6", system_prompt="be helpful", runner=mock_run
        )
        cmd = mock_run.call_args.args[0]
        assert "--system-prompt" in cmd
        assert "be helpful" in cmd

    def test_no_system_prompt_arg_when_none(self) -> None:
        mock_run = MagicMock(return_value=_completed("ok"))
        print_prompt("q", "claude-opus-4-6", system_prompt=None, runner=mock_run)
        cmd = mock_run.call_args.args[0]
        assert "--system-prompt" not in cmd

    def test_passes_model(self) -> None:
        mock_run = MagicMock(return_value=_completed("ok"))
        print_prompt("q", "claude-haiku-4-5-20251001", runner=mock_run)
        cmd = mock_run.call_args.args[0]
        assert "--model" in cmd
        assert "claude-haiku-4-5-20251001" in cmd

    def test_passes_timeout(self) -> None:
        mock_run = MagicMock(return_value=_completed("ok"))
        print_prompt("q", "claude-opus-4-6", timeout=7, runner=mock_run)
        assert mock_run.call_args.kwargs["timeout"] == 7

    def test_logs_stderr_at_warning_on_empty_output(self, caplog) -> None:
        mock_run = MagicMock(return_value=_completed("", stderr="Rate limit exceeded"))
        with caplog.at_level(logging.WARNING, logger="kennel.claude"):
            print_prompt("q", "claude-opus-4-6", runner=mock_run, _sleep=MagicMock())
        assert "Rate limit exceeded" in caplog.text

    def test_logs_raw_stdout_at_debug_on_empty_output(self, caplog) -> None:
        mock_run = MagicMock(return_value=_completed("   \n  "))
        with caplog.at_level(logging.DEBUG, logger="kennel.claude"):
            print_prompt("q", "claude-opus-4-6", runner=mock_run, _sleep=MagicMock())
        assert "stdout=" in caplog.text

    def test_no_stderr_log_when_stderr_empty(self, caplog) -> None:
        mock_run = MagicMock(return_value=_completed(""))
        with caplog.at_level(logging.WARNING, logger="kennel.claude"):
            print_prompt("q", "claude-opus-4-6", runner=mock_run, _sleep=MagicMock())
        assert "stderr=" not in caplog.text


class TestPrintPromptJson:
    def test_extracts_key_from_clean_json(self) -> None:
        mock_run = MagicMock(return_value=_completed('{"description": "Fixes bug."}'))
        assert (
            print_prompt_json("q", "description", "claude-opus-4-6", runner=mock_run)
            == "Fixes bug."
        )

    def test_extracts_key_when_preamble_present(self) -> None:
        mock_run = MagicMock(
            return_value=_completed(
                'Sure! Here you go: {"description": "Adds feature."} Done.'
            )
        )
        assert (
            print_prompt_json("q", "description", "claude-opus-4-6", runner=mock_run)
            == "Adds feature."
        )

    def test_returns_empty_when_key_missing(self) -> None:
        mock_run = MagicMock(return_value=_completed('{"other": "value"}'))
        assert (
            print_prompt_json("q", "description", "claude-opus-4-6", runner=mock_run)
            == ""
        )

    def test_returns_empty_on_no_json(self) -> None:
        mock_run = MagicMock(return_value=_completed("just plain text"))
        assert (
            print_prompt_json("q", "description", "claude-opus-4-6", runner=mock_run)
            == ""
        )

    def test_returns_empty_on_empty_output(self) -> None:
        mock_run = MagicMock(return_value=_completed(""))
        assert (
            print_prompt_json("q", "description", "claude-opus-4-6", runner=mock_run)
            == ""
        )

    def test_returns_empty_on_nonzero(self) -> None:
        mock_run = MagicMock(
            return_value=_completed('{"description": "x"}', returncode=1)
        )
        assert (
            print_prompt_json("q", "description", "claude-opus-4-6", runner=mock_run)
            == ""
        )

    def test_returns_empty_on_timeout(self) -> None:
        mock_run = MagicMock(side_effect=subprocess.TimeoutExpired("claude", 30))
        assert (
            print_prompt_json("q", "description", "claude-opus-4-6", runner=mock_run)
            == ""
        )

    def test_returns_empty_on_file_not_found(self) -> None:
        mock_run = MagicMock(side_effect=FileNotFoundError)
        assert (
            print_prompt_json("q", "description", "claude-opus-4-6", runner=mock_run)
            == ""
        )

    def test_appends_json_instruction_to_system_prompt(self) -> None:
        mock_run = MagicMock(return_value=_completed('{"description": "x"}'))
        print_prompt_json(
            "q",
            "description",
            "claude-opus-4-6",
            system_prompt="be helpful",
            runner=mock_run,
        )
        cmd = mock_run.call_args.args[0]
        assert "--system-prompt" in cmd
        idx = cmd.index("--system-prompt")
        combined = cmd[idx + 1]
        assert "be helpful" in combined
        assert "description" in combined

    def test_uses_json_instruction_as_only_system_prompt_when_none_given(self) -> None:
        mock_run = MagicMock(return_value=_completed('{"description": "x"}'))
        print_prompt_json(
            "q", "description", "claude-opus-4-6", system_prompt=None, runner=mock_run
        )
        cmd = mock_run.call_args.args[0]
        assert "--system-prompt" in cmd
        idx = cmd.index("--system-prompt")
        assert "description" in cmd[idx + 1]

    def test_non_string_value_is_ignored(self) -> None:
        mock_run = MagicMock(return_value=_completed('{"description": 42}'))
        assert (
            print_prompt_json("q", "description", "claude-opus-4-6", runner=mock_run)
            == ""
        )

    def test_passes_model_and_timeout(self) -> None:
        mock_run = MagicMock(return_value=_completed('{"k": "v"}'))
        print_prompt_json(
            "q", "k", "claude-haiku-4-5-20251001", timeout=7, runner=mock_run
        )
        cmd = mock_run.call_args.args[0]
        assert "claude-haiku-4-5-20251001" in cmd
        assert mock_run.call_args.kwargs["timeout"] == 7


class TestRunStreaming:
    def test_yields_output_lines(self, tmp_path: Path) -> None:
        from kennel.claude import _run_streaming

        stdin_file = tmp_path / "input.txt"
        stdin_file.write_text("hello")
        lines = list(_run_streaming(["echo", "hi"], stdin_file))
        assert "".join(lines).strip() == "hi"

    def test_raises_on_file_not_found(self, tmp_path: Path) -> None:
        from kennel.claude import _run_streaming

        stdin_file = tmp_path / "input.txt"
        stdin_file.write_text("")
        import pytest

        with pytest.raises(FileNotFoundError):
            list(_run_streaming(["/nonexistent/binary"], stdin_file))

    def test_raises_on_idle_timeout(self, tmp_path: Path) -> None:
        from kennel.claude import _run_streaming

        stdin_file = tmp_path / "input.txt"
        stdin_file.write_text("")
        import pytest

        with pytest.raises(ClaudeStreamError) as exc_info:
            list(_run_streaming(["sleep", "60"], stdin_file, idle_timeout=0.1))
        assert exc_info.value.returncode == -1

    def test_raises_on_nonzero_exit(self, tmp_path: Path) -> None:
        from kennel.claude import _run_streaming

        stdin_file = tmp_path / "input.txt"
        stdin_file.write_text("")
        import pytest

        with pytest.raises(ClaudeStreamError) as exc_info:
            list(_run_streaming(["false"], stdin_file))
        assert exc_info.value.returncode != 0

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

        mock_popen = MagicMock(return_value=mock_proc)
        mock_selector = MagicMock(return_value=([], [], []))

        result = "".join(
            _run_streaming(
                ["fake"], stdin_file, popen=mock_popen, selector=mock_selector
            )
        )
        assert "leftover" in result

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

        mock_popen = MagicMock(return_value=mock_proc)
        mock_selector = MagicMock(return_value=([mock_stdout], [], []))

        result = "".join(
            _run_streaming(
                ["fake"], stdin_file, popen=mock_popen, selector=mock_selector
            )
        )
        assert "line1" in result
        assert "extra" in result


class TestPrintPromptFromFile:
    def _files(self, tmp_path: Path) -> tuple[Path, Path]:
        sys = tmp_path / "system.md"
        sys.write_text("sys")
        prompt = tmp_path / "prompt.txt"
        prompt.write_text("p")
        return sys, prompt

    def test_returns_stdout_on_success(self, tmp_path: Path) -> None:
        sys, prompt = self._files(tmp_path)
        mock_stream = MagicMock(return_value=iter(["session output"]))
        result = print_prompt_from_file(
            sys, prompt, "claude-sonnet-4-6", streaming_runner=mock_stream
        )
        assert result == "session output"

    def test_raises_on_nonzero(self, tmp_path: Path) -> None:
        sys, prompt = self._files(tmp_path)
        mock_stream = MagicMock(side_effect=ClaudeStreamError(1))
        with pytest.raises(ClaudeStreamError):
            print_prompt_from_file(
                sys, prompt, "claude-sonnet-4-6", streaming_runner=mock_stream
            )

    def test_raises_on_idle_timeout(self, tmp_path: Path) -> None:
        sys, prompt = self._files(tmp_path)
        mock_stream = MagicMock(side_effect=ClaudeStreamError(-1))
        with pytest.raises(ClaudeStreamError):
            print_prompt_from_file(
                sys, prompt, "claude-sonnet-4-6", streaming_runner=mock_stream
            )

    def test_raises_on_file_not_found(self, tmp_path: Path) -> None:
        sys, prompt = self._files(tmp_path)
        mock_stream = MagicMock(side_effect=FileNotFoundError)
        with pytest.raises(FileNotFoundError):
            print_prompt_from_file(
                sys, prompt, "claude-sonnet-4-6", streaming_runner=mock_stream
            )

    def test_passes_correct_cmd(self, tmp_path: Path) -> None:
        sys, prompt = self._files(tmp_path)
        mock_stream = MagicMock(return_value=iter(["out"]))
        print_prompt_from_file(
            sys, prompt, "claude-sonnet-4-6", streaming_runner=mock_stream
        )
        cmd = mock_stream.call_args[0][0]
        assert "--model" in cmd
        assert "claude-sonnet-4-6" in cmd
        assert "--system-prompt-file" in cmd
        assert str(sys) in cmd
        assert "--print" in cmd

    def test_passes_idle_timeout(self, tmp_path: Path) -> None:
        sys, prompt = self._files(tmp_path)
        mock_stream = MagicMock(return_value=iter(["out"]))
        print_prompt_from_file(
            sys,
            prompt,
            "claude-sonnet-4-6",
            idle_timeout=600.0,
            streaming_runner=mock_stream,
        )
        assert mock_stream.call_args[0][2] == 600.0


class TestResumeSession:
    def test_returns_stdout_on_success(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("continue")
        mock_stream = MagicMock(return_value=iter(["continued"]))
        result = resume_session(
            "sess-123", prompt_file, "claude-sonnet-4-6", streaming_runner=mock_stream
        )
        assert result == "continued"

    def test_raises_on_nonzero(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("p")
        mock_stream = MagicMock(side_effect=ClaudeStreamError(1))
        with pytest.raises(ClaudeStreamError):
            resume_session(
                "sess-123",
                prompt_file,
                "claude-sonnet-4-6",
                streaming_runner=mock_stream,
            )

    def test_raises_on_idle_timeout(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("p")
        mock_stream = MagicMock(side_effect=ClaudeStreamError(-1))
        with pytest.raises(ClaudeStreamError):
            resume_session(
                "sess-123",
                prompt_file,
                "claude-sonnet-4-6",
                streaming_runner=mock_stream,
            )

    def test_raises_on_file_not_found(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("p")
        mock_stream = MagicMock(side_effect=FileNotFoundError)
        with pytest.raises(FileNotFoundError):
            resume_session(
                "sess-123",
                prompt_file,
                "claude-sonnet-4-6",
                streaming_runner=mock_stream,
            )

    def test_passes_correct_cmd(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("p")
        mock_stream = MagicMock(return_value=iter(["out"]))
        resume_session(
            "my-session-id",
            prompt_file,
            "claude-sonnet-4-6",
            streaming_runner=mock_stream,
        )
        cmd = mock_stream.call_args[0][0]
        assert "--resume" in cmd
        assert "my-session-id" in cmd
        assert "--print" in cmd

    def test_passes_idle_timeout(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("p")
        mock_stream = MagicMock(return_value=iter(["out"]))
        resume_session(
            "sess-1",
            prompt_file,
            "claude-sonnet-4-6",
            idle_timeout=900.0,
            streaming_runner=mock_stream,
        )
        assert mock_stream.call_args[0][2] == 900.0


class TestTriageComment:
    def test_returns_first_line(self) -> None:
        mock_run = MagicMock(return_value=_completed("ACT: fix the thing\nextra"))
        assert triage_comment("triage this", runner=mock_run) == "ACT: fix the thing"

    def test_returns_empty_on_nonzero(self) -> None:
        mock_run = MagicMock(return_value=_completed("ACT", returncode=1))
        assert triage_comment("triage this", runner=mock_run) == ""

    def test_returns_empty_on_empty_output(self) -> None:
        mock_run = MagicMock(return_value=_completed(""))
        assert triage_comment("triage this", runner=mock_run) == ""

    def test_returns_empty_on_timeout(self) -> None:
        mock_run = MagicMock(side_effect=subprocess.TimeoutExpired("claude", 15))
        assert triage_comment("triage", runner=mock_run) == ""

    def test_returns_empty_on_file_not_found(self) -> None:
        mock_run = MagicMock(side_effect=FileNotFoundError)
        assert triage_comment("triage", runner=mock_run) == ""

    def test_default_model_and_timeout(self) -> None:
        mock_run = MagicMock(return_value=_completed("ACT: thing"))
        triage_comment("triage this", runner=mock_run)
        cmd = mock_run.call_args.args[0]
        assert "claude-opus-4-6" in cmd
        assert mock_run.call_args.kwargs["timeout"] == 15

    def test_custom_model_and_timeout(self) -> None:
        mock_run = MagicMock(return_value=_completed("DO: fix"))
        triage_comment(
            "triage", model="claude-haiku-4-5-20251001", timeout=5, runner=mock_run
        )
        cmd = mock_run.call_args.args[0]
        assert "claude-haiku-4-5-20251001" in cmd
        assert mock_run.call_args.kwargs["timeout"] == 5


class TestGenerateReply:
    def test_returns_stripped_output(self) -> None:
        mock_run = MagicMock(return_value=_completed("  woof!  \n"))
        assert generate_reply("write a reply", runner=mock_run) == "woof!"

    def test_returns_empty_on_nonzero(self) -> None:
        mock_run = MagicMock(return_value=_completed("err", returncode=1))
        assert generate_reply("write", runner=mock_run) == ""

    def test_returns_empty_on_timeout(self) -> None:
        mock_run = MagicMock(side_effect=subprocess.TimeoutExpired("claude", 30))
        assert generate_reply("write", runner=mock_run) == ""

    def test_returns_empty_on_file_not_found(self) -> None:
        mock_run = MagicMock(side_effect=FileNotFoundError)
        assert generate_reply("write", runner=mock_run) == ""

    def test_default_model_and_timeout(self) -> None:
        mock_run = MagicMock(return_value=_completed("ok"))
        generate_reply("write a reply", runner=mock_run)
        cmd = mock_run.call_args.args[0]
        assert "claude-opus-4-6" in cmd
        assert mock_run.call_args.kwargs["timeout"] == 30

    def test_custom_model_and_timeout(self) -> None:
        mock_run = MagicMock(return_value=_completed("ok"))
        generate_reply("write", model="claude-sonnet-4-6", timeout=10, runner=mock_run)
        cmd = mock_run.call_args.args[0]
        assert "claude-sonnet-4-6" in cmd
        assert mock_run.call_args.kwargs["timeout"] == 10


class TestGenerateBranchName:
    def test_returns_first_line(self) -> None:
        mock_run = MagicMock(return_value=_completed("add-tests\nextra line"))
        assert (
            generate_branch_name("make branch for: add tests", runner=mock_run)
            == "add-tests"
        )

    def test_returns_empty_on_nonzero(self) -> None:
        mock_run = MagicMock(return_value=_completed("slug", returncode=1))
        assert generate_branch_name("make branch", runner=mock_run) == ""

    def test_returns_empty_on_empty_output(self) -> None:
        mock_run = MagicMock(return_value=_completed(""))
        assert generate_branch_name("make branch", runner=mock_run) == ""

    def test_returns_empty_on_timeout(self) -> None:
        mock_run = MagicMock(side_effect=subprocess.TimeoutExpired("claude", 15))
        assert generate_branch_name("make branch", runner=mock_run) == ""

    def test_returns_empty_on_file_not_found(self) -> None:
        mock_run = MagicMock(side_effect=FileNotFoundError)
        assert generate_branch_name("make branch", runner=mock_run) == ""

    def test_default_model_and_timeout(self) -> None:
        mock_run = MagicMock(return_value=_completed("fix-bug"))
        generate_branch_name("fix bug in parser", runner=mock_run)
        cmd = mock_run.call_args.args[0]
        assert "claude-haiku-4-5-20251001" in cmd
        assert mock_run.call_args.kwargs["timeout"] == 15

    def test_custom_model_and_timeout(self) -> None:
        mock_run = MagicMock(return_value=_completed("fix-bug"))
        generate_branch_name(
            "fix bug", model="claude-opus-4-6", timeout=20, runner=mock_run
        )
        cmd = mock_run.call_args.args[0]
        assert "claude-opus-4-6" in cmd
        assert mock_run.call_args.kwargs["timeout"] == 20


class TestGenerateStatus:
    def test_returns_two_lines(self) -> None:
        mock_run = MagicMock(return_value=_completed("🐶\ncoding up a storm"))
        result = generate_status("working on #42", "be fido", runner=mock_run)
        assert result == "🐶\ncoding up a storm"

    def test_returns_empty_on_failure(self) -> None:
        mock_run = MagicMock(return_value=_completed("", returncode=1))
        result = generate_status("working", "sys", runner=mock_run)
        assert result == ""

    def test_returns_empty_on_timeout(self) -> None:
        mock_run = MagicMock(side_effect=subprocess.TimeoutExpired("claude", 15))
        result = generate_status("working", "sys", runner=mock_run)
        assert result == ""

    def test_passes_system_prompt(self) -> None:
        mock_run = MagicMock(return_value=_completed("🚀\nworking"))
        generate_status("doing stuff", system_prompt="be a dog", runner=mock_run)
        cmd = mock_run.call_args.args[0]
        assert "--system-prompt" in cmd
        assert "be a dog" in cmd

    def test_default_model_and_timeout(self) -> None:
        mock_run = MagicMock(return_value=_completed("🐶\nwoof"))
        generate_status("working", "sys", runner=mock_run)
        cmd = mock_run.call_args.args[0]
        assert "claude-opus-4-6" in cmd
        assert mock_run.call_args.kwargs["timeout"] == 15

    def test_custom_model_and_timeout(self) -> None:
        mock_run = MagicMock(return_value=_completed("🐶\nwoof"))
        generate_status(
            "working", "sys", model="claude-sonnet-4-6", timeout=5, runner=mock_run
        )
        cmd = mock_run.call_args.args[0]
        assert "claude-sonnet-4-6" in cmd
        assert mock_run.call_args.kwargs["timeout"] == 5


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


class TestGenerateStatusWithSession:
    _RESULT_LINE = '{"type":"result","result":"🐶\\ncoding","session_id":"sess-42"}'

    def test_returns_text_and_session_id_on_success(self) -> None:
        mock_run = MagicMock(return_value=_completed(self._RESULT_LINE))
        text, sid = generate_status_with_session(
            "doing stuff", "be fido", runner=mock_run
        )
        assert text == "🐶\ncoding"
        assert sid == "sess-42"

    def test_returns_empty_pair_on_nonzero(self) -> None:
        mock_run = MagicMock(return_value=_completed(self._RESULT_LINE, returncode=1))
        mock_sleep = MagicMock()
        assert generate_status_with_session(
            "doing stuff", "sys", runner=mock_run, _sleep=mock_sleep
        ) == ("", "")
        mock_run.assert_called_once()
        mock_sleep.assert_not_called()

    def test_returns_empty_pair_on_timeout(self) -> None:
        mock_run = MagicMock(side_effect=subprocess.TimeoutExpired("claude", 15))
        mock_sleep = MagicMock()
        assert generate_status_with_session(
            "doing stuff", "sys", runner=mock_run, _sleep=mock_sleep
        ) == ("", "")
        mock_run.assert_called_once()
        mock_sleep.assert_not_called()

    def test_returns_empty_pair_on_file_not_found(self) -> None:
        mock_run = MagicMock(side_effect=FileNotFoundError)
        mock_sleep = MagicMock()
        assert generate_status_with_session(
            "doing stuff", "sys", runner=mock_run, _sleep=mock_sleep
        ) == ("", "")
        mock_run.assert_called_once()
        mock_sleep.assert_not_called()

    def test_passes_correct_flags(self) -> None:
        mock_run = MagicMock(return_value=_completed(self._RESULT_LINE))
        generate_status_with_session(
            "working on #42",
            "be a dog",
            model="claude-sonnet-4-6",
            timeout=10,
            runner=mock_run,
        )
        cmd = mock_run.call_args.args[0]
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
        assert mock_run.call_args.kwargs["timeout"] == 10

    def test_default_model_and_timeout(self) -> None:
        mock_run = MagicMock(return_value=_completed(self._RESULT_LINE))
        generate_status_with_session("working", "sys", runner=mock_run)
        cmd = mock_run.call_args.args[0]
        assert "claude-opus-4-6" in cmd
        assert mock_run.call_args.kwargs["timeout"] == 15

    def test_returns_empty_pair_when_no_result_field_after_retries(self) -> None:
        no_result = '{"type":"result","session_id":"sid"}'
        mock_run = MagicMock(return_value=_completed(no_result))
        mock_sleep = MagicMock()
        text, sid = generate_status_with_session(
            "working", "sys", runner=mock_run, _sleep=mock_sleep
        )
        assert text == ""
        assert sid == ""
        assert mock_run.call_count == 3
        assert mock_sleep.call_count == 2

    def test_returns_empty_session_when_no_session_id(self) -> None:
        no_sid = '{"type":"result","result":"🐶\\nwoof"}'
        mock_run = MagicMock(return_value=_completed(no_sid))
        text, sid = generate_status_with_session("working", "sys", runner=mock_run)
        assert text == "🐶\nwoof"
        assert sid == ""

    def test_retries_on_empty_output_then_succeeds(self) -> None:
        empty_line = '{"type":"result","session_id":"s1"}'
        mock_run = MagicMock(
            side_effect=[
                _completed(empty_line),
                _completed(empty_line),
                _completed(self._RESULT_LINE),
            ]
        )
        mock_sleep = MagicMock()
        text, sid = generate_status_with_session(
            "working", "sys", runner=mock_run, _sleep=mock_sleep
        )
        assert text == "🐶\ncoding"
        assert sid == "sess-42"
        assert mock_run.call_count == 3
        assert mock_sleep.call_count == 2

    def test_returns_empty_pair_after_all_retries_exhausted(self) -> None:
        empty_line = '{"type":"result","session_id":"s1"}'
        mock_run = MagicMock(return_value=_completed(empty_line))
        mock_sleep = MagicMock()
        assert generate_status_with_session(
            "working", "sys", runner=mock_run, _sleep=mock_sleep
        ) == ("", "")
        assert mock_run.call_count == 3
        assert mock_sleep.call_count == 2

    def test_logs_stderr_at_warning_on_empty_output(self, caplog) -> None:
        empty_line = '{"type":"result","session_id":"s1"}'
        mock_run = MagicMock(
            return_value=_completed(empty_line, stderr="Rate limit exceeded")
        )
        with caplog.at_level(logging.WARNING, logger="kennel.claude"):
            generate_status_with_session(
                "working", "sys", runner=mock_run, _sleep=MagicMock()
            )
        assert "Rate limit exceeded" in caplog.text

    def test_logs_raw_stdout_at_debug_on_empty_output(self, caplog) -> None:
        empty_line = '{"type":"result","session_id":"s1"}'
        mock_run = MagicMock(return_value=_completed(empty_line))
        with caplog.at_level(logging.DEBUG, logger="kennel.claude"):
            generate_status_with_session(
                "working", "sys", runner=mock_run, _sleep=MagicMock()
            )
        assert "stdout=" in caplog.text

    def test_no_stderr_log_when_stderr_empty(self, caplog) -> None:
        empty_line = '{"type":"result","session_id":"s1"}'
        mock_run = MagicMock(return_value=_completed(empty_line))
        with caplog.at_level(logging.WARNING, logger="kennel.claude"):
            generate_status_with_session(
                "working", "sys", runner=mock_run, _sleep=MagicMock()
            )
        assert "stderr=" not in caplog.text


class TestGenerateStatusEmoji:
    def test_returns_emoji(self) -> None:
        mock_run = MagicMock(return_value=_completed("🐕"))
        result = generate_status_emoji("pick emoji", "be fido", runner=mock_run)
        assert result == "🐕"

    def test_returns_empty_on_failure(self) -> None:
        mock_run = MagicMock(return_value=_completed("", returncode=1))
        result = generate_status_emoji("pick emoji", "sys", runner=mock_run)
        assert result == ""

    def test_passes_correct_flags(self) -> None:
        mock_run = MagicMock(return_value=_completed("🐕"))
        generate_status_emoji(
            "pick emoji",
            "be fido",
            model="claude-sonnet-4-6",
            timeout=10,
            runner=mock_run,
        )
        cmd = mock_run.call_args.args[0]
        assert "claude-sonnet-4-6" in cmd
        assert "be fido" in cmd
        assert mock_run.call_args.kwargs["timeout"] == 10


class TestResumeStatus:
    _RESULT_LINE = '{"type":"result","result":"🐕\\nfetching","session_id":"s-1"}'

    def test_returns_text_on_success(self) -> None:
        mock_run = MagicMock(return_value=_completed(self._RESULT_LINE))
        result = resume_status("s-1", "please shorten", runner=mock_run)
        assert result == "🐕\nfetching"

    def test_returns_empty_on_nonzero(self) -> None:
        mock_run = MagicMock(return_value=_completed(self._RESULT_LINE, returncode=1))
        assert resume_status("s-1", "shorten", runner=mock_run) == ""

    def test_returns_empty_on_timeout(self) -> None:
        mock_run = MagicMock(side_effect=subprocess.TimeoutExpired("claude", 15))
        assert resume_status("s-1", "shorten", runner=mock_run) == ""

    def test_returns_empty_on_file_not_found(self) -> None:
        mock_run = MagicMock(side_effect=FileNotFoundError)
        assert resume_status("s-1", "shorten", runner=mock_run) == ""

    def test_passes_correct_flags(self) -> None:
        mock_run = MagicMock(return_value=_completed(self._RESULT_LINE))
        resume_status(
            "my-session",
            "shorten to 80 chars",
            model="claude-sonnet-4-6",
            timeout=20,
            runner=mock_run,
        )
        cmd = mock_run.call_args.args[0]
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
        assert mock_run.call_args.kwargs["timeout"] == 20

    def test_default_model_and_timeout(self) -> None:
        mock_run = MagicMock(return_value=_completed(self._RESULT_LINE))
        resume_status("s-1", "shorten", runner=mock_run)
        cmd = mock_run.call_args.args[0]
        assert "claude-opus-4-6" in cmd
        assert mock_run.call_args.kwargs["timeout"] == 15

    def test_returns_empty_when_no_result_field(self) -> None:
        no_result = '{"type":"result","session_id":"s-1"}'
        mock_run = MagicMock(return_value=_completed(no_result))
        assert resume_status("s-1", "shorten", runner=mock_run) == ""


class TestKillActiveChildren:
    def _fake_proc(self, alive_after_term: bool = False) -> MagicMock:
        proc = MagicMock(spec=subprocess.Popen)
        proc.terminate = MagicMock()
        proc.kill = MagicMock()
        if alive_after_term:
            proc.wait = MagicMock(
                side_effect=[subprocess.TimeoutExpired(cmd="x", timeout=2), None]
            )
        else:
            proc.wait = MagicMock(return_value=None)
        return proc

    def test_no_children_is_noop(self) -> None:
        kill_active_children()  # nothing registered

    def test_terminates_registered_children(self) -> None:
        proc1 = self._fake_proc()
        proc2 = self._fake_proc()
        _register_child(proc1)
        _register_child(proc2)
        try:
            kill_active_children()
        finally:
            _unregister_child(proc1)
            _unregister_child(proc2)
        proc1.terminate.assert_called_once()
        proc2.terminate.assert_called_once()

    def test_kills_stragglers_after_grace(self) -> None:
        proc = self._fake_proc(alive_after_term=True)
        _register_child(proc)
        try:
            kill_active_children(grace_seconds=0.0)
        finally:
            _unregister_child(proc)
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()

    def test_swallows_oserror_on_terminate(self) -> None:
        proc = MagicMock(spec=subprocess.Popen)
        proc.terminate.side_effect = ProcessLookupError()
        proc.wait = MagicMock(return_value=None)
        _register_child(proc)
        try:
            kill_active_children()  # must not raise
        finally:
            _unregister_child(proc)

    def test_swallows_oserror_on_wait(self) -> None:
        proc = MagicMock(spec=subprocess.Popen)
        proc.terminate = MagicMock()
        proc.wait = MagicMock(side_effect=ProcessLookupError())
        _register_child(proc)
        try:
            kill_active_children()  # must not raise
        finally:
            _unregister_child(proc)

    def test_swallows_oserror_on_kill_after_timeout(self) -> None:
        proc = MagicMock(spec=subprocess.Popen)
        proc.terminate = MagicMock()
        proc.wait = MagicMock(
            side_effect=[subprocess.TimeoutExpired(cmd="x", timeout=2), None]
        )
        proc.kill.side_effect = ProcessLookupError()
        _register_child(proc)
        try:
            kill_active_children(grace_seconds=0.0)  # must not raise
        finally:
            _unregister_child(proc)


class TestRunStreamingTracksChildren:
    def _make_proc(self, lines: list[str]) -> MagicMock:
        proc = MagicMock()
        proc.stdout = MagicMock()
        proc.stdout.readline = MagicMock(side_effect=lines + [""])
        proc.stdout.read = MagicMock(return_value="")
        proc.poll = MagicMock(return_value=None)
        proc.wait = MagicMock(return_value=None)
        proc.returncode = 0
        return proc

    def test_registers_and_unregisters(self, tmp_path: Path) -> None:
        from kennel.claude import _active_children, _run_streaming

        stdin_file = tmp_path / "in"
        stdin_file.write_text("hi")
        proc = self._make_proc(["one\n"])
        captured: list = []

        def fake_popen(*args, **kwargs):
            captured.append(proc in _active_children)
            return proc

        def fake_select(rs, ws, xs, t):
            return (rs, [], [])

        list(
            _run_streaming(
                ["claude"],
                stdin_file,
                idle_timeout=1.0,
                popen=fake_popen,
                selector=fake_select,
            )
        )
        # Was registered (popen sees it not yet in set, but it is by the time
        # the generator runs); after exhaustion it should be unregistered.
        assert proc not in _active_children


# ── ClaudeSession tests ───────────────────────────────────────────────────────


def _make_session_proc(
    stdout_lines: list[str],
    poll_returns: int | None = None,
) -> MagicMock:
    """Build a mock Popen object for ClaudeSession tests.

    *stdout_lines* are returned by successive ``readline()`` calls; an empty
    string is appended automatically to signal EOF.  *poll_returns* is the
    value ``poll()`` yields (``None`` = still running; ``0`` = exited).
    """
    proc = MagicMock(spec=subprocess.Popen)
    proc.stdin = MagicMock()
    proc.stdin.closed = False
    proc.stdout = MagicMock()
    proc.stdout.readline = MagicMock(side_effect=stdout_lines + [""])
    proc.stderr = MagicMock()
    proc.poll = MagicMock(return_value=poll_returns)
    proc.wait = MagicMock(return_value=0)
    proc.returncode = 0
    return proc


def _make_session(
    tmp_path: Path,
    proc: MagicMock,
    *,
    idle_timeout: float = 1800.0,
) -> ClaudeSession:
    """Construct a ClaudeSession injecting *proc* as the subprocess."""
    system_file = tmp_path / "system.md"
    system_file.write_text("you are fido")
    fake_popen = MagicMock(return_value=proc)
    fake_selector = MagicMock(return_value=([proc.stdout], [], []))
    return ClaudeSession(
        system_file,
        work_dir=tmp_path,
        idle_timeout=idle_timeout,
        popen=fake_popen,
        selector=fake_selector,
    )


class TestClaudeSessionInit:
    def test_spawns_with_stream_json_flags(self, tmp_path: Path) -> None:
        system_file = tmp_path / "system.md"
        system_file.write_text("sys")
        proc = _make_session_proc([])
        fake_popen = MagicMock(return_value=proc)
        fake_selector = MagicMock(return_value=([], [], []))
        ClaudeSession(
            system_file, work_dir=tmp_path, popen=fake_popen, selector=fake_selector
        )
        cmd = fake_popen.call_args.args[0]
        assert cmd[0] == "claude"
        assert "--input-format" in cmd
        assert "stream-json" in cmd[cmd.index("--input-format") + 1]
        assert "--output-format" in cmd
        assert "stream-json" in cmd[cmd.index("--output-format") + 1]
        assert "--verbose" in cmd
        assert "--dangerously-skip-permissions" in cmd
        assert "--system-prompt-file" in cmd
        assert str(system_file) in cmd

    def test_opens_stdin_stdout_pipes(self, tmp_path: Path) -> None:
        system_file = tmp_path / "system.md"
        system_file.write_text("sys")
        proc = _make_session_proc([])
        fake_popen = MagicMock(return_value=proc)
        fake_selector = MagicMock(return_value=([], [], []))
        ClaudeSession(system_file, popen=fake_popen, selector=fake_selector)
        kwargs = fake_popen.call_args.kwargs
        assert kwargs["stdin"] == subprocess.PIPE
        assert kwargs["stdout"] == subprocess.PIPE
        assert kwargs["text"] is True

    def test_passes_cwd(self, tmp_path: Path) -> None:
        system_file = tmp_path / "system.md"
        system_file.write_text("sys")
        proc = _make_session_proc([])
        fake_popen = MagicMock(return_value=proc)
        fake_selector = MagicMock(return_value=([], [], []))
        ClaudeSession(
            system_file, work_dir=tmp_path, popen=fake_popen, selector=fake_selector
        )
        assert fake_popen.call_args.kwargs["cwd"] == tmp_path

    def test_registers_in_active_children(self, tmp_path: Path) -> None:
        system_file = tmp_path / "system.md"
        system_file.write_text("sys")
        proc = _make_session_proc([])
        fake_popen = MagicMock(return_value=proc)
        fake_selector = MagicMock(return_value=([], [], []))
        session = ClaudeSession(system_file, popen=fake_popen, selector=fake_selector)
        assert proc in _active_children
        # cleanup
        session.stop()


class TestClaudeSessionSend:
    def test_writes_json_user_message_to_stdin(self, tmp_path: Path) -> None:
        import json as _json

        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        session.send("hello fido")
        written = proc.stdin.write.call_args.args[0]
        obj = _json.loads(written.strip())
        assert obj["type"] == "user"
        assert obj["message"]["role"] == "user"
        assert obj["message"]["content"] == "hello fido"

    def test_flushes_after_write(self, tmp_path: Path) -> None:
        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        session.send("ping")
        proc.stdin.flush.assert_called()

    def test_appends_newline(self, tmp_path: Path) -> None:
        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        session.send("msg")
        written = proc.stdin.write.call_args.args[0]
        assert written.endswith("\n")


class TestClaudeSessionIterEvents:
    def test_yields_parsed_json_events(self, tmp_path: Path) -> None:
        import json as _json

        lines = [
            _json.dumps({"type": "assistant", "text": "thinking"}) + "\n",
            _json.dumps({"type": "result", "result": "done", "session_id": "s1"})
            + "\n",
        ]
        proc = _make_session_proc(lines)
        session = _make_session(tmp_path, proc)
        events = list(session.iter_events())
        assert events[0]["type"] == "assistant"
        assert events[1]["type"] == "result"

    def test_stops_after_result_event(self, tmp_path: Path) -> None:
        import json as _json

        lines = [
            _json.dumps({"type": "result", "result": "done"}) + "\n",
            _json.dumps({"type": "assistant", "text": "extra"}) + "\n",
        ]
        proc = _make_session_proc(lines)
        session = _make_session(tmp_path, proc)
        events = list(session.iter_events())
        # Only the result event should be yielded; extra line is not consumed
        assert len(events) == 1
        assert events[0]["type"] == "result"

    def test_stops_after_error_event(self, tmp_path: Path) -> None:
        import json as _json

        lines = [
            _json.dumps({"type": "error", "error": "oops"}) + "\n",
        ]
        proc = _make_session_proc(lines)
        session = _make_session(tmp_path, proc)
        events = list(session.iter_events())
        assert len(events) == 1
        assert events[0]["type"] == "error"

    def test_stops_on_eof(self, tmp_path: Path) -> None:
        proc = _make_session_proc([])  # immediately EOF
        session = _make_session(tmp_path, proc)
        events = list(session.iter_events())
        assert events == []

    def test_stops_when_process_exits(self, tmp_path: Path) -> None:
        system_file = tmp_path / "system.md"
        system_file.write_text("sys")
        proc = _make_session_proc([], poll_returns=0)
        fake_popen = MagicMock(return_value=proc)
        # selector never returns ready — forces poll() branch
        fake_selector = MagicMock(return_value=([], [], []))
        session = ClaudeSession(system_file, popen=fake_popen, selector=fake_selector)
        events = list(session.iter_events())
        assert events == []

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        import json as _json

        lines = [
            "\n",
            "   \n",
            _json.dumps({"type": "result", "result": "ok"}) + "\n",
        ]
        proc = _make_session_proc(lines)
        session = _make_session(tmp_path, proc)
        events = list(session.iter_events())
        assert len(events) == 1
        assert events[0]["type"] == "result"

    def test_skips_unparseable_lines(self, tmp_path: Path, caplog) -> None:
        import json as _json

        lines = [
            "not json at all\n",
            _json.dumps({"type": "result", "result": "ok"}) + "\n",
        ]
        proc = _make_session_proc(lines)
        session = _make_session(tmp_path, proc)
        with caplog.at_level(logging.WARNING, logger="kennel.claude"):
            events = list(session.iter_events())
        assert any("unparseable" in r.message for r in caplog.records)
        assert len(events) == 1

    def test_raises_on_idle_timeout(self, tmp_path: Path) -> None:
        system_file = tmp_path / "system.md"
        system_file.write_text("sys")
        proc = _make_session_proc([])
        proc.poll = MagicMock(return_value=None)  # never exits
        fake_popen = MagicMock(return_value=proc)
        # selector never ready, process never exits → triggers idle timeout
        fake_selector = MagicMock(return_value=([], [], []))
        session = ClaudeSession(
            system_file,
            idle_timeout=0.0,
            popen=fake_popen,
            selector=fake_selector,
        )
        with pytest.raises(ClaudeStreamError) as exc_info:
            list(session.iter_events())
        assert exc_info.value.returncode == -1
        proc.kill.assert_called()


class TestClaudeSessionStop:
    def test_closes_stdin_and_waits(self, tmp_path: Path) -> None:
        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        session.stop()
        proc.stdin.close.assert_called()
        proc.wait.assert_called()

    def test_unregisters_from_active_children(self, tmp_path: Path) -> None:
        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        assert proc in _active_children
        session.stop()
        assert proc not in _active_children

    def test_kills_if_wait_times_out(self, tmp_path: Path) -> None:
        proc = _make_session_proc([])
        proc.wait = MagicMock(
            side_effect=[subprocess.TimeoutExpired(cmd="claude", timeout=2), None]
        )
        session = _make_session(tmp_path, proc)
        session.stop(grace_seconds=0.0)
        proc.kill.assert_called()

    def test_swallows_oserror_on_stdin_close(self, tmp_path: Path) -> None:
        proc = _make_session_proc([])
        proc.stdin.close = MagicMock(side_effect=OSError("broken pipe"))
        session = _make_session(tmp_path, proc)
        session.stop()  # must not raise

    def test_swallows_oserror_on_wait(self, tmp_path: Path) -> None:
        proc = _make_session_proc([])
        proc.wait = MagicMock(side_effect=OSError("already gone"))
        session = _make_session(tmp_path, proc)
        session.stop()  # must not raise

    def test_swallows_errors_on_kill_after_timeout(self, tmp_path: Path) -> None:
        proc = _make_session_proc([])
        proc.wait = MagicMock(
            side_effect=[subprocess.TimeoutExpired(cmd="x", timeout=2), None]
        )
        proc.kill = MagicMock(side_effect=ProcessLookupError())
        session = _make_session(tmp_path, proc)
        session.stop(grace_seconds=0.0)  # must not raise

    def test_skips_close_when_stdin_already_closed(self, tmp_path: Path) -> None:
        proc = _make_session_proc([])
        proc.stdin.closed = True
        session = _make_session(tmp_path, proc)
        session.stop()
        proc.stdin.close.assert_not_called()
        proc.wait.assert_called()


class TestClaudeSessionSwitchModel:
    def test_sends_model_slash_command(self, tmp_path: Path) -> None:
        import json as _json

        result_line = _json.dumps({"type": "result", "result": ""}) + "\n"
        proc = _make_session_proc([result_line])
        session = _make_session(tmp_path, proc)
        session.switch_model("claude-opus-4-6")
        written = proc.stdin.write.call_args.args[0]
        obj = _json.loads(written.strip())
        assert obj["message"]["content"] == "/model claude-opus-4-6"

    def test_drains_response_events(self, tmp_path: Path) -> None:
        import json as _json

        lines = [
            _json.dumps({"type": "assistant", "text": "Switching..."}) + "\n",
            _json.dumps({"type": "result", "result": ""}) + "\n",
        ]
        proc = _make_session_proc(lines)
        session = _make_session(tmp_path, proc)
        # Must not raise or leave unread events blocking future reads
        session.switch_model("claude-sonnet-4-6")

    def test_works_when_command_produces_no_output(self, tmp_path: Path) -> None:
        proc = _make_session_proc([])  # immediate EOF
        session = _make_session(tmp_path, proc)
        session.switch_model("claude-haiku-4-5-20251001")  # must not raise


class TestClaudeSessionConsumeUntilResult:
    def test_returns_result_text(self, tmp_path: Path) -> None:
        import json as _json

        lines = [
            _json.dumps({"type": "assistant", "text": "thinking..."}) + "\n",
            _json.dumps({"type": "result", "result": "all done"}) + "\n",
        ]
        proc = _make_session_proc(lines)
        session = _make_session(tmp_path, proc)
        assert session.consume_until_result() == "all done"

    def test_returns_empty_on_eof_without_result(self, tmp_path: Path) -> None:
        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        assert session.consume_until_result() == ""

    def test_returns_empty_on_error_event(self, tmp_path: Path) -> None:
        import json as _json

        lines = [_json.dumps({"type": "error", "error": "something broke"}) + "\n"]
        proc = _make_session_proc(lines)
        session = _make_session(tmp_path, proc)
        assert session.consume_until_result() == ""

    def test_returns_empty_when_result_field_not_a_string(self, tmp_path: Path) -> None:
        import json as _json

        lines = [_json.dumps({"type": "result", "result": None}) + "\n"]
        proc = _make_session_proc(lines)
        session = _make_session(tmp_path, proc)
        assert session.consume_until_result() == ""

    def test_drains_all_intermediate_events(self, tmp_path: Path) -> None:
        import json as _json

        lines = [
            _json.dumps({"type": "system", "subtype": "init"}) + "\n",
            _json.dumps({"type": "assistant", "text": "a"}) + "\n",
            _json.dumps({"type": "assistant", "text": "b"}) + "\n",
            _json.dumps({"type": "result", "result": "output"}) + "\n",
        ]
        proc = _make_session_proc(lines)
        session = _make_session(tmp_path, proc)
        assert session.consume_until_result() == "output"
