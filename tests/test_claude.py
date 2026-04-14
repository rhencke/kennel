from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kennel.claude import (
    _LOG_LINE_TRUNCATE,
    _RETURNCODE_IDLE_TIMEOUT,
    ClaudeSession,
    ClaudeStreamError,
    _active_children,
    _claude,
    _register_child,
    _Trunc,
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


class TestTrunc:
    def test_str_truncates_long_string(self) -> None:
        t = _Trunc("x" * (_LOG_LINE_TRUNCATE + 50))
        assert str(t) == "x" * _LOG_LINE_TRUNCATE

    def test_str_passes_short_string_unchanged(self) -> None:
        assert str(_Trunc("hi")) == "hi"

    def test_repr_truncates_and_quotes(self) -> None:
        t = _Trunc("x" * (_LOG_LINE_TRUNCATE + 50))
        assert repr(t) == repr("x" * _LOG_LINE_TRUNCATE)

    def test_repr_passes_short_string_unchanged(self) -> None:
        assert repr(_Trunc("hi")) == repr("hi")


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

    def test_default_runner_uses_explicit_popen(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When no runner is overridden, _claude drives Popen directly so
        TimeoutExpired actually reaps the child (closes #489)."""
        import subprocess

        fake_proc = MagicMock()
        fake_proc.args = ["claude", "--print"]
        fake_proc.returncode = 0
        fake_proc.communicate = MagicMock(return_value=("hello", ""))
        fake_popen = MagicMock(return_value=fake_proc)
        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        result = _claude("--print", "-p", "hi", prompt="body", timeout=5)
        fake_popen.assert_called_once()
        fake_proc.communicate.assert_called_once_with(input="body", timeout=5)
        assert result.stdout == "hello"
        assert result.returncode == 0

    def test_default_runner_no_prompt_uses_devnull(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess

        fake_proc = MagicMock()
        fake_proc.args = ["claude", "--version"]
        fake_proc.returncode = 0
        fake_proc.communicate = MagicMock(return_value=("1.0", ""))
        fake_popen = MagicMock(return_value=fake_proc)
        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        _claude("--version")
        assert fake_popen.call_args.kwargs["stdin"] is subprocess.DEVNULL

    def test_default_runner_kills_on_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """TimeoutExpired from communicate → kill → re-raise with output."""
        import subprocess

        fake_proc = MagicMock()
        fake_proc.args = ["claude", "--print"]
        fake_proc.returncode = -9
        # First communicate call times out; second (after kill) returns.
        fake_proc.communicate = MagicMock(
            side_effect=[
                subprocess.TimeoutExpired("claude", 5),
                ("partial", "err"),
            ]
        )
        fake_proc.kill = MagicMock()
        monkeypatch.setattr(subprocess, "Popen", MagicMock(return_value=fake_proc))
        with pytest.raises(subprocess.TimeoutExpired) as exc_info:
            _claude("--print", "-p", "hi", prompt="x", timeout=5)
        fake_proc.kill.assert_called_once()
        assert exc_info.value.output == "partial"
        assert exc_info.value.stderr == "err"

    def test_default_runner_unresponsive_after_kill(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Child that ignores SIGKILL too — still re-raise TimeoutExpired."""
        import subprocess

        fake_proc = MagicMock()
        fake_proc.args = ["claude", "--print"]
        fake_proc.returncode = -9
        fake_proc.communicate = MagicMock(
            side_effect=[
                subprocess.TimeoutExpired("claude", 5),
                subprocess.TimeoutExpired("claude", 5),
            ]
        )
        fake_proc.kill = MagicMock()
        monkeypatch.setattr(subprocess, "Popen", MagicMock(return_value=fake_proc))
        with pytest.raises(subprocess.TimeoutExpired):
            _claude("--print", "-p", "hi", prompt="x", timeout=5)


@pytest.fixture
def session_resolver():
    """Install a session resolver that returns a MagicMock session for any
    repo, and wire the thread-local repo so ``print_prompt`` can find it.
    Yields the fake session so tests can assert on ``session.prompt.*``.
    """
    from kennel import claude as claude_module

    fake = MagicMock()
    fake.is_alive.return_value = True
    claude_module.set_session_resolver(lambda repo: fake)
    claude_module.set_thread_repo("owner/repo")
    try:
        yield fake
    finally:
        claude_module.set_session_resolver(None)
        claude_module.set_thread_repo(None)


class TestPrintPrompt:
    def test_delegates_to_session_prompt(self, session_resolver) -> None:
        session_resolver.prompt.return_value = "hello"
        assert print_prompt("hi", "claude-opus-4-6") == "hello"
        session_resolver.prompt.assert_called_once_with(
            "hi", model="claude-opus-4-6", system_prompt=None
        )

    def test_passes_system_prompt(self, session_resolver) -> None:
        session_resolver.prompt.return_value = "ok"
        print_prompt("q", "claude-opus-4-6", system_prompt="be fido")
        assert session_resolver.prompt.call_args.kwargs["system_prompt"] == "be fido"

    def test_raises_without_repo_set(self) -> None:
        """No thread_repo → wiring bug, raise loud."""
        with pytest.raises(RuntimeError, match="without a thread-local repo_name"):
            print_prompt("q", "claude-opus-4-6")

    def test_raises_without_resolver_installed(self) -> None:
        from kennel import claude as claude_module

        claude_module.set_thread_repo("owner/repo")
        try:
            with pytest.raises(RuntimeError, match="before set_session_resolver"):
                print_prompt("q", "claude-opus-4-6")
        finally:
            claude_module.set_thread_repo(None)

    def test_raises_when_session_missing(self) -> None:
        from kennel import claude as claude_module

        claude_module.set_session_resolver(lambda repo: None)
        claude_module.set_thread_repo("owner/repo")
        try:
            with pytest.raises(RuntimeError, match="no ClaudeSession registered"):
                print_prompt("q", "claude-opus-4-6")
        finally:
            claude_module.set_session_resolver(None)
            claude_module.set_thread_repo(None)

    def test_raises_when_session_not_alive(self) -> None:
        from kennel import claude as claude_module

        dead = MagicMock()
        dead.is_alive.return_value = False
        claude_module.set_session_resolver(lambda repo: dead)
        claude_module.set_thread_repo("owner/repo")
        try:
            with pytest.raises(RuntimeError, match="is not alive"):
                print_prompt("q", "claude-opus-4-6")
        finally:
            claude_module.set_session_resolver(None)
            claude_module.set_thread_repo(None)

    def test_session_errors_propagate(self, session_resolver) -> None:
        """Session errors must not be masked — fail open, not silently."""
        session_resolver.prompt.side_effect = ClaudeStreamError(1)
        with pytest.raises(ClaudeStreamError):
            print_prompt("q", "claude-opus-4-6")


class TestPrintPromptJson:
    def test_extracts_key_from_clean_json(self, session_resolver) -> None:
        session_resolver.prompt.return_value = '{"description": "Fixes bug."}'
        assert print_prompt_json("q", "description", "claude-opus-4-6") == "Fixes bug."

    def test_extracts_key_when_preamble_present(self, session_resolver) -> None:
        session_resolver.prompt.return_value = (
            'Sure! Here: {"description": "Adds feature."} Done.'
        )
        assert (
            print_prompt_json("q", "description", "claude-opus-4-6") == "Adds feature."
        )

    def test_returns_empty_when_key_missing(self, session_resolver) -> None:
        session_resolver.prompt.return_value = '{"other": "value"}'
        assert print_prompt_json("q", "description", "claude-opus-4-6") == ""

    def test_returns_empty_on_no_json(self, session_resolver) -> None:
        session_resolver.prompt.return_value = "just plain text"
        assert print_prompt_json("q", "description", "claude-opus-4-6") == ""

    def test_returns_empty_on_empty_output(self, session_resolver) -> None:
        session_resolver.prompt.return_value = ""
        assert print_prompt_json("q", "description", "claude-opus-4-6") == ""

    def test_non_string_value_is_ignored(self, session_resolver) -> None:
        session_resolver.prompt.return_value = '{"description": 42}'
        assert print_prompt_json("q", "description", "claude-opus-4-6") == ""

    def test_appends_json_instruction_to_system_prompt(self, session_resolver) -> None:
        session_resolver.prompt.return_value = '{"description": "x"}'
        print_prompt_json(
            "q", "description", "claude-opus-4-6", system_prompt="be helpful"
        )
        sent_system = session_resolver.prompt.call_args.kwargs["system_prompt"]
        assert "be helpful" in sent_system
        assert "description" in sent_system

    def test_uses_json_instruction_as_only_system_prompt_when_none_given(
        self, session_resolver
    ) -> None:
        session_resolver.prompt.return_value = '{"description": "x"}'
        print_prompt_json("q", "description", "claude-opus-4-6", system_prompt=None)
        sent_system = session_resolver.prompt.call_args.kwargs["system_prompt"]
        assert "description" in sent_system


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
        """Idle-timeout kills the subprocess when no output for N seconds.

        Uses a virtual clock and mocked popen/selector so the test runs in
        microseconds — no real ``sleep`` subprocess, no wall-clock waiting.
        """
        from kennel.claude import _run_streaming

        stdin_file = tmp_path / "input.txt"
        stdin_file.write_text("")

        proc = MagicMock()
        proc.stdout = MagicMock()
        proc.stderr = MagicMock()
        proc.poll = MagicMock(return_value=None)  # still alive
        proc.wait = MagicMock(return_value=None)
        proc.kill = MagicMock()
        proc.returncode = 0

        # Selector always reports stdout-not-ready → loop falls through to
        # the idle-timeout check on every iteration.
        fake_selector = MagicMock(return_value=([], [], []))
        # Virtual clock: first call seeds ``last_activity`` at 0; subsequent
        # calls jump past idle_timeout so the check trips on iteration one.
        times = iter([0.0, 1.0])
        fake_clock = MagicMock(side_effect=lambda: next(times))

        with pytest.raises(ClaudeStreamError) as exc_info:
            list(
                _run_streaming(
                    ["anything"],
                    stdin_file,
                    idle_timeout=0.1,
                    popen=MagicMock(return_value=proc),
                    selector=fake_selector,
                    clock=fake_clock,
                )
            )
        assert exc_info.value.returncode == _RETURNCODE_IDLE_TIMEOUT
        proc.kill.assert_called_once()
        proc.wait.assert_called_once()

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
        mock_stream = MagicMock(side_effect=ClaudeStreamError(_RETURNCODE_IDLE_TIMEOUT))
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
        mock_stream = MagicMock(side_effect=ClaudeStreamError(_RETURNCODE_IDLE_TIMEOUT))
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
    def test_delegates_to_session_prompt(self, session_resolver) -> None:
        session_resolver.prompt.return_value = "🐶\ncoding up a storm"
        assert generate_status("working on #42", "be fido") == "🐶\ncoding up a storm"
        session_resolver.prompt.assert_called_once_with(
            "working on #42", model="claude-opus-4-6", system_prompt="be fido"
        )

    def test_default_model(self, session_resolver) -> None:
        session_resolver.prompt.return_value = "🐶\nwoof"
        generate_status("working", "sys")
        assert session_resolver.prompt.call_args.kwargs["model"] == "claude-opus-4-6"

    def test_custom_model(self, session_resolver) -> None:
        session_resolver.prompt.return_value = "🐶\nwoof"
        generate_status("working", "sys", model="claude-sonnet-4-6")
        assert session_resolver.prompt.call_args.kwargs["model"] == "claude-sonnet-4-6"


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
    def test_delegates_to_session_prompt(self, session_resolver) -> None:
        session_resolver.prompt.return_value = "🐕"
        assert generate_status_emoji("pick emoji", "be fido") == "🐕"
        session_resolver.prompt.assert_called_once_with(
            "pick emoji", model="claude-opus-4-6", system_prompt="be fido"
        )

    def test_custom_model(self, session_resolver) -> None:
        session_resolver.prompt.return_value = "🐕"
        generate_status_emoji("pick emoji", "be fido", model="claude-sonnet-4-6")
        assert session_resolver.prompt.call_args.kwargs["model"] == "claude-sonnet-4-6"


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

    def test_session_for_current_repo_returns_live_session(self) -> None:
        from kennel import claude as claude_module
        from kennel.claude import (
            _session_for_current_repo,
            set_session_resolver,
            set_thread_repo,
        )

        live = MagicMock()
        live.is_alive.return_value = True
        set_session_resolver(lambda repo: live if repo == "owner/repo" else None)
        set_thread_repo("owner/repo")
        try:
            assert _session_for_current_repo() is live
        finally:
            set_thread_repo(None)
            claude_module.set_session_resolver(None)

    def test_thread_name_for_id_returns_none_when_not_found(self) -> None:
        from kennel.claude import _thread_name_for_id

        # Pick a thread_id that's very unlikely to match any live thread.
        assert _thread_name_for_id(0xDEADBEEF) is None

    def test_registers_and_unregisters_talker_when_thread_repo_set(
        self, tmp_path: Path
    ) -> None:
        """When thread-local repo_name is set, _run_streaming registers a
        webhook-kind ClaudeTalker for the duration of the subprocess and
        unregisters it on exit."""
        from kennel.claude import (
            _run_streaming,
            get_talker,
            set_thread_repo,
        )

        stdin_file = tmp_path / "in"
        stdin_file.write_text("hi")
        proc = self._make_proc(["one\n"])
        proc.pid = 77777

        observed: list = []

        def fake_popen(*args, **kwargs):
            return proc

        def fake_select(rs, ws, xs, t):
            observed.append(get_talker("owner/repo"))
            return (rs, [], [])

        set_thread_repo("owner/repo")
        try:
            list(
                _run_streaming(
                    ["claude"],
                    stdin_file,
                    idle_timeout=1.0,
                    popen=fake_popen,
                    selector=fake_select,
                )
            )
        finally:
            set_thread_repo(None)
        # During the subprocess lifetime the talker was present and described
        # the one-shot pid.
        assert observed[0] is not None
        assert observed[0].kind == "webhook"
        assert observed[0].claude_pid == 77777
        # Cleanly unregistered after the generator exhausts.
        assert get_talker("owner/repo") is None


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
    proc.pid = 99999
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
        repo_name="owner/repo",
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

    def test_raises_on_unparseable_line(self, tmp_path: Path) -> None:
        import json as _json

        lines = ["not json at all\n"]
        proc = _make_session_proc(lines)
        session = _make_session(tmp_path, proc)
        with pytest.raises(_json.JSONDecodeError):
            list(session.iter_events())

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
        assert exc_info.value.returncode == _RETURNCODE_IDLE_TIMEOUT
        proc.kill.assert_called()
        # restart() must spawn a replacement process after killing
        assert fake_popen.call_count == 2

    def test_cancel_cleared_at_iter_events_start(self, tmp_path: Path) -> None:
        # A cancel set before iter_events() is called is cleared at the start of
        # the method — it represents a stale signal from a previous turn, not a
        # request to abort the new turn.
        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        session._cancel.set()
        assert session._cancel.is_set()
        # Drain iter_events; proc has EOF so loop exits naturally
        list(session.iter_events())
        # cancel must have been cleared at start of iter_events
        assert not session._cancel.is_set()
        session.stop()

    def test_stops_when_cancel_set_during_turn(self, tmp_path: Path) -> None:
        # A cancel set AFTER iter_events() starts (i.e., during polling) must
        # abort the loop on the next cycle.
        system_file = tmp_path / "system.md"
        system_file.write_text("sys")
        proc = _make_session_proc([])
        proc.poll = MagicMock(return_value=None)  # never exits on its own
        fake_popen = MagicMock(return_value=proc)

        session_ref: list[ClaudeSession] = []

        def selector_that_cancels(
            *_args: object, **_kwargs: object
        ) -> tuple[list, list, list]:
            # Set cancel on first poll — simulates an interrupt arriving mid-turn
            session_ref[0]._cancel.set()
            return ([], [], [])

        session = ClaudeSession(
            system_file, popen=fake_popen, selector=selector_that_cancels
        )
        session_ref.append(session)
        events = list(session.iter_events())
        assert events == []
        session.stop()


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

    def test_raises_oserror_on_stdin_close(self, tmp_path: Path, caplog) -> None:
        proc = _make_session_proc([])
        proc.stdin.close = MagicMock(side_effect=OSError("broken pipe"))
        session = _make_session(tmp_path, proc)
        with caplog.at_level(logging.DEBUG, logger="kennel.claude"):
            with pytest.raises(OSError):
                session.stop()
        assert any("stdin close failed" in r.message for r in caplog.records)

    def test_unregisters_even_when_stdin_close_raises(self, tmp_path: Path) -> None:
        proc = _make_session_proc([])
        proc.stdin.close = MagicMock(side_effect=OSError("broken pipe"))
        session = _make_session(tmp_path, proc)
        assert proc in _active_children
        with pytest.raises(OSError):
            session.stop()
        assert proc not in _active_children  # outer finally must still unregister

    def test_logs_oserror_on_wait(self, tmp_path: Path, caplog) -> None:
        proc = _make_session_proc([])
        proc.wait = MagicMock(side_effect=OSError("already gone"))
        session = _make_session(tmp_path, proc)
        with caplog.at_level(logging.DEBUG, logger="kennel.claude"):
            session.stop()  # must not raise
        assert any("wait failed" in r.message for r in caplog.records)

    def test_raises_and_logs_on_kill_after_timeout(
        self, tmp_path: Path, caplog
    ) -> None:
        proc = _make_session_proc([])
        proc.wait = MagicMock(
            side_effect=[subprocess.TimeoutExpired(cmd="x", timeout=2), None]
        )
        proc.kill = MagicMock(side_effect=ProcessLookupError())
        session = _make_session(tmp_path, proc)
        with caplog.at_level(logging.DEBUG, logger="kennel.claude"):
            with pytest.raises(ProcessLookupError):
                session.stop(grace_seconds=0.0)
        assert any("kill/wait failed" in r.message for r in caplog.records)

    def test_unregisters_even_when_kill_raises(self, tmp_path: Path) -> None:
        proc = _make_session_proc([])
        proc.wait = MagicMock(
            side_effect=[subprocess.TimeoutExpired(cmd="x", timeout=2), None]
        )
        proc.kill = MagicMock(side_effect=ProcessLookupError())
        session = _make_session(tmp_path, proc)
        assert proc in _active_children
        with pytest.raises(ProcessLookupError):
            session.stop(grace_seconds=0.0)
        assert proc not in _active_children

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


class TestClaudeSessionIsAliveAndRestart:
    def test_is_alive_true_when_process_running(self, tmp_path: Path) -> None:
        proc = _make_session_proc([])
        proc.poll = MagicMock(return_value=None)
        session = _make_session(tmp_path, proc)
        assert session.is_alive() is True

    def test_is_alive_false_when_process_exited(self, tmp_path: Path) -> None:
        proc = _make_session_proc([])
        proc.poll = MagicMock(return_value=0)
        session = _make_session(tmp_path, proc)
        assert session.is_alive() is False

    def test_restart_spawns_new_process(self, tmp_path: Path) -> None:
        system_file = tmp_path / "system.md"
        system_file.write_text("sys")
        old_proc = _make_session_proc([])
        new_proc = _make_session_proc([])
        fake_popen = MagicMock(side_effect=[old_proc, new_proc])
        fake_selector = MagicMock(return_value=([], [], []))
        session = ClaudeSession(
            system_file, work_dir=tmp_path, popen=fake_popen, selector=fake_selector
        )
        session.restart()
        assert session._proc is new_proc
        assert fake_popen.call_count == 2

    def test_restart_registers_new_proc_in_active_children(
        self, tmp_path: Path
    ) -> None:
        system_file = tmp_path / "system.md"
        system_file.write_text("sys")
        old_proc = _make_session_proc([])
        new_proc = _make_session_proc([])
        fake_popen = MagicMock(side_effect=[old_proc, new_proc])
        fake_selector = MagicMock(return_value=([], [], []))
        session = ClaudeSession(
            system_file, work_dir=tmp_path, popen=fake_popen, selector=fake_selector
        )
        session.restart()
        assert new_proc in _active_children
        assert old_proc not in _active_children
        # cleanup
        session.stop()

    def test_restart_unregisters_old_proc(self, tmp_path: Path) -> None:
        system_file = tmp_path / "system.md"
        system_file.write_text("sys")
        old_proc = _make_session_proc([])
        new_proc = _make_session_proc([])
        fake_popen = MagicMock(side_effect=[old_proc, new_proc])
        fake_selector = MagicMock(return_value=([], [], []))
        session = ClaudeSession(
            system_file, work_dir=tmp_path, popen=fake_popen, selector=fake_selector
        )
        session.restart()
        assert old_proc not in _active_children
        # cleanup
        session.stop()

    def test_restart_logs_warning(self, tmp_path: Path, caplog) -> None:
        import logging as _logging

        system_file = tmp_path / "system.md"
        system_file.write_text("sys")
        old_proc = _make_session_proc([])
        new_proc = _make_session_proc([])
        fake_popen = MagicMock(side_effect=[old_proc, new_proc])
        fake_selector = MagicMock(return_value=([], [], []))
        session = ClaudeSession(
            system_file, work_dir=tmp_path, popen=fake_popen, selector=fake_selector
        )
        with caplog.at_level(_logging.WARNING, logger="kennel.claude"):
            session.restart()
        assert any("restart" in r.message.lower() for r in caplog.records)
        session.stop()

    def test_restart_skips_kill_when_process_already_dead(self, tmp_path: Path) -> None:
        system_file = tmp_path / "system.md"
        system_file.write_text("sys")
        old_proc = _make_session_proc([], poll_returns=0)  # already dead
        new_proc = _make_session_proc([])
        fake_popen = MagicMock(side_effect=[old_proc, new_proc])
        fake_selector = MagicMock(return_value=([], [], []))
        session = ClaudeSession(
            system_file, work_dir=tmp_path, popen=fake_popen, selector=fake_selector
        )
        session.restart()
        old_proc.kill.assert_not_called()
        assert session._proc is new_proc
        session.stop()

    def test_restart_raises_oserror_on_kill(self, tmp_path: Path, caplog) -> None:
        system_file = tmp_path / "system.md"
        system_file.write_text("sys")
        old_proc = _make_session_proc([])
        old_proc.kill = MagicMock(side_effect=OSError("already dead"))
        new_proc = _make_session_proc([])
        fake_popen = MagicMock(side_effect=[old_proc, new_proc])
        fake_selector = MagicMock(return_value=([], [], []))
        session = ClaudeSession(
            system_file, work_dir=tmp_path, popen=fake_popen, selector=fake_selector
        )
        with caplog.at_level(logging.WARNING, logger="kennel.claude"):
            with pytest.raises(OSError):
                session.restart()
        assert any("kill/wait failed" in r.message for r in caplog.records)

    def test_restart_raises_timeout_on_wait(self, tmp_path: Path, caplog) -> None:
        import subprocess as _subprocess

        system_file = tmp_path / "system.md"
        system_file.write_text("sys")
        old_proc = _make_session_proc([])
        old_proc.wait = MagicMock(side_effect=_subprocess.TimeoutExpired("x", 1))
        new_proc = _make_session_proc([])
        fake_popen = MagicMock(side_effect=[old_proc, new_proc])
        fake_selector = MagicMock(return_value=([], [], []))
        session = ClaudeSession(
            system_file, work_dir=tmp_path, popen=fake_popen, selector=fake_selector
        )
        with caplog.at_level(logging.WARNING, logger="kennel.claude"):
            with pytest.raises(_subprocess.TimeoutExpired):
                session.restart()
        assert any("kill/wait failed" in r.message for r in caplog.records)

    def test_stop_after_restart_cleans_up_new_proc(self, tmp_path: Path) -> None:
        system_file = tmp_path / "system.md"
        system_file.write_text("sys")
        old_proc = _make_session_proc([])
        new_proc = _make_session_proc([])
        fake_popen = MagicMock(side_effect=[old_proc, new_proc])
        fake_selector = MagicMock(return_value=([], [], []))
        session = ClaudeSession(
            system_file, work_dir=tmp_path, popen=fake_popen, selector=fake_selector
        )
        session.restart()
        session.stop()
        assert new_proc not in _active_children


class TestClaudeSessionLock:
    def test_enter_returns_self(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path, _make_session_proc([]))
        assert session.__enter__() is session
        session._lock.release()
        session.stop()

    def test_enter_preserves_cancel_event(self, tmp_path: Path) -> None:
        # __enter__ must NOT clear _cancel — a signal that lands between one
        # holder's __exit__ and the next holder's iter_events() must survive.
        session = _make_session(tmp_path, _make_session_proc([]))
        session._cancel.set()
        session.__enter__()
        assert session._cancel.is_set()  # still set; iter_events() will clear it
        session._lock.release()
        session.stop()

    def test_cancel_survives_lock_handoff(self, tmp_path: Path) -> None:
        import threading as _threading

        # Simulate the lost-interrupt race: interrupt() is called after holder
        # releases the lock but before the next holder calls iter_events().
        # The signal must still be set when the new holder enters __enter__.
        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)

        cancel_set_after_exit = _threading.Event()
        new_holder_entered = _threading.Event()
        new_holder_cancel_was_set: list[bool] = []

        def first_holder() -> None:
            session._lock.acquire()
            # Release without calling iter_events; then interrupt() sets cancel
            session._lock.release()
            # Simulate interrupt arriving right after release
            session._cancel.set()
            cancel_set_after_exit.set()

        def second_holder() -> None:
            cancel_set_after_exit.wait()
            session.__enter__()
            # Record whether cancel is still set before iter_events clears it
            new_holder_cancel_was_set.append(session._cancel.is_set())
            new_holder_entered.set()
            session._lock.release()

        t1 = _threading.Thread(target=first_holder)
        t2 = _threading.Thread(target=second_holder)
        t1.start()
        t2.start()
        t1.join(timeout=2)
        t2.join(timeout=2)

        assert new_holder_cancel_was_set == [True]
        session.stop()

    def test_exit_releases_lock(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path, _make_session_proc([]))
        session.__enter__()
        session.__exit__(None, None, None)
        assert session._lock.acquire(blocking=False)
        session._lock.release()
        session.stop()

    def test_owner_none_when_lock_free(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path, _make_session_proc([]))
        assert session.owner is None
        session.stop()

    def test_owner_set_to_thread_name_while_held(self, tmp_path: Path) -> None:
        import threading as _threading

        session = _make_session(tmp_path, _make_session_proc([]))
        with session:
            assert session.owner == _threading.current_thread().name
        session.stop()

    def test_owner_cleared_after_exit(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path, _make_session_proc([]))
        with session:
            pass
        assert session.owner is None
        session.stop()

    def test_last_turn_cancelled_false_initially(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path, _make_session_proc([]))
        assert session.last_turn_cancelled is False
        session.stop()

    def test_prompt_routes_through_session(self, tmp_path: Path) -> None:
        """ClaudeSession.prompt cancels, takes the lock, sends, and returns result."""
        from kennel.claude import ClaudeSession

        system_file = tmp_path / "system.md"
        system_file.write_text("persona")
        proc = _make_session_proc(
            [
                '{"type":"result","result":""}\n',  # drain after interrupt
                '{"type":"result","result":""}\n',  # /model ack
                '{"type":"result","result":"hello world"}\n',  # actual turn
            ]
        )
        proc.pid = 11111
        fake_popen = MagicMock(return_value=proc)
        fake_selector = MagicMock(return_value=([proc.stdout], [], []))
        session = ClaudeSession(
            system_file,
            work_dir=tmp_path,
            popen=fake_popen,
            selector=fake_selector,
            repo_name="owner/repo",
        )
        try:
            result = session.prompt("hi there", model="claude-opus-4-6")
            assert result == "hello world"
            # send() was called: /model line + main content.
            sent = [call.args[0] for call in proc.stdin.write.call_args_list]
            combined = "".join(sent)
            assert "/model claude-opus-4-6" in combined
            assert "hi there" in combined
        finally:
            session.stop()

    def test_prompt_prepends_system_prompt_to_body(self, tmp_path: Path) -> None:
        from kennel.claude import ClaudeSession

        system_file = tmp_path / "system.md"
        system_file.write_text("persona")
        proc = _make_session_proc(
            [
                '{"type":"result","result":""}\n',
                '{"type":"result","result":"ok"}\n',
            ]
        )
        proc.pid = 22222
        fake_popen = MagicMock(return_value=proc)
        fake_selector = MagicMock(return_value=([proc.stdout], [], []))
        session = ClaudeSession(
            system_file,
            work_dir=tmp_path,
            popen=fake_popen,
            selector=fake_selector,
            repo_name="owner/repo",
        )
        try:
            session.prompt("the question", system_prompt="extra instructions")
            sent = "".join(call.args[0] for call in proc.stdin.write.call_args_list)
            assert "extra instructions\\n\\n---\\n\\nthe question" in sent
        finally:
            session.stop()

    def test_pid_property_returns_popen_pid(self, tmp_path: Path) -> None:
        proc = _make_session_proc([])
        proc.pid = 424242
        session = _make_session(tmp_path, proc)
        assert session.pid == 424242
        session.stop()

    def test_repo_name_exposed(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path, _make_session_proc([]))
        assert session.repo_name == "owner/repo"
        session.stop()

    def test_owner_is_none_without_repo_name(self, tmp_path: Path) -> None:
        """Session without repo_name never registers a talker → owner None."""
        from kennel.claude import ClaudeSession

        system_file = tmp_path / "system.md"
        system_file.write_text("you are fido")
        proc = _make_session_proc([])
        fake_popen = MagicMock(return_value=proc)
        session = ClaudeSession(
            system_file,
            work_dir=tmp_path,
            popen=fake_popen,
            selector=MagicMock(return_value=([proc.stdout], [], [])),
        )
        with session:
            assert session.owner is None
        session.stop()

    def test_enter_raises_on_concurrent_talker_and_releases_lock(
        self, tmp_path: Path
    ) -> None:
        """__enter__ raises ClaudeLeakError if another talker is registered and
        releases the session lock so callers don't deadlock."""
        from datetime import datetime, timezone

        from kennel import claude as claude_module
        from kennel.claude import ClaudeLeakError, ClaudeTalker, register_talker

        session = _make_session(tmp_path, _make_session_proc([]))
        register_talker(
            ClaudeTalker(
                repo_name="owner/repo",
                thread_id=999,
                kind="webhook",
                description="leaked",
                claude_pid=555,
                started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
        )
        with pytest.raises(ClaudeLeakError):
            session.__enter__()
        # Session lock must be released so we don't deadlock future callers.
        assert session._lock.acquire(blocking=False)
        session._lock.release()
        with claude_module._talkers_lock:
            claude_module._talkers.clear()
        session.stop()

    def test_context_manager_blocks_second_thread(self, tmp_path: Path) -> None:
        import threading as _threading

        session = _make_session(tmp_path, _make_session_proc([]))
        entered = _threading.Event()
        done_checking = _threading.Event()
        could_not_acquire = _threading.Event()

        def first_thread() -> None:
            with session:
                entered.set()
                done_checking.wait()  # hold the lock until t2 is done checking

        def second_thread() -> None:
            entered.wait()  # t1 holds the lock and is blocked on done_checking
            acquired = session._lock.acquire(blocking=False)
            if not acquired:
                could_not_acquire.set()
            else:
                session._lock.release()
            done_checking.set()  # let t1 exit the context manager

        t1 = _threading.Thread(target=first_thread)
        t2 = _threading.Thread(target=second_thread)
        t1.start()
        t2.start()
        t1.join(timeout=2)
        t2.join(timeout=2)
        assert could_not_acquire.is_set()
        session.stop()


class TestClaudeSessionSendControlInterrupt:
    def test_writes_control_request_type(self, tmp_path: Path) -> None:
        import json as _json

        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        session._send_control_interrupt()
        written = proc.stdin.write.call_args.args[0]
        assert _json.loads(written.strip())["type"] == "control_request"
        session.stop()

    def test_writes_interrupt_subtype(self, tmp_path: Path) -> None:
        import json as _json

        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        session._send_control_interrupt()
        obj = _json.loads(proc.stdin.write.call_args.args[0].strip())
        assert obj["request"]["subtype"] == "interrupt"
        session.stop()

    def test_includes_request_id(self, tmp_path: Path) -> None:
        import json as _json

        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        session._send_control_interrupt()
        obj = _json.loads(proc.stdin.write.call_args.args[0].strip())
        assert isinstance(obj.get("request_id"), str) and obj["request_id"]
        session.stop()


class TestClaudeSessionInterrupt:
    def test_interrupt_sends_control_request_first(self, tmp_path: Path) -> None:
        """interrupt() sends control_request as the first stdin write."""
        import json as _json

        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        session.interrupt("abort now")
        first_write = proc.stdin.write.call_args_list[0].args[0]
        obj = _json.loads(first_write.strip())
        assert obj["type"] == "control_request"
        assert obj["request"]["subtype"] == "interrupt"
        session.stop()

    def test_interrupt_sends_follow_up_as_user_message(self, tmp_path: Path) -> None:
        """interrupt() sends follow-up content as the last stdin write."""
        import json as _json

        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        session.interrupt("abort now")
        last_write = proc.stdin.write.call_args.args[0]
        obj = _json.loads(last_write.strip())
        assert obj["type"] == "user"
        assert obj["message"]["content"] == "abort now"
        session.stop()

    def test_interrupt_drains_turn_before_follow_up(self, tmp_path: Path) -> None:
        """interrupt() reads all old-turn events before sending the follow-up."""
        import json as _json

        result_line = _json.dumps({"type": "result", "result": "done"})
        proc = _make_session_proc([result_line])
        session = _make_session(tmp_path, proc)
        session.interrupt("next")
        # stdout was read (drain happened — readline called at least once)
        assert proc.stdout.readline.call_count >= 1
        # and follow-up was still sent
        last_write = proc.stdin.write.call_args.args[0]
        obj = _json.loads(last_write.strip())
        assert obj["type"] == "user"
        assert obj["message"]["content"] == "next"
        session.stop()

    def test_interrupt_sets_cancel_while_waiting_for_lock(self, tmp_path: Path) -> None:
        """_cancel is set before interrupt() acquires the lock."""
        import threading as _threading
        import time as _time

        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        lock_held = _threading.Event()
        release = _threading.Event()
        cancel_seen: list[bool] = []

        def holder() -> None:
            session._lock.acquire()
            lock_held.set()
            release.wait()
            cancel_seen.append(session._cancel.is_set())
            session._lock.release()

        t_holder = _threading.Thread(target=holder)
        t_holder.start()
        lock_held.wait()

        t_int = _threading.Thread(target=lambda: session.interrupt("follow"))
        t_int.start()

        # _cancel.set() is the very first thing interrupt() does — 50ms is
        # far more than enough for the thread to reach it before we check.
        _time.sleep(0.05)
        release.set()
        t_holder.join(timeout=2)
        t_int.join(timeout=2)

        assert cancel_seen == [True]
        session.stop()

    def test_interrupt_waits_for_lock_holder_then_sends(self, tmp_path: Path) -> None:
        import threading as _threading

        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        acquired = _threading.Event()
        release = _threading.Event()

        def holder() -> None:
            session._lock.acquire()
            acquired.set()
            release.wait()
            session._lock.release()

        t_holder = _threading.Thread(target=holder)
        t_holder.start()
        acquired.wait()  # holder definitely has the lock

        t_interrupt = _threading.Thread(target=lambda: session.interrupt("msg"))
        t_interrupt.start()

        # interrupt is blocked on lock.acquire(); stdin must not be written yet
        assert proc.stdin.write.call_count == 0

        release.set()
        t_holder.join(timeout=2)
        t_interrupt.join(timeout=2)

        assert proc.stdin.write.call_count > 0
        session.stop()
