from __future__ import annotations

import logging
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kennel.claude import (
    _LOG_LINE_TRUNCATE,
    _RETURNCODE_IDLE_TIMEOUT,
    ClaudeAPI,
    ClaudeClient,
    ClaudeCode,
    ClaudeProviderError,
    ClaudeSession,
    ClaudeStreamError,
    _active_children,
    _claude,
    _default_claude_credentials_path,
    _load_claude_oauth_state,
    _register_child,
    _Trunc,
    _unregister_child,
    _usage_window,
    extract_result_text,
    extract_session_id,
    kill_active_children,
    raise_for_provider_error_output,
)
from kennel.provider import ProviderID, ProviderLimitSnapshot, ProviderLimitWindow


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

    def test_session_for_current_repo_raises_without_repo(self) -> None:
        from kennel import claude as claude_module
        from kennel.claude import _session_for_current_repo

        claude_module.set_thread_repo(None)
        with pytest.raises(RuntimeError, match="thread-local repo_name"):
            _session_for_current_repo()

    def test_session_for_current_repo_raises_without_resolver(self) -> None:
        from kennel import claude as claude_module
        from kennel.claude import _session_for_current_repo

        claude_module.set_session_resolver(None)
        claude_module.set_thread_repo("owner/repo")
        try:
            with pytest.raises(RuntimeError, match="set_session_resolver"):
                _session_for_current_repo()
        finally:
            claude_module.set_thread_repo(None)

    def test_session_for_current_repo_raises_when_no_session(self) -> None:
        from kennel import claude as claude_module
        from kennel.claude import _session_for_current_repo

        claude_module.set_session_resolver(lambda repo: None)
        claude_module.set_thread_repo("owner/repo")
        try:
            with pytest.raises(RuntimeError, match="no ClaudeSession registered"):
                _session_for_current_repo()
        finally:
            claude_module.set_thread_repo(None)
            claude_module.set_session_resolver(None)

    def test_session_for_current_repo_raises_when_not_alive(self) -> None:
        from kennel import claude as claude_module
        from kennel.claude import _session_for_current_repo

        dead = MagicMock()
        dead.is_alive.return_value = False
        claude_module.set_session_resolver(lambda repo: dead)
        claude_module.set_thread_repo("owner/repo")
        try:
            with pytest.raises(RuntimeError, match="not alive"):
                _session_for_current_repo()
        finally:
            claude_module.set_thread_repo(None)
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


class TestClaudeSessionWakeupPipe:
    def test_wake_writes_byte(self, tmp_path: Path) -> None:
        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        session._wake()
        data = os.read(session._wakeup_r, 16)
        assert data == b"\x00"
        session.stop()

    def test_wake_tolerates_closed_pipe(self, tmp_path: Path) -> None:
        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        os.close(session._wakeup_w)
        session._wake()  # should not raise
        session.stop()

    def test_drain_wakeup_clears_pipe(self, tmp_path: Path) -> None:
        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        os.write(session._wakeup_w, b"\x00\x00\x00")
        session._drain_wakeup()
        # Pipe should be empty — non-blocking read raises BlockingIOError
        with pytest.raises(BlockingIOError):
            os.read(session._wakeup_r, 1)
        session.stop()

    def test_drain_wakeup_tolerates_closed_pipe(self, tmp_path: Path) -> None:
        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        os.close(session._wakeup_r)
        session._drain_wakeup()  # should not raise
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

    def test_marks_in_turn_after_send(self, tmp_path: Path) -> None:
        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        session.send("hi")
        assert session._in_turn is True

    def test_drains_stale_turn_before_sending_new(self, tmp_path: Path) -> None:
        """Send() must drain any unfinished prior turn so the next
        consume_until_result doesn't read stale events as its own (#499)."""
        import json as _json

        stale_result = (
            _json.dumps({"type": "result", "result": "stale", "session_id": "s1"})
            + "\n"
        )
        proc = _make_session_proc([stale_result])
        session = _make_session(tmp_path, proc)
        session._in_turn = True  # simulate cancelled-prior-turn state
        session.send("fresh message")
        assert session._in_turn is True
        # control_request was written before the new user message
        writes = [c.args[0] for c in proc.stdin.write.call_args_list]
        assert any("control_request" in w for w in writes)
        assert any('"fresh message"' in w for w in writes)
        # The control_request must come first
        control_idx = next(i for i, w in enumerate(writes) if "control_request" in w)
        user_idx = next(i for i, w in enumerate(writes) if '"fresh message"' in w)
        assert control_idx < user_idx


class TestClaudeSessionDrainToBoundary:
    def test_returns_early_when_proc_dead(self, tmp_path: Path) -> None:
        proc = _make_session_proc([], poll_returns=0)
        session = _make_session(tmp_path, proc)
        session._in_turn = True
        session._drain_to_boundary()
        assert session._in_turn is False
        # No control_request sent to a dead process
        proc.stdin.write.assert_not_called()

    def test_returns_on_control_request_failure(self, tmp_path: Path) -> None:
        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        session._in_turn = True
        proc.stdin.write.side_effect = BrokenPipeError("pipe closed")
        session._drain_to_boundary()
        assert session._in_turn is False

    def test_reads_until_type_result(self, tmp_path: Path) -> None:
        import json as _json

        lines = [
            _json.dumps({"type": "assistant", "text": "thinking"}) + "\n",
            _json.dumps({"type": "result", "result": "abandoned", "session_id": "s9"})
            + "\n",
        ]
        proc = _make_session_proc(lines)
        session = _make_session(tmp_path, proc)
        session._in_turn = True
        session._drain_to_boundary()
        assert session._in_turn is False
        assert session._session_id == "s9"

    def test_reads_until_type_error(self, tmp_path: Path) -> None:
        import json as _json

        lines = [_json.dumps({"type": "error", "error": "boom"}) + "\n"]
        proc = _make_session_proc(lines)
        session = _make_session(tmp_path, proc)
        session._in_turn = True
        session._drain_to_boundary()
        assert session._in_turn is False

    def test_skips_blank_and_invalid_json(self, tmp_path: Path) -> None:
        import json as _json

        lines = [
            "\n",
            "not-json-at-all\n",
            _json.dumps({"type": "result", "result": "ok"}) + "\n",
        ]
        proc = _make_session_proc(lines)
        session = _make_session(tmp_path, proc)
        session._in_turn = True
        session._drain_to_boundary()
        assert session._in_turn is False

    def test_breaks_on_eof(self, tmp_path: Path) -> None:
        proc = _make_session_proc([])  # readline returns "" (EOF)
        session = _make_session(tmp_path, proc)
        session._in_turn = True
        session._drain_to_boundary()
        # EOF path restarts the session at the deadline check, but the
        # initial readline returning "" just exits the drain loop.
        # _in_turn gets cleared by the restart fallback path.

    def test_breaks_when_proc_exits_mid_drain(self, tmp_path: Path) -> None:
        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        session._in_turn = True
        # selector reports no pending data; proc.poll() says alive initially
        # (for the front-door check) then exited once drain loop polls.
        session._selector = MagicMock(return_value=([], [], []))
        poll_results = iter([None] + [0] * 10)
        proc.poll = MagicMock(side_effect=lambda: next(poll_results))
        session._drain_to_boundary(deadline=1.0)
        # Loop exits on proc.poll() == 0 (EOF). _in_turn stays True because
        # we only clear it on type=result/error — restart path would clear
        # it, but EOF-only exit doesn't.

    def test_restarts_on_deadline(self, tmp_path: Path) -> None:
        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        session._in_turn = True
        # No pending data, process stays alive → loop just times out
        session._selector = MagicMock(return_value=([], [], []))
        with patch.object(session, "restart") as mock_restart:
            session._drain_to_boundary(deadline=0.01)
        mock_restart.assert_called_once()
        assert session._in_turn is False

    def test_select_includes_wakeup_pipe(self, tmp_path: Path) -> None:
        import json as _json

        lines = [_json.dumps({"type": "result", "result": "done"}) + "\n"]
        proc = _make_session_proc(lines)
        select_inputs: list[list[object]] = []

        def tracking_selector(
            rlist: list[object],
            wlist: list[object],
            xlist: list[object],
            timeout: float,
        ) -> tuple[list[object], list[object], list[object]]:
            select_inputs.append(list(rlist))
            return ([proc.stdout], [], [])

        system_file = tmp_path / "system.md"
        system_file.write_text("sys")
        session = ClaudeSession(
            system_file, popen=MagicMock(return_value=proc), selector=tracking_selector
        )
        session._in_turn = True
        session._drain_to_boundary()
        assert all(session._wakeup_r in inputs for inputs in select_inputs)
        session.stop()


class TestClaudeSessionLogEvent:
    def test_assistant_text(self, tmp_path: Path, caplog) -> None:
        import logging as _l

        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        with caplog.at_level(_l.INFO, logger="kennel"):
            session._log_event(
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "text", "text": "thinking hard"}]},
                }
            )
        assert "claude>" in caplog.text and "thinking hard" in caplog.text

    def test_tool_use_command(self, tmp_path: Path, caplog) -> None:
        import logging as _l

        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        with caplog.at_level(_l.INFO, logger="kennel"):
            session._log_event(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "Bash",
                                "input": {"command": "ls -la"},
                            }
                        ]
                    },
                }
            )
        assert "claude tool: Bash" in caplog.text and "ls -la" in caplog.text

    def test_tool_use_file_path(self, tmp_path: Path, caplog) -> None:
        import logging as _l

        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        with caplog.at_level(_l.INFO, logger="kennel"):
            session._log_event(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "Read",
                                "input": {"file_path": "/tmp/foo.py"},
                            }
                        ]
                    },
                }
            )
        assert "claude tool: Read" in caplog.text and "/tmp/foo.py" in caplog.text

    def test_tool_use_fallback_first_value(self, tmp_path: Path, caplog) -> None:
        import logging as _l

        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        with caplog.at_level(_l.INFO, logger="kennel"):
            session._log_event(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "Thing",
                                "input": {"other": "value-xyz"},
                            }
                        ]
                    },
                }
            )
        assert "claude tool: Thing" in caplog.text and "value-xyz" in caplog.text

    def test_content_non_dict_skipped(self, tmp_path: Path) -> None:
        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        # Must not raise
        session._log_event(
            {"type": "assistant", "message": {"content": ["not a dict"]}}
        )

    def test_user_tool_result(self, tmp_path: Path, caplog) -> None:
        import logging as _l

        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        with caplog.at_level(_l.INFO, logger="kennel"):
            session._log_event(
                {
                    "type": "user",
                    "message": {
                        "content": [{"type": "tool_result", "content": "abcdefghij"}]
                    },
                }
            )
        assert "claude tool result" in caplog.text

    def test_system_event(self, tmp_path: Path, caplog) -> None:
        import logging as _l

        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        with caplog.at_level(_l.INFO, logger="kennel"):
            session._log_event({"type": "system", "subtype": "init"})
        assert "claude system: init" in caplog.text

    def test_result_event(self, tmp_path: Path, caplog) -> None:
        import logging as _l

        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        with caplog.at_level(_l.INFO, logger="kennel"):
            session._log_event({"type": "result", "result": "all done"})
        assert "claude result: all done" in caplog.text

    def test_error_event(self, tmp_path: Path, caplog) -> None:
        import logging as _l

        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        with caplog.at_level(_l.WARNING, logger="kennel"):
            session._log_event({"type": "error", "error": "kaboom"})
        assert "claude error: kaboom" in caplog.text


class TestClaudeSessionWaitForPendingPreempt:
    def test_returns_false_when_not_pending(self, tmp_path: Path) -> None:
        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        # _preempt_pending starts cleared
        assert session.wait_for_pending_preempt(timeout=0.01) is False

    def test_returns_true_when_pending_clears(self, tmp_path: Path) -> None:
        import threading as _t

        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        session._preempt_pending.set()

        # Clear the event from another thread after a brief delay.
        def _clearer() -> None:
            import time as _time

            _time.sleep(0.02)
            session._preempt_pending.clear()

        t = _t.Thread(target=_clearer)
        t.start()
        assert session.wait_for_pending_preempt(timeout=1.0) is True
        t.join()

    def test_returns_false_on_timeout(self, tmp_path: Path) -> None:
        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)
        session._preempt_pending.set()
        # Never cleared → waits out the deadline.
        assert session.wait_for_pending_preempt(timeout=0.05) is False


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

    def test_select_includes_wakeup_pipe(self, tmp_path: Path) -> None:
        import json as _json

        lines = [_json.dumps({"type": "result", "result": "ok"}) + "\n"]
        proc = _make_session_proc(lines)
        select_inputs: list[list[object]] = []

        def tracking_selector(
            rlist: list[object],
            wlist: list[object],
            xlist: list[object],
            timeout: float,
        ) -> tuple[list[object], list[object], list[object]]:
            select_inputs.append(list(rlist))
            return ([proc.stdout], [], [])

        system_file = tmp_path / "system.md"
        system_file.write_text("sys")
        session = ClaudeSession(
            system_file, popen=MagicMock(return_value=proc), selector=tracking_selector
        )
        list(session.iter_events())
        # Every select call must include the wakeup fd alongside stdout
        assert all(session._wakeup_r in inputs for inputs in select_inputs)
        session.stop()

    def test_wakeup_only_ready_continues_to_cancel_check(self, tmp_path: Path) -> None:
        """When only the wakeup pipe fires, iter_events loops back and checks cancel."""
        system_file = tmp_path / "system.md"
        system_file.write_text("sys")
        proc = _make_session_proc([])
        proc.poll = MagicMock(return_value=None)  # never exits
        fake_popen = MagicMock(return_value=proc)

        session_ref: list[ClaudeSession] = []
        call_count = [0]

        def staged_selector(
            rlist: list[object],
            wlist: list[object],
            xlist: list[object],
            timeout: float,
        ) -> tuple[list[object], list[object], list[object]]:
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: only wakeup pipe ready, then set cancel so next
                # iteration exits cleanly
                wakeup_r = next(x for x in rlist if x != proc.stdout)
                session_ref[0]._cancel.set()
                return ([wakeup_r], [], [])
            return ([], [], [])  # should not reach here

        session = ClaudeSession(system_file, popen=fake_popen, selector=staged_selector)
        session_ref.append(session)
        os.write(session._wakeup_w, b"\x00")
        events = list(session.iter_events())
        assert events == []
        assert session._last_turn_cancelled is True
        # readline was never called because stdout was not in the ready list
        assert proc.stdout.readline.call_count == 0
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
    def test_same_model_is_noop(self, tmp_path: Path) -> None:
        """When the target matches the current model, nothing happens."""
        proc = _make_session_proc([])
        session = _make_session(tmp_path, proc)  # default model claude-opus-4-6
        current_proc = session._proc
        session.switch_model("claude-opus-4-6")
        assert session._proc is current_proc
        # stdin.write should NOT have been called for a /model slash command.
        assert proc.stdin.write.call_count == 0

    def test_different_model_respawns_with_new_flag(self, tmp_path: Path) -> None:
        """Switching model kills the old proc and spawns a new one with
        --model <new>, passing --resume when session_id is known."""
        system_file = tmp_path / "system.md"
        system_file.write_text("sys")
        old_proc = _make_session_proc([])
        old_proc.pid = 1001
        new_proc = _make_session_proc([])
        new_proc.pid = 1002
        fake_popen = MagicMock(side_effect=[old_proc, new_proc])
        session = ClaudeSession(
            system_file,
            work_dir=tmp_path,
            popen=fake_popen,
            selector=MagicMock(return_value=([], [], [])),
            model="claude-opus-4-6",
            repo_name="owner/repo",
        )
        # Prior turn established a session_id — switch_model must preserve
        # conversation by passing --resume to the new subprocess.
        session._session_id = "sid-123"
        session.switch_model("claude-sonnet-4-6")
        old_proc.kill.assert_called_once()
        assert session._proc is new_proc
        assert session._model == "claude-sonnet-4-6"
        # Second spawn call had --model claude-sonnet-4-6 and --resume sid-123.
        second_cmd = fake_popen.call_args_list[1].args[0]
        assert "--model" in second_cmd
        assert second_cmd[second_cmd.index("--model") + 1] == "claude-sonnet-4-6"
        assert "--resume" in second_cmd
        assert second_cmd[second_cmd.index("--resume") + 1] == "sid-123"

    def test_switch_raises_when_kill_fails(self, tmp_path: Path) -> None:
        """kill/wait failure during switch_model re-raises so the caller
        can decide how to recover."""
        import subprocess

        system_file = tmp_path / "system.md"
        system_file.write_text("sys")
        old_proc = _make_session_proc([], poll_returns=None)
        old_proc.kill = MagicMock()
        old_proc.wait = MagicMock(side_effect=subprocess.TimeoutExpired("claude", 1.0))
        fake_popen = MagicMock(side_effect=[old_proc, _make_session_proc([])])
        session = ClaudeSession(
            system_file,
            work_dir=tmp_path,
            popen=fake_popen,
            selector=MagicMock(return_value=([], [], [])),
            model="claude-opus-4-6",
        )
        with pytest.raises(subprocess.TimeoutExpired):
            session.switch_model("claude-sonnet-4-6")

    def test_switch_with_no_prior_session_id_omits_resume(self, tmp_path: Path) -> None:
        system_file = tmp_path / "system.md"
        system_file.write_text("sys")
        old_proc = _make_session_proc([])
        new_proc = _make_session_proc([])
        fake_popen = MagicMock(side_effect=[old_proc, new_proc])
        session = ClaudeSession(
            system_file,
            work_dir=tmp_path,
            popen=fake_popen,
            selector=MagicMock(return_value=([], [], [])),
            model="claude-opus-4-6",
        )
        # No prior session_id (fresh session) — no --resume flag.
        session.switch_model("claude-sonnet-4-6")
        second_cmd = fake_popen.call_args_list[1].args[0]
        assert "--resume" not in second_cmd


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
        with patch.object(session, "restart") as mock_restart:
            with pytest.raises(ClaudeProviderError, match="something broke"):
                session.consume_until_result()
        mock_restart.assert_called_once_with()

    def test_raises_on_provider_error_result(self, tmp_path: Path) -> None:
        import json as _json

        lines = [
            _json.dumps(
                {
                    "type": "result",
                    "result": 'API Error: 500 {"type":"error","message":"Internal server error"}',
                }
            )
            + "\n"
        ]
        proc = _make_session_proc(lines)
        session = _make_session(tmp_path, proc)
        with patch.object(session, "restart") as mock_restart:
            with pytest.raises(ClaudeProviderError, match="500"):
                session.consume_until_result()
        mock_restart.assert_called_once_with()

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

    def test_restart_logs_info(self, tmp_path: Path, caplog) -> None:
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
        with caplog.at_level(_logging.INFO, logger="kennel.claude"):
            session.restart()
        assert any("restart" in r.message.lower() for r in caplog.records)
        session.stop()

    def test_restart_clears_session_id(self, tmp_path: Path) -> None:
        """restart drops session_id so the new spawn starts a fresh
        conversation (opposite of switch_model which preserves it)."""
        system_file = tmp_path / "system.md"
        system_file.write_text("sys")
        old_proc = _make_session_proc([])
        new_proc = _make_session_proc([])
        fake_popen = MagicMock(side_effect=[old_proc, new_proc])
        session = ClaudeSession(
            system_file,
            work_dir=tmp_path,
            popen=fake_popen,
            selector=MagicMock(return_value=([], [], [])),
        )
        session._session_id = "sid-123"
        session.restart()
        assert session._session_id == ""
        # Second spawn call had no --resume.
        second_cmd = fake_popen.call_args_list[1].args[0]
        assert "--resume" not in second_cmd

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
            model="claude-opus-4-6",
        )
        try:
            result = session.prompt("hi there", model="claude-opus-4-6")
            assert result == "hello world"
            # model param matches current → switch_model is a no-op;
            # stdin only carries the user message body.
            sent = "".join(call.args[0] for call in proc.stdin.write.call_args_list)
            assert "hi there" in sent
            assert "/model" not in sent
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


class TestClaudeSessionPreemptLatency:
    """Validate that webhook preemption completes well within the 30s latency target.

    The worker holds the session lock and blocks in iter_events (real select on
    a pipe). A preempter sets cancel and writes to the wakeup pipe. The worker's
    select must wake instantly, detect cancel, and release the lock — all well
    under the 30s budget from #559.
    """

    def test_preempt_handoff_under_worker_contention(self, tmp_path: Path) -> None:
        """Worker releases lock within 2s of a cancel+wake signal (target: <30s)."""
        import select as _select
        import threading
        import time

        stdout_r, stdout_w = os.pipe()
        os.set_blocking(stdout_r, False)

        proc = MagicMock(spec=subprocess.Popen)
        proc.pid = 99999
        proc.stdin = MagicMock()
        proc.stdin.closed = False
        proc.stdout = os.fdopen(stdout_r, "r")
        proc.stderr = MagicMock()
        proc.poll = MagicMock(return_value=None)  # never exits
        proc.wait = MagicMock(return_value=0)
        proc.returncode = 0

        system_file = tmp_path / "system.md"
        system_file.write_text("sys")

        session = ClaudeSession(
            system_file,
            work_dir=tmp_path,
            popen=MagicMock(return_value=proc),
            selector=_select.select,  # real select — wakeup pipe must work
            repo_name="test/latency",
            idle_timeout=30.0,
        )

        worker_in_select = threading.Event()
        worker_exited_lock = threading.Event()

        # Wrap selector so we know when the worker is blocking in select
        real_selector = session._selector

        def instrumented_selector(
            rlist: list[object],
            wlist: list[object],
            xlist: list[object],
            timeout: float,
        ) -> tuple[list[object], list[object], list[object]]:
            worker_in_select.set()
            return real_selector(rlist, wlist, xlist, timeout)

        session._selector = instrumented_selector

        def worker() -> None:
            with session:
                for _ in session.iter_events():
                    pass  # no events expected — just blocking on select
            worker_exited_lock.set()

        t = threading.Thread(target=worker)
        t.start()

        # Wait for worker to actually be blocking in select
        assert worker_in_select.wait(timeout=2.0), "worker never entered select"

        # Now preempt — this is the critical path we're measuring
        start = time.monotonic()
        session._cancel.set()
        session._wake()

        assert worker_exited_lock.wait(timeout=5.0), "worker did not release lock"
        elapsed = time.monotonic() - start

        # The 30s budget from #559 includes the actual Claude response; the
        # lock-handoff portion must be negligible.  Assert < 2s — generous
        # safety margin while still catching the old 10s+ poll-interval bug.
        assert elapsed < 2.0, (
            f"preempt handoff took {elapsed:.2f}s — must be well under 30s target"
        )

        t.join(timeout=2.0)
        os.close(stdout_w)
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


# ── ClaudeClient ─────────────────────────────────────────────────────────────


class TestClaudeClientPrintPrompt:
    def test_delegates_to_session_prompt(self) -> None:
        session = MagicMock()
        session.prompt.return_value = "hello"
        client = ClaudeClient(session_fn=lambda: session)
        assert client.print_prompt("hi", "claude-opus-4-6") == "hello"
        session.prompt.assert_called_once_with(
            "hi", model="claude-opus-4-6", system_prompt=None
        )

    def test_passes_system_prompt(self) -> None:
        session = MagicMock()
        session.prompt.return_value = "ok"
        client = ClaudeClient(session_fn=lambda: session)
        client.print_prompt("q", "claude-opus-4-6", system_prompt="be fido")
        assert session.prompt.call_args.kwargs["system_prompt"] == "be fido"

    def test_session_errors_propagate(self) -> None:
        session = MagicMock()
        session.prompt.side_effect = ClaudeStreamError(1)
        client = ClaudeClient(session_fn=lambda: session)
        with pytest.raises(ClaudeStreamError):
            client.print_prompt("q", "claude-opus-4-6")

    def test_uses_attached_session_before_resolver(self) -> None:
        attached = MagicMock()
        attached.prompt.return_value = "bound"
        resolver = MagicMock()
        client = ClaudeClient(session_fn=resolver, session=attached)
        assert client.print_prompt("hi", "claude-opus-4-6") == "bound"
        resolver.assert_not_called()


class TestClaudeClientSessionAttachment:
    def test_attach_session_updates_bound_session(self) -> None:
        attached = MagicMock()
        client = ClaudeClient()
        client.attach_session(attached)
        assert client.session is attached

    def test_detach_session_returns_and_clears_bound_session(self) -> None:
        attached = MagicMock()
        client = ClaudeClient(session=attached)
        assert client.detach_session() is attached
        assert client.session is None

    def test_session_owner_reads_from_bound_session(self) -> None:
        attached = MagicMock(owner="worker-home")
        client = ClaudeClient(session=attached)
        assert client.session_owner == "worker-home"

    def test_session_alive_reads_from_bound_session(self) -> None:
        attached = MagicMock()
        attached.is_alive.return_value = True
        client = ClaudeClient(session=attached)
        assert client.session_alive is True

    def test_session_pid_reads_from_bound_session(self) -> None:
        attached = MagicMock(pid=1234)
        client = ClaudeClient(session=attached)
        assert client.session_pid == 1234

    def test_provider_id_is_claude_code(self) -> None:
        client = ClaudeClient()
        assert str(client.provider_id) == "claude-code"


class TestClaudeOAuthState:
    def test_default_credentials_path_points_at_claude_dir(self) -> None:
        path = _default_claude_credentials_path()
        assert path.name == ".credentials.json"
        assert path.parent.name == ".claude"

    def test_load_oauth_state_returns_none_when_file_missing(
        self, tmp_path: Path
    ) -> None:
        assert _load_claude_oauth_state(tmp_path / "missing.json") is None

    def test_load_oauth_state_reads_access_token(self, tmp_path: Path) -> None:
        path = tmp_path / "creds.json"
        path.write_text('{"claudeAiOauth": {"accessToken": "tok-123"}}')
        state = _load_claude_oauth_state(path)
        assert state is not None
        assert state.access_token == "tok-123"

    def test_load_oauth_state_rejects_blank_access_token(self, tmp_path: Path) -> None:
        path = tmp_path / "creds.json"
        path.write_text('{"claudeAiOauth": {"accessToken": ""}}')
        with pytest.raises(ValueError, match="missing accessToken"):
            _load_claude_oauth_state(path)


class TestClaudeUsageWindow:
    def test_returns_none_for_missing_window(self) -> None:
        assert _usage_window("five_hour", None) is None

    def test_returns_none_for_null_utilization(self) -> None:
        assert (
            _usage_window(
                "five_hour",
                {"utilization": None, "resets_at": "2026-04-16T07:00:00+00:00"},
            )
            is None
        )

    def test_rejects_non_object_window(self) -> None:
        with pytest.raises(ValueError, match="must be an object or null"):
            _usage_window("five_hour", "nope")

    def test_rejects_non_numeric_utilization(self) -> None:
        with pytest.raises(ValueError, match="utilization must be numeric or null"):
            _usage_window("five_hour", {"utilization": "bad", "resets_at": None})

    def test_rejects_non_string_reset(self) -> None:
        with pytest.raises(ValueError, match="reset time must be a string or null"):
            _usage_window("five_hour", {"utilization": 37.2, "resets_at": 123})

    def test_parses_window(self) -> None:
        assert _usage_window(
            "five_hour",
            {
                "utilization": 37.2,
                "resets_at": "2026-04-16T07:00:00+00:00",
            },
        ) == ProviderLimitWindow(
            name="five_hour",
            used=37,
            limit=100,
            resets_at=datetime(2026, 4, 16, 7, 0, tzinfo=UTC),
            unit="%",
        )

    def test_parses_window_with_null_reset(self) -> None:
        assert _usage_window(
            "seven_day",
            {
                "utilization": 12.5,
                "resets_at": None,
            },
        ) == ProviderLimitWindow(
            name="seven_day",
            used=12,
            limit=100,
            resets_at=None,
            unit="%",
        )


class TestClaudeAPI:
    def test_provider_id_is_claude_code(self) -> None:
        assert (
            ClaudeAPI(oauth_state_fn=lambda: None).provider_id == ProviderID.CLAUDE_CODE
        )

    def test_limit_snapshot_marks_logged_out_accounts_unavailable(self) -> None:
        api = ClaudeAPI(oauth_state_fn=lambda: None)
        assert api.get_limit_snapshot() == ProviderLimitSnapshot(
            provider=ProviderID.CLAUDE_CODE,
            unavailable_reason="Claude Code is not logged in.",
        )

    def test_limit_snapshot_parses_usage_windows(self) -> None:
        response = MagicMock()
        response.json.return_value = {
            "five_hour": {
                "utilization": 37.0,
                "resets_at": "2026-04-16T07:00:00+00:00",
            },
            "seven_day": {
                "utilization": 98.0,
                "resets_at": "2026-04-20T12:00:00+00:00",
            },
            "seven_day_sonnet": {
                "utilization": 65.0,
                "resets_at": "2026-04-20T13:00:00+00:00",
            },
            "extra_usage": {
                "is_enabled": True,
                "monthly_limit": 2000,
                "used_credits": 2163.0,
                "utilization": 100.0,
            },
        }
        session = MagicMock()
        session.get.return_value = response
        api = ClaudeAPI(
            session=session,
            oauth_state_fn=lambda: type(
                "OAuthState", (), {"access_token": "tok-123"}
            )(),
        )
        assert api.get_limit_snapshot() == ProviderLimitSnapshot(
            provider=ProviderID.CLAUDE_CODE,
            windows=(
                ProviderLimitWindow(
                    name="five_hour",
                    used=37,
                    limit=100,
                    resets_at=datetime(2026, 4, 16, 7, 0, tzinfo=UTC),
                    unit="%",
                ),
                ProviderLimitWindow(
                    name="seven_day",
                    used=98,
                    limit=100,
                    resets_at=datetime(2026, 4, 20, 12, 0, tzinfo=UTC),
                    unit="%",
                ),
                ProviderLimitWindow(
                    name="seven_day_sonnet",
                    used=65,
                    limit=100,
                    resets_at=datetime(2026, 4, 20, 13, 0, tzinfo=UTC),
                    unit="%",
                ),
            ),
        )
        session.get.assert_called_once_with(
            "https://api.anthropic.com/api/oauth/usage",
            headers={
                "Authorization": "Bearer tok-123",
                "anthropic-beta": "oauth-2025-04-20",
                "Content-Type": "application/json",
                "User-Agent": "claude-code/2.1.110",
            },
            timeout=20,
        )

    def test_limit_snapshot_marks_non_subscription_accounts_unavailable(self) -> None:
        response = MagicMock()
        response.json.return_value = {
            "five_hour": None,
            "seven_day": None,
            "seven_day_sonnet": None,
        }
        session = MagicMock()
        session.get.return_value = response
        api = ClaudeAPI(
            session=session,
            oauth_state_fn=lambda: type(
                "OAuthState", (), {"access_token": "tok-123"}
            )(),
        )
        assert api.get_limit_snapshot() == ProviderLimitSnapshot(
            provider=ProviderID.CLAUDE_CODE,
            unavailable_reason="Claude usage is only available for subscription plans.",
        )

    def test_limit_snapshot_logs_and_raises_when_fetch_fails(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        response = MagicMock()
        response.raise_for_status.side_effect = RuntimeError("boom")
        session = MagicMock()
        session.get.return_value = response
        api = ClaudeAPI(
            session=session,
            oauth_state_fn=lambda: type(
                "OAuthState", (), {"access_token": "tok-123"}
            )(),
        )
        with caplog.at_level(logging.ERROR, logger="kennel.claude"):
            with pytest.raises(RuntimeError, match="boom"):
                api.get_limit_snapshot()
        assert "ClaudeAPI: failed to fetch usage snapshot" in caplog.text

    def test_limit_snapshot_logs_and_raises_when_payload_is_not_an_object(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        response = MagicMock()
        response.json.return_value = []
        session = MagicMock()
        session.get.return_value = response
        api = ClaudeAPI(
            session=session,
            oauth_state_fn=lambda: type(
                "OAuthState", (), {"access_token": "tok-123"}
            )(),
        )
        with caplog.at_level(logging.ERROR, logger="kennel.claude"):
            with pytest.raises(ValueError, match="must be a JSON object"):
                api.get_limit_snapshot()
        assert "ClaudeAPI: failed to fetch usage snapshot" in caplog.text


class TestClaudeCode:
    def test_provider_id_is_claude_code(self) -> None:
        assert ClaudeCode().provider_id == ProviderID.CLAUDE_CODE

    def test_exposes_injected_api_and_agent(self) -> None:
        api = MagicMock()
        agent = MagicMock()
        provider = ClaudeCode(api=api, agent=agent)
        assert provider.api is api
        assert provider.agent is agent

    def test_default_agent_receives_session(self) -> None:
        session = MagicMock()
        provider = ClaudeCode(session=session)
        assert provider.agent.session is session

    def test_attaches_session_to_injected_agent(self) -> None:
        agent = MagicMock()
        session = MagicMock()
        ClaudeCode(agent=agent, session=session)
        agent.attach_session.assert_called_once_with(session)


class TestClaudeClientPrintPromptJson:
    def test_extracts_json_value(self) -> None:
        session = MagicMock()
        session.prompt.return_value = '{"answer": "42"}'
        client = ClaudeClient(session_fn=lambda: session)
        assert client.print_prompt_json("q", "answer", "claude-opus-4-6") == "42"

    def test_returns_empty_on_empty_response(self) -> None:
        session = MagicMock()
        session.prompt.return_value = ""
        client = ClaudeClient(session_fn=lambda: session)
        assert client.print_prompt_json("q", "answer", "claude-opus-4-6") == ""

    def test_returns_empty_on_malformed_json(self) -> None:
        session = MagicMock()
        session.prompt.return_value = "not json"
        client = ClaudeClient(session_fn=lambda: session)
        assert client.print_prompt_json("q", "answer", "claude-opus-4-6") == ""

    def test_scans_for_json_in_preamble(self) -> None:
        session = MagicMock()
        session.prompt.return_value = 'Here is the answer: {"answer": "yes"}'
        client = ClaudeClient(session_fn=lambda: session)
        assert client.print_prompt_json("q", "answer", "claude-opus-4-6") == "yes"

    def test_appends_json_instruction_to_system(self) -> None:
        session = MagicMock()
        session.prompt.return_value = '{"k": "v"}'
        client = ClaudeClient(session_fn=lambda: session)
        client.print_prompt_json("q", "k", "claude-opus-4-6", system_prompt="sys")
        sp = session.prompt.call_args.kwargs["system_prompt"]
        assert sp.startswith("sys\n\n")
        assert '"k"' in sp

    def test_json_instruction_only_when_no_system(self) -> None:
        session = MagicMock()
        session.prompt.return_value = '{"k": "v"}'
        client = ClaudeClient(session_fn=lambda: session)
        client.print_prompt_json("q", "k", "claude-opus-4-6")
        sp = session.prompt.call_args.kwargs["system_prompt"]
        assert "Respond with ONLY" in sp

    def test_returns_empty_when_key_missing(self) -> None:
        session = MagicMock()
        session.prompt.return_value = '{"other": "val"}'
        client = ClaudeClient(session_fn=lambda: session)
        assert client.print_prompt_json("q", "answer", "claude-opus-4-6") == ""

    def test_returns_empty_when_value_not_string(self) -> None:
        session = MagicMock()
        session.prompt.return_value = '{"answer": 42}'
        client = ClaudeClient(session_fn=lambda: session)
        assert client.print_prompt_json("q", "answer", "claude-opus-4-6") == ""


class TestClaudeClientGenerateStatus:
    def test_delegates_to_print_prompt(self) -> None:
        session = MagicMock()
        session.prompt.return_value = "🐶\ncoding up a storm"
        client = ClaudeClient(session_fn=lambda: session)
        assert (
            client.generate_status("working on #42", "be fido")
            == "🐶\ncoding up a storm"
        )
        session.prompt.assert_called_once_with(
            "working on #42", model="claude-opus-4-6", system_prompt="be fido"
        )

    def test_custom_model(self) -> None:
        session = MagicMock()
        session.prompt.return_value = "🐶\nwoof"
        client = ClaudeClient(session_fn=lambda: session)
        client.generate_status("working", "sys", model="claude-sonnet-4-6")
        assert session.prompt.call_args.kwargs["model"] == "claude-sonnet-4-6"


class TestClaudeClientGenerateStatusEmoji:
    def test_delegates_to_print_prompt(self) -> None:
        session = MagicMock()
        session.prompt.return_value = "🐕"
        client = ClaudeClient(session_fn=lambda: session)
        assert client.generate_status_emoji("pick emoji", "be fido") == "🐕"
        session.prompt.assert_called_once_with(
            "pick emoji", model="claude-opus-4-6", system_prompt="be fido"
        )

    def test_custom_model(self) -> None:
        session = MagicMock()
        session.prompt.return_value = "🐕"
        client = ClaudeClient(session_fn=lambda: session)
        client.generate_status_emoji("pick", "sys", model="claude-sonnet-4-6")
        assert session.prompt.call_args.kwargs["model"] == "claude-sonnet-4-6"


class TestClaudeClientPrintPromptFromFile:
    def _files(self, tmp_path: Path) -> tuple[Path, Path]:
        sys = tmp_path / "system.md"
        sys.write_text("sys")
        prompt = tmp_path / "prompt.txt"
        prompt.write_text("p")
        return sys, prompt

    def test_returns_stdout_on_success(self, tmp_path: Path) -> None:
        sys, prompt = self._files(tmp_path)
        mock_stream = MagicMock(return_value=iter(["session output"]))
        client = ClaudeClient(streaming_runner=mock_stream)
        result = client.print_prompt_from_file(sys, prompt, "claude-sonnet-4-6")
        assert result == "session output"

    def test_raises_on_nonzero(self, tmp_path: Path) -> None:
        sys, prompt = self._files(tmp_path)
        mock_stream = MagicMock(side_effect=ClaudeStreamError(1))
        client = ClaudeClient(streaming_runner=mock_stream)
        with pytest.raises(ClaudeStreamError):
            client.print_prompt_from_file(sys, prompt, "claude-sonnet-4-6")

    def test_raises_on_idle_timeout(self, tmp_path: Path) -> None:
        sys, prompt = self._files(tmp_path)
        mock_stream = MagicMock(side_effect=ClaudeStreamError(_RETURNCODE_IDLE_TIMEOUT))
        client = ClaudeClient(streaming_runner=mock_stream)
        with pytest.raises(ClaudeStreamError):
            client.print_prompt_from_file(sys, prompt, "claude-sonnet-4-6")

    def test_raises_on_file_not_found(self, tmp_path: Path) -> None:
        sys, prompt = self._files(tmp_path)
        mock_stream = MagicMock(side_effect=FileNotFoundError)
        client = ClaudeClient(streaming_runner=mock_stream)
        with pytest.raises(FileNotFoundError):
            client.print_prompt_from_file(sys, prompt, "claude-sonnet-4-6")

    def test_passes_correct_cmd(self, tmp_path: Path) -> None:
        sys, prompt = self._files(tmp_path)
        mock_stream = MagicMock(return_value=iter(["out"]))
        client = ClaudeClient(streaming_runner=mock_stream)
        client.print_prompt_from_file(sys, prompt, "claude-sonnet-4-6")
        cmd = mock_stream.call_args[0][0]
        assert "--model" in cmd
        assert "claude-sonnet-4-6" in cmd
        assert "--system-prompt-file" in cmd
        assert str(sys) in cmd
        assert "--print" in cmd

    def test_passes_idle_timeout(self, tmp_path: Path) -> None:
        sys, prompt = self._files(tmp_path)
        mock_stream = MagicMock(return_value=iter(["out"]))
        client = ClaudeClient(streaming_runner=mock_stream)
        client.print_prompt_from_file(
            sys, prompt, "claude-sonnet-4-6", idle_timeout=600.0
        )
        assert mock_stream.call_args[0][2] == 600.0

    def test_passes_cwd(self, tmp_path: Path) -> None:
        sys, prompt = self._files(tmp_path)
        mock_stream = MagicMock(return_value=iter(["out"]))
        client = ClaudeClient(streaming_runner=mock_stream)
        client.print_prompt_from_file(sys, prompt, "claude-sonnet-4-6", cwd="/some/dir")
        assert mock_stream.call_args[1]["cwd"] == "/some/dir"

    def test_raises_on_provider_error_output(self, tmp_path: Path) -> None:
        sys, prompt = self._files(tmp_path)
        mock_stream = MagicMock(
            return_value=iter(
                ['API Error: 500 {"type":"error","message":"Internal server error"}']
            )
        )
        client = ClaudeClient(streaming_runner=mock_stream)
        with pytest.raises(ClaudeProviderError, match="500"):
            client.print_prompt_from_file(sys, prompt, "claude-sonnet-4-6")


class TestRaiseForProviderErrorOutput:
    def test_parses_plain_text_api_error(self) -> None:
        with pytest.raises(ClaudeProviderError) as exc_info:
            raise_for_provider_error_output("API Error: 500 upstream unavailable")
        assert exc_info.value.status_code == 500
        assert exc_info.value.request_id is None
        assert exc_info.value.payload == {}
        assert "upstream unavailable" in str(exc_info.value)

    def test_parses_result_event_with_json_payload(self) -> None:
        import json as _json

        output = "\n" + _json.dumps(
            {
                "type": "result",
                "result": 'API Error: 500 {"error":{"message":"Internal server error"},"request_id":"req_123"}',
            }
        )
        with pytest.raises(ClaudeProviderError) as exc_info:
            raise_for_provider_error_output(output)
        assert exc_info.value.status_code == 500
        assert exc_info.value.request_id == "req_123"
        assert exc_info.value.payload == {
            "error": {"message": "Internal server error"},
            "request_id": "req_123",
        }
        assert "Internal server error" in str(exc_info.value)

    def test_parses_error_event_dict(self) -> None:
        import json as _json

        output = _json.dumps(
            {"type": "error", "error": {"message": "boom"}, "request_id": "req_9"}
        )
        with pytest.raises(ClaudeProviderError) as exc_info:
            raise_for_provider_error_output(output)
        assert exc_info.value.request_id == "req_9"
        assert exc_info.value.payload["type"] == "error"
        assert "boom" in str(exc_info.value)


class TestClaudeClientResumeSession:
    def test_returns_stdout_on_success(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("continue")
        mock_stream = MagicMock(return_value=iter(["resumed output"]))
        client = ClaudeClient(streaming_runner=mock_stream)
        result = client.resume_session("sess-123", prompt_file, "claude-sonnet-4-6")
        assert result == "resumed output"

    def test_raises_on_nonzero(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("continue")
        mock_stream = MagicMock(side_effect=ClaudeStreamError(1))
        client = ClaudeClient(streaming_runner=mock_stream)
        with pytest.raises(ClaudeStreamError):
            client.resume_session("sess-123", prompt_file, "claude-sonnet-4-6")

    def test_passes_correct_cmd(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("continue")
        mock_stream = MagicMock(return_value=iter(["out"]))
        client = ClaudeClient(streaming_runner=mock_stream)
        client.resume_session("sess-123", prompt_file, "claude-sonnet-4-6")
        cmd = mock_stream.call_args[0][0]
        assert "--model" in cmd
        assert "claude-sonnet-4-6" in cmd
        assert "--resume" in cmd
        assert "sess-123" in cmd
        assert "--print" in cmd

    def test_raises_on_provider_error_output(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("continue")
        mock_stream = MagicMock(
            return_value=iter(
                ['API Error: 500 {"type":"error","message":"Internal server error"}']
            )
        )
        client = ClaudeClient(streaming_runner=mock_stream)
        with pytest.raises(ClaudeProviderError, match="500"):
            client.resume_session("sess-123", prompt_file, "claude-sonnet-4-6")

    def test_passes_idle_timeout(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("continue")
        mock_stream = MagicMock(return_value=iter(["out"]))
        client = ClaudeClient(streaming_runner=mock_stream)
        client.resume_session(
            "sess-123", prompt_file, "claude-sonnet-4-6", idle_timeout=600.0
        )
        assert mock_stream.call_args[0][2] == 600.0


class TestClaudeClientGenerateReply:
    def test_returns_stripped_output(self) -> None:
        mock_run = MagicMock(return_value=_completed("  woof!  \n"))
        client = ClaudeClient(runner=mock_run)
        assert client.generate_reply("write a reply") == "woof!"

    def test_returns_empty_on_nonzero(self) -> None:
        mock_run = MagicMock(return_value=_completed("err", returncode=1))
        client = ClaudeClient(runner=mock_run)
        assert client.generate_reply("write") == ""

    def test_returns_empty_on_timeout(self) -> None:
        mock_run = MagicMock(side_effect=subprocess.TimeoutExpired("claude", 30))
        client = ClaudeClient(runner=mock_run)
        assert client.generate_reply("write") == ""

    def test_returns_empty_on_file_not_found(self) -> None:
        mock_run = MagicMock(side_effect=FileNotFoundError)
        client = ClaudeClient(runner=mock_run)
        assert client.generate_reply("write") == ""

    def test_default_model_and_timeout(self) -> None:
        mock_run = MagicMock(return_value=_completed("ok"))
        client = ClaudeClient(runner=mock_run)
        client.generate_reply("write a reply")
        cmd = mock_run.call_args.args[0]
        assert "claude-opus-4-6" in cmd
        assert mock_run.call_args.kwargs["timeout"] == 30

    def test_custom_model_and_timeout(self) -> None:
        mock_run = MagicMock(return_value=_completed("ok"))
        client = ClaudeClient(runner=mock_run)
        client.generate_reply("write", model="claude-sonnet-4-6", timeout=10)
        cmd = mock_run.call_args.args[0]
        assert "claude-sonnet-4-6" in cmd
        assert mock_run.call_args.kwargs["timeout"] == 10


class TestClaudeClientGenerateBranchName:
    def test_returns_first_line(self) -> None:
        mock_run = MagicMock(return_value=_completed("fix-the-bug\nextra"))
        client = ClaudeClient(runner=mock_run)
        assert client.generate_branch_name("name this") == "fix-the-bug"

    def test_returns_empty_on_nonzero(self) -> None:
        mock_run = MagicMock(return_value=_completed("slug", returncode=1))
        client = ClaudeClient(runner=mock_run)
        assert client.generate_branch_name("name this") == ""

    def test_returns_empty_on_empty_output(self) -> None:
        mock_run = MagicMock(return_value=_completed(""))
        client = ClaudeClient(runner=mock_run)
        assert client.generate_branch_name("name this") == ""

    def test_returns_empty_on_timeout(self) -> None:
        mock_run = MagicMock(side_effect=subprocess.TimeoutExpired("claude", 15))
        client = ClaudeClient(runner=mock_run)
        assert client.generate_branch_name("name") == ""

    def test_returns_empty_on_file_not_found(self) -> None:
        mock_run = MagicMock(side_effect=FileNotFoundError)
        client = ClaudeClient(runner=mock_run)
        assert client.generate_branch_name("name") == ""

    def test_default_model_and_timeout(self) -> None:
        mock_run = MagicMock(return_value=_completed("slug"))
        client = ClaudeClient(runner=mock_run)
        client.generate_branch_name("name this")
        cmd = mock_run.call_args.args[0]
        assert "claude-haiku-4-5-20251001" in cmd
        assert mock_run.call_args.kwargs["timeout"] == 15

    def test_custom_model_and_timeout(self) -> None:
        mock_run = MagicMock(return_value=_completed("slug"))
        client = ClaudeClient(runner=mock_run)
        client.generate_branch_name("name", model="claude-opus-4-6", timeout=5)
        cmd = mock_run.call_args.args[0]
        assert "claude-opus-4-6" in cmd
        assert mock_run.call_args.kwargs["timeout"] == 5


class TestClaudeClientGenerateStatusWithSession:
    _RESULT_LINE = '{"type":"result","result":"🐶\\ncoding","session_id":"sess-42"}'

    def test_returns_text_and_session_id_on_success(self) -> None:
        mock_run = MagicMock(return_value=_completed(self._RESULT_LINE))
        client = ClaudeClient(runner=mock_run)
        text, sid = client.generate_status_with_session("doing stuff", "be fido")
        assert text == "🐶\ncoding"
        assert sid == "sess-42"

    def test_returns_empty_pair_on_nonzero(self) -> None:
        mock_run = MagicMock(return_value=_completed(self._RESULT_LINE, returncode=1))
        client = ClaudeClient(runner=mock_run, sleep_fn=MagicMock())
        assert client.generate_status_with_session("doing stuff", "sys") == ("", "")
        mock_run.assert_called_once()

    def test_returns_empty_pair_on_timeout(self) -> None:
        mock_run = MagicMock(side_effect=subprocess.TimeoutExpired("claude", 15))
        client = ClaudeClient(runner=mock_run, sleep_fn=MagicMock())
        assert client.generate_status_with_session("doing stuff", "sys") == ("", "")
        mock_run.assert_called_once()

    def test_returns_empty_pair_on_file_not_found(self) -> None:
        mock_run = MagicMock(side_effect=FileNotFoundError)
        client = ClaudeClient(runner=mock_run, sleep_fn=MagicMock())
        assert client.generate_status_with_session("doing stuff", "sys") == ("", "")
        mock_run.assert_called_once()

    def test_passes_correct_flags(self) -> None:
        mock_run = MagicMock(return_value=_completed(self._RESULT_LINE))
        client = ClaudeClient(runner=mock_run)
        client.generate_status_with_session(
            "working on #42", "be a dog", model="claude-sonnet-4-6", timeout=10
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
        client = ClaudeClient(runner=mock_run)
        client.generate_status_with_session("working", "sys")
        cmd = mock_run.call_args.args[0]
        assert "claude-opus-4-6" in cmd
        assert mock_run.call_args.kwargs["timeout"] == 15

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
        client = ClaudeClient(runner=mock_run, sleep_fn=mock_sleep)
        text, sid = client.generate_status_with_session("working", "sys")
        assert text == "🐶\ncoding"
        assert sid == "sess-42"
        assert mock_run.call_count == 3
        assert mock_sleep.call_count == 2

    def test_returns_empty_pair_after_all_retries_exhausted(self) -> None:
        empty_line = '{"type":"result","session_id":"s1"}'
        mock_run = MagicMock(return_value=_completed(empty_line))
        mock_sleep = MagicMock()
        client = ClaudeClient(runner=mock_run, sleep_fn=mock_sleep)
        assert client.generate_status_with_session("working", "sys") == ("", "")
        assert mock_run.call_count == 3
        assert mock_sleep.call_count == 2

    def test_returns_empty_session_when_no_session_id(self) -> None:
        no_sid = '{"type":"result","result":"🐶\\nwoof"}'
        mock_run = MagicMock(return_value=_completed(no_sid))
        client = ClaudeClient(runner=mock_run)
        text, sid = client.generate_status_with_session("working", "sys")
        assert text == "🐶\nwoof"
        assert sid == ""

    def test_logs_stderr_at_warning_on_empty_output(self, caplog) -> None:
        empty_line = '{"type":"result","session_id":"s1"}'
        mock_run = MagicMock(
            return_value=_completed(empty_line, stderr="Rate limit exceeded")
        )
        client = ClaudeClient(runner=mock_run, sleep_fn=MagicMock())
        with caplog.at_level(logging.WARNING, logger="kennel.claude"):
            client.generate_status_with_session("working", "sys")
        assert "Rate limit exceeded" in caplog.text

    def test_logs_raw_stdout_at_debug_on_empty_output(self, caplog) -> None:
        empty_line = '{"type":"result","session_id":"s1"}'
        mock_run = MagicMock(return_value=_completed(empty_line))
        client = ClaudeClient(runner=mock_run, sleep_fn=MagicMock())
        with caplog.at_level(logging.DEBUG, logger="kennel.claude"):
            client.generate_status_with_session("working", "sys")
        assert "stdout=" in caplog.text

    def test_no_stderr_log_when_stderr_empty(self, caplog) -> None:
        empty_line = '{"type":"result","session_id":"s1"}'
        mock_run = MagicMock(return_value=_completed(empty_line))
        client = ClaudeClient(runner=mock_run, sleep_fn=MagicMock())
        with caplog.at_level(logging.WARNING, logger="kennel.claude"):
            client.generate_status_with_session("working", "sys")
        assert "stderr=" not in caplog.text


class TestClaudeClientResumeStatus:
    _RESULT_LINE = '{"type":"result","result":"🐕\\nfetching","session_id":"s-1"}'

    def test_returns_text_on_success(self) -> None:
        mock_run = MagicMock(return_value=_completed(self._RESULT_LINE))
        client = ClaudeClient(runner=mock_run)
        assert client.resume_status("s-1", "please shorten") == "🐕\nfetching"

    def test_returns_empty_on_nonzero(self) -> None:
        mock_run = MagicMock(return_value=_completed(self._RESULT_LINE, returncode=1))
        client = ClaudeClient(runner=mock_run)
        assert client.resume_status("s-1", "shorten") == ""

    def test_returns_empty_on_timeout(self) -> None:
        mock_run = MagicMock(side_effect=subprocess.TimeoutExpired("claude", 15))
        client = ClaudeClient(runner=mock_run)
        assert client.resume_status("s-1", "shorten") == ""

    def test_returns_empty_on_file_not_found(self) -> None:
        mock_run = MagicMock(side_effect=FileNotFoundError)
        client = ClaudeClient(runner=mock_run)
        assert client.resume_status("s-1", "shorten") == ""

    def test_passes_correct_flags(self) -> None:
        mock_run = MagicMock(return_value=_completed(self._RESULT_LINE))
        client = ClaudeClient(runner=mock_run)
        client.resume_status(
            "my-session", "shorten to 80 chars", model="claude-sonnet-4-6", timeout=20
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
        client = ClaudeClient(runner=mock_run)
        client.resume_status("s-1", "shorten")
        cmd = mock_run.call_args.args[0]
        assert "claude-opus-4-6" in cmd
        assert mock_run.call_args.kwargs["timeout"] == 15

    def test_returns_empty_when_no_result_field(self) -> None:
        no_result = '{"type":"result","session_id":"s-1"}'
        mock_run = MagicMock(return_value=_completed(no_result))
        client = ClaudeClient(runner=mock_run)
        assert client.resume_status("s-1", "shorten") == ""
