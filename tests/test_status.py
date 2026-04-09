"""Tests for kennel.status — state-reading and formatting."""

from __future__ import annotations

import fcntl
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from kennel.config import RepoConfig
from kennel.status import (
    KennelStatus,
    RepoStatus,
    _claude_pid,
    _current_task,
    _fetch_activities,
    _fido_running,
    _format_uptime,
    _git_dir,
    _kennel_pid,
    _pgrep,
    _port_from_pid,
    _process_uptime_seconds,
    _read_state,
    _read_tasks,
    _repos_from_pid,
    collect,
    format_status,
    repo_status,
)


class TestFormatUptime:
    def test_seconds(self) -> None:
        assert _format_uptime(45) == "45s"

    def test_one_second(self) -> None:
        assert _format_uptime(1) == "1s"

    def test_fifty_nine_seconds(self) -> None:
        assert _format_uptime(59) == "59s"

    def test_minutes(self) -> None:
        assert _format_uptime(90) == "1m"

    def test_minutes_exact(self) -> None:
        assert _format_uptime(120) == "2m"

    def test_fifty_nine_minutes(self) -> None:
        assert _format_uptime(59 * 60) == "59m"

    def test_hours_and_minutes(self) -> None:
        assert _format_uptime(2 * 3600 + 13 * 60) == "2h13m"

    def test_hours_exact(self) -> None:
        assert _format_uptime(3 * 3600) == "3h"

    def test_one_hour_one_minute(self) -> None:
        assert _format_uptime(3600 + 60) == "1h1m"


class TestFidoRunning:
    def test_missing_lock_returns_false(self, tmp_path: Path) -> None:
        lock = tmp_path / "lock"
        assert _fido_running(lock) is False

    def test_unlocked_returns_false(self, tmp_path: Path) -> None:
        lock = tmp_path / "lock"
        lock.touch()
        assert _fido_running(lock) is False

    def test_locked_returns_true(self, tmp_path: Path) -> None:
        lock = tmp_path / "lock"
        lock.touch()
        fd = open(lock)  # noqa: SIM115
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            assert _fido_running(lock) is True
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            fd.close()

    def test_oserror_returns_false(self, tmp_path: Path) -> None:
        lock = tmp_path / "lock"
        lock.touch()
        with patch("builtins.open", side_effect=OSError("no perms")):
            assert _fido_running(lock) is False


class TestPgrep:
    def test_returns_pids(self) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="12345\n67890\n")
            result = _pgrep("some pattern")
        assert result == [12345, 67890]

    def test_strips_whitespace(self) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="  42  \n")
            result = _pgrep("pattern")
        assert result == [42]

    def test_empty_output(self) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="")
            result = _pgrep("pattern")
        assert result == []

    def test_skips_non_integer_lines(self) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="123\nnot-a-pid\n456\n")
            result = _pgrep("pattern")
        assert result == [123, 456]

    def test_oserror_returns_empty(self) -> None:
        with patch("subprocess.run", side_effect=OSError("no pgrep")):
            result = _pgrep("pattern")
        assert result == []

    def test_passes_pattern_to_pgrep(self) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="")
            _pgrep("kennel --port")
        mock_run.assert_called_once_with(
            ["pgrep", "-f", "kennel --port"],
            capture_output=True,
            text=True,
        )


class TestProcessUptimeSeconds:
    def test_returns_elapsed_time(self) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="  3742  ")
            result = _process_uptime_seconds(12345)
        assert result == 3742

    def test_none_when_no_output(self) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="")
            result = _process_uptime_seconds(99)
        assert result is None

    def test_none_on_oserror(self) -> None:
        with patch("subprocess.run", side_effect=OSError):
            result = _process_uptime_seconds(99)
        assert result is None

    def test_none_on_value_error(self) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="not-a-number")
            result = _process_uptime_seconds(99)
        assert result is None

    def test_calls_ps_with_pid(self) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="100")
            _process_uptime_seconds(42)
        mock_run.assert_called_once_with(
            ["ps", "-p", "42", "-o", "etimes="],
            capture_output=True,
            text=True,
        )


class TestReposFromPid:
    def test_parses_single_repo(self) -> None:
        cmdline = (
            b"kennel\x00--port\x009000\x00rhencke/confusio:/workspace/confusio\x00"
        )
        with patch.object(Path, "read_bytes", return_value=cmdline):
            result = _repos_from_pid(123)
        assert len(result) == 1
        assert result[0].name == "rhencke/confusio"
        assert result[0].work_dir == Path("/workspace/confusio")

    def test_parses_multiple_repos(self) -> None:
        cmdline = b"kennel\x00rhencke/a:/path/a\x00rhencke/b:/path/b\x00"
        with patch.object(Path, "read_bytes", return_value=cmdline):
            result = _repos_from_pid(123)
        assert len(result) == 2
        assert result[0].name == "rhencke/a"
        assert result[1].name == "rhencke/b"

    def test_oserror_returns_empty(self) -> None:
        with patch.object(Path, "read_bytes", side_effect=OSError("no proc")):
            result = _repos_from_pid(123)
        assert result == []

    def test_skips_args_without_colon(self) -> None:
        cmdline = b"kennel\x00--port\x009000\x00"
        with patch.object(Path, "read_bytes", return_value=cmdline):
            result = _repos_from_pid(123)
        assert result == []

    def test_skips_colon_args_without_slash_in_name(self) -> None:
        cmdline = b"something:value\x00"
        with patch.object(Path, "read_bytes", return_value=cmdline):
            result = _repos_from_pid(123)
        assert result == []

    def test_reads_proc_pid_cmdline(self) -> None:
        paths_read: list[Path] = []

        def capturing_read_bytes(self_path: Path) -> bytes:
            paths_read.append(self_path)
            return b""

        with patch.object(Path, "read_bytes", capturing_read_bytes):
            _repos_from_pid(789)
        assert paths_read == [Path("/proc/789/cmdline")]

    def test_expands_tilde_in_path(self) -> None:
        cmdline = b"rhencke/repo:~/workspace/repo\x00"
        with patch.object(Path, "read_bytes", return_value=cmdline):
            result = _repos_from_pid(1)
        assert result[0].work_dir == Path("~/workspace/repo").expanduser()

    def test_no_repos_returns_empty(self) -> None:
        cmdline = b"kennel\x00--port\x009000\x00--secret-file\x00/home/user/.kennel-secret\x00"
        with patch.object(Path, "read_bytes", return_value=cmdline):
            result = _repos_from_pid(1)
        assert result == []

    def test_skips_non_utf8_args(self) -> None:
        cmdline = b"\xff\xfe\x00rhencke/repo:/path\x00"
        with patch.object(Path, "read_bytes", return_value=cmdline):
            result = _repos_from_pid(1)
        assert len(result) == 1
        assert result[0].name == "rhencke/repo"


class TestKennelPid:
    def test_returns_first_pid(self) -> None:
        with patch("kennel.status._pgrep", return_value=[111, 222]):
            result = _kennel_pid()
        assert result == 111

    def test_returns_none_when_no_match(self) -> None:
        with patch("kennel.status._pgrep", return_value=[]):
            result = _kennel_pid()
        assert result is None

    def test_searches_for_kennel_port(self) -> None:
        with patch("kennel.status._pgrep", return_value=[]) as mock:
            _kennel_pid()
        mock.assert_called_once_with("kennel --port")


class TestPortFromPid:
    def test_returns_port(self) -> None:
        cmdline = b"kennel\x00--port\x009000\x00rhencke/repo:/path\x00"
        with patch.object(Path, "read_bytes", return_value=cmdline):
            assert _port_from_pid(42) == 9000

    def test_returns_none_when_no_port_flag(self) -> None:
        cmdline = b"kennel\x00rhencke/repo:/path\x00"
        with patch.object(Path, "read_bytes", return_value=cmdline):
            assert _port_from_pid(42) is None

    def test_returns_none_when_port_flag_last_arg(self) -> None:
        cmdline = b"kennel\x00--port\x00"
        with patch.object(Path, "read_bytes", return_value=cmdline):
            assert _port_from_pid(42) is None

    def test_returns_none_when_port_value_not_integer(self) -> None:
        cmdline = b"kennel\x00--port\x00notanumber\x00"
        with patch.object(Path, "read_bytes", return_value=cmdline):
            assert _port_from_pid(42) is None

    def test_oserror_returns_none(self) -> None:
        with patch.object(Path, "read_bytes", side_effect=OSError("no proc")):
            assert _port_from_pid(42) is None

    def test_reads_correct_pid_path(self) -> None:
        paths_read: list[Path] = []

        def capturing_read_bytes(self_path: Path) -> bytes:
            paths_read.append(self_path)
            return b""

        with patch.object(Path, "read_bytes", capturing_read_bytes):
            _port_from_pid(1234)
        assert paths_read == [Path("/proc/1234/cmdline")]


class TestFetchActivities:
    def test_returns_repo_what_map(self) -> None:
        data = json.dumps(
            [{"repo_name": "owner/repo", "what": "Working on: #1", "busy": True}]
        ).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = data
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = _fetch_activities(9000)
        assert result == {"owner/repo": "Working on: #1"}

    def test_returns_multiple_repos(self) -> None:
        data = json.dumps(
            [
                {"repo_name": "a/b", "what": "Napping", "busy": False},
                {"repo_name": "c/d", "what": "Fixing CI", "busy": True},
            ]
        ).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = data
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = _fetch_activities(9000)
        assert result == {"a/b": "Napping", "c/d": "Fixing CI"}

    def test_returns_empty_on_exception(self) -> None:
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            result = _fetch_activities(9000)
        assert result == {}

    def test_skips_items_without_repo_name(self) -> None:
        data = json.dumps([{"what": "something", "busy": True}]).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = data
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = _fetch_activities(9000)
        assert result == {}

    def test_skips_items_without_what(self) -> None:
        data = json.dumps([{"repo_name": "a/b", "busy": True}]).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = data
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = _fetch_activities(9000)
        assert result == {}

    def test_calls_correct_url(self) -> None:
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b"[]"
        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            _fetch_activities(8888)
        mock_open.assert_called_once_with("http://localhost:8888/status", timeout=2)


class TestClaudePid:
    def test_returns_first_pid(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        with patch("kennel.status._pgrep", return_value=[999]):
            result = _claude_pid(fido_dir)
        assert result == 999

    def test_returns_none_when_no_match(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        with patch("kennel.status._pgrep", return_value=[]):
            result = _claude_pid(fido_dir)
        assert result is None

    def test_searches_for_system_file(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        with patch("kennel.status._pgrep", return_value=[]) as mock:
            _claude_pid(fido_dir)
        mock.assert_called_once_with(str(fido_dir / "system"))


class TestGitDir:
    def test_returns_path(self, tmp_path: Path) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="/repo/.git\n")
            result = _git_dir(tmp_path)
        assert result == Path("/repo/.git")

    def test_strips_whitespace(self, tmp_path: Path) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="  /a/b/.git  \n")
            result = _git_dir(tmp_path)
        assert result == Path("/a/b/.git")

    def test_returns_none_on_error(self, tmp_path: Path) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(128, "git")
            result = _git_dir(tmp_path)
        assert result is None


class TestReadState:
    def test_absent_returns_empty(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        assert _read_state(fido_dir) == {}

    def test_reads_state(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (fido_dir / "state.json").write_text('{"issue": 42}')
        assert _read_state(fido_dir) == {"issue": 42}

    def test_corrupt_json_returns_empty(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (fido_dir / "state.json").write_text("not json {{{")
        assert _read_state(fido_dir) == {}

    def test_oserror_returns_empty(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        path = fido_dir / "state.json"
        path.touch()
        with patch.object(Path, "read_text", side_effect=OSError("oops")):
            assert _read_state(fido_dir) == {}


class TestReadTasks:
    def test_absent_returns_empty(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        assert _read_tasks(fido_dir) == []

    def test_reads_tasks(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        tasks = [{"id": "1", "title": "do thing", "status": "pending"}]
        (fido_dir / "tasks.json").write_text(json.dumps(tasks))
        assert _read_tasks(fido_dir) == tasks

    def test_corrupt_json_returns_empty(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (fido_dir / "tasks.json").write_text("not json")
        assert _read_tasks(fido_dir) == []

    def test_non_list_returns_empty(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (fido_dir / "tasks.json").write_text('{"not": "a list"}')
        assert _read_tasks(fido_dir) == []

    def test_oserror_returns_empty(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (fido_dir / "tasks.json").touch()
        with patch.object(Path, "read_text", side_effect=OSError("oops")):
            assert _read_tasks(fido_dir) == []


class TestCurrentTask:
    def test_empty_list(self) -> None:
        assert _current_task([]) is None

    def test_returns_in_progress_first(self) -> None:
        tasks = [
            {"status": "pending", "title": "pending task"},
            {"status": "in_progress", "title": "active task"},
        ]
        assert _current_task(tasks) == "active task"

    def test_returns_first_pending_when_no_in_progress(self) -> None:
        tasks = [
            {"status": "completed", "title": "done"},
            {"status": "pending", "title": "next"},
            {"status": "pending", "title": "later"},
        ]
        assert _current_task(tasks) == "next"

    def test_all_completed_returns_none(self) -> None:
        tasks = [{"status": "completed", "title": "done"}]
        assert _current_task(tasks) is None

    def test_in_progress_preferred_over_pending(self) -> None:
        tasks = [
            {"status": "pending", "title": "first pending"},
            {"status": "in_progress", "title": "in progress"},
            {"status": "pending", "title": "second pending"},
        ]
        assert _current_task(tasks) == "in progress"


class TestRepoStatus:
    def _make_config(self, tmp_path: Path) -> RepoConfig:
        return RepoConfig(name="owner/repo", work_dir=tmp_path)

    def test_no_git_dir_returns_empty_status(self, tmp_path: Path) -> None:
        cfg = self._make_config(tmp_path)
        with patch("kennel.status._git_dir", return_value=None):
            result = repo_status(cfg)
        assert result.name == "owner/repo"
        assert result.fido_running is False
        assert result.issue is None
        assert result.pending == 0
        assert result.completed == 0
        assert result.current_task is None
        assert result.claude_pid is None
        assert result.claude_uptime is None
        assert result.worker_what is None

    def test_no_git_dir_passes_worker_what(self, tmp_path: Path) -> None:
        cfg = self._make_config(tmp_path)
        with patch("kennel.status._git_dir", return_value=None):
            result = repo_status(cfg, worker_what="Napping")
        assert result.worker_what == "Napping"

    def test_with_running_fido_and_issue(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        fido_dir = git_dir / "fido"
        fido_dir.mkdir(parents=True)
        (fido_dir / "state.json").write_text('{"issue": 7}')
        tasks = [
            {"status": "completed", "title": "done task"},
            {"status": "pending", "title": "next task"},
        ]
        (fido_dir / "tasks.json").write_text(json.dumps(tasks))

        cfg = self._make_config(tmp_path)
        with (
            patch("kennel.status._git_dir", return_value=git_dir),
            patch("kennel.status._fido_running", return_value=True),
            patch("kennel.status._claude_pid", return_value=555),
            patch("kennel.status._process_uptime_seconds", return_value=180),
        ):
            result = repo_status(cfg, worker_what="Working on: #7 do thing")

        assert result.name == "owner/repo"
        assert result.fido_running is True
        assert result.issue == 7
        assert result.pending == 1
        assert result.completed == 1
        assert result.current_task == "next task"
        assert result.claude_pid == 555
        assert result.claude_uptime == 180
        assert result.worker_what == "Working on: #7 do thing"

    def test_worker_what_defaults_to_none(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        fido_dir = git_dir / "fido"
        fido_dir.mkdir(parents=True)

        cfg = self._make_config(tmp_path)
        with (
            patch("kennel.status._git_dir", return_value=git_dir),
            patch("kennel.status._fido_running", return_value=False),
            patch("kennel.status._claude_pid", return_value=None),
            patch("kennel.status._process_uptime_seconds", return_value=None),
        ):
            result = repo_status(cfg)
        assert result.worker_what is None

    def test_no_claude_pid_skips_uptime(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        fido_dir = git_dir / "fido"
        fido_dir.mkdir(parents=True)

        cfg = self._make_config(tmp_path)
        with (
            patch("kennel.status._git_dir", return_value=git_dir),
            patch("kennel.status._fido_running", return_value=False),
            patch("kennel.status._claude_pid", return_value=None),
            patch("kennel.status._process_uptime_seconds") as mock_uptime,
        ):
            result = repo_status(cfg)

        mock_uptime.assert_not_called()
        assert result.claude_pid is None
        assert result.claude_uptime is None


class TestCollect:
    def _fake_repo_status(self, name: str = "owner/repo") -> RepoStatus:
        return RepoStatus(
            name=name,
            fido_running=False,
            issue=None,
            pending=0,
            completed=0,
            current_task=None,
            claude_pid=None,
            claude_uptime=None,
            worker_what=None,
        )

    def test_kennel_up_with_uptime(self, tmp_path: Path) -> None:
        rc = RepoConfig(name="owner/repo", work_dir=tmp_path)
        with (
            patch("kennel.status._kennel_pid", return_value=42),
            patch("kennel.status._repos_from_pid", return_value=[rc]),
            patch("kennel.status._process_uptime_seconds", return_value=600),
            patch("kennel.status._port_from_pid", return_value=None),
            patch("kennel.status.repo_status", return_value=self._fake_repo_status()),
        ):
            result = collect()

        assert result.kennel_pid == 42
        assert result.kennel_uptime == 600
        assert len(result.repos) == 1

    def test_kennel_down(self) -> None:
        with (
            patch("kennel.status._kennel_pid", return_value=None),
            patch("kennel.status._repos_from_pid") as mock_repos,
            patch("kennel.status._process_uptime_seconds") as mock_uptime,
            patch("kennel.status._port_from_pid") as mock_port,
            patch("kennel.status.repo_status") as mock_repo_status,
        ):
            result = collect()

        mock_uptime.assert_not_called()
        mock_repos.assert_not_called()
        mock_repo_status.assert_not_called()
        mock_port.assert_not_called()
        assert result.kennel_pid is None
        assert result.kennel_uptime is None

    def test_fetches_activities_when_port_known(self, tmp_path: Path) -> None:
        rc = RepoConfig(name="owner/repo", work_dir=tmp_path)
        with (
            patch("kennel.status._kennel_pid", return_value=42),
            patch("kennel.status._repos_from_pid", return_value=[rc]),
            patch("kennel.status._process_uptime_seconds", return_value=0),
            patch("kennel.status._port_from_pid", return_value=9000),
            patch(
                "kennel.status._fetch_activities",
                return_value={"owner/repo": "Working on: #1"},
            ) as mock_fetch,
            patch("kennel.status.repo_status", return_value=self._fake_repo_status()),
        ):
            collect()
        mock_fetch.assert_called_once_with(9000)

    def test_skips_fetch_when_port_unknown(self, tmp_path: Path) -> None:
        rc = RepoConfig(name="owner/repo", work_dir=tmp_path)
        with (
            patch("kennel.status._kennel_pid", return_value=42),
            patch("kennel.status._repos_from_pid", return_value=[rc]),
            patch("kennel.status._process_uptime_seconds", return_value=0),
            patch("kennel.status._port_from_pid", return_value=None),
            patch("kennel.status._fetch_activities") as mock_fetch,
            patch("kennel.status.repo_status", return_value=self._fake_repo_status()),
        ):
            collect()
        mock_fetch.assert_not_called()

    def test_passes_worker_what_to_repo_status(self, tmp_path: Path) -> None:
        rc = RepoConfig(name="owner/repo", work_dir=tmp_path)
        with (
            patch("kennel.status._kennel_pid", return_value=42),
            patch("kennel.status._repos_from_pid", return_value=[rc]),
            patch("kennel.status._process_uptime_seconds", return_value=0),
            patch("kennel.status._port_from_pid", return_value=9000),
            patch(
                "kennel.status._fetch_activities",
                return_value={"owner/repo": "Fixing CI: tests"},
            ),
            patch(
                "kennel.status.repo_status", return_value=self._fake_repo_status()
            ) as mock_rs,
        ):
            collect()
        mock_rs.assert_called_once_with(rc, worker_what="Fixing CI: tests")

    def test_worker_what_none_for_unknown_repo(self, tmp_path: Path) -> None:
        rc = RepoConfig(name="owner/repo", work_dir=tmp_path)
        with (
            patch("kennel.status._kennel_pid", return_value=42),
            patch("kennel.status._repos_from_pid", return_value=[rc]),
            patch("kennel.status._process_uptime_seconds", return_value=0),
            patch("kennel.status._port_from_pid", return_value=9000),
            patch("kennel.status._fetch_activities", return_value={}),
            patch(
                "kennel.status.repo_status", return_value=self._fake_repo_status()
            ) as mock_rs,
        ):
            collect()
        mock_rs.assert_called_once_with(rc, worker_what=None)


class TestFormatStatus:
    def _repo(self, **kwargs) -> RepoStatus:
        defaults = dict(
            name="owner/repo",
            fido_running=False,
            issue=None,
            pending=0,
            completed=0,
            current_task=None,
            claude_pid=None,
            claude_uptime=None,
            worker_what=None,
        )
        defaults.update(kwargs)
        return RepoStatus(**defaults)

    def test_kennel_up_with_uptime(self) -> None:
        status = KennelStatus(kennel_pid=12345, kennel_uptime=7980, repos=[])
        output = format_status(status)
        assert output == "kennel: UP (pid 12345, uptime 2h13m)"

    def test_kennel_up_no_uptime(self) -> None:
        status = KennelStatus(kennel_pid=12345, kennel_uptime=None, repos=[])
        output = format_status(status)
        assert output == "kennel: UP (pid 12345)"

    def test_kennel_down(self) -> None:
        status = KennelStatus(kennel_pid=None, kennel_uptime=None, repos=[])
        output = format_status(status)
        assert output == "kennel: DOWN"

    def test_repo_fido_idle_no_issue(self) -> None:
        status = KennelStatus(
            kennel_pid=None,
            kennel_uptime=None,
            repos=[self._repo(name="owner/myrepo")],
        )
        output = format_status(status)
        assert "owner/myrepo: fido idle — no assigned issues" in output

    def test_repo_fido_running_no_issue(self) -> None:
        status = KennelStatus(
            kennel_pid=None,
            kennel_uptime=None,
            repos=[self._repo(fido_running=True)],
        )
        output = format_status(status)
        assert "fido running" in output

    def test_repo_with_issue_and_task(self) -> None:
        repo = self._repo(
            name="owner/repo",
            fido_running=True,
            issue=42,
            pending=1,
            completed=2,
            current_task="Do the thing",
        )
        status = KennelStatus(kennel_pid=None, kennel_uptime=None, repos=[repo])
        output = format_status(status)
        assert 'issue #42, task 3/3 "Do the thing"' in output

    def test_repo_issue_no_current_task_but_has_tasks(self) -> None:
        repo = self._repo(issue=5, pending=2, completed=0)
        status = KennelStatus(kennel_pid=None, kennel_uptime=None, repos=[repo])
        output = format_status(status)
        assert "issue #5, 2 pending" in output

    def test_repo_issue_no_tasks(self) -> None:
        repo = self._repo(issue=3)
        status = KennelStatus(kennel_pid=None, kennel_uptime=None, repos=[repo])
        output = format_status(status)
        assert "issue #3" in output
        assert "pending" not in output

    def test_claude_pid_with_uptime(self) -> None:
        repo = self._repo(claude_pid=9999, claude_uptime=185)
        status = KennelStatus(kennel_pid=None, kennel_uptime=None, repos=[repo])
        output = format_status(status)
        assert "claude pid 9999 (running 3m)" in output

    def test_claude_pid_no_uptime(self) -> None:
        repo = self._repo(claude_pid=9999, claude_uptime=None)
        status = KennelStatus(kennel_pid=None, kennel_uptime=None, repos=[repo])
        output = format_status(status)
        assert "claude pid 9999" in output
        assert "running" not in output

    def test_multiple_repos(self) -> None:
        repos = [
            self._repo(name="a/b"),
            self._repo(name="c/d"),
        ]
        status = KennelStatus(kennel_pid=1, kennel_uptime=60, repos=repos)
        lines = format_status(status).splitlines()
        assert len(lines) == 3
        assert lines[0].startswith("kennel:")
        assert lines[1].startswith("a/b:")
        assert lines[2].startswith("c/d:")

    def test_worker_what_included_in_output(self) -> None:
        repo = self._repo(fido_running=True, worker_what="Working on: #3 add widget")
        status = KennelStatus(kennel_pid=None, kennel_uptime=None, repos=[repo])
        output = format_status(status)
        assert "Working on: #3 add widget" in output

    def test_worker_what_none_not_included(self) -> None:
        repo = self._repo(fido_running=True, worker_what=None)
        status = KennelStatus(kennel_pid=None, kennel_uptime=None, repos=[repo])
        output = format_status(status)
        assert "Working on:" not in output

    def test_worker_what_appears_after_fido_status(self) -> None:
        repo = self._repo(fido_running=True, worker_what="Napping — waiting for work")
        status = KennelStatus(kennel_pid=None, kennel_uptime=None, repos=[repo])
        output = format_status(status)
        assert "fido running — Napping — waiting for work" in output
