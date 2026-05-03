"""Tests for fido.status — state-reading and formatting."""

# pyright: reportArgumentType=false, reportMissingParameterType=false, reportMissingTypeArgument=false, reportPrivateUsage=false, reportUnknownLambdaType=false

import fcntl
import json
import subprocess
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fido.color import _CODES
from fido.config import RepoConfig as _RepoConfig
from fido.provider import (
    ProviderID,
    ProviderLimitSnapshot,
    ProviderLimitWindow,
    ProviderPressureStatus,
)
from fido.status import (
    ClaudeTalkerInfo,
    FidoStatus,
    IssueCacheInfo,
    RateLimitInfo,
    RateLimitWindowInfo,
    RepoStatus,
    SystemResourceInfo,
    WebhookActivityInfo,
    _claude_pid,
    _current_task,
    _fetch_activities,
    _fido_pid,
    _fido_running,
    _format_agent_line,
    _format_cache_line,
    _format_duration_until,
    _format_gib,
    _format_percent,
    _format_rate_limit_line,
    _format_rate_limit_window,
    _format_resource_line,
    _format_uptime,
    _git_dir,
    _parse_iso_datetime,
    _parse_issue_cache,
    _parse_rate_limit,
    _pgrep,
    _port_from_pid,
    _process_uptime_seconds,
    _rate_limit_color,
    _read_state,
    _repos_from_pid,
    _system_resources,
    collect,
    format_status,
    provider_statuses_for_repo_configs,
    repo_status,
    running_repo_configs,
)


class RepoConfig(_RepoConfig):
    def __init__(self, *args, provider: ProviderID = ProviderID.CLAUDE_CODE, **kwargs):
        super().__init__(*args, provider=provider, **kwargs)


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
        mock_run = MagicMock(return_value=MagicMock(stdout="12345\n67890\n"))
        assert _pgrep("some pattern", _run=mock_run) == [12345, 67890]

    def test_strips_whitespace(self) -> None:
        mock_run = MagicMock(return_value=MagicMock(stdout="  42  \n"))
        assert _pgrep("pattern", _run=mock_run) == [42]

    def test_empty_output(self) -> None:
        mock_run = MagicMock(return_value=MagicMock(stdout=""))
        assert _pgrep("pattern", _run=mock_run) == []

    def test_skips_non_integer_lines(self) -> None:
        mock_run = MagicMock(return_value=MagicMock(stdout="123\nnot-a-pid\n456\n"))
        assert _pgrep("pattern", _run=mock_run) == [123, 456]

    def test_oserror_returns_empty(self) -> None:
        mock_run = MagicMock(side_effect=OSError("no pgrep"))
        assert _pgrep("pattern", _run=mock_run) == []

    def test_passes_pattern_to_pgrep(self) -> None:
        mock_run = MagicMock(return_value=MagicMock(stdout=""))
        _pgrep("fido --port", _run=mock_run)
        mock_run.assert_called_once_with(
            ["pgrep", "-f", "fido --port"],
            capture_output=True,
            text=True,
        )


class TestRunningRepoConfigs:
    def test_returns_empty_when_fido_not_running(self) -> None:
        assert running_repo_configs(_fido_pid_fn=lambda: None) == []

    def test_reads_repos_from_running_fido(self, tmp_path: Path) -> None:
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        assert running_repo_configs(
            _fido_pid_fn=lambda: 123,
            _repos_from_pid_fn=lambda pid: [repo_cfg],
        ) == [repo_cfg]


class TestProviderStatusesForRepoConfigs:
    def test_dedupes_by_provider(self, tmp_path: Path) -> None:
        claude_a = RepoConfig(name="owner/a", work_dir=tmp_path / "a")
        claude_b = RepoConfig(name="owner/b", work_dir=tmp_path / "b")
        copilot = RepoConfig(
            name="owner/c",
            work_dir=tmp_path / "c",
            provider=ProviderID.COPILOT_CLI,
        )
        factory = MagicMock()
        first = MagicMock()
        second = MagicMock()
        first.get_limit_snapshot.return_value = ProviderLimitSnapshot(
            provider=ProviderID.CLAUDE_CODE
        )
        second.get_limit_snapshot.return_value = ProviderLimitSnapshot(
            provider=ProviderID.COPILOT_CLI
        )
        factory.create_api.side_effect = [first, second]

        statuses = provider_statuses_for_repo_configs(
            [claude_a, claude_b, copilot],
            _provider_factory=factory,
        )

        assert list(statuses) == [ProviderID.CLAUDE_CODE, ProviderID.COPILOT_CLI]
        factory.create_api.assert_any_call(claude_a)
        factory.create_api.assert_any_call(copilot)

    def test_builds_default_factory_when_not_injected(self, tmp_path: Path) -> None:
        repo = RepoConfig(name="owner/repo", work_dir=tmp_path)
        factory = MagicMock()
        factory.create_api.return_value.get_limit_snapshot.return_value = (
            ProviderLimitSnapshot(provider=ProviderID.CLAUDE_CODE)
        )
        with patch("fido.status.DefaultProviderFactory", return_value=factory):
            statuses = provider_statuses_for_repo_configs([repo])
        assert list(statuses) == [ProviderID.CLAUDE_CODE]

    def test_collects_codex_provider_status(self, tmp_path: Path) -> None:
        repo = RepoConfig(
            name="owner/repo", work_dir=tmp_path, provider=ProviderID.CODEX
        )
        factory = MagicMock()
        api = MagicMock()
        api.get_limit_snapshot.return_value = ProviderLimitSnapshot(
            provider=ProviderID.CODEX,
            windows=(ProviderLimitWindow(name="codex_primary", used=25, limit=100),),
        )
        factory.create_api.return_value = api

        statuses = provider_statuses_for_repo_configs([repo], _provider_factory=factory)

        assert statuses[ProviderID.CODEX].provider == ProviderID.CODEX
        assert statuses[ProviderID.CODEX].percent_used == 25
        factory.create_api.assert_called_once_with(repo)


class TestProcessUptimeSeconds:
    def test_returns_elapsed_time(self) -> None:
        mock_run = MagicMock(return_value=MagicMock(stdout="  3742  "))
        assert _process_uptime_seconds(12345, _run=mock_run) == 3742

    def test_none_when_no_output(self) -> None:
        mock_run = MagicMock(return_value=MagicMock(stdout=""))
        assert _process_uptime_seconds(99, _run=mock_run) is None

    def test_none_on_oserror(self) -> None:
        mock_run = MagicMock(side_effect=OSError)
        assert _process_uptime_seconds(99, _run=mock_run) is None

    def test_none_on_value_error(self) -> None:
        mock_run = MagicMock(return_value=MagicMock(stdout="not-a-number"))
        assert _process_uptime_seconds(99, _run=mock_run) is None

    def test_calls_ps_with_pid(self) -> None:
        mock_run = MagicMock(return_value=MagicMock(stdout="100"))
        _process_uptime_seconds(42, _run=mock_run)
        mock_run.assert_called_once_with(
            ["ps", "-p", "42", "-o", "etimes="],
            capture_output=True,
            text=True,
        )


class TestReposFromPid:
    def test_parses_single_repo(self) -> None:
        cmdline = b"fido\x00--port\x009000\x00rhencke/confusio:/workspace/confusio:claude-code\x00"
        with patch.object(Path, "read_bytes", return_value=cmdline):
            result = _repos_from_pid(123)
        assert len(result) == 1
        assert result[0].name == "rhencke/confusio"
        assert result[0].work_dir == Path("/workspace/confusio")

    def test_parses_multiple_repos(self) -> None:
        cmdline = (
            b"fido\x00rhencke/a:/path/a:claude-code\x00"
            b"rhencke/b:/path/b:copilot-cli\x00"
        )
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
        cmdline = b"fido\x00--port\x009000\x00"
        with patch.object(Path, "read_bytes", return_value=cmdline):
            result = _repos_from_pid(123)
        assert result == []

    def test_skips_colon_args_without_slash_in_name(self) -> None:
        cmdline = b"something:value:claude-code\x00"
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
        cmdline = b"rhencke/repo:~/workspace/repo:claude-code\x00"
        with patch.object(Path, "read_bytes", return_value=cmdline):
            result = _repos_from_pid(1)
        assert result[0].work_dir == Path("~/workspace/repo").expanduser()

    def test_no_repos_returns_empty(self) -> None:
        cmdline = (
            b"fido\x00--port\x009000\x00--secret-file\x00/home/user/.fido-secret\x00"
        )
        with patch.object(Path, "read_bytes", return_value=cmdline):
            result = _repos_from_pid(1)
        assert result == []

    def test_skips_non_utf8_args(self) -> None:
        cmdline = b"\xff\xfe\x00rhencke/repo:/path:claude-code\x00"
        with patch.object(Path, "read_bytes", return_value=cmdline):
            result = _repos_from_pid(1)
        assert len(result) == 1
        assert result[0].name == "rhencke/repo"


class TestFidoPid:
    def test_returns_first_pid(self) -> None:
        with patch("fido.status._pgrep", return_value=[111, 222]):
            result = _fido_pid()
        assert result == 111

    def test_returns_none_when_no_match(self) -> None:
        with patch("fido.status._pgrep", return_value=[]):
            result = _fido_pid()
        assert result is None

    def test_searches_for_fido_port(self) -> None:
        with patch("fido.status._pgrep", return_value=[]) as mock:
            _fido_pid()
        mock.assert_called_once_with("fido --port")


class TestPortFromPid:
    def test_returns_port(self) -> None:
        cmdline = b"fido\x00--port\x009000\x00rhencke/repo:/path\x00"
        with patch.object(Path, "read_bytes", return_value=cmdline):
            assert _port_from_pid(42) == 9000

    def test_returns_none_when_no_port_flag(self) -> None:
        cmdline = b"fido\x00rhencke/repo:/path\x00"
        with patch.object(Path, "read_bytes", return_value=cmdline):
            assert _port_from_pid(42) is None

    def test_returns_none_when_port_flag_last_arg(self) -> None:
        cmdline = b"fido\x00--port\x00"
        with patch.object(Path, "read_bytes", return_value=cmdline):
            assert _port_from_pid(42) is None

    def test_returns_none_when_port_value_not_integer(self) -> None:
        cmdline = b"fido\x00--port\x00notanumber\x00"
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


class TestTaskPosition:
    def test_task_in_progress_takes_precedence(self) -> None:
        from fido.status import _task_position

        tasks = [
            {"status": "pending"},
            {"status": "in_progress"},
            {"status": "pending"},
        ]
        assert _task_position(tasks) == (2, 3)

    def test_first_pending_when_none_in_progress(self) -> None:
        from fido.status import _task_position

        tasks = [{"status": "pending"}, {"status": "pending"}]
        assert _task_position(tasks) == (1, 2)

    def test_empty_when_all_completed(self) -> None:
        from fido.status import _task_position

        tasks = [{"status": "completed"}, {"status": "completed"}]
        assert _task_position(tasks) == (None, None)

    def test_counts_up_past_completed_tasks(self) -> None:
        from fido.status import _task_position

        # Completed tasks do not offset the active queue position.
        tasks = [
            {"status": "completed"},
            {"status": "completed"},
            {"status": "in_progress"},
            {"status": "pending"},
        ]
        assert _task_position(tasks) == (1, 2)

    def test_pending_offsets_past_completed(self) -> None:
        from fido.status import _task_position

        # Completed tasks do not count toward the current queue length.
        tasks = [
            {"status": "completed"},
            {"status": "completed"},
            {"status": "completed"},
            {"status": "pending"},
            {"status": "pending"},
        ]
        assert _task_position(tasks) == (1, 2)


class TestElapsedSinceIso:
    def test_none_on_empty(self) -> None:
        from fido.status import _elapsed_since_iso

        assert _elapsed_since_iso(None) is None
        assert _elapsed_since_iso("") is None

    def test_none_on_bad_format(self) -> None:
        from fido.status import _elapsed_since_iso

        assert _elapsed_since_iso("not a date") is None

    def test_none_on_wrong_type(self) -> None:
        from fido.status import _elapsed_since_iso

        # datetime.fromisoformat raises TypeError for non-str input.
        assert _elapsed_since_iso(12345) is None  # type: ignore[arg-type]

    def test_returns_seconds_since(self) -> None:
        from datetime import datetime, timedelta, timezone

        from fido.status import _elapsed_since_iso

        ref = datetime(2026, 1, 1, tzinfo=timezone.utc)
        iso = (ref - timedelta(minutes=5)).isoformat()
        assert _elapsed_since_iso(iso, _now=lambda: ref) == 300

    def test_floors_at_zero_for_future_timestamps(self) -> None:
        from datetime import datetime, timedelta, timezone

        from fido.status import _elapsed_since_iso

        ref = datetime(2026, 1, 1, tzinfo=timezone.utc)
        future = (ref + timedelta(minutes=1)).isoformat()
        assert _elapsed_since_iso(future, _now=lambda: ref) == 0


class TestSystemResources:
    def test_collects_system_resources(self, tmp_path: Path) -> None:
        meminfo = tmp_path / "meminfo"
        meminfo.write_text("MemTotal:       1048576 kB\nMemAvailable:    262144 kB\n")
        loadavg = tmp_path / "loadavg"
        loadavg.write_text("1.25 0.50 0.25 1/100 123\n")

        result = _system_resources(
            meminfo_path=meminfo,
            loadavg_path=loadavg,
            disk_path=tmp_path,
            _disk_usage=lambda _path: SimpleNamespace(
                total=1024**3, free=256 * 1024**2
            ),
            _cpu_count=lambda: 4,
        )

        assert result == SystemResourceInfo(
            load_1=1.25,
            load_5=0.50,
            load_15=0.25,
            cpu_count=4,
            mem_total_bytes=1048576 * 1024,
            mem_available_bytes=262144 * 1024,
            disk_path=str(tmp_path),
            disk_total_bytes=1024**3,
            disk_free_bytes=256 * 1024**2,
        )

    def test_collect_returns_none_for_unreadable_resources(
        self, tmp_path: Path
    ) -> None:
        assert (
            _system_resources(
                meminfo_path=tmp_path / "missing",
                loadavg_path=tmp_path / "loadavg",
            )
            is None
        )

    def test_formats_resource_line(self) -> None:
        resources = SystemResourceInfo(
            load_1=1.25,
            load_5=0.50,
            load_15=0.25,
            cpu_count=4,
            mem_total_bytes=8 * 1024**3,
            mem_available_bytes=2 * 1024**3,
            disk_path="/",
            disk_total_bytes=100 * 1024**3,
            disk_free_bytes=25 * 1024**3,
        )

        assert _format_gib(1536 * 1024**2) == "1.5GiB"
        assert _format_percent(1, 0) == "n/a"
        assert _format_resource_line(resources) == (
            "host: cpu load 1.25/4 (0.50, 0.25), "
            "mem 6.0GiB/8.0GiB (75%), disk / 75.0GiB/100.0GiB (75%)"
        )
        assert _format_resource_line(None) is None


class TestCollectWebhookPropagation:
    """collect() forwards worker_uptime + webhook_activities into repo_status."""

    def test_worker_uptime_and_webhooks_forwarded(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        rc = RepoConfig(name="owner/repo", work_dir=tmp_path)
        activity = {
            "what": "Working on: #1",
            "crash_count": 0,
            "last_crash_error": None,
            "is_stuck": False,
            "worker_uptime_seconds": 120,
            "webhook_activities": [
                {"description": "triaging", "elapsed_seconds": 5.0},
            ],
        }
        fake_status = RepoStatus(
            name="owner/repo",
            fido_running=False,
            issue=None,
            pending=0,
            completed=0,
            current_task=None,
            claude_pid=None,
            claude_uptime=None,
            worker_what=None,
            crash_count=0,
            last_crash_error=None,
            worker_stuck=False,
        )
        with (
            patch("fido.status._fido_pid", return_value=42),
            patch("fido.status._repos_from_pid", return_value=[rc]),
            patch("fido.status._process_uptime_seconds", return_value=0),
            patch("fido.status._port_from_pid", return_value=9000),
            patch(
                "fido.status._fetch_activities",
                return_value=({"owner/repo": activity}, None),
            ),
            patch("fido.status.repo_status", return_value=fake_status) as mock_rs,
        ):
            collect()
        kwargs = mock_rs.call_args.kwargs
        assert kwargs["worker_uptime"] == 120
        assert len(kwargs["webhook_activities"]) == 1
        assert kwargs["webhook_activities"][0].description == "triaging"
        assert kwargs["webhook_activities"][0].elapsed_seconds == 5


class TestFetchActivities:
    def _make_urlopen(self, data: bytes) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = data
        return MagicMock(return_value=mock_resp)

    def _wrap(self, activities: list) -> bytes:
        """Wrap an activities list in the new ``/status.json`` envelope."""
        return json.dumps({"activities": activities, "rate_limit": None}).encode()

    def test_returns_repo_what_map(self) -> None:
        data = json.dumps(
            {
                "activities": [
                    {
                        "repo_name": "owner/repo",
                        "what": "Working on: #1",
                        "busy": True,
                        "crash_count": 0,
                        "last_crash_error": None,
                        "is_stuck": False,
                    }
                ],
                "rate_limit": None,
            }
        ).encode()
        result = _fetch_activities(9000, _urlopen=self._make_urlopen(data))
        assert result == (
            {
                "owner/repo": {
                    "what": "Working on: #1",
                    "crash_count": 0,
                    "last_crash_error": None,
                    "is_stuck": False,
                    "worker_uptime_seconds": None,
                    "webhook_activities": [],
                    "session_owner": None,
                    "session_alive": False,
                    "session_pid": None,
                    "session_dropped_count": 0,
                    "session_sent_count": 0,
                    "session_received_count": 0,
                    "claude_talker": None,
                    "provider_status": None,
                    "rescoping": False,
                    "issue_cache": None,
                }
            },
            None,
        )

    def test_returns_multiple_repos(self) -> None:
        data = json.dumps(
            {
                "activities": [
                    {
                        "repo_name": "a/b",
                        "what": "Napping",
                        "busy": False,
                        "crash_count": 0,
                        "last_crash_error": None,
                        "is_stuck": False,
                    },
                    {
                        "repo_name": "c/d",
                        "what": "Fixing CI",
                        "busy": True,
                        "crash_count": 2,
                        "last_crash_error": "RuntimeError: boom",
                        "is_stuck": True,
                    },
                ],
                "rate_limit": None,
            }
        ).encode()
        result = _fetch_activities(9000, _urlopen=self._make_urlopen(data))
        assert result == (
            {
                "a/b": {
                    "what": "Napping",
                    "crash_count": 0,
                    "last_crash_error": None,
                    "is_stuck": False,
                    "worker_uptime_seconds": None,
                    "webhook_activities": [],
                    "session_owner": None,
                    "session_alive": False,
                    "session_pid": None,
                    "session_dropped_count": 0,
                    "session_sent_count": 0,
                    "session_received_count": 0,
                    "claude_talker": None,
                    "provider_status": None,
                    "rescoping": False,
                    "issue_cache": None,
                },
                "c/d": {
                    "what": "Fixing CI",
                    "crash_count": 2,
                    "last_crash_error": "RuntimeError: boom",
                    "is_stuck": True,
                    "worker_uptime_seconds": None,
                    "webhook_activities": [],
                    "session_owner": None,
                    "session_alive": False,
                    "session_pid": None,
                    "session_dropped_count": 0,
                    "session_sent_count": 0,
                    "session_received_count": 0,
                    "claude_talker": None,
                    "provider_status": None,
                    "rescoping": False,
                    "issue_cache": None,
                },
            },
            None,
        )

    def test_parses_provider_status_from_json(self) -> None:
        data = json.dumps(
            {
                "activities": [
                    {
                        "repo_name": "owner/repo",
                        "what": "Working on: #1",
                        "busy": True,
                        "crash_count": 0,
                        "last_crash_error": None,
                        "is_stuck": False,
                        "provider_status": {
                            "provider": "claude-code",
                            "window_name": "five_hour",
                            "pressure": 0.96,
                            "percent_used": 96,
                            "resets_at": "2026-04-16T07:00:00+00:00",
                            "unavailable_reason": None,
                            "level": "paused",
                            "warning": False,
                            "paused": True,
                        },
                    }
                ],
                "rate_limit": None,
            }
        ).encode()
        result = _fetch_activities(9000, _urlopen=self._make_urlopen(data))
        assert result[0]["owner/repo"]["provider_status"] == ProviderPressureStatus(
            provider=ProviderID.CLAUDE_CODE,
            window_name="five_hour",
            pressure=0.96,
            resets_at=datetime(2026, 4, 16, 7, 0, tzinfo=UTC),
        )

    def test_invalid_provider_status_defaults_to_none(self) -> None:
        data = json.dumps(
            {
                "activities": [
                    {
                        "repo_name": "owner/repo",
                        "what": "Working on: #1",
                        "busy": True,
                        "crash_count": 0,
                        "last_crash_error": None,
                        "is_stuck": False,
                        "provider_status": {"provider": "nope"},
                    }
                ],
                "rate_limit": None,
            }
        ).encode()
        result = _fetch_activities(9000, _urlopen=self._make_urlopen(data))
        assert result[0]["owner/repo"]["provider_status"] is None

    def test_bad_provider_status_reset_time_defaults_to_none(self) -> None:
        data = json.dumps(
            {
                "activities": [
                    {
                        "repo_name": "owner/repo",
                        "what": "Working on: #1",
                        "busy": True,
                        "crash_count": 0,
                        "last_crash_error": None,
                        "is_stuck": False,
                        "provider_status": {
                            "provider": "claude-code",
                            "pressure": 0.5,
                            "resets_at": "not-a-date",
                        },
                    }
                ],
                "rate_limit": None,
            }
        ).encode()
        result = _fetch_activities(9000, _urlopen=self._make_urlopen(data))
        assert result[0]["owner/repo"]["provider_status"] == ProviderPressureStatus(
            provider=ProviderID.CLAUDE_CODE,
            pressure=0.5,
        )

    def test_returns_empty_on_exception(self) -> None:
        mock_urlopen = MagicMock(side_effect=OSError("refused"))
        assert _fetch_activities(9000, _urlopen=mock_urlopen) == ({}, None)

    def test_skips_items_without_repo_name(self) -> None:
        data = json.dumps(
            {"activities": [{"what": "something", "busy": True}], "rate_limit": None}
        ).encode()
        result = _fetch_activities(9000, _urlopen=self._make_urlopen(data))
        assert result == ({}, None)

    def test_skips_items_without_what(self) -> None:
        data = json.dumps(
            {"activities": [{"repo_name": "a/b", "busy": True}], "rate_limit": None}
        ).encode()
        result = _fetch_activities(9000, _urlopen=self._make_urlopen(data))
        assert result == ({}, None)

    def test_is_stuck_defaults_to_false_when_absent(self) -> None:
        data = json.dumps(
            {
                "activities": [
                    {
                        "repo_name": "owner/repo",
                        "what": "Napping",
                        "busy": False,
                        "crash_count": 0,
                        "last_crash_error": None,
                        # no "is_stuck" key — older server version
                    }
                ],
                "rate_limit": None,
            }
        ).encode()
        result = _fetch_activities(9000, _urlopen=self._make_urlopen(data))
        assert result[0]["owner/repo"]["is_stuck"] is False

    def test_calls_correct_url(self) -> None:
        mock_urlopen = self._make_urlopen(b'{"activities": [], "rate_limit": null}')
        _fetch_activities(8888, _urlopen=mock_urlopen)
        mock_urlopen.assert_called_once_with(
            "http://localhost:8888/status.json", timeout=2
        )


class TestClaudePid:
    def test_returns_first_pid(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        with patch("fido.status._pgrep", return_value=[999]):
            result = _claude_pid(fido_dir)
        assert result == 999

    def test_returns_none_when_no_match_and_no_state(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        with patch("fido.status._pgrep", return_value=[]):
            result = _claude_pid(fido_dir)
        assert result is None

    def test_searches_for_system_file_first(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        with patch("fido.status._pgrep", return_value=[]) as mock:
            _claude_pid(fido_dir)
        mock.assert_any_call(str(fido_dir / "system"))

    def test_falls_back_to_resumed_session_id(self, tmp_path: Path) -> None:
        """Resumed sessions run with --resume <id> and don't reference the
        system file — fall back to matching the session id from state.json."""
        from fido.state import State

        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        State(fido_dir).save({"setup_session_id": "abc-123", "issue": 7})

        def fake_pgrep(pattern: str) -> list[int]:
            return [42] if pattern == "abc-123" else []

        with patch("fido.status._pgrep", side_effect=fake_pgrep):
            result = _claude_pid(fido_dir)
        assert result == 42

    def test_resumed_fallback_returns_none_when_no_process(
        self, tmp_path: Path
    ) -> None:
        from fido.state import State

        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        State(fido_dir).save({"setup_session_id": "abc-123"})
        with patch("fido.status._pgrep", return_value=[]):
            result = _claude_pid(fido_dir)
        assert result is None

    def test_resumed_fallback_skipped_when_no_session_id(self, tmp_path: Path) -> None:
        from fido.state import State

        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        State(fido_dir).save({"issue": 7})
        with patch("fido.status._pgrep", return_value=[]) as mock:
            result = _claude_pid(fido_dir)
        assert result is None
        # Only the system-file pgrep fires when there's no session id.
        assert mock.call_count == 1

    def test_resumed_fallback_skipped_when_state_absent(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        with patch("fido.status._pgrep", return_value=[]):
            result = _claude_pid(fido_dir)
        assert result is None


class TestGitDir:
    def test_returns_path(self, tmp_path: Path) -> None:
        mock_run = MagicMock(return_value=MagicMock(stdout="/repo/.git\n"))
        assert _git_dir(tmp_path, _run=mock_run) == Path("/repo/.git")

    def test_strips_whitespace(self, tmp_path: Path) -> None:
        mock_run = MagicMock(return_value=MagicMock(stdout="  /a/b/.git  \n"))
        assert _git_dir(tmp_path, _run=mock_run) == Path("/a/b/.git")

    def test_returns_none_on_called_process_error(self, tmp_path: Path) -> None:
        mock_run = MagicMock(side_effect=subprocess.CalledProcessError(128, "git"))
        assert _git_dir(tmp_path, _run=mock_run) is None

    def test_returns_none_when_git_not_found(self, tmp_path: Path) -> None:
        mock_run = MagicMock(side_effect=FileNotFoundError("git not found"))
        assert _git_dir(tmp_path, _run=mock_run) is None


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
        with patch("fido.status._git_dir", return_value=None):
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
        assert result.crash_count == 0
        assert result.last_crash_error is None

    def test_no_git_dir_passes_worker_what(self, tmp_path: Path) -> None:
        cfg = self._make_config(tmp_path)
        with patch("fido.status._git_dir", return_value=None):
            result = repo_status(cfg, worker_what="Napping")
        assert result.worker_what == "Napping"

    def test_crash_count_defaults_to_zero(self, tmp_path: Path) -> None:
        cfg = self._make_config(tmp_path)
        with patch("fido.status._git_dir", return_value=None):
            result = repo_status(cfg)
        assert result.crash_count == 0

    def test_crash_fields_passed_through(self, tmp_path: Path) -> None:
        cfg = self._make_config(tmp_path)
        with patch("fido.status._git_dir", return_value=None):
            result = repo_status(
                cfg, crash_count=5, last_crash_error="ValueError: oops"
            )
        assert result.crash_count == 5
        assert result.last_crash_error == "ValueError: oops"

    def test_with_running_fido_and_issue(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        fido_dir = git_dir / "fido"
        fido_dir.mkdir(parents=True)
        (fido_dir / "state.json").write_text('{"issue": 7}')
        tasks = [
            {"status": "completed", "title": "done task", "type": "spec"},
            {"status": "pending", "title": "next task", "type": "spec"},
        ]
        (fido_dir / "tasks.json").write_text(json.dumps(tasks))

        cfg = self._make_config(tmp_path)
        with (
            patch("fido.status._git_dir", return_value=git_dir),
            patch("fido.status._fido_running", return_value=True),
            patch("fido.status._claude_pid", return_value=555),
            patch("fido.status._process_uptime_seconds", return_value=180),
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
            patch("fido.status._git_dir", return_value=git_dir),
            patch("fido.status._fido_running", return_value=False),
            patch("fido.status._claude_pid", return_value=None),
            patch("fido.status._process_uptime_seconds", return_value=None),
        ):
            result = repo_status(cfg)
        assert result.worker_what is None

    def test_no_claude_pid_skips_uptime(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        fido_dir = git_dir / "fido"
        fido_dir.mkdir(parents=True)

        cfg = self._make_config(tmp_path)
        with (
            patch("fido.status._git_dir", return_value=git_dir),
            patch("fido.status._fido_running", return_value=False),
            patch("fido.status._claude_pid", return_value=None),
            patch("fido.status._process_uptime_seconds") as mock_uptime,
        ):
            result = repo_status(cfg)

        mock_uptime.assert_not_called()
        assert result.claude_pid is None
        assert result.claude_uptime is None

    def test_worker_stuck_defaults_to_false(self, tmp_path: Path) -> None:
        cfg = self._make_config(tmp_path)
        with patch("fido.status._git_dir", return_value=None):
            result = repo_status(cfg)
        assert result.worker_stuck is False

    def test_worker_stuck_passed_through_no_git_dir(self, tmp_path: Path) -> None:
        cfg = self._make_config(tmp_path)
        with patch("fido.status._git_dir", return_value=None):
            result = repo_status(cfg, worker_stuck=True)
        assert result.worker_stuck is True

    def test_worker_stuck_passed_through_with_git_dir(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        fido_dir = git_dir / "fido"
        fido_dir.mkdir(parents=True)
        cfg = self._make_config(tmp_path)
        with (
            patch("fido.status._git_dir", return_value=git_dir),
            patch("fido.status._fido_running", return_value=False),
            patch("fido.status._claude_pid", return_value=None),
            patch("fido.status._process_uptime_seconds", return_value=None),
        ):
            result = repo_status(cfg, worker_stuck=True)
        assert result.worker_stuck is True

    def test_provider_status_passed_through(self, tmp_path: Path) -> None:
        cfg = self._make_config(tmp_path)
        status = ProviderPressureStatus(provider=ProviderID.CLAUDE_CODE, pressure=0.95)
        with patch("fido.status._git_dir", return_value=None):
            result = repo_status(cfg, provider_status=status)
        assert result.provider is ProviderID.CLAUDE_CODE
        assert result.provider_status == status

    def test_in_progress_task_propagates_position(self, tmp_path: Path) -> None:
        """repo_status() surfaces the in_progress task's position and total correctly.

        When the worker marks a task in_progress, _task_position() picks it up
        and the position is reflected in task_number / task_total on RepoStatus.
        An in_progress task that is not first in the non-completed list must
        still report its actual position, not 1.
        """
        git_dir = tmp_path / ".git"
        fido_dir = git_dir / "fido"
        fido_dir.mkdir(parents=True)
        tasks = [
            {"status": "completed", "title": "done", "type": "spec"},
            {"status": "pending", "title": "first pending", "type": "spec"},
            {"status": "in_progress", "title": "active task", "type": "spec"},
            {"status": "pending", "title": "last pending", "type": "spec"},
        ]
        (fido_dir / "tasks.json").write_text(json.dumps(tasks))

        cfg = self._make_config(tmp_path)
        with (
            patch("fido.status._git_dir", return_value=git_dir),
            patch("fido.status._fido_running", return_value=True),
            patch("fido.status._claude_pid", return_value=None),
            patch("fido.status._process_uptime_seconds", return_value=None),
        ):
            result = repo_status(cfg)

        # Non-completed: [pending, in_progress, pending] → 3 total; in_progress is #2.
        assert result.current_task == "active task"
        assert result.task_number == 2
        assert result.task_total == 3


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
            crash_count=0,
            last_crash_error=None,
            worker_stuck=False,
        )

    def test_fido_up_with_uptime(self, tmp_path: Path) -> None:
        rc = RepoConfig(name="owner/repo", work_dir=tmp_path)
        with (
            patch("fido.status._fido_pid", return_value=42),
            patch("fido.status._repos_from_pid", return_value=[rc]),
            patch("fido.status._process_uptime_seconds", return_value=600),
            patch("fido.status._port_from_pid", return_value=None),
            patch("fido.status.repo_status", return_value=self._fake_repo_status()),
        ):
            result = collect()

        assert result.fido_pid == 42
        assert result.fido_uptime == 600
        assert len(result.repos) == 1
        assert result.provider_statuses == []

    def test_fido_down(self) -> None:
        with (
            patch("fido.status._fido_pid", return_value=None),
            patch("fido.status._repos_from_pid") as mock_repos,
            patch("fido.status._process_uptime_seconds") as mock_uptime,
            patch("fido.status._port_from_pid") as mock_port,
            patch("fido.status.repo_status") as mock_repo_status,
        ):
            result = collect()

        mock_uptime.assert_not_called()
        mock_repos.assert_not_called()
        mock_repo_status.assert_not_called()
        mock_port.assert_not_called()
        assert result.fido_pid is None
        assert result.fido_uptime is None

    def test_passes_provider_status_to_repo_status(self, tmp_path: Path) -> None:
        rc = RepoConfig(name="owner/repo", work_dir=tmp_path)
        provider_status = ProviderPressureStatus(
            provider=ProviderID.CLAUDE_CODE,
            pressure=0.91,
        )
        with (
            patch("fido.status._fido_pid", return_value=42),
            patch("fido.status._repos_from_pid", return_value=[rc]),
            patch("fido.status._process_uptime_seconds", return_value=600),
            patch("fido.status._port_from_pid", return_value=9000),
            patch(
                "fido.status._fetch_activities",
                return_value=(
                    {
                        "owner/repo": self._activity_info(
                            provider_status=provider_status
                        )
                    },
                    None,
                ),
            ),
            patch(
                "fido.status.repo_status", return_value=self._fake_repo_status()
            ) as mock_repo,
        ):
            result = collect()

        assert result.provider_statuses == [provider_status]
        assert mock_repo.call_args.kwargs["provider_status"] == provider_status

    def _activity_info(
        self,
        what: str = "Working on: #1",
        crash_count: int = 0,
        last_crash_error: str | None = None,
        is_stuck: bool = False,
        worker_uptime_seconds: int | None = None,
        webhook_activities: list | None = None,
        session_owner: str | None = None,
        rescoping: bool = False,
        provider_status: ProviderPressureStatus | None = None,
    ) -> dict:
        return {
            "what": what,
            "crash_count": crash_count,
            "last_crash_error": last_crash_error,
            "is_stuck": is_stuck,
            "worker_uptime_seconds": worker_uptime_seconds,
            "webhook_activities": webhook_activities or [],
            "session_owner": session_owner,
            "rescoping": rescoping,
            "provider_status": provider_status,
        }

    def test_fetches_activities_when_port_known(self, tmp_path: Path) -> None:
        rc = RepoConfig(name="owner/repo", work_dir=tmp_path)
        with (
            patch("fido.status._fido_pid", return_value=42),
            patch("fido.status._repos_from_pid", return_value=[rc]),
            patch("fido.status._process_uptime_seconds", return_value=0),
            patch("fido.status._port_from_pid", return_value=9000),
            patch(
                "fido.status._fetch_activities",
                return_value=({"owner/repo": self._activity_info()}, None),
            ) as mock_fetch,
            patch("fido.status.repo_status", return_value=self._fake_repo_status()),
        ):
            collect()
        mock_fetch.assert_called_once_with(9000)

    def test_skips_fetch_when_port_unknown(self, tmp_path: Path) -> None:
        rc = RepoConfig(name="owner/repo", work_dir=tmp_path)
        with (
            patch("fido.status._fido_pid", return_value=42),
            patch("fido.status._repos_from_pid", return_value=[rc]),
            patch("fido.status._process_uptime_seconds", return_value=0),
            patch("fido.status._port_from_pid", return_value=None),
            patch("fido.status._fetch_activities") as mock_fetch,
            patch("fido.status.repo_status", return_value=self._fake_repo_status()),
        ):
            collect()
        mock_fetch.assert_not_called()

    def test_passes_worker_what_to_repo_status(self, tmp_path: Path) -> None:
        rc = RepoConfig(name="owner/repo", work_dir=tmp_path)
        with (
            patch("fido.status._fido_pid", return_value=42),
            patch("fido.status._repos_from_pid", return_value=[rc]),
            patch("fido.status._process_uptime_seconds", return_value=0),
            patch("fido.status._port_from_pid", return_value=9000),
            patch(
                "fido.status._fetch_activities",
                return_value=(
                    {"owner/repo": self._activity_info("Fixing CI: tests")},
                    None,
                ),
            ),
            patch(
                "fido.status.repo_status", return_value=self._fake_repo_status()
            ) as mock_rs,
        ):
            collect()
        mock_rs.assert_called_once_with(
            rc,
            worker_what="Fixing CI: tests",
            crash_count=0,
            last_crash_error=None,
            worker_stuck=False,
            worker_uptime=None,
            webhook_activities=[],
            provider_status=None,
            session_owner=None,
            session_alive=False,
            session_pid=None,
            session_dropped_count=0,
            session_sent_count=0,
            session_received_count=0,
            claude_talker=None,
            rescoping=False,
            issue_cache=None,
        )

    def test_passes_crash_info_to_repo_status(self, tmp_path: Path) -> None:
        rc = RepoConfig(name="owner/repo", work_dir=tmp_path)
        with (
            patch("fido.status._fido_pid", return_value=42),
            patch("fido.status._repos_from_pid", return_value=[rc]),
            patch("fido.status._process_uptime_seconds", return_value=0),
            patch("fido.status._port_from_pid", return_value=9000),
            patch(
                "fido.status._fetch_activities",
                return_value=(
                    {
                        "owner/repo": self._activity_info(
                            "Napping",
                            crash_count=3,
                            last_crash_error="ValueError: oops",
                        )
                    },
                    None,
                ),
            ),
            patch(
                "fido.status.repo_status", return_value=self._fake_repo_status()
            ) as mock_rs,
        ):
            collect()
        mock_rs.assert_called_once_with(
            rc,
            worker_what="Napping",
            crash_count=3,
            last_crash_error="ValueError: oops",
            worker_stuck=False,
            worker_uptime=None,
            webhook_activities=[],
            provider_status=None,
            session_owner=None,
            session_alive=False,
            session_pid=None,
            session_dropped_count=0,
            session_sent_count=0,
            session_received_count=0,
            claude_talker=None,
            rescoping=False,
            issue_cache=None,
        )

    def test_worker_what_none_for_unknown_repo(self, tmp_path: Path) -> None:
        rc = RepoConfig(name="owner/repo", work_dir=tmp_path)
        with (
            patch("fido.status._fido_pid", return_value=42),
            patch("fido.status._repos_from_pid", return_value=[rc]),
            patch("fido.status._process_uptime_seconds", return_value=0),
            patch("fido.status._port_from_pid", return_value=9000),
            patch("fido.status._fetch_activities", return_value=({}, None)),
            patch(
                "fido.status.repo_status", return_value=self._fake_repo_status()
            ) as mock_rs,
        ):
            collect()
        mock_rs.assert_called_once_with(
            rc,
            worker_what=None,
            crash_count=0,
            last_crash_error=None,
            worker_stuck=False,
            worker_uptime=None,
            webhook_activities=[],
            provider_status=None,
            session_owner=None,
            session_alive=False,
            session_pid=None,
            session_dropped_count=0,
            session_sent_count=0,
            session_received_count=0,
            claude_talker=None,
            rescoping=False,
            issue_cache=None,
        )

    def test_passes_claude_talker_to_repo_status(self, tmp_path: Path) -> None:
        """An active SessionTalker in /status → ClaudeTalkerInfo on RepoStatus."""
        rc = RepoConfig(name="owner/repo", work_dir=tmp_path)
        activity_info = {
            "what": "running",
            "crash_count": 0,
            "last_crash_error": None,
            "is_stuck": False,
            "worker_uptime_seconds": None,
            "webhook_activities": [],
            "session_owner": None,
            "session_alive": True,
            "session_pid": 42,
            "claude_talker": {
                "repo_name": "owner/repo",
                "thread_id": 1234,
                "kind": "worker",
                "description": "persistent session turn",
                "claude_pid": 42,
                "started_at": "2026-04-14T18:00:00+00:00",
            },
        }
        with (
            patch("fido.status._fido_pid", return_value=42),
            patch("fido.status._repos_from_pid", return_value=[rc]),
            patch("fido.status._process_uptime_seconds", return_value=0),
            patch("fido.status._port_from_pid", return_value=9000),
            patch(
                "fido.status._fetch_activities",
                return_value=({"owner/repo": activity_info}, None),
            ),
            patch(
                "fido.status.repo_status", return_value=self._fake_repo_status()
            ) as mock_rs,
        ):
            collect()
        kwargs = mock_rs.call_args.kwargs
        assert kwargs["claude_talker"] == ClaudeTalkerInfo(
            thread_id=1234,
            kind="worker",
            description="persistent session turn",
            claude_pid=42,
        )
        assert kwargs["session_pid"] == 42

    def test_passes_is_stuck_to_repo_status(self, tmp_path: Path) -> None:
        rc = RepoConfig(name="owner/repo", work_dir=tmp_path)
        with (
            patch("fido.status._fido_pid", return_value=42),
            patch("fido.status._repos_from_pid", return_value=[rc]),
            patch("fido.status._process_uptime_seconds", return_value=0),
            patch("fido.status._port_from_pid", return_value=9000),
            patch(
                "fido.status._fetch_activities",
                return_value=({"owner/repo": self._activity_info(is_stuck=True)}, None),
            ),
            patch(
                "fido.status.repo_status", return_value=self._fake_repo_status()
            ) as mock_rs,
        ):
            collect()
        mock_rs.assert_called_once_with(
            rc,
            worker_what="Working on: #1",
            crash_count=0,
            last_crash_error=None,
            worker_stuck=True,
            worker_uptime=None,
            webhook_activities=[],
            provider_status=None,
            session_owner=None,
            session_alive=False,
            session_pid=None,
            session_dropped_count=0,
            session_sent_count=0,
            session_received_count=0,
            claude_talker=None,
            rescoping=False,
            issue_cache=None,
        )

    def test_passes_rescoping_true_to_repo_status(self, tmp_path: Path) -> None:
        rc = RepoConfig(name="owner/repo", work_dir=tmp_path)
        with (
            patch("fido.status._fido_pid", return_value=42),
            patch("fido.status._repos_from_pid", return_value=[rc]),
            patch("fido.status._process_uptime_seconds", return_value=0),
            patch("fido.status._port_from_pid", return_value=9000),
            patch(
                "fido.status._fetch_activities",
                return_value=(
                    {"owner/repo": self._activity_info(rescoping=True)},
                    None,
                ),
            ),
            patch(
                "fido.status.repo_status", return_value=self._fake_repo_status()
            ) as mock_rs,
        ):
            collect()
        kwargs = mock_rs.call_args.kwargs
        assert kwargs["rescoping"] is True

    def test_rescoping_false_when_no_activity_info(self, tmp_path: Path) -> None:
        """Repos with no activity entry (unknown to the live server) get rescoping=False."""
        rc = RepoConfig(name="owner/repo", work_dir=tmp_path)
        with (
            patch("fido.status._fido_pid", return_value=42),
            patch("fido.status._repos_from_pid", return_value=[rc]),
            patch("fido.status._process_uptime_seconds", return_value=0),
            patch("fido.status._port_from_pid", return_value=9000),
            patch("fido.status._fetch_activities", return_value=({}, None)),
            patch(
                "fido.status.repo_status", return_value=self._fake_repo_status()
            ) as mock_rs,
        ):
            collect()
        kwargs = mock_rs.call_args.kwargs
        assert kwargs["rescoping"] is False


class TestFormatAgentLine:
    """Unit tests for _format_agent_line — the dedicated per-repo agent body line."""

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
            crash_count=0,
            last_crash_error=None,
            worker_stuck=False,
        )
        defaults.update(kwargs)
        return RepoStatus(**defaults)

    def test_returns_none_when_no_pid_and_session_not_alive(self) -> None:
        repo = self._repo(claude_pid=None, session_alive=False)
        assert _format_agent_line(repo) is None

    def test_returns_line_when_pid_present(self) -> None:
        repo = self._repo(claude_pid=1234)
        line = _format_agent_line(repo)
        assert line is not None
        assert "pid 1234" in line

    def test_returns_line_when_session_alive_no_pid(self) -> None:
        repo = self._repo(claude_pid=None, session_alive=True)
        line = _format_agent_line(repo)
        assert line is not None
        assert "agent" in line

    def test_includes_uptime_when_present(self) -> None:
        repo = self._repo(claude_pid=42, claude_uptime=90)
        line = _format_agent_line(repo)
        assert line is not None
        assert "running 1m" in line

    def test_omits_uptime_when_absent(self) -> None:
        repo = self._repo(claude_pid=42, claude_uptime=None)
        line = _format_agent_line(repo)
        assert line is not None
        assert "running" not in line

    def test_includes_session_idle_when_alive_and_no_talker(self) -> None:
        repo = self._repo(claude_pid=42, session_alive=True, claude_talker=None)
        line = _format_agent_line(repo)
        assert line is not None
        assert "session idle" in line

    def test_omits_session_idle_when_worker_is_agent_talker(self) -> None:
        repo = self._repo(
            claude_pid=42,
            session_alive=True,
            session_owner="worker-orly",
        )
        line = _format_agent_line(repo)
        assert line is not None
        assert "session idle" not in line

    def test_includes_dropped_count_singular(self) -> None:
        repo = self._repo(claude_pid=42, session_dropped_count=1)
        line = _format_agent_line(repo)
        assert line is not None
        assert "dropped session 1" in line

    def test_includes_dropped_count_plural(self) -> None:
        repo = self._repo(claude_pid=42, session_dropped_count=3)
        line = _format_agent_line(repo)
        assert line is not None
        assert "dropped sessions 3" in line

    def test_uses_provider_name_as_label(self) -> None:
        from fido.provider import ProviderID

        repo = self._repo(claude_pid=42, provider=ProviderID.COPILOT_CLI)
        line = _format_agent_line(repo)
        assert line is not None
        assert "copilot-cli" in line

    def test_indented_with_two_spaces(self) -> None:
        repo = self._repo(claude_pid=42)
        line = _format_agent_line(repo)
        assert line is not None
        assert line.startswith("  ")

    def test_includes_sent_and_received_counts_when_nonzero(self) -> None:
        repo = self._repo(
            claude_pid=42, session_sent_count=5, session_received_count=12
        )
        line = _format_agent_line(repo)
        assert line is not None
        assert "5 sent, 12 received" in line

    def test_omits_sent_and_received_counts_when_both_zero(self) -> None:
        repo = self._repo(claude_pid=42, session_sent_count=0, session_received_count=0)
        line = _format_agent_line(repo)
        assert line is not None
        assert "sent" not in line
        assert "received" not in line

    def test_includes_counts_when_only_sent_is_nonzero(self) -> None:
        repo = self._repo(claude_pid=42, session_sent_count=3, session_received_count=0)
        line = _format_agent_line(repo)
        assert line is not None
        assert "3 sent, 0 received" in line

    def test_includes_counts_when_only_received_is_nonzero(self) -> None:
        repo = self._repo(claude_pid=42, session_sent_count=0, session_received_count=7)
        line = _format_agent_line(repo)
        assert line is not None
        assert "0 sent, 7 received" in line


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
            crash_count=0,
            last_crash_error=None,
            worker_stuck=False,
        )
        defaults.update(kwargs)
        return RepoStatus(**defaults)

    def test_fido_up_with_uptime(self) -> None:
        status = FidoStatus(fido_pid=12345, fido_uptime=7980, repos=[])
        output = format_status(status)
        assert output == "fido: UP (pid 12345, uptime 2h13m)"

    def test_includes_host_resources(self) -> None:
        status = FidoStatus(
            fido_pid=12345,
            fido_uptime=7980,
            repos=[],
            resources=SystemResourceInfo(
                load_1=0.25,
                load_5=0.50,
                load_15=0.75,
                cpu_count=2,
                mem_total_bytes=4 * 1024**3,
                mem_available_bytes=1024**3,
                disk_path="/",
                disk_total_bytes=20 * 1024**3,
                disk_free_bytes=5 * 1024**3,
            ),
        )

        output = format_status(status)

        assert "host: cpu load 0.25/2 (0.50, 0.75)" in output
        assert "mem 3.0GiB/4.0GiB (75%)" in output
        assert "disk / 15.0GiB/20.0GiB (75%)" in output

    def test_includes_provider_limits_summary_and_repo_provider(self) -> None:
        provider_status = ProviderPressureStatus(
            provider=ProviderID.CLAUDE_CODE,
            pressure=0.91,
            window_name="five_hour",
        )
        status = FidoStatus(
            fido_pid=12345,
            fido_uptime=None,
            repos=[self._repo(provider_status=provider_status)],
            provider_statuses=[provider_status],
        )
        output = format_status(status)
        assert "limits: claude-code 91% (five hour)" in output
        assert "owner/repo: claude-code" in output
        assert "owner/repo: claude-code 91% (five hour)" not in output

    def test_includes_provider_reset_time_in_summary(self) -> None:
        provider_status = ProviderPressureStatus(
            provider=ProviderID.CLAUDE_CODE,
            pressure=0.96,
            window_name="five_hour",
            resets_at=datetime(2026, 4, 16, 7, 0, tzinfo=UTC),
        )
        status = FidoStatus(
            fido_pid=None,
            fido_uptime=None,
            repos=[self._repo(provider_status=provider_status)],
            provider_statuses=[provider_status],
        )
        output = format_status(status)
        assert "resets 2026-04-16 07:00 UTC" in output

    def test_includes_provider_unavailable_summary(self) -> None:
        provider_status = ProviderPressureStatus(
            provider=ProviderID.CLAUDE_CODE,
            unavailable_reason="limits unavailable",
        )
        status = FidoStatus(
            fido_pid=None,
            fido_uptime=None,
            repos=[self._repo(provider_status=provider_status)],
            provider_statuses=[provider_status],
        )
        output = format_status(status)
        assert "claude-code unavailable" in output

    def test_includes_provider_unknown_summary(self) -> None:
        provider_status = ProviderPressureStatus(provider=ProviderID.CLAUDE_CODE)
        status = FidoStatus(
            fido_pid=None,
            fido_uptime=None,
            repos=[self._repo(provider_status=provider_status)],
            provider_statuses=[provider_status],
        )
        output = format_status(status)
        assert "claude-code limits unknown" in output
        assert "owner/repo: claude-code" in output
        assert "owner/repo: claude-code limits unknown" not in output

    def test_includes_copilot_unknown_summary(self) -> None:
        provider_status = ProviderPressureStatus(provider=ProviderID.COPILOT_CLI)
        status = FidoStatus(
            fido_pid=None,
            fido_uptime=None,
            repos=[
                self._repo(
                    provider=ProviderID.COPILOT_CLI, provider_status=provider_status
                )
            ],
            provider_statuses=[provider_status],
        )
        output = format_status(status)
        assert "copilot-cli limits unknown" in output
        assert "owner/repo: copilot-cli" in output
        assert "owner/repo: copilot-cli limits unknown" not in output

    def test_fido_up_no_uptime(self) -> None:
        status = FidoStatus(fido_pid=12345, fido_uptime=None, repos=[])
        output = format_status(status)
        assert output == "fido: UP (pid 12345)"

    def test_fido_down(self) -> None:
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[])
        output = format_status(status)
        assert output == "fido: DOWN"

    def test_repo_fido_idle_no_issue(self) -> None:
        status = FidoStatus(
            fido_pid=None,
            fido_uptime=None,
            repos=[self._repo(name="owner/myrepo")],
        )
        output = format_status(status)
        assert "owner/myrepo: claude-code" in output
        assert "no assigned issues" in output

    def test_repo_fido_running_no_issue(self) -> None:
        status = FidoStatus(
            fido_pid=None,
            fido_uptime=None,
            repos=[self._repo(fido_running=True)],
        )
        output = format_status(status)
        assert "owner/repo: claude-code" in output

    def test_repo_with_issue_and_task(self) -> None:
        repo = self._repo(
            name="owner/repo",
            fido_running=True,
            issue=42,
            pending=1,
            completed=2,
            current_task="Do the thing",
            task_number=1,
            task_total=1,
            issue_title="Add widget",
            worker_what="working",
        )
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        output = format_status(status)
        assert "Issue: #42 — Add widget" in output
        assert "Worker: task 1/1 — Do the thing" in output

    def test_paused_provider_overrides_worker_state(self) -> None:
        provider_status = ProviderPressureStatus(
            provider=ProviderID.CLAUDE_CODE,
            pressure=0.97,
        )
        repo = self._repo(
            issue=42,
            current_task="Do the thing",
            task_number=3,
            task_total=3,
            provider_status=provider_status,
            worker_what="waiting",
        )
        status = FidoStatus(
            fido_pid=None,
            fido_uptime=None,
            repos=[repo],
            provider_statuses=[provider_status],
        )
        output = format_status(status)
        assert "Worker: paused for claude-code reset" in output

    def test_task_count_shows_question_mark_when_rescoping(self) -> None:
        repo = self._repo(
            issue=1,
            current_task="Reorder spec",
            task_number=2,
            task_total=5,
            rescoping=True,
            worker_what="working",
        )
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        output = format_status(status)
        assert "task 2/5?" in output

    def test_task_count_no_question_mark_when_not_rescoping(self) -> None:
        repo = self._repo(
            issue=1,
            current_task="Do stuff",
            task_number=2,
            task_total=5,
            rescoping=False,
            worker_what="working",
        )
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        output = format_status(status)
        assert "task 2/5" in output
        assert "task 2/5?" not in output

    def test_repo_issue_without_title(self) -> None:
        repo = self._repo(
            issue=5,
            pending=2,
            completed=0,
            task_number=1,
            task_total=2,
            worker_what="working",
        )
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        output = format_status(status)
        assert "Issue: #5" in output
        assert "Worker: task 1/2" in output

    def test_repo_issue_no_tasks(self) -> None:
        repo = self._repo(issue=3)
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        output = format_status(status)
        assert "Issue: #3" in output
        # No worker running → Worker line hidden entirely
        assert "Worker:" not in output

    def test_repo_issue_no_tasks_with_running_worker(self) -> None:
        repo = self._repo(issue=3, worker_what="waiting for work")
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        output = format_status(status)
        assert "Issue: #3" in output
        # Worker is running but has no task → hidden (idle workers are noise)
        assert "Worker:" not in output

    def test_claude_pid_on_agent_line_when_no_talker(self) -> None:
        """Agent info appears on a dedicated body line, not as a header suffix."""
        repo = self._repo(issue=1, claude_pid=9999, claude_uptime=185)
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        output = format_status(status)
        assert "pid 9999" in output
        assert "running 3m" in output
        # Info is on a body line, not the repo header.
        assert not any(
            line.startswith("owner/repo:") and "pid 9999" in line
            for line in output.splitlines()
        )

    def test_claude_pid_no_uptime_on_agent_line(self) -> None:
        repo = self._repo(issue=1, claude_pid=9999, claude_uptime=None)
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        output = format_status(status)
        assert "pid 9999" in output
        assert "running" not in output

    def test_worker_talker_shows_agent_line_and_no_pid_on_header(self) -> None:
        """Worker-kind talker → agent line still shown; repo header carries no pid info."""
        repo = self._repo(
            issue=1,
            claude_pid=9999,
            claude_uptime=60,
            session_alive=True,
            worker_what="working",
            claude_talker=ClaudeTalkerInfo(
                thread_id=111,
                kind="worker",
                description="persistent session turn",
                claude_pid=9999,
            ),
        )
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        output = format_status(status)
        # Agent line is always present when there's a pid.
        assert "pid 9999" in output
        # Active-agent rows start with "* Worker:" (NO_COLOR-friendly marker);
        # inactive rows start with "  Worker:".  This repo's worker is the
        # active talker, so check both prefixes for forward-compat.
        worker_lines = [
            line
            for line in output.splitlines()
            if line.startswith("  Worker:") or line.startswith("* Worker:")
        ]
        assert any("→ pid" not in line for line in worker_lines)
        # Header does not carry pid info — that belongs to the agent line.
        header = next(line for line in output.splitlines() if line.startswith("owner"))
        assert "pid 9999" not in header

    def test_agent_line_session_alive_no_talker(self) -> None:
        """Session alive but nobody holds the lock → dedicated line with 'session idle'."""
        repo = self._repo(
            issue=1,
            claude_pid=9999,
            claude_uptime=60,
            session_alive=True,
            claude_talker=None,
        )
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        output = format_status(status)
        # Appears on the dedicated agent line, not as a header suffix.
        assert "pid 9999 (running 1m, session idle)" in output
        assert not any(
            line.startswith("owner/repo:") and "pid 9999" in line
            for line in output.splitlines()
        )

    def test_session_alive_without_claude_pid(self) -> None:
        """Session_alive with no pid still signals idle agent presence on its own line."""
        repo = self._repo(
            issue=1,
            claude_pid=None,
            session_alive=True,
        )
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        output = format_status(status)
        assert "agent (session idle)" in output

    def test_agent_line_shows_dropped_session_count(self) -> None:
        repo = self._repo(
            issue=1,
            claude_pid=9999,
            claude_uptime=60,
            session_dropped_count=2,
        )
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        output = format_status(status)
        assert "pid 9999 (running 1m, dropped sessions 2)" in output

    def test_multiple_repos(self) -> None:
        # Each repo emits a header + "no assigned issues" body line.
        repos = [
            self._repo(name="a/b"),
            self._repo(name="c/d"),
        ]
        status = FidoStatus(fido_pid=1, fido_uptime=60, repos=repos)
        lines = format_status(status).splitlines()
        assert lines[0].startswith("fido:")
        assert any(line.startswith("a/b:") for line in lines)
        assert any(line.startswith("c/d:") for line in lines)

    def test_worker_line_hidden_when_only_worker_what_set(self) -> None:
        """Per #1029: the Worker line hides when ``worker_what`` is the
        only signal — "waiting for work" is the default state and not
        worth a line.  The visibility gate (``_should_show_worker_line``)
        now requires an active state: current_task, task numbering, the
        agent talker, or a paused provider.
        """
        repo = self._repo(
            fido_running=True, issue=1, worker_what="Working on: #3 add widget"
        )
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        output = format_status(status)
        assert "Worker:" not in output, output

    def test_worker_what_hidden_when_task_present(self) -> None:
        """worker_what is redundant when there's a specific Task line."""
        repo = self._repo(
            fido_running=True,
            issue=1,
            current_task="Do thing",
            task_number=1,
            task_total=1,
            worker_what="Working on something",
        )
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        output = format_status(status)
        assert "Working on something" not in output

    def test_crash_count_zero_not_shown(self) -> None:
        repo = self._repo(crash_count=0, last_crash_error=None)
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        output = format_status(status)
        assert "crashes" not in output
        assert "crash" not in output.lower()

    def test_crash_count_nonzero_shown_with_error(self) -> None:
        repo = self._repo(crash_count=3, last_crash_error="RuntimeError: boom")
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        output = format_status(status)
        assert "crashes 3" in output
        assert "RuntimeError: boom" in output

    def test_crash_count_without_error(self) -> None:
        repo = self._repo(crash_count=1, last_crash_error=None)
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        output = format_status(status)
        assert "crashes 1" in output
        assert "last crash" not in output

    def test_worker_uptime_shown_in_header(self) -> None:
        repo = self._repo(fido_running=True, worker_uptime=7320)
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        output = format_status(status)
        # 7320s = 2h02m
        assert "up 2h2m" in output

    def test_issue_elapsed_shown_in_body(self) -> None:
        repo = self._repo(issue=1, issue_elapsed_seconds=3900)
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        output = format_status(status)
        # 3900s = 1h05m
        assert "elapsed 1h5m" in output

    def test_pr_line_rendered_when_pr_number_set(self) -> None:
        repo = self._repo(
            issue=1,
            pr_number=42,
            pr_title="Refactor widget",
        )
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        output = format_status(status)
        assert "PR:     #42 — Refactor widget" in output

    def test_pr_line_without_title(self) -> None:
        repo = self._repo(issue=1, pr_number=7, pr_title=None)
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        output = format_status(status)
        assert "PR:     #7" in output

    def test_current_task_line_without_numbering(self) -> None:
        repo = self._repo(
            issue=1,
            current_task="Freeform task",
            task_number=None,
            task_total=None,
            worker_what="working",
        )
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        output = format_status(status)
        # Worker line shows the free-form task title.
        assert "Worker: task: Freeform task" in output

    def test_webhook_talker_sorts_to_top_without_pid_suffix(self) -> None:
        """When a webhook is the talker, its line sorts to the top and is highlighted."""
        repo = self._repo(
            issue=1,
            claude_pid=8888,
            claude_uptime=45,
            session_alive=True,
            webhook_activities=[
                WebhookActivityInfo(
                    description="triaging comment", elapsed_seconds=30, thread_id=1
                ),
                WebhookActivityInfo(
                    description="replying to review", elapsed_seconds=2, thread_id=2
                ),
            ],
            claude_talker=ClaudeTalkerInfo(
                thread_id=2,
                kind="webhook",
                description="one-shot claude --print",
                claude_pid=8888,
            ),
        )
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        webhook_lines = [
            line for line in format_status(status).splitlines() if "webhook:" in line
        ]
        assert "replying to review" in webhook_lines[0]
        assert "→ pid" not in webhook_lines[0]
        assert "triaging comment" in webhook_lines[1]
        assert "→ pid" not in webhook_lines[1]

    def test_worker_line_has_no_provider_suffix_when_worker_is_talker(self) -> None:
        """Worker line has no '<- provider' suffix even when the worker owns the session."""
        repo = self._repo(
            issue=1,
            current_task="Do thing",
            task_number=1,
            task_total=1,
            worker_what="working",
            claude_talker=ClaudeTalkerInfo(
                thread_id=100,
                kind="worker",
                description="persistent session turn",
                claude_pid=42,
            ),
        )
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        output = format_status(status)
        worker_line = next(ln for ln in output.splitlines() if "Worker:" in ln)
        assert "<-" not in worker_line

    def test_worker_line_no_marker_when_not_talker(self) -> None:
        """Worker line has no provider marker when the worker is on a task but not talking."""
        repo = self._repo(
            issue=1,
            current_task="Do thing",
            task_number=1,
            task_total=1,
            worker_what="working",
        )
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        output = format_status(status)
        worker_line = next(ln for ln in output.splitlines() if "Worker:" in ln)
        assert "<-" not in worker_line

    def test_webhook_talker_sorts_first_and_has_no_provider_suffix(self) -> None:
        """Active webhook talker sorts to top; neither line has a '<- provider' suffix."""
        repo = self._repo(
            issue=1,
            webhook_activities=[
                WebhookActivityInfo(
                    description="triaging comment", elapsed_seconds=5, thread_id=1
                ),
                WebhookActivityInfo(
                    description="replying to review", elapsed_seconds=2, thread_id=2
                ),
            ],
            claude_talker=ClaudeTalkerInfo(
                thread_id=2,
                kind="webhook",
                description="one-shot",
                claude_pid=99,
            ),
        )
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        output = format_status(status)
        webhook_lines = [ln for ln in output.splitlines() if "webhook:" in ln]
        # Active talker (replying to review, tid=2) sorts to top.
        assert "replying to review" in webhook_lines[0]
        assert "<-" not in webhook_lines[0]
        # Non-talker also has no suffix.
        assert "triaging comment" in webhook_lines[1]
        assert "<-" not in webhook_lines[1]

    def test_webhook_overflow_summary_when_more_than_five(self) -> None:
        """More than 5 webhook activities → first 5 shown + '+N more' line."""
        repo = self._repo(
            issue=1,
            webhook_activities=[
                WebhookActivityInfo(
                    description=f"wh{i}", elapsed_seconds=i, thread_id=i
                )
                for i in range(9)
            ],
        )
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        lines = [
            line
            for line in format_status(status).splitlines()
            if "webhook" in line or "more" in line
        ]
        # 5 shown + 1 overflow summary.
        assert len(lines) == 6
        assert "+4 more webhooks" in lines[-1]

    def test_webhook_overflow_singular_for_one_extra(self) -> None:
        """Overflow of exactly 1 uses singular 'webhook'."""
        repo = self._repo(
            issue=1,
            webhook_activities=[
                WebhookActivityInfo(
                    description=f"wh{i}", elapsed_seconds=i, thread_id=i
                )
                for i in range(6)
            ],
        )
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        assert "+1 more webhook" in format_status(status)
        assert "+1 more webhooks" not in format_status(status)

    def test_webhook_activities_rendered_as_sub_bullets(self) -> None:
        repo = self._repo(
            issue=1,
            webhook_activities=[
                WebhookActivityInfo(
                    description="triaging comment on PR #9",
                    elapsed_seconds=12,
                    thread_id=1,
                ),
                WebhookActivityInfo(
                    description="replying to review",
                    elapsed_seconds=3,
                    thread_id=2,
                ),
            ],
        )
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        output = format_status(status)
        # Webhook lines are plain peer siblings without tree characters.
        assert "webhook: triaging comment on PR #9 (12s)" in output
        assert "webhook: replying to review (3s)" in output
        assert "├─" not in output
        assert "└─" not in output

    def test_worker_busy_false_not_shown(self) -> None:
        repo = self._repo(worker_stuck=False)
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        output = format_status(status)
        assert "BUSY" not in output

    def test_worker_busy_true_shown(self) -> None:
        repo = self._repo(worker_stuck=True)
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        output = format_status(status)
        assert "BUSY" in output

    def test_busy_appears_in_header_with_crash(self) -> None:
        repo = self._repo(worker_stuck=True, crash_count=1, last_crash_error="err")
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        output = format_status(status)
        # Both appear in the comma-separated top-line stats.
        assert "crashes 1" in output
        assert "BUSY" in output
        assert "last crash: err" in output


class TestFormatStatusColor:
    """Color tests: verify ANSI codes appear under FORCE_COLOR=1."""

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
            crash_count=0,
            last_crash_error=None,
            worker_stuck=False,
        )
        defaults.update(kwargs)
        return RepoStatus(**defaults)

    def _color_env(self) -> dict[str, str]:
        return {"FORCE_COLOR": "1"}

    def test_fido_up_header_bold(self) -> None:
        status = FidoStatus(fido_pid=42, fido_uptime=60, repos=[])
        with patch.dict("os.environ", self._color_env(), clear=True):
            output = format_status(status)
        assert output.startswith(_CODES["bold"])

    def test_fido_down_header_bold(self) -> None:
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[])
        with patch.dict("os.environ", self._color_env(), clear=True):
            output = format_status(status)
        assert _CODES["bold"] in output

    def test_repo_running_bold(self) -> None:
        repo = self._repo(fido_running=True)
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        with patch.dict("os.environ", self._color_env(), clear=True):
            output = format_status(status)
        header = [ln for ln in output.splitlines() if "owner/repo" in ln][0]
        assert _CODES["bold"] in header

    def test_repo_idle_when_fido_down(self) -> None:
        """When fido itself is down (``fido_running=False``), the repo
        header is rendered with a brown-background highlight rather
        than the plain dim styling — the dim styling has been
        repurposed for inactive providers, brown signals "fido itself
        isn't running so this repo can't make progress".
        """
        repo = self._repo(fido_running=False)
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        with patch.dict("os.environ", self._color_env(), clear=True):
            output = format_status(status)
        header = [ln for ln in output.splitlines() if "owner/repo" in ln][0]
        # 24-bit brown background (60, 30, 5) — the "fido is down" highlight.
        assert "\x1b[48;2;60;30;5m" in header

    def test_issue_number_cyan(self) -> None:
        repo = self._repo(issue=42)
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        with patch.dict("os.environ", self._color_env(), clear=True):
            output = format_status(status)
        assert f"{_CODES['cyan']}#42" in output

    def test_pr_number_magenta(self) -> None:
        repo = self._repo(issue=1, pr_number=99)
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        with patch.dict("os.environ", self._color_env(), clear=True):
            output = format_status(status)
        assert f"{_CODES['magenta']}#99" in output

    def test_elapsed_dim(self) -> None:
        repo = self._repo(issue=1, issue_elapsed_seconds=120)
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        with patch.dict("os.environ", self._color_env(), clear=True):
            output = format_status(status)
        assert f"{_CODES['dim']}(elapsed 2m)" in output
        assert f"  {_CODES['dim']}(elapsed 2m)" not in output
        assert f" {_CODES['dim']}(elapsed 2m)" in output

    def test_busy_red(self) -> None:
        repo = self._repo(worker_stuck=True)
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        with patch.dict("os.environ", self._color_env(), clear=True):
            output = format_status(status)
        assert f"{_CODES['red']}BUSY" in output

    def test_provider_warning_dim(self) -> None:
        provider_status = ProviderPressureStatus(
            provider=ProviderID.CLAUDE_CODE,
            pressure=0.9,
        )
        repo = self._repo(provider_status=provider_status)
        status = FidoStatus(
            fido_pid=None,
            fido_uptime=None,
            repos=[repo],
            provider_statuses=[provider_status],
        )
        with patch.dict("os.environ", self._color_env(), clear=True):
            output = format_status(status)
        assert f"{_CODES['dim']}claude-code 90%" in output

    def test_provider_pause_dark_gray(self) -> None:
        provider_status = ProviderPressureStatus(
            provider=ProviderID.CLAUDE_CODE,
            pressure=0.95,
        )
        repo = self._repo(provider_status=provider_status)
        status = FidoStatus(
            fido_pid=None,
            fido_uptime=None,
            repos=[repo],
            provider_statuses=[provider_status],
        )
        with patch.dict("os.environ", self._color_env(), clear=True):
            output = format_status(status)
        assert f"{_CODES['dark_gray']}claude-code 95%" in output

    def test_provider_ok_has_no_warning_color(self) -> None:
        provider_status = ProviderPressureStatus(
            provider=ProviderID.CLAUDE_CODE,
            pressure=0.5,
        )
        status = FidoStatus(
            fido_pid=None,
            fido_uptime=None,
            repos=[self._repo(provider_status=provider_status)],
            provider_statuses=[provider_status],
        )
        with patch.dict("os.environ", self._color_env(), clear=True):
            output = format_status(status)
        assert f"{_CODES['dim']}claude-code 50%" not in output
        assert f"{_CODES['dark_gray']}claude-code 50%" not in output

    def test_crash_red_bold(self) -> None:
        repo = self._repo(crash_count=2, last_crash_error="RuntimeError: boom")
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        with patch.dict("os.environ", self._color_env(), clear=True):
            output = format_status(status)
        assert f"{_CODES['red_bold']}crashes 2" in output
        assert f"{_CODES['red_bold']}last crash: RuntimeError: boom" in output

    def test_task_counter_bold(self) -> None:
        repo = self._repo(
            issue=1,
            current_task="Do thing",
            task_number=2,
            task_total=5,
            worker_what="working",
        )
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        with patch.dict("os.environ", self._color_env(), clear=True):
            output = format_status(status)
        assert f"{_CODES['bold']}task 2/5" in output

    def test_worker_label_green_when_talker(self) -> None:
        repo = self._repo(
            issue=1,
            claude_pid=999,
            session_alive=True,
            worker_what="working",
            claude_talker=ClaudeTalkerInfo(
                thread_id=1, kind="worker", description="turn", claude_pid=999
            ),
        )
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        with patch.dict("os.environ", self._color_env(), clear=True):
            output = format_status(status)
        worker_line = [ln for ln in output.splitlines() if "Worker:" in ln][0]
        assert f"{_CODES['green_bg']}Worker:" in worker_line

    def test_worker_label_green_bg_during_python_only_gap(self) -> None:
        """Per #1029: ``current_task`` alone keeps the green_bg on so
        the worker label doesn't flicker off during the Python-only
        gaps between ``session.prompt`` calls (where talker briefly
        unregisters).  Anti-regression for the previous "highlight iff
        talker" semantics.
        """
        repo = self._repo(
            issue=1,
            current_task="Refactor",
            session_alive=True,
            session_owner="webhook-handler",  # someone else holds the lock
            worker_what="waiting",
        )
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        with patch.dict("os.environ", self._color_env(), clear=True):
            output = format_status(status)
        worker_line = [ln for ln in output.splitlines() if "Worker:" in ln][0]
        assert _CODES["green_bg"] in worker_line

    def test_worker_label_green_bg_when_session_owner_is_worker(self) -> None:
        repo = self._repo(
            issue=1,
            session_owner="worker-orly",
            session_alive=True,
            worker_what="working",
        )
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        with patch.dict("os.environ", self._color_env(), clear=True):
            output = format_status(status)
        worker_line = [ln for ln in output.splitlines() if "Worker:" in ln][0]
        assert f"{_CODES['green_bg']}Worker:" in worker_line

    def test_worker_label_green_bg_when_on_task_even_without_talker(self) -> None:
        """Highlight stays on during the Python-only gap between
        ``session.prompt`` calls — the talker is unregistered but the
        worker is still assigned to a task (``current_task`` set)."""
        repo = self._repo(
            issue=1,
            current_task="Do thing",
            task_number=1,
            task_total=3,
            worker_what="working",
            # No claude_talker, no session_owner — purely between-turn state.
        )
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        with patch.dict("os.environ", self._color_env(), clear=True):
            output = format_status(status)
        worker_line = [ln for ln in output.splitlines() if "Worker:" in ln][0]
        assert f"{_CODES['green_bg']}Worker:" in worker_line

    def test_worker_line_hidden_when_napping_with_no_task(self) -> None:
        """Per #1029: napping (no task, no talker) → no Worker line at
        all.  The previous behaviour of emitting a plain "waiting for
        work" line was repurposed to use the line's absence as the
        idle signal.
        """
        repo = self._repo(
            issue=1,
            current_task=None,
            session_alive=False,
            session_owner=None,
            worker_what="waiting for work",
        )
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        with patch.dict("os.environ", self._color_env(), clear=True):
            output = format_status(status)
        assert "Worker:" not in output, output

    def test_webhook_label_yellow_when_talker(self) -> None:
        repo = self._repo(
            issue=1,
            claude_pid=888,
            session_alive=True,
            webhook_activities=[
                WebhookActivityInfo(
                    description="triaging", elapsed_seconds=10, thread_id=5
                ),
            ],
            claude_talker=ClaudeTalkerInfo(
                thread_id=5, kind="webhook", description="one-shot", claude_pid=888
            ),
        )
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        with patch.dict("os.environ", self._color_env(), clear=True):
            output = format_status(status)
        wh_line = [ln for ln in output.splitlines() if "webhook:" in ln][0]
        assert f"{_CODES['yellow_bg']}webhook:" in wh_line

    def test_webhook_label_plain_when_not_talker(self) -> None:
        repo = self._repo(
            issue=1,
            webhook_activities=[
                WebhookActivityInfo(
                    description="triaging", elapsed_seconds=10, thread_id=5
                ),
            ],
        )
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        with patch.dict("os.environ", self._color_env(), clear=True):
            output = format_status(status)
        wh_line = [ln for ln in output.splitlines() if "webhook:" in ln][0]
        assert _CODES["yellow_bg"] not in wh_line

    def test_session_idle_dim(self) -> None:
        repo = self._repo(issue=1, claude_pid=999, session_alive=True)
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        with patch.dict("os.environ", self._color_env(), clear=True):
            output = format_status(status)
        assert f"{_CODES['dim']}session idle" in output

    def test_session_idle_hidden_while_worker_owns_agent(self) -> None:
        repo = self._repo(
            issue=1,
            claude_pid=999,
            session_alive=True,
            session_owner="worker-orly",
        )
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        with patch.dict("os.environ", self._color_env(), clear=True):
            output = format_status(status)
        assert "session idle" not in output

    def test_claude_running_uptime_dim(self) -> None:
        repo = self._repo(issue=1, claude_pid=999, claude_uptime=120)
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        with patch.dict("os.environ", self._color_env(), clear=True):
            output = format_status(status)
        assert f"{_CODES['dim']}running 2m" in output

    def test_no_color_env_suppresses_ansi(self) -> None:
        repo = self._repo(fido_running=True, issue=42, worker_stuck=True)
        status = FidoStatus(fido_pid=1, fido_uptime=60, repos=[repo])
        with patch.dict("os.environ", {"NO_COLOR": ""}, clear=True):
            output = format_status(status)
        assert "\033[" not in output


class TestProviderColoredStatus:
    """Provider-specific section-bg tinting + limits-line fg highlighting.

    Feature: repo sections get the provider's dim_bg across all their
    lines; the limits-line provider tokens get the provider's bright_fg;
    the active-agent "Worker:" row carries an ASCII ``*`` marker under
    ``NO_COLOR`` (GREEN_BG is the signal when color is enabled, so no
    asterisk is needed in color mode).
    """

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
            crash_count=0,
            last_crash_error=None,
            worker_stuck=False,
        )
        defaults.update(kwargs)
        return RepoStatus(**defaults)

    def test_active_worker_line_starts_with_asterisk_marker(self) -> None:
        # NO_COLOR alternate for the GREEN_BG highlight: active rows carry
        # a leading ``* `` that's visible regardless of ANSI support.
        with patch.dict("os.environ", {"NO_COLOR": ""}, clear=True):
            repo = self._repo(
                fido_running=True,
                issue=7,
                current_task={"title": "implement foo", "index": 1, "total": 2},
                worker_what="working",
            )
            status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
            output = format_status(status)
        worker_lines = [ln for ln in output.splitlines() if "Worker:" in ln]
        assert worker_lines, f"no Worker line in:\n{output}"
        assert worker_lines[0].startswith("* "), worker_lines[0]

    def test_active_worker_line_has_no_asterisk_when_color_enabled(self) -> None:
        # When color is on, GREEN_BG provides the active-worker signal;
        # the ``*`` marker must NOT appear (it would be redundant clutter).
        with patch.dict("os.environ", {"FORCE_COLOR": "1"}, clear=True):
            repo = self._repo(
                fido_running=True,
                issue=7,
                current_task={"title": "implement foo", "index": 1, "total": 2},
                worker_what="working",
            )
            status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
            output = format_status(status)
        worker_lines = [ln for ln in output.splitlines() if "Worker:" in ln]
        assert worker_lines, f"no Worker line in:\n{output}"
        # Strip ANSI codes to check prefix cleanly.
        import re

        plain = re.sub(r"\033\[[^m]*m", "", worker_lines[0])
        assert not plain.startswith("* "), (
            f"unexpected asterisk in color mode: {plain!r}"
        )
        assert plain.startswith("  "), f"expected two-space indent: {plain!r}"

    def test_active_worker_line_has_marker_in_no_color_mode(self) -> None:
        """In NO_COLOR mode the active Worker line is prefixed with
        ``* `` so the active state is still distinguishable when the
        green_bg highlight is suppressed.  Per #1029 the inactive form
        no longer renders, so this is the only Worker-line shape.
        """
        with patch.dict("os.environ", {"NO_COLOR": ""}, clear=True):
            repo = self._repo(
                fido_running=True,
                issue=7,
                current_task="Refactor",
                session_alive=True,
                session_owner="webhook-handler",
                worker_what="waiting",
            )
            status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
            output = format_status(status)
        worker_lines = [ln for ln in output.splitlines() if "Worker:" in ln]
        assert worker_lines, f"no Worker line in:\n{output}"
        assert worker_lines[0].startswith("* Worker:"), worker_lines[0]

    def test_limits_line_colors_claude_code_token_when_color_enabled(self) -> None:
        from fido.color import rgb_fg
        from fido.provider import PROVIDER_PALETTES

        with patch.dict("os.environ", {"FORCE_COLOR": "1"}, clear=True):
            # pressure=0.50 keeps the status healthy — no warning/paused
            # overlay — so the provider-color highlight wins.
            provider_status = ProviderPressureStatus(
                provider=ProviderID.CLAUDE_CODE,
                pressure=0.50,
                window_name="five_hour",
            )
            status = FidoStatus(
                fido_pid=None,
                fido_uptime=None,
                repos=[self._repo(provider_status=provider_status)],
                provider_statuses=[provider_status],
            )
            output = format_status(status)
        palette = PROVIDER_PALETTES[ProviderID.CLAUDE_CODE]
        expected_prefix = rgb_fg(*palette.bright_fg) + "claude-code"
        limits_line = next(ln for ln in output.splitlines() if "limits:" in ln)
        assert expected_prefix in limits_line

    def test_limits_line_highlights_copilot_token(self) -> None:
        from fido.color import rgb_fg
        from fido.provider import PROVIDER_PALETTES

        with patch.dict("os.environ", {"FORCE_COLOR": "1"}, clear=True):
            provider_status = ProviderPressureStatus(provider=ProviderID.COPILOT_CLI)
            status = FidoStatus(
                fido_pid=None,
                fido_uptime=None,
                repos=[
                    self._repo(
                        provider=ProviderID.COPILOT_CLI, provider_status=provider_status
                    )
                ],
                provider_statuses=[provider_status],
            )
            output = format_status(status)
        palette = PROVIDER_PALETTES[ProviderID.COPILOT_CLI]
        expected_prefix = rgb_fg(*palette.bright_fg) + "copilot-cli"
        limits_line = next(ln for ln in output.splitlines() if "limits:" in ln)
        assert expected_prefix in limits_line

    def test_limits_line_respects_paused_style_over_provider_fg(self) -> None:
        # A paused / warning status wins over the provider highlight so
        # state signalling isn't lost to identity coloring.
        with patch.dict("os.environ", {"FORCE_COLOR": "1"}, clear=True):
            # pressure=0.99 crosses the pause threshold (0.95) → paused.
            provider_status = ProviderPressureStatus(
                provider=ProviderID.CLAUDE_CODE,
                pressure=0.99,
            )
            status = FidoStatus(
                fido_pid=None,
                fido_uptime=None,
                repos=[self._repo(provider_status=provider_status)],
                provider_statuses=[provider_status],
            )
            output = format_status(status)
        # DARK_GRAY code is \033[90m — must be present; truecolor provider
        # prefix must NOT be (identity color suppressed while paused).
        assert "\033[90m" in output

    def test_repo_section_lines_get_provider_bg_when_color_enabled(self) -> None:
        from fido.color import rgb_bg
        from fido.provider import PROVIDER_PALETTES

        with patch.dict("os.environ", {"FORCE_COLOR": "1"}, clear=True):
            repo = self._repo(fido_running=True, issue=7, issue_title="do thing")
            status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
            output = format_status(status)
        palette = PROVIDER_PALETTES[ProviderID.CLAUDE_CODE]
        expected_bg = rgb_bg(*palette.dim_bg)
        repo_lines = [
            ln for ln in output.splitlines() if "owner/repo" in ln or "Issue:" in ln
        ]
        assert repo_lines, f"no repo/issue lines:\n{output}"
        for line in repo_lines:
            assert expected_bg in line, f"bg missing from: {line!r}"

    def test_repo_section_lines_get_codex_provider_bg_when_color_enabled(self) -> None:
        from fido.color import rgb_bg
        from fido.provider import PROVIDER_PALETTES

        with patch.dict("os.environ", {"FORCE_COLOR": "1"}, clear=True):
            repo = self._repo(provider=ProviderID.CODEX, fido_running=True, issue=7)
            status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
            output = format_status(status)
        palette = PROVIDER_PALETTES[ProviderID.CODEX]
        expected_bg = rgb_bg(*palette.dim_bg)
        repo_lines = [
            ln for ln in output.splitlines() if "owner/repo" in ln or "Issue:" in ln
        ]
        assert repo_lines, f"no repo/issue lines:\n{output}"
        for line in repo_lines:
            assert expected_bg in line, f"bg missing from: {line!r}"

    def test_limits_line_highlights_codex_token(self) -> None:
        from fido.color import rgb_fg
        from fido.provider import PROVIDER_PALETTES

        with patch.dict("os.environ", {"FORCE_COLOR": "1"}, clear=True):
            provider_status = ProviderPressureStatus(
                provider=ProviderID.CODEX,
                pressure=0.50,
                window_name="five_hour",
            )
            status = FidoStatus(
                fido_pid=None,
                fido_uptime=None,
                repos=[
                    self._repo(
                        provider=ProviderID.CODEX, provider_status=provider_status
                    )
                ],
                provider_statuses=[provider_status],
            )
            output = format_status(status)
        palette = PROVIDER_PALETTES[ProviderID.CODEX]
        expected_prefix = rgb_fg(*palette.bright_fg) + "codex"
        limits_line = next(ln for ln in output.splitlines() if "limits:" in ln)
        assert expected_prefix in limits_line

    def test_repo_header_colors_provider_name_with_bright_fg(self) -> None:
        # The provider token in the repo header stats line gets the provider's
        # bright_fg color, matching the visual identity on the global limits line.
        from fido.color import rgb_fg
        from fido.provider import PROVIDER_PALETTES

        with patch.dict("os.environ", {"FORCE_COLOR": "1"}, clear=True):
            provider_status = ProviderPressureStatus(
                provider=ProviderID.CLAUDE_CODE,
                pressure=0.50,
                window_name="five_hour",
            )
            repo = self._repo(provider_status=provider_status)
            status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
            output = format_status(status)
        palette = PROVIDER_PALETTES[ProviderID.CLAUDE_CODE]
        expected = rgb_fg(*palette.bright_fg) + "claude-code"
        header_line = next(ln for ln in output.splitlines() if "owner/repo" in ln)
        assert expected in header_line

    def test_repo_header_provider_name_respects_paused_style_over_bright_fg(
        self,
    ) -> None:
        # Warning/paused state wins over the provider-color highlight in the
        # repo header, same precedence as the global limits line.
        with patch.dict("os.environ", {"FORCE_COLOR": "1"}, clear=True):
            provider_status = ProviderPressureStatus(
                provider=ProviderID.CLAUDE_CODE,
                pressure=0.99,
            )
            repo = self._repo(provider_status=provider_status)
            status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
            output = format_status(status)
        header_line = next(ln for ln in output.splitlines() if "owner/repo" in ln)
        # DARK_GRAY code (\033[90m) must be present on the header for the paused state.
        assert "\033[90m" in header_line

    def test_repo_header_codex_provider_name_colored(self) -> None:
        from fido.color import rgb_fg
        from fido.provider import PROVIDER_PALETTES

        with patch.dict("os.environ", {"FORCE_COLOR": "1"}, clear=True):
            repo = self._repo(provider=ProviderID.CODEX)
            status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
            output = format_status(status)
        palette = PROVIDER_PALETTES[ProviderID.CODEX]
        expected = rgb_fg(*palette.bright_fg) + "codex"
        header_line = next(ln for ln in output.splitlines() if "owner/repo" in ln)
        assert expected in header_line

    def test_repo_header_provider_name_colored_when_no_provider_status(self) -> None:
        # Even without a provider_status, the provider name gets bright_fg color
        # from the palette when the palette exists.
        from fido.color import rgb_fg
        from fido.provider import PROVIDER_PALETTES

        with patch.dict("os.environ", {"FORCE_COLOR": "1"}, clear=True):
            repo = self._repo(provider=ProviderID.CLAUDE_CODE, provider_status=None)
            status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
            output = format_status(status)
        palette = PROVIDER_PALETTES[ProviderID.CLAUDE_CODE]
        expected = rgb_fg(*palette.bright_fg) + "claude-code"
        header_line = next(ln for ln in output.splitlines() if "owner/repo" in ln)
        assert expected in header_line


# ── Issue cache display (closes #812) ────────────────────────────────────────


class TestParseIsoDatetime:
    def test_returns_none_for_non_string(self) -> None:
        assert _parse_iso_datetime(None) is None
        assert _parse_iso_datetime(123) is None

    def test_returns_none_for_invalid_string(self) -> None:
        assert _parse_iso_datetime("not a timestamp") is None

    def test_parses_valid_iso_string(self) -> None:
        d = _parse_iso_datetime("2026-04-19T12:00:00+00:00")
        assert d is not None
        assert d.year == 2026


class TestParseIssueCache:
    def test_returns_none_for_non_dict(self) -> None:
        assert _parse_issue_cache(None) is None
        assert _parse_issue_cache("oops") is None

    def test_parses_minimal_payload(self) -> None:
        info = _parse_issue_cache(
            {
                "loaded": True,
                "open_issues": 5,
                "events_applied": 10,
                "events_dropped_stale": 2,
                "last_event_at": "2026-04-19T12:00:00+00:00",
                "last_reconcile_at": None,
                "last_reconcile_drift": 0,
            }
        )
        assert info is not None
        assert info.loaded is True
        assert info.open_issues == 5
        assert info.events_applied == 10
        assert info.events_dropped_stale == 2
        assert info.last_event_at is not None
        assert info.last_reconcile_at is None
        assert info.last_reconcile_drift == 0

    def test_defaults_when_keys_missing(self) -> None:
        info = _parse_issue_cache({})
        assert info is not None
        assert info.loaded is False
        assert info.open_issues == 0


class TestFormatCacheLine:
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
            crash_count=0,
            last_crash_error=None,
            worker_stuck=False,
        )
        defaults.update(kwargs)
        return RepoStatus(**defaults)

    def test_returns_none_when_no_cache(self) -> None:
        assert _format_cache_line(self._repo(issue_cache=None)) is None

    def test_returns_none_when_cache_unloaded(self) -> None:
        info = IssueCacheInfo(
            loaded=False,
            open_issues=0,
            events_applied=0,
            events_dropped_stale=0,
            last_event_at=None,
            last_reconcile_at=None,
            last_reconcile_drift=0,
        )
        assert _format_cache_line(self._repo(issue_cache=info)) is None

    def test_renders_basic_loaded_cache(self) -> None:
        info = IssueCacheInfo(
            loaded=True,
            open_issues=42,
            events_applied=7,
            events_dropped_stale=0,
            last_event_at=None,
            last_reconcile_at=None,
            last_reconcile_drift=0,
        )
        line = _format_cache_line(self._repo(issue_cache=info))
        assert line is not None
        assert "42 open" in line
        assert "applied 7" in line
        assert "stale-dropped" not in line
        assert "reconciled" not in line

    def test_includes_stale_dropped_when_nonzero(self) -> None:
        info = IssueCacheInfo(
            loaded=True,
            open_issues=10,
            events_applied=20,
            events_dropped_stale=3,
            last_event_at=None,
            last_reconcile_at=None,
            last_reconcile_drift=0,
        )
        line = _format_cache_line(self._repo(issue_cache=info))
        assert line is not None
        assert "stale-dropped 3" in line

    def test_includes_reconcile_when_present(self) -> None:
        info = IssueCacheInfo(
            loaded=True,
            open_issues=10,
            events_applied=20,
            events_dropped_stale=0,
            last_event_at=None,
            last_reconcile_at=datetime(2026, 4, 19, tzinfo=UTC),
            last_reconcile_drift=2,
        )
        line = _format_cache_line(self._repo(issue_cache=info))
        assert line is not None
        assert "reconciled drift 2" in line


class TestFormatStatusCacheLineIntegration:
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
            crash_count=0,
            last_crash_error=None,
            worker_stuck=False,
        )
        defaults.update(kwargs)
        return RepoStatus(**defaults)

    def test_format_status_includes_cache_line_when_loaded(self) -> None:
        info = IssueCacheInfo(
            loaded=True,
            open_issues=99,
            events_applied=1,
            events_dropped_stale=0,
            last_event_at=None,
            last_reconcile_at=None,
            last_reconcile_drift=0,
        )
        repo = self._repo(issue_cache=info)
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        out = format_status(status)
        assert "Cache:" in out
        assert "99 open" in out

    def test_format_status_omits_cache_line_when_absent(self) -> None:
        repo = self._repo(issue_cache=None)
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        out = format_status(status)
        assert "Cache:" not in out


# ── GitHub rate-limit display (closes #812 follow-up) ─────────────────────────


def _window(
    name: str = "core",
    used: int = 100,
    limit: int = 5000,
    reset_offset_seconds: int = 3600,
) -> RateLimitWindowInfo:
    """Build a RateLimitWindowInfo with reset time relative to now."""
    return RateLimitWindowInfo(
        name=name,
        used=used,
        limit=limit,
        resets_at=datetime.now(tz=timezone.utc).replace(microsecond=0)
        + timedelta(seconds=reset_offset_seconds),
    )


class TestParseRateLimit:
    def test_returns_none_for_non_dict(self) -> None:
        assert _parse_rate_limit(None) is None
        assert _parse_rate_limit("nope") is None

    def test_returns_none_when_rest_missing(self) -> None:
        raw = {"graphql": {}, "fetched_at": "2026-04-19T12:00:00+00:00"}
        assert _parse_rate_limit(raw) is None

    def test_returns_none_when_graphql_missing(self) -> None:
        raw = {"rest": {}, "fetched_at": "2026-04-19T12:00:00+00:00"}
        assert _parse_rate_limit(raw) is None

    def test_returns_none_when_fetched_at_missing(self) -> None:
        raw = {"rest": {}, "graphql": {}}
        assert _parse_rate_limit(raw) is None

    def test_parses_full_payload(self) -> None:
        raw = {
            "rest": {
                "name": "core",
                "used": 5,
                "limit": 5000,
                "resets_at": "2026-04-19T13:00:00+00:00",
            },
            "graphql": {
                "name": "graphql",
                "used": 12,
                "limit": 5000,
                "resets_at": "2026-04-19T14:00:00+00:00",
            },
            "fetched_at": "2026-04-19T12:00:00+00:00",
        }
        info = _parse_rate_limit(raw)
        assert info is not None
        assert info.rest.used == 5
        assert info.graphql.used == 12
        assert info.fetched_at.year == 2026

    def test_window_falls_back_to_epoch_when_resets_at_invalid(self) -> None:
        raw = {
            "rest": {"name": "core", "used": 0, "limit": 5000, "resets_at": "garbage"},
            "graphql": {"name": "graphql", "used": 0, "limit": 5000},
            "fetched_at": "2026-04-19T12:00:00+00:00",
        }
        info = _parse_rate_limit(raw)
        assert info is not None
        assert info.rest.resets_at == datetime.fromtimestamp(0, tz=timezone.utc)
        assert info.graphql.resets_at == datetime.fromtimestamp(0, tz=timezone.utc)


class TestFormatDurationUntil:
    def test_past_target_renders_now(self) -> None:
        past = datetime(2020, 1, 1, tzinfo=timezone.utc)
        assert _format_duration_until(past) == "now"

    def test_future_seconds(self) -> None:
        now = datetime(2026, 4, 19, tzinfo=timezone.utc)
        assert _format_duration_until(now.replace(second=30), now=now) == "30s"

    def test_future_minutes(self) -> None:
        now = datetime(2026, 4, 19, 12, 0, tzinfo=timezone.utc)
        assert _format_duration_until(now.replace(minute=10), now=now) == "10m"

    def test_future_hours(self) -> None:
        now = datetime(2026, 4, 19, 12, 0, tzinfo=timezone.utc)
        target = datetime(2026, 4, 19, 16, 23, tzinfo=timezone.utc)
        assert _format_duration_until(target, now=now) == "4h23m"


class TestRateLimitColor:
    def test_dim_when_plenty_remaining(self) -> None:
        from fido.color import DIM

        assert _rate_limit_color(_window(used=100, limit=5000)) == DIM

    def test_yellow_when_at_or_below_25_percent(self) -> None:
        from fido.color import YELLOW

        # 25% remaining (75% used)
        assert _rate_limit_color(_window(used=3750, limit=5000)) == YELLOW

    def test_red_when_at_or_below_5_percent(self) -> None:
        from fido.color import RED_BOLD

        # 5% remaining (95% used)
        assert _rate_limit_color(_window(used=4750, limit=5000)) == RED_BOLD

    def test_dim_when_zero_limit(self) -> None:
        # zero-limit (no data yet) — pct_remaining is 0.0, which is ≤ 5
        # so it actually goes RED. That's fine — it signals "no data".
        from fido.color import RED_BOLD

        assert _rate_limit_color(_window(used=0, limit=0)) == RED_BOLD


class TestFormatRateLimitWindow:
    def test_includes_label_used_limit_and_reset_text(self) -> None:
        w = _window(used=12, limit=5000)
        out = _format_rate_limit_window(w, "REST")
        assert "REST" in out
        assert "12/5000" in out
        assert "to reset" in out


class TestFormatRateLimitLine:
    def test_returns_none_when_no_rate_limit(self) -> None:
        assert _format_rate_limit_line(None) is None

    def test_renders_both_windows(self) -> None:
        info = RateLimitInfo(
            rest=_window("core", used=5, limit=5000),
            graphql=_window("graphql", used=10, limit=5000),
            fetched_at=datetime.now(tz=timezone.utc),
        )
        out = _format_rate_limit_line(info)
        assert out is not None
        assert "GitHub:" in out
        assert "REST" in out
        assert "GraphQL" in out
        assert "5/5000" in out
        assert "10/5000" in out


class TestFormatStatusRateLimitIntegration:
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
            crash_count=0,
            last_crash_error=None,
            worker_stuck=False,
        )
        defaults.update(kwargs)
        return RepoStatus(**defaults)

    def test_format_status_includes_github_line_when_present(self) -> None:
        info = RateLimitInfo(
            rest=_window("core", used=5, limit=5000),
            graphql=_window("graphql", used=12, limit=5000),
            fetched_at=datetime.now(tz=timezone.utc),
        )
        status = FidoStatus(
            fido_pid=None,
            fido_uptime=None,
            repos=[self._repo()],
            rate_limit=info,
        )
        out = format_status(status)
        assert "GitHub:" in out

    def test_format_status_omits_github_line_when_absent(self) -> None:
        status = FidoStatus(
            fido_pid=None,
            fido_uptime=None,
            repos=[self._repo()],
            rate_limit=None,
        )
        out = format_status(status)
        assert "GitHub:" not in out


class TestFetchActivitiesRateLimit:
    def _wrap(self, activities: list, rate_limit: dict | None) -> bytes:
        return json.dumps({"activities": activities, "rate_limit": rate_limit}).encode()

    def _make_urlopen(self, data: bytes) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = data
        return MagicMock(return_value=mock_resp)

    def test_returns_none_when_rate_limit_missing(self) -> None:
        data = self._wrap([], None)
        _, info = _fetch_activities(9000, _urlopen=self._make_urlopen(data))
        assert info is None

    def test_parses_rate_limit_payload(self) -> None:
        rate = {
            "rest": {
                "name": "core",
                "used": 7,
                "limit": 5000,
                "resets_at": "2026-04-19T13:00:00+00:00",
            },
            "graphql": {
                "name": "graphql",
                "used": 22,
                "limit": 5000,
                "resets_at": "2026-04-19T14:00:00+00:00",
            },
            "fetched_at": "2026-04-19T12:00:00+00:00",
        }
        data = self._wrap([], rate)
        _, info = _fetch_activities(9000, _urlopen=self._make_urlopen(data))
        assert info is not None
        assert info.rest.used == 7
        assert info.graphql.used == 22
