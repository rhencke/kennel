import hashlib
import hmac
import json
import subprocess
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from http.server import HTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import ANY, MagicMock, patch

import pytest

from fido import provider
from fido.claude import ClaudeClient
from fido.config import Config
from fido.config import RepoConfig as _RepoConfig
from fido.events import (
    Action,
    recover_reply_promises,
    reply_to_comment,
    reply_to_issue_comment,
)
from fido.infra import Infra
from fido.provider import ProviderID
from fido.server import PreflightError, WebhookHandler, _repo_status
from fido.store import FidoStore


class RepoConfig(_RepoConfig):
    def __init__(self, *args, provider: ProviderID = ProviderID.CLAUDE_CODE, **kwargs):
        super().__init__(*args, provider=provider, **kwargs)


def _client(return_value: str = "", *, side_effect=None) -> MagicMock:
    client = MagicMock(spec=ClaudeClient)
    client.voice_model = "claude-opus-4-6"
    client.work_model = "claude-sonnet-4-6"
    client.brief_model = "claude-haiku-4-5"
    if side_effect is not None:
        client.run_turn.side_effect = side_effect
    else:
        client.run_turn.return_value = return_value
    return client


# Thread-capture and do_POST synchronisation helpers ---------------------------
#
# The wfile socket is unbuffered (wbufsize=0), so HTTP response bytes reach the
# client the instant _respond() writes them — before do_POST() finishes.  Tests
# that assert on work done after the ack (background threads, _self_restart)
# therefore cannot rely on urlopen() returning to mean "server is done".
#
# Solution: _restore_handler_fns (autouse) overrides two injectables:
#   _fn_after_do_post → signals _post_done so _post_webhook can wait
#   _fn_spawn_bg      → captures the background thread so _post_webhook can join
#
# _post_webhook:  wait for _post_done → join any captured threads → return status
#
_bg_threads: list[threading.Thread] = []
_bg_threads_lock: threading.Lock = threading.Lock()
_post_done: threading.Event = threading.Event()


def _capturing_spawn_bg(fn: Callable[..., Any], args: tuple[Any, ...]) -> None:
    t = threading.Thread(target=fn, args=args, daemon=True)
    t.start()
    with _bg_threads_lock:
        _bg_threads.append(t)


def _signal_post_done() -> None:
    _post_done.set()


# ------------------------------------------------------------------------------


def _config(tmp_path: Path) -> Config:
    from fido.config import RepoMembership

    return Config(
        port=0,  # will bind to random port
        secret=b"test-secret",
        repos={
            "owner/repo": RepoConfig(
                name="owner/repo",
                work_dir=tmp_path,
                membership=RepoMembership(collaborators=frozenset({"owner"})),
            ),
        },
        allowed_bots=frozenset({"copilot[bot]"}),
        log_level="WARNING",
        sub_dir=tmp_path / "sub",
    )


def _sign(body: bytes, secret: bytes) -> str:
    return "sha256=" + hmac.new(secret, body, hashlib.sha256).hexdigest()


@pytest.fixture(autouse=True)
def _restore_handler_fns():
    saved = {
        "gh": WebhookHandler.gh,
        "_fn_dispatch": WebhookHandler._fn_dispatch,
        "_fn_reply_to_comment": WebhookHandler._fn_reply_to_comment,
        "_fn_reply_to_review": WebhookHandler._fn_reply_to_review,
        "_fn_reply_to_issue_comment": WebhookHandler._fn_reply_to_issue_comment,
        "_fn_create_task": WebhookHandler._fn_create_task,
        "_fn_launch_worker": WebhookHandler._fn_launch_worker,
        "_fn_unblock_tasks": WebhookHandler._fn_unblock_tasks,
        "_fn_spawn_bg": WebhookHandler._fn_spawn_bg,
        "_fn_after_do_post": WebhookHandler._fn_after_do_post,
        "_fn_runner_dir": WebhookHandler._fn_runner_dir,
        "infra": WebhookHandler.infra,
        "static_files": WebhookHandler.static_files,
        "fido_started_at": WebhookHandler.fido_started_at,
    }
    # Override _fn_after_do_post for all tests so _post_webhook can wait for
    # do_POST to complete without sleeping (see module-level comment above).
    WebhookHandler._fn_after_do_post = staticmethod(  # type: ignore[assignment]
        _signal_post_done
    )
    _post_done.clear()
    yield
    with _bg_threads_lock:
        _bg_threads.clear()
    for attr, val in saved.items():
        setattr(WebhookHandler, attr, val)


@pytest.fixture(autouse=True)
def _stub_provider_statuses():
    with patch("fido.server.provider_statuses_for_repo_configs", return_value={}):
        yield


@pytest.fixture()
def server(tmp_path: Path):
    cfg = _config(tmp_path)
    WebhookHandler.config = cfg
    WebhookHandler.registry = MagicMock()
    WebhookHandler._fn_spawn_bg = staticmethod(_capturing_spawn_bg)  # type: ignore[assignment]
    srv = HTTPServer(("127.0.0.1", 0), WebhookHandler)
    port = srv.server_address[1]
    t = threading.Thread(
        target=srv.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    t.start()
    yield f"http://127.0.0.1:{port}", cfg
    srv.shutdown()


class TestSignatureVerification:
    def test_valid_signature(self, server: tuple) -> None:
        url, cfg = server
        body = json.dumps({"hook_id": 1}).encode()
        sig = _sign(body, cfg.secret)
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-GitHub-Event": "ping",
                "X-GitHub-Delivery": "test",
                "X-Hub-Signature-256": sig,
            },
        )
        resp = urllib.request.urlopen(req)
        assert resp.status == 200

    def test_bad_signature(self, server: tuple) -> None:
        url, cfg = server
        body = b'{"hook_id": 1}'
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-GitHub-Event": "ping",
                "X-GitHub-Delivery": "test",
                "X-Hub-Signature-256": "sha256=bad",
            },
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == 401

    def test_missing_signature(self, server: tuple) -> None:
        url, cfg = server
        body = b'{"hook_id": 1}'
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-GitHub-Event": "ping",
                "X-GitHub-Delivery": "test",
            },
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == 401


class TestGetEndpoint:
    def test_health_check(self, server: tuple) -> None:
        url, _ = server
        resp = urllib.request.urlopen(url)
        assert resp.status == 200
        assert b"fido is running" in resp.read()

    def test_status_endpoint_returns_activities(self, server: tuple) -> None:
        from datetime import datetime, timezone

        from fido.registry import WorkerActivity

        url, _ = server
        WebhookHandler.registry.get_all_activities.return_value = [
            WorkerActivity(
                repo_name="owner/repo",
                what="Working on: #1",
                busy=True,
                last_progress_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ]
        WebhookHandler.registry.get_crash_info.return_value = None
        WebhookHandler.registry.is_stale.return_value = False
        WebhookHandler.registry.thread_started_at.return_value = None
        WebhookHandler.registry.get_webhook_activities.return_value = []
        WebhookHandler.registry.get_session_owner.return_value = None
        WebhookHandler.registry.get_session_alive.return_value = False
        WebhookHandler.registry.get_session_pid.return_value = None
        WebhookHandler.registry.get_session_dropped_count.return_value = 0
        WebhookHandler.registry.is_rescoping.return_value = False
        resp = urllib.request.urlopen(f"{url}/status.json")
        assert resp.status == 200
        data = json.loads(resp.read())
        assert len(data["activities"]) == 1
        entry = data["activities"][0]
        assert entry["repo_name"] == "owner/repo"
        assert entry["what"] == "Working on: #1"
        assert entry["busy"] is True
        assert entry["crash_count"] == 0
        assert entry["last_crash_error"] is None
        assert entry["is_stuck"] is False
        assert entry["worker_uptime_seconds"] is None
        assert entry["webhook_activities"] == []
        assert entry["session_owner"] is None
        assert entry["session_dropped_count"] == 0

    def test_status_endpoint_includes_session_owner(self, server: tuple) -> None:
        from datetime import datetime, timezone

        from fido.registry import WorkerActivity

        url, _ = server
        WebhookHandler.registry.get_all_activities.return_value = [
            WorkerActivity(
                repo_name="owner/repo",
                what="Working on: #1",
                busy=True,
                last_progress_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ]
        WebhookHandler.registry.get_crash_info.return_value = None
        WebhookHandler.registry.is_stale.return_value = False
        WebhookHandler.registry.thread_started_at.return_value = None
        WebhookHandler.registry.get_webhook_activities.return_value = []
        WebhookHandler.registry.get_session_owner.return_value = "worker-home"
        WebhookHandler.registry.get_session_alive.return_value = True
        WebhookHandler.registry.get_session_pid.return_value = None
        WebhookHandler.registry.get_session_dropped_count.return_value = 3
        WebhookHandler.registry.is_rescoping.return_value = False
        resp = urllib.request.urlopen(f"{url}/status.json")
        data = json.loads(resp.read())
        assert data["activities"][0]["session_owner"] == "worker-home"
        assert data["activities"][0]["session_dropped_count"] == 3

    def test_status_endpoint_includes_session_alive(self, server: tuple) -> None:
        from datetime import datetime, timezone

        from fido.registry import WorkerActivity

        url, _ = server
        WebhookHandler.registry.get_all_activities.return_value = [
            WorkerActivity(
                repo_name="owner/repo",
                what="idle",
                busy=False,
                last_progress_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ]
        WebhookHandler.registry.get_crash_info.return_value = None
        WebhookHandler.registry.is_stale.return_value = False
        WebhookHandler.registry.thread_started_at.return_value = None
        WebhookHandler.registry.get_webhook_activities.return_value = []
        WebhookHandler.registry.get_session_owner.return_value = None
        WebhookHandler.registry.get_session_alive.return_value = True
        WebhookHandler.registry.get_session_pid.return_value = None
        WebhookHandler.registry.is_rescoping.return_value = False
        resp = urllib.request.urlopen(f"{url}/status.json")
        data = json.loads(resp.read())
        assert data["activities"][0]["session_alive"] is True
        assert data["activities"][0]["session_owner"] is None

    def test_status_endpoint_includes_crash_info(self, server: tuple) -> None:
        from datetime import datetime, timezone

        from fido.registry import WorkerActivity, WorkerCrash

        url, _ = server
        WebhookHandler.registry.get_all_activities.return_value = [
            WorkerActivity(
                repo_name="owner/repo",
                what="Napping",
                busy=False,
                last_progress_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ]
        WebhookHandler.registry.get_crash_info.return_value = WorkerCrash(
            death_count=3,
            last_error="RuntimeError: boom",
            last_crash_time=datetime(2026, 1, 1),
        )
        WebhookHandler.registry.is_stale.return_value = False
        WebhookHandler.registry.thread_started_at.return_value = None
        WebhookHandler.registry.get_webhook_activities.return_value = []
        WebhookHandler.registry.get_session_owner.return_value = None
        WebhookHandler.registry.get_session_alive.return_value = False
        WebhookHandler.registry.get_session_pid.return_value = None
        WebhookHandler.registry.is_rescoping.return_value = False
        resp = urllib.request.urlopen(f"{url}/status.json")
        data = json.loads(resp.read())
        assert data["activities"][0]["crash_count"] == 3
        assert data["activities"][0]["last_crash_error"] == "RuntimeError: boom"

    def test_status_endpoint_empty_when_no_activities(self, server: tuple) -> None:
        url, _ = server
        WebhookHandler.registry.get_all_activities.return_value = []
        resp = urllib.request.urlopen(f"{url}/status.json")
        assert resp.status == 200
        assert json.loads(resp.read()) == {
            "activities": [],
            "rate_limit": None,
            "fido_uptime_seconds": None,
        }

    def test_status_endpoint_content_type_json(self, server: tuple) -> None:
        url, _ = server
        WebhookHandler.registry.get_all_activities.return_value = []
        resp = urllib.request.urlopen(f"{url}/status.json")
        assert resp.headers.get("Content-Type") == "application/json"

    def test_status_endpoint_is_stuck_true_when_stale(self, server: tuple) -> None:
        from datetime import datetime, timezone

        from fido.registry import WorkerActivity

        url, _ = server
        WebhookHandler.registry.get_all_activities.return_value = [
            WorkerActivity(
                repo_name="owner/repo",
                what="Working on: #1",
                busy=True,
                last_progress_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ]
        WebhookHandler.registry.get_crash_info.return_value = None
        WebhookHandler.registry.is_stale.return_value = True
        WebhookHandler.registry.thread_started_at.return_value = None
        WebhookHandler.registry.get_webhook_activities.return_value = []
        WebhookHandler.registry.get_session_owner.return_value = None
        WebhookHandler.registry.get_session_alive.return_value = False
        WebhookHandler.registry.get_session_pid.return_value = None
        WebhookHandler.registry.is_rescoping.return_value = False
        resp = urllib.request.urlopen(f"{url}/status.json")
        data = json.loads(resp.read())
        assert data["activities"][0]["is_stuck"] is True

    def test_status_endpoint_includes_rescoping_flag(self, server: tuple) -> None:
        from datetime import datetime, timezone

        from fido.registry import WorkerActivity

        url, _ = server
        WebhookHandler.registry.get_all_activities.return_value = [
            WorkerActivity(
                repo_name="owner/repo",
                what="Working on: #1",
                busy=True,
                last_progress_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ]
        WebhookHandler.registry.get_crash_info.return_value = None
        WebhookHandler.registry.is_stale.return_value = False
        WebhookHandler.registry.thread_started_at.return_value = None
        WebhookHandler.registry.get_webhook_activities.return_value = []
        WebhookHandler.registry.get_session_owner.return_value = None
        WebhookHandler.registry.get_session_alive.return_value = False
        WebhookHandler.registry.get_session_pid.return_value = None
        WebhookHandler.registry.is_rescoping.return_value = True
        resp = urllib.request.urlopen(f"{url}/status.json")
        data = json.loads(resp.read())
        assert data["activities"][0]["rescoping"] is True

    def test_status_endpoint_includes_rate_limit_when_monitor_present(
        self, server: tuple
    ) -> None:
        """A real ``RateLimitMonitor`` with a refreshed snapshot serializes
        into ``/status.json`` under the top-level ``rate_limit`` key
        (closes #812 follow-up)."""
        from datetime import datetime, timezone

        from fido.rate_limit import RateLimitMonitor
        from fido.registry import WorkerActivity

        url, _ = server
        WebhookHandler.registry.get_all_activities.return_value = [
            WorkerActivity(
                repo_name="owner/repo",
                what="idle",
                busy=False,
                last_progress_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ]
        WebhookHandler.registry.get_crash_info.return_value = None
        WebhookHandler.registry.is_stale.return_value = False
        WebhookHandler.registry.thread_started_at.return_value = None
        WebhookHandler.registry.get_webhook_activities.return_value = []
        WebhookHandler.registry.get_session_owner.return_value = None
        WebhookHandler.registry.get_session_alive.return_value = False
        WebhookHandler.registry.get_session_pid.return_value = None
        WebhookHandler.registry.is_rescoping.return_value = False

        gh = MagicMock()
        gh.get_rate_limit.return_value = {
            "core": {"used": 5, "limit": 5000, "reset": 1700000000},
            "graphql": {"used": 12, "limit": 5000, "reset": 1700003600},
        }
        monitor = RateLimitMonitor(gh)
        monitor.refresh()
        WebhookHandler.rate_limit_monitor = monitor
        try:
            resp = urllib.request.urlopen(f"{url}/status.json")
            data = json.loads(resp.read())
            rl = data["rate_limit"]
            assert rl is not None
            assert rl["rest"]["used"] == 5
            assert rl["rest"]["limit"] == 5000
            assert rl["graphql"]["used"] == 12
            assert "fetched_at" in rl
        finally:
            WebhookHandler.rate_limit_monitor = None

    def test_status_endpoint_omits_rate_limit_when_monitor_absent(
        self, server: tuple
    ) -> None:
        """No monitor instance attached → ``rate_limit`` is ``None``."""
        from datetime import datetime, timezone

        from fido.registry import WorkerActivity

        url, _ = server
        WebhookHandler.registry.get_all_activities.return_value = [
            WorkerActivity(
                repo_name="owner/repo",
                what="idle",
                busy=False,
                last_progress_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ]
        WebhookHandler.registry.get_crash_info.return_value = None
        WebhookHandler.registry.is_stale.return_value = False
        WebhookHandler.registry.thread_started_at.return_value = None
        WebhookHandler.registry.get_webhook_activities.return_value = []
        WebhookHandler.registry.get_session_owner.return_value = None
        WebhookHandler.registry.get_session_alive.return_value = False
        WebhookHandler.registry.get_session_pid.return_value = None
        WebhookHandler.registry.is_rescoping.return_value = False
        WebhookHandler.rate_limit_monitor = None

        resp = urllib.request.urlopen(f"{url}/status.json")
        data = json.loads(resp.read())
        assert data["rate_limit"] is None

    def test_status_endpoint_omits_rate_limit_before_first_refresh(
        self, server: tuple
    ) -> None:
        """Monitor present but ``latest()`` returns None → rate_limit None."""
        from datetime import datetime, timezone

        from fido.rate_limit import RateLimitMonitor
        from fido.registry import WorkerActivity

        url, _ = server
        WebhookHandler.registry.get_all_activities.return_value = [
            WorkerActivity(
                repo_name="owner/repo",
                what="idle",
                busy=False,
                last_progress_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ]
        WebhookHandler.registry.get_crash_info.return_value = None
        WebhookHandler.registry.is_stale.return_value = False
        WebhookHandler.registry.thread_started_at.return_value = None
        WebhookHandler.registry.get_webhook_activities.return_value = []
        WebhookHandler.registry.get_session_owner.return_value = None
        WebhookHandler.registry.get_session_alive.return_value = False
        WebhookHandler.registry.get_session_pid.return_value = None
        WebhookHandler.registry.is_rescoping.return_value = False

        monitor = RateLimitMonitor(MagicMock())
        WebhookHandler.rate_limit_monitor = monitor
        try:
            resp = urllib.request.urlopen(f"{url}/status.json")
            data = json.loads(resp.read())
            assert data["rate_limit"] is None
        finally:
            WebhookHandler.rate_limit_monitor = None

    def test_status_endpoint_serializes_loaded_issue_cache(self, server: tuple) -> None:
        """Wire a real loaded :class:`IssueTreeCache` through the registry
        and verify the /status.json payload includes the cache snapshot
        (closes #812 status half).
        """
        from datetime import datetime, timezone

        from fido.issue_cache import IssueTreeCache
        from fido.registry import WorkerActivity

        url, _ = server
        WebhookHandler.registry.get_all_activities.return_value = [
            WorkerActivity(
                repo_name="owner/repo",
                what="Working on: #1",
                busy=True,
                last_progress_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ]
        WebhookHandler.registry.get_crash_info.return_value = None
        WebhookHandler.registry.is_stale.return_value = False
        WebhookHandler.registry.thread_started_at.return_value = None
        WebhookHandler.registry.get_webhook_activities.return_value = []
        WebhookHandler.registry.get_session_owner.return_value = None
        WebhookHandler.registry.get_session_alive.return_value = False
        WebhookHandler.registry.get_session_pid.return_value = None
        WebhookHandler.registry.is_rescoping.return_value = False

        cache = IssueTreeCache("owner/repo")
        cache.load_inventory(
            [
                {
                    "number": 7,
                    "title": "demo",
                    "createdAt": "2026-04-01T00:00:00Z",
                    "assignees": {"nodes": []},
                    "subIssues": {"nodes": []},
                }
            ],
            snapshot_started_at=datetime(2026, 4, 19, tzinfo=timezone.utc),
        )
        WebhookHandler.registry.get_issue_cache.return_value = cache

        resp = urllib.request.urlopen(f"{url}/status.json")
        data = json.loads(resp.read())
        cache_blob = data["activities"][0]["issue_cache"]
        assert cache_blob is not None
        assert cache_blob["loaded"] is True
        assert cache_blob["open_issues"] == 1
        assert cache_blob["events_applied"] == 0
        assert cache_blob["last_event_at"] is None
        assert cache_blob["last_reconcile_at"] is None
        assert cache_blob["last_reconcile_drift"] == 0

    def test_status_endpoint_omits_issue_cache_when_registry_returns_non_cache(
        self, server: tuple
    ) -> None:
        """A MagicMock registry (default in these tests) returns a Mock
        from ``get_issue_cache``; ``_serialize_issue_cache`` must reject
        it as non-cache and emit ``None`` rather than raise.
        """
        from datetime import datetime, timezone

        from fido.registry import WorkerActivity

        url, _ = server
        WebhookHandler.registry.get_all_activities.return_value = [
            WorkerActivity(
                repo_name="owner/repo",
                what="Working on: #1",
                busy=True,
                last_progress_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ]
        WebhookHandler.registry.get_crash_info.return_value = None
        WebhookHandler.registry.is_stale.return_value = False
        WebhookHandler.registry.thread_started_at.return_value = None
        WebhookHandler.registry.get_webhook_activities.return_value = []
        WebhookHandler.registry.get_session_owner.return_value = None
        WebhookHandler.registry.get_session_alive.return_value = False
        WebhookHandler.registry.get_session_pid.return_value = None
        WebhookHandler.registry.is_rescoping.return_value = False
        # registry.get_issue_cache returns a MagicMock by default, not a
        # real IssueTreeCache — must serialize to None.
        resp = urllib.request.urlopen(f"{url}/status.json")
        data = json.loads(resp.read())
        assert data["activities"][0]["issue_cache"] is None

    def test_status_endpoint_includes_provider_status(self, server: tuple) -> None:
        from datetime import UTC, datetime

        from fido.registry import WorkerActivity

        url, _ = server
        WebhookHandler.registry.get_all_activities.return_value = [
            WorkerActivity(
                repo_name="owner/repo",
                what="Working on: #1",
                busy=True,
                last_progress_at=datetime(2026, 1, 1, tzinfo=UTC),
            ),
        ]
        WebhookHandler.registry.get_crash_info.return_value = None
        WebhookHandler.registry.is_stale.return_value = False
        WebhookHandler.registry.thread_started_at.return_value = None
        WebhookHandler.registry.get_webhook_activities.return_value = []
        WebhookHandler.registry.get_session_owner.return_value = None
        WebhookHandler.registry.get_session_alive.return_value = False
        WebhookHandler.registry.get_session_pid.return_value = None
        WebhookHandler.registry.is_rescoping.return_value = False
        with patch(
            "fido.server.provider_statuses_for_repo_configs",
            return_value={
                ProviderID.CLAUDE_CODE: MagicMock(
                    provider=ProviderID.CLAUDE_CODE,
                    window_name="five_hour",
                    pressure=0.96,
                    percent_used=96,
                    resets_at=datetime(2026, 4, 16, 7, 0, tzinfo=UTC),
                    unavailable_reason=None,
                    level="paused",
                    warning=False,
                    paused=True,
                )
            },
        ):
            resp = urllib.request.urlopen(f"{url}/status.json")
        data = json.loads(resp.read())
        assert data["activities"][0]["provider"] == ProviderID.CLAUDE_CODE
        assert data["activities"][0]["provider_status"]["level"] == "paused"
        assert data["activities"][0]["provider_status"]["percent_used"] == 96

    def test_status_endpoint_rescoping_false_by_default(self, server: tuple) -> None:
        from datetime import datetime, timezone

        from fido.registry import WorkerActivity

        url, _ = server
        WebhookHandler.registry.get_all_activities.return_value = [
            WorkerActivity(
                repo_name="owner/repo",
                what="Napping",
                busy=False,
                last_progress_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ]
        WebhookHandler.registry.get_crash_info.return_value = None
        WebhookHandler.registry.is_stale.return_value = False
        WebhookHandler.registry.thread_started_at.return_value = None
        WebhookHandler.registry.get_webhook_activities.return_value = []
        WebhookHandler.registry.get_session_owner.return_value = None
        WebhookHandler.registry.get_session_alive.return_value = False
        WebhookHandler.registry.get_session_pid.return_value = None
        WebhookHandler.registry.is_rescoping.return_value = False
        resp = urllib.request.urlopen(f"{url}/status.json")
        data = json.loads(resp.read())
        assert data["activities"][0]["rescoping"] is False

    def test_status_endpoint_includes_fido_state_defaults_when_no_files(
        self, server: tuple
    ) -> None:
        """With no state.json or tasks.json on disk, fido_state fields default."""
        from datetime import datetime, timezone

        from fido.registry import WorkerActivity

        url, _ = server
        WebhookHandler.registry.get_all_activities.return_value = [
            WorkerActivity(
                repo_name="owner/repo",
                what="Working on: #1",
                busy=True,
                last_progress_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ]
        WebhookHandler.registry.get_crash_info.return_value = None
        WebhookHandler.registry.is_stale.return_value = False
        WebhookHandler.registry.thread_started_at.return_value = None
        WebhookHandler.registry.get_webhook_activities.return_value = []
        WebhookHandler.registry.get_session_owner.return_value = None
        WebhookHandler.registry.get_session_alive.return_value = False
        WebhookHandler.registry.get_session_pid.return_value = None
        WebhookHandler.registry.get_session_dropped_count.return_value = 0
        WebhookHandler.registry.is_rescoping.return_value = False
        resp = urllib.request.urlopen(f"{url}/status.json")
        data = json.loads(resp.read())
        entry = data["activities"][0]
        assert entry["fido_running"] is False
        assert entry["issue"] is None
        assert entry["issue_title"] is None
        assert entry["issue_elapsed_seconds"] is None
        assert entry["pr_number"] is None
        assert entry["pr_title"] is None
        assert entry["pending"] == 0
        assert entry["completed"] == 0
        assert entry["current_task"] is None
        assert entry["task_number"] is None
        assert entry["task_total"] is None

    def test_status_endpoint_includes_fido_state_from_files(
        self, server: tuple, tmp_path: Path
    ) -> None:
        """state.json and tasks.json on disk are surfaced in the activity entry."""
        from datetime import datetime, timezone

        from fido.registry import WorkerActivity

        # Write state.json
        fido_dir = tmp_path / ".git" / "fido"
        fido_dir.mkdir(parents=True)
        state = {
            "issue": 42,
            "issue_title": "Fix the thing",
            "issue_started_at": "2026-04-01T00:00:00+00:00",
            "pr_number": 99,
            "pr_title": "Fixes the thing",
        }
        (fido_dir / "state.json").write_text(json.dumps(state))

        # Write tasks.json
        tasks_path = fido_dir / "tasks.json"
        tasks = [
            {
                "id": "1",
                "title": "first task",
                "type": "spec",
                "status": "completed",
                "description": "",
            },
            {
                "id": "2",
                "title": "active task",
                "type": "spec",
                "status": "in_progress",
                "description": "",
            },
            {
                "id": "3",
                "title": "later task",
                "type": "spec",
                "status": "pending",
                "description": "",
            },
        ]
        tasks_path.write_text(json.dumps(tasks))

        url, _ = server
        WebhookHandler.registry.get_all_activities.return_value = [
            WorkerActivity(
                repo_name="owner/repo",
                what="Working on: #42",
                busy=True,
                last_progress_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ]
        WebhookHandler.registry.get_crash_info.return_value = None
        WebhookHandler.registry.is_stale.return_value = False
        WebhookHandler.registry.thread_started_at.return_value = None
        WebhookHandler.registry.get_webhook_activities.return_value = []
        WebhookHandler.registry.get_session_owner.return_value = None
        WebhookHandler.registry.get_session_alive.return_value = False
        WebhookHandler.registry.get_session_pid.return_value = None
        WebhookHandler.registry.get_session_dropped_count.return_value = 0
        WebhookHandler.registry.is_rescoping.return_value = False
        resp = urllib.request.urlopen(f"{url}/status.json")
        data = json.loads(resp.read())
        entry = data["activities"][0]
        assert entry["issue"] == 42
        assert entry["issue_title"] == "Fix the thing"
        assert entry["issue_elapsed_seconds"] is not None
        assert entry["issue_elapsed_seconds"] >= 0
        assert entry["pr_number"] == 99
        assert entry["pr_title"] == "Fixes the thing"
        assert entry["pending"] == 1
        assert entry["completed"] == 1
        assert entry["current_task"] == "active task"
        assert entry["task_number"] == 1
        assert entry["task_total"] == 2

    def test_status_endpoint_fido_state_defaults_when_repo_cfg_missing(
        self, server: tuple
    ) -> None:
        """When a repo has no config entry, fido_state fields are all defaults."""
        from datetime import datetime, timezone

        from fido.registry import WorkerActivity

        url, _ = server
        # Report an activity for a repo not in config
        WebhookHandler.registry.get_all_activities.return_value = [
            WorkerActivity(
                repo_name="unknown/repo",
                what="idle",
                busy=False,
                last_progress_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ]
        WebhookHandler.registry.get_crash_info.return_value = None
        WebhookHandler.registry.is_stale.return_value = False
        WebhookHandler.registry.thread_started_at.return_value = None
        WebhookHandler.registry.get_webhook_activities.return_value = []
        WebhookHandler.registry.get_session_owner.return_value = None
        WebhookHandler.registry.get_session_alive.return_value = False
        WebhookHandler.registry.get_session_pid.return_value = None
        WebhookHandler.registry.get_session_dropped_count.return_value = 0
        WebhookHandler.registry.is_rescoping.return_value = False
        resp = urllib.request.urlopen(f"{url}/status.json")
        data = json.loads(resp.read())
        entry = data["activities"][0]
        assert entry["fido_running"] is False
        assert entry["issue"] is None
        assert entry["pending"] == 0
        assert entry["task_number"] is None


class TestCollectFidoState:
    """Unit tests for _collect_fido_state covering edge-case branches."""

    def _now(self) -> Any:
        from datetime import datetime, timezone

        return datetime(2026, 4, 19, 12, 0, 0, tzinfo=timezone.utc)

    def test_defaults_when_no_files(self, tmp_path: Path) -> None:
        from fido.server import (
            _collect_fido_state,  # pyright: ignore[reportPrivateUsage]
        )

        result = _collect_fido_state(tmp_path, self._now())
        assert result["fido_running"] is False
        assert result["issue"] is None
        assert result["issue_title"] is None
        assert result["issue_elapsed_seconds"] is None
        assert result["pr_number"] is None
        assert result["pr_title"] is None
        assert result["pending"] == 0
        assert result["completed"] == 0
        assert result["current_task"] is None
        assert result["task_number"] is None
        assert result["task_total"] is None

    def test_lock_file_exists_not_held(self, tmp_path: Path) -> None:
        """Lock file exists but is not held by another process → fido_running=False."""
        from fido.server import (
            _collect_fido_state,  # pyright: ignore[reportPrivateUsage]
        )

        fido_dir = tmp_path / ".git" / "fido"
        fido_dir.mkdir(parents=True)
        (fido_dir / "lock").touch()
        result = _collect_fido_state(tmp_path, self._now())
        assert result["fido_running"] is False

    def test_lock_file_held(self, tmp_path: Path) -> None:
        """Lock file held by another thread → fido_running=True."""
        import fcntl
        import threading

        from fido.server import (
            _collect_fido_state,  # pyright: ignore[reportPrivateUsage]
        )

        fido_dir = tmp_path / ".git" / "fido"
        fido_dir.mkdir(parents=True)
        lock_path = fido_dir / "lock"
        lock_path.touch()

        ready = threading.Event()
        release = threading.Event()

        def hold_lock() -> None:
            with open(lock_path) as fd:
                fcntl.flock(fd, fcntl.LOCK_EX)
                ready.set()
                release.wait(timeout=5)
                fcntl.flock(fd, fcntl.LOCK_UN)

        t = threading.Thread(target=hold_lock, daemon=True)
        t.start()
        ready.wait(timeout=5)
        try:
            result = _collect_fido_state(tmp_path, self._now())
            assert result["fido_running"] is True
        finally:
            release.set()
            t.join(timeout=5)

    def test_lock_file_oserror(self, tmp_path: Path) -> None:
        """If opening the lock file raises OSError, fido_running stays False."""
        from fido.server import (
            _collect_fido_state,  # pyright: ignore[reportPrivateUsage]
        )

        fido_dir = tmp_path / ".git" / "fido"
        fido_dir.mkdir(parents=True)
        # Create a directory at the lock path to cause OSError on open()
        (fido_dir / "lock").mkdir()
        result = _collect_fido_state(tmp_path, self._now())
        assert result["fido_running"] is False

    def test_state_load_exception(self, tmp_path: Path) -> None:
        """If State.load() raises, state fields default gracefully."""
        from unittest.mock import patch

        from fido.server import (
            _collect_fido_state,  # pyright: ignore[reportPrivateUsage]
        )

        with patch("fido.server.State") as mock_state_cls:
            mock_state_cls.return_value.load.side_effect = RuntimeError("disk error")
            result = _collect_fido_state(tmp_path, self._now())
        assert result["issue"] is None
        assert result["issue_title"] is None
        assert result["pr_number"] is None

    def test_invalid_issue_started_at(self, tmp_path: Path) -> None:
        """Malformed issue_started_at is silently ignored."""
        from fido.server import (
            _collect_fido_state,  # pyright: ignore[reportPrivateUsage]
        )

        fido_dir = tmp_path / ".git" / "fido"
        fido_dir.mkdir(parents=True)
        (fido_dir / "state.json").write_text(
            json.dumps({"issue": 1, "issue_started_at": "not-a-date"})
        )
        result = _collect_fido_state(tmp_path, self._now())
        assert result["issue"] == 1
        assert result["issue_elapsed_seconds"] is None

    def test_tasks_load_exception(self, tmp_path: Path) -> None:
        """If Tasks.list() raises, task fields default gracefully."""
        from unittest.mock import patch

        from fido.server import (
            _collect_fido_state,  # pyright: ignore[reportPrivateUsage]
        )

        with patch("fido.server.Tasks") as mock_tasks_cls:
            mock_tasks_cls.return_value.list.side_effect = RuntimeError("disk error")
            result = _collect_fido_state(tmp_path, self._now())
        assert result["pending"] == 0
        assert result["current_task"] is None
        assert result["task_number"] is None

    def test_current_task_from_pending_when_no_in_progress(
        self, tmp_path: Path
    ) -> None:
        """current_task comes from the first pending task when none are in_progress."""
        from fido.server import (
            _collect_fido_state,  # pyright: ignore[reportPrivateUsage]
        )

        fido_dir = tmp_path / ".git" / "fido"
        fido_dir.mkdir(parents=True)
        tasks = [
            {
                "id": "1",
                "title": "do this",
                "type": "spec",
                "status": "pending",
                "description": "",
            },
            {
                "id": "2",
                "title": "then that",
                "type": "spec",
                "status": "pending",
                "description": "",
            },
        ]
        (fido_dir / "tasks.json").write_text(json.dumps(tasks))
        result = _collect_fido_state(tmp_path, self._now())
        assert result["current_task"] == "do this"
        assert result["task_number"] == 1
        assert result["task_total"] == 2

    def test_task_number_from_in_progress(self, tmp_path: Path) -> None:
        """task_number correctly reflects the in_progress task's position."""
        from fido.server import (
            _collect_fido_state,  # pyright: ignore[reportPrivateUsage]
        )

        fido_dir = tmp_path / ".git" / "fido"
        fido_dir.mkdir(parents=True)
        tasks = [
            {
                "id": "1",
                "title": "done",
                "type": "spec",
                "status": "completed",
                "description": "",
            },
            {
                "id": "2",
                "title": "first pending",
                "type": "spec",
                "status": "pending",
                "description": "",
            },
            {
                "id": "3",
                "title": "active",
                "type": "spec",
                "status": "in_progress",
                "description": "",
            },
            {
                "id": "4",
                "title": "later",
                "type": "spec",
                "status": "pending",
                "description": "",
            },
        ]
        (fido_dir / "tasks.json").write_text(json.dumps(tasks))
        result = _collect_fido_state(tmp_path, self._now())
        assert result["current_task"] == "active"
        assert result["task_number"] == 2  # position in non-completed list
        assert result["task_total"] == 3


class TestRepoStatus:
    @pytest.mark.parametrize(
        ("act", "expected"),
        [
            (
                {
                    "provider_status": {"paused": True},
                    "is_stuck": False,
                    "crash_count": 0,
                    "busy": True,
                },
                "paused",
            ),
            ({"is_stuck": True, "crash_count": 2, "busy": True}, "stuck"),
            ({"is_stuck": False, "crash_count": 3, "busy": True}, "crashed"),
            ({"is_stuck": False, "crash_count": 0, "busy": True}, "busy"),
            ({"is_stuck": False, "crash_count": 0, "busy": False}, "waiting"),
            (
                {
                    "is_stuck": False,
                    "crash_count": 0,
                    "busy": False,
                    "what": "waiting: no issues found",
                },
                "waiting: no issues found",
            ),
            (
                {
                    "is_stuck": False,
                    "crash_count": 0,
                    "busy": False,
                    "what": "scanning for work",
                },
                "scanning for work",
            ),
        ],
        ids=[
            "paused",
            "stuck",
            "crashed",
            "busy",
            "waiting",
            "waiting-what",
            "scanning",
        ],
    )
    def test_repo_status_priority(self, act: dict, expected: str) -> None:
        assert _repo_status(act) == expected


class TestStatusXml:
    def test_status_returns_namespaced_xml_with_xslt_pi(self, server: tuple) -> None:
        url, _ = server
        WebhookHandler.registry.get_all_activities.return_value = []
        resp = urllib.request.urlopen(f"{url}/status")
        body = resp.read().decode()
        assert resp.headers.get("Content-Type") == "application/xml; charset=utf-8"
        assert '<?xml version="1.0" encoding="UTF-8"?>' in body
        assert '<?xml-stylesheet type="text/xsl" href="/static/status.xsl"?>' in body
        assert "<fido" in body
        assert 'xmlns="https://fidocancode.dog/fido"' in body

    def test_status_xml_contains_repo_data_with_namespaces(self, server: tuple) -> None:
        from datetime import datetime, timezone

        from fido.registry import WorkerActivity

        url, _ = server
        WebhookHandler.registry.get_all_activities.return_value = [
            WorkerActivity(
                repo_name="owner/repo",
                what="Working on: #1",
                busy=True,
                last_progress_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ]
        WebhookHandler.registry.get_crash_info.return_value = None
        WebhookHandler.registry.is_stale.return_value = False
        WebhookHandler.registry.thread_started_at.return_value = None
        WebhookHandler.registry.get_webhook_activities.return_value = []
        WebhookHandler.registry.get_session_owner.return_value = None
        WebhookHandler.registry.get_session_alive.return_value = False
        WebhookHandler.registry.get_session_pid.return_value = None
        WebhookHandler.registry.is_rescoping.return_value = False
        resp = urllib.request.urlopen(f"{url}/status")
        body = resp.read().decode()
        assert "<repo_name>owner/repo</repo_name>" in body
        assert "<what>Working on: #1</what>" in body
        assert "<busy>true</busy>" in body
        assert 'dog:status="busy"' in body

    def test_status_xml_empty_fido(self, server: tuple) -> None:
        url, _ = server
        WebhookHandler.registry.get_all_activities.return_value = []
        resp = urllib.request.urlopen(f"{url}/status")
        body = resp.read().decode()
        assert 'xmlns="https://fidocancode.dog/fido"' in body
        # Empty fido — self-closing root element (with namespace attrs)
        assert "/>" in body

    def test_status_xml_includes_claude_talker(self, server: tuple) -> None:
        from datetime import datetime, timezone

        from fido.provider import SessionTalker
        from fido.registry import WorkerActivity

        url, _ = server
        talker = SessionTalker(
            repo_name="owner/repo",
            thread_id=42,
            kind="worker",
            description="implementing task",
            claude_pid=9999,
            started_at=datetime(2026, 4, 14, 16, 0, tzinfo=timezone.utc),
        )
        WebhookHandler.registry.get_all_activities.return_value = [
            WorkerActivity(
                repo_name="owner/repo",
                what="working",
                busy=True,
                last_progress_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ]
        WebhookHandler.registry.get_crash_info.return_value = None
        WebhookHandler.registry.is_stale.return_value = False
        WebhookHandler.registry.thread_started_at.return_value = None
        WebhookHandler.registry.get_webhook_activities.return_value = []
        WebhookHandler.registry.get_session_owner.return_value = None
        WebhookHandler.registry.get_session_alive.return_value = False
        WebhookHandler.registry.get_session_pid.return_value = None
        WebhookHandler.registry.is_rescoping.return_value = False
        with patch("fido.provider.get_talker", return_value=talker):
            resp = urllib.request.urlopen(f"{url}/status")
        body = resp.read().decode()
        assert "<kind>worker</kind>" in body
        assert "<claude_pid>9999</claude_pid>" in body

    def test_status_xml_includes_provider_status(self, server: tuple) -> None:
        from datetime import UTC, datetime

        from fido.registry import WorkerActivity

        url, _ = server
        WebhookHandler.registry.get_all_activities.return_value = [
            WorkerActivity(
                repo_name="owner/repo",
                what="working",
                busy=False,
                last_progress_at=datetime(2026, 1, 1, tzinfo=UTC),
            ),
        ]
        WebhookHandler.registry.get_crash_info.return_value = None
        WebhookHandler.registry.is_stale.return_value = False
        WebhookHandler.registry.thread_started_at.return_value = None
        WebhookHandler.registry.get_webhook_activities.return_value = []
        WebhookHandler.registry.get_session_owner.return_value = None
        WebhookHandler.registry.get_session_alive.return_value = False
        WebhookHandler.registry.get_session_pid.return_value = None
        WebhookHandler.registry.is_rescoping.return_value = False
        with patch(
            "fido.server.provider_statuses_for_repo_configs",
            return_value={
                ProviderID.CLAUDE_CODE: MagicMock(
                    provider=ProviderID.CLAUDE_CODE,
                    window_name="five_hour",
                    pressure=0.96,
                    percent_used=96,
                    resets_at=datetime(2026, 4, 16, 7, 0, tzinfo=UTC),
                    unavailable_reason=None,
                    level="paused",
                    warning=False,
                    paused=True,
                )
            },
        ):
            resp = urllib.request.urlopen(f"{url}/status")
        body = resp.read().decode()
        assert "<provider>claude-code</provider>" in body
        assert "<provider_status>" in body
        assert "<percent_used>96</percent_used>" in body

    def test_status_xml_includes_webhooks(self, server: tuple) -> None:
        from datetime import datetime, timezone

        from fido.registry import WebhookActivity, WorkerActivity

        url, _ = server
        WebhookHandler.registry.get_all_activities.return_value = [
            WorkerActivity(
                repo_name="owner/repo",
                what="working",
                busy=True,
                last_progress_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ]
        WebhookHandler.registry.get_crash_info.return_value = None
        WebhookHandler.registry.is_stale.return_value = False
        WebhookHandler.registry.thread_started_at.return_value = None
        WebhookHandler.registry.get_webhook_activities.return_value = [
            WebhookActivity(
                handle_id=1,
                description="replying to review",
                started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                thread_id=789,
            ),
        ]
        WebhookHandler.registry.get_session_owner.return_value = None
        WebhookHandler.registry.get_session_alive.return_value = False
        WebhookHandler.registry.get_session_pid.return_value = None
        WebhookHandler.registry.is_rescoping.return_value = False
        resp = urllib.request.urlopen(f"{url}/status")
        body = resp.read().decode()
        assert "<description>replying to review</description>" in body
        assert "<thread_id>789</thread_id>" in body

    def test_status_xml_includes_fido_uptime(self, server: tuple) -> None:
        """<fido_uptime_seconds> appears in the root element when fido_started_at is set."""
        from datetime import datetime, timezone

        url, _ = server
        WebhookHandler.registry.get_all_activities.return_value = []
        WebhookHandler.fido_started_at = datetime(
            2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc
        )
        resp = urllib.request.urlopen(f"{url}/status")
        body = resp.read().decode()
        assert "<fido_uptime_seconds>" in body

    def test_status_xml_no_fido_uptime_when_not_started(self, server: tuple) -> None:
        """<fido_uptime_seconds> is absent when fido_started_at is None (default)."""
        url, _ = server
        WebhookHandler.registry.get_all_activities.return_value = []
        assert WebhookHandler.fido_started_at is None
        resp = urllib.request.urlopen(f"{url}/status")
        body = resp.read().decode()
        assert "<fido_uptime_seconds>" not in body

    def test_status_xml_includes_rate_limit(self, server: tuple) -> None:
        """<rate_limit> with nested windows appears in the root element when available."""
        from datetime import UTC, datetime

        from fido.rate_limit import (
            RateLimitMonitor,
            RateLimitSnapshot,
            RateLimitWindow,
        )

        url, _ = server
        WebhookHandler.registry.get_all_activities.return_value = []
        snap = RateLimitSnapshot(
            rest=RateLimitWindow(
                name="rest",
                used=100,
                limit=5000,
                resets_at=datetime(2026, 4, 19, 13, 0, tzinfo=UTC),
            ),
            graphql=RateLimitWindow(
                name="graphql",
                used=5,
                limit=5000,
                resets_at=datetime(2026, 4, 19, 13, 0, tzinfo=UTC),
            ),
            fetched_at=datetime(2026, 4, 19, 12, 0, tzinfo=UTC),
        )
        monitor = MagicMock(spec=RateLimitMonitor)
        monitor.latest.return_value = snap
        WebhookHandler.rate_limit_monitor = monitor
        resp = urllib.request.urlopen(f"{url}/status")
        body = resp.read().decode()
        assert "<rate_limit>" in body
        assert "<rest>" in body
        assert "<used>100</used>" in body
        assert "<graphql>" in body
        assert "<used>5</used>" in body

    def test_status_json_includes_fido_uptime(self, server: tuple) -> None:
        """fido_uptime_seconds appears in /status.json when fido_started_at is set."""
        from datetime import datetime, timezone

        url, _ = server
        WebhookHandler.registry.get_all_activities.return_value = []
        WebhookHandler.fido_started_at = datetime(
            2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc
        )
        resp = urllib.request.urlopen(f"{url}/status.json")
        data = json.loads(resp.read())
        assert data["fido_uptime_seconds"] is not None
        assert data["fido_uptime_seconds"] >= 0

    def test_status_json_fido_uptime_null_when_not_started(self, server: tuple) -> None:
        """fido_uptime_seconds is null in /status.json when fido_started_at is None."""
        url, _ = server
        WebhookHandler.registry.get_all_activities.return_value = []
        assert WebhookHandler.fido_started_at is None
        resp = urllib.request.urlopen(f"{url}/status.json")
        data = json.loads(resp.read())
        assert data["fido_uptime_seconds"] is None

    def test_status_xml_includes_issue_cache_as_nested_elements(
        self, server: tuple
    ) -> None:
        """issue_cache dict is emitted as nested XML children, not as str(dict)."""
        from datetime import datetime, timezone

        from fido.issue_cache import IssueTreeCache
        from fido.registry import WorkerActivity

        url, _ = server
        WebhookHandler.registry.get_all_activities.return_value = [
            WorkerActivity(
                repo_name="owner/repo",
                what="working",
                busy=True,
                last_progress_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ]
        WebhookHandler.registry.get_crash_info.return_value = None
        WebhookHandler.registry.is_stale.return_value = False
        WebhookHandler.registry.thread_started_at.return_value = None
        WebhookHandler.registry.get_webhook_activities.return_value = []
        WebhookHandler.registry.get_session_owner.return_value = None
        WebhookHandler.registry.get_session_alive.return_value = False
        WebhookHandler.registry.get_session_pid.return_value = None
        WebhookHandler.registry.is_rescoping.return_value = False

        cache = IssueTreeCache("owner/repo")
        cache.load_inventory(
            [
                {
                    "number": 3,
                    "title": "demo",
                    "createdAt": "2026-04-01T00:00:00Z",
                    "assignees": {"nodes": []},
                    "subIssues": {"nodes": []},
                }
            ],
            snapshot_started_at=datetime(2026, 4, 19, tzinfo=timezone.utc),
        )
        WebhookHandler.registry.get_issue_cache.return_value = cache

        resp = urllib.request.urlopen(f"{url}/status")
        body = resp.read().decode()
        assert "<issue_cache>" in body
        assert "<open_issues>1</open_issues>" in body
        assert "<loaded>true</loaded>" in body


class TestStaticFileServing:
    def test_serves_static_css(self, server: tuple, tmp_path: Path) -> None:
        url, _ = server
        static_dir = tmp_path / "static"
        static_dir.mkdir()
        (static_dir / "style.css").write_text("body { color: red; }")

        from fido.static_files import StaticFiles

        WebhookHandler.static_files = StaticFiles(static_dir)
        resp = urllib.request.urlopen(f"{url}/static/style.css")
        assert resp.status == 200
        assert resp.read() == b"body { color: red; }"
        assert resp.headers["Content-Type"] == "text/css; charset=utf-8"

    def test_static_includes_caching_headers(
        self, server: tuple, tmp_path: Path
    ) -> None:
        url, _ = server
        static_dir = tmp_path / "static"
        static_dir.mkdir()
        (static_dir / "style.css").write_text("x")

        from fido.static_files import StaticFiles

        WebhookHandler.static_files = StaticFiles(static_dir)
        resp = urllib.request.urlopen(f"{url}/static/style.css")
        assert resp.headers["ETag"] is not None
        assert resp.headers["Last-Modified"] is not None
        assert resp.headers["Cache-Control"] == "public, max-age=3600"

    def test_static_304_on_etag_match(self, server: tuple, tmp_path: Path) -> None:
        url, _ = server
        static_dir = tmp_path / "static"
        static_dir.mkdir()
        content = b"body { color: blue; }"
        (static_dir / "style.css").write_bytes(content)

        import hashlib

        from fido.static_files import StaticFiles

        WebhookHandler.static_files = StaticFiles(static_dir)
        etag = '"' + hashlib.sha256(content).hexdigest()[:16] + '"'
        req = urllib.request.Request(
            f"{url}/static/style.css",
            headers={"If-None-Match": etag},
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == 304

    def test_static_404_when_no_static_files(self, server: tuple) -> None:
        url, _ = server
        WebhookHandler.static_files = None
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"{url}/static/style.css")
        assert exc_info.value.code == 404

    def test_static_404_for_missing_file(self, server: tuple, tmp_path: Path) -> None:
        url, _ = server
        static_dir = tmp_path / "static"
        static_dir.mkdir()

        from fido.static_files import StaticFiles

        WebhookHandler.static_files = StaticFiles(static_dir)
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"{url}/static/nope.css")
        assert exc_info.value.code == 404


class TestEmptyBody:
    def test_returns_400(self, server: tuple) -> None:
        url, _ = server
        req = urllib.request.Request(
            url,
            data=b"",
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-GitHub-Event": "ping",
                "Content-Length": "0",
            },
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == 400


class TestInvalidJson:
    def test_returns_400(self, server: tuple) -> None:
        url, cfg = server
        body = b"not json at all"
        sig = _sign(body, cfg.secret)
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-GitHub-Event": "ping",
                "X-GitHub-Delivery": "test",
                "X-Hub-Signature-256": sig,
            },
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == 400


def _post_webhook(url: str, cfg: Config, event: str, payload: dict) -> int:
    body = json.dumps(payload).encode()
    sig = _sign(body, cfg.secret)
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-GitHub-Event": event,
            "X-GitHub-Delivery": "test",
            "X-Hub-Signature-256": sig,
        },
    )
    _post_done.clear()
    resp = urllib.request.urlopen(req)
    # The wfile socket is unbuffered: _respond() sends bytes immediately, so
    # urlopen() returns before do_POST() finishes.  Wait for _fn_after_do_post
    # to fire (set by _restore_handler_fns autouse fixture) before asserting.
    _post_done.wait(timeout=5.0)
    # Join any background thread captured by _capturing_spawn_bg.
    with _bg_threads_lock:
        threads = list(_bg_threads)
        _bg_threads.clear()
    for t in threads:
        t.join()
    return resp.status


class TestReplyPromiseKey:
    def test_returns_none_without_replyable_thread(self) -> None:
        handler = object.__new__(WebhookHandler)
        assert handler._reply_promise(Action(prompt="x")) is None

    def test_raises_for_invalid_thread_data(self) -> None:
        handler = object.__new__(WebhookHandler)
        action = Action(prompt="x", thread={"comment_type": "wat", "comment_id": "5"})
        with pytest.raises(ValueError, match="invalid reply promise comment type"):
            handler._reply_promise(action)

    def test_raises_for_non_integer_comment_id(self) -> None:
        handler = object.__new__(WebhookHandler)
        action = Action(prompt="x", thread={"comment_type": "pulls", "comment_id": "5"})
        with pytest.raises(TypeError, match="invalid reply promise comment id"):
            handler._reply_promise(action)

    def test_prepare_and_ack_ignore_non_replyable_action(self, tmp_path: Path) -> None:
        handler = object.__new__(WebhookHandler)
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        action = Action(prompt="x")

        assert handler._prepare_reply(repo_cfg, action) is None
        handler._ack_reply(repo_cfg, None)


class TestPatchIssueCache:
    """Tests for ``_patch_issue_cache`` — webhook → cache event patcher (#812)."""

    def _payload(self, repo_owner: str = "owner") -> dict:
        return {
            "repository": {
                "full_name": f"{repo_owner}/repo",
                "owner": {"login": repo_owner},
            },
        }

    def test_assigned_event_patches_cache(self, server: tuple) -> None:
        url, cfg = server
        cache = MagicMock()
        WebhookHandler.registry.get_issue_cache.return_value = cache
        payload = {
            **self._payload(),
            "action": "assigned",
            "issue": {
                "number": 42,
                "title": "x",
                "updated_at": "2026-04-18T22:00:00Z",
                "created_at": "2026-04-15T00:00:00Z",
            },
            "assignee": {"login": "fido"},
        }
        status = _post_webhook(url, cfg, "issues", payload)
        assert status == 200
        # _patch_issue_cache called → cache.apply_event called once with
        # the translator's ('assigned', payload) output.
        cache.apply_event.assert_called()
        args, _ = cache.apply_event.call_args
        assert args[0] == "assigned"
        assert args[1]["issue_number"] == 42
        assert args[1]["login"] == "fido"

    def test_non_issue_event_does_not_touch_cache(self, server: tuple) -> None:
        url, cfg = server
        cache = MagicMock()
        WebhookHandler.registry.get_issue_cache.return_value = cache
        payload = {
            **self._payload(),
            "action": "synchronize",
            "pull_request": {"number": 7},
        }
        status = _post_webhook(url, cfg, "pull_request", payload)
        assert status == 200
        cache.apply_event.assert_not_called()

    def test_irrelevant_issue_action_does_not_touch_cache(self, server: tuple) -> None:
        url, cfg = server
        cache = MagicMock()
        WebhookHandler.registry.get_issue_cache.return_value = cache
        payload = {
            **self._payload(),
            "action": "labeled",  # not a picker-relevant action
            "issue": {
                "number": 42,
                "title": "x",
                "updated_at": "2026-04-18T22:00:00Z",
                "created_at": "2026-04-15T00:00:00Z",
            },
        }
        status = _post_webhook(url, cfg, "issues", payload)
        assert status == 200
        cache.apply_event.assert_not_called()

    def test_cache_apply_failure_does_not_500(self, server: tuple) -> None:
        """Cache patching is best-effort — failure logs and returns 200
        so the dispatch + ack still happen and the hourly reconcile heals."""
        url, cfg = server
        cache = MagicMock()
        cache.apply_event.side_effect = RuntimeError("boom")
        WebhookHandler.registry.get_issue_cache.return_value = cache
        payload = {
            **self._payload(),
            "action": "assigned",
            "issue": {
                "number": 42,
                "title": "x",
                "updated_at": "2026-04-18T22:00:00Z",
                "created_at": "2026-04-15T00:00:00Z",
            },
            "assignee": {"login": "fido"},
        }
        status = _post_webhook(url, cfg, "issues", payload)
        assert status == 200


class TestProcessAction:
    """Tests for _process_action — the background thread that dispatches actions."""

    def _payload(self, repo_owner: str = "owner") -> dict:
        return {
            "repository": {
                "full_name": f"{repo_owner}/repo",
                "owner": {"login": repo_owner},
            },
        }

    def test_dispatch_triggers_worker_on_merged_pr(self, server: tuple) -> None:
        url, cfg = server
        payload = {
            **self._payload(),
            "action": "closed",
            "pull_request": {"number": 7, "merged": True},
        }
        mock_worker = MagicMock()
        WebhookHandler._fn_launch_worker = mock_worker
        WebhookHandler._fn_create_task = MagicMock()
        status = _post_webhook(url, cfg, "pull_request", payload)
        assert status == 200
        mock_worker.assert_called()

    def test_reply_to_comment_creates_task_for_act(self, server: tuple) -> None:
        url, cfg = server
        payload = {
            **self._payload(),
            "action": "created",
            "comment": {
                "id": 200,
                "body": "please add logging",
                "user": {"login": "owner"},
                "html_url": "https://example.com",
                "path": "foo.py",
                "line": 5,
                "diff_hunk": "@@ @@",
            },
            "pull_request": {"number": 3, "title": "My PR", "body": "desc"},
        }
        mock_reply = MagicMock(return_value=("ACT", ["add logging"]))
        mock_task = MagicMock()
        WebhookHandler._fn_reply_to_comment = mock_reply
        WebhookHandler._fn_create_task = mock_task
        WebhookHandler._fn_launch_worker = MagicMock()
        status = _post_webhook(url, cfg, "pull_request_review_comment", payload)
        assert status == 200
        mock_reply.assert_called()
        mock_task.assert_called()

    def test_reply_to_comment_no_task_for_dump(self, server: tuple) -> None:
        url, cfg = server
        payload = {
            **self._payload(),
            "action": "created",
            "comment": {
                "id": 201,
                "body": "use erlang instead",
                "user": {"login": "owner"},
                "html_url": "https://example.com",
                "path": "foo.py",
                "line": 1,
                "diff_hunk": "@@ @@",
            },
            "pull_request": {"number": 4, "title": "My PR", "body": ""},
        }
        mock_reply = MagicMock(return_value=("DUMP", ["nope"]))
        mock_task = MagicMock()
        WebhookHandler._fn_reply_to_comment = mock_reply
        WebhookHandler._fn_create_task = mock_task
        WebhookHandler._fn_launch_worker = MagicMock()
        status = _post_webhook(url, cfg, "pull_request_review_comment", payload)
        assert status == 200
        mock_reply.assert_called()
        mock_task.assert_not_called()

    def test_reply_to_comment_defer_skips_task(self, server: tuple) -> None:
        """DEFER files a GitHub issue instead — no tasks.json entry."""
        url, cfg = server
        payload = {
            **self._payload(),
            "action": "created",
            "comment": {
                "id": 202,
                "body": "refactor the whole module",
                "user": {"login": "owner"},
                "html_url": "https://example.com",
                "path": "foo.py",
                "line": 1,
                "diff_hunk": "@@ @@",
            },
            "pull_request": {"number": 5, "title": "My PR", "body": ""},
        }
        mock_task = MagicMock()
        WebhookHandler._fn_reply_to_comment = MagicMock(
            return_value=("DEFER", ["big refactor"])
        )
        WebhookHandler._fn_create_task = mock_task
        WebhookHandler._fn_launch_worker = MagicMock()
        status = _post_webhook(url, cfg, "pull_request_review_comment", payload)
        assert status == 200
        mock_task.assert_not_called()

    def test_reply_to_comment_failure_skips_task(self, server: tuple) -> None:
        """If reply posting raises, queue recovery and skip task creation."""
        url, cfg = server
        payload = {
            **self._payload(),
            "action": "created",
            "comment": {
                "id": 205,
                "body": "please add logging",
                "user": {"login": "owner"},
                "html_url": "https://example.com",
                "path": "foo.py",
                "line": 1,
                "diff_hunk": "@@ @@",
            },
            "pull_request": {"number": 5, "title": "My PR", "body": ""},
        }
        mock_task = MagicMock()
        WebhookHandler._fn_reply_to_comment = MagicMock(
            side_effect=RuntimeError("network down")
        )
        WebhookHandler._fn_create_task = mock_task
        WebhookHandler._fn_launch_worker = MagicMock()
        WebhookHandler.gh = MagicMock()
        status = _post_webhook(url, cfg, "pull_request_review_comment", payload)
        assert status == 200
        mock_task.assert_not_called()
        store = FidoStore(cfg.repos["owner/repo"].work_dir)
        promises = store.recoverable_promises()
        assert len(promises) == 1
        assert promises[0].anchor_comment_id == 205
        assert promises[0].state == "failed"
        assert store.claim_state(205) == "retryable_failed"

    def test_review_comment_promise_written_before_attempt(self, server: tuple) -> None:
        """Promise is durable before the reply attempt, not just on exception."""
        url, cfg = server
        store = FidoStore(cfg.repos["owner/repo"].work_dir)
        payload = {
            **self._payload(),
            "action": "created",
            "comment": {
                "id": 207,
                "body": "please add logging",
                "user": {"login": "owner"},
                "html_url": "https://example.com",
                "path": "foo.py",
                "line": 1,
                "diff_hunk": "@@ @@",
            },
            "pull_request": {"number": 5, "title": "My PR", "body": ""},
        }
        promise_existed_during_call = []

        def check_promise_during_call(*args, **kwargs):
            promise_existed_during_call.append(store.claim_state(207))
            return ("ACT", ["do the thing"])

        WebhookHandler._fn_reply_to_comment = check_promise_during_call
        WebhookHandler._fn_create_task = MagicMock()
        WebhookHandler._fn_launch_worker = MagicMock()
        WebhookHandler.gh = MagicMock()
        status = _post_webhook(url, cfg, "pull_request_review_comment", payload)
        assert status == 200
        assert promise_existed_during_call == ["in_progress"], (
            "promise must exist before attempt"
        )
        assert store.claim_state(207) == "completed"
        assert store.recoverable_promises() == []

    def test_review_comment_success_records_reply_artifact(self, server: tuple) -> None:
        url, cfg = server
        payload = {
            **self._payload(),
            "action": "created",
            "comment": {
                "id": 208,
                "body": "please add logging",
                "user": {"login": "owner"},
                "html_url": "https://github.com/owner/repo/pull/5#discussion_r208",
                "path": "foo.py",
                "line": 1,
                "diff_hunk": "@@ @@",
            },
            "pull_request": {"number": 5, "title": "My PR", "body": ""},
        }
        mock_gh = MagicMock()
        mock_gh.fetch_comment_thread.return_value = [
            {"id": 208, "body": "please add logging", "author": "owner"}
        ]
        mock_gh.reply_to_review_comment.return_value = {"id": 9208}
        WebhookHandler.gh = mock_gh
        WebhookHandler._fn_reply_to_comment = reply_to_comment
        WebhookHandler._fn_create_task = MagicMock()
        WebhookHandler._fn_launch_worker = MagicMock()

        def fake_pp(prompt, model, **kwargs):
            if model == "claude-haiku-4-5":
                return "NO"
            if "Triage" in prompt:
                return "ANSWER: looks good"
            return "One review reply."

        with (
            patch(
                "fido.events._configured_agent",
                return_value=_client(side_effect=fake_pp),
            ),
            patch("fido.events.maybe_react"),
        ):
            status = _post_webhook(url, cfg, "pull_request_review_comment", payload)

        assert status == 200
        store = FidoStore(cfg.repos["owner/repo"].work_dir)
        with store._connect() as conn:
            row = conn.execute(
                """
                SELECT a.artifact_comment_id, a.lane_key, COUNT(ap.promise_id) AS covered
                  FROM reply_artifacts AS a
                  JOIN reply_artifact_promises AS ap
                    ON ap.artifact_comment_id = a.artifact_comment_id
              GROUP BY a.artifact_comment_id, a.lane_key
                """,
            ).fetchone()
        assert row is not None
        assert int(row["artifact_comment_id"]) == 9208
        assert row["lane_key"] == "pulls:owner/repo:5:thread:208"
        assert int(row["covered"]) == 1
        assert store.claim_state(208) == "completed"

    def test_successful_redelivery_clears_stale_review_promise(
        self, server: tuple
    ) -> None:
        url, cfg = server
        payload = {
            **self._payload(),
            "action": "created",
            "comment": {
                "id": 206,
                "body": "please add logging",
                "user": {"login": "owner"},
                "html_url": "https://example.com",
                "path": "foo.py",
                "line": 1,
                "diff_hunk": "@@ @@",
            },
            "pull_request": {"number": 5, "title": "My PR", "body": ""},
        }
        mock_task = MagicMock()
        WebhookHandler._fn_reply_to_comment = MagicMock(
            side_effect=[RuntimeError("network down"), ("DO", ["from redelivery"])]
        )
        WebhookHandler._fn_create_task = mock_task
        WebhookHandler._fn_launch_worker = MagicMock()
        WebhookHandler.gh = MagicMock()
        assert _post_webhook(url, cfg, "pull_request_review_comment", payload) == 200
        assert _post_webhook(url, cfg, "pull_request_review_comment", payload) == 200
        store = FidoStore(cfg.repos["owner/repo"].work_dir)
        assert store.claim_state(206) == "retryable_failed"
        assert WebhookHandler._fn_reply_to_comment.call_count == 1
        mock_task.assert_not_called()

    def test_failed_review_comment_webhook_recovers_once_from_live_state(
        self, server: tuple
    ) -> None:
        url, cfg = server
        payload = {
            **self._payload(),
            "action": "created",
            "comment": {
                "id": 205,
                "body": "please add logging",
                "user": {"login": "owner"},
                "html_url": "https://example.com",
                "path": "foo.py",
                "line": 1,
                "diff_hunk": "@@ @@",
            },
            "pull_request": {"number": 5, "title": "My PR", "body": ""},
        }
        WebhookHandler._fn_reply_to_comment = MagicMock(
            side_effect=RuntimeError("network down")
        )
        WebhookHandler._fn_create_task = MagicMock()
        WebhookHandler._fn_launch_worker = MagicMock()
        WebhookHandler.gh = MagicMock()
        assert _post_webhook(url, cfg, "pull_request_review_comment", payload) == 200
        assert _post_webhook(url, cfg, "pull_request_review_comment", payload) == 200

        store = FidoStore(cfg.repos["owner/repo"].work_dir)
        assert [p.anchor_comment_id for p in store.recoverable_promises()] == [205]

        recovery_gh = MagicMock()
        recovery_gh.view_issue.return_value = {"title": "My PR", "body": "body"}
        recovery_gh.get_pull_comment.return_value = {
            "id": 205,
            "body": "edited after webhook",
            "path": "foo.py",
            "line": 2,
            "diff_hunk": "@@ @@",
            "pull_request_url": "https://api.github.com/repos/owner/repo/pulls/5",
            "html_url": "https://github.com/owner/repo/pull/5#discussion_r205",
            "user": {"login": "owner"},
        }

        def fake_reply(action, *args, **kwargs):
            assert action.comment_body == "edited after webhook"
            return ("DO", ["task from recovery"])

        with (
            patch("fido.events.reply_to_comment", side_effect=fake_reply) as mock_reply,
            patch("fido.events.create_task") as mock_create_task,
        ):
            assert recover_reply_promises(
                cfg.repos["owner/repo"].work_dir / ".git" / "fido",
                cfg,
                cfg.repos["owner/repo"],
                recovery_gh,
                5,
            )
            assert not recover_reply_promises(
                cfg.repos["owner/repo"].work_dir / ".git" / "fido",
                cfg,
                cfg.repos["owner/repo"],
                recovery_gh,
                5,
            )
        assert mock_reply.call_count == 1
        mock_create_task.assert_called_once_with(
            "task from recovery",
            cfg,
            cfg.repos["owner/repo"],
            recovery_gh,
            thread={
                "repo": "owner/repo",
                "pr": 5,
                "comment_id": 205,
                "url": "https://github.com/owner/repo/pull/5#discussion_r205",
                "author": "owner",
                "comment_type": "pulls",
            },
            registry=None,
        )
        assert store.recoverable_promises() == []

    def test_reply_to_comment_do_creates_task(self, server: tuple) -> None:
        """DO adds to tasks.json."""
        url, cfg = server
        payload = {
            **self._payload(),
            "action": "created",
            "comment": {
                "id": 204,
                "body": "cache the results",
                "user": {"login": "owner"},
                "html_url": "https://example.com",
                "path": "foo.py",
                "line": 1,
                "diff_hunk": "@@ @@",
            },
            "pull_request": {"number": 5, "title": "My PR", "body": ""},
        }
        task_titles = []

        def capture_task(title, *args, **kwargs):
            task_titles.append(title)

        WebhookHandler._fn_reply_to_comment = MagicMock(
            return_value=("DO", ["add result caching"])
        )
        WebhookHandler._fn_create_task = capture_task
        WebhookHandler._fn_launch_worker = MagicMock()
        status = _post_webhook(url, cfg, "pull_request_review_comment", payload)
        assert status == 200
        assert task_titles == ["add result caching"]

    def test_already_replied_comment_skipped(self, server: tuple) -> None:
        url, cfg = server
        promise = FidoStore(cfg.repos["owner/repo"].work_dir).prepare_reply(
            owner="webhook", comment_type="pulls", anchor_comment_id=203
        )
        assert promise is not None
        payload = {
            **self._payload(),
            "action": "created",
            "comment": {
                "id": 203,
                "body": "please add logging",
                "user": {"login": "owner"},
                "html_url": "https://example.com",
                "path": "foo.py",
                "line": 5,
                "diff_hunk": "@@ @@",
            },
            "pull_request": {"number": 6, "title": "My PR", "body": ""},
        }
        mock_reply = MagicMock()
        WebhookHandler._fn_reply_to_comment = mock_reply
        WebhookHandler._fn_create_task = MagicMock()
        WebhookHandler._fn_launch_worker = MagicMock()
        status = _post_webhook(url, cfg, "pull_request_review_comment", payload)
        assert status == 200
        mock_reply.assert_not_called()

    def test_preempted_review_comment_creates_no_phantom_task(
        self, server: tuple
    ) -> None:
        """When reply_to_comment returns ('ACT', []) due to per-comment lock
        contention (preempted reply), _process_action_inner must not enqueue any
        tasks — the process that holds the lock is responsible for reply and task
        creation."""
        url, cfg = server
        payload = {
            **self._payload(),
            "action": "created",
            "comment": {
                "id": 510,
                "body": "please rename this variable",
                "user": {"login": "owner"},
                "html_url": "https://example.com",
                "path": "foo.py",
                "line": 3,
                "diff_hunk": "@@ @@",
            },
            "pull_request": {"number": 51, "title": "My PR", "body": ""},
        }
        mock_task = MagicMock()
        # Simulate lock-contention: another process holds the lock, so
        # reply_to_comment returns ACT with empty titles instead of a task title.
        WebhookHandler._fn_reply_to_comment = MagicMock(return_value=("ACT", []))
        WebhookHandler._fn_create_task = mock_task
        WebhookHandler._fn_launch_worker = MagicMock()
        status = _post_webhook(url, cfg, "pull_request_review_comment", payload)
        assert status == 200
        mock_task.assert_not_called()

    def test_swallowed_comment_outcome_is_logged(
        self, server: tuple, caplog: pytest.LogCaptureFixture
    ) -> None:
        """_process_action_inner must emit an 'action outcome:' summary log for
        every processed comment.  Without this log a comment could be silently
        swallowed — consumed by the webhook handler without any visible trace."""
        import logging

        url, cfg = server
        payload = {
            **self._payload(),
            "action": "created",
            "comment": {
                "id": 511,
                "body": "looks good to me",
                "user": {"login": "owner"},
                "html_url": "https://github.com/owner/repo/pull/52#issuecomment-511",
            },
            "issue": {
                "number": 52,
                "title": "my pr",
                "body": "",
                "pull_request": {"url": "https://api.github.com/..."},
            },
        }
        WebhookHandler._fn_reply_to_issue_comment = MagicMock(
            return_value=("ANSWER", [])
        )
        WebhookHandler._fn_create_task = MagicMock()
        WebhookHandler._fn_launch_worker = MagicMock()
        with caplog.at_level(logging.INFO, logger="fido.server"):
            status = _post_webhook(url, cfg, "issue_comment", payload)
        assert status == 200
        assert "action outcome:" in caplog.text

    def test_duplicate_issue_comment_delivery_skips_second_reply(
        self, server: tuple
    ) -> None:
        url, cfg = server
        payload = {
            **self._payload(),
            "action": "created",
            "comment": {
                "id": 303,
                "body": "looks good",
                "user": {"login": "owner"},
                "html_url": "https://github.com/owner/repo/pull/11#issuecomment-303",
            },
            "issue": {
                "number": 11,
                "title": "my pr",
                "body": "",
                "pull_request": {"url": "https://api.github.com/..."},
            },
        }
        mock_gh = MagicMock()
        mock_ic = MagicMock(return_value=("ANSWER", ["because"]))
        WebhookHandler.gh = mock_gh
        WebhookHandler._fn_reply_to_issue_comment = mock_ic
        WebhookHandler._fn_create_task = MagicMock()
        WebhookHandler._fn_launch_worker = MagicMock()
        assert _post_webhook(url, cfg, "issue_comment", payload) == 200
        assert _post_webhook(url, cfg, "issue_comment", payload) == 200
        mock_ic.assert_called_once()

    def test_review_comments_handled(self, server: tuple) -> None:
        url, cfg = server
        payload = {
            **self._payload(),
            "action": "submitted",
            "review": {
                "id": 888,
                "state": "changes_requested",
                "user": {"login": "owner"},
            },
            "pull_request": {"number": 9},
        }
        mock_review = MagicMock()
        WebhookHandler._fn_reply_to_review = mock_review
        WebhookHandler._fn_launch_worker = MagicMock()
        status = _post_webhook(url, cfg, "pull_request_review", payload)
        assert status == 200
        mock_review.assert_called()

    def test_issue_comment_handled(self, server: tuple) -> None:
        url, cfg = server
        payload = {
            **self._payload(),
            "action": "created",
            "comment": {
                "id": 300,
                "body": "looks good",
                "user": {"login": "owner"},
                "html_url": "https://github.com/owner/repo/pull/11#issuecomment-300",
            },
            "issue": {
                "number": 11,
                "title": "my pr",
                "body": "",
                "pull_request": {"url": "https://api.github.com/..."},
            },
        }
        mock_gh = MagicMock()
        mock_ic = MagicMock(return_value=("ACT", ["do it"]))
        mock_task = MagicMock()
        WebhookHandler.gh = mock_gh
        WebhookHandler._fn_reply_to_issue_comment = mock_ic
        WebhookHandler._fn_create_task = mock_task
        WebhookHandler._fn_launch_worker = MagicMock()
        status = _post_webhook(url, cfg, "issue_comment", payload)
        assert status == 200
        mock_ic.assert_called()
        mock_task.assert_called_once_with(
            "do it",
            cfg,
            cfg.repos["owner/repo"],
            mock_gh,
            thread={
                "repo": "owner/repo",
                "pr": 11,
                "comment_id": 300,
                "url": "https://github.com/owner/repo/pull/11#issuecomment-300",
                "author": "owner",
                "comment_type": "issues",
            },
            registry=WebhookHandler.registry,
        )

    def test_issue_comment_no_task_for_answer(self, server: tuple) -> None:
        url, cfg = server
        payload = {
            **self._payload(),
            "action": "created",
            "comment": {"id": 301, "body": "why?", "user": {"login": "owner"}},
            "issue": {
                "number": 12,
                "title": "my pr",
                "body": "",
                "pull_request": {"url": "https://api.github.com/..."},
            },
        }
        mock_ic = MagicMock(return_value=("ANSWER", ["because"]))
        mock_task = MagicMock()
        WebhookHandler._fn_reply_to_issue_comment = mock_ic
        WebhookHandler._fn_create_task = mock_task
        WebhookHandler._fn_launch_worker = MagicMock()
        status = _post_webhook(url, cfg, "issue_comment", payload)
        assert status == 200
        mock_ic.assert_called()
        mock_task.assert_not_called()

    def test_reply_to_issue_comment_failure_skips_task(self, server: tuple) -> None:
        """If issue comment reply raises, queue recovery and skip task creation."""
        url, cfg = server
        payload = {
            **self._payload(),
            "action": "created",
            "comment": {"id": 302, "body": "please fix", "user": {"login": "owner"}},
            "issue": {
                "number": 13,
                "title": "my pr",
                "body": "",
                "pull_request": {"url": "https://api.github.com/..."},
            },
        }
        mock_task = MagicMock()
        WebhookHandler._fn_reply_to_issue_comment = MagicMock(
            side_effect=RuntimeError("network down")
        )
        WebhookHandler._fn_create_task = mock_task
        WebhookHandler._fn_launch_worker = MagicMock()
        WebhookHandler.gh = MagicMock()
        status = _post_webhook(url, cfg, "issue_comment", payload)
        assert status == 200
        mock_task.assert_not_called()
        store = FidoStore(cfg.repos["owner/repo"].work_dir)
        promises = store.recoverable_promises()
        assert len(promises) == 1
        assert promises[0].anchor_comment_id == 302
        assert promises[0].state == "failed"
        assert store.claim_state(302) == "retryable_failed"

    def test_issue_comment_promise_written_before_attempt(self, server: tuple) -> None:
        """Promise is durable before the issue-comment reply attempt, not just on exception."""
        url, cfg = server
        store = FidoStore(cfg.repos["owner/repo"].work_dir)
        payload = {
            **self._payload(),
            "action": "created",
            "comment": {
                "id": 305,
                "body": "please fix this",
                "user": {"login": "owner"},
                "html_url": "https://github.com/owner/repo/pull/14#issuecomment-305",
            },
            "issue": {
                "number": 14,
                "title": "my pr",
                "body": "",
                "pull_request": {"url": "https://api.github.com/..."},
            },
        }
        promise_existed_during_call = []

        def check_promise_during_call(*args, **kwargs):
            promise_existed_during_call.append(store.claim_state(305))
            return ("ACT", ["fix the thing"])

        WebhookHandler._fn_reply_to_issue_comment = check_promise_during_call
        WebhookHandler._fn_create_task = MagicMock()
        WebhookHandler._fn_launch_worker = MagicMock()
        WebhookHandler.gh = MagicMock()
        status = _post_webhook(url, cfg, "issue_comment", payload)
        assert status == 200
        assert promise_existed_during_call == ["in_progress"], (
            "promise must exist before attempt"
        )
        assert store.claim_state(305) == "completed"
        assert store.recoverable_promises() == []

    def test_issue_comment_success_records_reply_artifact(self, server: tuple) -> None:
        url, cfg = server
        payload = {
            **self._payload(),
            "action": "created",
            "comment": {
                "id": 306,
                "body": "please fix this",
                "user": {"login": "owner"},
                "html_url": "https://github.com/owner/repo/pull/14#issuecomment-306",
            },
            "issue": {
                "number": 14,
                "title": "my pr",
                "body": "",
                "pull_request": {"url": "https://api.github.com/..."},
            },
        }
        mock_gh = MagicMock()
        mock_gh.get_repo_info.return_value = "owner/repo"
        mock_gh.comment_issue.return_value = {"id": 9306}
        WebhookHandler.gh = mock_gh
        WebhookHandler._fn_reply_to_issue_comment = reply_to_issue_comment
        WebhookHandler._fn_create_task = MagicMock()
        WebhookHandler._fn_launch_worker = MagicMock()

        def fake_pp(prompt, model, **kwargs):
            if "Triage" in prompt:
                return "ANSWER: looks good"
            return "One issue reply."

        with (
            patch(
                "fido.events._configured_agent",
                return_value=_client(side_effect=fake_pp),
            ),
            patch("fido.events.maybe_react"),
        ):
            status = _post_webhook(url, cfg, "issue_comment", payload)

        assert status == 200
        store = FidoStore(cfg.repos["owner/repo"].work_dir)
        with store._connect() as conn:
            row = conn.execute(
                """
                SELECT a.artifact_comment_id, a.lane_key, COUNT(ap.promise_id) AS covered
                  FROM reply_artifacts AS a
                  JOIN reply_artifact_promises AS ap
                    ON ap.artifact_comment_id = a.artifact_comment_id
              GROUP BY a.artifact_comment_id, a.lane_key
                """,
            ).fetchone()
        assert row is not None
        assert int(row["artifact_comment_id"]) == 9306
        assert row["lane_key"] == "issues:owner/repo:14"
        assert int(row["covered"]) == 1
        assert store.claim_state(306) == "completed"

    def test_successful_redelivery_clears_stale_issue_promise(
        self, server: tuple
    ) -> None:
        url, cfg = server
        payload = {
            **self._payload(),
            "action": "created",
            "comment": {
                "id": 304,
                "body": "please fix",
                "user": {"login": "owner"},
                "html_url": "https://github.com/owner/repo/pull/13#issuecomment-304",
            },
            "issue": {
                "number": 13,
                "title": "my pr",
                "body": "",
                "pull_request": {"url": "https://api.github.com/..."},
            },
        }
        mock_task = MagicMock()
        WebhookHandler._fn_reply_to_issue_comment = MagicMock(
            side_effect=[
                RuntimeError("network down"),
                ("ACT", ["from issue redelivery"]),
            ]
        )
        WebhookHandler._fn_create_task = mock_task
        WebhookHandler._fn_launch_worker = MagicMock()
        WebhookHandler.gh = MagicMock()
        assert _post_webhook(url, cfg, "issue_comment", payload) == 200
        assert _post_webhook(url, cfg, "issue_comment", payload) == 200
        store = FidoStore(cfg.repos["owner/repo"].work_dir)
        assert store.claim_state(304) == "retryable_failed"
        assert WebhookHandler._fn_reply_to_issue_comment.call_count == 1
        mock_task.assert_not_called()

    def test_failed_issue_comment_webhook_recovers_once_from_live_state(
        self, server: tuple
    ) -> None:
        url, cfg = server
        payload = {
            **self._payload(),
            "action": "created",
            "comment": {"id": 302, "body": "please fix", "user": {"login": "owner"}},
            "issue": {
                "number": 13,
                "title": "my pr",
                "body": "",
                "pull_request": {"url": "https://api.github.com/..."},
            },
        }
        WebhookHandler._fn_reply_to_issue_comment = MagicMock(
            side_effect=RuntimeError("network down")
        )
        WebhookHandler._fn_create_task = MagicMock()
        WebhookHandler._fn_launch_worker = MagicMock()
        WebhookHandler.gh = MagicMock()
        assert _post_webhook(url, cfg, "issue_comment", payload) == 200
        assert _post_webhook(url, cfg, "issue_comment", payload) == 200

        store = FidoStore(cfg.repos["owner/repo"].work_dir)
        assert [p.anchor_comment_id for p in store.recoverable_promises()] == [302]

        recovery_gh = MagicMock()
        recovery_gh.view_issue.return_value = {"title": "my pr", "body": "body"}
        recovery_gh.get_issue_comment.return_value = {
            "id": 302,
            "body": "edited top-level comment",
            "html_url": "https://github.com/owner/repo/pull/13#issuecomment-302",
            "issue_url": "https://api.github.com/repos/owner/repo/issues/13",
            "user": {"login": "owner"},
        }

        def fake_reply(action, *args, **kwargs):
            assert action.comment_body == "edited top-level comment"
            return ("ACT", ["task from issue recovery"])

        with (
            patch(
                "fido.events.reply_to_issue_comment", side_effect=fake_reply
            ) as mock_reply,
            patch("fido.events.create_task") as mock_create_task,
        ):
            assert recover_reply_promises(
                cfg.repos["owner/repo"].work_dir / ".git" / "fido",
                cfg,
                cfg.repos["owner/repo"],
                recovery_gh,
                13,
            )
            assert not recover_reply_promises(
                cfg.repos["owner/repo"].work_dir / ".git" / "fido",
                cfg,
                cfg.repos["owner/repo"],
                recovery_gh,
                13,
            )
        assert mock_reply.call_count == 1
        mock_create_task.assert_called_once_with(
            "task from issue recovery",
            cfg,
            cfg.repos["owner/repo"],
            recovery_gh,
            thread={
                "repo": "owner/repo",
                "pr": 13,
                "comment_id": 302,
                "url": "https://github.com/owner/repo/pull/13#issuecomment-302",
                "author": "owner",
                "comment_type": "issues",
            },
            registry=None,
        )
        assert store.recoverable_promises() == []

    def test_process_action_does_not_overwrite_worker_what(self, server: tuple) -> None:
        """_process_action must not call report_activity — the webhook runs on
        a separate thread and writing the worker's own worker_what field
        from here clobbers the worker thread's state.  Webhook activity is
        tracked via registry.webhook_activity instead.
        """
        url, cfg = server
        payload = {
            **self._payload(),
            "action": "closed",
            "pull_request": {"number": 14, "merged": True},
        }
        WebhookHandler._fn_launch_worker = MagicMock()
        status = _post_webhook(url, cfg, "pull_request", payload)
        assert status == 200
        for call in WebhookHandler.registry.report_activity.call_args_list:
            args = call.args
            assert "handling webhook action" not in args

    def test_status_endpoint_includes_claude_talker(self, server: tuple) -> None:
        """Active SessionTalker appears in /status as a structured object."""
        from datetime import datetime, timezone

        from fido.provider import SessionTalker
        from fido.registry import WorkerActivity

        url, _ = server
        talker = SessionTalker(
            repo_name="owner/repo",
            thread_id=12321,
            kind="worker",
            description="persistent session turn",
            claude_pid=12345,
            started_at=datetime(2026, 4, 14, 16, 0, tzinfo=timezone.utc),
        )
        WebhookHandler.registry.get_all_activities.return_value = [
            WorkerActivity(
                repo_name="owner/repo",
                what="running",
                busy=True,
                last_progress_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ]
        WebhookHandler.registry.get_crash_info.return_value = None
        WebhookHandler.registry.is_stale.return_value = False
        WebhookHandler.registry.thread_started_at.return_value = None
        WebhookHandler.registry.get_webhook_activities.return_value = []
        WebhookHandler.registry.get_session_owner.return_value = None
        WebhookHandler.registry.get_session_alive.return_value = True
        WebhookHandler.registry.get_session_pid.return_value = None
        WebhookHandler.registry.is_rescoping.return_value = False
        with patch("fido.provider.get_talker", return_value=talker):
            resp = urllib.request.urlopen(f"{url}/status.json")
        data = json.loads(resp.read())
        talker_data = data["activities"][0]["claude_talker"]
        assert talker_data["repo_name"] == "owner/repo"
        assert talker_data["thread_id"] == 12321
        assert talker_data["kind"] == "worker"
        assert talker_data["claude_pid"] == 12345

    def test_claude_leak_halts_process(
        self,
        server: tuple,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """SessionLeakError from a webhook handler calls os._exit(3)."""
        from fido import server as server_module

        url, cfg = server
        payload = {
            **self._payload(),
            "action": "closed",
            "pull_request": {"number": 99, "merged": True},
        }
        WebhookHandler.gh = MagicMock()
        WebhookHandler._fn_launch_worker = MagicMock(
            side_effect=provider.SessionLeakError("leaked")
        )
        exits: list[int] = []
        monkeypatch.setattr(server_module.os, "_exit", exits.append)
        _post_webhook(url, cfg, "pull_request", payload)
        assert exits == [3]

    def test_exception_in_process_action_does_not_crash(self, server: tuple) -> None:
        url, cfg = server
        payload = {
            **self._payload(),
            "action": "closed",
            "pull_request": {"number": 13, "merged": True},
        }
        WebhookHandler.gh = MagicMock()
        WebhookHandler._fn_launch_worker = MagicMock(side_effect=Exception("explode"))
        status = _post_webhook(url, cfg, "pull_request", payload)
        assert status == 200
        # server still alive — no crash

    def test_process_action_error_reacts_on_reply_to(self, server: tuple) -> None:
        """On exception with a reply_to comment, adds a confused reaction."""
        url, cfg = server
        payload = {
            **self._payload(),
            "action": "created",
            "comment": {
                "id": 500,
                "body": "looks bad",
                "user": {"login": "owner"},
                "html_url": "https://example.com",
                "path": "x.py",
                "line": 1,
                "diff_hunk": "@@ @@",
            },
            "pull_request": {"number": 20, "title": "T", "body": ""},
        }
        mock_gh = MagicMock()
        WebhookHandler.gh = mock_gh
        WebhookHandler._fn_reply_to_comment = MagicMock(
            side_effect=RuntimeError("boom")
        )
        WebhookHandler._fn_launch_worker = MagicMock()
        _post_webhook(url, cfg, "pull_request_review_comment", payload)
        mock_gh.add_reaction.assert_called_once_with(
            "owner/repo", "pulls", 500, "confused"
        )

    def test_process_action_error_reacts_on_thread(self, server: tuple) -> None:
        """On exception with a thread comment (issue_comment), adds a confused reaction."""
        url, cfg = server
        payload = {
            **self._payload(),
            "action": "created",
            "comment": {
                "id": 501,
                "body": "question?",
                "user": {"login": "owner"},
                "html_url": "https://github.com/owner/repo/pull/21#issuecomment-501",
            },
            "issue": {
                "number": 21,
                "title": "my pr",
                "body": "",
                "pull_request": {"url": "https://api.github.com/..."},
            },
        }
        mock_gh = MagicMock()
        WebhookHandler.gh = mock_gh
        WebhookHandler._fn_reply_to_issue_comment = MagicMock(
            side_effect=RuntimeError("boom")
        )
        WebhookHandler._fn_launch_worker = MagicMock()
        _post_webhook(url, cfg, "issue_comment", payload)
        mock_gh.add_reaction.assert_called_once_with(
            "owner/repo", "issues", 501, "confused"
        )

    def test_process_action_error_no_reaction_without_comment(
        self, server: tuple
    ) -> None:
        """On exception with no comment context (e.g., merged PR), no reaction."""
        url, cfg = server
        payload = {
            **self._payload(),
            "action": "closed",
            "pull_request": {"number": 22, "merged": True},
        }
        mock_gh = MagicMock()
        WebhookHandler.gh = mock_gh
        WebhookHandler._fn_launch_worker = MagicMock(side_effect=RuntimeError("boom"))
        _post_webhook(url, cfg, "pull_request", payload)
        mock_gh.add_reaction.assert_not_called()

    def test_process_action_error_no_reaction_when_comment_id_missing(
        self, server: tuple
    ) -> None:
        """No reaction when thread has no comment_id (e.g., review submission)."""
        url, cfg = server
        # review submission: reply_to=None, thread=None, review_comments set
        payload = {
            **self._payload(),
            "action": "submitted",
            "review": {
                "id": 888,
                "state": "changes_requested",
                "user": {"login": "owner"},
            },
            "pull_request": {"number": 24},
        }
        mock_gh = MagicMock()
        WebhookHandler.gh = mock_gh
        WebhookHandler._fn_reply_to_review = MagicMock(side_effect=RuntimeError("boom"))
        WebhookHandler._fn_launch_worker = MagicMock()
        _post_webhook(url, cfg, "pull_request_review", payload)
        mock_gh.add_reaction.assert_not_called()

    def test_process_action_error_no_reaction_when_thread_lacks_comment_id(
        self, server: tuple
    ) -> None:
        """No reaction when thread dict exists but has no comment_id key."""
        from fido.events import Action

        url, cfg = server
        handler = WebhookHandler.__new__(WebhookHandler)
        mock_gh = MagicMock()
        handler.gh = mock_gh
        action = Action(
            prompt="test",
            thread={"repo": "owner/repo", "pr": 1},
        )
        handler._signal_action_error(action)
        mock_gh.add_reaction.assert_not_called()

    def test_process_action_error_reaction_failure_doesnt_crash(
        self, server: tuple
    ) -> None:
        """add_reaction failure is caught — server stays alive."""
        url, cfg = server
        payload = {
            **self._payload(),
            "action": "created",
            "comment": {
                "id": 502,
                "body": "yo",
                "user": {"login": "owner"},
                "html_url": "https://example.com",
                "path": "x.py",
                "line": 1,
                "diff_hunk": "@@ @@",
            },
            "pull_request": {"number": 23, "title": "T", "body": ""},
        }
        mock_gh = MagicMock()
        mock_gh.add_reaction.side_effect = RuntimeError("reaction failed")
        WebhookHandler.gh = mock_gh
        WebhookHandler._fn_reply_to_comment = MagicMock(
            side_effect=RuntimeError("process boom")
        )
        WebhookHandler._fn_launch_worker = MagicMock()
        status = _post_webhook(url, cfg, "pull_request_review_comment", payload)
        assert status == 200
        mock_gh.add_reaction.assert_called_once()

    def test_issue_comment_webhook_activity_tracks_phase(self, tmp_path: Path) -> None:
        from fido.events import Action
        from fido.registry import WorkerRegistry

        cfg = _config(tmp_path)
        handler = WebhookHandler.__new__(WebhookHandler)
        handler.config = cfg
        handler.registry = WorkerRegistry(MagicMock())
        handler.gh = MagicMock()
        phases: list[str] = []

        def reply_to_issue_comment(*args: object) -> tuple[str, list[str]]:
            del args
            phases.append(
                handler.registry.get_webhook_activities("owner/repo")[0].description
            )
            return "ACT", ["do it"]

        def create_task(*args: object, **kwargs: object) -> None:
            del args, kwargs
            phases.append(
                handler.registry.get_webhook_activities("owner/repo")[0].description
            )

        WebhookHandler._fn_reply_to_issue_comment = reply_to_issue_comment
        WebhookHandler._fn_create_task = create_task
        WebhookHandler._fn_launch_worker = MagicMock()
        action = Action(
            prompt="test",
            comment_body="please fix",
            thread={
                "repo": "owner/repo",
                "pr": 11,
                "comment_id": 9300,
                "url": "https://github.com/owner/repo/pull/11#issuecomment-9300",
                "author": "owner",
                "comment_type": "issues",
            },
        )

        handler._process_action(action, cfg.repos["owner/repo"])

        assert phases == ["triaging PR comment", "queuing PR comment task"]
        assert handler.registry.get_webhook_activities("owner/repo") == []

    def test_dispatch_error_returns_500(self, server: tuple) -> None:
        """When dispatch raises, return 500 so GitHub retries the delivery."""
        url, cfg = server
        payload = {**self._payload(), "action": "created"}
        WebhookHandler._fn_dispatch = MagicMock(side_effect=RuntimeError("boom"))
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _post_webhook(url, cfg, "pull_request_review_comment", payload)
        assert exc_info.value.code == 500

    def test_dispatch_called_before_ack(self, server: tuple) -> None:
        """dispatch() must be called before the HTTP response is written."""
        url, cfg = server
        payload = {
            **self._payload(),
            "action": "created",
            "comment": {
                "id": 999,
                "body": "hey",
                "user": {"login": "owner"},
                "html_url": "https://example.com",
                "path": "x.py",
                "line": 1,
                "diff_hunk": "@@ @@",
            },
            "pull_request": {"number": 99, "title": "T", "body": ""},
        }
        call_order: list[str] = []

        def fake_dispatch(*_args, **_kwargs):
            call_order.append("dispatch")
            return None

        original_respond = WebhookHandler._respond

        def fake_respond(self, code, msg):
            call_order.append(f"respond_{code}")
            original_respond(self, code, msg)

        WebhookHandler._fn_dispatch = fake_dispatch
        WebhookHandler._respond = fake_respond  # type: ignore[method-assign]
        try:
            _post_webhook(url, cfg, "pull_request_review_comment", payload)
        finally:
            WebhookHandler._respond = original_respond  # type: ignore[method-assign]
        assert call_order == ["dispatch", "respond_200"]

    def test_review_comment_calls_unblock_tasks(self, server: tuple) -> None:
        """A pull_request_review_comment triggers unblock_tasks so BLOCKED tasks resume."""
        url, cfg = server
        payload = {
            **self._payload(),
            "action": "created",
            "comment": {
                "id": 700,
                "body": "here is the answer",
                "user": {"login": "owner"},
                "html_url": "https://example.com",
                "path": "foo.py",
                "line": 1,
                "diff_hunk": "@@ @@",
            },
            "pull_request": {"number": 71, "title": "My PR", "body": ""},
        }
        mock_unblock = MagicMock(return_value=0)
        WebhookHandler._fn_reply_to_comment = MagicMock(return_value=("ANSWER", []))
        WebhookHandler._fn_launch_worker = MagicMock()
        WebhookHandler._fn_unblock_tasks = mock_unblock
        status = _post_webhook(url, cfg, "pull_request_review_comment", payload)
        assert status == 200
        mock_unblock.assert_called_once_with(cfg.repos["owner/repo"].work_dir)

    def test_issue_comment_calls_unblock_tasks(self, server: tuple) -> None:
        """A top-level PR comment triggers unblock_tasks so BLOCKED tasks resume."""
        url, cfg = server
        payload = {
            **self._payload(),
            "action": "created",
            "comment": {
                "id": 701,
                "body": "here is what you need",
                "user": {"login": "owner"},
                "html_url": "https://github.com/owner/repo/pull/71#issuecomment-701",
            },
            "issue": {
                "number": 71,
                "title": "my pr",
                "body": "",
                "pull_request": {"url": "https://api.github.com/..."},
            },
        }
        mock_unblock = MagicMock(return_value=0)
        WebhookHandler.gh = MagicMock()
        WebhookHandler._fn_reply_to_issue_comment = MagicMock(
            return_value=("ANSWER", [])
        )
        WebhookHandler._fn_launch_worker = MagicMock()
        WebhookHandler._fn_unblock_tasks = mock_unblock
        status = _post_webhook(url, cfg, "issue_comment", payload)
        assert status == 200
        mock_unblock.assert_called_once_with(cfg.repos["owner/repo"].work_dir)

    def test_non_comment_event_does_not_call_unblock_tasks(self, server: tuple) -> None:
        """A PR merge event (no comment body) must NOT trigger unblock_tasks."""
        url, cfg = server
        payload = {
            **self._payload(),
            "action": "closed",
            "pull_request": {"number": 72, "merged": True},
        }
        mock_unblock = MagicMock(return_value=0)
        WebhookHandler._fn_launch_worker = MagicMock()
        WebhookHandler._fn_unblock_tasks = mock_unblock
        status = _post_webhook(url, cfg, "pull_request", payload)
        assert status == 200
        mock_unblock.assert_not_called()


class TestPopulateMemberships:
    def test_populates_from_get_collaborators(self, tmp_path: Path) -> None:
        from fido.config import RepoMembership
        from fido.server import populate_memberships

        cfg = Config(
            port=0,
            secret=b"s",
            repos={
                "owner/repo": RepoConfig(name="owner/repo", work_dir=tmp_path),
            },
            allowed_bots=frozenset(),
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )
        mock_gh = MagicMock()
        mock_gh.get_user.return_value = "fido-bot"
        mock_gh.get_collaborators.return_value = ["alice", "bob"]
        populate_memberships(cfg, mock_gh)
        assert cfg.repos["owner/repo"].membership == RepoMembership(
            collaborators=frozenset({"alice", "bob"})
        )

    def test_filters_bot_user_from_collaborators(self, tmp_path: Path) -> None:
        from fido.server import populate_memberships

        cfg = Config(
            port=0,
            secret=b"s",
            repos={
                "owner/repo": RepoConfig(name="owner/repo", work_dir=tmp_path),
            },
            allowed_bots=frozenset(),
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )
        mock_gh = MagicMock()
        mock_gh.get_user.return_value = "fido-bot"
        mock_gh.get_collaborators.return_value = ["alice", "fido-bot", "bob"]
        populate_memberships(cfg, mock_gh)
        result = cfg.repos["owner/repo"].membership.collaborators
        assert result == frozenset({"alice", "bob"})
        assert "fido-bot" not in result

    def test_iterates_all_repos(self, tmp_path: Path) -> None:
        from fido.server import populate_memberships

        cfg = Config(
            port=0,
            secret=b"s",
            repos={
                "o/r1": RepoConfig(name="o/r1", work_dir=tmp_path),
                "o/r2": RepoConfig(name="o/r2", work_dir=tmp_path),
            },
            allowed_bots=frozenset(),
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )
        mock_gh = MagicMock()
        mock_gh.get_user.return_value = "bot"
        mock_gh.get_collaborators.side_effect = lambda name: {
            "o/r1": ["alice"],
            "o/r2": ["bob", "carol"],
        }[name]
        populate_memberships(cfg, mock_gh)
        assert cfg.repos["o/r1"].membership.collaborators == frozenset({"alice"})
        assert cfg.repos["o/r2"].membership.collaborators == frozenset({"bob", "carol"})


class TestBootstrapIssueCaches:
    """Tests for bootstrap_issue_caches() (#837)."""

    def _make_repos(self, tmp_path: Path) -> dict[str, RepoConfig]:
        return {"owner/repo": RepoConfig(name="owner/repo", work_dir=tmp_path)}

    def test_calls_find_all_open_issues_with_owner_and_repo_name(
        self, tmp_path: Path
    ) -> None:
        from fido.server import bootstrap_issue_caches

        repos = self._make_repos(tmp_path)
        mock_gh = MagicMock()
        mock_gh.find_all_open_issues.return_value = []
        mock_registry = MagicMock()
        mock_registry.get_issue_cache.return_value = MagicMock()

        bootstrap_issue_caches(repos, mock_gh, mock_registry)

        mock_gh.find_all_open_issues.assert_called_once_with("owner", "repo")

    def test_calls_load_inventory_on_cache_with_returned_issues(
        self, tmp_path: Path
    ) -> None:
        from fido.server import bootstrap_issue_caches

        repos = self._make_repos(tmp_path)
        mock_gh = MagicMock()
        fake_inventory = [{"number": 1}]
        mock_gh.find_all_open_issues.return_value = fake_inventory
        mock_cache = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get_issue_cache.return_value = mock_cache

        bootstrap_issue_caches(repos, mock_gh, mock_registry)

        mock_registry.get_issue_cache.assert_called_once_with("owner/repo")
        mock_cache.load_inventory.assert_called_once_with(
            fake_inventory, snapshot_started_at=ANY
        )

    def test_bootstraps_all_repos(self, tmp_path: Path) -> None:
        from fido.server import bootstrap_issue_caches

        repos = {
            "a/r1": RepoConfig(name="a/r1", work_dir=tmp_path),
            "b/r2": RepoConfig(name="b/r2", work_dir=tmp_path),
        }
        mock_gh = MagicMock()
        mock_gh.find_all_open_issues.return_value = []
        mock_registry = MagicMock()

        bootstrap_issue_caches(repos, mock_gh, mock_registry)

        assert mock_gh.find_all_open_issues.call_count == 2
        mock_gh.find_all_open_issues.assert_any_call("a", "r1")
        mock_gh.find_all_open_issues.assert_any_call("b", "r2")

    def test_per_repo_failure_is_swallowed(self, tmp_path: Path) -> None:
        """A single GitHub API error must not prevent fido from starting."""
        from fido.server import bootstrap_issue_caches

        repos = {
            "a/r1": RepoConfig(name="a/r1", work_dir=tmp_path),
            "b/r2": RepoConfig(name="b/r2", work_dir=tmp_path),
        }
        mock_gh = MagicMock()
        mock_gh.find_all_open_issues.side_effect = [RuntimeError("API down"), []]
        mock_cache_r2 = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get_issue_cache.side_effect = lambda name: (
            MagicMock() if name == "a/r1" else mock_cache_r2
        )

        # Must not raise despite the first repo failing.
        bootstrap_issue_caches(repos, mock_gh, mock_registry)

        # Second repo should still be bootstrapped.
        mock_cache_r2.load_inventory.assert_called_once()


class TestRun:
    """Tests for the run() entry point."""

    def _fake_cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=0,
            secret=b"test",
            repos={"owner/repo": RepoConfig(name="owner/repo", work_dir=tmp_path)},
            allowed_bots=frozenset(),
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )

    def test_run_starts_server(self, tmp_path: Path) -> None:
        from fido.server import run

        fake_cfg = self._fake_cfg(tmp_path)
        mock_server = MagicMock()
        mock_server.serve_forever.side_effect = KeyboardInterrupt

        run(
            _from_args=lambda: fake_cfg,
            _HTTPServer=lambda *a, **kw: mock_server,
            _make_registry=MagicMock(),
            _path_home=lambda: tmp_path,
            _basic_config=MagicMock(),
            _populate_memberships=MagicMock(),
            _startup_pull=MagicMock(),
            _preflight_repo_identity=MagicMock(),
            _preflight_tools=MagicMock(),
            _preflight_sub_dir=MagicMock(),
            _preflight_gh_auth=MagicMock(),
            _GitHub=MagicMock,
        )

        mock_server.serve_forever.assert_called_once()
        mock_server.server_close.assert_called_once()

    def test_run_keyboard_interrupt_kills_children(self, tmp_path: Path) -> None:
        from fido.server import run

        fake_cfg = self._fake_cfg(tmp_path)
        mock_server = MagicMock()
        mock_server.serve_forever.side_effect = KeyboardInterrupt
        mock_kill = MagicMock()

        run(
            _from_args=lambda: fake_cfg,
            _HTTPServer=lambda *a, **kw: mock_server,
            _make_registry=MagicMock(),
            _path_home=lambda: tmp_path,
            _basic_config=MagicMock(),
            _populate_memberships=MagicMock(),
            _startup_pull=MagicMock(),
            _preflight_repo_identity=MagicMock(),
            _preflight_tools=MagicMock(),
            _preflight_sub_dir=MagicMock(),
            _preflight_gh_auth=MagicMock(),
            _GitHub=MagicMock,
            _signal=MagicMock(),
            _kill_active_children=mock_kill,
        )
        mock_kill.assert_called_once()

    def test_run_installs_sigterm_and_sigint_handlers(self, tmp_path: Path) -> None:
        import signal as _sig

        from fido.server import run

        fake_cfg = self._fake_cfg(tmp_path)
        mock_server = MagicMock()
        mock_server.serve_forever.side_effect = KeyboardInterrupt
        installed: dict[int, object] = {}

        def fake_signal(signum, handler):
            installed[signum] = handler

        run(
            _from_args=lambda: fake_cfg,
            _HTTPServer=lambda *a, **kw: mock_server,
            _make_registry=MagicMock(),
            _path_home=lambda: tmp_path,
            _basic_config=MagicMock(),
            _populate_memberships=MagicMock(),
            _startup_pull=MagicMock(),
            _preflight_repo_identity=MagicMock(),
            _preflight_tools=MagicMock(),
            _preflight_sub_dir=MagicMock(),
            _preflight_gh_auth=MagicMock(),
            _GitHub=MagicMock,
            _signal=fake_signal,
            _kill_active_children=MagicMock(),
        )
        assert _sig.SIGTERM in installed
        assert _sig.SIGINT in installed

    def test_shutdown_handler_kills_children_and_exits(self, tmp_path: Path) -> None:
        from fido.server import run

        fake_cfg = self._fake_cfg(tmp_path)
        mock_server = MagicMock()
        mock_server.serve_forever.side_effect = KeyboardInterrupt
        mock_kill = MagicMock()
        captured: dict[int, object] = {}

        def fake_signal(signum, handler):
            captured[signum] = handler

        run(
            _from_args=lambda: fake_cfg,
            _HTTPServer=lambda *a, **kw: mock_server,
            _make_registry=MagicMock(),
            _path_home=lambda: tmp_path,
            _basic_config=MagicMock(),
            _populate_memberships=MagicMock(),
            _startup_pull=MagicMock(),
            _preflight_repo_identity=MagicMock(),
            _preflight_tools=MagicMock(),
            _preflight_sub_dir=MagicMock(),
            _preflight_gh_auth=MagicMock(),
            _GitHub=MagicMock,
            _signal=fake_signal,
            _kill_active_children=mock_kill,
        )
        # Reset call counts from the KeyboardInterrupt path so we can verify
        # the signal handler invokes the same teardown.
        mock_kill.reset_mock()
        mock_server.server_close.reset_mock()

        import signal as _sig

        handler = captured[_sig.SIGTERM]
        with pytest.raises(SystemExit):
            handler(_sig.SIGTERM, None)
        mock_kill.assert_called_once()
        mock_server.server_close.assert_called_once()

    def test_run_format_includes_repo_name(self, tmp_path: Path) -> None:
        from fido.server import run

        fake_cfg = self._fake_cfg(tmp_path)
        mock_server = MagicMock()
        mock_server.serve_forever.side_effect = KeyboardInterrupt

        captured_kwargs: list[dict] = []

        def fake_basic_config(**kwargs):
            captured_kwargs.append(kwargs)

        run(
            _from_args=lambda: fake_cfg,
            _HTTPServer=lambda *a, **kw: mock_server,
            _make_registry=MagicMock(),
            _path_home=lambda: tmp_path,
            _basic_config=fake_basic_config,
            _populate_memberships=MagicMock(),
            _startup_pull=MagicMock(),
            _preflight_repo_identity=MagicMock(),
            _preflight_tools=MagicMock(),
            _preflight_sub_dir=MagicMock(),
            _preflight_gh_auth=MagicMock(),
            _GitHub=MagicMock,
        )

        assert len(captured_kwargs) == 1
        assert "%(repo_name)s" in captured_kwargs[0]["format"]

    def test_run_adds_repo_context_filter_to_handlers(self, tmp_path: Path) -> None:
        from fido.server import run
        from fido.worker import RepoContextFilter

        fake_cfg = self._fake_cfg(tmp_path)
        mock_server = MagicMock()
        mock_server.serve_forever.side_effect = KeyboardInterrupt

        captured_handlers: list = []

        def fake_basic_config(**kwargs):
            captured_handlers.extend(kwargs.get("handlers", []))

        run(
            _from_args=lambda: fake_cfg,
            _HTTPServer=lambda *a, **kw: mock_server,
            _make_registry=MagicMock(),
            _path_home=lambda: tmp_path,
            _basic_config=fake_basic_config,
            _populate_memberships=MagicMock(),
            _startup_pull=MagicMock(),
            _preflight_repo_identity=MagicMock(),
            _preflight_tools=MagicMock(),
            _preflight_sub_dir=MagicMock(),
            _preflight_gh_auth=MagicMock(),
            _GitHub=MagicMock,
        )

        assert len(captured_handlers) >= 1
        for handler in captured_handlers:
            assert any(isinstance(f, RepoContextFilter) for f in handler.filters)

    def test_run_stderr_tty_adds_stream_handler(self, tmp_path: Path) -> None:
        from fido.server import run

        fake_cfg = self._fake_cfg(tmp_path)
        mock_server = MagicMock()
        mock_server.serve_forever.side_effect = KeyboardInterrupt
        mock_stderr = MagicMock()
        mock_stderr.isatty.return_value = True

        run(
            _from_args=lambda: fake_cfg,
            _HTTPServer=lambda *a, **kw: mock_server,
            _make_registry=MagicMock(),
            _path_home=lambda: tmp_path,
            _basic_config=MagicMock(),
            _stderr=mock_stderr,
            _populate_memberships=MagicMock(),
            _startup_pull=MagicMock(),
            _preflight_repo_identity=MagicMock(),
            _preflight_tools=MagicMock(),
            _preflight_sub_dir=MagicMock(),
            _preflight_gh_auth=MagicMock(),
            _GitHub=MagicMock,
        )

        mock_server.serve_forever.assert_called_once()

    def test_run_logs_to_stderr_stream_only(self, tmp_path: Path) -> None:
        from fido.server import run
        from fido.worker import RepoContextFilter

        fake_cfg = Config(
            port=0,
            secret=b"test",
            repos={
                "owner/myrepo": RepoConfig(name="owner/myrepo", work_dir=tmp_path),
                "owner/other": RepoConfig(name="owner/other", work_dir=tmp_path),
            },
            allowed_bots=frozenset(),
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )

        mock_server = MagicMock()
        mock_server.serve_forever.side_effect = KeyboardInterrupt

        captured_handlers: list = []
        fake_stderr = MagicMock()

        def fake_basic_config(**kwargs):
            captured_handlers.extend(kwargs.get("handlers", []))

        run(
            _from_args=lambda: fake_cfg,
            _HTTPServer=lambda *a, **kw: mock_server,
            _make_registry=MagicMock(),
            _path_home=lambda: tmp_path,
            _basic_config=fake_basic_config,
            _populate_memberships=MagicMock(),
            _startup_pull=MagicMock(),
            _preflight_repo_identity=MagicMock(),
            _preflight_tools=MagicMock(),
            _preflight_sub_dir=MagicMock(),
            _preflight_gh_auth=MagicMock(),
            _GitHub=MagicMock,
            _stderr=fake_stderr,
        )

        import logging

        assert len(captured_handlers) == 1
        assert isinstance(captured_handlers[0], logging.StreamHandler)
        assert captured_handlers[0].stream is fake_stderr
        assert any(
            isinstance(f, RepoContextFilter) for f in captured_handlers[0].filters
        )

    def test_run_does_not_create_file_log_handlers(self, tmp_path: Path) -> None:
        from fido.server import run

        fake_cfg = Config(
            port=0,
            secret=b"test",
            repos={"owner/myrepo": RepoConfig(name="owner/myrepo", work_dir=tmp_path)},
            allowed_bots=frozenset(),
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )

        mock_server = MagicMock()
        mock_server.serve_forever.side_effect = KeyboardInterrupt

        captured_handlers: list = []

        def fake_basic_config(**kwargs):
            captured_handlers.extend(kwargs.get("handlers", []))

        run(
            _from_args=lambda: fake_cfg,
            _HTTPServer=lambda *a, **kw: mock_server,
            _make_registry=MagicMock(),
            _path_home=lambda: tmp_path,
            _basic_config=fake_basic_config,
            _populate_memberships=MagicMock(),
            _startup_pull=MagicMock(),
            _preflight_repo_identity=MagicMock(),
            _preflight_tools=MagicMock(),
            _preflight_sub_dir=MagicMock(),
            _preflight_gh_auth=MagicMock(),
            _GitHub=MagicMock,
        )

        import logging

        assert not any(isinstance(h, logging.FileHandler) for h in captured_handlers)

    def test_run_starts_watchdog_with_registry_and_repos(self, tmp_path: Path) -> None:
        from fido.server import run

        fake_cfg = self._fake_cfg(tmp_path)
        mock_server = MagicMock()
        mock_server.serve_forever.side_effect = KeyboardInterrupt
        mock_registry = MagicMock()
        mock_make_registry = MagicMock(return_value=mock_registry)
        mock_watchdog_cls = MagicMock()

        run(
            _from_args=lambda: fake_cfg,
            _HTTPServer=lambda *a, **kw: mock_server,
            _make_registry=mock_make_registry,
            _path_home=lambda: tmp_path,
            _basic_config=MagicMock(),
            _populate_memberships=MagicMock(),
            _startup_pull=MagicMock(),
            _preflight_repo_identity=MagicMock(),
            _preflight_tools=MagicMock(),
            _preflight_sub_dir=MagicMock(),
            _preflight_gh_auth=MagicMock(),
            _GitHub=MagicMock,
            _Watchdog=mock_watchdog_cls,
            _ReconcileWatchdog=MagicMock(),
        )

        mock_watchdog_cls.assert_called_once_with(mock_registry, fake_cfg.repos)
        mock_watchdog_cls.return_value.start_thread.assert_called_once()

    def test_run_starts_rate_limit_monitor_with_gh(self, tmp_path: Path) -> None:
        from fido.server import WebhookHandler, run

        fake_cfg = self._fake_cfg(tmp_path)
        mock_server = MagicMock()
        mock_server.serve_forever.side_effect = KeyboardInterrupt
        mock_make_registry = MagicMock(return_value=MagicMock())
        mock_rl_cls = MagicMock()
        mock_gh_instance = MagicMock()

        run(
            _from_args=lambda: fake_cfg,
            _HTTPServer=lambda *a, **kw: mock_server,
            _make_registry=mock_make_registry,
            _path_home=lambda: tmp_path,
            _basic_config=MagicMock(),
            _populate_memberships=MagicMock(),
            _startup_pull=MagicMock(),
            _preflight_repo_identity=MagicMock(),
            _preflight_tools=MagicMock(),
            _preflight_sub_dir=MagicMock(),
            _preflight_gh_auth=MagicMock(),
            _GitHub=lambda: mock_gh_instance,
            _Watchdog=MagicMock(),
            _ReconcileWatchdog=MagicMock(),
            _RateLimitMonitor=mock_rl_cls,
        )

        mock_rl_cls.assert_called_once_with(mock_gh_instance)
        mock_rl_cls.return_value.start_thread.assert_called_once()
        assert WebhookHandler.rate_limit_monitor is mock_rl_cls.return_value

    def test_run_starts_reconcile_watchdog_with_registry_repos_and_gh(
        self, tmp_path: Path
    ) -> None:
        from fido.server import run

        fake_cfg = self._fake_cfg(tmp_path)
        mock_server = MagicMock()
        mock_server.serve_forever.side_effect = KeyboardInterrupt
        mock_registry = MagicMock()
        mock_make_registry = MagicMock(return_value=mock_registry)
        mock_reconcile_cls = MagicMock()
        mock_gh_instance = MagicMock()

        run(
            _from_args=lambda: fake_cfg,
            _HTTPServer=lambda *a, **kw: mock_server,
            _make_registry=mock_make_registry,
            _path_home=lambda: tmp_path,
            _basic_config=MagicMock(),
            _populate_memberships=MagicMock(),
            _startup_pull=MagicMock(),
            _preflight_repo_identity=MagicMock(),
            _preflight_tools=MagicMock(),
            _preflight_sub_dir=MagicMock(),
            _preflight_gh_auth=MagicMock(),
            _GitHub=lambda: mock_gh_instance,
            _Watchdog=MagicMock(),
            _ReconcileWatchdog=mock_reconcile_cls,
        )

        mock_reconcile_cls.assert_called_once_with(
            mock_registry, fake_cfg.repos, mock_gh_instance
        )
        mock_reconcile_cls.return_value.start_thread.assert_called_once()

    def test_run_installs_excepthooks(self, tmp_path: Path) -> None:
        """Uncaught exceptions (main thread and worker threads) should route
        through the logger so tracebacks land in fido.log, not just stderr."""
        import sys as _sys
        import threading as _threading

        from fido.server import run

        fake_cfg = self._fake_cfg(tmp_path)
        mock_server = MagicMock()
        mock_server.serve_forever.side_effect = KeyboardInterrupt

        saved_sys = _sys.excepthook
        saved_thr = _threading.excepthook
        try:
            run(
                _from_args=lambda: fake_cfg,
                _HTTPServer=lambda *a, **kw: mock_server,
                _make_registry=MagicMock(),
                _path_home=lambda: tmp_path,
                _basic_config=MagicMock(),
                _populate_memberships=MagicMock(),
                _startup_pull=MagicMock(),
                _preflight_repo_identity=MagicMock(),
                _preflight_tools=MagicMock(),
                _preflight_sub_dir=MagicMock(),
                _preflight_gh_auth=MagicMock(),
                _GitHub=MagicMock,
                _Watchdog=MagicMock(),
            )
            assert _sys.excepthook is not saved_sys
            assert _threading.excepthook is not saved_thr

            # Call the hooks to confirm they don't raise and they go through logging.
            import fido.server as srv_mod

            with patch.object(srv_mod, "log") as mock_log:
                try:
                    raise ValueError("boom")
                except ValueError:
                    _sys.excepthook(*_sys.exc_info())
                mock_log.critical.assert_called_once()

            with patch.object(srv_mod, "log") as mock_log:
                fake_args = MagicMock()
                fake_args.thread = MagicMock(name="tname")
                fake_args.exc_type = ValueError
                fake_args.exc_value = ValueError("boom")
                fake_args.exc_traceback = None
                _threading.excepthook(fake_args)
                mock_log.critical.assert_called_once()

            # Also cover the branch where thread is None.
            with patch.object(srv_mod, "log") as mock_log:
                fake_args = MagicMock()
                fake_args.thread = None
                fake_args.exc_type = ValueError
                fake_args.exc_value = ValueError("boom")
                fake_args.exc_traceback = None
                _threading.excepthook(fake_args)
                mock_log.critical.assert_called_once()
        finally:
            _sys.excepthook = saved_sys
            _threading.excepthook = saved_thr

    def test_run_calls_bootstrap_issue_caches_with_repos_gh_and_registry(
        self, tmp_path: Path
    ) -> None:
        from fido.server import run

        fake_cfg = self._fake_cfg(tmp_path)
        mock_server = MagicMock()
        mock_server.serve_forever.side_effect = KeyboardInterrupt
        mock_registry = MagicMock()
        mock_make_registry = MagicMock(return_value=mock_registry)
        mock_bootstrap = MagicMock()
        mock_gh_instance = MagicMock()

        run(
            _from_args=lambda: fake_cfg,
            _HTTPServer=lambda *a, **kw: mock_server,
            _make_registry=mock_make_registry,
            _path_home=lambda: tmp_path,
            _basic_config=MagicMock(),
            _populate_memberships=MagicMock(),
            _startup_pull=MagicMock(),
            _preflight_repo_identity=MagicMock(),
            _preflight_tools=MagicMock(),
            _preflight_sub_dir=MagicMock(),
            _preflight_gh_auth=MagicMock(),
            _GitHub=lambda: mock_gh_instance,
            _Watchdog=MagicMock(),
            _ReconcileWatchdog=MagicMock(),
            _bootstrap_issue_caches=mock_bootstrap,
        )

        mock_bootstrap.assert_called_once_with(
            fake_cfg.repos, mock_gh_instance, mock_registry
        )


def _self_restart_cfg(tmp_path: Path) -> Config:
    return Config(
        port=0,
        secret=b"test-secret",
        repos={"owner/fido": RepoConfig(name="owner/fido", work_dir=tmp_path)},
        allowed_bots=frozenset(),
        log_level="WARNING",
        sub_dir=tmp_path / "sub",
    )


_MERGE_PAYLOAD = {
    "repository": {
        "full_name": "owner/fido",
        "owner": {"login": "owner"},
        "default_branch": "main",
    },
    "action": "closed",
    "pull_request": {"number": 1, "merged": True},
}

_PUSH_PAYLOAD = {
    "repository": {
        "full_name": "owner/fido",
        "owner": {"login": "owner"},
        "default_branch": "main",
    },
    "ref": "refs/heads/main",
}


class TestNoop:
    def test_noop_after_post_is_callable_and_silent(self) -> None:
        from fido.server import _noop_after_post

        _noop_after_post()  # default hook — must not raise


class TestParseRepoFromUrl:
    def test_parses_ssh_url(self) -> None:
        from fido.server import _parse_repo_from_url

        assert _parse_repo_from_url("git@github.com:owner/repo.git") == "owner/repo"

    def test_parses_https_url(self) -> None:
        from fido.server import _parse_repo_from_url

        assert _parse_repo_from_url("https://github.com/owner/repo.git") == "owner/repo"

    def test_parses_url_without_git_suffix(self) -> None:
        from fido.server import _parse_repo_from_url

        assert _parse_repo_from_url("https://github.com/owner/repo") == "owner/repo"

    def test_returns_none_for_garbage(self) -> None:
        from fido.server import _parse_repo_from_url

        assert _parse_repo_from_url("garbage") is None


class _FakeProcessRunner:
    """Minimal ProcessRunner fake.

    Results are consumed in order.  Each item is either a return value or an
    exception instance — exceptions are raised, everything else is returned.
    The ``error`` parameter overrides the results list and raises the same
    exception on every call (backward-compat shorthand for single-error tests).
    """

    def __init__(
        self, results: list[Any] | None = None, error: Exception | None = None
    ) -> None:
        self._results = list(results or [])
        self._error = error
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def run(self, cmd: Any, **kwargs: Any) -> Any:
        self.calls.append((cmd, kwargs))
        if self._error is not None:
            raise self._error
        result = self._results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class _FakeClock:
    """Minimal Clock fake: returns pre-configured monotonic values, records sleeps."""

    def __init__(self, times: list[float] | None = None) -> None:
        self._times = iter(times or [])
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return next(self._times)

    def sleep(self, secs: float) -> None:
        self.slept.append(secs)


class _FakeOsProcess:
    """Minimal OsProcess fake: captures execvp, exit, and chdir calls."""

    def __init__(self) -> None:
        self.execvp_calls: list[tuple[str, list[str]]] = []
        self.exit_calls: list[int] = []
        self.chdir_calls: list[Any] = []

    def execvp(self, file: str, args: list[str]) -> None:
        self.execvp_calls.append((file, args))

    def exit(self, code: int) -> None:
        self.exit_calls.append(code)

    def chdir(self, path: Any) -> None:
        self.chdir_calls.append(path)

    def install_signal(self, signum: Any, handler: Any) -> Any:
        return None


class TestPreflightRepoIdentity:
    def test_succeeds_when_remote_matches(self, tmp_path: Path) -> None:
        from fido.server import preflight_repo_identity

        repos = {"owner/repo": RepoConfig(name="owner/repo", work_dir=tmp_path)}
        fake = _FakeProcessRunner([MagicMock(stdout="git@github.com:owner/repo.git\n")])
        preflight_repo_identity(repos, fake)  # no exception

    def test_raises_on_remote_mismatch(self, tmp_path: Path) -> None:
        from fido.server import preflight_repo_identity

        repos = {"owner/repo": RepoConfig(name="owner/repo", work_dir=tmp_path)}
        fake = _FakeProcessRunner(
            [MagicMock(stdout="git@github.com:other/thing.git\n")]
        )
        with pytest.raises(PreflightError, match="other/thing"):
            preflight_repo_identity(repos, fake)

    def test_raises_on_subprocess_error(self, tmp_path: Path) -> None:
        from fido.server import preflight_repo_identity

        repos = {"owner/repo": RepoConfig(name="owner/repo", work_dir=tmp_path)}
        fake = _FakeProcessRunner(error=subprocess.CalledProcessError(128, []))
        with pytest.raises(PreflightError):
            preflight_repo_identity(repos, fake)

    def test_raises_when_git_not_found(self, tmp_path: Path) -> None:
        from fido.server import preflight_repo_identity

        repos = {"owner/repo": RepoConfig(name="owner/repo", work_dir=tmp_path)}
        fake = _FakeProcessRunner(error=FileNotFoundError())
        with pytest.raises(PreflightError):
            preflight_repo_identity(repos, fake)

    def test_raises_on_unparseable_url(self, tmp_path: Path) -> None:
        from fido.server import preflight_repo_identity

        repos = {"owner/repo": RepoConfig(name="owner/repo", work_dir=tmp_path)}
        fake = _FakeProcessRunner([MagicMock(stdout="garbage\n")])
        with pytest.raises(PreflightError):
            preflight_repo_identity(repos, fake)

    def test_checks_all_repos(self, tmp_path: Path) -> None:
        from fido.server import preflight_repo_identity

        repos = {
            "owner/repo1": RepoConfig(name="owner/repo1", work_dir=tmp_path),
            "owner/repo2": RepoConfig(name="owner/repo2", work_dir=tmp_path),
        }
        fake = _FakeProcessRunner(
            [
                MagicMock(stdout="git@github.com:owner/repo1.git\n"),
                MagicMock(stdout="git@github.com:owner/repo2.git\n"),
            ]
        )
        preflight_repo_identity(repos, fake)  # no exception
        assert fake.call_count == 2

    def test_raises_on_second_repo_mismatch(self, tmp_path: Path) -> None:
        from fido.server import preflight_repo_identity

        repos = {
            "owner/repo1": RepoConfig(name="owner/repo1", work_dir=tmp_path),
            "owner/repo2": RepoConfig(name="owner/repo2", work_dir=tmp_path),
        }
        fake = _FakeProcessRunner(
            [
                MagicMock(stdout="git@github.com:owner/repo1.git\n"),
                MagicMock(stdout="git@github.com:other/thing.git\n"),
            ]
        )
        with pytest.raises(PreflightError, match="other/thing"):
            preflight_repo_identity(repos, fake)

    def test_run_calls_preflight_repo_identity(self, tmp_path: Path) -> None:
        from fido.server import run

        fake_cfg = Config(
            port=0,
            secret=b"test",
            repos={"owner/repo": RepoConfig(name="owner/repo", work_dir=tmp_path)},
            allowed_bots=frozenset(),
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )
        mock_server = MagicMock()
        mock_server.serve_forever.side_effect = KeyboardInterrupt
        mock_preflight = MagicMock()

        run(
            _from_args=lambda: fake_cfg,
            _HTTPServer=lambda *a, **kw: mock_server,
            _make_registry=MagicMock(),
            _path_home=lambda: tmp_path,
            _basic_config=MagicMock(),
            _populate_memberships=MagicMock(),
            _startup_pull=MagicMock(),
            _preflight_repo_identity=mock_preflight,
            _preflight_tools=MagicMock(),
            _preflight_sub_dir=MagicMock(),
            _preflight_gh_auth=MagicMock(),
            _GitHub=MagicMock,
        )

        mock_preflight.assert_called_once_with(fake_cfg.repos, ANY)

    def test_run_calls_preflight_tools(self, tmp_path: Path) -> None:
        from fido.server import run

        fake_cfg = Config(
            port=0,
            secret=b"test",
            repos={"owner/repo": RepoConfig(name="owner/repo", work_dir=tmp_path)},
            allowed_bots=frozenset(),
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )
        mock_server = MagicMock()
        mock_server.serve_forever.side_effect = KeyboardInterrupt
        mock_preflight = MagicMock()

        run(
            _from_args=lambda: fake_cfg,
            _HTTPServer=lambda *a, **kw: mock_server,
            _make_registry=MagicMock(),
            _path_home=lambda: tmp_path,
            _basic_config=MagicMock(),
            _populate_memberships=MagicMock(),
            _startup_pull=MagicMock(),
            _preflight_repo_identity=MagicMock(),
            _preflight_tools=mock_preflight,
            _preflight_sub_dir=MagicMock(),
            _preflight_gh_auth=MagicMock(),
            _GitHub=MagicMock,
        )

        mock_preflight.assert_called_once_with(ANY)

    def test_run_calls_preflight_gh_auth(self, tmp_path: Path) -> None:
        from fido.server import run

        fake_cfg = Config(
            port=0,
            secret=b"test",
            repos={"owner/repo": RepoConfig(name="owner/repo", work_dir=tmp_path)},
            allowed_bots=frozenset(),
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )
        mock_server = MagicMock()
        mock_server.serve_forever.side_effect = KeyboardInterrupt
        mock_preflight = MagicMock()

        run(
            _from_args=lambda: fake_cfg,
            _HTTPServer=lambda *a, **kw: mock_server,
            _make_registry=MagicMock(),
            _path_home=lambda: tmp_path,
            _basic_config=MagicMock(),
            _populate_memberships=MagicMock(),
            _startup_pull=MagicMock(),
            _preflight_repo_identity=MagicMock(),
            _preflight_tools=MagicMock(),
            _preflight_sub_dir=MagicMock(),
            _preflight_gh_auth=mock_preflight,
            _GitHub=MagicMock,
        )

        mock_preflight.assert_called_once()

    def test_run_calls_preflight_sub_dir(self, tmp_path: Path) -> None:
        from fido.server import run

        fake_cfg = Config(
            port=0,
            secret=b"test",
            repos={"owner/repo": RepoConfig(name="owner/repo", work_dir=tmp_path)},
            allowed_bots=frozenset(),
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )
        mock_server = MagicMock()
        mock_server.serve_forever.side_effect = KeyboardInterrupt
        mock_preflight = MagicMock()

        run(
            _from_args=lambda: fake_cfg,
            _HTTPServer=lambda *a, **kw: mock_server,
            _make_registry=MagicMock(),
            _path_home=lambda: tmp_path,
            _basic_config=MagicMock(),
            _populate_memberships=MagicMock(),
            _startup_pull=MagicMock(),
            _preflight_repo_identity=MagicMock(),
            _preflight_tools=MagicMock(),
            _preflight_sub_dir=mock_preflight,
            _preflight_gh_auth=MagicMock(),
            _GitHub=MagicMock,
        )

        mock_preflight.assert_called_once_with(fake_cfg, ANY)

    def test_run_converts_preflight_error_to_system_exit(self, tmp_path: Path) -> None:
        from fido.server import run

        fake_cfg = Config(
            port=0,
            secret=b"test",
            repos={"owner/repo": RepoConfig(name="owner/repo", work_dir=tmp_path)},
            allowed_bots=frozenset(),
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )
        mock_server = MagicMock()

        with pytest.raises(SystemExit, match="something went wrong"):
            run(
                _from_args=lambda: fake_cfg,
                _HTTPServer=lambda *a, **kw: mock_server,
                _make_registry=MagicMock(),
                _path_home=lambda: tmp_path,
                _basic_config=MagicMock(),
                _populate_memberships=MagicMock(),
                _startup_pull=MagicMock(),
                _preflight_tools=MagicMock(
                    side_effect=PreflightError("something went wrong")
                ),
                _preflight_sub_dir=MagicMock(),
                _preflight_gh_auth=MagicMock(),
                _GitHub=MagicMock,
                _preflight_repo_identity=MagicMock(),
            )


class _FakeFilesystem:
    """Minimal Filesystem fake for preflight tests."""

    def __init__(
        self, missing: set[str] | None = None, dirs: set[Path] | None = None
    ) -> None:
        self._missing = missing or set()
        self._dirs = dirs or set()

    def which(self, name: str) -> str | None:
        return None if name in self._missing else f"/usr/bin/{name}"

    def is_dir(self, path: Path) -> bool:
        return path in self._dirs


class TestPreflightTools:
    def test_succeeds_when_all_tools_found(self) -> None:
        from fido.server import preflight_tools

        preflight_tools(_FakeFilesystem())  # no exception

    def test_raises_when_git_missing(self) -> None:
        from fido.server import preflight_tools

        missing = "git"
        with pytest.raises(PreflightError, match=repr(missing)):
            preflight_tools(_FakeFilesystem(missing={missing}))

    def test_raises_when_gh_missing(self) -> None:
        from fido.server import preflight_tools

        missing = "gh"
        with pytest.raises(PreflightError, match=repr(missing)):
            preflight_tools(_FakeFilesystem(missing={missing}))

    def test_raises_when_claude_missing(self) -> None:
        from fido.server import preflight_tools

        missing = "claude"
        with pytest.raises(PreflightError, match=repr(missing)):
            preflight_tools(_FakeFilesystem(missing={missing}))

    def test_raises_when_copilot_missing(self) -> None:
        from fido.server import preflight_tools

        missing = "copilot"
        with pytest.raises(PreflightError, match=repr(missing)):
            preflight_tools(_FakeFilesystem(missing={missing}))

    def test_required_tools_constant(self) -> None:
        from fido.server import _REQUIRED_TOOLS

        assert set(_REQUIRED_TOOLS) == {"git", "gh", "claude", "copilot"}


class TestPreflightSubDir:
    def test_succeeds_when_sub_dir_exists(self, tmp_path: Path) -> None:
        from fido.server import preflight_sub_dir

        sub = tmp_path / "sub"
        cfg = Config(
            port=0,
            secret=b"s",
            repos={},
            allowed_bots=frozenset(),
            log_level="INFO",
            sub_dir=sub,
        )
        preflight_sub_dir(cfg, _FakeFilesystem(dirs={sub}))  # no exception

    def test_raises_when_sub_dir_missing(self, tmp_path: Path) -> None:
        from fido.server import preflight_sub_dir

        cfg = Config(
            port=0,
            secret=b"s",
            repos={},
            allowed_bots=frozenset(),
            log_level="INFO",
            sub_dir=tmp_path / "sub",
        )
        with pytest.raises(PreflightError, match="skill-files directory not found"):
            preflight_sub_dir(cfg, _FakeFilesystem())

    def test_error_message_includes_path(self, tmp_path: Path) -> None:
        from fido.server import preflight_sub_dir

        sub = tmp_path / "my-sub"
        cfg = Config(
            port=0,
            secret=b"s",
            repos={},
            allowed_bots=frozenset(),
            log_level="INFO",
            sub_dir=sub,
        )
        with pytest.raises(PreflightError, match=str(sub)):
            preflight_sub_dir(cfg, _FakeFilesystem())


class TestPreflightGhAuth:
    def test_succeeds_when_get_user_works(self) -> None:
        from fido.server import preflight_gh_auth

        mock_gh = MagicMock()
        mock_gh.get_user.return_value = "fido-bot"
        preflight_gh_auth(mock_gh)  # no exception

    def test_raises_when_get_user_raises_runtime_error(self) -> None:
        from fido.server import preflight_gh_auth

        mock_gh = MagicMock()
        mock_gh.get_user.side_effect = RuntimeError("not logged in")
        with pytest.raises(PreflightError, match="not logged in"):
            preflight_gh_auth(mock_gh)

    def test_raises_when_get_user_raises_any_exception(self) -> None:
        from fido.server import preflight_gh_auth

        mock_gh = MagicMock()
        mock_gh.get_user.side_effect = Exception("network error")
        with pytest.raises(PreflightError, match="network error"):
            preflight_gh_auth(mock_gh)


class TestGetSelfRepo:
    def test_parses_ssh_remote(self, tmp_path: Path) -> None:
        from fido.server import _get_self_repo

        proc = _FakeProcessRunner([MagicMock(stdout="git@github.com:owner/fido.git\n")])
        assert _get_self_repo(tmp_path, proc) == "owner/fido"

    def test_parses_https_remote(self, tmp_path: Path) -> None:
        from fido.server import _get_self_repo

        proc = _FakeProcessRunner(
            [MagicMock(stdout="https://github.com/owner/fido.git\n")]
        )
        assert _get_self_repo(tmp_path, proc) == "owner/fido"

    def test_parses_remote_without_git_suffix(self, tmp_path: Path) -> None:
        from fido.server import _get_self_repo

        proc = _FakeProcessRunner([MagicMock(stdout="https://github.com/owner/fido\n")])
        assert _get_self_repo(tmp_path, proc) == "owner/fido"

    def test_returns_none_on_subprocess_error(self, tmp_path: Path) -> None:
        from fido.server import _get_self_repo

        proc = _FakeProcessRunner([subprocess.CalledProcessError(128, [])])
        assert _get_self_repo(tmp_path, proc) is None

    def test_returns_none_on_file_not_found(self, tmp_path: Path) -> None:
        from fido.server import _get_self_repo

        proc = _FakeProcessRunner([FileNotFoundError()])
        assert _get_self_repo(tmp_path, proc) is None

    def test_returns_none_on_unparseable_url(self, tmp_path: Path) -> None:
        from fido.server import _get_self_repo

        proc = _FakeProcessRunner([MagicMock(stdout="garbage\n")])
        assert _get_self_repo(tmp_path, proc) is None


class TestGetHead:
    def test_returns_sha(self, tmp_path: Path) -> None:
        from fido.server import _get_head

        proc = _FakeProcessRunner([MagicMock(stdout="abc123def456\n")])
        assert _get_head(tmp_path, proc) == "abc123def456"

    def test_returns_none_on_subprocess_error(self, tmp_path: Path) -> None:
        from fido.server import _get_head

        proc = _FakeProcessRunner([subprocess.CalledProcessError(128, [])])
        assert _get_head(tmp_path, proc) is None

    def test_returns_none_on_file_not_found(self, tmp_path: Path) -> None:
        from fido.server import _get_head

        proc = _FakeProcessRunner([FileNotFoundError()])
        assert _get_head(tmp_path, proc) is None


class TestRunnerDir:
    def test_returns_package_parent(self) -> None:
        from fido.server import _runner_dir

        result = _runner_dir()
        assert (result / "fido" / "server.py").exists()


class TestPullWithBackoff:
    def test_success_on_first_try(self, tmp_path: Path) -> None:
        from fido.server import _pull_with_backoff

        proc = _FakeProcessRunner([MagicMock(), MagicMock()])  # fetch + reset
        clock = _FakeClock(times=[0.0, 0.0])  # start + success log
        assert _pull_with_backoff(tmp_path, proc, clock)
        # Each sync attempt runs two git commands: fetch + reset.
        assert proc.call_count == 2
        cmds = [c[0] for c in proc.calls]
        assert cmds[0] == ["git", "fetch", "origin", "main"]
        assert cmds[1] == ["git", "reset", "--hard", "origin/main"]
        assert clock.slept == []

    def test_success_after_retry(self, tmp_path: Path) -> None:
        from fido.server import _pull_with_backoff

        # First attempt: fetch fails.  Second attempt: fetch + reset both succeed.
        proc = _FakeProcessRunner(
            [subprocess.CalledProcessError(1, []), MagicMock(), MagicMock()]
        )
        clock = _FakeClock(times=[0.0, 1.0, 1.0])  # start, fail-elapsed, success-log
        assert _pull_with_backoff(tmp_path, proc, clock)
        assert proc.call_count == 3
        assert clock.slept == [10]

    def test_recovers_from_divergent_local(self, tmp_path: Path) -> None:
        """The whole point of using fetch+reset: divergent local branch is fine.

        Previously git pull would fail with `fatal: Need to specify how to
        reconcile divergent branches` and the runner would never sync.  With
        reset --hard the local state is forcibly replaced, so the worker
        catches up regardless of how the runner clone got into a weird state.
        """
        from fido.server import _pull_with_backoff

        proc = _FakeProcessRunner([MagicMock(), MagicMock()])  # fetch + reset
        clock = _FakeClock(times=[0.0, 0.0])
        assert _pull_with_backoff(tmp_path, proc, clock)
        assert ["git", "reset", "--hard", "origin/main"] in [c[0] for c in proc.calls]
        assert clock.slept == []

    def test_gives_up_after_all_retries_fail(self, tmp_path: Path) -> None:
        from fido.server import _pull_with_backoff

        # Fetch fails on every attempt — 4 attempts total.
        proc = _FakeProcessRunner(error=subprocess.CalledProcessError(1, []))
        # start + one elapsed read per failed attempt (4 total)
        clock = _FakeClock(times=[0.0, 1.0, 12.0, 43.0, 104.0])
        assert not _pull_with_backoff(tmp_path, proc, clock)
        # 4 attempts × 1 failing fetch each = 4 calls.
        assert proc.call_count == 4
        # 3 sleeps at 10s, 30s, 60s between retries.
        assert clock.slept == [10, 30, 60]

    def test_reset_failure_retries(self, tmp_path: Path) -> None:
        """Fetch succeeds but reset fails — both commands matter for the result."""
        from fido.server import _pull_with_backoff

        proc = _FakeProcessRunner(
            [
                MagicMock(),  # fetch
                subprocess.CalledProcessError(1, []),  # reset
                MagicMock(),  # fetch retry
                MagicMock(),  # reset retry
            ]
        )
        clock = _FakeClock(times=[0.0, 1.0, 1.0])  # start, fail-elapsed, success-log
        assert _pull_with_backoff(tmp_path, proc, clock)
        assert proc.call_count == 4
        assert clock.slept == [10]

    def test_gives_up_when_budget_exhausted(self, tmp_path: Path) -> None:
        from fido.server import _pull_with_backoff

        proc = _FakeProcessRunner(error=subprocess.CalledProcessError(1, []))
        # First attempt at t=0, elapsed=595; next delay of 10s would exceed 600s budget.
        clock = _FakeClock(times=[0.0, 595.0])
        assert not _pull_with_backoff(tmp_path, proc, clock)
        # Slept zero times because budget was exhausted before any sleep.
        assert clock.slept == []

    def test_returns_false_on_file_not_found(self, tmp_path: Path) -> None:
        from fido.server import _pull_with_backoff

        proc = _FakeProcessRunner(error=FileNotFoundError())
        clock = _FakeClock(times=[0.0, 1.0, 12.0, 43.0, 104.0])
        assert not _pull_with_backoff(tmp_path, proc, clock)


class TestStartupPull:
    def test_exits_for_supervisor_when_head_changes(self) -> None:
        from fido.server import _startup_pull

        # rev-parse before, fetch, reset, rev-parse after — different SHAs
        proc = _FakeProcessRunner(
            [
                MagicMock(stdout="sha1\n"),
                MagicMock(),
                MagicMock(),
                MagicMock(stdout="sha2\n"),
            ]
        )
        clock = _FakeClock(times=[0.0, 0.0])  # start + success log
        os_proc = _FakeOsProcess()
        _startup_pull(proc, clock, os_proc)
        assert os_proc.exit_calls == [75]

    def test_skips_exit_when_head_unchanged(self) -> None:
        from fido.server import _startup_pull

        proc = _FakeProcessRunner(
            [
                MagicMock(stdout="same\n"),
                MagicMock(),
                MagicMock(),
                MagicMock(stdout="same\n"),
            ]
        )
        clock = _FakeClock(times=[0.0, 0.0])
        os_proc = _FakeOsProcess()
        _startup_pull(proc, clock, os_proc)
        assert os_proc.exit_calls == []

    def test_continues_on_pull_failure(self) -> None:
        from fido.server import _startup_pull

        # get_head fails → None; fetch fails with budget exhausted immediately
        proc = _FakeProcessRunner(
            [FileNotFoundError(), subprocess.CalledProcessError(1, [])]
        )
        clock = _FakeClock(times=[0.0, 601.0])  # budget exhausted on first attempt
        os_proc = _FakeOsProcess()
        _startup_pull(proc, clock, os_proc)
        assert os_proc.exit_calls == []

    def test_does_not_exit_when_head_unknown(self) -> None:
        from fido.server import _startup_pull

        # Both rev-parse calls fail → head_before and head_after are None
        proc = _FakeProcessRunner(
            [FileNotFoundError(), MagicMock(), MagicMock(), FileNotFoundError()]
        )
        clock = _FakeClock(times=[0.0, 0.0])
        os_proc = _FakeOsProcess()
        _startup_pull(proc, clock, os_proc)
        # Can't compare HEAD — pull succeeded, log and continue without exit
        assert os_proc.exit_calls == []


class TestSelfRestart:
    """Tests for the self-restart flow."""

    def _make_server(self, tmp_path: Path):
        cfg = _self_restart_cfg(tmp_path)
        mock_registry = MagicMock()
        WebhookHandler.config = cfg
        WebhookHandler.registry = mock_registry
        srv = HTTPServer(("127.0.0.1", 0), WebhookHandler)
        port = srv.server_address[1]
        t = threading.Thread(
            target=srv.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        t.start()
        return srv, f"http://127.0.0.1:{port}", cfg, mock_registry

    def test_triggers_restart_exit_on_matching_repo(self, tmp_path: Path) -> None:
        srv, url, cfg, mock_registry = self._make_server(tmp_path)
        try:
            WebhookHandler._fn_runner_dir = lambda: tmp_path  # type: ignore[assignment]
            os_proc = _FakeOsProcess()
            WebhookHandler.infra = Infra(
                proc=_FakeProcessRunner(
                    [
                        MagicMock(stdout="git@github.com:owner/fido.git\n"),
                        MagicMock(),  # fetch
                        MagicMock(),  # reset
                    ]
                ),
                clock=_FakeClock(times=[0.0, 0.0]),
                fs=_FakeFilesystem(),
                os_proc=os_proc,
            )
            status = _post_webhook(url, cfg, "pull_request", _MERGE_PAYLOAD)
            assert status == 200
            mock_registry.stop_and_join.assert_called_once_with("owner/fido")
            assert os_proc.chdir_calls == [tmp_path]
            assert os_proc.exit_calls == [75]
        finally:
            srv.shutdown()

    def test_kills_active_children_before_restart_exit(self, tmp_path: Path) -> None:
        """Self-restart must stop every worker and SIGTERM every tracked child
        BEFORE exiting for the host supervisor."""
        srv, url, cfg, mock_registry = self._make_server(tmp_path)
        try:
            WebhookHandler._fn_runner_dir = lambda: tmp_path  # type: ignore[assignment]
            call_log: list[str] = []

            def _kill():
                call_log.append("kill_active_children")

            real_stop_all = mock_registry.stop_all
            real_stop_and_join = mock_registry.stop_and_join
            real_stop_all.side_effect = lambda: call_log.append("stop_all")
            real_stop_and_join.side_effect = lambda repo: call_log.append(
                f"stop_and_join:{repo}"
            )
            WebhookHandler._fn_kill_active_children = staticmethod(_kill)  # type: ignore[assignment]

            os_proc = _FakeOsProcess()

            def _tracking_exit(code):  # type: ignore[no-untyped-def]
                call_log.append(f"exit:{code}")
                os_proc.exit_calls.append(code)

            os_proc.exit = _tracking_exit  # type: ignore[method-assign]
            WebhookHandler.infra = Infra(
                proc=_FakeProcessRunner(
                    [
                        MagicMock(stdout="git@github.com:owner/fido.git\n"),
                        MagicMock(),  # fetch
                        MagicMock(),  # reset
                    ]
                ),
                clock=_FakeClock(times=[0.0, 0.0]),
                fs=_FakeFilesystem(),
                os_proc=os_proc,
            )
            status = _post_webhook(url, cfg, "pull_request", _MERGE_PAYLOAD)
            assert status == 200
            # stop_and_join + stop_all + kill_active_children MUST all run
            # before host-supervised restart exit.
            exit_idx = call_log.index("exit:75")
            assert "stop_and_join:owner/fido" in call_log[:exit_idx]
            assert "stop_all" in call_log[:exit_idx]
            assert "kill_active_children" in call_log[:exit_idx]
            # Defensive: kill_active_children should be the last thing
            # before exit so it's the most recent TERM.
            assert call_log[exit_idx - 1] == "kill_active_children"
        finally:
            srv.shutdown()

    def test_skips_when_self_repo_mismatch(self, tmp_path: Path) -> None:
        srv, url, cfg, mock_registry = self._make_server(tmp_path)
        try:
            WebhookHandler._fn_runner_dir = lambda: tmp_path  # type: ignore[assignment]
            proc = _FakeProcessRunner(
                [MagicMock(stdout="git@github.com:other/repo.git\n")]
            )
            os_proc = _FakeOsProcess()
            WebhookHandler.infra = Infra(
                proc=proc,
                clock=_FakeClock(),
                fs=_FakeFilesystem(),
                os_proc=os_proc,
            )
            _post_webhook(url, cfg, "pull_request", _MERGE_PAYLOAD)
            mock_registry.stop_and_join.assert_not_called()
            assert proc.call_count == 1  # only the get-url call; no pull attempted
            assert os_proc.exit_calls == []
        finally:
            srv.shutdown()

    def test_skips_when_self_repo_unknown(self, tmp_path: Path) -> None:
        srv, url, cfg, mock_registry = self._make_server(tmp_path)
        try:
            WebhookHandler._fn_runner_dir = lambda: tmp_path  # type: ignore[assignment]
            os_proc = _FakeOsProcess()
            WebhookHandler.infra = Infra(
                proc=_FakeProcessRunner([FileNotFoundError()]),
                clock=_FakeClock(),
                fs=_FakeFilesystem(),
                os_proc=os_proc,
            )
            _post_webhook(url, cfg, "pull_request", _MERGE_PAYLOAD)
            mock_registry.stop_and_join.assert_not_called()
            assert os_proc.exit_calls == []
        finally:
            srv.shutdown()

    def test_pull_failure_leaves_worker_alone(self, tmp_path: Path) -> None:
        """When pull gives up, do NOT tear down the worker thread.

        Previously the worker for the merged repo was stopped before the
        pull, and stayed stopped after pull failure — silently leaving
        fido without a fido worker on its own repo.  Now the pull runs
        first and the worker is only torn down on success.
        """
        srv, url, cfg, mock_registry = self._make_server(tmp_path)
        try:
            WebhookHandler._fn_runner_dir = lambda: tmp_path  # type: ignore[assignment]
            os_proc = _FakeOsProcess()
            WebhookHandler.infra = Infra(
                proc=_FakeProcessRunner(
                    [
                        MagicMock(stdout="git@github.com:owner/fido.git\n"),
                        subprocess.CalledProcessError(
                            1, []
                        ),  # fetch fails; budget exhausted
                    ]
                ),
                clock=_FakeClock(times=[0.0, 601.0]),
                fs=_FakeFilesystem(),
                os_proc=os_proc,
            )
            _post_webhook(url, cfg, "pull_request", _MERGE_PAYLOAD)
            mock_registry.stop_and_join.assert_not_called()
            assert os_proc.exit_calls == []
        finally:
            srv.shutdown()

    def test_pull_precedes_stop_and_join(self, tmp_path: Path) -> None:
        """Pull MUST complete before stop_and_join so a failed pull leaves the
        worker intact."""
        srv, url, cfg, mock_registry = self._make_server(tmp_path)
        try:
            WebhookHandler._fn_runner_dir = lambda: tmp_path  # type: ignore[assignment]
            proc = _FakeProcessRunner(
                [
                    MagicMock(stdout="git@github.com:owner/fido.git\n"),
                    MagicMock(),  # fetch
                    MagicMock(),  # reset
                ]
            )
            os_proc = _FakeOsProcess()
            WebhookHandler.infra = Infra(
                proc=proc,
                clock=_FakeClock(times=[0.0, 0.0]),
                fs=_FakeFilesystem(),
                os_proc=os_proc,
            )
            _post_webhook(url, cfg, "pull_request", _MERGE_PAYLOAD)
            # All 3 proc calls (get-url + fetch + reset) happened before stop_and_join.
            assert proc.call_count == 3
            mock_registry.stop_and_join.assert_called_once_with("owner/fido")
        finally:
            srv.shutdown()

    def test_push_to_default_branch_triggers_restart(self, tmp_path: Path) -> None:
        srv, url, cfg, mock_registry = self._make_server(tmp_path)
        try:
            WebhookHandler._fn_runner_dir = lambda: tmp_path  # type: ignore[assignment]
            os_proc = _FakeOsProcess()
            WebhookHandler.infra = Infra(
                proc=_FakeProcessRunner(
                    [
                        MagicMock(stdout="git@github.com:owner/fido.git\n"),
                        MagicMock(),  # fetch
                        MagicMock(),  # reset
                    ]
                ),
                clock=_FakeClock(times=[0.0, 0.0]),
                fs=_FakeFilesystem(),
                os_proc=os_proc,
            )
            _post_webhook(url, cfg, "push", _PUSH_PAYLOAD)
            mock_registry.stop_and_join.assert_called_once_with("owner/fido")
            assert os_proc.exit_calls == [75]
        finally:
            srv.shutdown()

    def test_push_to_non_default_branch_ignored(self, tmp_path: Path) -> None:
        srv, url, cfg, mock_registry = self._make_server(tmp_path)
        try:
            WebhookHandler._fn_runner_dir = lambda: tmp_path  # type: ignore[assignment]
            proc = _FakeProcessRunner([])
            os_proc = _FakeOsProcess()
            WebhookHandler.infra = Infra(
                proc=proc,
                clock=_FakeClock(),
                fs=_FakeFilesystem(),
                os_proc=os_proc,
            )
            payload = {**_PUSH_PAYLOAD, "ref": "refs/heads/feature-branch"}
            _post_webhook(url, cfg, "push", payload)
            assert proc.calls == []
            assert os_proc.exit_calls == []
        finally:
            srv.shutdown()

    def _make_unregistered_server(self, tmp_path: Path):
        """Server whose config does NOT include the fido repo."""
        from fido.config import RepoMembership

        cfg = Config(
            port=0,
            secret=b"test-secret",
            repos={
                "owner/other": RepoConfig(
                    name="owner/other",
                    work_dir=tmp_path,
                    membership=RepoMembership(collaborators=frozenset()),
                )
            },
            allowed_bots=frozenset(),
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )
        mock_registry = MagicMock()
        WebhookHandler.config = cfg
        WebhookHandler.registry = mock_registry
        srv = HTTPServer(("127.0.0.1", 0), WebhookHandler)
        port = srv.server_address[1]
        t = threading.Thread(
            target=srv.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        t.start()
        return srv, f"http://127.0.0.1:{port}", cfg

    def test_merged_pr_on_unregistered_repo_triggers_restart(
        self, tmp_path: Path
    ) -> None:
        """Self-restart fires for merged PR even when the repo is not registered."""
        srv, url, cfg = self._make_unregistered_server(tmp_path)
        try:
            WebhookHandler._fn_runner_dir = lambda: tmp_path  # type: ignore[assignment]
            os_proc = _FakeOsProcess()
            WebhookHandler.infra = Infra(
                proc=_FakeProcessRunner(
                    [
                        MagicMock(stdout="git@github.com:owner/fido.git\n"),
                        MagicMock(),  # fetch
                        MagicMock(),  # reset
                    ]
                ),
                clock=_FakeClock(times=[0.0, 0.0]),
                fs=_FakeFilesystem(),
                os_proc=os_proc,
            )
            status = _post_webhook(url, cfg, "pull_request", _MERGE_PAYLOAD)
            assert status == 200
            assert os_proc.exit_calls == [75]
        finally:
            srv.shutdown()

    def test_push_to_default_on_unregistered_repo_triggers_restart(
        self, tmp_path: Path
    ) -> None:
        """Self-restart fires for push to default branch even when repo is not registered."""
        srv, url, cfg = self._make_unregistered_server(tmp_path)
        try:
            WebhookHandler._fn_runner_dir = lambda: tmp_path  # type: ignore[assignment]
            os_proc = _FakeOsProcess()
            WebhookHandler.infra = Infra(
                proc=_FakeProcessRunner(
                    [
                        MagicMock(stdout="git@github.com:owner/fido.git\n"),
                        MagicMock(),  # fetch
                        MagicMock(),  # reset
                    ]
                ),
                clock=_FakeClock(times=[0.0, 0.0]),
                fs=_FakeFilesystem(),
                os_proc=os_proc,
            )
            status = _post_webhook(url, cfg, "push", _PUSH_PAYLOAD)
            assert status == 200
            assert os_proc.exit_calls == [75]
        finally:
            srv.shutdown()
