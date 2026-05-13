import dataclasses
import hashlib
import hmac
import json
import logging
import socket
import subprocess
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from http.server import HTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import ANY, MagicMock, patch

import pytest
from frozendict import frozendict

from fido import provider
from fido.appstate import (
    _ZERO_GITHUB_LIMITS,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    FidoState,
    GitHubLimit,
    IssueSnapshot,
    ProviderLimitWindow,
    ProviderSnapshot,
    RepoState,
    WebhookActivity,
    WorkerActivity,
    WorkerCrash,
    zero_repo_state,
)
from fido.claude import ClaudeClient
from fido.config import Config
from fido.config import RepoConfig as _RepoConfig
from fido.events import (
    Action,
    Dispatcher,
    WebhookIngressOracle,
)
from fido.infra import Infra
from fido.provider import ProviderID
from fido.server import FidoHTTPServer, PreflightError, WebhookHandler
from fido.store import FidoStore
from fido.tasks import Tasks
from fido.types import TaskStatus, TaskType
from tests.fakes import _FakeDispatcher

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _zero_crash() -> WorkerCrash:
    return WorkerCrash(death_count=0, last_error="", last_crash_time=_EPOCH)


def _repo_state(
    repo_name: str,
    what: str = "Working on: #1",
    busy: bool = True,
    crash_count: int = 0,
    last_error: str = "",
    stale: bool = False,
    started_at: datetime | None = None,
    webhook_activities: tuple[WebhookActivity, ...] = (),
    provider: ProviderSnapshot | None = None,
    issue: IssueSnapshot | None = None,
) -> RepoState:
    progress_at = (
        datetime(2020, 1, 1, tzinfo=timezone.utc)
        if stale
        else datetime.now(tz=timezone.utc)
    )
    base = zero_repo_state(repo_name, started_at=started_at or _EPOCH)
    return dataclasses.replace(
        base,
        activity=WorkerActivity(
            repo_name=repo_name,
            what=what,
            busy=busy,
            last_progress_at=progress_at,
        ),
        crash_record=WorkerCrash(
            death_count=crash_count,
            last_error=last_error,
            last_crash_time=_EPOCH,
        ),
        webhook_activities=webhook_activities,
        provider=provider if provider is not None else base.provider,
        issue=issue if issue is not None else base.issue,
    )


def _fido_state(
    *repo_states: RepoState,
    github_limits: GitHubLimit = _ZERO_GITHUB_LIMITS,
) -> FidoState:
    return FidoState(
        repos=frozendict({rs.key: rs for rs in repo_states}),
        github_limits=github_limits,
        process_started_at=_EPOCH,
    )


class RepoConfig(_RepoConfig):
    def __init__(
        self,
        *args: object,
        provider: ProviderID = ProviderID.CLAUDE_CODE,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, provider=provider, **kwargs)


def _client(return_value: str = "", *, side_effect: object = None) -> MagicMock:
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

    # Registry.start resolves the canonical git_dir via
    # ``git rev-parse --absolute-git-dir`` (#1696 codex P1 round 5)
    # so the work_dir must be a real git repo.  ``git init`` is
    # idempotent — re-initialising an existing repo is safe.
    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

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
def _restore_handler_fns() -> object:
    saved = {
        "gh": WebhookHandler.gh,
        "dispatchers": WebhookHandler.dispatchers,
        "_fn_reply_to_comment": WebhookHandler._fn_reply_to_comment,
        "_fn_reply_to_review": WebhookHandler._fn_reply_to_review,
        "_fn_reply_to_issue_comment": WebhookHandler._fn_reply_to_issue_comment,
        "_fn_create_task": WebhookHandler._fn_create_task,
        "_fn_launch_worker": WebhookHandler._fn_launch_worker,
        "_fn_spawn_bg": WebhookHandler._fn_spawn_bg,
        "_fn_after_do_post": WebhookHandler._fn_after_do_post,
        "_fn_runner_dir": WebhookHandler._fn_runner_dir,
        "infra": WebhookHandler.infra,
        "static_files": WebhookHandler.static_files,
        "_restart_fsm_state": WebhookHandler._restart_fsm_state,
    }
    # Reset the ingress oracle so each test starts with a clean delivery-ID
    # table — tests reuse the same delivery ID ("test") so without a reset,
    # the second test to run would see the delivery as already dispatched and
    # suppress it via the Redeliver path.
    WebhookHandler.ingress_oracle = WebhookIngressOracle()
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


@pytest.fixture()
def server(tmp_path: Path) -> object:
    cfg = _config(tmp_path)
    repo_cfg = cfg.repos["owner/repo"]
    WebhookHandler.config = cfg
    WebhookHandler.registry = MagicMock()
    WebhookHandler.state_reader = MagicMock()
    WebhookHandler.dispatchers = {"owner/repo": Dispatcher(cfg, repo_cfg, MagicMock())}
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
        body = json.dumps(
            {
                "hook_id": 1,
                "repository": {"full_name": "other/repo", "default_branch": "main"},
            }
        ).encode()
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
    """``/status.json`` is now a pure :class:`FidoState` dump.

    The compatibility layer (``_collect_status_payload``,
    ``_collect_activities``, ``_collect_fido_state``, ``_repo_status``,
    ``_serialize_*``, the XML renderer) was removed in #1696.  These
    tests exercise the new shape — a JSON serialization of the
    snapshot dataclass tree via :func:`fido.server._jsonable`.
    """

    def test_health_check(self, server: tuple) -> None:
        url, _ = server
        resp = urllib.request.urlopen(url)
        assert resp.status == 200
        assert b"fido is running" in resp.read()

    def test_status_endpoint_returns_json(self, server: tuple) -> None:
        url, _ = server
        WebhookHandler.state_reader.get.return_value = _fido_state()
        resp = urllib.request.urlopen(f"{url}/status.json")
        assert resp.status == 200
        assert resp.headers.get("Content-Type") == "application/json"
        data = json.loads(resp.read())
        assert data == {
            "repos": {},
            "github_limits": {
                "rest": {
                    "name": "rest",
                    "used": 0,
                    "limit": 0,
                    "resets_at": "1970-01-01T00:00:00+00:00",
                    "unit": "",
                },
                "graphql": {
                    "name": "graphql",
                    "used": 0,
                    "limit": 0,
                    "resets_at": "1970-01-01T00:00:00+00:00",
                    "unit": "",
                },
            },
            "process_started_at": "1970-01-01T00:00:00+00:00",
        }

    def test_status_endpoint_dumps_repo_state(self, server: tuple) -> None:
        url, _ = server
        WebhookHandler.state_reader.get.return_value = _fido_state(
            _repo_state("owner/repo")
        )
        resp = urllib.request.urlopen(f"{url}/status.json")
        data = json.loads(resp.read())

        assert "owner/repo" in data["repos"]
        repo = data["repos"]["owner/repo"]
        assert repo["key"] == "owner/repo"
        assert repo["activity"]["repo_name"] == "owner/repo"
        assert repo["activity"]["what"] == "Working on: #1"
        assert repo["activity"]["busy"] is True
        assert repo["crash_record"]["death_count"] == 0
        assert repo["webhook_activities"] == []
        # Zero-sentinel leaves: every snapshot field carries a real
        # value (#1696 — no None on FidoState), and the zeros are
        # documented per leaf in :mod:`fido.appstate`.
        assert repo["thread"] == {
            "is_alive": False,
            "was_stopped": False,
            "crash_error": "",
        }
        assert repo["provider"]["session_owner"] == ""
        assert repo["provider"]["session_alive"] is False
        assert repo["issue"]["issue"] == 0
        assert repo["task_list"]["pending_task_count"] == 0
        assert repo["rescoping"] is False

    def test_status_endpoint_dumps_provider_snapshot(self, server: tuple) -> None:
        url, _ = server
        WebhookHandler.state_reader.get.return_value = _fido_state(
            _repo_state(
                "owner/repo",
                provider=ProviderSnapshot(
                    session_owner="worker-home",
                    session_alive=True,
                    session_pid=4321,
                    session_dropped_count=3,
                    session_sent_count=10,
                    session_received_count=8,
                ),
            )
        )
        resp = urllib.request.urlopen(f"{url}/status.json")
        data = json.loads(resp.read())
        provider = data["repos"]["owner/repo"]["provider"]
        assert provider == {
            "session_owner": "worker-home",
            "session_alive": True,
            "session_pid": 4321,
            "session_dropped_count": 3,
            "session_sent_count": 10,
            "session_received_count": 8,
        }

    def test_status_endpoint_dumps_issue_snapshot(self, server: tuple) -> None:
        url, _ = server
        snapshot = IssueSnapshot(
            issue=42,
            issue_title="Fix the thing",
            issue_started_at="2026-04-01T00:00:00+00:00",
            pr_number=99,
            pr_title="Fixes the thing",
        )
        WebhookHandler.state_reader.get.return_value = _fido_state(
            _repo_state("owner/repo", issue=snapshot)
        )
        resp = urllib.request.urlopen(f"{url}/status.json")
        data = json.loads(resp.read())
        assert data["repos"]["owner/repo"]["issue"] == {
            "issue": 42,
            "issue_title": "Fix the thing",
            "issue_started_at": "2026-04-01T00:00:00+00:00",
            "pr_number": 99,
            "pr_title": "Fixes the thing",
        }

    def test_status_endpoint_dumps_github_limits(self, server: tuple) -> None:
        url, _ = server
        limits = GitHubLimit(
            rest=ProviderLimitWindow(
                name="rest",
                used=5,
                limit=5000,
                resets_at=datetime(2024, 11, 14, 12, 0, tzinfo=timezone.utc),
                unit="",
            ),
            graphql=ProviderLimitWindow(
                name="graphql",
                used=12,
                limit=5000,
                resets_at=datetime(2024, 11, 14, 13, 0, tzinfo=timezone.utc),
                unit="",
            ),
        )
        WebhookHandler.state_reader.get.return_value = _fido_state(github_limits=limits)
        resp = urllib.request.urlopen(f"{url}/status.json")
        data = json.loads(resp.read())
        assert data["github_limits"]["rest"]["used"] == 5
        assert data["github_limits"]["rest"]["limit"] == 5000
        assert data["github_limits"]["rest"]["resets_at"] == "2024-11-14T12:00:00+00:00"
        assert data["github_limits"]["graphql"]["used"] == 12

    def test_status_endpoint_dumps_process_started_at(self, server: tuple) -> None:
        url, _ = server
        started_at = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        WebhookHandler.state_reader.get.return_value = FidoState(
            repos=frozendict(),
            github_limits=_ZERO_GITHUB_LIMITS,
            process_started_at=started_at,
        )
        resp = urllib.request.urlopen(f"{url}/status.json")
        data = json.loads(resp.read())
        assert data["process_started_at"] == "2026-01-01T12:00:00+00:00"

    def test_status_endpoint_dumps_webhook_activity(self, server: tuple) -> None:
        url, _ = server
        wa = WebhookActivity(
            handle_id=1,
            description="issue_comment.created",
            started_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
            thread_id=12345,
        )
        WebhookHandler.state_reader.get.return_value = _fido_state(
            _repo_state("owner/repo", webhook_activities=(wa,))
        )
        resp = urllib.request.urlopen(f"{url}/status.json")
        data = json.loads(resp.read())
        webhooks = data["repos"]["owner/repo"]["webhook_activities"]
        assert webhooks == [
            {
                "handle_id": 1,
                "description": "issue_comment.created",
                "started_at": "2026-04-01T00:00:00+00:00",
                "thread_id": 12345,
            }
        ]


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


class TestMalformedPayload:
    """Payloads missing schema-required keys return 500 so GitHub retries."""

    def _post_signed(self, url: str, cfg: Config, event: str, payload: dict) -> int:
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
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        return exc_info.value.code

    def test_missing_repository_returns_500(self, server: tuple) -> None:
        url, cfg = server
        assert self._post_signed(url, cfg, "issues", {"action": "opened"}) == 500

    def test_missing_default_branch_returns_500(self, server: tuple) -> None:
        url, cfg = server
        assert (
            self._post_signed(
                url, cfg, "issues", {"repository": {"full_name": "owner/repo"}}
            )
            == 500
        )

    def test_missing_action_on_pull_request_returns_500(self, server: tuple) -> None:
        url, cfg = server
        assert (
            self._post_signed(
                url,
                cfg,
                "pull_request",
                {
                    "repository": {
                        "full_name": "owner/repo",
                        "default_branch": "main",
                    }
                },
            )
            == 500
        )


def _payload(repo_owner: str = "owner") -> dict:
    """Base webhook payload fragment with the repository block every event needs."""
    return {
        "repository": {
            "full_name": f"{repo_owner}/repo",
            "owner": {"login": repo_owner},
            "default_branch": "main",
        },
    }


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

    def test_assigned_event_patches_cache(self, server: tuple) -> None:
        url, cfg = server
        cache = MagicMock()
        WebhookHandler.registry.get_issue_cache.return_value = cache
        payload = {
            **_payload(),
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
            **_payload(),
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
            **_payload(),
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
            **_payload(),
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

    def test_dispatch_triggers_worker_on_merged_pr(self, server: tuple) -> None:
        url, cfg = server
        payload = {
            **_payload(),
            "action": "closed",
            "pull_request": {"number": 7, "merged": True},
        }
        mock_worker = MagicMock()
        WebhookHandler._fn_launch_worker = mock_worker
        WebhookHandler._fn_create_task = MagicMock()
        status = _post_webhook(url, cfg, "pull_request", payload)
        assert status == 200
        mock_worker.assert_called()

    def test_reply_to_comment_defer_skips_task(self, server: tuple) -> None:
        """DEFER files a GitHub issue instead — no tasks.json entry."""
        url, cfg = server
        payload = {
            **_payload(),
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

    def test_already_replied_comment_skipped(self, server: tuple) -> None:
        url, cfg = server
        promise = FidoStore(cfg.repos["owner/repo"].work_dir).prepare_reply(
            owner="webhook", comment_type="pulls", anchor_comment_id=203
        )
        assert promise is not None
        payload = {
            **_payload(),
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
            **_payload(),
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
            **_payload(),
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

    def test_review_comments_handled(self, server: tuple) -> None:
        # pull_request_review / submitted with state="commented" is collapsed by
        # the ingress oracle (CollapseReview) so reply_to_review is never called
        # — inline comments are handled individually by pull_request_review_comment
        # events.  Decisive states (approved, changes_requested, dismissed) are
        # NOT collapsed and do wake the worker.
        url, cfg = server
        payload = {
            **_payload(),
            "action": "submitted",
            "review": {
                "id": 888,
                "state": "commented",
                "user": {"login": "owner"},
            },
            "pull_request": {"number": 9},
        }
        mock_review = MagicMock()
        WebhookHandler._fn_reply_to_review = mock_review
        WebhookHandler._fn_launch_worker = MagicMock()
        status = _post_webhook(url, cfg, "pull_request_review", payload)
        assert status == 200
        mock_review.assert_not_called()

    def test_process_action_does_not_overwrite_worker_what(self, server: tuple) -> None:
        """_process_action must not call report_activity — the webhook runs on
        a separate thread and writing the worker's own worker_what field
        from here clobbers the worker thread's state.  Webhook activity is
        tracked via registry.webhook_activity instead.
        """
        url, cfg = server
        payload = {
            **_payload(),
            "action": "closed",
            "pull_request": {"number": 14, "merged": True},
        }
        WebhookHandler._fn_launch_worker = MagicMock()
        status = _post_webhook(url, cfg, "pull_request", payload)
        assert status == 200
        for call in WebhookHandler.registry.report_activity.call_args_list:
            args = call.args
            assert "handling webhook action" not in args

    def test_claude_leak_halts_process(
        self,
        server: tuple,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """SessionLeakError from a webhook handler calls os._exit(3)."""
        from fido import server as server_module

        url, cfg = server
        payload = {
            **_payload(),
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
            **_payload(),
            "action": "closed",
            "pull_request": {"number": 13, "merged": True},
        }
        WebhookHandler.gh = MagicMock()
        WebhookHandler._fn_launch_worker = MagicMock(side_effect=Exception("explode"))
        status = _post_webhook(url, cfg, "pull_request", payload)
        assert status == 200
        # server still alive — no crash

    def test_process_action_error_no_reaction_without_comment(
        self, server: tuple
    ) -> None:
        """On exception with no comment context (e.g., merged PR), no reaction."""
        url, cfg = server
        payload = {
            **_payload(),
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
            **_payload(),
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

    def test_issue_comment_webhook_activity_tracks_phase(self, tmp_path: Path) -> None:
        from fido.appstate import FidoState
        from fido.atomic import create_atomic
        from fido.events import Action
        from fido.registry import WorkerRegistry

        cfg = _config(tmp_path)
        handler = WebhookHandler.__new__(WebhookHandler)
        handler.config = cfg
        _, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        handler.registry = WorkerRegistry(MagicMock(), updater)
        handler.registry.start(cfg.repos["owner/repo"])
        handler.gh = MagicMock()
        handler.dispatchers = {"owner/repo": _FakeDispatcher()}
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
        payload = {**_payload(), "action": "created"}
        mock_dispatcher = _FakeDispatcher(dispatch_side_effect=RuntimeError("boom"))
        WebhookHandler.dispatchers = {"owner/repo": mock_dispatcher}
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _post_webhook(url, cfg, "pull_request_review_comment", payload)
        assert exc_info.value.code == 500

    def test_dispatch_called_before_ack(self, server: tuple) -> None:
        """dispatch() must be called before the HTTP response is written."""
        url, cfg = server
        payload = {
            **_payload(),
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

        def fake_dispatch(*_args: object, **_kwargs: object) -> None:
            call_order.append("dispatch")

        original_respond = WebhookHandler._respond

        def fake_respond(self: object, code: int, msg: str) -> None:
            call_order.append(f"respond_{code}")
            original_respond(self, code, msg)

        mock_dispatcher = _FakeDispatcher(
            dispatch_side_effect=lambda *_a, **_kw: call_order.append("dispatch")
        )
        WebhookHandler.dispatchers = {"owner/repo": mock_dispatcher}
        WebhookHandler._respond = fake_respond  # type: ignore[method-assign]
        try:
            _post_webhook(url, cfg, "pull_request_review_comment", payload)
        finally:
            WebhookHandler._respond = original_respond  # type: ignore[method-assign]
        assert call_order == ["dispatch", "respond_200"]

    def test_review_comment_calls_unblock_tasks(self, server: tuple) -> None:
        """A pull_request_review_comment transitions BLOCKED tasks to PENDING."""
        url, cfg = server
        work_dir = cfg.repos["owner/repo"].work_dir
        Tasks(work_dir).add("blocked task", TaskType.SPEC, status=TaskStatus.BLOCKED)
        payload = {
            **_payload(),
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
        WebhookHandler._fn_reply_to_comment = MagicMock(return_value=("ANSWER", []))
        WebhookHandler._fn_launch_worker = MagicMock()
        status = _post_webhook(url, cfg, "pull_request_review_comment", payload)
        assert status == 200
        tasks = Tasks(work_dir).list()
        assert all(t["status"] == str(TaskStatus.PENDING) for t in tasks)

    def test_issue_comment_calls_unblock_tasks(self, server: tuple) -> None:
        """A top-level PR comment transitions BLOCKED tasks to PENDING."""
        url, cfg = server
        work_dir = cfg.repos["owner/repo"].work_dir
        Tasks(work_dir).add("blocked task", TaskType.SPEC, status=TaskStatus.BLOCKED)
        payload = {
            **_payload(),
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
        WebhookHandler.gh = MagicMock()
        WebhookHandler._fn_reply_to_issue_comment = MagicMock(
            return_value=("ANSWER", [])
        )
        WebhookHandler._fn_launch_worker = MagicMock()
        status = _post_webhook(url, cfg, "issue_comment", payload)
        assert status == 200
        tasks = Tasks(work_dir).list()
        assert all(t["status"] == str(TaskStatus.PENDING) for t in tasks)

    def test_non_comment_event_does_not_call_unblock_tasks(self, server: tuple) -> None:
        """A PR merge event (no comment body) must NOT unblock BLOCKED tasks."""
        url, cfg = server
        work_dir = cfg.repos["owner/repo"].work_dir
        Tasks(work_dir).add("blocked task", TaskType.SPEC, status=TaskStatus.BLOCKED)
        payload = {
            **_payload(),
            "action": "closed",
            "pull_request": {"number": 72, "merged": True},
        }
        WebhookHandler._fn_launch_worker = MagicMock()
        status = _post_webhook(url, cfg, "pull_request", payload)
        assert status == 200
        tasks = Tasks(work_dir).list()
        assert all(t["status"] == str(TaskStatus.BLOCKED) for t in tasks)

    def test_review_comment_handler_stays_off_provider(self, server: tuple) -> None:
        """Review-comment ingestion queues durably without entering provider."""
        url, cfg = server
        mock_session = MagicMock()
        WebhookHandler.registry.get_session.return_value = mock_session
        mock_reply = MagicMock(return_value=("ANSWER", []))
        WebhookHandler._fn_reply_to_comment = mock_reply
        WebhookHandler._fn_create_task = MagicMock()
        WebhookHandler._fn_launch_worker = MagicMock()
        payload = {
            **_payload(),
            "action": "created",
            "comment": {
                "id": 950,
                "body": "please refactor this",
                "user": {"login": "owner"},
                "html_url": "https://example.com",
                "path": "src/foo.py",
                "line": 1,
                "diff_hunk": "@@ @@",
            },
            "pull_request": {"number": 10, "title": "My PR", "body": ""},
        }
        status = _post_webhook(url, cfg, "pull_request_review_comment", payload)
        assert status == 200
        mock_session.hold_for_handler.assert_not_called()
        mock_reply.assert_not_called()
        WebhookHandler._fn_launch_worker.assert_called_once()

    def test_issue_comment_handler_stays_off_provider(self, server: tuple) -> None:
        """Top-level PR comment ingestion queues durably without provider use."""
        url, cfg = server
        mock_session = MagicMock()
        WebhookHandler.registry.get_session.return_value = mock_session
        mock_reply = MagicMock(return_value=("ANSWER", []))
        WebhookHandler._fn_reply_to_issue_comment = mock_reply
        WebhookHandler._fn_create_task = MagicMock()
        WebhookHandler._fn_launch_worker = MagicMock()
        payload = {
            **_payload(),
            "action": "created",
            "comment": {
                "id": 1007,
                "body": "please update the docs",
                "user": {"login": "owner"},
                "html_url": "https://github.com/owner/repo/pull/90#issuecomment-1007",
            },
            "issue": {
                "number": 90,
                "title": "my pr",
                "body": "",
                "pull_request": {
                    "url": "https://api.github.com/repos/owner/repo/pulls/90"
                },
            },
        }
        status = _post_webhook(url, cfg, "issue_comment", payload)
        assert status == 200
        mock_session.hold_for_handler.assert_not_called()
        mock_reply.assert_not_called()
        WebhookHandler._fn_launch_worker.assert_called_once()

    def test_publish_provider_snapshot_inside_hold_for_handler(
        self, tmp_path: Path
    ) -> None:
        """Snapshot published inside hold_for_handler carries the thread's actual session state.

        Both publish calls fire inside the hold_for_handler block (enter then
        exit).  The important thing is not just that publish was called — it's
        that the FidoState snapshot at each publish point contains the real
        session field values from the WorkerThread, not stale data or a
        no-op placeholder.
        """
        from contextlib import contextmanager

        from fido.appstate import FidoState, ProviderSnapshot
        from fido.atomic import create_atomic
        from fido.events import Action
        from fido.registry import WorkerRegistry

        cfg = _config(tmp_path)
        handler = WebhookHandler.__new__(WebhookHandler)
        handler.config = cfg

        reader, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )

        # Build a mock thread with known, non-default session field values so
        # we can verify the snapshot contains exactly these values.
        thread_mock = MagicMock()
        thread_mock.session_owner = "the-worker"
        thread_mock.session_alive = True
        thread_mock.session_pid = 42
        thread_mock.session_dropped_count = 0
        thread_mock.session_sent_count = 5
        thread_mock.session_received_count = 3

        reg = WorkerRegistry(MagicMock(return_value=thread_mock), updater)
        reg.start(cfg.repos["owner/repo"])
        handler.registry = reg
        handler.gh = MagicMock()
        handler.dispatchers = {"owner/repo": _FakeDispatcher()}

        # Install the spy AFTER start() so start()'s own publish isn't counted.
        # server.py calls publish_provider_snapshot (public) which delegates to
        # _publish_provider_snapshot (private); patching the private method
        # intercepts both the enter and exit publishes.
        captured_snapshots: list[ProviderSnapshot | None] = []
        original_private_publish = reg._publish_provider_snapshot  # type: ignore[method-assign]

        def spy_private_publish(repo_name: str) -> None:
            original_private_publish(repo_name)
            captured_snapshots.append(reader.get().repos[repo_name].provider)

        reg._publish_provider_snapshot = spy_private_publish  # type: ignore[method-assign]

        mock_session = MagicMock()

        @contextmanager  # type: ignore[misc]
        def noop_hold_for_handler() -> object:
            yield mock_session

        mock_session.hold_for_handler = noop_hold_for_handler

        def patched_get_session(repo_name: str) -> object:
            del repo_name
            return mock_session

        reg.get_session = patched_get_session  # type: ignore[method-assign]

        WebhookHandler._fn_reply_to_issue_comment = MagicMock(  # type: ignore[assignment]
            return_value=("ACT", [])
        )
        WebhookHandler._fn_create_task = MagicMock()  # type: ignore[assignment]
        WebhookHandler._fn_launch_worker = MagicMock()  # type: ignore[assignment]

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

        # Exactly two publishes from _process_action: one on hold enter, one on exit.
        assert len(captured_snapshots) == 2

        # Both snapshots must carry the mock thread's real session field values.
        expected = ProviderSnapshot(
            session_owner="the-worker",
            session_alive=True,
            session_pid=42,
            session_dropped_count=0,
            session_sent_count=5,
            session_received_count=3,
        )
        assert captured_snapshots[0] == expected  # published on hold enter
        assert captured_snapshots[1] == expected  # published on hold exit (finally)

    def test_publish_provider_snapshot_inside_hold_for_handler_on_exception(
        self, tmp_path: Path
    ) -> None:
        """Snapshot carries real values on both publishes even when the action raises.

        The finally block in _process_action guarantees the exit publish fires
        whether or not _process_action_inner raises.  Verify the snapshot
        actually contains the thread's session state on both captures — not just
        that publish was called twice.
        """
        from contextlib import contextmanager

        from fido.appstate import FidoState, ProviderSnapshot
        from fido.atomic import create_atomic
        from fido.events import Action
        from fido.registry import WorkerRegistry

        cfg = _config(tmp_path)
        handler = WebhookHandler.__new__(WebhookHandler)
        handler.config = cfg

        reader, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )

        # Use session_owner=None to exercise the provider-returns-None path
        # (providers filter out non-worker talkers; clearing stale worker info
        # is still correct even without a positive "owned by webhook" value).
        thread_mock = MagicMock()
        thread_mock.session_owner = None
        thread_mock.session_alive = True
        thread_mock.session_pid = 99
        thread_mock.session_dropped_count = 1
        thread_mock.session_sent_count = 2
        thread_mock.session_received_count = 4

        reg = WorkerRegistry(MagicMock(return_value=thread_mock), updater)
        reg.start(cfg.repos["owner/repo"])
        handler.registry = reg
        handler.gh = MagicMock()
        handler.dispatchers = {"owner/repo": _FakeDispatcher()}

        captured_snapshots: list[ProviderSnapshot | None] = []
        original_private_publish = reg._publish_provider_snapshot  # type: ignore[method-assign]

        def spy_private_publish(repo_name: str) -> None:
            original_private_publish(repo_name)
            captured_snapshots.append(reader.get().repos[repo_name].provider)

        reg._publish_provider_snapshot = spy_private_publish  # type: ignore[method-assign]

        mock_session = MagicMock()

        @contextmanager  # type: ignore[misc]
        def noop_hold_for_handler() -> object:
            yield mock_session

        mock_session.hold_for_handler = noop_hold_for_handler

        def patched_get_session(repo_name: str) -> object:
            del repo_name
            return mock_session

        reg.get_session = patched_get_session  # type: ignore[method-assign]

        def raise_in_handler(*args: object, **kwargs: object) -> tuple[str, list[str]]:
            del args, kwargs
            raise RuntimeError("boom")

        WebhookHandler._fn_reply_to_issue_comment = raise_in_handler  # type: ignore[assignment]
        WebhookHandler._fn_create_task = MagicMock()  # type: ignore[assignment]
        WebhookHandler._fn_launch_worker = MagicMock()  # type: ignore[assignment]

        action = Action(
            prompt="test",
            comment_body="please fix",
            thread={
                "repo": "owner/repo",
                "pr": 12,
                "comment_id": 9301,
                "url": "https://github.com/owner/repo/pull/12#issuecomment-9301",
                "author": "owner",
                "comment_type": "issues",
            },
        )

        with pytest.raises(RuntimeError, match="boom"):
            handler._process_action(action, cfg.repos["owner/repo"])

        # Two publishes must have fired even though the action raised.
        assert len(captured_snapshots) == 2

        # Both snapshots must carry the mock thread's real session field values.
        # session_owner falls back to "" when the provider returns None — the
        # ProviderSnapshot field is now non-optional (#1696, no None on
        # appstate types).
        expected = ProviderSnapshot(
            session_owner="",
            session_alive=True,
            session_pid=99,
            session_dropped_count=1,
            session_sent_count=2,
            session_received_count=4,
        )
        assert captured_snapshots[0] == expected  # published on hold enter
        assert captured_snapshots[1] == expected  # published on hold exit (finally)


class TestProcessActionInner:
    """Direct unit tests for ``_process_action_inner`` — covers the
    reply-to / review-comments / comment-body / signal-action-error
    branches that the deleted webhook-driven TestProcessAction tests
    used to cover.

    Tests construct synthetic Actions with the relevant fields
    populated and call ``_process_action_inner`` directly, without going
    through the HTTP webhook path.  This decouples the coverage of
    server.py's dispatch logic from the webhook → queue architecture.
    """

    @pytest.fixture
    def repo_cfg(self, tmp_path: Path) -> RepoConfig:
        from fido.config import RepoMembership

        return RepoConfig(
            name="owner/repo",
            work_dir=tmp_path,
            membership=RepoMembership(collaborators=frozenset({"owner"})),
        )

    @pytest.fixture
    def cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=0,
            secret=b"test",
            repos={"owner/repo": RepoConfig(name="owner/repo", work_dir=tmp_path)},
            allowed_bots=frozenset(),
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )

    def _handler(self, cfg: Config) -> WebhookHandler:
        handler = object.__new__(WebhookHandler)
        handler.config = cfg
        handler.gh = MagicMock()
        handler.registry = MagicMock()
        handler.dispatchers = {"owner/repo": _FakeDispatcher()}
        return handler

    def _activity(self) -> MagicMock:
        # The activity handle is just a "set_description" sink — easy
        # to mock without depending on the concrete class.
        return MagicMock()

    def test_action_with_reply_to_calls_reply_to_comment_and_creates_tasks(
        self, cfg: Config, repo_cfg: RepoConfig, tmp_path: Path
    ) -> None:
        """Action.reply_to set → reply_to_comment + create_task fire."""
        action = Action(
            prompt="comment",
            reply_to={
                "repo": "owner/repo",
                "pr": 1,
                "comment_id": 100,
                "url": "https://example.com",
                "author": "owner",
                "comment_type": "pulls",
            },
            comment_body="please add logging",
            is_bot=False,
        )
        mock_reply = MagicMock(return_value=("ACT", ["add logging"]))
        mock_task = MagicMock()
        WebhookHandler._fn_reply_to_comment = mock_reply
        WebhookHandler._fn_create_task = mock_task
        WebhookHandler._fn_launch_worker = MagicMock()
        handler = self._handler(cfg)
        handler._process_action_inner(action, repo_cfg, self._activity())
        mock_reply.assert_called_once()

    def test_action_with_reply_to_skips_when_promise_is_none(
        self, cfg: Config, repo_cfg: RepoConfig, tmp_path: Path
    ) -> None:
        """Action.reply_to set but already-claimed → skip the reply call."""
        # Pre-claim the comment so prepare_reply returns None.
        store = FidoStore(tmp_path)
        store.prepare_reply(
            owner="webhook", comment_type="pulls", anchor_comment_id=100
        )
        action = Action(
            prompt="comment",
            reply_to={
                "repo": "owner/repo",
                "pr": 1,
                "comment_id": 100,
                "url": "https://example.com",
                "author": "owner",
                "comment_type": "pulls",
            },
            comment_body="please add logging",
            is_bot=False,
        )
        mock_reply = MagicMock()
        WebhookHandler._fn_reply_to_comment = mock_reply
        WebhookHandler._fn_create_task = MagicMock()
        WebhookHandler._fn_launch_worker = MagicMock()
        handler = self._handler(cfg)
        handler._process_action_inner(action, repo_cfg, self._activity())
        mock_reply.assert_not_called()  # already-claimed → skipped

    def test_action_with_reply_to_failure_marks_promise_retryable(
        self, cfg: Config, repo_cfg: RepoConfig, tmp_path: Path
    ) -> None:
        """When reply_to_comment raises a recoverable error, _fail_reply marks
        the promise retryable and the narrowed except swallows the exception
        (signaling a 'confused' reaction via _signal_action_error).
        Logic bugs (KeyError, TypeError, etc.) are NOT swallowed — only
        requests.RequestException and friends are caught."""
        import requests

        action = Action(
            prompt="comment",
            reply_to={
                "repo": "owner/repo",
                "pr": 1,
                "comment_id": 200,
                "url": "https://example.com",
                "author": "owner",
                "comment_type": "pulls",
            },
            comment_body="boom",
        )
        mock_reply = MagicMock(side_effect=requests.RequestException("API down"))
        WebhookHandler._fn_reply_to_comment = mock_reply
        WebhookHandler._fn_launch_worker = MagicMock()
        handler = self._handler(cfg)
        # _process_action_inner swallows the recoverable exception (logs + signal).
        handler._process_action_inner(action, repo_cfg, self._activity())
        # _signal_action_error fired → confused reaction posted.
        handler.gh.add_reaction.assert_called_with(
            "owner/repo", "pulls", 200, "confused"
        )

    def test_action_with_reply_to_logic_bug_propagates(
        self, cfg: Config, repo_cfg: RepoConfig, tmp_path: Path
    ) -> None:
        """Logic bugs (e.g. KeyError) from reply_to_comment are NOT swallowed —
        they propagate so the watchdog sees the crash."""
        action = Action(
            prompt="comment",
            reply_to={
                "repo": "owner/repo",
                "pr": 1,
                "comment_id": 201,
                "url": "https://example.com",
                "author": "owner",
                "comment_type": "pulls",
            },
            comment_body="boom",
        )
        mock_reply = MagicMock(side_effect=KeyError("missing_key"))
        WebhookHandler._fn_reply_to_comment = mock_reply
        WebhookHandler._fn_launch_worker = MagicMock()
        handler = self._handler(cfg)
        with pytest.raises(KeyError):
            handler._process_action_inner(action, repo_cfg, self._activity())

    def test_action_with_review_comments_calls_reply_to_review(
        self, cfg: Config, repo_cfg: RepoConfig
    ) -> None:
        """Action.review_comments set → reply_to_review fires."""
        action = Action(
            prompt="review",
            review_comments=[{"id": 9, "body": "lgtm"}],
        )
        mock_reply = MagicMock()
        WebhookHandler._fn_reply_to_review = mock_reply
        WebhookHandler._fn_launch_worker = MagicMock()
        handler = self._handler(cfg)
        handler._process_action_inner(action, repo_cfg, self._activity())
        mock_reply.assert_called_once()

    def test_action_with_comment_body_only_calls_reply_to_issue_comment(
        self, cfg: Config, repo_cfg: RepoConfig, tmp_path: Path
    ) -> None:
        """Action.comment_body set without reply_to → reply_to_issue_comment fires."""
        action = Action(
            prompt="issue comment",
            thread={
                "repo": "owner/repo",
                "pr": 1,
                "comment_id": 300,
                "url": "https://example.com",
                "author": "owner",
                "comment_type": "issues",
            },
            comment_body="any thoughts?",
        )
        mock_reply = MagicMock(return_value=("ANSWER", []))
        WebhookHandler._fn_reply_to_issue_comment = mock_reply
        WebhookHandler._fn_create_task = MagicMock()
        WebhookHandler._fn_launch_worker = MagicMock()
        handler = self._handler(cfg)
        handler._process_action_inner(action, repo_cfg, self._activity())
        mock_reply.assert_called_once()

    def test_action_with_comment_body_skips_when_promise_is_none(
        self, cfg: Config, repo_cfg: RepoConfig, tmp_path: Path
    ) -> None:
        """Issue-comment path: already-claimed → skip the reply call."""
        store = FidoStore(tmp_path)
        store.prepare_reply(
            owner="webhook", comment_type="issues", anchor_comment_id=400
        )
        action = Action(
            prompt="issue comment",
            thread={
                "repo": "owner/repo",
                "pr": 1,
                "comment_id": 400,
                "url": "https://example.com",
                "author": "owner",
                "comment_type": "issues",
            },
            comment_body="dup",
        )
        mock_reply = MagicMock()
        WebhookHandler._fn_reply_to_issue_comment = mock_reply
        WebhookHandler._fn_launch_worker = MagicMock()
        handler = self._handler(cfg)
        handler._process_action_inner(action, repo_cfg, self._activity())
        mock_reply.assert_not_called()

    def test_action_with_comment_body_failure_marks_promise_retryable(
        self, cfg: Config, repo_cfg: RepoConfig, tmp_path: Path
    ) -> None:
        """Issue-comment path: recoverable error raises → _fail_reply marks
        retryable, narrowed except swallows + signals.  Logic bugs propagate."""
        import requests

        action = Action(
            prompt="issue comment",
            thread={
                "repo": "owner/repo",
                "pr": 1,
                "comment_id": 500,
                "url": "https://example.com",
                "author": "owner",
                "comment_type": "issues",
            },
            comment_body="boom",
        )
        mock_reply = MagicMock(side_effect=requests.RequestException("API down"))
        WebhookHandler._fn_reply_to_issue_comment = mock_reply
        WebhookHandler._fn_launch_worker = MagicMock()
        handler = self._handler(cfg)
        handler._process_action_inner(action, repo_cfg, self._activity())
        handler.gh.add_reaction.assert_called_with(
            "owner/repo", "issues", 500, "confused"
        )

    def test_eyes_reaction_posted_for_review_comment(
        self, cfg: Config, repo_cfg: RepoConfig
    ) -> None:
        """review-comment action → eyes posted before triage begins (#1243)."""
        action = Action(
            prompt="review comment",
            reply_to={
                "repo": "owner/repo",
                "pr": 1,
                "comment_id": 101,
                "url": "https://example.com",
                "author": "owner",
                "comment_type": "pulls",
            },
            comment_body="please fix this",
            is_bot=False,
        )
        WebhookHandler._fn_reply_to_comment = MagicMock(return_value=("ANSWER", []))
        WebhookHandler._fn_launch_worker = MagicMock()
        handler = self._handler(cfg)
        handler._process_action_inner(action, repo_cfg, self._activity())
        handler.gh.add_reaction.assert_any_call("owner/repo", "pulls", 101, "eyes")

    def test_eyes_reaction_posted_for_top_level_pr_comment(
        self, cfg: Config, repo_cfg: RepoConfig
    ) -> None:
        """top-level PR comment action → eyes posted before triage begins (#1243)."""
        action = Action(
            prompt="issue comment",
            thread={
                "repo": "owner/repo",
                "pr": 1,
                "comment_id": 301,
                "url": "https://example.com",
                "author": "owner",
                "comment_type": "issues",
            },
            comment_body="any thoughts?",
            is_bot=False,
        )
        WebhookHandler._fn_reply_to_issue_comment = MagicMock(
            return_value=("ANSWER", [])
        )
        WebhookHandler._fn_launch_worker = MagicMock()
        handler = self._handler(cfg)
        handler._process_action_inner(action, repo_cfg, self._activity())
        handler.gh.add_reaction.assert_any_call("owner/repo", "issues", 301, "eyes")

    def test_eyes_reaction_skipped_when_other_open_comment_in_queue(
        self, cfg: Config, repo_cfg: RepoConfig
    ) -> None:
        """Batch arrival: another pending comment in queue → eager eyes skipped.

        With #1662, when a second comment arrives for a repo that already
        has an open queued or in-progress comment, the dispatcher does not
        post eager eyes — the worker posts eyes when claiming each comment
        in pickup order, so at most one comment per repo carries eyes at a
        time.
        """
        from fido.store import FidoStore

        # Pre-populate the queue with another pending comment for the
        # same repo — simulates a sibling in a multi-comment review.
        FidoStore(repo_cfg.work_dir).enqueue_pr_comment(
            delivery_id="delivery-sibling",
            repo="owner/repo",
            pr_number=1,
            comment_type="pulls",
            comment_id=400,
            author="owner",
            is_bot=False,
            body="sibling comment",
            github_created_at="2026-04-30T10:00:00Z",
        )
        action = Action(
            prompt="batch review comment",
            reply_to={
                "repo": "owner/repo",
                "pr": 1,
                "comment_id": 401,
                "url": "https://example.com",
                "author": "owner",
                "comment_type": "pulls",
            },
            comment_body="please fix this",
            is_bot=False,
        )
        WebhookHandler._fn_reply_to_comment = MagicMock(return_value=("ANSWER", []))
        WebhookHandler._fn_launch_worker = MagicMock()
        handler = self._handler(cfg)
        handler._process_action_inner(action, repo_cfg, self._activity())
        for call in handler.gh.add_reaction.call_args_list:
            assert call.args[3] != "eyes", (
                f"eyes was posted despite sibling open comment: {call}"
            )

    def test_eyes_reaction_skipped_for_bot_action(
        self, cfg: Config, repo_cfg: RepoConfig
    ) -> None:
        """is_bot=True → eyes reaction must not be posted."""
        action = Action(
            prompt="bot comment",
            reply_to={
                "repo": "owner/repo",
                "pr": 1,
                "comment_id": 201,
                "url": "https://example.com",
                "author": "somebot[bot]",
                "comment_type": "pulls",
            },
            comment_body="automated feedback",
            is_bot=True,
        )
        WebhookHandler._fn_reply_to_comment = MagicMock(return_value=("ANSWER", []))
        WebhookHandler._fn_launch_worker = MagicMock()
        handler = self._handler(cfg)
        handler._process_action_inner(action, repo_cfg, self._activity())
        for call in handler.gh.add_reaction.call_args_list:
            assert call.args[3] != "eyes", f"eyes was posted for bot action: {call}"

    def test_eyes_reaction_skipped_for_non_comment_action(
        self, cfg: Config, repo_cfg: RepoConfig
    ) -> None:
        """Non-comment webhook actions (no reply_to, no comment_body) → no reaction."""
        action = Action(prompt="push event")
        WebhookHandler._fn_launch_worker = MagicMock()
        handler = self._handler(cfg)
        handler._process_action_inner(action, repo_cfg, self._activity())
        handler.gh.add_reaction.assert_not_called()

    def test_eyes_reaction_failure_does_not_abort_handler(
        self, cfg: Config, repo_cfg: RepoConfig
    ) -> None:
        """add_reaction failure for eyes must not abort the webhook handler."""
        import requests

        action = Action(
            prompt="review comment",
            reply_to={
                "repo": "owner/repo",
                "pr": 1,
                "comment_id": 102,
                "url": "https://example.com",
                "author": "owner",
                "comment_type": "pulls",
            },
            comment_body="please fix",
            is_bot=False,
        )
        WebhookHandler._fn_reply_to_comment = MagicMock(return_value=("ANSWER", []))
        WebhookHandler._fn_launch_worker = MagicMock()
        handler = self._handler(cfg)
        handler.gh.add_reaction.side_effect = requests.RequestException("network down")
        # Should not raise — eyes failure is best-effort.
        handler._process_action_inner(action, repo_cfg, self._activity())

    def test_eyes_removed_on_review_comment_dedup_skip(
        self, cfg: Config, repo_cfg: RepoConfig
    ) -> None:
        """When _prepare_reply returns None (dedup skip), the eyes reaction is
        removed so it doesn't linger on a comment that won't receive a reply."""
        action = Action(
            prompt="review comment",
            reply_to={
                "repo": "owner/repo",
                "pr": 1,
                "comment_id": 111,
                "url": "https://example.com",
                "author": "owner",
                "comment_type": "pulls",
            },
            comment_body="please fix",
            is_bot=False,
        )
        WebhookHandler._fn_launch_worker = MagicMock()
        handler = self._handler(cfg)
        # Simulate dedup skip: _prepare_reply returns None
        with patch.object(handler, "_prepare_reply", return_value=None):
            with patch(
                "fido.server.SynthesisExecutor.remove_eyes_reaction"
            ) as mock_remove:
                handler._process_action_inner(action, repo_cfg, self._activity())
        mock_remove.assert_called_once()

    def test_eyes_removed_on_pr_comment_dedup_skip(
        self, cfg: Config, repo_cfg: RepoConfig
    ) -> None:
        """When _prepare_reply returns None for a top-level PR comment (dedup skip),
        the eyes reaction is removed."""
        action = Action(
            prompt="pr comment",
            comment_body="please fix",
            thread={
                "repo": "owner/repo",
                "pr": 1,
                "comment_id": 222,
                "url": "https://example.com",
                "author": "owner",
                "comment_type": "issues",
            },
            is_bot=False,
        )
        WebhookHandler._fn_launch_worker = MagicMock()
        handler = self._handler(cfg)
        with patch.object(handler, "_prepare_reply", return_value=None):
            with patch(
                "fido.server.SynthesisExecutor.remove_eyes_reaction"
            ) as mock_remove:
                handler._process_action_inner(action, repo_cfg, self._activity())
        mock_remove.assert_called_once()

    def test_eyes_removed_on_recoverable_exception(
        self, cfg: Config, repo_cfg: RepoConfig
    ) -> None:
        """When a recoverable exception is raised during action processing, the
        eyes reaction is removed before signalling the confused reaction."""
        import requests

        action = Action(
            prompt="review comment",
            reply_to={
                "repo": "owner/repo",
                "pr": 1,
                "comment_id": 333,
                "url": "https://example.com",
                "author": "owner",
                "comment_type": "pulls",
            },
            comment_body="please fix",
            is_bot=False,
        )
        WebhookHandler._fn_reply_to_comment = MagicMock(
            side_effect=requests.RequestException("network down")
        )
        WebhookHandler._fn_launch_worker = MagicMock()
        handler = self._handler(cfg)
        remove_order: list[str] = []
        with patch(
            "fido.server.SynthesisExecutor.remove_eyes_reaction",
            side_effect=lambda _t: remove_order.append("remove_eyes"),
        ):
            with patch.object(
                handler,
                "_signal_action_error",
                side_effect=lambda _a: remove_order.append("signal_error"),
            ):
                # Recoverable exception — should not propagate
                handler._process_action_inner(action, repo_cfg, self._activity())
        # Eyes must be removed before the confused reaction is signalled
        assert remove_order == ["remove_eyes", "signal_error"]

    def test_eyes_reaction_posted_for_queued_comment_action(
        self, cfg: Config, repo_cfg: RepoConfig
    ) -> None:
        """Queued comment actions (thread set, comment_body=None, preempts_worker=True)
        get the immediate eyes reaction — the production shape from the dispatcher."""
        action = Action(
            prompt="Queued review comment on PR #1 by owner (human/owner)",
            preempts_worker=True,
            is_bot=False,
            thread={
                "repo": "owner/repo",
                "pr": 1,
                "comment_id": 444,
                "url": "https://example.com",
                "author": "owner",
                "comment_type": "pulls",
            },
            # comment_body intentionally None — matches _queued_pr_comment_action shape
        )
        WebhookHandler._fn_launch_worker = MagicMock()
        handler = self._handler(cfg)
        handler._process_action_inner(action, repo_cfg, self._activity())
        handler.gh.add_reaction.assert_any_call("owner/repo", "pulls", 444, "eyes")

    def test_remove_eyes_best_effort_noop_with_missing_fields(
        self, cfg: Config
    ) -> None:
        """_remove_eyes_best_effort is a no-op when repo or comment_id is absent."""
        handler = self._handler(cfg)
        # Missing repo — should not call remove_eyes_reaction at all.
        with patch("fido.server.SynthesisExecutor.remove_eyes_reaction") as mock_remove:
            handler._remove_eyes_best_effort(handler.gh, {"comment_id": 99})
        mock_remove.assert_not_called()
        # Missing comment_id — same.
        with patch("fido.server.SynthesisExecutor.remove_eyes_reaction") as mock_remove:
            handler._remove_eyes_best_effort(handler.gh, {"repo": "owner/repo"})
        mock_remove.assert_not_called()

    def test_describe_action_handles_each_action_shape(self, cfg: Config) -> None:
        """Cover all branches of _describe_action."""
        handler = self._handler(cfg)
        thread = {
            "repo": "owner/repo",
            "pr": 1,
            "comment_id": 1,
            "url": "",
            "author": "owner",
            "comment_type": "pulls",
        }
        assert (
            handler._describe_action(Action(prompt="x", reply_to=thread))
            == "handling review comment"
        )
        assert (
            handler._describe_action(Action(prompt="x", review_comments=[{}]))
            == "handling review thread"
        )
        assert (
            handler._describe_action(Action(prompt="x", comment_body="hi"))
            == "handling PR comment"
        )
        assert (
            handler._describe_action(
                Action(prompt="x", thread=thread, preempts_worker=True)
            )
            == "ingesting PR comment"
        )
        assert handler._describe_action(Action(prompt="x")) == "handling webhook action"

    def test_signal_action_error_posts_confused_reaction(self, cfg: Config) -> None:
        """_signal_action_error posts a 'confused' reaction on the
        triggering comment when one is present."""
        handler = self._handler(cfg)
        action = Action(
            prompt="x",
            reply_to={
                "repo": "owner/repo",
                "pr": 1,
                "comment_id": 99,
                "url": "",
                "author": "owner",
                "comment_type": "pulls",
            },
        )
        handler._signal_action_error(action)
        handler.gh.add_reaction.assert_called_once_with(
            "owner/repo", "pulls", 99, "confused"
        )

    def test_signal_action_error_swallows_reaction_failure(self, cfg: Config) -> None:
        """A failed add_reaction during _signal_action_error must not
        propagate — we don't want the error-signaling path to mask the
        original error or crash the handler."""
        handler = self._handler(cfg)
        handler.gh.add_reaction.side_effect = RuntimeError("network down")
        action = Action(
            prompt="x",
            thread={
                "repo": "owner/repo",
                "pr": 1,
                "comment_id": 88,
                "url": "",
                "author": "owner",
                "comment_type": "issues",
            },
        )
        handler._signal_action_error(action)  # should not raise

    def test_signal_action_error_no_op_when_no_thread(self, cfg: Config) -> None:
        """_signal_action_error is a no-op when the action has no thread or
        reply_to — non-comment events have no comment to react on."""
        handler = self._handler(cfg)
        handler._signal_action_error(Action(prompt="push event"))
        handler.gh.add_reaction.assert_not_called()


class TestSynchronousPreemption:
    """Verify that preemption fires on the HTTP handler thread, before the
    background thread spawns (#955).

    ``_do_post_inner`` calls ``session.preempt_worker()`` synchronously, then
    delegates to ``_fn_spawn_bg``.  The ordering guarantee is what fixes the
    race: the cancel signal reaches the provider *before* the handler thread
    can be de-scheduled and the worker turn can complete.
    """

    def _issue_comment_payload(self, comment_id: int = 900) -> dict:
        """An issue_comment on a PR produces a durable-demand wakeup action."""
        return {
            **_payload(),
            "action": "created",
            "comment": {
                "id": comment_id,
                "body": "please fix this",
                "user": {"login": "owner"},
                "html_url": f"https://github.com/owner/repo/pull/80#issuecomment-{comment_id}",
            },
            "issue": {
                "number": 80,
                "title": "my pr",
                "body": "",
                "pull_request": {
                    "url": "https://api.github.com/repos/owner/repo/pulls/80"
                },
            },
        }

    def _record_durable_demand_order(self, call_order: list[str]) -> None:
        WebhookHandler.registry.note_durable_demand.side_effect = lambda repo: (
            call_order.append(f"durable:{repo}")
        )

    def _record_interrupt_order(self, call_order: list[str]) -> None:
        WebhookHandler.registry.note_provider_interrupt_requested.side_effect = (
            lambda repo: call_order.append(f"interrupt:{repo}")
        )

    def _record_background_spawn_order(self, call_order: list[str]) -> None:
        original_spawn = _capturing_spawn_bg

        def tracking_spawn(fn: Callable[..., Any], args: tuple[Any, ...]) -> None:
            call_order.append("spawn")
            original_spawn(fn, args)

        WebhookHandler._fn_spawn_bg = staticmethod(tracking_spawn)  # type: ignore[assignment]

    def test_preempt_fires_before_background_spawn(self, server: tuple) -> None:
        """``session.preempt_worker()`` must be called before ``_fn_spawn_bg``
        for a webhook action that reports durable demand."""
        url, cfg = server

        call_order: list[str] = []

        mock_session = MagicMock()
        self._record_durable_demand_order(call_order)
        self._record_interrupt_order(call_order)
        mock_session.preempt_worker.side_effect = lambda: call_order.append("preempt")
        WebhookHandler.registry.get_session.return_value = mock_session

        self._record_background_spawn_order(call_order)
        WebhookHandler._fn_reply_to_issue_comment = MagicMock(
            return_value=("ANSWER", [])
        )
        WebhookHandler._fn_create_task = MagicMock()
        WebhookHandler._fn_launch_worker = MagicMock()

        status = _post_webhook(url, cfg, "issue_comment", self._issue_comment_payload())
        assert status == 200
        assert call_order == [
            "durable:owner/repo",
            "interrupt:owner/repo",
            "preempt",
            "spawn",
        ]
        WebhookHandler.registry.note_durable_demand.assert_called_once_with(
            "owner/repo"
        )
        WebhookHandler.registry.note_provider_interrupt_requested.assert_called_once_with(
            "owner/repo"
        )

    def test_preempt_failure_keeps_enqueued_comment_and_spawns(
        self, server: tuple, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A provider interrupt failure must not discard durable webhook demand."""
        url, cfg = server

        call_order: list[str] = []

        mock_session = MagicMock()
        self._record_durable_demand_order(call_order)
        self._record_interrupt_order(call_order)
        WebhookHandler.registry.get_session.return_value = mock_session

        def failing_preempt() -> None:
            call_order.append("preempt")
            raise RuntimeError("interrupt failed")

        mock_session.preempt_worker.side_effect = failing_preempt
        self._record_background_spawn_order(call_order)
        WebhookHandler._fn_reply_to_issue_comment = MagicMock(
            return_value=("ANSWER", [])
        )
        WebhookHandler._fn_create_task = MagicMock()
        WebhookHandler._fn_launch_worker = MagicMock()

        with caplog.at_level(logging.ERROR, logger="fido.server"):
            status = _post_webhook(
                url, cfg, "issue_comment", self._issue_comment_payload(901)
            )

        assert status == 200
        assert call_order == [
            "durable:owner/repo",
            "interrupt:owner/repo",
            "preempt",
            "spawn",
        ]
        WebhookHandler.registry.note_durable_demand.assert_called_once_with(
            "owner/repo"
        )
        WebhookHandler.registry.note_provider_interrupt_requested.assert_called_once_with(
            "owner/repo"
        )
        WebhookHandler.registry.enter_untriaged.assert_called_with("owner/repo")
        WebhookHandler.registry.recover_provider.assert_not_called()
        queued = FidoStore(cfg.repos["owner/repo"].work_dir).pending_pr_comments(
            repo="owner/repo"
        )
        assert [entry.comment_id for entry in queued] == [901]
        assert (
            "provider preempt failed for owner/repo after durable webhook enqueue"
            in caplog.text
        )
        assert "provider preempt wedged" not in caplog.text
        assert "provider recovery requested" not in caplog.text

    def test_preempt_wedge_recovers_provider_and_spawns(
        self, server: tuple, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Recoverable provider wedges trigger provider recovery after enqueue."""
        url, cfg = server

        call_order: list[str] = []

        mock_session = MagicMock()
        self._record_durable_demand_order(call_order)
        self._record_interrupt_order(call_order)
        WebhookHandler.registry.get_session.return_value = mock_session
        WebhookHandler.registry.recover_provider.side_effect = lambda repo: (
            call_order.append(f"recover:{repo}") or True
        )

        def wedged_preempt() -> None:
            call_order.append("preempt")
            raise provider.ProviderInterruptTimeout("interrupt timed out")

        mock_session.preempt_worker.side_effect = wedged_preempt
        self._record_background_spawn_order(call_order)
        WebhookHandler._fn_reply_to_issue_comment = MagicMock(
            return_value=("ANSWER", [])
        )
        WebhookHandler._fn_create_task = MagicMock()
        WebhookHandler._fn_launch_worker = MagicMock()

        with caplog.at_level(logging.WARNING, logger="fido.server"):
            status = _post_webhook(
                url, cfg, "issue_comment", self._issue_comment_payload(902)
            )

        assert status == 200
        assert call_order == [
            "durable:owner/repo",
            "interrupt:owner/repo",
            "preempt",
            "recover:owner/repo",
            "spawn",
        ]
        WebhookHandler.registry.note_durable_demand.assert_called_once_with(
            "owner/repo"
        )
        WebhookHandler.registry.note_provider_interrupt_requested.assert_called_once_with(
            "owner/repo"
        )
        WebhookHandler.registry.recover_provider.assert_called_once_with("owner/repo")
        WebhookHandler.registry.enter_untriaged.assert_called_with("owner/repo")
        queued = FidoStore(cfg.repos["owner/repo"].work_dir).pending_pr_comments(
            repo="owner/repo"
        )
        assert [entry.comment_id for entry in queued] == [902]
        assert (
            "provider preempt wedged for owner/repo after durable webhook enqueue "
            "— recovering provider"
        ) in caplog.text
        assert (
            "provider recovery requested for owner/repo after preempt wedge"
            in caplog.text
        )
        assert "provider preempt failed" not in caplog.text

    def test_preempt_wedge_logs_unavailable_recovery(
        self, server: tuple, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Recoverable wedges log loudly when no provider recovery is available."""
        url, cfg = server

        call_order: list[str] = []

        mock_session = MagicMock()
        self._record_durable_demand_order(call_order)
        self._record_interrupt_order(call_order)
        WebhookHandler.registry.get_session.return_value = mock_session
        WebhookHandler.registry.recover_provider.side_effect = lambda repo: (
            call_order.append(f"recover:{repo}") or False
        )

        def wedged_preempt() -> None:
            call_order.append("preempt")
            raise provider.ProviderInterruptTimeout("interrupt timed out")

        mock_session.preempt_worker.side_effect = wedged_preempt
        self._record_background_spawn_order(call_order)
        WebhookHandler._fn_reply_to_issue_comment = MagicMock(
            return_value=("ANSWER", [])
        )
        WebhookHandler._fn_create_task = MagicMock()
        WebhookHandler._fn_launch_worker = MagicMock()

        with caplog.at_level(logging.WARNING, logger="fido.server"):
            status = _post_webhook(
                url, cfg, "issue_comment", self._issue_comment_payload(903)
            )

        assert status == 200
        assert call_order == [
            "durable:owner/repo",
            "interrupt:owner/repo",
            "preempt",
            "recover:owner/repo",
            "spawn",
        ]
        WebhookHandler.registry.recover_provider.assert_called_once_with("owner/repo")
        assert (
            "provider recovery unavailable for owner/repo after preempt wedge"
            in caplog.text
        )
        assert "provider recovery requested" not in caplog.text

    def test_no_preempt_for_non_model_action(self, server: tuple) -> None:
        """A PR-merge webhook does not use the model — ``preempt_worker`` must
        NOT be called."""
        url, cfg = server

        mock_session = MagicMock()
        WebhookHandler.registry.get_session.return_value = mock_session
        WebhookHandler._fn_launch_worker = MagicMock()
        WebhookHandler._fn_create_task = MagicMock()

        payload = {
            **_payload(),
            "action": "closed",
            "pull_request": {"number": 81, "merged": True},
        }
        status = _post_webhook(url, cfg, "pull_request", payload)
        assert status == 200
        WebhookHandler.registry.note_durable_demand.assert_not_called()
        WebhookHandler.registry.note_provider_interrupt_requested.assert_not_called()
        mock_session.preempt_worker.assert_not_called()

    def test_no_preempt_when_session_is_none(self, server: tuple) -> None:
        """If ``registry.get_session()`` returns None, the handler must not
        crash — there is simply no session to preempt."""
        url, cfg = server

        WebhookHandler.registry.get_session.return_value = None
        WebhookHandler._fn_reply_to_issue_comment = MagicMock(
            return_value=("ANSWER", [])
        )
        WebhookHandler._fn_create_task = MagicMock()
        WebhookHandler._fn_launch_worker = MagicMock()

        status = _post_webhook(
            url, cfg, "issue_comment", self._issue_comment_payload(901)
        )
        assert status == 200
        WebhookHandler.registry.note_durable_demand.assert_called_once_with(
            "owner/repo"
        )
        WebhookHandler.registry.note_provider_interrupt_requested.assert_called_once_with(
            "owner/repo"
        )


class TestUntriagedInboxWiring:
    """Verify that _do_post_inner / _process_action correctly enter/exit the
    per-repo untriaged inbox for preempting webhook actions (#1067)."""

    def _issue_comment_payload(self, comment_id: int = 950) -> dict:
        """An issue_comment on a PR produces a durable-demand wakeup action."""
        return {
            **_payload(),
            "action": "created",
            "comment": {
                "id": comment_id,
                "body": "please fix this",
                "user": {"login": "owner"},
                "html_url": f"https://github.com/owner/repo/pull/80#issuecomment-{comment_id}",
            },
            "issue": {
                "number": 80,
                "title": "my pr",
                "body": "",
                "pull_request": {
                    "url": "https://api.github.com/repos/owner/repo/pulls/80"
                },
            },
        }

    def _check_run_failure_payload(self) -> dict:
        return {
            **_payload(),
            "action": "completed",
            "check_run": {
                "name": "ci",
                "conclusion": "failure",
                "pull_requests": [{"number": 80}],
            },
        }

    def test_enter_untriaged_called_for_comment_demand(self, server: tuple) -> None:
        """enter_untriaged must be called synchronously for durable demand."""
        url, cfg = server
        WebhookHandler._fn_reply_to_issue_comment = MagicMock(
            return_value=("ANSWER", [])
        )
        WebhookHandler._fn_create_task = MagicMock()
        WebhookHandler._fn_launch_worker = MagicMock()

        status = _post_webhook(url, cfg, "issue_comment", self._issue_comment_payload())
        assert status == 200
        WebhookHandler.registry.enter_untriaged.assert_called_with("owner/repo")

    def test_enter_untriaged_called_for_ci_failure(self, server: tuple) -> None:
        """CI failures do not use the model, but they still preempt workers."""
        url, cfg = server
        WebhookHandler._fn_launch_worker = MagicMock()

        status = _post_webhook(url, cfg, "check_run", self._check_run_failure_payload())

        assert status == 200
        WebhookHandler.registry.enter_untriaged.assert_called_with("owner/repo")
        WebhookHandler.registry.exit_untriaged.assert_called_with("owner/repo")

    def test_enter_untriaged_not_called_for_non_model_action(
        self, server: tuple
    ) -> None:
        """A PR-merge webhook does not use the model — enter_untriaged must NOT
        be called."""
        url, cfg = server
        WebhookHandler._fn_launch_worker = MagicMock()

        payload = {
            **_payload(),
            "action": "closed",
            "pull_request": {"number": 81, "merged": True},
        }
        status = _post_webhook(url, cfg, "pull_request", payload)
        assert status == 200
        WebhookHandler.registry.enter_untriaged.assert_not_called()

    def test_exit_untriaged_called_on_handler_success(self, server: tuple) -> None:
        """exit_untriaged must be called in the _process_action finally block
        when the handler completes successfully."""
        url, cfg = server
        WebhookHandler._fn_reply_to_issue_comment = MagicMock(
            return_value=("ANSWER", [])
        )
        WebhookHandler._fn_create_task = MagicMock()
        WebhookHandler._fn_launch_worker = MagicMock()

        status = _post_webhook(
            url, cfg, "issue_comment", self._issue_comment_payload(951)
        )
        assert status == 200
        WebhookHandler.registry.exit_untriaged.assert_called_with("owner/repo")

    def test_exit_untriaged_called_on_handler_exception(self, server: tuple) -> None:
        """exit_untriaged must still be called even when _process_action_inner
        raises — the finally block must fire on all non-os._exit paths."""
        url, cfg = server
        WebhookHandler._fn_reply_to_issue_comment = MagicMock(
            side_effect=RuntimeError("boom")
        )
        WebhookHandler._fn_create_task = MagicMock()
        WebhookHandler._fn_launch_worker = MagicMock()

        status = _post_webhook(
            url, cfg, "issue_comment", self._issue_comment_payload(952)
        )
        assert status == 200
        WebhookHandler.registry.exit_untriaged.assert_called_with("owner/repo")

    def test_exit_untriaged_not_called_for_non_model_action(
        self, server: tuple
    ) -> None:
        """exit_untriaged must NOT be called when the action does not use the
        model — there was no matching enter."""
        url, cfg = server
        WebhookHandler._fn_launch_worker = MagicMock()

        payload = {
            **_payload(),
            "action": "closed",
            "pull_request": {"number": 82, "merged": True},
        }
        status = _post_webhook(url, cfg, "pull_request", payload)
        assert status == 200
        WebhookHandler.registry.exit_untriaged.assert_not_called()

    def test_enter_fires_before_background_spawn(self, server: tuple) -> None:
        """enter_untriaged must be called before _fn_spawn_bg so the inbox is
        non-empty before the background thread runs."""
        url, cfg = server

        call_order: list[str] = []

        mock_session = MagicMock()
        WebhookHandler.registry.get_session.return_value = mock_session
        WebhookHandler.registry.enter_untriaged.side_effect = lambda _: (
            call_order.append("enter")
        )

        original_spawn = _capturing_spawn_bg

        def tracking_spawn(fn: Callable[..., Any], args: tuple[Any, ...]) -> None:
            call_order.append("spawn")
            original_spawn(fn, args)

        WebhookHandler._fn_spawn_bg = staticmethod(tracking_spawn)  # type: ignore[assignment]
        WebhookHandler._fn_reply_to_issue_comment = MagicMock(
            return_value=("ANSWER", [])
        )
        WebhookHandler._fn_create_task = MagicMock()
        WebhookHandler._fn_launch_worker = MagicMock()

        status = _post_webhook(
            url, cfg, "issue_comment", self._issue_comment_payload(953)
        )
        assert status == 200
        assert call_order == ["enter", "spawn"]


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
        """A single transient GitHub API error must not prevent fido from starting."""
        import requests

        from fido.server import bootstrap_issue_caches

        repos = {
            "a/r1": RepoConfig(name="a/r1", work_dir=tmp_path),
            "b/r2": RepoConfig(name="b/r2", work_dir=tmp_path),
        }
        mock_gh = MagicMock()
        mock_gh.find_all_open_issues.side_effect = [
            requests.RequestException("API down"),
            [],
        ]
        mock_cache_r2 = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get_issue_cache.side_effect = lambda name: (
            MagicMock() if name == "a/r1" else mock_cache_r2
        )

        # Must not raise despite the first repo failing.
        bootstrap_issue_caches(repos, mock_gh, mock_registry)

        # Second repo should still be bootstrapped.
        mock_cache_r2.load_inventory.assert_called_once()

    def test_logic_bug_in_bootstrap_propagates(self, tmp_path: Path) -> None:
        """Non-transient errors (logic bugs, auth failures) must propagate
        loudly so misconfiguration is caught at startup, not silently deferred."""
        from fido.server import bootstrap_issue_caches

        repos = {"a/r1": RepoConfig(name="a/r1", work_dir=tmp_path)}
        mock_gh = MagicMock()
        mock_gh.find_all_open_issues.side_effect = RuntimeError("not logged in")
        mock_registry = MagicMock()

        with pytest.raises(RuntimeError, match="not logged in"):
            bootstrap_issue_caches(repos, mock_gh, mock_registry)

    def test_wakes_worker_after_successful_load(self, tmp_path: Path) -> None:
        """Worker is woken after a successful cache load so it rescans immediately (#995)."""
        from fido.server import bootstrap_issue_caches

        repos = self._make_repos(tmp_path)
        mock_gh = MagicMock()
        mock_gh.find_all_open_issues.return_value = []
        mock_registry = MagicMock()
        mock_registry.get_issue_cache.return_value = MagicMock()

        bootstrap_issue_caches(repos, mock_gh, mock_registry)

        mock_registry.wake.assert_called_once_with("owner/repo")

    def test_wakes_each_repo_worker_on_success(self, tmp_path: Path) -> None:
        """Each successfully-loaded repo wakes its own worker thread (#995)."""
        from fido.server import bootstrap_issue_caches

        repos = {
            "a/r1": RepoConfig(name="a/r1", work_dir=tmp_path),
            "b/r2": RepoConfig(name="b/r2", work_dir=tmp_path),
        }
        mock_gh = MagicMock()
        mock_gh.find_all_open_issues.return_value = []
        mock_registry = MagicMock()

        bootstrap_issue_caches(repos, mock_gh, mock_registry)

        assert mock_registry.wake.call_count == 2
        mock_registry.wake.assert_any_call("a/r1")
        mock_registry.wake.assert_any_call("b/r2")

    def test_does_not_wake_worker_on_failed_load(self, tmp_path: Path) -> None:
        """A failed bootstrap must not wake the worker — the cache is cold (#995)."""
        import requests

        from fido.server import bootstrap_issue_caches

        repos = {
            "a/r1": RepoConfig(name="a/r1", work_dir=tmp_path),
            "b/r2": RepoConfig(name="b/r2", work_dir=tmp_path),
        }
        mock_gh = MagicMock()
        mock_gh.find_all_open_issues.side_effect = [
            requests.RequestException("API down"),
            [],
        ]
        mock_registry = MagicMock()
        mock_registry.get_issue_cache.return_value = MagicMock()

        bootstrap_issue_caches(repos, mock_gh, mock_registry)

        # Only b/r2 succeeded — only that worker should be woken.
        mock_registry.wake.assert_called_once_with("b/r2")


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
            _preflight_repo_identity=MagicMock(),
            _preflight_tools=MagicMock(),
            _preflight_sub_dir=MagicMock(),
            _preflight_gh_auth=MagicMock(),
            _GitHub=MagicMock,
            _ProviderPressureMonitor=MagicMock(),
            _RateLimitMonitor=MagicMock(),
        )

        mock_server.serve_forever.assert_called_once()
        mock_server.server_close.assert_called_once()

    def test_default_server_does_not_block_behind_slow_client(self) -> None:
        srv = FidoHTTPServer(("127.0.0.1", 0), WebhookHandler)
        srv.request_timeout_seconds = 0.2
        port = srv.server_address[1]
        thread = threading.Thread(
            target=srv.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()
        slow = socket.create_connection(("127.0.0.1", port), timeout=1)
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/", timeout=1
            ) as response:
                assert response.read() == b"fido is running"
        finally:
            slow.close()
            srv.shutdown()
            srv.server_close()
            thread.join(timeout=1)

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
            _preflight_repo_identity=MagicMock(),
            _preflight_tools=MagicMock(),
            _preflight_sub_dir=MagicMock(),
            _preflight_gh_auth=MagicMock(),
            _GitHub=MagicMock,
            _signal=MagicMock(),
            _kill_active_children=mock_kill,
            _ProviderPressureMonitor=MagicMock(),
            _RateLimitMonitor=MagicMock(),
        )
        mock_kill.assert_called_once()

    def test_run_installs_sigterm_and_sigint_handlers(self, tmp_path: Path) -> None:
        import signal as _sig

        from fido.server import run

        fake_cfg = self._fake_cfg(tmp_path)
        mock_server = MagicMock()
        mock_server.serve_forever.side_effect = KeyboardInterrupt
        installed: dict[int, object] = {}

        def fake_signal(signum: int, handler: object) -> None:
            installed[signum] = handler

        run(
            _from_args=lambda: fake_cfg,
            _HTTPServer=lambda *a, **kw: mock_server,
            _make_registry=MagicMock(),
            _path_home=lambda: tmp_path,
            _basic_config=MagicMock(),
            _populate_memberships=MagicMock(),
            _preflight_repo_identity=MagicMock(),
            _preflight_tools=MagicMock(),
            _preflight_sub_dir=MagicMock(),
            _preflight_gh_auth=MagicMock(),
            _GitHub=MagicMock,
            _signal=fake_signal,
            _kill_active_children=MagicMock(),
            _ProviderPressureMonitor=MagicMock(),
            _RateLimitMonitor=MagicMock(),
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

        def fake_signal(signum: int, handler: object) -> None:
            captured[signum] = handler

        run(
            _from_args=lambda: fake_cfg,
            _HTTPServer=lambda *a, **kw: mock_server,
            _make_registry=MagicMock(),
            _path_home=lambda: tmp_path,
            _basic_config=MagicMock(),
            _populate_memberships=MagicMock(),
            _preflight_repo_identity=MagicMock(),
            _preflight_tools=MagicMock(),
            _preflight_sub_dir=MagicMock(),
            _preflight_gh_auth=MagicMock(),
            _GitHub=MagicMock,
            _signal=fake_signal,
            _kill_active_children=mock_kill,
            _ProviderPressureMonitor=MagicMock(),
            _RateLimitMonitor=MagicMock(),
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

        def fake_basic_config(**kwargs: object) -> None:
            captured_kwargs.append(kwargs)

        run(
            _from_args=lambda: fake_cfg,
            _HTTPServer=lambda *a, **kw: mock_server,
            _make_registry=MagicMock(),
            _path_home=lambda: tmp_path,
            _basic_config=fake_basic_config,
            _populate_memberships=MagicMock(),
            _preflight_repo_identity=MagicMock(),
            _preflight_tools=MagicMock(),
            _preflight_sub_dir=MagicMock(),
            _preflight_gh_auth=MagicMock(),
            _GitHub=MagicMock,
            _ProviderPressureMonitor=MagicMock(),
            _RateLimitMonitor=MagicMock(),
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

        def fake_basic_config(**kwargs: object) -> None:
            captured_handlers.extend(kwargs.get("handlers", []))

        run(
            _from_args=lambda: fake_cfg,
            _HTTPServer=lambda *a, **kw: mock_server,
            _make_registry=MagicMock(),
            _path_home=lambda: tmp_path,
            _basic_config=fake_basic_config,
            _populate_memberships=MagicMock(),
            _preflight_repo_identity=MagicMock(),
            _preflight_tools=MagicMock(),
            _preflight_sub_dir=MagicMock(),
            _preflight_gh_auth=MagicMock(),
            _GitHub=MagicMock,
            _ProviderPressureMonitor=MagicMock(),
            _RateLimitMonitor=MagicMock(),
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
            _preflight_repo_identity=MagicMock(),
            _preflight_tools=MagicMock(),
            _preflight_sub_dir=MagicMock(),
            _preflight_gh_auth=MagicMock(),
            _GitHub=MagicMock,
            _ProviderPressureMonitor=MagicMock(),
            _RateLimitMonitor=MagicMock(),
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

        def fake_basic_config(**kwargs: object) -> None:
            captured_handlers.extend(kwargs.get("handlers", []))

        run(
            _from_args=lambda: fake_cfg,
            _HTTPServer=lambda *a, **kw: mock_server,
            _make_registry=MagicMock(),
            _path_home=lambda: tmp_path,
            _basic_config=fake_basic_config,
            _populate_memberships=MagicMock(),
            _preflight_repo_identity=MagicMock(),
            _preflight_tools=MagicMock(),
            _preflight_sub_dir=MagicMock(),
            _preflight_gh_auth=MagicMock(),
            _GitHub=MagicMock,
            _stderr=fake_stderr,
            _ProviderPressureMonitor=MagicMock(),
            _RateLimitMonitor=MagicMock(),
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

        def fake_basic_config(**kwargs: object) -> None:
            captured_handlers.extend(kwargs.get("handlers", []))

        run(
            _from_args=lambda: fake_cfg,
            _HTTPServer=lambda *a, **kw: mock_server,
            _make_registry=MagicMock(),
            _path_home=lambda: tmp_path,
            _basic_config=fake_basic_config,
            _populate_memberships=MagicMock(),
            _preflight_repo_identity=MagicMock(),
            _preflight_tools=MagicMock(),
            _preflight_sub_dir=MagicMock(),
            _preflight_gh_auth=MagicMock(),
            _GitHub=MagicMock,
            _ProviderPressureMonitor=MagicMock(),
            _RateLimitMonitor=MagicMock(),
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
            _preflight_repo_identity=MagicMock(),
            _preflight_tools=MagicMock(),
            _preflight_sub_dir=MagicMock(),
            _preflight_gh_auth=MagicMock(),
            _GitHub=MagicMock,
            _Watchdog=mock_watchdog_cls,
            _ReconcileWatchdog=MagicMock(),
            _ProviderPressureMonitor=MagicMock(),
            _RateLimitMonitor=MagicMock(),
        )

        mock_watchdog_cls.assert_called_once_with(mock_registry, fake_cfg.repos)
        mock_watchdog_cls.return_value.start_thread.assert_called_once()

    def test_run_starts_rate_limit_monitor_with_gh(self, tmp_path: Path) -> None:
        from fido.server import run

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
            _preflight_repo_identity=MagicMock(),
            _preflight_tools=MagicMock(),
            _preflight_sub_dir=MagicMock(),
            _preflight_gh_auth=MagicMock(),
            _GitHub=lambda: mock_gh_instance,
            _Watchdog=MagicMock(),
            _ReconcileWatchdog=MagicMock(),
            _RateLimitMonitor=mock_rl_cls,
            _ProviderPressureMonitor=MagicMock(),
        )

        # The state_updater is the second element returned by create_atomic(...).
        # RateLimitMonitor must be wired with the same updater that was passed to
        # make_registry — verify positional args rather than the updater value.
        assert mock_rl_cls.call_count == 1
        rl_args = mock_rl_cls.call_args[0]
        assert rl_args[0] is mock_gh_instance
        # The updater passed to RateLimitMonitor must be the same object that was
        # passed as state_updater to make_registry.
        assert rl_args[1] is mock_make_registry.call_args.kwargs["state_updater"]
        mock_rl_cls.return_value.start_thread.assert_called_once()

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
            _preflight_repo_identity=MagicMock(),
            _preflight_tools=MagicMock(),
            _preflight_sub_dir=MagicMock(),
            _preflight_gh_auth=MagicMock(),
            _GitHub=lambda: mock_gh_instance,
            _Watchdog=MagicMock(),
            _ReconcileWatchdog=mock_reconcile_cls,
            _ProviderPressureMonitor=MagicMock(),
            _RateLimitMonitor=MagicMock(),
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
                _preflight_repo_identity=MagicMock(),
                _preflight_tools=MagicMock(),
                _preflight_sub_dir=MagicMock(),
                _preflight_gh_auth=MagicMock(),
                _GitHub=MagicMock,
                _Watchdog=MagicMock(),
                _ProviderPressureMonitor=MagicMock(),
                _RateLimitMonitor=MagicMock(),
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
            _preflight_repo_identity=MagicMock(),
            _preflight_tools=MagicMock(),
            _preflight_sub_dir=MagicMock(),
            _preflight_gh_auth=MagicMock(),
            _GitHub=lambda: mock_gh_instance,
            _Watchdog=MagicMock(),
            _ReconcileWatchdog=MagicMock(),
            _bootstrap_issue_caches=mock_bootstrap,
            _ProviderPressureMonitor=MagicMock(),
            _RateLimitMonitor=MagicMock(),
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

    def run(self, cmd: object, **kwargs: object) -> object:
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

    def chdir(self, path: object) -> None:
        self.chdir_calls.append(path)

    def install_signal(self, signum: int, handler: object) -> object:
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
            _preflight_repo_identity=mock_preflight,
            _preflight_tools=MagicMock(),
            _preflight_sub_dir=MagicMock(),
            _preflight_gh_auth=MagicMock(),
            _GitHub=MagicMock,
            _ProviderPressureMonitor=MagicMock(),
            _RateLimitMonitor=MagicMock(),
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
            _preflight_repo_identity=MagicMock(),
            _preflight_tools=mock_preflight,
            _preflight_sub_dir=MagicMock(),
            _preflight_gh_auth=MagicMock(),
            _GitHub=MagicMock,
            _ProviderPressureMonitor=MagicMock(),
            _RateLimitMonitor=MagicMock(),
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
            _preflight_repo_identity=MagicMock(),
            _preflight_tools=MagicMock(),
            _preflight_sub_dir=MagicMock(),
            _preflight_gh_auth=mock_preflight,
            _GitHub=MagicMock,
            _ProviderPressureMonitor=MagicMock(),
            _RateLimitMonitor=MagicMock(),
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
            _preflight_repo_identity=MagicMock(),
            _preflight_tools=MagicMock(),
            _preflight_sub_dir=mock_preflight,
            _preflight_gh_auth=MagicMock(),
            _GitHub=MagicMock,
            _ProviderPressureMonitor=MagicMock(),
            _RateLimitMonitor=MagicMock(),
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
                _preflight_tools=MagicMock(
                    side_effect=PreflightError("something went wrong")
                ),
                _preflight_sub_dir=MagicMock(),
                _preflight_gh_auth=MagicMock(),
                _GitHub=MagicMock,
                _preflight_repo_identity=MagicMock(),
                _ProviderPressureMonitor=MagicMock(),
                _RateLimitMonitor=MagicMock(),
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

        assert set(_REQUIRED_TOOLS) == {"git", "gh", "claude", "copilot", "codex"}


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


class TestSelfRestart:
    """Tests for the self-restart flow."""

    def _make_server(self, tmp_path: Path) -> object:
        cfg = _self_restart_cfg(tmp_path)
        repo_cfg = cfg.repos["owner/fido"]
        mock_registry = MagicMock()
        WebhookHandler.config = cfg
        WebhookHandler.registry = mock_registry
        WebhookHandler.dispatchers = {
            "owner/fido": Dispatcher(cfg, repo_cfg, MagicMock())
        }
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

            def _kill() -> None:
                call_log.append("kill_active_children")

            real_stop_all = mock_registry.stop_all
            real_stop_and_join = mock_registry.stop_and_join
            real_stop_all.side_effect = lambda: call_log.append("stop_all")
            real_stop_and_join.side_effect = lambda repo: call_log.append(
                f"stop_and_join:{repo}"
            )
            WebhookHandler._fn_kill_active_children = staticmethod(_kill)  # type: ignore[assignment]

            os_proc = _FakeOsProcess()

            def _tracking_exit(code: int) -> None:  # type: ignore[no-untyped-def]
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

    def _make_unregistered_server(self, tmp_path: Path) -> object:
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
