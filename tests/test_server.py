from __future__ import annotations

import hashlib
import hmac
import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kennel.config import Config, RepoConfig
from kennel.server import PreflightError, WebhookHandler


def _config(tmp_path: Path) -> Config:
    from kennel.config import RepoMembership

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
        "_fn_runner_dir": WebhookHandler._fn_runner_dir,
        "_fn_get_self_repo": WebhookHandler._fn_get_self_repo,
        "_fn_get_head": WebhookHandler._fn_get_head,
        "_fn_pull_with_backoff": WebhookHandler._fn_pull_with_backoff,
        "_fn_os_chdir": WebhookHandler._fn_os_chdir,
        "_fn_os_execvp": WebhookHandler._fn_os_execvp,
    }
    yield
    for attr, val in saved.items():
        setattr(WebhookHandler, attr, val)


@pytest.fixture()
def server(tmp_path: Path):
    cfg = _config(tmp_path)
    WebhookHandler.config = cfg
    WebhookHandler.registry = MagicMock()
    srv = HTTPServer(("127.0.0.1", 0), WebhookHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
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
        assert b"kennel is running" in resp.read()

    def test_status_endpoint_returns_activities(self, server: tuple) -> None:
        from datetime import datetime, timezone

        from kennel.registry import WorkerActivity

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
        resp = urllib.request.urlopen(f"{url}/status")
        assert resp.status == 200
        data = json.loads(resp.read())
        assert len(data) == 1
        entry = data[0]
        assert entry["repo_name"] == "owner/repo"
        assert entry["what"] == "Working on: #1"
        assert entry["busy"] is True
        assert entry["crash_count"] == 0
        assert entry["last_crash_error"] is None
        assert entry["is_stuck"] is False
        assert entry["worker_uptime_seconds"] is None
        assert entry["webhook_activities"] == []

    def test_status_endpoint_includes_crash_info(self, server: tuple) -> None:
        from datetime import datetime, timezone

        from kennel.registry import WorkerActivity, WorkerCrash

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
        resp = urllib.request.urlopen(f"{url}/status")
        data = json.loads(resp.read())
        assert data[0]["crash_count"] == 3
        assert data[0]["last_crash_error"] == "RuntimeError: boom"

    def test_status_endpoint_empty_when_no_activities(self, server: tuple) -> None:
        url, _ = server
        WebhookHandler.registry.get_all_activities.return_value = []
        resp = urllib.request.urlopen(f"{url}/status")
        assert resp.status == 200
        assert json.loads(resp.read()) == []

    def test_status_endpoint_content_type_json(self, server: tuple) -> None:
        url, _ = server
        WebhookHandler.registry.get_all_activities.return_value = []
        resp = urllib.request.urlopen(f"{url}/status")
        assert resp.headers.get("Content-Type") == "application/json"

    def test_status_endpoint_is_stuck_true_when_stale(self, server: tuple) -> None:
        from datetime import datetime, timezone

        from kennel.registry import WorkerActivity

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
        resp = urllib.request.urlopen(f"{url}/status")
        data = json.loads(resp.read())
        assert data[0]["is_stuck"] is True


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
    resp = urllib.request.urlopen(req)
    return resp.status


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
        time.sleep(0.2)
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
        time.sleep(0.2)
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
        time.sleep(0.2)
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
        time.sleep(0.2)
        mock_task.assert_not_called()

    def test_reply_to_comment_failure_skips_task(self, server: tuple) -> None:
        """If reply posting raises, no task is created (fail closed)."""
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
        time.sleep(0.2)
        mock_task.assert_not_called()

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
        time.sleep(0.2)
        assert task_titles == ["add result caching"]

    def test_already_replied_comment_skipped(self, server: tuple) -> None:
        url, cfg = server
        import kennel.server as ks

        ks._replied_comments.add(203)
        try:
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
            time.sleep(0.2)
            mock_reply.assert_not_called()
        finally:
            ks._replied_comments.discard(203)

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
        time.sleep(0.2)
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
        time.sleep(0.2)
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
        time.sleep(0.2)
        mock_ic.assert_called()
        mock_task.assert_not_called()

    def test_reply_to_issue_comment_failure_skips_task(self, server: tuple) -> None:
        """If issue comment reply raises, no task is created (fail closed)."""
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
        time.sleep(0.2)
        mock_task.assert_not_called()

    def test_process_action_emits_heartbeat(self, server: tuple) -> None:
        """_process_action calls registry.report_activity at the start."""
        url, cfg = server
        payload = {
            **self._payload(),
            "action": "closed",
            "pull_request": {"number": 14, "merged": True},
        }
        WebhookHandler._fn_launch_worker = MagicMock()
        status = _post_webhook(url, cfg, "pull_request", payload)
        assert status == 200
        time.sleep(0.2)
        WebhookHandler.registry.report_activity.assert_called_with(
            "owner/repo", "handling webhook action", busy=True
        )

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
        time.sleep(0.2)
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
        time.sleep(0.2)
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
        time.sleep(0.2)
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
        time.sleep(0.2)
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
        time.sleep(0.2)
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
        time.sleep(0.2)
        mock_gh.add_reaction.assert_called_once()

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


class TestPopulateMemberships:
    def test_populates_from_get_collaborators(self, tmp_path: Path) -> None:
        from kennel.config import RepoMembership
        from kennel.server import populate_memberships

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
        populate_memberships(cfg, _gh_factory=lambda: mock_gh)
        assert cfg.repos["owner/repo"].membership == RepoMembership(
            collaborators=frozenset({"alice", "bob"})
        )

    def test_filters_bot_user_from_collaborators(self, tmp_path: Path) -> None:
        from kennel.server import populate_memberships

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
        populate_memberships(cfg, _gh_factory=lambda: mock_gh)
        result = cfg.repos["owner/repo"].membership.collaborators
        assert result == frozenset({"alice", "bob"})
        assert "fido-bot" not in result

    def test_iterates_all_repos(self, tmp_path: Path) -> None:
        from kennel.server import populate_memberships

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
        populate_memberships(cfg, _gh_factory=lambda: mock_gh)
        assert cfg.repos["o/r1"].membership.collaborators == frozenset({"alice"})
        assert cfg.repos["o/r2"].membership.collaborators == frozenset({"bob", "carol"})


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
        from kennel.server import run

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
        )

        mock_server.serve_forever.assert_called_once()
        mock_server.server_close.assert_called_once()

    def test_run_keyboard_interrupt_kills_children(self, tmp_path: Path) -> None:
        from kennel.server import run

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
            _signal=MagicMock(),
            _kill_active_children=mock_kill,
        )
        mock_kill.assert_called_once()

    def test_run_installs_sigterm_and_sigint_handlers(self, tmp_path: Path) -> None:
        import signal as _sig

        from kennel.server import run

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
            _signal=fake_signal,
            _kill_active_children=MagicMock(),
        )
        assert _sig.SIGTERM in installed
        assert _sig.SIGINT in installed

    def test_shutdown_handler_kills_children_and_exits(self, tmp_path: Path) -> None:
        from kennel.server import run

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
        from kennel.server import run

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
        )

        assert len(captured_kwargs) == 1
        assert "%(repo_name)s" in captured_kwargs[0]["format"]

    def test_run_adds_repo_context_filter_to_handlers(self, tmp_path: Path) -> None:
        from kennel.server import run
        from kennel.worker import RepoContextFilter

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
        )

        assert len(captured_handlers) >= 1
        for handler in captured_handlers:
            assert any(isinstance(f, RepoContextFilter) for f in handler.filters)

    def test_run_stderr_tty_adds_stream_handler(self, tmp_path: Path) -> None:
        from kennel.server import run

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
        )

        mock_server.serve_forever.assert_called_once()

    def test_run_creates_per_repo_log_handlers(self, tmp_path: Path) -> None:
        from kennel.server import run
        from kennel.worker import RepoContextFilter, RepoNameFilter

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
        )

        # Two shared handlers (file + no tty stderr) + two per-repo handlers
        assert len(captured_handlers) == 3  # shared file + 2 per-repo (no tty)
        repo_handlers = [
            h
            for h in captured_handlers
            if any(isinstance(f, RepoNameFilter) for f in h.filters)
        ]
        assert len(repo_handlers) == 2
        short_names = {
            f.short_name
            for h in repo_handlers
            for f in h.filters
            if isinstance(f, RepoNameFilter)
        }
        assert short_names == {"myrepo", "other"}
        for handler in repo_handlers:
            assert any(isinstance(f, RepoContextFilter) for f in handler.filters)

    def test_run_per_repo_log_file_path(self, tmp_path: Path) -> None:
        from kennel.server import run
        from kennel.worker import RepoNameFilter

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
        )

        repo_handler = next(
            h
            for h in captured_handlers
            if any(isinstance(f, RepoNameFilter) for f in h.filters)
        )
        import logging

        assert isinstance(repo_handler, logging.FileHandler)
        assert repo_handler.baseFilename.endswith("kennel-myrepo.log")

    def test_run_starts_watchdog_with_registry_and_repos(self, tmp_path: Path) -> None:
        from kennel.server import run

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
            _Watchdog=mock_watchdog_cls,
        )

        mock_watchdog_cls.assert_called_once_with(mock_registry, fake_cfg.repos)
        mock_watchdog_cls.return_value.start_thread.assert_called_once()

    def test_run_installs_excepthooks(self, tmp_path: Path) -> None:
        """Uncaught exceptions (main thread and worker threads) should route
        through the logger so tracebacks land in kennel.log, not just stderr."""
        import sys as _sys
        import threading as _threading

        from kennel.server import run

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
                _Watchdog=MagicMock(),
            )
            assert _sys.excepthook is not saved_sys
            assert _threading.excepthook is not saved_thr

            # Call the hooks to confirm they don't raise and they go through logging.
            import kennel.server as srv_mod

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


def _self_restart_cfg(tmp_path: Path) -> Config:
    return Config(
        port=0,
        secret=b"test-secret",
        repos={"owner/kennel": RepoConfig(name="owner/kennel", work_dir=tmp_path)},
        allowed_bots=frozenset(),
        log_level="WARNING",
        sub_dir=tmp_path / "sub",
    )


_MERGE_PAYLOAD = {
    "repository": {
        "full_name": "owner/kennel",
        "owner": {"login": "owner"},
        "default_branch": "main",
    },
    "action": "closed",
    "pull_request": {"number": 1, "merged": True},
}

_PUSH_PAYLOAD = {
    "repository": {
        "full_name": "owner/kennel",
        "owner": {"login": "owner"},
        "default_branch": "main",
    },
    "ref": "refs/heads/main",
}


class TestParseRepoFromUrl:
    def test_parses_ssh_url(self) -> None:
        from kennel.server import _parse_repo_from_url

        assert _parse_repo_from_url("git@github.com:owner/repo.git") == "owner/repo"

    def test_parses_https_url(self) -> None:
        from kennel.server import _parse_repo_from_url

        assert _parse_repo_from_url("https://github.com/owner/repo.git") == "owner/repo"

    def test_parses_url_without_git_suffix(self) -> None:
        from kennel.server import _parse_repo_from_url

        assert _parse_repo_from_url("https://github.com/owner/repo") == "owner/repo"

    def test_returns_none_for_garbage(self) -> None:
        from kennel.server import _parse_repo_from_url

        assert _parse_repo_from_url("garbage") is None


class TestPreflightRepoIdentity:
    def test_succeeds_when_remote_matches(self, tmp_path: Path) -> None:
        from kennel.server import preflight_repo_identity

        repos = {"owner/repo": RepoConfig(name="owner/repo", work_dir=tmp_path)}
        mock_run = MagicMock(
            return_value=MagicMock(stdout="git@github.com:owner/repo.git\n")
        )
        preflight_repo_identity(repos, _run=mock_run)  # no exception

    def test_raises_on_remote_mismatch(self, tmp_path: Path) -> None:
        from kennel.server import preflight_repo_identity

        repos = {"owner/repo": RepoConfig(name="owner/repo", work_dir=tmp_path)}
        mock_run = MagicMock(
            return_value=MagicMock(stdout="git@github.com:other/thing.git\n")
        )
        with pytest.raises(PreflightError, match="other/thing"):
            preflight_repo_identity(repos, _run=mock_run)

    def test_raises_on_subprocess_error(self, tmp_path: Path) -> None:
        from kennel.server import preflight_repo_identity

        repos = {"owner/repo": RepoConfig(name="owner/repo", work_dir=tmp_path)}
        mock_run = MagicMock(side_effect=subprocess.CalledProcessError(128, []))
        with pytest.raises(PreflightError):
            preflight_repo_identity(repos, _run=mock_run)

    def test_raises_when_git_not_found(self, tmp_path: Path) -> None:
        from kennel.server import preflight_repo_identity

        repos = {"owner/repo": RepoConfig(name="owner/repo", work_dir=tmp_path)}
        mock_run = MagicMock(side_effect=FileNotFoundError())
        with pytest.raises(PreflightError):
            preflight_repo_identity(repos, _run=mock_run)

    def test_raises_on_unparseable_url(self, tmp_path: Path) -> None:
        from kennel.server import preflight_repo_identity

        repos = {"owner/repo": RepoConfig(name="owner/repo", work_dir=tmp_path)}
        mock_run = MagicMock(return_value=MagicMock(stdout="garbage\n"))
        with pytest.raises(PreflightError):
            preflight_repo_identity(repos, _run=mock_run)

    def test_checks_all_repos(self, tmp_path: Path) -> None:
        from kennel.server import preflight_repo_identity

        repos = {
            "owner/repo1": RepoConfig(name="owner/repo1", work_dir=tmp_path),
            "owner/repo2": RepoConfig(name="owner/repo2", work_dir=tmp_path),
        }
        mock_run = MagicMock(
            side_effect=[
                MagicMock(stdout="git@github.com:owner/repo1.git\n"),
                MagicMock(stdout="git@github.com:owner/repo2.git\n"),
            ]
        )
        preflight_repo_identity(repos, _run=mock_run)  # no exception
        assert mock_run.call_count == 2

    def test_raises_on_second_repo_mismatch(self, tmp_path: Path) -> None:
        from kennel.server import preflight_repo_identity

        repos = {
            "owner/repo1": RepoConfig(name="owner/repo1", work_dir=tmp_path),
            "owner/repo2": RepoConfig(name="owner/repo2", work_dir=tmp_path),
        }
        mock_run = MagicMock(
            side_effect=[
                MagicMock(stdout="git@github.com:owner/repo1.git\n"),
                MagicMock(stdout="git@github.com:other/thing.git\n"),
            ]
        )
        with pytest.raises(PreflightError, match="other/thing"):
            preflight_repo_identity(repos, _run=mock_run)

    def test_run_calls_preflight_repo_identity(self, tmp_path: Path) -> None:
        from kennel.server import run

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
        )

        mock_preflight.assert_called_once_with(fake_cfg.repos)

    def test_run_calls_preflight_tools(self, tmp_path: Path) -> None:
        from kennel.server import run

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
        )

        mock_preflight.assert_called_once_with()

    def test_run_calls_preflight_gh_auth(self, tmp_path: Path) -> None:
        from kennel.server import run

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
        )

        mock_preflight.assert_called_once_with()

    def test_run_calls_preflight_sub_dir(self, tmp_path: Path) -> None:
        from kennel.server import run

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
        )

        mock_preflight.assert_called_once_with(fake_cfg)

    def test_run_converts_preflight_error_to_system_exit(self, tmp_path: Path) -> None:
        from kennel.server import run

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
                _preflight_repo_identity=MagicMock(),
            )


class TestPreflightTools:
    def test_succeeds_when_all_tools_found(self) -> None:
        from kennel.server import preflight_tools

        preflight_tools(_which=lambda _: "/usr/bin/git")  # no exception

    def test_raises_when_git_missing(self) -> None:
        from kennel.server import preflight_tools

        missing = "git"
        with pytest.raises(PreflightError, match=repr(missing)):
            preflight_tools(_which=lambda t: None if t == missing else f"/usr/bin/{t}")

    def test_raises_when_gh_missing(self) -> None:
        from kennel.server import preflight_tools

        missing = "gh"
        with pytest.raises(PreflightError, match=repr(missing)):
            preflight_tools(_which=lambda t: None if t == missing else f"/usr/bin/{t}")

    def test_raises_when_claude_missing(self) -> None:
        from kennel.server import preflight_tools

        missing = "claude"
        with pytest.raises(PreflightError, match=repr(missing)):
            preflight_tools(_which=lambda t: None if t == missing else f"/usr/bin/{t}")

    def test_required_tools_constant(self) -> None:
        from kennel.server import _REQUIRED_TOOLS

        assert set(_REQUIRED_TOOLS) == {"git", "gh", "claude"}


class TestPreflightSubDir:
    def test_succeeds_when_sub_dir_exists(self, tmp_path: Path) -> None:
        from kennel.server import preflight_sub_dir

        cfg = Config(
            port=0,
            secret=b"s",
            repos={},
            allowed_bots=frozenset(),
            log_level="INFO",
            sub_dir=tmp_path / "sub",
        )
        preflight_sub_dir(cfg, _is_dir=lambda _: True)  # no exception

    def test_raises_when_sub_dir_missing(self, tmp_path: Path) -> None:
        from kennel.server import preflight_sub_dir

        cfg = Config(
            port=0,
            secret=b"s",
            repos={},
            allowed_bots=frozenset(),
            log_level="INFO",
            sub_dir=tmp_path / "sub",
        )
        with pytest.raises(PreflightError, match="skill-files directory not found"):
            preflight_sub_dir(cfg, _is_dir=lambda _: False)

    def test_error_message_includes_path(self, tmp_path: Path) -> None:
        from kennel.server import preflight_sub_dir

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
            preflight_sub_dir(cfg, _is_dir=lambda _: False)


class TestPreflightGhAuth:
    def test_succeeds_when_get_user_works(self) -> None:
        from kennel.server import preflight_gh_auth

        mock_gh = MagicMock()
        mock_gh.return_value.get_user.return_value = "fido-bot"
        preflight_gh_auth(_gh_factory=mock_gh)  # no exception

    def test_raises_when_get_user_raises_runtime_error(self) -> None:
        from kennel.server import preflight_gh_auth

        mock_gh = MagicMock()
        mock_gh.return_value.get_user.side_effect = RuntimeError("not logged in")
        with pytest.raises(PreflightError, match="not logged in"):
            preflight_gh_auth(_gh_factory=mock_gh)

    def test_raises_when_gh_factory_raises(self) -> None:
        from kennel.server import preflight_gh_auth

        mock_gh = MagicMock(side_effect=RuntimeError("gh auth token failed"))
        with pytest.raises(PreflightError, match="gh auth token failed"):
            preflight_gh_auth(_gh_factory=mock_gh)

    def test_raises_when_get_user_raises_any_exception(self) -> None:
        from kennel.server import preflight_gh_auth

        mock_gh = MagicMock()
        mock_gh.return_value.get_user.side_effect = Exception("network error")
        with pytest.raises(PreflightError, match="network error"):
            preflight_gh_auth(_gh_factory=mock_gh)


class TestGetSelfRepo:
    def test_parses_ssh_remote(self, tmp_path: Path) -> None:
        from kennel.server import _get_self_repo

        mock_run = MagicMock(
            return_value=MagicMock(
                stdout="git@github.com:owner/kennel.git\n", returncode=0
            )
        )
        assert _get_self_repo(tmp_path, _run=mock_run) == "owner/kennel"

    def test_parses_https_remote(self, tmp_path: Path) -> None:
        from kennel.server import _get_self_repo

        mock_run = MagicMock(
            return_value=MagicMock(
                stdout="https://github.com/owner/kennel.git\n", returncode=0
            )
        )
        assert _get_self_repo(tmp_path, _run=mock_run) == "owner/kennel"

    def test_parses_remote_without_git_suffix(self, tmp_path: Path) -> None:
        from kennel.server import _get_self_repo

        mock_run = MagicMock(
            return_value=MagicMock(
                stdout="https://github.com/owner/kennel\n", returncode=0
            )
        )
        assert _get_self_repo(tmp_path, _run=mock_run) == "owner/kennel"

    def test_returns_none_on_subprocess_error(self, tmp_path: Path) -> None:
        from kennel.server import _get_self_repo

        mock_run = MagicMock(side_effect=subprocess.CalledProcessError(128, []))
        assert _get_self_repo(tmp_path, _run=mock_run) is None

    def test_returns_none_on_file_not_found(self, tmp_path: Path) -> None:
        from kennel.server import _get_self_repo

        mock_run = MagicMock(side_effect=FileNotFoundError())
        assert _get_self_repo(tmp_path, _run=mock_run) is None

    def test_returns_none_on_unparseable_url(self, tmp_path: Path) -> None:
        from kennel.server import _get_self_repo

        mock_run = MagicMock(return_value=MagicMock(stdout="garbage\n", returncode=0))
        assert _get_self_repo(tmp_path, _run=mock_run) is None


class TestGetHead:
    def test_returns_sha(self, tmp_path: Path) -> None:
        from kennel.server import _get_head

        run = MagicMock(return_value=MagicMock(stdout="abc123def456\n"))
        assert _get_head(tmp_path, _run=run) == "abc123def456"

    def test_returns_none_on_subprocess_error(self, tmp_path: Path) -> None:
        from kennel.server import _get_head

        run = MagicMock(side_effect=subprocess.CalledProcessError(128, []))
        assert _get_head(tmp_path, _run=run) is None

    def test_returns_none_on_file_not_found(self, tmp_path: Path) -> None:
        from kennel.server import _get_head

        run = MagicMock(side_effect=FileNotFoundError())
        assert _get_head(tmp_path, _run=run) is None


class TestRunnerDir:
    def test_returns_package_parent(self) -> None:
        from kennel.server import _runner_dir

        result = _runner_dir()
        assert (result / "kennel" / "server.py").exists()


class TestPullWithBackoff:
    def test_success_on_first_try(self, tmp_path: Path) -> None:
        from kennel.server import _pull_with_backoff

        mock_run = MagicMock(return_value=MagicMock(returncode=0))
        mock_sleep = MagicMock()
        assert _pull_with_backoff(
            tmp_path, _run=mock_run, _sleep=mock_sleep, _monotonic=lambda: 0.0
        )
        # Each sync attempt runs two git commands: fetch + reset.
        assert mock_run.call_count == 2
        cmds = [c.args[0] for c in mock_run.call_args_list]
        assert cmds[0] == ["git", "fetch", "origin", "main"]
        assert cmds[1] == ["git", "reset", "--hard", "origin/main"]
        mock_sleep.assert_not_called()

    def test_success_after_retry(self, tmp_path: Path) -> None:
        from kennel.server import _pull_with_backoff

        # First attempt: fetch fails.  Second attempt: fetch + reset both succeed.
        mock_run = MagicMock(
            side_effect=[
                subprocess.CalledProcessError(1, []),
                MagicMock(returncode=0),
                MagicMock(returncode=0),
            ]
        )
        mock_sleep = MagicMock()
        times = iter([0.0, 1.0, 1.0])
        assert _pull_with_backoff(
            tmp_path,
            _run=mock_run,
            _sleep=mock_sleep,
            _monotonic=lambda: next(times),
        )
        assert mock_run.call_count == 3
        mock_sleep.assert_called_once_with(10)

    def test_recovers_from_divergent_local(self, tmp_path: Path) -> None:
        """The whole point of using fetch+reset: divergent local branch is fine.

        Previously git pull would fail with `fatal: Need to specify how to
        reconcile divergent branches` and the runner would never sync.  With
        reset --hard the local state is forcibly replaced, so the worker
        catches up regardless of how the runner clone got into a weird state.
        """
        from kennel.server import _pull_with_backoff

        mock_run = MagicMock(return_value=MagicMock(returncode=0))
        mock_sleep = MagicMock()
        assert _pull_with_backoff(
            tmp_path, _run=mock_run, _sleep=mock_sleep, _monotonic=lambda: 0.0
        )
        assert ["git", "reset", "--hard", "origin/main"] in [
            c.args[0] for c in mock_run.call_args_list
        ]
        mock_sleep.assert_not_called()

    def test_gives_up_after_all_retries_fail(self, tmp_path: Path) -> None:
        from kennel.server import _pull_with_backoff

        # Fetch fails on every attempt — 4 attempts total.
        mock_run = MagicMock(side_effect=subprocess.CalledProcessError(1, []))
        mock_sleep = MagicMock()
        times = iter([0.0, 1.0, 1.0, 12.0, 12.0, 43.0, 43.0, 104.0])
        assert not _pull_with_backoff(
            tmp_path,
            _run=mock_run,
            _sleep=mock_sleep,
            _monotonic=lambda: next(times),
        )
        # 4 attempts × 1 failing fetch each = 4 calls.
        assert mock_run.call_count == 4
        # 3 sleeps at 10s, 30s, 60s between retries.
        assert [c.args[0] for c in mock_sleep.call_args_list] == [10, 30, 60]

    def test_reset_failure_retries(self, tmp_path: Path) -> None:
        """Fetch succeeds but reset fails — both commands matter for the result."""
        from kennel.server import _pull_with_backoff

        mock_run = MagicMock(
            side_effect=[
                MagicMock(returncode=0),  # fetch
                subprocess.CalledProcessError(1, []),  # reset
                MagicMock(returncode=0),  # fetch retry
                MagicMock(returncode=0),  # reset retry
            ]
        )
        mock_sleep = MagicMock()
        times = iter([0.0, 1.0, 1.0])
        assert _pull_with_backoff(
            tmp_path,
            _run=mock_run,
            _sleep=mock_sleep,
            _monotonic=lambda: next(times),
        )
        assert mock_run.call_count == 4
        mock_sleep.assert_called_once_with(10)

    def test_gives_up_when_budget_exhausted(self, tmp_path: Path) -> None:
        from kennel.server import _pull_with_backoff

        mock_run = MagicMock(side_effect=subprocess.CalledProcessError(1, []))
        mock_sleep = MagicMock()
        # First attempt at t=0, elapsed=595; next delay of 10s would exceed 600s budget.
        times = iter([0.0, 595.0, 595.0])
        assert not _pull_with_backoff(
            tmp_path,
            _run=mock_run,
            _sleep=mock_sleep,
            _monotonic=lambda: next(times),
        )
        # Slept zero times because budget was exhausted before any sleep.
        mock_sleep.assert_not_called()

    def test_returns_false_on_file_not_found(self, tmp_path: Path) -> None:
        from kennel.server import _pull_with_backoff

        mock_run = MagicMock(side_effect=FileNotFoundError())
        mock_sleep = MagicMock()
        times = iter([0.0, 1.0, 1.0, 12.0, 12.0, 43.0, 43.0, 104.0])
        assert not _pull_with_backoff(
            tmp_path,
            _run=mock_run,
            _sleep=mock_sleep,
            _monotonic=lambda: next(times),
        )


class TestStartupPull:
    def test_reexecs_when_head_changes(self) -> None:
        from kennel.server import _startup_pull

        heads = iter(["aaa", "bbb"])
        mock_exec = MagicMock()
        _startup_pull(
            _runner_dir=lambda: Path("/fake"),
            _get_head=lambda _d: next(heads),
            _pull=lambda _d: True,
            _execvp=mock_exec,
        )
        mock_exec.assert_called_once()
        assert mock_exec.call_args.args[0] == "uv"

    def test_skips_exec_when_head_unchanged(self) -> None:
        from kennel.server import _startup_pull

        mock_exec = MagicMock()
        _startup_pull(
            _runner_dir=lambda: Path("/fake"),
            _get_head=lambda _d: "same_sha",
            _pull=lambda _d: True,
            _execvp=mock_exec,
        )
        mock_exec.assert_not_called()

    def test_continues_on_pull_failure(self) -> None:
        from kennel.server import _startup_pull

        mock_exec = MagicMock()
        _startup_pull(
            _runner_dir=lambda: Path("/fake"),
            _get_head=lambda _d: "aaa",
            _pull=lambda _d: False,
            _execvp=mock_exec,
        )
        mock_exec.assert_not_called()

    def test_execs_when_head_unknown(self) -> None:
        from kennel.server import _startup_pull

        mock_exec = MagicMock()
        _startup_pull(
            _runner_dir=lambda: Path("/fake"),
            _get_head=lambda _d: None,
            _pull=lambda _d: True,
            _execvp=mock_exec,
        )
        # Can't compare HEAD — but pull succeeded, so log and continue
        mock_exec.assert_not_called()


class TestSelfRestart:
    """Tests for the self-restart flow."""

    def _make_server(self, tmp_path: Path):
        cfg = _self_restart_cfg(tmp_path)
        mock_registry = MagicMock()
        WebhookHandler.config = cfg
        WebhookHandler.registry = mock_registry
        srv = HTTPServer(("127.0.0.1", 0), WebhookHandler)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        return srv, f"http://127.0.0.1:{port}", cfg, mock_registry

    def test_triggers_exec_on_matching_repo(self, tmp_path: Path) -> None:
        srv, url, cfg, mock_registry = self._make_server(tmp_path)
        try:
            WebhookHandler._fn_runner_dir = lambda: tmp_path
            WebhookHandler._fn_get_self_repo = lambda _d: "owner/kennel"
            WebhookHandler._fn_pull_with_backoff = lambda _d: True
            mock_chdir = MagicMock()
            mock_exec = MagicMock()
            WebhookHandler._fn_os_chdir = mock_chdir
            WebhookHandler._fn_os_execvp = mock_exec
            status = _post_webhook(url, cfg, "pull_request", _MERGE_PAYLOAD)
            assert status == 200
            time.sleep(0.2)
            mock_registry.stop_and_join.assert_called_once_with("owner/kennel")
            mock_chdir.assert_called_once_with(tmp_path)
            mock_exec.assert_called_once()
            args = mock_exec.call_args.args
            assert args[0] == "uv"
            assert args[1][:3] == ["uv", "run", "kennel"]
        finally:
            srv.shutdown()

    def test_skips_when_self_repo_mismatch(self, tmp_path: Path) -> None:
        srv, url, cfg, mock_registry = self._make_server(tmp_path)
        try:
            WebhookHandler._fn_runner_dir = lambda: tmp_path
            WebhookHandler._fn_get_self_repo = lambda _d: "other/repo"
            mock_pull = MagicMock()
            mock_exec = MagicMock()
            WebhookHandler._fn_pull_with_backoff = mock_pull
            WebhookHandler._fn_os_execvp = mock_exec
            _post_webhook(url, cfg, "pull_request", _MERGE_PAYLOAD)
            time.sleep(0.2)
            mock_registry.stop_and_join.assert_not_called()
            mock_pull.assert_not_called()
            mock_exec.assert_not_called()
        finally:
            srv.shutdown()

    def test_skips_when_self_repo_unknown(self, tmp_path: Path) -> None:
        srv, url, cfg, mock_registry = self._make_server(tmp_path)
        try:
            WebhookHandler._fn_runner_dir = lambda: tmp_path
            WebhookHandler._fn_get_self_repo = lambda _d: None
            mock_exec = MagicMock()
            WebhookHandler._fn_os_execvp = mock_exec
            _post_webhook(url, cfg, "pull_request", _MERGE_PAYLOAD)
            time.sleep(0.2)
            mock_registry.stop_and_join.assert_not_called()
            mock_exec.assert_not_called()
        finally:
            srv.shutdown()

    def test_pull_failure_leaves_worker_alone(self, tmp_path: Path) -> None:
        """When pull gives up, do NOT tear down the worker thread.

        Previously the worker for the merged repo was stopped before the
        pull, and stayed stopped after pull failure — silently leaving
        kennel without a fido worker on its own repo.  Now the pull runs
        first and the worker is only torn down on success.
        """
        srv, url, cfg, mock_registry = self._make_server(tmp_path)
        try:
            WebhookHandler._fn_runner_dir = lambda: tmp_path
            WebhookHandler._fn_get_self_repo = lambda _d: "owner/kennel"
            WebhookHandler._fn_pull_with_backoff = lambda _d: False
            mock_exec = MagicMock()
            WebhookHandler._fn_os_execvp = mock_exec
            _post_webhook(url, cfg, "pull_request", _MERGE_PAYLOAD)
            time.sleep(0.2)
            mock_registry.stop_and_join.assert_not_called()
            mock_exec.assert_not_called()
        finally:
            srv.shutdown()

    def test_pull_precedes_stop_and_join(self, tmp_path: Path) -> None:
        """Pull MUST run before stop_and_join, so a failed pull leaves the
        worker intact."""
        call_order: list[str] = []
        srv, url, cfg, mock_registry = self._make_server(tmp_path)
        mock_registry.stop_and_join.side_effect = lambda *_a, **_kw: call_order.append(
            "stop_and_join"
        )
        try:
            WebhookHandler._fn_runner_dir = lambda: tmp_path
            WebhookHandler._fn_get_self_repo = lambda _d: "owner/kennel"

            def fake_pull(_d):
                call_order.append("pull")
                return True

            WebhookHandler._fn_pull_with_backoff = fake_pull
            WebhookHandler._fn_os_chdir = MagicMock()
            WebhookHandler._fn_os_execvp = MagicMock()
            _post_webhook(url, cfg, "pull_request", _MERGE_PAYLOAD)
            time.sleep(0.2)
        finally:
            srv.shutdown()
        assert call_order == ["pull", "stop_and_join"]

    def test_push_to_default_branch_triggers_restart(self, tmp_path: Path) -> None:
        srv, url, cfg, mock_registry = self._make_server(tmp_path)
        try:
            WebhookHandler._fn_runner_dir = lambda: tmp_path
            WebhookHandler._fn_get_self_repo = lambda _d: "owner/kennel"
            WebhookHandler._fn_pull_with_backoff = lambda _d: True
            mock_chdir = MagicMock()
            mock_exec = MagicMock()
            WebhookHandler._fn_os_chdir = mock_chdir
            WebhookHandler._fn_os_execvp = mock_exec
            _post_webhook(url, cfg, "push", _PUSH_PAYLOAD)
            time.sleep(0.2)
            mock_registry.stop_and_join.assert_called_once_with("owner/kennel")
            mock_exec.assert_called_once()
        finally:
            srv.shutdown()

    def test_push_to_non_default_branch_ignored(self, tmp_path: Path) -> None:
        srv, url, cfg, mock_registry = self._make_server(tmp_path)
        try:
            WebhookHandler._fn_runner_dir = lambda: tmp_path
            mock_pull = MagicMock()
            mock_exec = MagicMock()
            WebhookHandler._fn_pull_with_backoff = mock_pull
            WebhookHandler._fn_os_execvp = mock_exec
            payload = {**_PUSH_PAYLOAD, "ref": "refs/heads/feature-branch"}
            _post_webhook(url, cfg, "push", payload)
            time.sleep(0.2)
            mock_pull.assert_not_called()
            mock_exec.assert_not_called()
        finally:
            srv.shutdown()

    def _make_unregistered_server(self, tmp_path: Path):
        """Server whose config does NOT include the kennel repo."""
        from kennel.config import RepoMembership

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
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        return srv, f"http://127.0.0.1:{port}", cfg

    def test_merged_pr_on_unregistered_repo_triggers_restart(
        self, tmp_path: Path
    ) -> None:
        """Self-restart fires for merged PR even when the repo is not registered."""
        srv, url, cfg = self._make_unregistered_server(tmp_path)
        try:
            WebhookHandler._fn_runner_dir = lambda: tmp_path
            WebhookHandler._fn_get_self_repo = lambda _d: "owner/kennel"
            WebhookHandler._fn_pull_with_backoff = lambda _d: True
            mock_chdir = MagicMock()
            mock_exec = MagicMock()
            WebhookHandler._fn_os_chdir = mock_chdir
            WebhookHandler._fn_os_execvp = mock_exec
            status = _post_webhook(url, cfg, "pull_request", _MERGE_PAYLOAD)
            assert status == 200
            time.sleep(0.2)
            mock_exec.assert_called_once()
        finally:
            srv.shutdown()

    def test_push_to_default_on_unregistered_repo_triggers_restart(
        self, tmp_path: Path
    ) -> None:
        """Self-restart fires for push to default branch even when repo is not registered."""
        srv, url, cfg = self._make_unregistered_server(tmp_path)
        try:
            WebhookHandler._fn_runner_dir = lambda: tmp_path
            WebhookHandler._fn_get_self_repo = lambda _d: "owner/kennel"
            WebhookHandler._fn_pull_with_backoff = lambda _d: True
            mock_chdir = MagicMock()
            mock_exec = MagicMock()
            WebhookHandler._fn_os_chdir = mock_chdir
            WebhookHandler._fn_os_execvp = mock_exec
            status = _post_webhook(url, cfg, "push", _PUSH_PAYLOAD)
            assert status == 200
            time.sleep(0.2)
            mock_exec.assert_called_once()
        finally:
            srv.shutdown()
