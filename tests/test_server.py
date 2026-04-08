from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kennel.config import Config, RepoConfig
from kennel.server import WebhookHandler


def _config(tmp_path: Path) -> Config:
    return Config(
        port=0,  # will bind to random port
        secret=b"test-secret",
        repos={"owner/repo": RepoConfig(name="owner/repo", work_dir=tmp_path)},
        allowed_bots=frozenset({"copilot[bot]"}),
        log_level="WARNING",
        self_repo=None,
        sub_dir=tmp_path / "sub",
    )


def _sign(body: bytes, secret: bytes) -> str:
    return "sha256=" + hmac.new(secret, body, hashlib.sha256).hexdigest()


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
        with (
            patch("kennel.server.launch_worker") as mock_worker,
            patch("kennel.server.create_task"),
        ):
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
        with (
            patch(
                "kennel.server.reply_to_comment",
                return_value=(True, "ACT", "add logging"),
            ) as mock_reply,
            patch("kennel.server.create_task") as mock_task,
            patch("kennel.server.launch_worker"),
        ):
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
        with (
            patch(
                "kennel.server.reply_to_comment", return_value=(True, "DUMP", "nope")
            ) as mock_reply,
            patch("kennel.server.create_task") as mock_task,
            patch("kennel.server.launch_worker"),
        ):
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
        with (
            patch(
                "kennel.server.reply_to_comment",
                return_value=(True, "DEFER", "big refactor"),
            ),
            patch("kennel.server.create_task") as mock_task,
            patch("kennel.server.launch_worker"),
        ):
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

        with (
            patch(
                "kennel.server.reply_to_comment",
                return_value=(True, "DO", "add result caching"),
            ),
            patch("kennel.server.create_task", side_effect=capture_task),
            patch("kennel.server.launch_worker"),
        ):
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
            with (
                patch("kennel.server.reply_to_comment") as mock_reply,
                patch("kennel.server.create_task"),
                patch("kennel.server.launch_worker"),
            ):
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
        with (
            patch("kennel.server.reply_to_review") as mock_review,
            patch("kennel.server.launch_worker"),
        ):
            status = _post_webhook(url, cfg, "pull_request_review", payload)
            assert status == 200
            time.sleep(0.2)
            mock_review.assert_called()

    def test_issue_comment_handled(self, server: tuple) -> None:
        url, cfg = server
        payload = {
            **self._payload(),
            "action": "created",
            "comment": {"id": 300, "body": "looks good", "user": {"login": "owner"}},
            "issue": {
                "number": 11,
                "title": "my pr",
                "body": "",
                "pull_request": {"url": "https://api.github.com/..."},
            },
        }
        with (
            patch(
                "kennel.server.reply_to_issue_comment", return_value=("ACT", "do it")
            ) as mock_ic,
            patch("kennel.server.create_task") as mock_task,
            patch("kennel.server.launch_worker"),
        ):
            status = _post_webhook(url, cfg, "issue_comment", payload)
            assert status == 200
            time.sleep(0.2)
            mock_ic.assert_called()
            mock_task.assert_called()

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
        with (
            patch(
                "kennel.server.reply_to_issue_comment",
                return_value=("ANSWER", "because"),
            ) as mock_ic,
            patch("kennel.server.create_task") as mock_task,
            patch("kennel.server.launch_worker"),
        ):
            status = _post_webhook(url, cfg, "issue_comment", payload)
            assert status == 200
            time.sleep(0.2)
            mock_ic.assert_called()
            mock_task.assert_not_called()

    def test_exception_in_process_action_does_not_crash(self, server: tuple) -> None:
        url, cfg = server
        payload = {
            **self._payload(),
            "action": "closed",
            "pull_request": {"number": 13, "merged": True},
        }
        with patch("kennel.server.launch_worker", side_effect=Exception("explode")):
            status = _post_webhook(url, cfg, "pull_request", payload)
            assert status == 200
            time.sleep(0.2)
            # server still alive — no crash


class TestRun:
    """Tests for the run() entry point."""

    def test_run_starts_server(self, tmp_path: Path) -> None:
        from kennel.config import Config, RepoConfig
        from kennel.server import run

        fake_cfg = Config(
            port=0,
            secret=b"test",
            repos={"owner/repo": RepoConfig(name="owner/repo", work_dir=tmp_path)},
            allowed_bots=frozenset(),
            log_level="WARNING",
            self_repo=None,
            sub_dir=tmp_path / "sub",
        )

        mock_server = MagicMock()
        mock_server.serve_forever.side_effect = KeyboardInterrupt

        tmp_path / "log"

        with (
            patch("kennel.server.Config.from_args", return_value=fake_cfg),
            patch("kennel.server.HTTPServer", return_value=mock_server),
            patch("kennel.server.make_registry"),
            patch("pathlib.Path.home", return_value=tmp_path),
        ):
            run()

        mock_server.serve_forever.assert_called_once()
        mock_server.server_close.assert_called_once()

    def test_run_stderr_tty_adds_stream_handler(self, tmp_path: Path) -> None:
        from kennel.config import Config, RepoConfig
        from kennel.server import run

        fake_cfg = Config(
            port=0,
            secret=b"test",
            repos={"owner/repo": RepoConfig(name="owner/repo", work_dir=tmp_path)},
            allowed_bots=frozenset(),
            log_level="WARNING",
            self_repo=None,
            sub_dir=tmp_path / "sub",
        )

        mock_server = MagicMock()
        mock_server.serve_forever.side_effect = KeyboardInterrupt

        with (
            patch("kennel.server.Config.from_args", return_value=fake_cfg),
            patch("kennel.server.HTTPServer", return_value=mock_server),
            patch("kennel.server.make_registry"),
            patch("pathlib.Path.home", return_value=tmp_path),
            patch("sys.stderr") as mock_stderr,
        ):
            mock_stderr.isatty.return_value = True
            run()

        mock_server.serve_forever.assert_called_once()


class TestSelfRestart:
    """Tests for the self-restart flow (self_repo merge triggers exec)."""

    def test_self_restart_triggers_on_kennel_merge(self, tmp_path: Path) -> None:
        cfg = Config(
            port=0,
            secret=b"test-secret",
            repos={"owner/kennel": RepoConfig(name="owner/kennel", work_dir=tmp_path)},
            allowed_bots=frozenset(),
            log_level="WARNING",
            self_repo="owner/kennel",
            sub_dir=tmp_path / "sub",
        )
        WebhookHandler.config = cfg
        WebhookHandler.registry = MagicMock()
        srv = HTTPServer(("127.0.0.1", 0), WebhookHandler)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        url = f"http://127.0.0.1:{port}"
        try:
            payload = {
                "repository": {
                    "full_name": "owner/kennel",
                    "owner": {"login": "owner"},
                },
                "action": "closed",
                "pull_request": {"number": 1, "merged": True},
            }
            with (
                patch("kennel.server.subprocess.run"),
                patch("kennel.server.os.execv") as mock_exec,
            ):
                status = _post_webhook(url, cfg, "pull_request", payload)
                assert status == 200
                time.sleep(0.2)
                mock_exec.assert_called_once()
        finally:
            srv.shutdown()
