from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kennel.github import (
    _HTTP_TIMEOUT,  # noqa: PLC2701
    GitHub,
    GraphQLError,
    _gh_token,
    _TimeoutSession,  # noqa: PLC2701
)


def _completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=""
    )


class TestGhToken:
    def test_uses_env_var(self) -> None:
        assert _gh_token(environ={"GITHUB_TOKEN": "mytoken"}) == "mytoken"

    def test_falls_back_to_gh_cli(self) -> None:
        mock_run = MagicMock(return_value=_completed("ghp_abc\n"))
        assert _gh_token(runner=mock_run, environ={}) == "ghp_abc"

    def test_gh_cli_strips_whitespace(self) -> None:
        mock_run = MagicMock(return_value=_completed("  tok  \n"))
        assert _gh_token(runner=mock_run, environ={}) == "tok"

    def test_raises_on_nonzero_exit(self) -> None:
        mock_run = MagicMock(return_value=_completed("", returncode=1))
        with pytest.raises(RuntimeError, match="gh auth token failed"):
            _gh_token(runner=mock_run, environ={})

    def test_error_message_includes_exit_code(self) -> None:
        mock_run = MagicMock(return_value=_completed("", returncode=4))
        with pytest.raises(RuntimeError, match=r"exit 4"):
            _gh_token(runner=mock_run, environ={})


class TestTimeoutSession:
    def test_injects_default_timeout(self) -> None:
        from unittest.mock import patch

        with patch("requests.Session.request") as mock_req:
            mock_req.return_value = MagicMock()
            s = _TimeoutSession()
            s.request("GET", "https://example.com")
        assert mock_req.call_args.kwargs.get("timeout") == _HTTP_TIMEOUT

    def test_does_not_override_caller_timeout(self) -> None:
        from unittest.mock import patch

        with patch("requests.Session.request") as mock_req:
            mock_req.return_value = MagicMock()
            s = _TimeoutSession()
            s.request("GET", "https://example.com", timeout=5)
        assert mock_req.call_args.kwargs.get("timeout") == 5

    def test_uses_timeout_session_when_no_session_injected(self) -> None:
        gh = GitHub("tok")
        assert isinstance(gh._s, _TimeoutSession)


class TestGitHubClass:
    def _gh(self) -> tuple[GitHub, MagicMock]:
        mock_s = MagicMock()
        return GitHub("test-token", session=mock_s), mock_s

    def test_sets_auth_header(self) -> None:
        gh = GitHub("test-token")
        assert gh._s.headers["Authorization"] == "Bearer test-token"

    def test_sets_accept_header(self) -> None:
        gh = GitHub("test-token")
        assert gh._s.headers["Accept"] == "application/vnd.github+json"

    def test_get_calls_session(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"id": 1}]
        mock_s.get.return_value = mock_resp
        result = gh._get("/repos/o/r/issues")
        mock_s.get.assert_called_once_with("https://api.github.com/repos/o/r/issues")
        assert result == [{"id": 1}]

    def test_get_raises_on_error(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("404")
        mock_s.get.return_value = mock_resp
        try:
            gh._get("/bad")
            assert False, "should have raised"
        except Exception as e:
            assert "404" in str(e)

    def test_post_calls_session(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_s.post.return_value = mock_resp
        gh._post("/repos/o/r/issues/1/comments", body="hi")
        mock_s.post.assert_called_once_with(
            "https://api.github.com/repos/o/r/issues/1/comments",
            json={"body": "hi"},
        )

    def test_post_raises_on_error(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("422")
        mock_s.post.return_value = mock_resp
        try:
            gh._post("/bad")
            assert False, "should have raised"
        except Exception as e:
            assert "422" in str(e)

    def test_post_json_returns_response(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "id": 42,
            "html_url": "https://github.com/o/r/pull/42",
        }
        mock_s.post.return_value = mock_resp
        result = gh._post_json("/repos/o/r/pulls", title="t", body="b")
        assert result["id"] == 42

    def test_patch_calls_session(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_s.patch.return_value = mock_resp
        gh._patch("/repos/o/r/issues/1", state="closed")
        mock_s.patch.assert_called_once_with(
            "https://api.github.com/repos/o/r/issues/1",
            json={"state": "closed"},
        )

    def test_patch_raises_on_error(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("404")
        mock_s.patch.return_value = mock_resp
        try:
            gh._patch("/bad")
            assert False, "should have raised"
        except Exception as e:
            assert "404" in str(e)

    def test_put_calls_session(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_s.put.return_value = mock_resp
        gh._put("/repos/o/r/pulls/1/merge", merge_method="squash")
        mock_s.put.assert_called_once_with(
            "https://api.github.com/repos/o/r/pulls/1/merge",
            json={"merge_method": "squash"},
        )

    def test_put_raises_on_error(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("405")
        mock_s.put.return_value = mock_resp
        try:
            gh._put("/bad")
            assert False, "should have raised"
        except Exception as e:
            assert "405" in str(e)

    def test_graphql_posts_to_graphql_endpoint(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {}}
        mock_s.post.return_value = mock_resp
        result = gh._graphql("query { viewer { login } }", login="fido")
        url = mock_s.post.call_args.args[0]
        assert url == "https://api.github.com/graphql"
        body = mock_s.post.call_args.kwargs["json"]
        assert body["query"] == "query { viewer { login } }"
        assert body["variables"] == {"login": "fido"}
        assert result == {"data": {}}

    def test_graphql_raises_on_http_error(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("500")
        mock_s.post.return_value = mock_resp
        with pytest.raises(Exception, match="500"):
            gh._graphql("query {}")

    def test_graphql_raises_on_payload_error(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"errors": [{"message": "not found"}]}
        mock_s.post.return_value = mock_resp
        with pytest.raises(GraphQLError) as exc_info:
            gh._graphql("query {}")
        assert exc_info.value.errors == [{"message": "not found"}]

    def test_add_reaction_pulls(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_s.post.return_value = mock_resp
        gh.add_reaction("o/r", "pulls", 42, "rocket")
        url = mock_s.post.call_args.args[0]
        assert "repos/o/r/pulls/comments/42/reactions" in url
        assert mock_s.post.call_args.kwargs["json"]["content"] == "rocket"

    def test_add_reaction_issues(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_s.post.return_value = mock_resp
        gh.add_reaction("o/r", "issues", 7, "+1")
        url = mock_s.post.call_args.args[0]
        assert "repos/o/r/issues/comments/7/reactions" in url

    def test_reply_to_review_comment(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_s.post.return_value = mock_resp
        gh.reply_to_review_comment("o/r", 10, "lgtm", 55)
        url = mock_s.post.call_args.args[0]
        assert "repos/o/r/pulls/10/comments" in url
        body = mock_s.post.call_args.kwargs["json"]
        assert body["body"] == "lgtm"
        assert body["in_reply_to"] == 55

    def test_reply_to_review_comment_converts_in_reply_to(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_s.post.return_value = mock_resp
        gh.reply_to_review_comment("o/r", 10, "ok", "99")
        body = mock_s.post.call_args.kwargs["json"]
        assert body["in_reply_to"] == 99

    def test_edit_review_comment(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_s.patch.return_value = mock_resp
        gh.edit_review_comment("o/r", 77, "updated body")
        url = mock_s.patch.call_args.args[0]
        assert "repos/o/r/pulls/comments/77" in url
        body = mock_s.patch.call_args.kwargs["json"]
        assert body["body"] == "updated body"

    def test_get_pull_comments(self) -> None:
        gh, mock_s = self._gh()
        comments = [{"id": 42, "body": "looks good"}]
        mock_resp = MagicMock()
        mock_resp.json.return_value = comments
        mock_resp.headers = {}
        mock_s.get.return_value = mock_resp
        result = gh.get_pull_comments("o/r", 7)
        assert result == comments

    def test_get_pull_comments_url(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.headers = {}
        mock_s.get.return_value = mock_resp
        gh.get_pull_comments("o/r", 7)
        url = mock_s.get.call_args.args[0]
        assert "repos/o/r/pulls/7/comments" in url

    def test_get_pull_comment(self) -> None:
        gh, mock_s = self._gh()
        comment = {"id": 10, "body": "hi"}
        mock_resp = MagicMock()
        mock_resp.json.return_value = comment
        mock_s.get.return_value = mock_resp
        assert gh.get_pull_comment("o/r", 10) == comment

    def test_get_pull_comment_returns_none_on_404(self) -> None:
        import requests

        gh, mock_s = self._gh()
        response = MagicMock(status_code=404)
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError(response=response)
        mock_s.get.return_value = mock_resp
        assert gh.get_pull_comment("o/r", 10) is None

    def test_get_pull_comment_reraises_non_404(self) -> None:
        import requests

        gh, mock_s = self._gh()
        response = MagicMock(status_code=500)
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError(response=response)
        mock_s.get.return_value = mock_resp
        with pytest.raises(requests.HTTPError):
            gh.get_pull_comment("o/r", 10)

    def test_get_review_comments(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"id": 101, "body": "nit"},
            {"id": 102, "body": "fix"},
        ]
        mock_resp.headers = {}
        mock_s.get.return_value = mock_resp
        result = gh.get_review_comments("o/r", 10, 99)
        assert result == [(101, "nit"), (102, "fix")]

    def test_get_review_comments_empty(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.headers = {}
        mock_s.get.return_value = mock_resp
        result = gh.get_review_comments("o/r", 10, 99)
        assert result == []

    def test_get_review_comments_url(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.headers = {}
        mock_s.get.return_value = mock_resp
        gh.get_review_comments("o/r", 10, 99)
        url = mock_s.get.call_args.args[0]
        assert "repos/o/r/pulls/10/reviews/99/comments" in url

    def test_comment_issue(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_s.post.return_value = mock_resp
        gh.comment_issue("o/r", 7, "hello")
        url = mock_s.post.call_args.args[0]
        assert "repos/o/r/issues/7/comments" in url
        assert mock_s.post.call_args.kwargs["json"]["body"] == "hello"

    def test_fetch_sibling_threads_returns_threads(self) -> None:
        gh, mock_s = self._gh()
        comments = [
            {
                "id": 10,
                "in_reply_to_id": None,
                "path": "a.py",
                "line": 5,
                "user": {"login": "alice"},
                "body": "root",
            },
            {
                "id": 11,
                "in_reply_to_id": 10,
                "path": "a.py",
                "line": 5,
                "user": {"login": "bob"},
                "body": "reply",
            },
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = comments
        mock_resp.headers = {}
        mock_s.get.return_value = mock_resp
        result = gh.fetch_sibling_threads("o/r", 7)
        assert len(result) == 1
        assert result[0]["path"] == "a.py"
        assert len(result[0]["comments"]) == 2

    def test_fetch_sibling_threads_skips_reply_to_unknown_root(self) -> None:
        gh, mock_s = self._gh()
        comments = [
            {
                "id": 20,
                "in_reply_to_id": 99,
                "path": "b.py",
                "line": 1,
                "user": {"login": "alice"},
                "body": "orphan reply",
            },
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = comments
        mock_resp.headers = {}
        mock_s.get.return_value = mock_resp
        result = gh.fetch_sibling_threads("o/r", 7)
        assert result == []

    def test_fetch_sibling_threads_returns_empty_on_error(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("403")
        mock_s.get.return_value = mock_resp
        result = gh.fetch_sibling_threads("o/r", 7)
        assert result == []

    def test_fetch_comment_thread_returns_thread(self) -> None:
        gh, mock_s = self._gh()
        comments = [
            {
                "id": 10,
                "in_reply_to_id": None,
                "user": {"login": "alice"},
                "body": "root comment",
            },
            {
                "id": 11,
                "in_reply_to_id": 10,
                "user": {"login": "fido"},
                "body": "reply one",
            },
            {
                "id": 12,
                "in_reply_to_id": 10,
                "user": {"login": "alice"},
                "body": "reply two",
            },
            {
                "id": 20,
                "in_reply_to_id": None,
                "user": {"login": "bob"},
                "body": "other thread",
            },
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = comments
        mock_resp.headers = {}
        mock_s.get.return_value = mock_resp
        result = gh.fetch_comment_thread("o/r", 7, 10)
        assert result == [
            {"id": 10, "author": "alice", "body": "root comment"},
            {"id": 11, "author": "fido", "body": "reply one"},
            {"id": 12, "author": "alice", "body": "reply two"},
        ]

    def test_fetch_comment_thread_finds_thread_by_reply_id(self) -> None:
        """When comment_id is a reply, finds the root and returns the full thread."""
        gh, mock_s = self._gh()
        comments = [
            {
                "id": 10,
                "in_reply_to_id": None,
                "user": {"login": "alice"},
                "body": "root",
            },
            {
                "id": 11,
                "in_reply_to_id": 10,
                "user": {"login": "fido"},
                "body": "reply",
            },
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = comments
        mock_resp.headers = {}
        mock_s.get.return_value = mock_resp
        result = gh.fetch_comment_thread("o/r", 7, 11)
        assert result == [
            {"id": 10, "author": "alice", "body": "root"},
            {"id": 11, "author": "fido", "body": "reply"},
        ]

    def test_fetch_comment_thread_returns_empty_when_not_found(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.headers = {}
        mock_s.get.return_value = mock_resp
        result = gh.fetch_comment_thread("o/r", 7, 999)
        assert result == []

    def test_fetch_comment_thread_returns_empty_on_error(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("403")
        mock_s.get.return_value = mock_resp
        result = gh.fetch_comment_thread("o/r", 7, 10)
        assert result == []

    def test_get_run_log_skips_non_failing_jobs(self) -> None:
        gh, mock_s = self._gh()
        jobs_resp = MagicMock()
        jobs_resp.json.return_value = {
            "jobs": [
                {"id": 1, "conclusion": "success"},
                {"id": 2, "conclusion": "failure"},
            ]
        }
        log_resp = MagicMock()
        log_resp.text = "failure log\n"
        mock_s.get.side_effect = [jobs_resp, log_resp]
        result = gh.get_run_log("o/r", 42)
        assert result == "failure log\n"
        # Only one GET for logs (the success job was skipped)
        assert mock_s.get.call_count == 2

    def test_paginate_single_page(self) -> None:
        gh, mock_s = self._gh()
        resp = MagicMock()
        resp.json.return_value = [{"id": 1}, {"id": 2}]
        resp.headers = {}
        mock_s.get.return_value = resp
        result = list(gh._paginate("https://api.github.com/repos/o/r/items"))
        assert result == [{"id": 1}, {"id": 2}]

    def test_paginate_follows_next_link(self) -> None:
        gh, mock_s = self._gh()
        page1 = MagicMock()
        page1.json.return_value = [{"id": 1}]
        next_url = "https://api.github.com/repos/o/r/items?page=2"
        page1.headers = {"Link": f'<{next_url}>; rel="next"'}
        page2 = MagicMock()
        page2.json.return_value = [{"id": 2}]
        page2.headers = {}
        mock_s.get.side_effect = [page1, page2]
        result = list(gh._paginate("https://api.github.com/repos/o/r/items"))
        assert result == [{"id": 1}, {"id": 2}]
        assert mock_s.get.call_count == 2
        assert mock_s.get.call_args_list[1].args[0] == next_url

    def _transient_resp(self, status: int) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status
        resp.reason = "Server Error"
        return resp

    def _ok_resp(self, body: list[dict], link: str = "") -> MagicMock:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = body
        resp.headers = {"Link": link} if link else {}
        return resp

    def test_retryable_get_retries_on_5xx_then_succeeds(self) -> None:
        # Regression for #664: transient 5xx on a read-only GET must be
        # retried, not propagated as an uncaught worker-thread crash.
        import requests as _requests

        sleeper = MagicMock()
        mock_s = MagicMock()
        mock_s.get.side_effect = [
            self._transient_resp(500),
            self._transient_resp(503),
            self._ok_resp([{"id": 7}]),
        ]
        gh = GitHub("tok", session=mock_s, sleeper=sleeper)
        assert gh._get("/anything") == [{"id": 7}]
        assert mock_s.get.call_count == 3
        # Two failures → two sleeps from the retry schedule.
        assert sleeper.call_count == 2
        assert sleeper.call_args_list[0].args[0] == 1.0
        assert sleeper.call_args_list[1].args[0] == 3.0
        # Sanity: no real network-error import drift.
        assert issubclass(_requests.ConnectionError, Exception)

    def test_retryable_get_gives_up_after_budget(self) -> None:
        import requests as _requests

        sleeper = MagicMock()
        mock_s = MagicMock()
        mock_s.get.return_value = self._transient_resp(502)
        gh = GitHub("tok", session=mock_s, sleeper=sleeper)
        with pytest.raises(_requests.HTTPError, match="502"):
            gh._get("/always-bad")
        # Initial attempt + len(_GET_RETRY_DELAYS) retries.
        assert mock_s.get.call_count == 4
        assert sleeper.call_count == 3

    def test_retryable_get_retries_on_connection_error(self) -> None:
        import requests as _requests

        sleeper = MagicMock()
        mock_s = MagicMock()
        mock_s.get.side_effect = [
            _requests.ConnectionError("boom"),
            self._ok_resp([{"ok": True}]),
        ]
        gh = GitHub("tok", session=mock_s, sleeper=sleeper)
        assert gh._get("/flaky") == [{"ok": True}]
        assert mock_s.get.call_count == 2
        assert sleeper.call_count == 1

    def test_retryable_get_non_retryable_4xx_propagates_immediately(self) -> None:
        import requests as _requests

        sleeper = MagicMock()
        mock_s = MagicMock()
        resp = MagicMock()
        resp.status_code = 404
        resp.raise_for_status.side_effect = _requests.HTTPError("404")
        mock_s.get.return_value = resp
        gh = GitHub("tok", session=mock_s, sleeper=sleeper)
        with pytest.raises(_requests.HTTPError, match="404"):
            gh._get("/nope")
        # No retry on 4xx — single GET, no sleep.
        assert mock_s.get.call_count == 1
        sleeper.assert_not_called()

    def test_paginate_retries_on_5xx_mid_stream(self) -> None:
        # Multi-page pagination: a 5xx on page 2 must not abort the
        # whole walk — it retries just that page.
        sleeper = MagicMock()
        mock_s = MagicMock()
        next_url = "https://api.github.com/repos/o/r/items?page=2"
        page1 = self._ok_resp([{"id": 1}], link=f'<{next_url}>; rel="next"')
        bad = self._transient_resp(502)
        page2 = self._ok_resp([{"id": 2}])
        mock_s.get.side_effect = [page1, bad, page2]
        gh = GitHub("tok", session=mock_s, sleeper=sleeper)
        result = list(gh._paginate("https://api.github.com/repos/o/r/items"))
        assert result == [{"id": 1}, {"id": 2}]
        assert mock_s.get.call_count == 3
        assert sleeper.call_count == 1

    def _gql_pr(
        self, number: int, ref: str, state: str, user: str, body: str = ""
    ) -> dict:
        return {
            "__typename": "PullRequest",
            "number": number,
            "headRefName": ref,
            "state": state,
            "author": {"login": user},
            "body": body,
        }

    def _gql_timeline(
        self,
        nodes: list[dict],
        has_next: bool = False,
        cursor: str | None = None,
    ) -> dict:
        return {
            "data": {
                "repository": {
                    "issue": {
                        "timelineItems": {
                            "pageInfo": {
                                "hasNextPage": has_next,
                                "endCursor": cursor,
                            },
                            "nodes": nodes,
                        }
                    }
                }
            }
        }

    def _cross_ref_node(self, pr: dict) -> dict:
        return {"__typename": "CrossReferencedEvent", "source": pr}

    def _connected_node(self, pr: dict) -> dict:
        pr_no_body = {k: v for k, v in pr.items() if k != "body"}
        return {"__typename": "ConnectedEvent", "subject": pr_no_body}

    def _disconnected_node(self, pr_number: int) -> dict:
        return {
            "__typename": "DisconnectedEvent",
            "subject": {"__typename": "PullRequest", "number": pr_number},
        }

    def test_find_pr_returns_match(self) -> None:
        gh, mock_s = self._gh()
        pr = self._gql_pr(1, "feat", "OPEN", "fido", "closes #5")
        mock_s.post.return_value.json.return_value = self._gql_timeline(
            [self._cross_ref_node(pr)]
        )
        result = gh.find_pr("o/r", 5, "fido")
        assert result == {
            "number": 1,
            "headRefName": "feat",
            "state": "OPEN",
            "author": {"login": "fido"},
        }

    def test_find_pr_skips_merged(self) -> None:
        """Merged PRs are skipped — a reopened issue should get a fresh PR."""
        gh, mock_s = self._gh()
        pr = self._gql_pr(3, "fix", "MERGED", "fido", "closes #2")
        mock_s.post.return_value.json.return_value = self._gql_timeline(
            [self._cross_ref_node(pr)]
        )
        assert gh.find_pr("o/r", 2, "fido") is None

    def test_find_pr_skips_closed(self) -> None:
        """Closed (not merged) PRs are skipped — a reopened issue should get a fresh PR."""
        gh, mock_s = self._gh()
        pr = self._gql_pr(4, "fix", "CLOSED", "fido", "closes #2")
        mock_s.post.return_value.json.return_value = self._gql_timeline(
            [self._cross_ref_node(pr)]
        )
        assert gh.find_pr("o/r", 2, "fido") is None

    def test_find_pr_skips_non_open_returns_subsequent_open_pr(self) -> None:
        """When a merged/closed PR and a later open PR reference the issue, returns open."""
        gh, mock_s = self._gh()
        merged = self._gql_pr(3, "fix", "MERGED", "fido", "closes #2")
        open_pr = self._gql_pr(7, "fix-retry", "OPEN", "fido", "closes #2")
        mock_s.post.return_value.json.return_value = self._gql_timeline(
            [self._cross_ref_node(merged), self._cross_ref_node(open_pr)]
        )
        result = gh.find_pr("o/r", 2, "fido")
        assert result is not None
        assert result["number"] == 7
        assert result["state"] == "OPEN"

    def test_find_pr_filters_by_user(self) -> None:
        gh, mock_s = self._gh()
        pr = self._gql_pr(1, "feat", "OPEN", "other", "closes #5")
        mock_s.post.return_value.json.return_value = self._gql_timeline(
            [self._cross_ref_node(pr)]
        )
        assert gh.find_pr("o/r", 5, "fido") is None

    def test_find_pr_returns_none_on_empty(self) -> None:
        gh, mock_s = self._gh()
        mock_s.post.return_value.json.return_value = self._gql_timeline([])
        assert gh.find_pr("o/r", 1, "fido") is None

    def test_find_pr_skips_non_pr_cross_reference(self) -> None:
        """Cross-referenced events from plain issues (not PRs) are ignored."""
        gh, mock_s = self._gh()
        issue_ref = {"__typename": "Issue", "number": 9}
        node = {"__typename": "CrossReferencedEvent", "source": issue_ref}
        mock_s.post.return_value.json.return_value = self._gql_timeline([node])
        assert gh.find_pr("o/r", 5, "fido") is None

    def test_find_pr_skips_substring_match(self) -> None:
        """#9 must not match a PR body that only contains #90."""
        gh, mock_s = self._gh()
        pr = self._gql_pr(99, "feat", "OPEN", "fido", "closes #90")
        mock_s.post.return_value.json.return_value = self._gql_timeline(
            [self._cross_ref_node(pr)]
        )
        assert gh.find_pr("o/r", 9, "fido") is None

    def test_find_pr_skips_prefix_match(self) -> None:
        """#100 must not match a PR body that only contains closes #10."""
        gh, mock_s = self._gh()
        pr = self._gql_pr(99, "feat", "OPEN", "fido", "closes #10")
        mock_s.post.return_value.json.return_value = self._gql_timeline(
            [self._cross_ref_node(pr)]
        )
        assert gh.find_pr("o/r", 100, "fido") is None

    def test_find_pr_requires_closing_keyword(self) -> None:
        """Bare #N in PR body (no closing keyword) must not match."""
        gh, mock_s = self._gh()
        pr = self._gql_pr(7, "feat", "OPEN", "fido", "see #5 for context")
        mock_s.post.return_value.json.return_value = self._gql_timeline(
            [self._cross_ref_node(pr)]
        )
        assert gh.find_pr("o/r", 5, "fido") is None

    def test_find_pr_title_keyword_matches(self) -> None:
        """Issue reference in PR title (not body) still matches."""
        gh, mock_s = self._gh()
        pr = self._gql_pr(7, "feat", "OPEN", "fido", "")
        node = self._cross_ref_node(pr)
        node["source"]["title"] = "closes #5"
        mock_s.post.return_value.json.return_value = self._gql_timeline([node])
        result = gh.find_pr("o/r", 5, "fido")
        assert result is not None
        assert result["number"] == 7

    def test_find_pr_no_keyword_in_body_or_title(self) -> None:
        """No closing keyword in body or title — not matched."""
        gh, mock_s = self._gh()
        pr = self._gql_pr(7, "feat", "OPEN", "fido", "some other text")
        node = self._cross_ref_node(pr)
        node["source"]["title"] = "unrelated title"
        mock_s.post.return_value.json.return_value = self._gql_timeline([node])
        assert gh.find_pr("o/r", 5, "fido") is None

    def test_find_pr_uses_graphql(self) -> None:
        gh, mock_s = self._gh()
        mock_s.post.return_value.json.return_value = self._gql_timeline([])
        gh.find_pr("o/r", 5, "fido")
        url = mock_s.post.call_args.args[0]
        assert url.endswith("/graphql")
        body = mock_s.post.call_args.kwargs["json"]
        assert body["variables"]["owner"] == "o"
        assert body["variables"]["repo"] == "r"
        assert body["variables"]["number"] == 5

    def test_find_pr_follows_pagination(self) -> None:
        """find_pr fetches additional pages when hasNextPage is true."""
        gh, mock_s = self._gh()
        pr = self._gql_pr(7, "feat", "OPEN", "fido", "closes #5")
        page1 = self._gql_timeline(
            [{"__typename": "LabeledEvent"}], has_next=True, cursor="abc"
        )
        page2 = self._gql_timeline([self._cross_ref_node(pr)])
        mock_s.post.return_value.json.side_effect = [page1, page2]
        result = gh.find_pr("o/r", 5, "fido")
        assert result is not None
        assert result["number"] == 7
        assert mock_s.post.call_count == 2
        assert (
            mock_s.post.call_args_list[1].kwargs["json"]["variables"]["cursor"] == "abc"
        )

    def test_find_pr_connected_event(self) -> None:
        """PRs linked via Development sidebar (connected event) are found."""
        gh, mock_s = self._gh()
        pr = self._gql_pr(7, "feat", "OPEN", "fido")
        mock_s.post.return_value.json.return_value = self._gql_timeline(
            [self._connected_node(pr)]
        )
        result = gh.find_pr("o/r", 5, "fido")
        assert result is not None
        assert result["number"] == 7

    def test_find_pr_disconnected_event_excludes_pr(self) -> None:
        """A sidebar-connected-then-disconnected PR is not returned."""
        gh, mock_s = self._gh()
        pr = self._gql_pr(7, "feat", "OPEN", "fido")
        nodes = [self._connected_node(pr), self._disconnected_node(7)]
        mock_s.post.return_value.json.return_value = self._gql_timeline(nodes)
        assert gh.find_pr("o/r", 5, "fido") is None

    def test_find_pr_disconnected_does_not_affect_keyword_linked(self) -> None:
        """DisconnectedEvent does not remove a keyword-linked (cross-referenced) PR."""
        gh, mock_s = self._gh()
        pr = self._gql_pr(7, "feat", "OPEN", "fido", "closes #5")
        nodes = [self._cross_ref_node(pr), self._disconnected_node(7)]
        mock_s.post.return_value.json.return_value = self._gql_timeline(nodes)
        result = gh.find_pr("o/r", 5, "fido")
        assert result is not None
        assert result["number"] == 7

    def test_find_pr_connected_filters_by_user(self) -> None:
        gh, mock_s = self._gh()
        pr = self._gql_pr(7, "feat", "OPEN", "other")
        mock_s.post.return_value.json.return_value = self._gql_timeline(
            [self._connected_node(pr)]
        )
        assert gh.find_pr("o/r", 5, "fido") is None

    def test_find_pr_skips_non_pr_connected(self) -> None:
        """Connected events where the subject is an Issue (not a PR) are ignored."""
        gh, mock_s = self._gh()
        node = {
            "__typename": "ConnectedEvent",
            "subject": {"__typename": "Issue", "number": 9},
        }
        mock_s.post.return_value.json.return_value = self._gql_timeline([node])
        assert gh.find_pr("o/r", 5, "fido") is None

    def test_find_pr_raises_on_graphql_error(self) -> None:
        gh, mock_s = self._gh()
        mock_s.post.return_value.json.return_value = {
            "errors": [{"message": "something went wrong"}]
        }
        with pytest.raises(GraphQLError):
            gh.find_pr("o/r", 1, "fido")

    def test_find_pr_returns_none_on_empty_timeline(self) -> None:
        gh, mock_s = self._gh()
        mock_s.post.return_value.json.return_value = {
            "data": {"repository": {"issue": {"timelineItems": {}}}}
        }
        assert gh.find_pr("o/r", 1, "fido") is None

    def test_get_user(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"login": "fido", "id": 1}
        mock_s.get.return_value = mock_resp
        result = gh.get_user()
        url = mock_s.get.call_args.args[0]
        assert url.endswith("/user")
        assert result == "fido"

    def test_get_collaborators_filters_by_permission(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"login": "admin-user", "role_name": "admin"},
            {"login": "maint-user", "role_name": "maintain"},
            {"login": "write-user", "role_name": "write"},
            {"login": "triage-user", "role_name": "triage"},
            {"login": "read-user", "role_name": "read"},
        ]
        mock_resp.headers = {}
        mock_s.get.return_value = mock_resp
        result = gh.get_collaborators("owner/repo")
        assert result == ["admin-user", "maint-user", "write-user"]
        url = mock_s.get.call_args.args[0]
        assert url.endswith("/repos/owner/repo/collaborators")

    def test_get_collaborators_preserves_order(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"login": "second", "role_name": "write"},
            {"login": "first", "role_name": "admin"},
        ]
        mock_resp.headers = {}
        mock_s.get.return_value = mock_resp
        # API order preserved — caller uses [0] as "primary reviewer"
        assert gh.get_collaborators("o/r") == ["second", "first"]

    def test_get_collaborators_skips_users_missing_login(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"role_name": "admin"},
            {"login": "alice", "role_name": "write"},
        ]
        mock_resp.headers = {}
        mock_s.get.return_value = mock_resp
        assert gh.get_collaborators("o/r") == ["alice"]

    def test_get_collaborators_handles_missing_role(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"login": "alice"},
            {"login": "bob", "role_name": "admin"},
        ]
        mock_resp.headers = {}
        mock_s.get.return_value = mock_resp
        assert gh.get_collaborators("o/r") == ["bob"]

    def test_get_repo_info_https(self) -> None:
        gh = GitHub("test-token")
        mock_run = MagicMock(
            return_value=_completed("https://github.com/owner/repo.git\n")
        )
        assert gh.get_repo_info(runner=mock_run) == "owner/repo"

    def test_get_repo_info_ssh(self) -> None:
        gh = GitHub("test-token")
        mock_run = MagicMock(return_value=_completed("git@github.com:owner/repo.git\n"))
        assert gh.get_repo_info(runner=mock_run) == "owner/repo"

    def test_get_repo_info_passes_cwd(self) -> None:
        gh = GitHub("test-token")
        mock_run = MagicMock(return_value=_completed("https://github.com/o/r.git"))
        gh.get_repo_info(cwd="/tmp/repo", runner=mock_run)
        assert mock_run.call_args.kwargs["cwd"] == "/tmp/repo"

    def test_get_repo_info_raises_unknown(self) -> None:
        gh = GitHub("test-token")
        mock_run = MagicMock(return_value=_completed("https://example.com/repo.git"))
        with pytest.raises(ValueError, match="Cannot parse"):
            gh.get_repo_info(runner=mock_run)

    def test_get_repo_info_raises_on_git_failure(self) -> None:
        gh = GitHub("test-token")
        mock_run = MagicMock(side_effect=subprocess.CalledProcessError(128, "git"))
        with pytest.raises(subprocess.CalledProcessError):
            gh.get_repo_info(runner=mock_run)

    def test_get_repo_info_raises_on_file_not_found(self) -> None:
        gh = GitHub("test-token")
        mock_run = MagicMock(side_effect=FileNotFoundError("git not found"))
        with pytest.raises(FileNotFoundError):
            gh.get_repo_info(runner=mock_run)

    def test_get_default_branch(self) -> None:
        gh, mock_s = self._gh()
        remote_resp = _completed("https://github.com/o/r.git\n")
        repo_resp = MagicMock()
        repo_resp.json.return_value = {"default_branch": "main"}
        mock_run = MagicMock(return_value=remote_resp)
        mock_s.get.return_value = repo_resp
        assert gh.get_default_branch(runner=mock_run) == "main"

    def test_get_default_branch_passes_cwd(self) -> None:
        gh, mock_s = self._gh()
        remote_resp = _completed("https://github.com/o/r.git\n")
        repo_resp = MagicMock()
        repo_resp.json.return_value = {"default_branch": "main"}
        mock_run = MagicMock(return_value=remote_resp)
        mock_s.get.return_value = repo_resp
        gh.get_default_branch(cwd=Path("/repo"), runner=mock_run)
        assert mock_run.call_args.kwargs["cwd"] == Path("/repo")

    def test_set_user_status_graphql(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {}}
        mock_s.post.return_value = mock_resp
        gh.set_user_status("working", "🚀", busy=True)
        url = mock_s.post.call_args.args[0]
        assert url.endswith("/graphql")
        body = mock_s.post.call_args.kwargs["json"]
        assert "changeUserStatus" in body["query"]
        assert body["variables"]["msg"] == "working"
        assert body["variables"]["emoji"] == "🚀"
        assert body["variables"]["busy"] is True

    def test_graphql_paginate_follows_next_cursor(self) -> None:
        gh, mock_s = self._gh()
        page1 = MagicMock()
        page1.json.return_value = {
            "data": {
                "repository": {
                    "issues": {
                        "nodes": [{"number": 1}],
                        "pageInfo": {"hasNextPage": True, "endCursor": "CUR1"},
                    }
                }
            }
        }
        page2 = MagicMock()
        page2.json.return_value = {
            "data": {
                "repository": {
                    "issues": {
                        "nodes": [{"number": 2}],
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                    }
                }
            }
        }
        mock_s.post.side_effect = [page1, page2]
        query = "query($cursor:String){repository{issues(after:$cursor){nodes{number} pageInfo{endCursor hasNextPage}}}}"
        results = list(gh._graphql_paginate(query, ("repository", "issues")))
        assert results == [{"number": 1}, {"number": 2}]
        # Second call carries the cursor from page 1's endCursor.
        assert (
            mock_s.post.call_args_list[1].kwargs["json"]["variables"]["cursor"]
            == "CUR1"
        )

    def test_get_sub_issues_paginates(self) -> None:
        gh, mock_s = self._gh()
        resp = MagicMock()
        resp.json.return_value = {
            "data": {
                "repository": {
                    "issue": {
                        "subIssues": {
                            "nodes": [{"number": 10, "title": "child"}],
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                        }
                    }
                }
            }
        }
        mock_s.post.return_value = resp
        result = list(gh.get_sub_issues("owner", "repo", 5))
        assert result == [{"number": 10, "title": "child"}]
        assert mock_s.post.call_args.kwargs["json"]["variables"]["number"] == 5

    def test_get_issue_node_returns_full_shape(self) -> None:
        gh, mock_s = self._gh()
        node = {
            "number": 99,
            "title": "parent",
            "state": "OPEN",
            "subIssues": {
                "nodes": [{"number": 100, "title": "child"}],
                "pageInfo": {"hasNextPage": False},
            },
        }
        resp = MagicMock()
        resp.json.return_value = {"data": {"repository": {"issue": node}}}
        mock_s.post.return_value = resp
        result = gh.get_issue_node("owner", "repo", 99)
        assert result["number"] == 99
        assert result["subIssues"]["nodes"] == [{"number": 100, "title": "child"}]

    def test_get_issue_node_hydrates_paginated_sub_issues(self) -> None:
        """When the embedded sub-issues page has more, refetch with get_sub_issues."""
        gh, mock_s = self._gh()
        first_resp = MagicMock()
        first_resp.json.return_value = {
            "data": {
                "repository": {
                    "issue": {
                        "number": 99,
                        "title": "parent",
                        "state": "OPEN",
                        "subIssues": {
                            "nodes": [{"number": 100}],
                            "pageInfo": {"hasNextPage": True, "endCursor": None},
                        },
                    }
                }
            }
        }
        hydrate_resp = MagicMock()
        hydrate_resp.json.return_value = {
            "data": {
                "repository": {
                    "issue": {
                        "subIssues": {
                            "nodes": [{"number": 100}, {"number": 101}],
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                        }
                    }
                }
            }
        }
        mock_s.post.side_effect = [first_resp, hydrate_resp]
        result = gh.get_issue_node("owner", "repo", 99)
        # Hydrated list replaces the paginated stub.
        assert [n["number"] for n in result["subIssues"]["nodes"]] == [100, 101]

    def test_find_issues_hydrates_paginated_sub_issues(self) -> None:
        gh, mock_s = self._gh()
        first_resp = MagicMock()
        first_resp.json.return_value = {
            "data": {
                "repository": {
                    "issues": {
                        "nodes": [
                            {
                                "number": 1,
                                "title": "parent",
                                "subIssues": {
                                    "nodes": [{"number": 10}],
                                    "pageInfo": {
                                        "hasNextPage": True,
                                        "endCursor": None,
                                    },
                                },
                            }
                        ],
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                    }
                }
            }
        }
        hydrate_resp = MagicMock()
        hydrate_resp.json.return_value = {
            "data": {
                "repository": {
                    "issue": {
                        "subIssues": {
                            "nodes": [{"number": 10}, {"number": 11}],
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                        }
                    }
                }
            }
        }
        mock_s.post.side_effect = [first_resp, hydrate_resp]
        result = gh.find_issues("owner", "repo", "fido")
        assert [n["number"] for n in result[0]["subIssues"]["nodes"]] == [10, 11]

    def test_add_assignee_posts_to_rest_endpoint(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_s.post.return_value = mock_resp
        gh.add_assignee("owner/repo", 42, "fido")
        assert "/repos/owner/repo/issues/42/assignees" in mock_s.post.call_args.args[0]
        assert mock_s.post.call_args.kwargs["json"] == {"assignees": ["fido"]}

    def test_find_issues_graphql(self) -> None:
        gh, mock_s = self._gh()
        nodes = [
            {
                "number": 1,
                "title": "bug",
                "subIssues": {"nodes": [], "pageInfo": {"hasNextPage": False}},
            }
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": {
                "repository": {
                    "issues": {
                        "nodes": nodes,
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                    }
                }
            }
        }
        mock_s.post.return_value = mock_resp
        result = gh.find_issues("owner", "repo", "fido")
        body = mock_s.post.call_args.kwargs["json"]
        assert body["variables"] == {
            "owner": "owner",
            "repo": "repo",
            "login": "fido",
            "cursor": None,
        }
        assert result == nodes

    def test_find_all_open_issues_returns_all_issues(self) -> None:
        gh, mock_s = self._gh()
        nodes = [
            {
                "number": 1,
                "title": "bug",
                "state": "OPEN",
                "subIssues": {"nodes": []},
            },
            {
                "number": 2,
                "title": "feature",
                "state": "OPEN",
                "subIssues": {"nodes": []},
            },
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": {
                "repository": {
                    "issues": {
                        "nodes": nodes,
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                    }
                }
            }
        }
        mock_s.post.return_value = mock_resp
        result = gh.find_all_open_issues("owner", "repo")
        body = mock_s.post.call_args.kwargs["json"]
        # No login/assignee filter in variables
        assert body["variables"] == {
            "owner": "owner",
            "repo": "repo",
            "cursor": None,
        }
        assert result == nodes

    def test_view_issue(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "state": "open",
            "title": "Bug",
            "body": "desc",
            "created_at": "2024-01-01T00:00:00Z",
        }
        mock_s.get.return_value = mock_resp
        result = gh.view_issue("o/r", 5)
        assert result == {
            "state": "OPEN",
            "title": "Bug",
            "body": "desc",
            "created_at": "2024-01-01T00:00:00Z",
        }

    def test_get_issue_comments(self) -> None:
        gh, mock_s = self._gh()
        comments = [{"id": 1, "body": "hi"}]
        mock_resp = MagicMock()
        mock_resp.json.return_value = comments
        mock_resp.headers = {}
        mock_s.get.return_value = mock_resp
        result = gh.get_issue_comments("o/r", 9)
        url = mock_s.get.call_args.args[0]
        assert "repos/o/r/issues/9/comments" in url
        assert result == comments

    def test_get_issue_comment(self) -> None:
        gh, mock_s = self._gh()
        comment = {"id": 9, "body": "hi"}
        mock_resp = MagicMock()
        mock_resp.json.return_value = comment
        mock_s.get.return_value = mock_resp
        assert gh.get_issue_comment("o/r", 9) == comment

    def test_get_issue_comment_returns_none_on_404(self) -> None:
        import requests

        gh, mock_s = self._gh()
        response = MagicMock(status_code=404)
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError(response=response)
        mock_s.get.return_value = mock_resp
        assert gh.get_issue_comment("o/r", 9) is None

    def test_get_issue_comment_reraises_non_404(self) -> None:
        import requests

        gh, mock_s = self._gh()
        response = MagicMock(status_code=500)
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError(response=response)
        mock_s.get.return_value = mock_resp
        with pytest.raises(requests.HTTPError):
            gh.get_issue_comment("o/r", 9)

    def test_get_issue_events(self) -> None:
        gh, mock_s = self._gh()
        events = [{"event": "reopened", "created_at": "2024-06-01T00:00:00Z"}]
        mock_resp = MagicMock()
        mock_resp.json.return_value = events
        mock_resp.headers = {}
        mock_s.get.return_value = mock_resp
        result = gh.get_issue_events("o/r", 3)
        url = mock_s.get.call_args.args[0]
        assert "repos/o/r/issues/3/events" in url
        assert result == events

    def test_create_issue(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"html_url": "https://github.com/o/r/issues/42"}
        mock_s.post.return_value = mock_resp
        url = gh.create_issue("o/r", "My feature request", "body text")
        assert url == "https://github.com/o/r/issues/42"
        post_url = mock_s.post.call_args.args[0]
        assert "repos/o/r/issues" in post_url
        payload = mock_s.post.call_args.kwargs["json"]
        assert payload["title"] == "My feature request"
        assert payload["body"] == "body text"

    def test_create_pr_returns_url(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "html_url": "https://github.com/o/r/pull/10",
            "number": 10,
        }
        mock_s.post.return_value = mock_resp
        result = gh.create_pr("o/r", "title", "body", "main", "feat")
        url = mock_s.post.call_args.args[0]
        assert "repos/o/r/pulls" in url
        body = mock_s.post.call_args.kwargs["json"]
        assert body["draft"] is True
        assert body["title"] == "title"
        assert body["base"] == "main"
        assert body["head"] == "feat"
        assert result == "https://github.com/o/r/pull/10"

    def test_edit_pr_body(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_s.patch.return_value = mock_resp
        gh.edit_pr_body("o/r", 10, "new body")
        url = mock_s.patch.call_args.args[0]
        assert "repos/o/r/pulls/10" in url
        assert mock_s.patch.call_args.kwargs["json"]["body"] == "new body"

    def test_get_pr_body(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"body": "PR body text"}
        mock_s.get.return_value = mock_resp
        result = gh.get_pr_body("o/r", 10)
        url = mock_s.get.call_args.args[0]
        assert "repos/o/r/pulls/10" in url
        assert result == "PR body text"

    def test_get_pr_body_none_returns_empty_string(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"body": None}
        mock_s.get.return_value = mock_resp
        result = gh.get_pr_body("o/r", 10)
        assert result == ""

    def test_add_pr_reviewers(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_s.post.return_value = mock_resp
        gh.add_pr_reviewers("o/r", 10, ["rhencke", "alice"])
        url = mock_s.post.call_args.args[0]
        assert "repos/o/r/pulls/10/requested_reviewers" in url
        assert mock_s.post.call_args.kwargs["json"]["reviewers"] == [
            "rhencke",
            "alice",
        ]

    def test_add_pr_reviewers_empty_skips_post(self) -> None:
        gh, mock_s = self._gh()
        gh.add_pr_reviewers("o/r", 10, [])
        mock_s.post.assert_not_called()

    def test_pr_checks_returns_list(self) -> None:
        gh, mock_s = self._gh()
        pr_resp = MagicMock()
        pr_resp.json.return_value = {"head": {"sha": "abc123"}}
        checks_resp = MagicMock()
        checks_resp.json.return_value = {
            "check_runs": [
                {
                    "name": "ci",
                    "status": "completed",
                    "conclusion": "success",
                    "html_url": "http://...",
                },
            ]
        }
        mock_s.get.side_effect = [pr_resp, checks_resp]
        result = gh.pr_checks("o/r", 10)
        assert result == [{"name": "ci", "state": "SUCCESS", "link": "http://..."}]

    def test_pr_checks_in_progress(self) -> None:
        gh, mock_s = self._gh()
        pr_resp = MagicMock()
        pr_resp.json.return_value = {"head": {"sha": "abc123"}}
        checks_resp = MagicMock()
        checks_resp.json.return_value = {
            "check_runs": [
                {
                    "name": "build",
                    "status": "in_progress",
                    "conclusion": None,
                    "html_url": "http://...",
                },
            ]
        }
        mock_s.get.side_effect = [pr_resp, checks_resp]
        result = gh.pr_checks("o/r", 10)
        assert result[0]["state"] == "IN_PROGRESS"

    def test_get_required_checks_returns_context_names(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "required_status_checks": {
                "checks": [
                    {"context": "ci / test", "app_id": 1},
                    {"context": "ci / lint", "app_id": 1},
                ]
            }
        }
        mock_s.get.return_value = mock_resp
        result = gh.get_required_checks("o/r", "main")
        assert result == ["ci / test", "ci / lint"]

    def test_get_required_checks_no_protection_returns_empty(self) -> None:
        import requests

        gh, mock_s = self._gh()
        err_resp = MagicMock()
        err_resp.status_code = 404
        exc = requests.HTTPError(response=err_resp)
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = exc
        mock_s.get.return_value = mock_resp
        result = gh.get_required_checks("o/r", "main")
        assert result == []

    def test_get_required_checks_no_required_status_checks_returns_empty(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_s.get.return_value = mock_resp
        result = gh.get_required_checks("o/r", "main")
        assert result == []

    def test_get_required_checks_reraises_non_404(self) -> None:
        import requests

        gh, mock_s = self._gh()
        err_resp = MagicMock()
        err_resp.status_code = 500
        exc = requests.HTTPError(response=err_resp)
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = exc
        mock_s.get.return_value = mock_resp
        with pytest.raises(requests.HTTPError):
            gh.get_required_checks("o/r", "main")

    def test_pr_ready(self) -> None:
        gh, mock_s = self._gh()
        pr_resp = MagicMock()
        pr_resp.json.return_value = {"node_id": "PR_xyz"}
        graphql_resp = MagicMock()
        graphql_resp.json.return_value = {"data": {}}
        mock_s.get.return_value = pr_resp
        mock_s.post.return_value = graphql_resp
        gh.pr_ready("o/r", 10)
        body = mock_s.post.call_args.kwargs["json"]
        assert "markPullRequestReadyForReview" in body["query"]
        assert body["variables"]["prId"] == "PR_xyz"

    def test_pr_merge_squash(self) -> None:
        gh, mock_s = self._gh()
        get_resp = MagicMock()
        get_resp.json.return_value = {"merged": False, "node_id": "PR_abc"}
        mock_s.get.return_value = get_resp
        put_resp = MagicMock()
        put_resp.json.return_value = {}
        mock_s.put.return_value = put_resp
        gh.pr_merge("o/r", 10)
        url = mock_s.put.call_args.args[0]
        assert "repos/o/r/pulls/10/merge" in url
        assert mock_s.put.call_args.kwargs["json"]["merge_method"] == "squash"

    def test_pr_merge_no_squash(self) -> None:
        gh, mock_s = self._gh()
        get_resp = MagicMock()
        get_resp.json.return_value = {"merged": False, "node_id": "PR_abc"}
        mock_s.get.return_value = get_resp
        put_resp = MagicMock()
        put_resp.json.return_value = {}
        mock_s.put.return_value = put_resp
        gh.pr_merge("o/r", 10, squash=False)
        assert mock_s.put.call_args.kwargs["json"]["merge_method"] == "merge"

    def test_pr_merge_auto(self) -> None:
        gh, mock_s = self._gh()
        pr_resp = MagicMock()
        pr_resp.json.return_value = {"merged": False, "node_id": "PR_abc"}
        graphql_resp = MagicMock()
        graphql_resp.json.return_value = {"data": {}}
        mock_s.get.return_value = pr_resp
        mock_s.post.return_value = graphql_resp
        gh.pr_merge("o/r", 10, auto=True)
        body = mock_s.post.call_args.kwargs["json"]
        assert "enablePullRequestAutoMerge" in body["query"]
        assert body["variables"]["mergeMethod"] == "SQUASH"
        assert body["variables"]["prId"] == "PR_abc"

    def test_pr_merge_already_merged_returns_early(self) -> None:
        """If the PR is already merged, skip the merge call silently."""
        gh, mock_s = self._gh()
        get_resp = MagicMock()
        get_resp.json.return_value = {"merged": True, "node_id": "PR_abc"}
        mock_s.get.return_value = get_resp
        gh.pr_merge("o/r", 10)
        mock_s.put.assert_not_called()
        mock_s.post.assert_not_called()

    def test_pr_merge_already_merged_auto_returns_early(self) -> None:
        gh, mock_s = self._gh()
        get_resp = MagicMock()
        get_resp.json.return_value = {"merged": True, "node_id": "PR_abc"}
        mock_s.get.return_value = get_resp
        gh.pr_merge("o/r", 10, auto=True)
        mock_s.post.assert_not_called()

    def test_pr_merge_405_after_concurrent_merge_treated_as_success(self) -> None:
        """Race: get_pr returns not-merged, but PR merges between our check and
        the merge call.  405 response + recheck showing merged = treat as success."""
        import requests as _requests

        gh, mock_s = self._gh()
        get_responses = [
            MagicMock(
                json=MagicMock(return_value={"merged": False, "node_id": "PR_x"})
            ),
            MagicMock(json=MagicMock(return_value={"merged": True, "node_id": "PR_x"})),
        ]
        mock_s.get.side_effect = get_responses
        err = _requests.HTTPError("405 Method Not Allowed")
        err.response = MagicMock(status_code=405)
        put_resp = MagicMock()
        put_resp.raise_for_status.side_effect = err
        mock_s.put.return_value = put_resp
        # Should not raise — 405 + recheck shows merged.
        gh.pr_merge("o/r", 10)
        assert mock_s.get.call_count == 2

    def test_pr_merge_405_but_still_not_merged_reraises(self) -> None:
        """405 with recheck showing not-merged means something else is wrong;
        re-raise so the caller sees it."""
        import requests as _requests

        gh, mock_s = self._gh()
        get_responses = [
            MagicMock(
                json=MagicMock(return_value={"merged": False, "node_id": "PR_x"})
            ),
            MagicMock(
                json=MagicMock(return_value={"merged": False, "node_id": "PR_x"})
            ),
        ]
        mock_s.get.side_effect = get_responses
        err = _requests.HTTPError("405 Method Not Allowed")
        err.response = MagicMock(status_code=405)
        put_resp = MagicMock()
        put_resp.raise_for_status.side_effect = err
        mock_s.put.return_value = put_resp
        import pytest as _pytest

        with _pytest.raises(_requests.HTTPError):
            gh.pr_merge("o/r", 10)

    def test_pr_merge_auto_falls_back_on_auto_merge_disabled(self) -> None:
        """#643: if the repo has auto-merge disabled (GraphQL UNPROCESSABLE),
        fall back to an immediate squash merge rather than crashing."""
        gh, mock_s = self._gh()
        pr_resp = MagicMock()
        pr_resp.json.return_value = {"merged": False, "node_id": "PR_abc"}
        mock_s.get.return_value = pr_resp
        graphql_resp = MagicMock()
        graphql_resp.json.return_value = {
            "data": None,
            "errors": [
                {
                    "type": "UNPROCESSABLE",
                    "path": ["enablePullRequestAutoMerge"],
                    "message": "Pull request Auto merge is not allowed for this repository",
                }
            ],
        }
        mock_s.post.return_value = graphql_resp
        put_resp = MagicMock()
        put_resp.raise_for_status = MagicMock()
        mock_s.put.return_value = put_resp
        gh.pr_merge("o/r", 10, auto=True)
        mock_s.put.assert_called_once()
        # PUT body should request a squash merge via REST.
        call = mock_s.put.call_args
        assert "/pulls/10/merge" in call.args[0]
        assert call.kwargs["json"]["merge_method"] == "squash"

    def test_pr_merge_auto_non_dict_error_entries_skipped(self) -> None:
        """#643 helper ignores non-dict entries in GraphQLError.errors (GitHub's
        error list occasionally contains bare strings)."""
        from kennel.github import GraphQLError, _auto_merge_unavailable

        exc = GraphQLError(["string-error", {"type": "OTHER"}])
        assert _auto_merge_unavailable(exc) is False

    def test_pr_merge_auto_reraises_other_graphql_errors(self) -> None:
        """Non-\"auto-merge disabled\" GraphQL errors must not be swallowed."""
        from kennel.github import GraphQLError

        gh, mock_s = self._gh()
        pr_resp = MagicMock()
        pr_resp.json.return_value = {"merged": False, "node_id": "PR_abc"}
        mock_s.get.return_value = pr_resp
        graphql_resp = MagicMock()
        graphql_resp.json.return_value = {
            "data": None,
            "errors": [
                {
                    "type": "FORBIDDEN",
                    "message": "not allowed",
                }
            ],
        }
        mock_s.post.return_value = graphql_resp
        import pytest as _pytest

        with _pytest.raises(GraphQLError):
            gh.pr_merge("o/r", 10, auto=True)
        mock_s.put.assert_not_called()

    def test_pr_merge_non_405_error_reraises(self) -> None:
        """Non-405 errors (e.g. 500) should not be swallowed."""
        import requests as _requests

        gh, mock_s = self._gh()
        get_resp = MagicMock()
        get_resp.json.return_value = {"merged": False, "node_id": "PR_x"}
        mock_s.get.return_value = get_resp
        err = _requests.HTTPError("500 Internal Server Error")
        err.response = MagicMock(status_code=500)
        put_resp = MagicMock()
        put_resp.raise_for_status.side_effect = err
        mock_s.put.return_value = put_resp
        import pytest as _pytest

        with _pytest.raises(_requests.HTTPError):
            gh.pr_merge("o/r", 10)

    def test_get_pr_returns_dict(self) -> None:
        gh, mock_s = self._gh()
        pr_resp = MagicMock()
        pr_resp.json.return_value = {
            "draft": False,
            "mergeable_state": "clean",
            "body": "desc",
            "node_id": "PR_1",
        }
        reviews_resp = MagicMock()
        reviews_resp.json.return_value = [
            {
                "user": {"login": "alice"},
                "state": "APPROVED",
                "submitted_at": "2024-01-01T00:00:00Z",
                "body": "Looks good!",
            }
        ]
        reviews_resp.headers = {}
        commits_resp = MagicMock()
        commits_resp.json.return_value = [
            {
                "sha": "abc",
                "commit": {
                    "message": "Fix bug\n\nDetails",
                    "committer": {"date": "2024-01-02T00:00:00Z"},
                },
            }
        ]
        commits_resp.headers = {}
        mock_s.get.side_effect = [pr_resp, reviews_resp, commits_resp]
        result = gh.get_pr("o/r", 10)
        assert result["isDraft"] is False
        assert result["mergeStateStatus"] == "CLEAN"
        assert result["body"] == "desc"
        assert result["reviews"] == [
            {
                "author": {"login": "alice"},
                "state": "APPROVED",
                "submittedAt": "2024-01-01T00:00:00Z",
                "body": "Looks good!",
            }
        ]
        assert result["commits"] == [
            {
                "messageHeadline": "Fix bug",
                "oid": "abc",
                "committedDate": "2024-01-02T00:00:00Z",
            }
        ]

    def test_get_pr_null_body(self) -> None:
        gh, mock_s = self._gh()
        pr_resp = MagicMock()
        pr_resp.json.return_value = {
            "draft": True,
            "mergeable_state": None,
            "body": None,
        }
        reviews_resp = MagicMock()
        reviews_resp.json.return_value = []
        reviews_resp.headers = {}
        commits_resp = MagicMock()
        commits_resp.json.return_value = []
        commits_resp.headers = {}
        mock_s.get.side_effect = [pr_resp, reviews_resp, commits_resp]
        result = gh.get_pr("o/r", 10)
        assert result["body"] == ""
        assert result["mergeStateStatus"] == ""

    def test_get_review_threads(self) -> None:
        gh, mock_s = self._gh()
        nodes = [{"id": "T_1", "isResolved": False, "comments": {"nodes": []}}]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": nodes}}}}
        }
        mock_s.post.return_value = mock_resp
        result = gh.get_review_threads("o", "r", 10)
        assert result == nodes
        body = mock_s.post.call_args.kwargs["json"]
        assert body["variables"] == {"owner": "o", "repo": "r", "pr": 10}

    def test_resolve_thread(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {}}
        mock_s.post.return_value = mock_resp
        gh.resolve_thread("T_abc")
        body = mock_s.post.call_args.kwargs["json"]
        assert "resolveReviewThread" in body["query"]
        assert body["variables"] == {"id": "T_abc"}

    def test_is_thread_resolved_for_comment_resolved(self) -> None:
        gh, mock_s = self._gh()
        nodes = [
            {
                "isResolved": True,
                "comments": {"nodes": [{"databaseId": 42}, {"databaseId": 43}]},
            },
            {
                "isResolved": False,
                "comments": {"nodes": [{"databaseId": 99}]},
            },
        ]
        resp = MagicMock()
        resp.json.return_value = {
            "data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": nodes}}}}
        }
        mock_s.post.return_value = resp
        assert gh.is_thread_resolved_for_comment("o/r", 10, 43) is True

    def test_is_thread_resolved_for_comment_unresolved(self) -> None:
        gh, mock_s = self._gh()
        nodes = [
            {
                "isResolved": False,
                "comments": {"nodes": [{"databaseId": 99}]},
            },
        ]
        resp = MagicMock()
        resp.json.return_value = {
            "data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": nodes}}}}
        }
        mock_s.post.return_value = resp
        assert gh.is_thread_resolved_for_comment("o/r", 10, 99) is False

    def test_is_thread_resolved_for_comment_not_found(self) -> None:
        gh, mock_s = self._gh()
        nodes: list[dict[str, object]] = []
        resp = MagicMock()
        resp.json.return_value = {
            "data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": nodes}}}}
        }
        mock_s.post.return_value = resp
        assert gh.is_thread_resolved_for_comment("o/r", 10, 7777) is False

    def test_get_reviews_returns_dict(self) -> None:
        gh, mock_s = self._gh()
        pr_resp = MagicMock()
        pr_resp.json.return_value = {
            "draft": True,
            "requested_reviewers": [{"login": "alice"}],
        }
        reviews_resp = MagicMock()
        reviews_resp.json.return_value = [
            {
                "user": {"login": "bob"},
                "state": "CHANGES_REQUESTED",
                "submitted_at": "2024-01-01T00:00:00Z",
            }
        ]
        reviews_resp.headers = {}
        commits_resp = MagicMock()
        commits_resp.json.return_value = [
            {"commit": {"committer": {"date": "2024-01-02T00:00:00Z"}}}
        ]
        commits_resp.headers = {}
        mock_s.get.side_effect = [pr_resp, reviews_resp, commits_resp]
        result = gh.get_reviews("o/r", 10)
        assert result["isDraft"] is True
