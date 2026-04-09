from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kennel.github import (
    GH,
    GitHub,
    _get_gh,
    _gh_token,
    get_github,
)


def _completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=""
    )


@pytest.fixture(autouse=True)
def _reset_gh_cache() -> None:
    """Reset caches before each test for isolation."""
    _get_gh.cache_clear()
    get_github.cache_clear()


class TestGhToken:
    def test_uses_env_var(self) -> None:
        assert _gh_token(environ={"GITHUB_TOKEN": "mytoken"}) == "mytoken"

    def test_falls_back_to_gh_cli(self) -> None:
        mock_run = MagicMock(return_value=_completed("ghp_abc\n"))
        assert _gh_token(runner=mock_run, environ={}) == "ghp_abc"

    def test_gh_cli_strips_whitespace(self) -> None:
        mock_run = MagicMock(return_value=_completed("  tok  \n"))
        assert _gh_token(runner=mock_run, environ={}) == "tok"


class TestGetGh:
    def test_creates_instance_lazily(self) -> None:
        result = _get_gh(token="tok")
        assert isinstance(result, GH)
        assert result._s.headers["Authorization"] == "Bearer tok"

    def test_returns_cached_instance(self) -> None:
        first = _get_gh(token="tok")
        second = _get_gh(token="tok")
        assert first is second


class TestGetGithub:
    def test_creates_instance_lazily(self) -> None:
        result = get_github(token="tok")
        assert isinstance(result, GitHub)

    def test_returns_cached_instance(self) -> None:
        first = get_github(token="tok")
        second = get_github(token="tok")
        assert first is second


class TestGitHubClass:
    def _github(self) -> tuple[GitHub, MagicMock]:
        mock_s = MagicMock()
        return GitHub("test-token", session=mock_s), mock_s

    def test_stores_gh_as_attribute(self) -> None:
        gh = GitHub("test-token")
        assert isinstance(gh._gh, GH)

    def test_uses_provided_token(self) -> None:
        gh = GitHub("my-token")
        assert gh._gh._s.headers["Authorization"] == "Bearer my-token"

    def test_get_repo_info_delegates(self) -> None:
        gh, _ = self._github()
        mock_run = MagicMock(return_value=_completed("https://github.com/o/r.git\n"))
        assert gh.get_repo_info(runner=mock_run) == "o/r"

    def test_get_user_delegates(self) -> None:
        gh, mock_s = self._github()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"login": "fido"}
        mock_s.get.return_value = mock_resp
        assert gh.get_user() == "fido"

    def test_get_default_branch_delegates(self) -> None:
        gh, mock_s = self._github()
        remote_resp = _completed("https://github.com/o/r.git\n")
        repo_resp = MagicMock()
        repo_resp.json.return_value = {"default_branch": "main"}
        mock_run = MagicMock(return_value=remote_resp)
        mock_s.get.return_value = repo_resp
        assert gh.get_default_branch(runner=mock_run) == "main"

    def test_get_default_branch_passes_cwd(self) -> None:
        gh, mock_s = self._github()
        remote_resp = _completed("https://github.com/o/r.git\n")
        repo_resp = MagicMock()
        repo_resp.json.return_value = {"default_branch": "main"}
        mock_run = MagicMock(return_value=remote_resp)
        mock_s.get.return_value = repo_resp
        gh.get_default_branch(cwd=Path("/repo"), runner=mock_run)
        assert mock_run.call_args.kwargs["cwd"] == Path("/repo")

    def test_set_user_status_delegates(self) -> None:
        gh, mock_s = self._github()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_s.post.return_value = mock_resp
        gh.set_user_status("working", "🐕", busy=True)
        body = mock_s.post.call_args.kwargs["json"]
        assert body["variables"]["busy"] is True

    def test_find_issues_delegates(self) -> None:
        gh, mock_s = self._github()
        nodes = [{"number": 1, "title": "t"}]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": {"repository": {"issues": {"nodes": nodes}}}
        }
        mock_s.post.return_value = mock_resp
        assert gh.find_issues("o", "r", "fido") == nodes

    def test_view_issue_delegates(self) -> None:
        gh, mock_s = self._github()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "state": "open",
            "title": "T",
            "body": "b",
            "created_at": "2024-01-01T00:00:00Z",
        }
        mock_s.get.return_value = mock_resp
        result = gh.view_issue("o/r", 1)
        assert result["state"] == "OPEN"
        assert result["created_at"] == "2024-01-01T00:00:00Z"

    def test_comment_issue_delegates(self) -> None:
        gh, mock_s = self._github()
        mock_resp = MagicMock()
        mock_s.post.return_value = mock_resp
        gh.comment_issue("o/r", 7, "hi")
        assert "repos/o/r/issues/7/comments" in mock_s.post.call_args.args[0]

    def test_get_issue_comments_delegates(self) -> None:
        gh, mock_s = self._github()
        comments = [{"id": 1}]
        mock_resp = MagicMock()
        mock_resp.json.return_value = comments
        mock_resp.headers = {}
        mock_s.get.return_value = mock_resp
        assert gh.get_issue_comments("o/r", 9) == comments

    def test_get_issue_events_delegates(self) -> None:
        gh, mock_s = self._github()
        events = [{"event": "reopened", "created_at": "2024-06-01T00:00:00Z"}]
        mock_resp = MagicMock()
        mock_resp.json.return_value = events
        mock_resp.headers = {}
        mock_s.get.return_value = mock_resp
        assert gh.get_issue_events("o/r", 3) == events

    def test_create_issue_delegates(self) -> None:
        gh, mock_s = self._github()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"html_url": "https://github.com/o/r/issues/10"}
        mock_s.post.return_value = mock_resp
        url = gh.create_issue("o/r", "My suggestion", "some body")
        assert url == "https://github.com/o/r/issues/10"
        assert "repos/o/r/issues" in mock_s.post.call_args.args[0]

    def test_get_pull_comments_delegates(self) -> None:
        gh, mock_s = self._github()
        comments = [{"id": 42}]
        mock_resp = MagicMock()
        mock_resp.json.return_value = comments
        mock_resp.headers = {}
        mock_s.get.return_value = mock_resp
        assert gh.get_pull_comments("o/r", 7) == comments

    def test_fetch_sibling_threads_delegates(self) -> None:
        gh, mock_s = self._github()
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.headers = {}
        mock_s.get.return_value = mock_resp
        result = gh.fetch_sibling_threads("o/r", 7)
        assert result == []

    def test_find_pr_delegates(self) -> None:
        gh, mock_s = self._github()
        timeline_resp = MagicMock()
        timeline_resp.json.return_value = [
            {
                "event": "cross-referenced",
                "source": {
                    "issue": {
                        "number": 1,
                        "user": {"login": "fido"},
                        "body": "closes #5",
                        "pull_request": {},
                    }
                },
            }
        ]
        timeline_resp.headers = {}
        pr_resp = MagicMock()
        pr_resp.json.return_value = {
            "number": 1,
            "head": {"ref": "feat"},
            "state": "open",
            "merged": False,
            "user": {"login": "fido"},
        }
        mock_s.get.side_effect = [timeline_resp, pr_resp]
        result = gh.find_pr("o/r", 5, "fido")
        assert result is not None
        assert result["number"] == 1

    def test_create_pr_delegates(self) -> None:
        gh, mock_s = self._github()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"html_url": "https://github.com/o/r/pull/1"}
        mock_s.post.return_value = mock_resp
        result = gh.create_pr("o/r", "t", "b", "main", "feat")
        assert result == "https://github.com/o/r/pull/1"

    def test_edit_pr_body_delegates(self) -> None:
        gh, mock_s = self._github()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_s.patch.return_value = mock_resp
        gh.edit_pr_body("o/r", 10, "new")
        assert mock_s.patch.call_args.kwargs["json"]["body"] == "new"

    def test_get_pr_body_delegates(self) -> None:
        gh, mock_s = self._github()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"body": "some body"}
        mock_s.get.return_value = mock_resp
        result = gh.get_pr_body("o/r", 10)
        assert result == "some body"

    def test_add_pr_reviewer_delegates(self) -> None:
        gh, mock_s = self._github()
        mock_resp = MagicMock()
        mock_s.post.return_value = mock_resp
        gh.add_pr_reviewer("o/r", 10, "alice")
        assert mock_s.post.call_args.kwargs["json"]["reviewers"] == ["alice"]

    def test_pr_checks_delegates(self) -> None:
        gh, mock_s = self._github()
        pr_resp = MagicMock()
        pr_resp.json.return_value = {"head": {"sha": "abc"}}
        checks_resp = MagicMock()
        checks_resp.json.return_value = {"check_runs": []}
        mock_s.get.side_effect = [pr_resp, checks_resp]
        assert gh.pr_checks("o/r", 10) == []

    def test_get_required_checks_delegates(self) -> None:
        gh, mock_s = self._github()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "required_status_checks": {
                "checks": [{"context": "ci / test", "app_id": 1}]
            }
        }
        mock_s.get.return_value = mock_resp
        result = gh.get_required_checks("o/r", "main")
        assert result == ["ci / test"]

    def test_pr_ready_delegates(self) -> None:
        gh, mock_s = self._github()
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

    def test_pr_merge_delegates(self) -> None:
        gh, mock_s = self._github()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_s.put.return_value = mock_resp
        gh.pr_merge("o/r", 10)
        assert mock_s.put.call_args.kwargs["json"]["merge_method"] == "squash"

    def test_get_pr_delegates(self) -> None:
        gh, mock_s = self._github()
        pr_resp = MagicMock()
        pr_resp.json.return_value = {
            "draft": False,
            "mergeable_state": "clean",
            "body": "b",
        }
        reviews_resp = MagicMock()
        reviews_resp.json.return_value = []
        reviews_resp.headers = {}
        commits_resp = MagicMock()
        commits_resp.json.return_value = []
        commits_resp.headers = {}
        mock_s.get.side_effect = [pr_resp, reviews_resp, commits_resp]
        result = gh.get_pr("o/r", 10)
        assert result["isDraft"] is False

    def test_get_reviews_delegates(self) -> None:
        gh, mock_s = self._github()
        pr_resp = MagicMock()
        pr_resp.json.return_value = {"draft": True}
        reviews_resp = MagicMock()
        reviews_resp.json.return_value = []
        reviews_resp.headers = {}
        commits_resp = MagicMock()
        commits_resp.json.return_value = []
        commits_resp.headers = {}
        mock_s.get.side_effect = [pr_resp, reviews_resp, commits_resp]
        result = gh.get_reviews("o/r", 10)
        assert result["isDraft"] is True

    def test_get_review_comments_delegates(self) -> None:
        gh, mock_s = self._github()
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"id": 1, "body": "fix this"},
            {"id": 2, "body": "nit"},
        ]
        mock_resp.headers = {}
        mock_s.get.return_value = mock_resp
        assert gh.get_review_comments("o/r", 10, 99) == [
            (1, "fix this"),
            (2, "nit"),
        ]

    def test_reply_to_review_comment_delegates(self) -> None:
        gh, mock_s = self._github()
        mock_resp = MagicMock()
        mock_s.post.return_value = mock_resp
        gh.reply_to_review_comment("o/r", 10, "lgtm", 55)
        assert mock_s.post.call_args.kwargs["json"]["body"] == "lgtm"

    def test_add_reaction_delegates(self) -> None:
        gh, mock_s = self._github()
        mock_resp = MagicMock()
        mock_s.post.return_value = mock_resp
        gh.add_reaction("o/r", "pulls", 42, "rocket")
        assert mock_s.post.call_args.kwargs["json"]["content"] == "rocket"

    def test_get_review_threads_delegates(self) -> None:
        gh, mock_s = self._github()
        payload = {
            "data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": []}}}}
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = payload
        mock_s.post.return_value = mock_resp
        result = gh.get_review_threads("o", "r", 10)
        assert result == payload

    def test_resolve_thread_delegates(self) -> None:
        gh, mock_s = self._github()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {}}
        mock_s.post.return_value = mock_resp
        gh.resolve_thread("T_abc")
        assert "resolveReviewThread" in mock_s.post.call_args.kwargs["json"]["query"]

    def test_get_run_log_delegates(self) -> None:
        gh, mock_s = self._github()
        jobs_resp = MagicMock()
        jobs_resp.json.return_value = {"jobs": [{"id": 1, "conclusion": "failure"}]}
        log_resp = MagicMock()
        log_resp.text = "log\n"
        mock_s.get.side_effect = [jobs_resp, log_resp]
        assert gh.get_run_log("o/r", 1) == "log\n"


class TestGHClass:
    def _gh(self) -> tuple[GH, MagicMock]:
        mock_s = MagicMock()
        return GH("test-token", session=mock_s), mock_s

    def test_sets_auth_header(self) -> None:
        gh = GH("test-token")
        assert gh._s.headers["Authorization"] == "Bearer test-token"

    def test_sets_accept_header(self) -> None:
        gh = GH("test-token")
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

    def test_graphql_raises_on_error(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("500")
        mock_s.post.return_value = mock_resp
        try:
            gh._graphql("query {}")
            assert False, "should have raised"
        except Exception as e:
            assert "500" in str(e)

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

    def _timeline_resp(self, events: list[dict]) -> MagicMock:
        """Build a mock timeline response with no next-page Link header."""
        resp = MagicMock()
        resp.json.return_value = events
        resp.headers = {}
        return resp

    def _cross_ref_event(
        self, pr_number: int, user: str, body: str, has_pull_request: bool = True
    ) -> dict:
        source_issue: dict = {
            "number": pr_number,
            "user": {"login": user},
            "body": body,
        }
        if has_pull_request:
            source_issue["pull_request"] = {}
        return {"event": "cross-referenced", "source": {"issue": source_issue}}

    def _pr_resp(
        self, number: int, ref: str, state: str, merged: bool, user: str
    ) -> MagicMock:
        resp = MagicMock()
        resp.json.return_value = {
            "number": number,
            "head": {"ref": ref},
            "state": state,
            "merged": merged,
            "user": {"login": user},
        }
        return resp

    def test_find_pr_returns_match(self) -> None:
        gh, mock_s = self._gh()
        timeline = self._timeline_resp([self._cross_ref_event(1, "fido", "closes #5")])
        pr = self._pr_resp(1, "feat", "open", False, "fido")
        mock_s.get.side_effect = [timeline, pr]
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
        timeline = self._timeline_resp([self._cross_ref_event(3, "fido", "closes #2")])
        pr = self._pr_resp(3, "fix", "closed", True, "fido")
        mock_s.get.side_effect = [timeline, pr]
        result = gh.find_pr("o/r", 2, "fido")
        assert result is None

    def test_find_pr_skips_closed(self) -> None:
        """Closed (not merged) PRs are skipped — a reopened issue should get a fresh PR."""
        gh, mock_s = self._gh()
        timeline = self._timeline_resp([self._cross_ref_event(4, "fido", "closes #2")])
        pr = self._pr_resp(4, "fix", "closed", False, "fido")
        mock_s.get.side_effect = [timeline, pr]
        result = gh.find_pr("o/r", 2, "fido")
        assert result is None

    def test_find_pr_skips_non_open_returns_subsequent_open_pr(self) -> None:
        """When a merged/closed PR and a later open PR reference the issue, returns open."""
        gh, mock_s = self._gh()
        timeline = self._timeline_resp(
            [
                self._cross_ref_event(3, "fido", "closes #2"),
                self._cross_ref_event(7, "fido", "closes #2"),
            ]
        )
        merged_pr = self._pr_resp(3, "fix", "closed", True, "fido")
        open_pr = self._pr_resp(7, "fix-retry", "open", False, "fido")
        mock_s.get.side_effect = [timeline, merged_pr, open_pr]
        result = gh.find_pr("o/r", 2, "fido")
        assert result is not None
        assert result["number"] == 7
        assert result["state"] == "OPEN"

    def test_find_pr_filters_by_user(self) -> None:
        gh, mock_s = self._gh()
        timeline = self._timeline_resp([self._cross_ref_event(1, "other", "closes #5")])
        mock_s.get.return_value = timeline
        assert gh.find_pr("o/r", 5, "fido") is None

    def test_find_pr_returns_none_on_empty(self) -> None:
        gh, mock_s = self._gh()
        timeline = self._timeline_resp([])
        mock_s.get.return_value = timeline
        assert gh.find_pr("o/r", 1, "fido") is None

    def test_find_pr_skips_non_pr_cross_reference(self) -> None:
        """Cross-referenced events from plain issues (not PRs) are ignored."""
        gh, mock_s = self._gh()
        event = self._cross_ref_event(9, "fido", "closes #5", has_pull_request=False)
        timeline = self._timeline_resp([event])
        mock_s.get.return_value = timeline
        assert gh.find_pr("o/r", 5, "fido") is None

    def test_find_pr_skips_substring_match(self) -> None:
        """#9 must not match a PR body that only contains #90."""
        gh, mock_s = self._gh()
        timeline = self._timeline_resp(
            [self._cross_ref_event(99, "fido", "closes #90")]
        )
        mock_s.get.return_value = timeline
        assert gh.find_pr("o/r", 9, "fido") is None

    def test_find_pr_skips_prefix_match(self) -> None:
        """#100 must not match a PR body that only contains closes #10."""
        gh, mock_s = self._gh()
        timeline = self._timeline_resp(
            [self._cross_ref_event(99, "fido", "closes #10")]
        )
        mock_s.get.return_value = timeline
        assert gh.find_pr("o/r", 100, "fido") is None

    def test_find_pr_requires_closing_keyword(self) -> None:
        """Bare #N in PR body (no closing keyword) must not match."""
        gh, mock_s = self._gh()
        timeline = self._timeline_resp(
            [self._cross_ref_event(7, "fido", "see #5 for context")]
        )
        mock_s.get.return_value = timeline
        assert gh.find_pr("o/r", 5, "fido") is None

    def test_find_pr_body_only_not_title(self) -> None:
        """Issue reference in PR title alone does not count — body only."""
        gh, mock_s = self._gh()
        event = self._cross_ref_event(7, "fido", "")  # empty body
        # Inject a title field to simulate a PR where #5 appears only in the title
        event["source"]["issue"]["title"] = "closes #5"
        timeline = self._timeline_resp([event])
        mock_s.get.return_value = timeline
        assert gh.find_pr("o/r", 5, "fido") is None

    def test_find_pr_timeline_url(self) -> None:
        gh, mock_s = self._gh()
        timeline = self._timeline_resp([])
        mock_s.get.return_value = timeline
        gh.find_pr("o/r", 5, "fido")
        url = mock_s.get.call_args.args[0]
        assert "/repos/o/r/issues/5/timeline" in url

    def test_find_pr_follows_pagination(self) -> None:
        """find_pr fetches additional pages when a Link: next header is present."""
        gh, mock_s = self._gh()
        page1 = MagicMock()
        page1.json.return_value = [{"event": "committed"}]  # no match on page 1
        next_url = "https://api.github.com/repos/o/r/issues/5/timeline?page=2"
        page1.headers = {"Link": f'<{next_url}>; rel="next"'}
        page2 = self._timeline_resp([self._cross_ref_event(7, "fido", "closes #5")])
        pr = self._pr_resp(7, "feat", "open", False, "fido")
        mock_s.get.side_effect = [page1, page2, pr]
        result = gh.find_pr("o/r", 5, "fido")
        assert result is not None
        assert result["number"] == 7

    def test_get_user(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"login": "fido", "id": 1}
        mock_s.get.return_value = mock_resp
        result = gh.get_user()
        url = mock_s.get.call_args.args[0]
        assert url.endswith("/user")
        assert result == "fido"

    def test_get_repo_info_https(self) -> None:
        gh = GH("test-token")
        mock_run = MagicMock(
            return_value=_completed("https://github.com/owner/repo.git\n")
        )
        assert gh.get_repo_info(runner=mock_run) == "owner/repo"

    def test_get_repo_info_ssh(self) -> None:
        gh = GH("test-token")
        mock_run = MagicMock(return_value=_completed("git@github.com:owner/repo.git\n"))
        assert gh.get_repo_info(runner=mock_run) == "owner/repo"

    def test_get_repo_info_passes_cwd(self) -> None:
        gh = GH("test-token")
        mock_run = MagicMock(return_value=_completed("https://github.com/o/r.git"))
        gh.get_repo_info(cwd="/tmp/repo", runner=mock_run)
        assert mock_run.call_args.kwargs["cwd"] == "/tmp/repo"

    def test_get_repo_info_raises_unknown(self) -> None:
        gh = GH("test-token")
        mock_run = MagicMock(return_value=_completed("https://example.com/repo.git"))
        with pytest.raises(ValueError, match="Cannot parse"):
            gh.get_repo_info(runner=mock_run)

    def test_get_default_branch(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"default_branch": "main", "name": "repo"}
        mock_s.get.return_value = mock_resp
        result = gh.get_default_branch("o/r")
        url = mock_s.get.call_args.args[0]
        assert "repos/o/r" in url
        assert result == "main"

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

    def test_find_issues_graphql(self) -> None:
        gh, mock_s = self._gh()
        nodes = [{"number": 1, "title": "bug"}]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": {"repository": {"issues": {"nodes": nodes}}}
        }
        mock_s.post.return_value = mock_resp
        result = gh.find_issues("owner", "repo", "fido")
        body = mock_s.post.call_args.kwargs["json"]
        assert body["variables"] == {"owner": "owner", "repo": "repo", "login": "fido"}
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

    def test_add_pr_reviewer(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_s.post.return_value = mock_resp
        gh.add_pr_reviewer("o/r", 10, "rhencke")
        url = mock_s.post.call_args.args[0]
        assert "repos/o/r/pulls/10/requested_reviewers" in url
        assert mock_s.post.call_args.kwargs["json"]["reviewers"] == ["rhencke"]

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
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_s.put.return_value = mock_resp
        gh.pr_merge("o/r", 10)
        url = mock_s.put.call_args.args[0]
        assert "repos/o/r/pulls/10/merge" in url
        assert mock_s.put.call_args.kwargs["json"]["merge_method"] == "squash"

    def test_pr_merge_no_squash(self) -> None:
        gh, mock_s = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_s.put.return_value = mock_resp
        gh.pr_merge("o/r", 10, squash=False)
        assert mock_s.put.call_args.kwargs["json"]["merge_method"] == "merge"

    def test_pr_merge_auto(self) -> None:
        gh, mock_s = self._gh()
        pr_resp = MagicMock()
        pr_resp.json.return_value = {"node_id": "PR_abc"}
        graphql_resp = MagicMock()
        graphql_resp.json.return_value = {"data": {}}
        mock_s.get.return_value = pr_resp
        mock_s.post.return_value = graphql_resp
        gh.pr_merge("o/r", 10, auto=True)
        body = mock_s.post.call_args.kwargs["json"]
        assert "enablePullRequestAutoMerge" in body["query"]
        assert body["variables"]["mergeMethod"] == "SQUASH"
        assert body["variables"]["prId"] == "PR_abc"

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
