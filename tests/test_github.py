from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    """Reset caches before each test and seed with real instances."""
    _get_gh.cache_clear()
    get_github.cache_clear()
    with patch("kennel.github._gh_token", return_value="test-token"):
        _get_gh()
        get_github()


class TestGhToken:
    def test_uses_env_var(self) -> None:
        with patch("os.environ.get", return_value="mytoken"):
            assert _gh_token() == "mytoken"

    def test_falls_back_to_gh_cli(self) -> None:
        with (
            patch("os.environ.get", return_value=""),
            patch("subprocess.run", return_value=_completed("ghp_abc\n")),
        ):
            assert _gh_token() == "ghp_abc"

    def test_gh_cli_strips_whitespace(self) -> None:
        with (
            patch("os.environ.get", return_value=""),
            patch("subprocess.run", return_value=_completed("  tok  \n")),
        ):
            assert _gh_token() == "tok"


class TestGetGh:
    def test_creates_instance_lazily(self) -> None:
        mock_instance = MagicMock()
        _get_gh.cache_clear()
        with (
            patch("kennel.github._gh_token", return_value="tok"),
            patch("kennel.github.GH", return_value=mock_instance) as mock_cls,
        ):
            result = _get_gh()
        mock_cls.assert_called_once_with("tok")
        assert result is mock_instance

    def test_returns_cached_instance(self) -> None:
        mock_instance = MagicMock()
        _get_gh.cache_clear()
        with (
            patch("kennel.github._gh_token", return_value="tok"),
            patch("kennel.github.GH", return_value=mock_instance) as mock_cls,
        ):
            first = _get_gh()
            second = _get_gh()
        mock_cls.assert_called_once()
        assert first is second is mock_instance


class TestGetGithub:
    def test_creates_instance_lazily(self) -> None:
        mock_instance = MagicMock()
        get_github.cache_clear()
        with (
            patch("kennel.github._gh_token", return_value="tok"),
            patch("kennel.github.GitHub", return_value=mock_instance) as mock_cls,
        ):
            result = get_github()
        mock_cls.assert_called_once_with()
        assert result is mock_instance

    def test_returns_cached_instance(self) -> None:
        mock_instance = MagicMock()
        get_github.cache_clear()
        with (
            patch("kennel.github._gh_token", return_value="tok"),
            patch("kennel.github.GitHub", return_value=mock_instance) as mock_cls,
        ):
            first = get_github()
            second = get_github()
        mock_cls.assert_called_once()
        assert first is second is mock_instance


class TestGitHubClass:
    def _github(self) -> GitHub:
        with patch("kennel.github._gh_token", return_value="test-token"):
            return GitHub()

    def test_stores_gh_as_attribute(self) -> None:
        gh = self._github()
        assert isinstance(gh._gh, GH)

    def test_uses_provided_token(self) -> None:
        gh = GitHub("my-token")
        assert gh._gh._s.headers["Authorization"] == "Bearer my-token"

    def test_get_repo_info_delegates(self) -> None:
        gh = self._github()
        with patch(
            "subprocess.run",
            return_value=_completed("https://github.com/o/r.git\n"),
        ):
            assert gh.get_repo_info() == "o/r"

    def test_get_user_delegates(self) -> None:
        gh = self._github()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"login": "fido"}
        with patch.object(gh._gh._s, "get", return_value=mock_resp):
            assert gh.get_user() == "fido"

    def test_get_default_branch_delegates(self) -> None:
        gh = self._github()
        remote_resp = _completed("https://github.com/o/r.git\n")
        repo_resp = MagicMock()
        repo_resp.json.return_value = {"default_branch": "main"}
        with (
            patch("subprocess.run", return_value=remote_resp),
            patch.object(gh._gh._s, "get", return_value=repo_resp),
        ):
            assert gh.get_default_branch() == "main"

    def test_get_default_branch_passes_cwd(self) -> None:
        gh = self._github()
        remote_resp = _completed("https://github.com/o/r.git\n")
        repo_resp = MagicMock()
        repo_resp.json.return_value = {"default_branch": "main"}
        with (
            patch("subprocess.run", return_value=remote_resp) as mock_sub,
            patch.object(gh._gh._s, "get", return_value=repo_resp),
        ):
            gh.get_default_branch(cwd=Path("/repo"))
        assert mock_sub.call_args.kwargs["cwd"] == Path("/repo")

    def test_set_user_status_delegates(self) -> None:
        gh = self._github()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        with patch.object(gh._gh._s, "post", return_value=mock_resp) as mock_post:
            gh.set_user_status("working", "🐕", busy=True)
        body = mock_post.call_args.kwargs["json"]
        assert body["variables"]["busy"] is True

    def test_find_issues_delegates(self) -> None:
        gh = self._github()
        nodes = [{"number": 1, "title": "t"}]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": {"repository": {"issues": {"nodes": nodes}}}
        }
        with patch.object(gh._gh._s, "post", return_value=mock_resp):
            assert gh.find_issues("o", "r", "fido") == nodes

    def test_view_issue_delegates(self) -> None:
        gh = self._github()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"state": "open", "title": "T", "body": "b"}
        with patch.object(gh._gh._s, "get", return_value=mock_resp):
            result = gh.view_issue("o/r", 1)
        assert result["state"] == "OPEN"

    def test_comment_issue_delegates(self) -> None:
        gh = self._github()
        mock_resp = MagicMock()
        with patch.object(gh._gh._s, "post", return_value=mock_resp) as mock_post:
            gh.comment_issue("o/r", 7, "hi")
        assert "repos/o/r/issues/7/comments" in mock_post.call_args.args[0]

    def test_get_issue_comments_delegates(self) -> None:
        gh = self._github()
        comments = [{"id": 1}]
        mock_resp = MagicMock()
        mock_resp.json.return_value = comments
        with patch.object(gh._gh._s, "get", return_value=mock_resp):
            assert gh.get_issue_comments("o/r", 9) == comments

    def test_create_issue_delegates(self) -> None:
        gh = self._github()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"html_url": "https://github.com/o/r/issues/10"}
        with patch.object(gh._gh._s, "post", return_value=mock_resp) as mock_post:
            url = gh.create_issue("o/r", "My suggestion", "some body")
        assert url == "https://github.com/o/r/issues/10"
        assert "repos/o/r/issues" in mock_post.call_args.args[0]

    def test_get_pull_comments_delegates(self) -> None:
        gh = self._github()
        comments = [{"id": 42}]
        mock_resp = MagicMock()
        mock_resp.json.return_value = comments
        with patch.object(gh._gh._s, "get", return_value=mock_resp):
            assert gh.get_pull_comments("o/r", 7) == comments

    def test_fetch_sibling_threads_delegates(self) -> None:
        gh = self._github()
        with patch.object(gh._gh, "fetch_sibling_threads", return_value=[]) as mock_m:
            result = gh.fetch_sibling_threads("o/r", 7)
        mock_m.assert_called_once_with("o/r", 7)
        assert result == []

    def test_find_pr_delegates(self) -> None:
        gh = self._github()
        search_resp = MagicMock()
        search_resp.json.return_value = {
            "items": [{"number": 1, "user": {"login": "fido"}}]
        }
        pr_resp = MagicMock()
        pr_resp.json.return_value = {
            "number": 1,
            "head": {"ref": "feat"},
            "state": "open",
            "merged": False,
            "user": {"login": "fido"},
        }
        with patch.object(gh._gh._s, "get", side_effect=[search_resp, pr_resp]):
            result = gh.find_pr("o/r", 5, "fido")
        assert result is not None
        assert result["number"] == 1

    def test_create_pr_delegates(self) -> None:
        gh = self._github()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"html_url": "https://github.com/o/r/pull/1"}
        with patch.object(gh._gh._s, "post", return_value=mock_resp):
            result = gh.create_pr("o/r", "t", "b", "main", "feat")
        assert result == "https://github.com/o/r/pull/1"

    def test_edit_pr_body_delegates(self) -> None:
        gh = self._github()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        with patch.object(gh._gh._s, "patch", return_value=mock_resp) as mock_patch:
            gh.edit_pr_body("o/r", 10, "new")
        assert mock_patch.call_args.kwargs["json"]["body"] == "new"

    def test_get_pr_body_delegates(self) -> None:
        gh = self._github()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"body": "some body"}
        with patch.object(gh._gh._s, "get", return_value=mock_resp):
            result = gh.get_pr_body("o/r", 10)
        assert result == "some body"

    def test_add_pr_reviewer_delegates(self) -> None:
        gh = self._github()
        mock_resp = MagicMock()
        with patch.object(gh._gh._s, "post", return_value=mock_resp) as mock_post:
            gh.add_pr_reviewer("o/r", 10, "alice")
        assert mock_post.call_args.kwargs["json"]["reviewers"] == ["alice"]

    def test_pr_checks_delegates(self) -> None:
        gh = self._github()
        pr_resp = MagicMock()
        pr_resp.json.return_value = {"head": {"sha": "abc"}}
        checks_resp = MagicMock()
        checks_resp.json.return_value = {"check_runs": []}
        with patch.object(gh._gh._s, "get", side_effect=[pr_resp, checks_resp]):
            assert gh.pr_checks("o/r", 10) == []

    def test_pr_ready_delegates(self) -> None:
        gh = self._github()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        with patch.object(gh._gh._s, "patch", return_value=mock_resp) as mock_patch:
            gh.pr_ready("o/r", 10)
        assert mock_patch.call_args.kwargs["json"]["draft"] is False

    def test_pr_merge_delegates(self) -> None:
        gh = self._github()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        with patch.object(gh._gh._s, "put", return_value=mock_resp) as mock_put:
            gh.pr_merge("o/r", 10)
        assert mock_put.call_args.kwargs["json"]["merge_method"] == "squash"

    def test_get_pr_delegates(self) -> None:
        gh = self._github()
        pr_resp = MagicMock()
        pr_resp.json.return_value = {
            "draft": False,
            "mergeable_state": "clean",
            "body": "b",
        }
        reviews_resp = MagicMock()
        reviews_resp.json.return_value = []
        commits_resp = MagicMock()
        commits_resp.json.return_value = []
        with patch.object(
            gh._gh._s, "get", side_effect=[pr_resp, reviews_resp, commits_resp]
        ):
            result = gh.get_pr("o/r", 10)
        assert result["isDraft"] is False

    def test_get_reviews_delegates(self) -> None:
        gh = self._github()
        pr_resp = MagicMock()
        pr_resp.json.return_value = {"draft": True}
        reviews_resp = MagicMock()
        reviews_resp.json.return_value = []
        commits_resp = MagicMock()
        commits_resp.json.return_value = []
        with patch.object(
            gh._gh._s, "get", side_effect=[pr_resp, reviews_resp, commits_resp]
        ):
            result = gh.get_reviews("o/r", 10)
        assert result["isDraft"] is True

    def test_get_review_comments_delegates(self) -> None:
        gh = self._github()
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"id": 1, "body": "fix this"},
            {"id": 2, "body": "nit"},
        ]
        with patch.object(gh._gh._s, "get", return_value=mock_resp):
            assert gh.get_review_comments("o/r", 10, 99) == [
                (1, "fix this"),
                (2, "nit"),
            ]

    def test_reply_to_review_comment_delegates(self) -> None:
        gh = self._github()
        mock_resp = MagicMock()
        with patch.object(gh._gh._s, "post", return_value=mock_resp) as mock_post:
            gh.reply_to_review_comment("o/r", 10, "lgtm", 55)
        assert mock_post.call_args.kwargs["json"]["body"] == "lgtm"

    def test_add_reaction_delegates(self) -> None:
        gh = self._github()
        mock_resp = MagicMock()
        with patch.object(gh._gh._s, "post", return_value=mock_resp) as mock_post:
            gh.add_reaction("o/r", "pulls", 42, "rocket")
        assert mock_post.call_args.kwargs["json"]["content"] == "rocket"

    def test_get_review_threads_delegates(self) -> None:
        gh = self._github()
        payload = {
            "data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": []}}}}
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = payload
        with patch.object(gh._gh._s, "post", return_value=mock_resp):
            result = gh.get_review_threads("o", "r", 10)
        assert result == payload

    def test_resolve_thread_delegates(self) -> None:
        gh = self._github()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {}}
        with patch.object(gh._gh._s, "post", return_value=mock_resp) as mock_post:
            gh.resolve_thread("T_abc")
        assert "resolveReviewThread" in mock_post.call_args.kwargs["json"]["query"]

    def test_get_run_log_delegates(self) -> None:
        gh = self._github()
        jobs_resp = MagicMock()
        jobs_resp.json.return_value = {"jobs": [{"id": 1, "conclusion": "failure"}]}
        log_resp = MagicMock()
        log_resp.text = "log\n"
        with patch.object(gh._gh._s, "get", side_effect=[jobs_resp, log_resp]):
            assert gh.get_run_log("o/r", 1) == "log\n"


class TestGHClass:
    def _gh(self) -> GH:
        return GH("test-token")

    def test_sets_auth_header(self) -> None:
        gh = self._gh()
        assert gh._s.headers["Authorization"] == "Bearer test-token"

    def test_sets_accept_header(self) -> None:
        gh = self._gh()
        assert gh._s.headers["Accept"] == "application/vnd.github+json"

    def test_get_calls_session(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"id": 1}]
        with patch.object(gh._s, "get", return_value=mock_resp) as mock_get:
            result = gh._get("/repos/o/r/issues")
        mock_get.assert_called_once_with("https://api.github.com/repos/o/r/issues")
        assert result == [{"id": 1}]

    def test_get_raises_on_error(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("404")
        with patch.object(gh._s, "get", return_value=mock_resp):
            try:
                gh._get("/bad")
                assert False, "should have raised"
            except Exception as e:
                assert "404" in str(e)

    def test_post_calls_session(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        with patch.object(gh._s, "post", return_value=mock_resp) as mock_post:
            gh._post("/repos/o/r/issues/1/comments", body="hi")
        mock_post.assert_called_once_with(
            "https://api.github.com/repos/o/r/issues/1/comments",
            json={"body": "hi"},
        )

    def test_post_raises_on_error(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("422")
        with patch.object(gh._s, "post", return_value=mock_resp):
            try:
                gh._post("/bad")
                assert False, "should have raised"
            except Exception as e:
                assert "422" in str(e)

    def test_post_json_returns_response(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "id": 42,
            "html_url": "https://github.com/o/r/pull/42",
        }
        with patch.object(gh._s, "post", return_value=mock_resp):
            result = gh._post_json("/repos/o/r/pulls", title="t", body="b")
        assert result["id"] == 42

    def test_patch_calls_session(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        with patch.object(gh._s, "patch", return_value=mock_resp) as mock_patch:
            gh._patch("/repos/o/r/issues/1", state="closed")
        mock_patch.assert_called_once_with(
            "https://api.github.com/repos/o/r/issues/1",
            json={"state": "closed"},
        )

    def test_patch_raises_on_error(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("404")
        with patch.object(gh._s, "patch", return_value=mock_resp):
            try:
                gh._patch("/bad")
                assert False, "should have raised"
            except Exception as e:
                assert "404" in str(e)

    def test_put_calls_session(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        with patch.object(gh._s, "put", return_value=mock_resp) as mock_put:
            gh._put("/repos/o/r/pulls/1/merge", merge_method="squash")
        mock_put.assert_called_once_with(
            "https://api.github.com/repos/o/r/pulls/1/merge",
            json={"merge_method": "squash"},
        )

    def test_put_raises_on_error(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("405")
        with patch.object(gh._s, "put", return_value=mock_resp):
            try:
                gh._put("/bad")
                assert False, "should have raised"
            except Exception as e:
                assert "405" in str(e)

    def test_graphql_posts_to_graphql_endpoint(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {}}
        with patch.object(gh._s, "post", return_value=mock_resp) as mock_post:
            result = gh._graphql("query { viewer { login } }", login="fido")
        url = mock_post.call_args.args[0]
        assert url == "https://api.github.com/graphql"
        body = mock_post.call_args.kwargs["json"]
        assert body["query"] == "query { viewer { login } }"
        assert body["variables"] == {"login": "fido"}
        assert result == {"data": {}}

    def test_graphql_raises_on_error(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("500")
        with patch.object(gh._s, "post", return_value=mock_resp):
            try:
                gh._graphql("query {}")
                assert False, "should have raised"
            except Exception as e:
                assert "500" in str(e)

    def test_add_reaction_pulls(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        with patch.object(gh._s, "post", return_value=mock_resp) as mock_post:
            gh.add_reaction("o/r", "pulls", 42, "rocket")
        url = mock_post.call_args.args[0]
        assert "repos/o/r/pulls/comments/42/reactions" in url
        assert mock_post.call_args.kwargs["json"]["content"] == "rocket"

    def test_add_reaction_issues(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        with patch.object(gh._s, "post", return_value=mock_resp) as mock_post:
            gh.add_reaction("o/r", "issues", 7, "+1")
        url = mock_post.call_args.args[0]
        assert "repos/o/r/issues/comments/7/reactions" in url

    def test_reply_to_review_comment(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        with patch.object(gh._s, "post", return_value=mock_resp) as mock_post:
            gh.reply_to_review_comment("o/r", 10, "lgtm", 55)
        url = mock_post.call_args.args[0]
        assert "repos/o/r/pulls/10/comments" in url
        body = mock_post.call_args.kwargs["json"]
        assert body["body"] == "lgtm"
        assert body["in_reply_to"] == 55

    def test_reply_to_review_comment_converts_in_reply_to(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        with patch.object(gh._s, "post", return_value=mock_resp) as mock_post:
            gh.reply_to_review_comment("o/r", 10, "ok", "99")
        body = mock_post.call_args.kwargs["json"]
        assert body["in_reply_to"] == 99

    def test_get_pull_comments(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        comments = [{"id": 42, "body": "looks good"}]
        mock_resp.json.return_value = comments
        with patch.object(gh._s, "get", return_value=mock_resp):
            result = gh.get_pull_comments("o/r", 7)
        assert result == comments

    def test_get_pull_comments_url(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        with patch.object(gh._s, "get", return_value=mock_resp) as mock_get:
            gh.get_pull_comments("o/r", 7)
        url = mock_get.call_args.args[0]
        assert "repos/o/r/pulls/7/comments" in url

    def test_get_review_comments(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"id": 101, "body": "nit"},
            {"id": 102, "body": "fix"},
        ]
        with patch.object(gh._s, "get", return_value=mock_resp):
            result = gh.get_review_comments("o/r", 10, 99)
        assert result == [(101, "nit"), (102, "fix")]

    def test_get_review_comments_empty(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        with patch.object(gh._s, "get", return_value=mock_resp):
            result = gh.get_review_comments("o/r", 10, 99)
        assert result == []

    def test_get_review_comments_url(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        with patch.object(gh._s, "get", return_value=mock_resp) as mock_get:
            gh.get_review_comments("o/r", 10, 99)
        url = mock_get.call_args.args[0]
        assert "repos/o/r/pulls/10/reviews/99/comments" in url

    def test_comment_issue(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        with patch.object(gh._s, "post", return_value=mock_resp) as mock_post:
            gh.comment_issue("o/r", 7, "hello")
        url = mock_post.call_args.args[0]
        assert "repos/o/r/issues/7/comments" in url
        assert mock_post.call_args.kwargs["json"]["body"] == "hello"

    def test_find_pr_returns_match(self) -> None:
        gh = self._gh()
        search_resp = MagicMock()
        search_resp.json.return_value = {
            "items": [{"number": 1, "user": {"login": "fido"}}]
        }
        pr_resp = MagicMock()
        pr_resp.json.return_value = {
            "number": 1,
            "head": {"ref": "feat"},
            "state": "open",
            "merged": False,
            "user": {"login": "fido"},
        }
        with patch.object(gh._s, "get", side_effect=[search_resp, pr_resp]):
            result = gh.find_pr("o/r", 5, "fido")
        assert result == {
            "number": 1,
            "headRefName": "feat",
            "state": "OPEN",
            "author": {"login": "fido"},
        }

    def test_find_pr_merged_state(self) -> None:
        gh = self._gh()
        search_resp = MagicMock()
        search_resp.json.return_value = {
            "items": [{"number": 3, "user": {"login": "fido"}}]
        }
        pr_resp = MagicMock()
        pr_resp.json.return_value = {
            "number": 3,
            "head": {"ref": "fix"},
            "state": "closed",
            "merged": True,
            "user": {"login": "fido"},
        }
        with patch.object(gh._s, "get", side_effect=[search_resp, pr_resp]):
            result = gh.find_pr("o/r", 2, "fido")
        assert result is not None
        assert result["state"] == "MERGED"

    def test_find_pr_filters_by_user(self) -> None:
        gh = self._gh()
        search_resp = MagicMock()
        search_resp.json.return_value = {
            "items": [{"number": 1, "user": {"login": "other"}}]
        }
        with patch.object(gh._s, "get", return_value=search_resp):
            assert gh.find_pr("o/r", 5, "fido") is None

    def test_find_pr_returns_none_on_empty(self) -> None:
        gh = self._gh()
        search_resp = MagicMock()
        search_resp.json.return_value = {"items": []}
        with patch.object(gh._s, "get", return_value=search_resp):
            assert gh.find_pr("o/r", 1, "fido") is None

    def test_find_pr_search_url(self) -> None:
        gh = self._gh()
        search_resp = MagicMock()
        search_resp.json.return_value = {"items": []}
        with patch.object(gh._s, "get", return_value=search_resp) as mock_get:
            gh.find_pr("o/r", 5, "fido")
        url = mock_get.call_args.args[0]
        assert "/search/issues?q=" in url
        assert "type%3Apr" in url or "type:pr" in url

    def test_get_user(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"login": "fido", "id": 1}
        with patch.object(gh._s, "get", return_value=mock_resp) as mock_get:
            result = gh.get_user()
        url = mock_get.call_args.args[0]
        assert url.endswith("/user")
        assert result == "fido"

    def test_get_repo_info_https(self) -> None:
        gh = self._gh()
        with patch(
            "subprocess.run",
            return_value=_completed("https://github.com/owner/repo.git\n"),
        ):
            assert gh.get_repo_info() == "owner/repo"

    def test_get_repo_info_ssh(self) -> None:
        gh = self._gh()
        with patch(
            "subprocess.run", return_value=_completed("git@github.com:owner/repo.git\n")
        ):
            assert gh.get_repo_info() == "owner/repo"

    def test_get_repo_info_passes_cwd(self) -> None:
        gh = self._gh()
        with patch(
            "subprocess.run", return_value=_completed("https://github.com/o/r.git")
        ) as mock:
            gh.get_repo_info(cwd="/tmp/repo")
        assert mock.call_args.kwargs["cwd"] == "/tmp/repo"

    def test_get_repo_info_raises_unknown(self) -> None:
        gh = self._gh()
        with patch(
            "subprocess.run", return_value=_completed("https://example.com/repo.git")
        ):
            with pytest.raises(ValueError, match="Cannot parse"):
                gh.get_repo_info()

    def test_get_default_branch(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"default_branch": "main", "name": "repo"}
        with patch.object(gh._s, "get", return_value=mock_resp) as mock_get:
            result = gh.get_default_branch("o/r")
        url = mock_get.call_args.args[0]
        assert "repos/o/r" in url
        assert result == "main"

    def test_set_user_status_graphql(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {}}
        with patch.object(gh._s, "post", return_value=mock_resp) as mock_post:
            gh.set_user_status("working", "🚀", busy=True)
        url = mock_post.call_args.args[0]
        assert url.endswith("/graphql")
        body = mock_post.call_args.kwargs["json"]
        assert "changeUserStatus" in body["query"]
        assert body["variables"]["msg"] == "working"
        assert body["variables"]["emoji"] == "🚀"
        assert body["variables"]["busy"] is True

    def test_find_issues_graphql(self) -> None:
        gh = self._gh()
        nodes = [{"number": 1, "title": "bug"}]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": {"repository": {"issues": {"nodes": nodes}}}
        }
        with patch.object(gh._s, "post", return_value=mock_resp) as mock_post:
            result = gh.find_issues("owner", "repo", "fido")
        body = mock_post.call_args.kwargs["json"]
        assert body["variables"] == {"owner": "owner", "repo": "repo", "login": "fido"}
        assert result == nodes

    def test_view_issue(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"state": "open", "title": "Bug", "body": "desc"}
        with patch.object(gh._s, "get", return_value=mock_resp):
            result = gh.view_issue("o/r", 5)
        assert result == {"state": "OPEN", "title": "Bug", "body": "desc"}

    def test_get_issue_comments(self) -> None:
        gh = self._gh()
        comments = [{"id": 1, "body": "hi"}]
        mock_resp = MagicMock()
        mock_resp.json.return_value = comments
        with patch.object(gh._s, "get", return_value=mock_resp) as mock_get:
            result = gh.get_issue_comments("o/r", 9)
        url = mock_get.call_args.args[0]
        assert "repos/o/r/issues/9/comments" in url
        assert result == comments

    def test_create_issue(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"html_url": "https://github.com/o/r/issues/42"}
        with patch.object(gh._s, "post", return_value=mock_resp) as mock_post:
            url = gh.create_issue("o/r", "My feature request", "body text")
        assert url == "https://github.com/o/r/issues/42"
        post_url = mock_post.call_args.args[0]
        assert "repos/o/r/issues" in post_url
        payload = mock_post.call_args.kwargs["json"]
        assert payload["title"] == "My feature request"
        assert payload["body"] == "body text"

    def test_create_pr_returns_url(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "html_url": "https://github.com/o/r/pull/10",
            "number": 10,
        }
        with patch.object(gh._s, "post", return_value=mock_resp) as mock_post:
            result = gh.create_pr("o/r", "title", "body", "main", "feat")
        url = mock_post.call_args.args[0]
        assert "repos/o/r/pulls" in url
        body = mock_post.call_args.kwargs["json"]
        assert body["draft"] is True
        assert body["title"] == "title"
        assert body["base"] == "main"
        assert body["head"] == "feat"
        assert result == "https://github.com/o/r/pull/10"

    def test_edit_pr_body(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        with patch.object(gh._s, "patch", return_value=mock_resp) as mock_patch:
            gh.edit_pr_body("o/r", 10, "new body")
        url = mock_patch.call_args.args[0]
        assert "repos/o/r/pulls/10" in url
        assert mock_patch.call_args.kwargs["json"]["body"] == "new body"

    def test_get_pr_body(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"body": "PR body text"}
        with patch.object(gh._s, "get", return_value=mock_resp) as mock_get:
            result = gh.get_pr_body("o/r", 10)
        url = mock_get.call_args.args[0]
        assert "repos/o/r/pulls/10" in url
        assert result == "PR body text"

    def test_get_pr_body_none_returns_empty_string(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"body": None}
        with patch.object(gh._s, "get", return_value=mock_resp):
            result = gh.get_pr_body("o/r", 10)
        assert result == ""

    def test_add_pr_reviewer(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        with patch.object(gh._s, "post", return_value=mock_resp) as mock_post:
            gh.add_pr_reviewer("o/r", 10, "rhencke")
        url = mock_post.call_args.args[0]
        assert "repos/o/r/pulls/10/requested_reviewers" in url
        assert mock_post.call_args.kwargs["json"]["reviewers"] == ["rhencke"]

    def test_pr_checks_returns_list(self) -> None:
        gh = self._gh()
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
        with patch.object(gh._s, "get", side_effect=[pr_resp, checks_resp]):
            result = gh.pr_checks("o/r", 10)
        assert result == [{"name": "ci", "state": "SUCCESS", "link": "http://..."}]

    def test_pr_checks_in_progress(self) -> None:
        gh = self._gh()
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
        with patch.object(gh._s, "get", side_effect=[pr_resp, checks_resp]):
            result = gh.pr_checks("o/r", 10)
        assert result[0]["state"] == "IN_PROGRESS"

    def test_pr_ready(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        with patch.object(gh._s, "patch", return_value=mock_resp) as mock_patch:
            gh.pr_ready("o/r", 10)
        url = mock_patch.call_args.args[0]
        assert "repos/o/r/pulls/10" in url
        assert mock_patch.call_args.kwargs["json"]["draft"] is False

    def test_pr_merge_squash(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        with patch.object(gh._s, "put", return_value=mock_resp) as mock_put:
            gh.pr_merge("o/r", 10)
        url = mock_put.call_args.args[0]
        assert "repos/o/r/pulls/10/merge" in url
        assert mock_put.call_args.kwargs["json"]["merge_method"] == "squash"

    def test_pr_merge_no_squash(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        with patch.object(gh._s, "put", return_value=mock_resp) as mock_put:
            gh.pr_merge("o/r", 10, squash=False)
        assert mock_put.call_args.kwargs["json"]["merge_method"] == "merge"

    def test_pr_merge_auto(self) -> None:
        gh = self._gh()
        pr_resp = MagicMock()
        pr_resp.json.return_value = {"node_id": "PR_abc"}
        graphql_resp = MagicMock()
        graphql_resp.json.return_value = {"data": {}}
        with (
            patch.object(gh._s, "get", return_value=pr_resp),
            patch.object(gh._s, "post", return_value=graphql_resp) as mock_post,
        ):
            gh.pr_merge("o/r", 10, auto=True)
        body = mock_post.call_args.kwargs["json"]
        assert "enablePullRequestAutoMerge" in body["query"]
        assert body["variables"]["mergeMethod"] == "SQUASH"
        assert body["variables"]["prId"] == "PR_abc"

    def test_get_pr_returns_dict(self) -> None:
        gh = self._gh()
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
            }
        ]
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
        with patch.object(
            gh._s, "get", side_effect=[pr_resp, reviews_resp, commits_resp]
        ):
            result = gh.get_pr("o/r", 10)
        assert result["isDraft"] is False
        assert result["mergeStateStatus"] == "CLEAN"
        assert result["body"] == "desc"
        assert result["reviews"] == [
            {
                "author": {"login": "alice"},
                "state": "APPROVED",
                "submittedAt": "2024-01-01T00:00:00Z",
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
        gh = self._gh()
        pr_resp = MagicMock()
        pr_resp.json.return_value = {
            "draft": True,
            "mergeable_state": None,
            "body": None,
        }
        reviews_resp = MagicMock()
        reviews_resp.json.return_value = []
        commits_resp = MagicMock()
        commits_resp.json.return_value = []
        with patch.object(
            gh._s, "get", side_effect=[pr_resp, reviews_resp, commits_resp]
        ):
            result = gh.get_pr("o/r", 10)
        assert result["body"] == ""
        assert result["mergeStateStatus"] == ""

    def test_get_reviews_returns_dict(self) -> None:
        gh = self._gh()
        pr_resp = MagicMock()
        pr_resp.json.return_value = {"draft": True}
        reviews_resp = MagicMock()
        reviews_resp.json.return_value = [
            {
                "user": {"login": "bob"},
                "state": "CHANGES_REQUESTED",
                "submitted_at": "2024-01-01T00:00:00Z",
            }
        ]
        commits_resp = MagicMock()
        commits_resp.json.return_value = [
            {"commit": {"committer": {"date": "2024-01-02T00:00:00Z"}}}
        ]
        with patch.object(
            gh._s, "get", side_effect=[pr_resp, reviews_resp, commits_resp]
        ):
            result = gh.get_reviews("o/r", 10)
        assert result["isDraft"] is True
        assert result["reviews"] == [
            {
                "author": {"login": "bob"},
                "state": "CHANGES_REQUESTED",
                "submittedAt": "2024-01-01T00:00:00Z",
            }
        ]
        assert result["commits"] == [{"committedDate": "2024-01-02T00:00:00Z"}]

    def test_get_review_threads_graphql(self) -> None:
        gh = self._gh()
        payload = {
            "data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": []}}}}
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = payload
        with patch.object(gh._s, "post", return_value=mock_resp) as mock_post:
            result = gh.get_review_threads("owner", "repo", 10)
        body = mock_post.call_args.kwargs["json"]
        assert "reviewThreads" in body["query"]
        assert body["variables"] == {"owner": "owner", "repo": "repo", "pr": 10}
        assert result == payload

    def test_get_review_threads_coerces_pr_to_int(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        with patch.object(gh._s, "post", return_value=mock_resp) as mock_post:
            gh.get_review_threads("o", "r", "13")
        body = mock_post.call_args.kwargs["json"]
        assert body["variables"]["pr"] == 13

    def test_resolve_thread_graphql(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {}}
        with patch.object(gh._s, "post", return_value=mock_resp) as mock_post:
            gh.resolve_thread("T_kwDOABC123")
        body = mock_post.call_args.kwargs["json"]
        assert "resolveReviewThread" in body["query"]
        assert body["variables"]["id"] == "T_kwDOABC123"

    def test_get_run_log_returns_failed_job_logs(self) -> None:
        gh = self._gh()
        jobs_resp = MagicMock()
        jobs_resp.json.return_value = {
            "jobs": [
                {"id": 1, "name": "test", "conclusion": "failure"},
                {"id": 2, "name": "build", "conclusion": "success"},
            ]
        }
        log_resp = MagicMock()
        log_resp.text = "log output\n"
        with patch.object(gh._s, "get", side_effect=[jobs_resp, log_resp]):
            result = gh.get_run_log("o/r", 12345)
        assert result == "log output\n"

    def test_get_run_log_skips_passing_jobs(self) -> None:
        gh = self._gh()
        jobs_resp = MagicMock()
        jobs_resp.json.return_value = {
            "jobs": [
                {"id": 1, "name": "build", "conclusion": "success"},
            ]
        }
        with patch.object(gh._s, "get", return_value=jobs_resp):
            result = gh.get_run_log("o/r", 99)
        assert result == ""

    def test_get_run_log_timed_out(self) -> None:
        gh = self._gh()
        jobs_resp = MagicMock()
        jobs_resp.json.return_value = {
            "jobs": [
                {"id": 5, "name": "flaky", "conclusion": "timed_out"},
            ]
        }
        log_resp = MagicMock()
        log_resp.text = "timed out log\n"
        with patch.object(gh._s, "get", side_effect=[jobs_resp, log_resp]):
            result = gh.get_run_log("o/r", 55)
        assert result == "timed out log\n"

    def _raw_comment(
        self,
        cid: int,
        body: str,
        path: str = "foo.py",
        line: int = 10,
        author: str = "reviewer",
        parent_id: int | None = None,
    ) -> dict:
        c = {
            "id": cid,
            "body": body,
            "path": path,
            "line": line,
            "user": {"login": author},
        }
        if parent_id is not None:
            c["in_reply_to_id"] = parent_id
        return c

    def test_fetch_sibling_threads_groups_root_and_replies(self) -> None:
        gh = self._gh()
        raw = [
            self._raw_comment(1, "fix this", path="a.py", line=5),
            self._raw_comment(2, "agreed", parent_id=1, author="fido"),
            self._raw_comment(3, "nit here", path="b.py", line=20),
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = raw
        with patch.object(gh._s, "get", return_value=mock_resp):
            threads = gh.fetch_sibling_threads("o/r", 7)
        assert len(threads) == 2
        thread_a = next(t for t in threads if t["path"] == "a.py")
        assert thread_a["line"] == 5
        assert len(thread_a["comments"]) == 2
        assert thread_a["comments"][0] == {"author": "reviewer", "body": "fix this"}
        assert thread_a["comments"][1] == {"author": "fido", "body": "agreed"}
        thread_b = next(t for t in threads if t["path"] == "b.py")
        assert len(thread_b["comments"]) == 1

    def test_fetch_sibling_threads_returns_empty_on_exception(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = RuntimeError("network fail")
        with patch.object(gh._s, "get", return_value=mock_resp):
            result = gh.fetch_sibling_threads("o/r", 7)
        assert result == []

    def test_fetch_sibling_threads_empty_pr(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        with patch.object(gh._s, "get", return_value=mock_resp):
            result = gh.fetch_sibling_threads("o/r", 7)
        assert result == []

    def test_fetch_sibling_threads_orphan_reply_skipped(self) -> None:
        """A reply whose parent_id has no root in the dict is silently skipped."""
        gh = self._gh()
        raw = [self._raw_comment(99, "orphan reply", parent_id=9999)]
        mock_resp = MagicMock()
        mock_resp.json.return_value = raw
        with patch.object(gh._s, "get", return_value=mock_resp):
            result = gh.fetch_sibling_threads("o/r", 7)
        assert result == []
