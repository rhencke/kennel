from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kennel.github import (
    GH,
    GitHub,
    _get_gh,
    _get_github,
    _gh_token,
    add_pr_reviewer,
    add_reaction,
    close_issue,
    comment_issue,
    create_pr,
    edit_pr_body,
    find_issues,
    find_pr,
    get_default_branch,
    get_issue_comments,
    get_pr,
    get_pull_comments,
    get_repo_info,
    get_review_comments,
    get_review_threads,
    get_reviews,
    get_run_log,
    get_user,
    pr_checks,
    pr_merge,
    pr_ready,
    reply_to_review_comment,
    resolve_thread,
    set_user_status,
    view_issue,
)


def _completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=""
    )


@pytest.fixture(autouse=True)
def _reset_gh_cache() -> None:
    """Reset caches before each test and seed with real instances."""
    _get_gh.cache_clear()
    _get_github.cache_clear()
    with patch("kennel.github._gh_token", return_value="test-token"):
        _get_gh()
        _get_github()


class TestGetRepoInfo:
    def test_returns_https_remote(self) -> None:
        with patch(
            "subprocess.run",
            return_value=_completed("https://github.com/owner/repo.git\n"),
        ):
            assert get_repo_info() == "owner/repo"

    def test_returns_ssh_remote(self) -> None:
        with patch(
            "subprocess.run", return_value=_completed("git@github.com:owner/repo.git\n")
        ):
            assert get_repo_info() == "owner/repo"

    def test_https_without_git_suffix(self) -> None:
        with patch(
            "subprocess.run", return_value=_completed("https://github.com/owner/repo\n")
        ):
            assert get_repo_info() == "owner/repo"

    def test_passes_cwd(self) -> None:
        with patch(
            "subprocess.run", return_value=_completed("https://github.com/o/r.git")
        ) as mock:
            get_repo_info(cwd="/some/path")
        assert mock.call_args.kwargs["cwd"] == "/some/path"

    def test_raises_on_unknown_url(self) -> None:
        with patch(
            "subprocess.run", return_value=_completed("https://example.com/repo.git")
        ):
            with pytest.raises(ValueError, match="Cannot parse"):
                get_repo_info()


class TestGetUser:
    def test_returns_login(self) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"login": "fido"}
        with patch.object(_get_github()._gh._s, "get", return_value=mock_resp):
            assert get_user() == "fido"


class TestGetDefaultBranch:
    def test_returns_branch(self) -> None:
        remote_resp = _completed("https://github.com/o/r.git\n")
        repo_resp = MagicMock()
        repo_resp.json.return_value = {"default_branch": "main"}
        with (
            patch("subprocess.run", return_value=remote_resp),
            patch.object(_get_github()._gh._s, "get", return_value=repo_resp),
        ):
            assert get_default_branch() == "main"

    def test_passes_cwd(self) -> None:
        remote_resp = _completed("https://github.com/o/r.git\n")
        repo_resp = MagicMock()
        repo_resp.json.return_value = {"default_branch": "main"}
        with (
            patch("subprocess.run", return_value=remote_resp) as mock_sub,
            patch.object(_get_github()._gh._s, "get", return_value=repo_resp),
        ):
            get_default_branch(cwd=Path("/repo"))
        assert mock_sub.call_args.kwargs["cwd"] == Path("/repo")


class TestSetUserStatus:
    def test_busy_true(self) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        with patch.object(
            _get_github()._gh._s, "post", return_value=mock_resp
        ) as mock_post:
            set_user_status("coding", "🐶", busy=True)
        body = mock_post.call_args.kwargs["json"]
        assert body["variables"]["busy"] is True

    def test_busy_false(self) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        with patch.object(
            _get_github()._gh._s, "post", return_value=mock_resp
        ) as mock_post:
            set_user_status("napping", "💤", busy=False)
        body = mock_post.call_args.kwargs["json"]
        assert body["variables"]["busy"] is False

    def test_passes_msg_and_emoji(self) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        with patch.object(
            _get_github()._gh._s, "post", return_value=mock_resp
        ) as mock_post:
            set_user_status("working", "🚀")
        body = mock_post.call_args.kwargs["json"]
        assert body["variables"]["msg"] == "working"
        assert body["variables"]["emoji"] == "🚀"


class TestFindIssues:
    def test_returns_nodes(self) -> None:
        nodes = [{"number": 1, "title": "Fix it", "subIssues": {"nodes": []}}]
        payload = {"data": {"repository": {"issues": {"nodes": nodes}}}}
        mock_resp = MagicMock()
        mock_resp.json.return_value = payload
        with patch.object(_get_github()._gh._s, "post", return_value=mock_resp):
            result = find_issues("owner", "repo", "fido")
        assert result == nodes

    def test_passes_variables(self) -> None:
        payload = {"data": {"repository": {"issues": {"nodes": []}}}}
        mock_resp = MagicMock()
        mock_resp.json.return_value = payload
        with patch.object(
            _get_github()._gh._s, "post", return_value=mock_resp
        ) as mock_post:
            find_issues("myowner", "myrepo", "mylogin")
        body = mock_post.call_args.kwargs["json"]
        assert body["variables"]["owner"] == "myowner"
        assert body["variables"]["repo"] == "myrepo"
        assert body["variables"]["login"] == "mylogin"


class TestViewIssue:
    def test_returns_parsed_json(self) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"state": "open", "title": "Bug", "body": "desc"}
        with patch.object(_get_github()._gh._s, "get", return_value=mock_resp):
            assert view_issue("o/r", 5) == {
                "state": "OPEN",
                "title": "Bug",
                "body": "desc",
            }

    def test_uppercases_state(self) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"state": "closed", "title": "T", "body": ""}
        with patch.object(_get_github()._gh._s, "get", return_value=mock_resp):
            assert view_issue("o/r", 5)["state"] == "CLOSED"

    def test_null_body_becomes_empty_string(self) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"state": "open", "title": "T", "body": None}
        with patch.object(_get_github()._gh._s, "get", return_value=mock_resp):
            assert view_issue("o/r", 5)["body"] == ""

    def test_uses_correct_url(self) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"state": "open", "title": "T", "body": ""}
        with patch.object(
            _get_github()._gh._s, "get", return_value=mock_resp
        ) as mock_get:
            view_issue("o/r", 42)
        url = mock_get.call_args.args[0]
        assert "repos/o/r/issues/42" in url


class TestCloseIssue:
    def test_patches_state_closed(self) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        with patch.object(
            _get_github()._gh._s, "patch", return_value=mock_resp
        ) as mock_patch:
            close_issue("o/r", 3)
        url = mock_patch.call_args.args[0]
        assert "repos/o/r/issues/3" in url
        assert mock_patch.call_args.kwargs["json"]["state"] == "closed"


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
        _get_github.cache_clear()
        with (
            patch("kennel.github._gh_token", return_value="tok"),
            patch("kennel.github.GitHub", return_value=mock_instance) as mock_cls,
        ):
            result = _get_github()
        mock_cls.assert_called_once_with()
        assert result is mock_instance

    def test_returns_cached_instance(self) -> None:
        mock_instance = MagicMock()
        _get_github.cache_clear()
        with (
            patch("kennel.github._gh_token", return_value="tok"),
            patch("kennel.github.GitHub", return_value=mock_instance) as mock_cls,
        ):
            first = _get_github()
            second = _get_github()
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

    def test_close_issue_delegates(self) -> None:
        gh = self._github()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        with patch.object(gh._gh._s, "patch", return_value=mock_resp) as mock_patch:
            gh.close_issue("o/r", 3)
        assert "repos/o/r/issues/3" in mock_patch.call_args.args[0]

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

    def test_get_pull_comments_delegates(self) -> None:
        gh = self._github()
        comments = [{"id": 42}]
        mock_resp = MagicMock()
        mock_resp.json.return_value = comments
        with patch.object(gh._gh._s, "get", return_value=mock_resp):
            assert gh.get_pull_comments("o/r", 7) == comments

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
        with patch.object(gh._gh._s, "get", side_effect=[pr_resp, reviews_resp]):
            result = gh.get_reviews("o/r", 10)
        assert result["isDraft"] is True

    def test_get_review_comments_delegates(self) -> None:
        gh = self._github()
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"id": 1}, {"id": 2}]
        with patch.object(gh._gh._s, "get", return_value=mock_resp):
            assert gh.get_review_comments("o/r", 10, 99) == [1, 2]

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
        mock_resp.json.return_value = [{"id": 101}, {"id": 102}]
        with patch.object(gh._s, "get", return_value=mock_resp):
            result = gh.get_review_comments("o/r", 10, 99)
        assert result == [101, 102]

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

    def test_close_issue(self) -> None:
        gh = self._gh()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        with patch.object(gh._s, "patch", return_value=mock_resp) as mock_patch:
            gh.close_issue("o/r", 3)
        url = mock_patch.call_args.args[0]
        assert "repos/o/r/issues/3" in url
        assert mock_patch.call_args.kwargs["json"]["state"] == "closed"

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
            {"sha": "abc", "commit": {"message": "Fix bug\n\nDetails"}}
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
        assert result["commits"] == [{"messageHeadline": "Fix bug", "oid": "abc"}]

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
        with patch.object(gh._s, "get", side_effect=[pr_resp, reviews_resp]):
            result = gh.get_reviews("o/r", 10)
        assert result["isDraft"] is True
        assert result["reviews"] == [
            {
                "author": {"login": "bob"},
                "state": "CHANGES_REQUESTED",
                "submittedAt": "2024-01-01T00:00:00Z",
            }
        ]

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


class TestCommentIssue:
    def test_calls_gh_class(self) -> None:
        mock_gh = MagicMock()
        with patch("kennel.github._get_github", return_value=mock_gh):
            comment_issue("o/r", 7, "hello")
        mock_gh.comment_issue.assert_called_once_with("o/r", 7, "hello")


class TestGetIssueComments:
    def test_delegates_to_gh_class(self) -> None:
        comments = [{"id": 1, "body": "hi"}]
        mock_gh = MagicMock()
        mock_gh.get_issue_comments.return_value = comments
        with patch("kennel.github._get_github", return_value=mock_gh):
            assert get_issue_comments("o/r", 9) == comments
        mock_gh.get_issue_comments.assert_called_once_with("o/r", 9)


class TestGetPullComments:
    def test_delegates_to_gh_class(self) -> None:
        comments = [{"id": 42, "body": "looks good"}]
        mock_gh = MagicMock()
        mock_gh.get_pull_comments.return_value = comments
        with patch("kennel.github._get_github", return_value=mock_gh):
            assert get_pull_comments("o/r", 7) == comments
        mock_gh.get_pull_comments.assert_called_once_with("o/r", 7)


class TestFindPr:
    def test_delegates_to_gh_class(self) -> None:
        expected = {
            "number": 1,
            "headRefName": "feat",
            "state": "OPEN",
            "author": {"login": "fido"},
        }
        mock_gh = MagicMock()
        mock_gh.find_pr.return_value = expected
        with patch("kennel.github._get_github", return_value=mock_gh):
            result = find_pr("o/r", 5, "fido")
        mock_gh.find_pr.assert_called_once_with("o/r", 5, "fido")
        assert result == expected

    def test_returns_none_when_gh_returns_none(self) -> None:
        mock_gh = MagicMock()
        mock_gh.find_pr.return_value = None
        with patch("kennel.github._get_github", return_value=mock_gh):
            assert find_pr("o/r", 1, "fido") is None


class TestCreatePr:
    def test_delegates_to_gh_class(self) -> None:
        mock_gh = MagicMock()
        mock_gh.create_pr.return_value = "https://github.com/o/r/pull/10"
        with patch("kennel.github._get_github", return_value=mock_gh):
            result = create_pr("o/r", "title", "body", "main", "feat")
        mock_gh.create_pr.assert_called_once_with(
            "o/r", "title", "body", "main", "feat"
        )
        assert result == "https://github.com/o/r/pull/10"


class TestEditPrBody:
    def test_delegates_to_gh_class(self) -> None:
        mock_gh = MagicMock()
        with patch("kennel.github._get_github", return_value=mock_gh):
            edit_pr_body("o/r", 10, "new body")
        mock_gh.edit_pr_body.assert_called_once_with("o/r", 10, "new body")


class TestAddPrReviewer:
    def test_delegates_to_gh_class(self) -> None:
        mock_gh = MagicMock()
        with patch("kennel.github._get_github", return_value=mock_gh):
            add_pr_reviewer("o/r", 10, "rhencke")
        mock_gh.add_pr_reviewer.assert_called_once_with("o/r", 10, "rhencke")


class TestPrChecks:
    def test_delegates_to_gh_class(self) -> None:
        checks = [{"name": "ci", "state": "SUCCESS", "link": "http://..."}]
        mock_gh = MagicMock()
        mock_gh.pr_checks.return_value = checks
        with patch("kennel.github._get_github", return_value=mock_gh):
            assert pr_checks("o/r", 10) == checks
        mock_gh.pr_checks.assert_called_once_with("o/r", 10)


class TestPrReady:
    def test_delegates_to_gh_class(self) -> None:
        mock_gh = MagicMock()
        with patch("kennel.github._get_github", return_value=mock_gh):
            pr_ready("o/r", 10)
        mock_gh.pr_ready.assert_called_once_with("o/r", 10)


class TestPrMerge:
    def test_squash_default(self) -> None:
        mock_gh = MagicMock()
        with patch("kennel.github._get_github", return_value=mock_gh):
            pr_merge("o/r", 10)
        mock_gh.pr_merge.assert_called_once_with("o/r", 10, squash=True, auto=False)

    def test_auto_flag(self) -> None:
        mock_gh = MagicMock()
        with patch("kennel.github._get_github", return_value=mock_gh):
            pr_merge("o/r", 10, auto=True)
        mock_gh.pr_merge.assert_called_once_with("o/r", 10, squash=True, auto=True)

    def test_no_squash(self) -> None:
        mock_gh = MagicMock()
        with patch("kennel.github._get_github", return_value=mock_gh):
            pr_merge("o/r", 10, squash=False)
        mock_gh.pr_merge.assert_called_once_with("o/r", 10, squash=False, auto=False)


class TestGetPr:
    def test_delegates_to_gh_class(self) -> None:
        data = {
            "reviews": [],
            "isDraft": True,
            "mergeStateStatus": "CLEAN",
            "body": "",
            "commits": [],
        }
        mock_gh = MagicMock()
        mock_gh.get_pr.return_value = data
        with patch("kennel.github._get_github", return_value=mock_gh):
            assert get_pr("o/r", 10) == data
        mock_gh.get_pr.assert_called_once_with("o/r", 10)


class TestGetReviews:
    def test_delegates_to_gh_class(self) -> None:
        data = {"reviews": [], "isDraft": False}
        mock_gh = MagicMock()
        mock_gh.get_reviews.return_value = data
        with patch("kennel.github._get_github", return_value=mock_gh):
            assert get_reviews("o/r", 10) == data
        mock_gh.get_reviews.assert_called_once_with("o/r", 10)


class TestGetReviewComments:
    def test_delegates_to_gh_class(self) -> None:
        mock_gh = MagicMock()
        mock_gh.get_review_comments.return_value = [101, 102, 103]
        with patch("kennel.github._get_github", return_value=mock_gh):
            assert get_review_comments("o/r", 10, 99) == [101, 102, 103]
        mock_gh.get_review_comments.assert_called_once_with("o/r", 10, 99)

    def test_empty_result(self) -> None:
        mock_gh = MagicMock()
        mock_gh.get_review_comments.return_value = []
        with patch("kennel.github._get_github", return_value=mock_gh):
            assert get_review_comments("o/r", 10, 99) == []


class TestReplyToReviewComment:
    def test_delegates_to_gh_class(self) -> None:
        mock_gh = MagicMock()
        with patch("kennel.github._get_github", return_value=mock_gh):
            reply_to_review_comment("o/r", 10, "lgtm", 55)
        mock_gh.reply_to_review_comment.assert_called_once_with("o/r", 10, "lgtm", 55)


class TestAddReaction:
    def test_delegates_to_gh_class_pulls(self) -> None:
        mock_gh = MagicMock()
        with patch("kennel.github._get_github", return_value=mock_gh):
            add_reaction("o/r", "pulls", 42, "rocket")
        mock_gh.add_reaction.assert_called_once_with("o/r", "pulls", 42, "rocket")

    def test_delegates_to_gh_class_issues(self) -> None:
        mock_gh = MagicMock()
        with patch("kennel.github._get_github", return_value=mock_gh):
            add_reaction("o/r", "issues", 7, "+1")
        mock_gh.add_reaction.assert_called_once_with("o/r", "issues", 7, "+1")


class TestGetReviewThreads:
    def test_delegates_to_gh_class(self) -> None:
        data = {
            "data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": []}}}}
        }
        mock_gh = MagicMock()
        mock_gh.get_review_threads.return_value = data
        with patch("kennel.github._get_github", return_value=mock_gh):
            assert get_review_threads("owner", "repo", 10) == data
        mock_gh.get_review_threads.assert_called_once_with("owner", "repo", 10)


class TestResolveThread:
    def test_delegates_to_gh_class(self) -> None:
        mock_gh = MagicMock()
        with patch("kennel.github._get_github", return_value=mock_gh):
            resolve_thread("T_kwDOABC123")
        mock_gh.resolve_thread.assert_called_once_with("T_kwDOABC123")


class TestGetRunLog:
    def test_delegates_to_gh_class(self) -> None:
        mock_gh = MagicMock()
        mock_gh.get_run_log.return_value = "log output\n"
        with patch("kennel.github._get_github", return_value=mock_gh):
            assert get_run_log("o/r", 12345) == "log output\n"
        mock_gh.get_run_log.assert_called_once_with("o/r", 12345)

    def test_passes_string_run_id(self) -> None:
        mock_gh = MagicMock()
        mock_gh.get_run_log.return_value = ""
        with patch("kennel.github._get_github", return_value=mock_gh):
            get_run_log("o/r", "99")
        mock_gh.get_run_log.assert_called_once_with("o/r", "99")
