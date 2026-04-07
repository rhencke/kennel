from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from kennel.github import (
    GH,
    _gh,
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


class TestGhHelper:
    def test_calls_subprocess_run(self) -> None:
        with patch("subprocess.run", return_value=_completed("out")) as mock:
            result = _gh("api", "user", cwd="/tmp", timeout=5)
        mock.assert_called_once_with(
            ["gh", "api", "user"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd="/tmp",
        )
        assert result.stdout == "out"

    def test_defaults(self) -> None:
        with patch("subprocess.run", return_value=_completed()) as mock:
            _gh("version")
        _, kwargs = mock.call_args
        assert kwargs["timeout"] == 30
        assert kwargs["cwd"] is None


class TestGetRepoInfo:
    def test_returns_stripped_name(self) -> None:
        with patch("subprocess.run", return_value=_completed("owner/repo\n")):
            assert get_repo_info() == "owner/repo"

    def test_passes_cwd(self) -> None:
        with patch("subprocess.run", return_value=_completed("o/r")) as mock:
            get_repo_info(cwd="/some/path")
        assert mock.call_args.kwargs["cwd"] == "/some/path"


class TestGetUser:
    def test_returns_login(self) -> None:
        with patch("subprocess.run", return_value=_completed("fido\n")):
            assert get_user() == "fido"


class TestGetDefaultBranch:
    def test_returns_branch(self) -> None:
        with patch("subprocess.run", return_value=_completed("main\n")):
            assert get_default_branch() == "main"

    def test_passes_cwd(self) -> None:
        with patch("subprocess.run", return_value=_completed("main")) as mock:
            get_default_branch(cwd=Path("/repo"))
        assert mock.call_args.kwargs["cwd"] == Path("/repo")


class TestSetUserStatus:
    def test_busy_true(self) -> None:
        with patch("subprocess.run", return_value=_completed()) as mock:
            set_user_status("coding", "🐶", busy=True)
        cmd = mock.call_args.args[0]
        assert "busy=true" in cmd

    def test_busy_false(self) -> None:
        with patch("subprocess.run", return_value=_completed()) as mock:
            set_user_status("napping", "💤", busy=False)
        cmd = mock.call_args.args[0]
        assert "busy=false" in cmd

    def test_passes_msg_and_emoji(self) -> None:
        with patch("subprocess.run", return_value=_completed()) as mock:
            set_user_status("working", "🚀")
        cmd = mock.call_args.args[0]
        assert "msg=working" in cmd
        assert "emoji=🚀" in cmd


class TestFindIssues:
    def test_returns_nodes(self) -> None:
        nodes = [{"number": 1, "title": "Fix it", "subIssues": {"nodes": []}}]
        payload = {"data": {"repository": {"issues": {"nodes": nodes}}}}
        with patch("subprocess.run", return_value=_completed(json.dumps(payload))):
            result = find_issues("owner", "repo", "fido")
        assert result == nodes

    def test_passes_variables(self) -> None:
        payload = {"data": {"repository": {"issues": {"nodes": []}}}}
        with patch(
            "subprocess.run", return_value=_completed(json.dumps(payload))
        ) as mock:
            find_issues("myowner", "myrepo", "mylogin")
        cmd = mock.call_args.args[0]
        assert "-F" in cmd
        assert "owner=myowner" in cmd
        assert "repo=myrepo" in cmd
        assert "login=mylogin" in cmd


class TestViewIssue:
    def test_returns_parsed_json(self) -> None:
        issue = {"state": "OPEN", "title": "Bug", "body": "desc"}
        with patch("subprocess.run", return_value=_completed(json.dumps(issue))):
            assert view_issue("o/r", 5) == issue

    def test_converts_number_to_str(self) -> None:
        issue = {"state": "OPEN", "title": "T", "body": ""}
        with patch(
            "subprocess.run", return_value=_completed(json.dumps(issue))
        ) as mock:
            view_issue("o/r", 42)
        cmd = mock.call_args.args[0]
        assert "42" in cmd


class TestCloseIssue:
    def test_calls_gh(self) -> None:
        with patch("subprocess.run", return_value=_completed()) as mock:
            close_issue("o/r", 3)
        cmd = mock.call_args.args[0]
        assert cmd == ["gh", "issue", "close", "3", "--repo", "o/r"]


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


class TestCommentIssue:
    def test_calls_gh_class(self) -> None:
        mock_gh = MagicMock()
        with (
            patch("kennel.github._gh_token", return_value="tok"),
            patch("kennel.github.GH", return_value=mock_gh) as mock_cls,
        ):
            comment_issue("o/r", 7, "hello")
        mock_cls.assert_called_once_with("tok")
        mock_gh.comment_issue.assert_called_once_with("o/r", 7, "hello")


class TestGetIssueComments:
    def test_returns_list(self) -> None:
        comments = [{"id": 1, "body": "hi"}]
        with patch("subprocess.run", return_value=_completed(json.dumps(comments))):
            assert get_issue_comments("o/r", 1) == comments

    def test_uses_api_endpoint(self) -> None:
        with patch("subprocess.run", return_value=_completed("[]")) as mock:
            get_issue_comments("o/r", 9)
        cmd = mock.call_args.args[0]
        assert "repos/o/r/issues/9/comments" in cmd


class TestGetPullComments:
    def test_returns_list(self) -> None:
        comments = [{"id": 42, "body": "looks good"}]
        with patch("subprocess.run", return_value=_completed(json.dumps(comments))):
            assert get_pull_comments("o/r", 7) == comments

    def test_uses_api_endpoint(self) -> None:
        with patch("subprocess.run", return_value=_completed("[]")) as mock:
            get_pull_comments("o/r", 7)
        cmd = mock.call_args.args[0]
        assert "repos/o/r/pulls/7/comments" in cmd


class TestFindPr:
    def test_returns_matching_pr(self) -> None:
        prs = [
            {
                "number": 1,
                "headRefName": "feat",
                "state": "OPEN",
                "author": {"login": "fido"},
            },
            {
                "number": 2,
                "headRefName": "other",
                "state": "OPEN",
                "author": {"login": "other"},
            },
        ]
        with patch("subprocess.run", return_value=_completed(json.dumps(prs))):
            result = find_pr("o/r", 5, "fido")
        assert result == prs[0]

    def test_returns_none_if_not_found(self) -> None:
        prs = [
            {
                "number": 1,
                "headRefName": "feat",
                "state": "OPEN",
                "author": {"login": "other"},
            }
        ]
        with patch("subprocess.run", return_value=_completed(json.dumps(prs))):
            assert find_pr("o/r", 5, "fido") is None

    def test_returns_none_on_empty(self) -> None:
        with patch("subprocess.run", return_value=_completed("[]")):
            assert find_pr("o/r", 1, "fido") is None


class TestCreatePr:
    def test_returns_url(self) -> None:
        with patch(
            "subprocess.run",
            return_value=_completed("https://github.com/o/r/pull/10\n"),
        ):
            assert (
                create_pr("o/r", "title", "body", "main", "feat")
                == "https://github.com/o/r/pull/10"
            )

    def test_passes_args(self) -> None:
        with patch("subprocess.run", return_value=_completed("url")) as mock:
            create_pr("o/r", "T", "B", "main", "branch")
        cmd = mock.call_args.args[0]
        assert "--draft" in cmd
        assert "--title" in cmd
        assert "T" in cmd
        assert "--base" in cmd
        assert "main" in cmd
        assert "--head" in cmd
        assert "branch" in cmd


class TestEditPrBody:
    def test_calls_gh(self) -> None:
        with patch("subprocess.run", return_value=_completed()) as mock:
            edit_pr_body("o/r", 10, "new body")
        cmd = mock.call_args.args[0]
        assert cmd == ["gh", "pr", "edit", "10", "--repo", "o/r", "--body", "new body"]


class TestAddPrReviewer:
    def test_calls_gh(self) -> None:
        with patch("subprocess.run", return_value=_completed()) as mock:
            add_pr_reviewer("o/r", 10, "rhencke")
        cmd = mock.call_args.args[0]
        assert cmd == [
            "gh",
            "pr",
            "edit",
            "10",
            "--repo",
            "o/r",
            "--add-reviewer",
            "rhencke",
        ]


class TestPrChecks:
    def test_returns_list(self) -> None:
        checks = [{"name": "ci", "state": "SUCCESS", "link": "http://..."}]
        with patch("subprocess.run", return_value=_completed(json.dumps(checks))):
            assert pr_checks("o/r", 10) == checks


class TestPrReady:
    def test_calls_gh(self) -> None:
        with patch("subprocess.run", return_value=_completed()) as mock:
            pr_ready("o/r", 10)
        cmd = mock.call_args.args[0]
        assert cmd == ["gh", "pr", "ready", "10", "--repo", "o/r"]


class TestPrMerge:
    def test_squash_default(self) -> None:
        with patch("subprocess.run", return_value=_completed()) as mock:
            pr_merge("o/r", 10)
        cmd = mock.call_args.args[0]
        assert "--squash" in cmd
        assert "--auto" not in cmd

    def test_auto_flag(self) -> None:
        with patch("subprocess.run", return_value=_completed()) as mock:
            pr_merge("o/r", 10, auto=True)
        cmd = mock.call_args.args[0]
        assert "--auto" in cmd

    def test_no_squash(self) -> None:
        with patch("subprocess.run", return_value=_completed()) as mock:
            pr_merge("o/r", 10, squash=False)
        cmd = mock.call_args.args[0]
        assert "--squash" not in cmd


class TestGetPr:
    def test_returns_dict(self) -> None:
        data = {
            "reviews": [],
            "isDraft": True,
            "mergeStateStatus": "CLEAN",
            "body": "",
            "commits": [],
        }
        with patch("subprocess.run", return_value=_completed(json.dumps(data))):
            assert get_pr("o/r", 10) == data

    def test_requests_all_fields(self) -> None:
        data = {
            "reviews": [],
            "isDraft": False,
            "mergeStateStatus": "CLEAN",
            "body": "",
            "commits": [],
        }
        with patch("subprocess.run", return_value=_completed(json.dumps(data))) as mock:
            get_pr("o/r", 10)
        cmd = mock.call_args.args[0]
        assert "reviews,isDraft,mergeStateStatus,body,commits" in cmd


class TestGetReviews:
    def test_returns_dict(self) -> None:
        data = {"reviews": [], "isDraft": False}
        with patch("subprocess.run", return_value=_completed(json.dumps(data))):
            assert get_reviews("o/r", 10) == data


class TestGetReviewComments:
    def test_delegates_to_gh_class(self) -> None:
        mock_gh = MagicMock()
        mock_gh.get_review_comments.return_value = [101, 102, 103]
        with (
            patch("kennel.github._gh_token", return_value="tok"),
            patch("kennel.github.GH", return_value=mock_gh),
        ):
            assert get_review_comments("o/r", 10, 99) == [101, 102, 103]
        mock_gh.get_review_comments.assert_called_once_with("o/r", 10, 99)

    def test_empty_result(self) -> None:
        mock_gh = MagicMock()
        mock_gh.get_review_comments.return_value = []
        with (
            patch("kennel.github._gh_token", return_value="tok"),
            patch("kennel.github.GH", return_value=mock_gh),
        ):
            assert get_review_comments("o/r", 10, 99) == []


class TestReplyToReviewComment:
    def test_delegates_to_gh_class(self) -> None:
        mock_gh = MagicMock()
        with (
            patch("kennel.github._gh_token", return_value="tok"),
            patch("kennel.github.GH", return_value=mock_gh) as mock_cls,
        ):
            reply_to_review_comment("o/r", 10, "lgtm", 55)
        mock_cls.assert_called_once_with("tok")
        mock_gh.reply_to_review_comment.assert_called_once_with("o/r", 10, "lgtm", 55)


class TestAddReaction:
    def test_delegates_to_gh_class_pulls(self) -> None:
        mock_gh = MagicMock()
        with (
            patch("kennel.github._gh_token", return_value="tok"),
            patch("kennel.github.GH", return_value=mock_gh) as mock_cls,
        ):
            add_reaction("o/r", "pulls", 42, "rocket")
        mock_cls.assert_called_once_with("tok")
        mock_gh.add_reaction.assert_called_once_with("o/r", "pulls", 42, "rocket")

    def test_delegates_to_gh_class_issues(self) -> None:
        mock_gh = MagicMock()
        with (
            patch("kennel.github._gh_token", return_value="tok"),
            patch("kennel.github.GH", return_value=mock_gh),
        ):
            add_reaction("o/r", "issues", 7, "+1")
        mock_gh.add_reaction.assert_called_once_with("o/r", "issues", 7, "+1")


class TestGetReviewThreads:
    def test_returns_parsed_json(self) -> None:
        data = {
            "data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": []}}}}
        }
        with patch("subprocess.run", return_value=_completed(json.dumps(data))):
            assert get_review_threads("owner", "repo", 10) == data

    def test_passes_variables(self) -> None:
        data = {"data": {}}
        with patch("subprocess.run", return_value=_completed(json.dumps(data))) as mock:
            get_review_threads("myowner", "myrepo", 10)
        cmd = mock.call_args.args[0]
        assert "owner=myowner" in cmd
        assert "repo=myrepo" in cmd
        assert "pr=10" in cmd


class TestResolveThread:
    def test_calls_graphql(self) -> None:
        with patch("subprocess.run", return_value=_completed()) as mock:
            resolve_thread("T_kwDOABC123")
        cmd = mock.call_args.args[0]
        assert "graphql" in cmd
        assert "id=T_kwDOABC123" in cmd
        assert any("resolveReviewThread" in a for a in cmd)


class TestGetRunLog:
    def test_returns_stdout(self) -> None:
        with patch("subprocess.run", return_value=_completed("log output\n")):
            assert get_run_log(12345) == "log output\n"

    def test_uses_timeout_60(self) -> None:
        with patch("subprocess.run", return_value=_completed()) as mock:
            get_run_log("99")
        assert mock.call_args.kwargs["timeout"] == 60

    def test_converts_run_id(self) -> None:
        with patch("subprocess.run", return_value=_completed()) as mock:
            get_run_log(42)
        cmd = mock.call_args.args[0]
        assert "42" in cmd
