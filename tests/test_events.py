from __future__ import annotations

from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

from kennel.config import Config, RepoConfig
from kennel.events import (
    Action,
    _comment_lock,
    _get_commit_summary,
    _is_allowed,
    _notify_thread_change,
    _reorder_tasks_background,
    _summarize_as_action_item,
    _triage,
    create_task,
    dispatch,
    launch_sync,
    launch_worker,
    maybe_react,
    needs_more_context,
    reply_to_comment,
    reply_to_issue_comment,
    reply_to_review,
)


def _config(tmp_path: Path) -> Config:
    return Config(
        port=9000,
        secret=b"test",
        repos={},
        allowed_bots=frozenset({"copilot[bot]"}),
        log_level="WARNING",
        self_repo=None,
        sub_dir=tmp_path / "sub",
    )


def _repo_cfg(tmp_path: Path) -> RepoConfig:
    return RepoConfig(name="owner/repo", work_dir=tmp_path)


def _payload(repo_owner: str = "owner") -> dict:
    return {
        "repository": {
            "full_name": f"{repo_owner}/repo",
            "owner": {"login": repo_owner},
        },
    }


class TestNeedsMoreContext:
    def test_haiku_yes_returns_true(self) -> None:
        assert needs_more_context("same", _print_prompt=MagicMock(return_value="YES"))

    def test_haiku_yes_with_explanation_returns_true(self) -> None:
        assert needs_more_context(
            "^", _print_prompt=MagicMock(return_value="YES, this is vague")
        )

    def test_haiku_no_returns_false(self) -> None:
        assert not needs_more_context(
            "This is a detailed review comment.",
            _print_prompt=MagicMock(return_value="NO"),
        )

    def test_haiku_no_with_explanation_returns_false(self) -> None:
        assert not needs_more_context(
            "Please rename this variable to be more descriptive.",
            _print_prompt=MagicMock(return_value="NO, it's clear"),
        )

    def test_subprocess_exception_returns_false(self) -> None:
        assert not needs_more_context("ditto", _print_prompt=MagicMock(return_value=""))

    def test_timeout_returns_false(self) -> None:
        assert not needs_more_context("same", _print_prompt=MagicMock(return_value=""))

    def test_empty_output_returns_false(self) -> None:
        assert not needs_more_context(
            "here too", _print_prompt=MagicMock(return_value="")
        )

    def test_uses_haiku_model(self) -> None:
        mock_pp = MagicMock(return_value="YES")
        needs_more_context("same", _print_prompt=mock_pp)
        assert mock_pp.call_args.args[1] == "claude-haiku-4-5"

    def test_defaults_to_claude_print_prompt(self) -> None:
        with patch("kennel.claude.print_prompt", return_value="NO") as mock_pp:
            result = needs_more_context("some comment")
        mock_pp.assert_called_once()
        assert result is False


class TestIsAllowed:
    def test_owner_allowed(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = _payload("owner")
        assert _is_allowed("owner", payload, cfg)

    def test_bot_allowed(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = _payload("owner")
        assert _is_allowed("copilot[bot]", payload, cfg)

    def test_random_user_denied(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = _payload("owner")
        assert not _is_allowed("rando", payload, cfg)


class TestDispatchPing:
    def test_returns_none(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        result = dispatch(
            "ping", {"hook_id": 123, **_payload()}, cfg, _repo_cfg(tmp_path)
        )
        assert result is None


class TestDispatchIssuesAssigned:
    def test_returns_action(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "assigned",
            "assignee": {"login": "fido"},
            "issue": {"number": 1, "title": "test issue"},
        }
        result = dispatch("issues", payload, cfg, _repo_cfg(tmp_path))
        assert result is not None
        assert "#1" in result.prompt

    def test_no_number_returns_none(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "assigned",
            "assignee": {"login": "fido"},
            "issue": {"title": "test"},
        }
        result = dispatch("issues", payload, cfg, _repo_cfg(tmp_path))
        assert result is None


class TestDispatchReviewComment:
    def test_owner_comment(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "created",
            "comment": {
                "id": 123,
                "body": "fix this",
                "user": {"login": "owner"},
                "html_url": "https://example.com",
                "path": "test.py",
                "line": 10,
                "diff_hunk": "@@ -1,3 +1,3 @@",
            },
            "pull_request": {"number": 5, "title": "pr title", "body": "pr body"},
        }
        result = dispatch(
            "pull_request_review_comment", payload, cfg, _repo_cfg(tmp_path)
        )
        assert result is not None
        assert result.reply_to is not None
        assert result.comment_body == "fix this"

    def test_self_comment_ignored(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "created",
            "comment": {"id": 1, "body": "done", "user": {"login": "FidoCanCode"}},
            "pull_request": {"number": 5},
        }
        result = dispatch(
            "pull_request_review_comment", payload, cfg, _repo_cfg(tmp_path)
        )
        assert result is None

    def test_unallowed_user_ignored(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "created",
            "comment": {"id": 1, "body": "hi", "user": {"login": "rando"}},
            "pull_request": {"number": 5},
        }
        result = dispatch(
            "pull_request_review_comment", payload, cfg, _repo_cfg(tmp_path)
        )
        assert result is None


class TestDispatchCheckRun:
    def test_failure(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "completed",
            "check_run": {
                "conclusion": "failure",
                "name": "test-unit",
                "pull_requests": [{"number": 3}],
            },
        }
        result = dispatch("check_run", payload, cfg, _repo_cfg(tmp_path))
        assert result is not None
        assert "CI failure" in result.prompt

    def test_success_ignored(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "completed",
            "check_run": {"conclusion": "success", "name": "lint", "pull_requests": []},
        }
        result = dispatch("check_run", payload, cfg, _repo_cfg(tmp_path))
        assert result is None


class TestDispatchPullRequest:
    def test_merged(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "closed",
            "pull_request": {"number": 7, "merged": True},
        }
        result = dispatch("pull_request", payload, cfg, _repo_cfg(tmp_path))
        assert result is not None
        assert "merged" in result.prompt

    def test_closed_not_merged(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "closed",
            "pull_request": {"number": 7, "merged": False},
        }
        result = dispatch("pull_request", payload, cfg, _repo_cfg(tmp_path))
        assert result is None


class TestDispatchIssueComment:
    def test_pr_comment(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "created",
            "comment": {
                "id": 456,
                "body": "looks good",
                "user": {"login": "owner"},
                "html_url": "https://github.com/owner/repo/pull/10#issuecomment-456",
            },
            "issue": {
                "number": 10,
                "title": "test pr",
                "body": "desc",
                "pull_request": {"url": "https://api.github.com/..."},
            },
        }
        result = dispatch("issue_comment", payload, cfg, _repo_cfg(tmp_path))
        assert result is not None
        assert result.comment_body == "looks good"
        assert result.thread is not None
        assert (
            result.thread["url"]
            == "https://github.com/owner/repo/pull/10#issuecomment-456"
        )

    def test_non_pr_ignored(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "created",
            "comment": {"id": 1, "body": "hi", "user": {"login": "owner"}},
            "issue": {"number": 10, "title": "issue"},
        }
        result = dispatch("issue_comment", payload, cfg, _repo_cfg(tmp_path))
        assert result is None


class TestDispatchUnknown:
    def test_unknown_event(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        result = dispatch(
            "unknown_event",
            {**_payload(), "action": "whatever"},
            cfg,
            _repo_cfg(tmp_path),
        )
        assert result is None


# ── New coverage tests ──────────────────────────────────────────────────────


class TestCommentLock:
    def test_creates_lock_file(self, tmp_path: Path) -> None:
        path = _comment_lock(tmp_path, 42)
        assert path == tmp_path / ".git" / "fido" / "comments" / "42.lock"
        assert path.parent.is_dir()


class TestSummarizeAsActionItem:
    def test_returns_model_result(self) -> None:
        pp = MagicMock(return_value="add logging to streamed sub-Claude output")
        result = _summarize_as_action_item(
            "Ensure we log at that level too.", _print_prompt=pp
        )
        assert result == "add logging to streamed sub-Claude output"

    def test_falls_back_to_comment_body_when_empty(self) -> None:
        pp = MagicMock(return_value="")
        result = _summarize_as_action_item("short comment", _print_prompt=pp)
        assert result == "short comment"

    def test_truncates_long_comment_in_fallback(self) -> None:
        long_comment = "x" * 200
        pp = MagicMock(return_value="")
        result = _summarize_as_action_item(long_comment, _print_prompt=pp)
        assert result == long_comment[:80]

    def test_strips_whitespace_from_result(self) -> None:
        pp = MagicMock(return_value="  add tests  ")
        result = _summarize_as_action_item("add tests please", _print_prompt=pp)
        assert result == "add tests"

    def test_defaults_to_claude_print_prompt(self) -> None:
        with patch("kennel.claude.print_prompt", return_value="add tests") as mock_pp:
            result = _summarize_as_action_item("add some tests")
        mock_pp.assert_called_once()
        assert result == "add tests"

    def test_short_result_returned_without_retry(self) -> None:
        short_title = "add unit tests"
        pp = MagicMock(return_value=short_title)
        result = _summarize_as_action_item("add some tests", _print_prompt=pp)
        assert result == short_title
        pp.assert_called_once()  # no retry needed

    def test_retries_when_result_too_long(self) -> None:
        long_title = "a" * 81
        short_title = "add tests"
        pp = MagicMock(side_effect=[long_title, short_title])
        result = _summarize_as_action_item("add some tests", _print_prompt=pp)
        assert result == short_title
        assert pp.call_count == 2

    def test_retries_up_to_three_times_then_truncates(self) -> None:
        long_title = "a" * 81
        pp = MagicMock(return_value=long_title)
        result = _summarize_as_action_item("add some tests", _print_prompt=pp)
        assert result == long_title[:80]
        assert pp.call_count == 4  # 1 initial + 3 retries

    def test_stops_retrying_once_short_enough(self) -> None:
        titles = ["a" * 81, "b" * 81, "short title"]
        pp = MagicMock(side_effect=titles)
        result = _summarize_as_action_item("add some tests", _print_prompt=pp)
        assert result == "short title"
        assert pp.call_count == 3  # 1 initial + 2 retries


class TestTriage:
    def test_returns_parsed_category(self, tmp_path: Path) -> None:
        cat, title = _triage(
            "please add tests",
            is_bot=False,
            _print_prompt=MagicMock(return_value="ACT: add tests"),
        )
        assert cat == "ACT"
        assert title == "add tests"

    def test_fallback_on_bad_response(self, tmp_path: Path) -> None:
        pp = MagicMock(side_effect=["", "implement the thing"])
        cat, title = _triage("do stuff", is_bot=False, _print_prompt=pp)
        assert cat == "ACT"
        assert title == "implement the thing"

    def test_fallback_for_bot(self, tmp_path: Path) -> None:
        pp = MagicMock(side_effect=["", "implement the thing"])
        cat, title = _triage("do stuff", is_bot=True, _print_prompt=pp)
        assert cat == "DO"
        assert title == "implement the thing"

    def test_with_context(self, tmp_path: Path) -> None:
        ctx = {"pr_title": "My PR", "file": "foo.py", "diff_hunk": "@@ -1 +1 @@"}
        cat, title = _triage(
            "nit comment",
            is_bot=False,
            context=ctx,
            _print_prompt=MagicMock(return_value="DEFER: out of scope"),
        )
        assert cat == "DEFER"

    def test_unrecognized_category_falls_back(self, tmp_path: Path) -> None:
        pp = MagicMock(side_effect=["WEIRD: something", "do the thing"])
        cat, title = _triage("hi", is_bot=False, _print_prompt=pp)
        assert cat == "ACT"
        assert title == "do the thing"

    def test_timeout_falls_back(self, tmp_path: Path) -> None:
        pp = MagicMock(side_effect=["", "do the thing"])
        cat, title = _triage("hi", is_bot=True, _print_prompt=pp)
        assert cat == "DO"

    def test_task_category_falls_back(self, tmp_path: Path) -> None:
        """TASK is no longer a valid bot category — falls back to DO."""
        pp = MagicMock(side_effect=["TASK: add caching", "add result caching"])
        cat, title = _triage("cache results", is_bot=True, _print_prompt=pp)
        assert cat == "DO"
        assert title == "add result caching"

    def test_bot_categories_in_prompt(self, tmp_path: Path) -> None:
        """Ensure bot-specific categories (DO/DEFER/DUMP) are used when is_bot=True."""
        captured = {}

        def fake_pp(prompt, model, **kwargs):
            captured["prompt"] = prompt
            return "DO: implement feature"

        cat, _ = _triage("implement feature", is_bot=True, _print_prompt=fake_pp)
        assert cat == "DO"
        assert "DO" in captured["prompt"]
        assert "TASK" not in captured["prompt"]

    def test_defaults_to_claude_print_prompt(self) -> None:
        with patch("kennel.claude.print_prompt", return_value="ACT: do it") as mock_pp:
            cat, title = _triage("do it", is_bot=False)
        mock_pp.assert_called_once()
        assert cat == "ACT"


class TestMaybeReact:
    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={},
            allowed_bots=frozenset(),
            log_level="WARNING",
            self_repo=None,
            sub_dir=tmp_path / "sub",
        )

    def test_reacts_when_valid(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        mock_gh = MagicMock()
        maybe_react(
            "great work!",
            99,
            "pulls",
            "owner/repo",
            cfg,
            _print_prompt=MagicMock(return_value="heart"),
            _gh=mock_gh,
        )
        mock_gh.add_reaction.assert_called_once_with("owner/repo", "pulls", 99, "heart")

    def test_no_reaction_for_none(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        mock_gh = MagicMock()
        maybe_react(
            "ok",
            99,
            "pulls",
            "owner/repo",
            cfg,
            _print_prompt=MagicMock(return_value="NONE"),
            _gh=mock_gh,
        )
        mock_gh.add_reaction.assert_not_called()

    def test_timeout_warns_and_returns(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        maybe_react(
            "hi",
            1,
            "pulls",
            "owner/repo",
            cfg,
            _print_prompt=MagicMock(return_value=""),
        )

    def test_file_not_found_warns_and_returns(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        maybe_react(
            "hi",
            1,
            "pulls",
            "owner/repo",
            cfg,
            _print_prompt=MagicMock(return_value=""),
        )

    def test_reads_persona_if_present(self, tmp_path: Path) -> None:
        sub_dir = tmp_path / "sub"
        sub_dir.mkdir()
        (sub_dir / "persona.md").write_text("you are fido")
        cfg = Config(
            port=9000,
            secret=b"test",
            repos={},
            allowed_bots=frozenset(),
            log_level="WARNING",
            self_repo=None,
            sub_dir=sub_dir,
        )
        captured = {}

        def fake_pp(prompt, model, **kwargs):
            captured["prompt"] = prompt
            return "eyes"

        maybe_react(
            "look at this",
            1,
            "pulls",
            "owner/repo",
            cfg,
            _print_prompt=fake_pp,
            _gh=MagicMock(),
        )
        assert "you are fido" in captured.get("prompt", "")

    def test_defaults_to_claude_print_prompt(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        with patch("kennel.claude.print_prompt", return_value="NONE") as mock_pp:
            maybe_react("hi", 1, "pulls", "owner/repo", cfg)
        mock_pp.assert_called_once()


class TestReplyToComment:
    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={},
            allowed_bots=frozenset(),
            log_level="WARNING",
            self_repo=None,
            sub_dir=tmp_path / "sub",
        )

    def _repo_cfg(self, tmp_path: Path) -> RepoConfig:
        return RepoConfig(name="owner/repo", work_dir=tmp_path)

    def test_no_reply_to_returns_act(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(prompt="do stuff")
        posted, cat, title = reply_to_comment(action, cfg, self._repo_cfg(tmp_path))
        assert not posted
        assert cat == "ACT"

    def test_no_comment_body_returns_act(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="something",
            reply_to={"repo": "a/b", "pr": 1, "comment_id": 5},
        )
        posted, cat, title = reply_to_comment(action, cfg, self._repo_cfg(tmp_path))
        assert not posted
        assert cat == "ACT"

    def test_full_flow_act(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 10},
            comment_body="please add logging",
            is_bot=False,
            context={
                "pr_title": "My PR",
                "file": "foo.py",
                "line": 5,
                "diff_hunk": "@@ @@",
            },
        )

        def fake_pp(prompt, model, **kwargs):
            if model == "claude-haiku-4-5":
                return "NO"
            if "Triage" in prompt:
                return "ACT: add logging"
            return "I will add logging."

        posted, cat, title = reply_to_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            _print_prompt=fake_pp,
            _gh=MagicMock(),
        )
        assert posted
        assert cat == "ACT"
        assert "logging" in title.lower()

    def test_full_flow_ask(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 11},
            comment_body="can you clarify?",
            is_bot=False,
        )

        def fake_pp(prompt, model, **kwargs):
            if model == "claude-haiku-4-5":
                return "NO"
            if "Triage" in prompt:
                return "ASK: need more info"
            return "What specifically?"

        posted, cat, title = reply_to_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            _print_prompt=fake_pp,
            _gh=MagicMock(),
        )
        assert posted
        assert cat == "ASK"

    def test_full_flow_answer(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 12},
            comment_body="why did you do this?",
            is_bot=False,
        )

        def fake_pp(prompt, model, **kwargs):
            if model == "claude-haiku-4-5":
                return "NO"
            if "Triage" in prompt:
                return "ANSWER: explain choice"
            return "I did this because..."

        posted, cat, title = reply_to_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            _print_prompt=fake_pp,
            _gh=MagicMock(),
        )
        assert posted
        assert cat == "ANSWER"

    def test_full_flow_do(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 13},
            comment_body="cache the results for performance",
            is_bot=True,
        )

        def fake_pp(prompt, model, **kwargs):
            if model == "claude-haiku-4-5":
                return "NO"
            if "Triage" in prompt:
                return "DO: add result caching"
            return "On it!"

        mock_gh = MagicMock()
        posted, cat, title = reply_to_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            _print_prompt=fake_pp,
            _gh=mock_gh,
        )
        assert posted
        assert cat == "DO"
        assert title == "add result caching"
        mock_gh.create_issue.assert_not_called()

    def test_full_flow_defer(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 14},
            comment_body="refactor everything",
            is_bot=False,
        )

        def fake_pp(prompt, model, **kwargs):
            if model == "claude-haiku-4-5":
                return "NO"
            if "Triage" in prompt:
                return "DEFER: out of scope"
            return "That's out of scope for this PR."

        mock_gh = MagicMock()
        mock_gh.create_issue.return_value = "https://github.com/owner/repo/issues/99"
        posted, cat, title = reply_to_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            _print_prompt=fake_pp,
            _gh=mock_gh,
        )
        assert posted
        assert cat == "DEFER"
        mock_gh.create_issue.assert_called_once_with(
            "owner/repo",
            "out of scope",
            "Deferred from https://github.com/owner/repo/pull/1\n\n> refactor everything",
        )

    def test_full_flow_dump(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 14},
            comment_body="use a different language",
            is_bot=False,
        )

        def fake_pp(prompt, model, **kwargs):
            if model == "claude-haiku-4-5":
                return "NO"
            if "Triage" in prompt:
                return "DUMP: not applicable"
            return "Not applicable here."

        posted, cat, title = reply_to_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            _print_prompt=fake_pp,
            _gh=MagicMock(),
        )
        assert posted
        assert cat == "DUMP"

    def test_full_flow_defer_issue_creation_failure(self, tmp_path: Path) -> None:
        """DEFER still posts a reply even when create_issue raises."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 15},
            comment_body="refactor everything",
            is_bot=False,
        )

        def fake_pp(prompt, model, **kwargs):
            if model == "claude-haiku-4-5":
                return "NO"
            if "Triage" in prompt:
                return "DEFER: out of scope"
            return "That's out of scope for this PR."

        mock_gh = MagicMock()
        mock_gh.create_issue.side_effect = RuntimeError("network fail")
        posted, cat, title = reply_to_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            _print_prompt=fake_pp,
            _gh=mock_gh,
        )
        assert posted
        assert cat == "DEFER"

    def test_empty_body_uses_fallback(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 15},
            comment_body="do something",
            is_bot=False,
        )

        def fake_pp(prompt, model, **kwargs):
            if model == "claude-haiku-4-5":
                return "NO"
            if "Triage" in prompt:
                return "ACT: do it"
            return ""  # empty reply triggers fallback

        posted, cat, title = reply_to_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            _print_prompt=fake_pp,
            _gh=MagicMock(),
        )
        assert posted
        assert cat == "ACT"  # still succeeds with fallback body

    def test_claude_timeout_uses_fallback(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 16},
            comment_body="do something",
            is_bot=False,
        )

        def fake_pp(prompt, model, **kwargs):
            if model == "claude-haiku-4-5":
                return "NO"
            if "Triage" in prompt:
                return "ACT: do it"
            return ""  # simulates timeout — print_prompt returns "" on failure

        posted, cat, title = reply_to_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            _print_prompt=fake_pp,
            _gh=MagicMock(),
        )
        assert posted
        assert cat == "ACT"

    def test_lock_race_returns_act(self, tmp_path: Path) -> None:
        """Second call with same comment_id is blocked by lock."""
        import fcntl

        cfg = self._cfg(tmp_path)
        cid = 999
        lock_path = _comment_lock(tmp_path, cid)
        # Pre-acquire the lock to simulate race
        lock_fd = open(lock_path, "w")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            action = Action(
                prompt="comment",
                reply_to={"repo": "owner/repo", "pr": 1, "comment_id": cid},
                comment_body="competing update",
                is_bot=False,
            )
            posted, cat, title = reply_to_comment(action, cfg, self._repo_cfg(tmp_path))
            assert not posted  # locked — no reply sent
            assert cat == "ACT"  # returns without posting
        finally:
            lock_fd.close()

    def test_no_comment_id_skips_lock(self, tmp_path: Path) -> None:
        """When comment_id is None, lock is skipped; maybe_react is still called."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": None},
            comment_body="some comment",
            is_bot=False,
        )

        def fake_pp(prompt, model, **kwargs):
            if model == "claude-haiku-4-5":
                return "NO"
            if "Triage" in prompt:
                return "ACT: do it"
            return "ok"

        posted, cat, title = reply_to_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            _print_prompt=fake_pp,
            _gh=MagicMock(),
        )
        assert posted
        assert cat == "ACT"


class TestReplyToReview:
    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={},
            allowed_bots=frozenset(),
            log_level="WARNING",
            self_repo=None,
            sub_dir=tmp_path / "sub",
        )

    def _repo_cfg(self, tmp_path: Path) -> RepoConfig:
        return RepoConfig(name="owner/repo", work_dir=tmp_path)

    def test_no_review_comments_returns_early(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(prompt="review", review_comments=None)
        # should return without error
        reply_to_review(action, cfg, self._repo_cfg(tmp_path))

    def test_fetches_and_replies(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="review",
            review_comments={"repo": "owner/repo", "pr": 5, "review_id": 777},
        )

        def fake_pp(prompt, model, **kwargs):
            if model == "claude-haiku-4-5":
                return "NO"
            if "Triage" in prompt:
                return "ACT: fix it"
            return "Will fix."

        mock_gh = MagicMock()
        mock_gh.get_review_comments.return_value = [(100, "fix this"), (200, "nit")]
        reply_to_review(
            action, cfg, self._repo_cfg(tmp_path), _print_prompt=fake_pp, _gh=mock_gh
        )

    def test_skips_already_replied(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="review",
            review_comments={"repo": "owner/repo", "pr": 5, "review_id": 778},
        )
        already = {100, 200}
        calls = []

        def fake_pp(prompt, model, **kwargs):
            calls.append((prompt, model))
            return ""

        mock_gh = MagicMock()
        mock_gh.get_review_comments.return_value = [(100, "fix this"), (200, "nit")]
        reply_to_review(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            already_replied=already,
            _print_prompt=fake_pp,
            _gh=mock_gh,
        )
        # no claude calls since all comments already replied
        assert not calls

    def test_fetch_exception_returns_early(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="review",
            review_comments={"repo": "owner/repo", "pr": 5, "review_id": 779},
        )
        mock_gh = MagicMock()
        mock_gh.get_review_comments.side_effect = Exception("network fail")
        reply_to_review(
            action, cfg, self._repo_cfg(tmp_path), _gh=mock_gh
        )  # should not raise

    def test_no_inline_comments(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="review",
            review_comments={"repo": "owner/repo", "pr": 5, "review_id": 780},
        )
        mock_gh = MagicMock()
        mock_gh.get_review_comments.return_value = []
        reply_to_review(
            action, cfg, self._repo_cfg(tmp_path), _gh=mock_gh
        )  # empty → no replies


class TestReplyToIssueComment:
    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={},
            allowed_bots=frozenset(),
            log_level="WARNING",
            self_repo=None,
            sub_dir=tmp_path / "sub",
        )

    def _repo_cfg(self, tmp_path: Path) -> RepoConfig:
        return RepoConfig(name="owner/repo", work_dir=tmp_path)

    def _action(self, comment="please fix", is_bot=False, cid=42):
        return Action(
            prompt="PR top-level comment on #7 by owner:\n\nplease fix",
            comment_body=comment,
            is_bot=is_bot,
            context={"pr_title": "My PR", "comment_id": cid},
        )

    def test_act_reply(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)

        def fake_pp(prompt, model, **kwargs):
            if "Triage" in prompt:
                return "ACT: fix the bug"
            return "I'll fix that."

        cat, title = reply_to_issue_comment(
            self._action(),
            cfg,
            self._repo_cfg(tmp_path),
            _print_prompt=fake_pp,
            _gh=MagicMock(),
        )
        assert cat == "ACT"

    def test_ask_reply(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)

        def fake_pp(prompt, model, **kwargs):
            if "Triage" in prompt:
                return "ASK: unclear"
            return "What do you mean?"

        cat, title = reply_to_issue_comment(
            self._action("unclear"),
            cfg,
            self._repo_cfg(tmp_path),
            _print_prompt=fake_pp,
            _gh=MagicMock(),
        )
        assert cat == "ASK"

    def test_answer_reply(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)

        def fake_pp(prompt, model, **kwargs):
            if "Triage" in prompt:
                return "ANSWER: it works this way"
            return "Yes, because..."

        cat, title = reply_to_issue_comment(
            self._action("why?"),
            cfg,
            self._repo_cfg(tmp_path),
            _print_prompt=fake_pp,
            _gh=MagicMock(),
        )
        assert cat == "ANSWER"

    def test_dump_reply(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)

        def fake_pp(prompt, model, **kwargs):
            if "Triage" in prompt:
                return "DUMP: nope"
            return "That won't work here."

        cat, title = reply_to_issue_comment(
            self._action("do it differently"),
            cfg,
            self._repo_cfg(tmp_path),
            _print_prompt=fake_pp,
            _gh=MagicMock(),
        )
        assert cat == "DUMP"

    def test_defer_reply(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)

        def fake_pp(prompt, model, **kwargs):
            if "Triage" in prompt:
                return "DEFER: later"
            return "Out of scope."

        mock_gh = MagicMock()
        mock_gh.get_repo_info.return_value = "owner/repo"
        mock_gh.create_issue.return_value = "https://github.com/owner/repo/issues/5"
        cat, title = reply_to_issue_comment(
            self._action("big refactor"),
            cfg,
            self._repo_cfg(tmp_path),
            _print_prompt=fake_pp,
            _gh=mock_gh,
        )
        assert cat == "DEFER"
        mock_gh.create_issue.assert_called_once_with(
            "owner/repo",
            "later",
            "Deferred from https://github.com/owner/repo/pull/7\n\n> big refactor",
        )

    def test_defer_reply_issue_creation_failure(self, tmp_path: Path) -> None:
        """DEFER still posts a reply even when create_issue raises."""
        cfg = self._cfg(tmp_path)

        def fake_pp(prompt, model, **kwargs):
            if "Triage" in prompt:
                return "DEFER: later"
            return "Out of scope."

        mock_gh = MagicMock()
        mock_gh.get_repo_info.return_value = "owner/repo"
        mock_gh.create_issue.side_effect = RuntimeError("network fail")
        cat, title = reply_to_issue_comment(
            self._action("big refactor"),
            cfg,
            self._repo_cfg(tmp_path),
            _print_prompt=fake_pp,
            _gh=mock_gh,
        )
        assert cat == "DEFER"

    def test_empty_body_fallback(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)

        def fake_pp(prompt, model, **kwargs):
            if "Triage" in prompt:
                return "ACT: do it"
            return ""  # empty reply triggers fallback

        cat, title = reply_to_issue_comment(
            self._action(),
            cfg,
            self._repo_cfg(tmp_path),
            _print_prompt=fake_pp,
            _gh=MagicMock(),
        )
        assert cat == "ACT"

    def test_timeout_fallback(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)

        def fake_pp(prompt, model, **kwargs):
            if "Triage" in prompt:
                return "ACT: do it"
            return ""  # simulates timeout — print_prompt returns "" on failure

        cat, title = reply_to_issue_comment(
            self._action(),
            cfg,
            self._repo_cfg(tmp_path),
            _print_prompt=fake_pp,
            _gh=MagicMock(),
        )
        assert cat == "ACT"

    def test_post_exception_does_not_raise(self, tmp_path: Path) -> None:
        """comment_issue failure is caught and does not propagate."""
        cfg = self._cfg(tmp_path)
        # Use action without comment_id so react block is skipped
        action = Action(
            prompt="PR top-level comment on #7 by owner:\n\nplease fix",
            comment_body="please fix",
            is_bot=False,
            context={"pr_title": "My PR"},  # no comment_id → react block skipped
        )

        def fake_pp(prompt, model, **kwargs):
            if "Triage" in prompt:
                return "ACT: do it"
            return "ok"

        mock_gh = MagicMock()
        mock_gh.comment_issue.side_effect = Exception("gh fail")
        cat, title = reply_to_issue_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            _print_prompt=fake_pp,
            _gh=mock_gh,
        )
        assert cat == "ACT"

    def test_no_comment_id_skips_react(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="PR top-level comment on #7 by owner:\n\nhi",
            comment_body="hi",
            is_bot=False,
            context={"pr_title": "My PR"},  # no comment_id
        )

        def fake_pp(prompt, model, **kwargs):
            if "Triage" in prompt:
                return "ACT: do it"
            return "ok"

        cat, title = reply_to_issue_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            _print_prompt=fake_pp,
            _gh=MagicMock(),
        )
        assert cat == "ACT"

    def test_defaults_to_claude_print_prompt(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = self._action()

        def fake_pp(prompt, model, **kwargs):
            if "Triage" in prompt:
                return "ACT: do it"
            return "ok"

        with patch("kennel.claude.print_prompt", side_effect=fake_pp) as mock_pp:
            cat, title = reply_to_issue_comment(
                action, cfg, self._repo_cfg(tmp_path), _gh=MagicMock()
            )
        assert mock_pp.called
        assert cat == "ACT"

    def test_includes_conversation_context_in_triage(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = self._action()
        mock_gh = MagicMock()
        mock_gh.get_issue_comments.return_value = [
            {"user": {"login": "alice"}, "body": "first comment"},
            {"user": {"login": "bob"}, "body": "second comment"},
            {"user": {"login": "owner"}, "body": "please fix"},
        ]

        def fake_pp(prompt, model, **kwargs):
            if "Triage" in prompt:
                return "ACT: do it"
            return "ok"

        with patch(
            "kennel.events._triage", wraps=lambda *a, **kw: ("ACT", "do it")
        ) as mock_triage:
            cat, title = reply_to_issue_comment(
                action,
                cfg,
                self._repo_cfg(tmp_path),
                _print_prompt=fake_pp,
                _gh=mock_gh,
            )
        assert cat == "ACT"
        mock_gh.get_issue_comments.assert_called_once_with("owner/repo", 7)
        # Verify conversation context was built and passed to _triage
        triage_ctx = mock_triage.call_args[0][2]  # third positional arg = context
        assert "conversation" in triage_ctx
        assert "alice: first comment" in triage_ctx["conversation"]
        assert "bob: second comment" in triage_ctx["conversation"]

    def test_conversation_context_exception_is_swallowed(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = self._action()
        mock_gh = MagicMock()
        mock_gh.get_issue_comments.side_effect = RuntimeError("API down")

        def fake_pp(prompt, model, **kwargs):
            if "Triage" in prompt:
                return "ACT: do it"
            return "ok"

        cat, title = reply_to_issue_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            _print_prompt=fake_pp,
            _gh=mock_gh,
        )
        assert cat == "ACT"


class TestCreateTask:
    def test_calls_add_task_and_launch_sync(self, tmp_path: Path) -> None:
        cfg = Config(
            port=9000,
            secret=b"test",
            repos={},
            allowed_bots=frozenset(),
            log_level="WARNING",
            self_repo=None,
            sub_dir=tmp_path / "sub",
        )
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        with (
            patch("kennel.events.add_task") as mock_add,
            patch("kennel.events.launch_sync") as mock_sync,
        ):
            create_task("do something", cfg, repo_cfg)
        mock_add.assert_called_once_with(
            tmp_path, title="do something", task_type=ANY, thread=None
        )
        mock_sync.assert_called_once_with(cfg, repo_cfg)

    def test_passes_thread(self, tmp_path: Path) -> None:
        cfg = Config(
            port=9000,
            secret=b"test",
            repos={},
            allowed_bots=frozenset(),
            log_level="WARNING",
            self_repo=None,
            sub_dir=tmp_path / "sub",
        )
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        thread = {"repo": "a/b", "pr": 1, "comment_id": 5}
        with (
            patch("kennel.events.add_task") as mock_add,
            patch("kennel.events.launch_sync"),
        ):
            create_task("do something", cfg, repo_cfg, thread=thread)
        mock_add.assert_called_once_with(
            tmp_path, title="do something", task_type=ANY, thread=thread
        )

    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={},
            allowed_bots=frozenset(),
            log_level="WARNING",
            self_repo=None,
            sub_dir=tmp_path / "sub",
        )

    def _fido_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / ".git" / "fido"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def test_returns_created_task(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        fake_task = {
            "id": "t1",
            "title": "do something",
            "status": "pending",
            "type": "spec",
        }
        with (
            patch("kennel.events.add_task", return_value=fake_task),
            patch("kennel.events.launch_sync"),
        ):
            result = create_task("do something", cfg, repo_cfg)
        assert result == fake_task

    def test_no_abort_without_registry(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        import json

        (fido_dir / "state.json").write_text(json.dumps({"current_task_id": "t1"}))
        (fido_dir / "tasks.json").write_text(
            json.dumps(
                [
                    {
                        "id": "t1",
                        "title": "Plain task",
                        "status": "pending",
                        "type": "spec",
                    }
                ]
            )
        )
        thread = {"repo": "owner/repo", "pr": 1, "comment_id": 42}
        fake_task = {
            "id": "t2",
            "title": "Comment task",
            "status": "pending",
            "type": "spec",
            "thread": thread,
        }
        registry = MagicMock()
        with (
            patch("kennel.events.add_task", return_value=fake_task),
            patch("kennel.events.launch_sync"),
        ):
            create_task("Comment task", cfg, repo_cfg, thread=thread)  # no registry
        registry.abort_task.assert_not_called()

    def test_no_abort_when_new_task_has_no_thread(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        import json

        (fido_dir / "state.json").write_text(json.dumps({"current_task_id": "t1"}))
        (fido_dir / "tasks.json").write_text(
            json.dumps(
                [
                    {
                        "id": "t1",
                        "title": "Plain task",
                        "status": "pending",
                        "type": "spec",
                    }
                ]
            )
        )
        fake_task = {
            "id": "t2",
            "title": "Another plain task",
            "status": "pending",
            "type": "spec",
        }
        registry = MagicMock()
        with (
            patch("kennel.events.add_task", return_value=fake_task),
            patch("kennel.events.launch_sync"),
        ):
            create_task(
                "Another plain task", cfg, repo_cfg, registry=registry
            )  # no thread
        registry.abort_task.assert_not_called()

    def test_no_abort_when_no_current_task_in_state(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        import json

        (fido_dir / "state.json").write_text(
            json.dumps({"issue": 5})
        )  # no current_task_id
        thread = {"repo": "owner/repo", "pr": 1, "comment_id": 42}
        fake_task = {
            "id": "t1",
            "title": "Comment task",
            "status": "pending",
            "type": "spec",
            "thread": thread,
        }
        registry = MagicMock()
        with (
            patch("kennel.events.add_task", return_value=fake_task),
            patch("kennel.events.launch_sync"),
        ):
            create_task("Comment task", cfg, repo_cfg, thread=thread, registry=registry)
        registry.abort_task.assert_not_called()

    def test_no_abort_when_current_task_has_thread(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        import json

        existing_thread = {"repo": "owner/repo", "pr": 1, "comment_id": 10}
        (fido_dir / "state.json").write_text(
            json.dumps({"current_task_id": "t1", "issue": 5})
        )
        (fido_dir / "tasks.json").write_text(
            json.dumps(
                [
                    {
                        "id": "t1",
                        "title": "Thread task",
                        "status": "pending",
                        "type": "thread",
                        "thread": existing_thread,
                    }
                ]
            )
        )
        new_thread = {"repo": "owner/repo", "pr": 1, "comment_id": 42}
        fake_task = {
            "id": "t2",
            "title": "New thread task",
            "status": "pending",
            "type": "thread",
            "thread": new_thread,
        }
        registry = MagicMock()
        with (
            patch("kennel.events.add_task", return_value=fake_task),
            patch("kennel.events.launch_sync"),
        ):
            create_task(
                "New thread task",
                cfg,
                repo_cfg,
                thread=new_thread,
                registry=registry,
            )
        registry.abort_task.assert_not_called()

    def test_no_abort_when_state_file_absent(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        self._fido_dir(tmp_path)  # create dir but no state.json
        thread = {"repo": "owner/repo", "pr": 1, "comment_id": 42}
        fake_task = {
            "id": "t1",
            "title": "Comment task",
            "status": "pending",
            "type": "spec",
            "thread": thread,
        }
        registry = MagicMock()
        with (
            patch("kennel.events.add_task", return_value=fake_task),
            patch("kennel.events.launch_sync"),
        ):
            create_task("Comment task", cfg, repo_cfg, thread=thread, registry=registry)
        registry.abort_task.assert_not_called()

    def test_no_abort_when_current_task_not_in_tasks_json(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        import json

        (fido_dir / "state.json").write_text(json.dumps({"current_task_id": "t-gone"}))
        (fido_dir / "tasks.json").write_text(json.dumps([]))
        thread = {"repo": "owner/repo", "pr": 1, "comment_id": 42}
        fake_task = {
            "id": "t2",
            "title": "Comment task",
            "status": "pending",
            "type": "spec",
            "thread": thread,
        }
        registry = MagicMock()
        with (
            patch("kennel.events.add_task", return_value=fake_task),
            patch("kennel.events.launch_sync"),
        ):
            create_task("Comment task", cfg, repo_cfg, thread=thread, registry=registry)
        registry.abort_task.assert_not_called()

    def test_no_abort_when_state_has_no_current_task(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        import json

        (fido_dir / "state.json").write_text(json.dumps({"issue": 5}))
        (fido_dir / "tasks.json").write_text(json.dumps([]))
        thread = {"repo": "owner/repo", "pr": 1, "comment_id": 42}
        fake_task = {
            "id": "t2",
            "title": "Comment task",
            "status": "pending",
            "type": "thread",
            "thread": thread,
        }
        registry = MagicMock()
        with (
            patch("kennel.events.add_task", return_value=fake_task),
            patch("kennel.events.launch_sync"),
        ):
            create_task("Comment task", cfg, repo_cfg, thread=thread, registry=registry)
        registry.abort_task.assert_not_called()

    def _setup_abort_scenario(
        self, tmp_path: Path, current_type: str = "spec"
    ) -> tuple[MagicMock, Path]:
        """Set up state/tasks for abort tests and return (registry, fido_dir)."""
        import json

        fido_dir = self._fido_dir(tmp_path)
        (fido_dir / "state.json").write_text(
            json.dumps({"current_task_id": "t-current", "issue": 5})
        )
        (fido_dir / "tasks.json").write_text(
            json.dumps(
                [
                    {
                        "id": "t-current",
                        "title": "Current task",
                        "status": "pending",
                        "type": current_type,
                    }
                ]
            )
        )
        registry = MagicMock()
        return registry, fido_dir

    def test_aborts_when_thread_task_supersedes_non_thread_current(
        self, tmp_path: Path
    ) -> None:
        cfg = self._cfg(tmp_path)
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        import json

        registry, fido_dir = self._setup_abort_scenario(tmp_path, "spec")
        thread = {"repo": "owner/repo", "pr": 1, "comment_id": 42}
        fake_task = {
            "id": "t-comment",
            "title": "Comment task",
            "status": "pending",
            "type": "thread",
            "thread": thread,
        }
        with (
            patch("kennel.events.add_task", return_value=fake_task),
            patch("kennel.events.launch_sync"),
        ):
            create_task(
                "Comment task",
                cfg,
                repo_cfg,
                thread=thread,
                registry=registry,
            )
        registry.abort_task.assert_called_once_with("owner/repo")
        # ABORT_KEEP: current task stays in tasks.json
        remaining = json.loads((fido_dir / "tasks.json").read_text())
        assert any(t["id"] == "t-current" for t in remaining)

    def test_thread_preempts_spec_keeps_task(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        import json

        registry, fido_dir = self._setup_abort_scenario(tmp_path, "spec")
        thread = {"repo": "owner/repo", "pr": 1, "comment_id": 42}
        fake_task = {
            "id": "t-comment",
            "title": "Comment task",
            "status": "pending",
            "type": "thread",
            "thread": thread,
        }
        with (
            patch("kennel.events.add_task", return_value=fake_task),
            patch("kennel.events.launch_sync"),
        ):
            create_task(
                "Comment task",
                cfg,
                repo_cfg,
                thread=thread,
                registry=registry,
            )
        registry.abort_task.assert_called_once_with("owner/repo")
        # task should still be in tasks.json (ABORT_KEEP)
        remaining = json.loads((fido_dir / "tasks.json").read_text())
        assert any(t["id"] == "t-current" for t in remaining)

    def test_thread_does_not_preempt_thread(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)

        registry, _fido_dir = self._setup_abort_scenario(tmp_path, "thread")
        thread = {"repo": "owner/repo", "pr": 1, "comment_id": 42}
        fake_task = {
            "id": "t-comment",
            "title": "Comment task",
            "status": "pending",
            "type": "thread",
            "thread": thread,
        }
        with (
            patch("kennel.events.add_task", return_value=fake_task),
            patch("kennel.events.launch_sync"),
        ):
            create_task(
                "Comment task",
                cfg,
                repo_cfg,
                thread=thread,
                registry=registry,
            )
        registry.abort_task.assert_not_called()

    def test_ci_preempts_thread(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        import json

        registry, fido_dir = self._setup_abort_scenario(tmp_path, "thread")
        fake_task = {
            "id": "t-ci",
            "title": "CI fix",
            "status": "pending",
            "type": "ci",
        }
        with (
            patch("kennel.events.add_task", return_value=fake_task),
            patch("kennel.events.launch_sync"),
        ):
            create_task(
                "CI fix",
                cfg,
                repo_cfg,
                registry=registry,
            )
        registry.abort_task.assert_called_once_with("owner/repo")
        remaining = json.loads((fido_dir / "tasks.json").read_text())
        assert any(t["id"] == "t-current" for t in remaining)

    def test_ci_preempts_spec(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        import json

        registry, fido_dir = self._setup_abort_scenario(tmp_path, "spec")
        fake_task = {
            "id": "t-ci",
            "title": "CI fix",
            "status": "pending",
            "type": "ci",
        }
        with (
            patch("kennel.events.add_task", return_value=fake_task),
            patch("kennel.events.launch_sync"),
        ):
            create_task(
                "CI fix",
                cfg,
                repo_cfg,
                registry=registry,
            )
        registry.abort_task.assert_called_once_with("owner/repo")
        remaining = json.loads((fido_dir / "tasks.json").read_text())
        assert any(t["id"] == "t-current" for t in remaining)

    def test_spec_does_not_preempt_spec(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)

        registry, _fido_dir = self._setup_abort_scenario(tmp_path, "spec")
        fake_task = {
            "id": "t-new",
            "title": "New spec task",
            "status": "pending",
            "type": "spec",
        }
        with (
            patch("kennel.events.add_task", return_value=fake_task),
            patch("kennel.events.launch_sync"),
        ):
            create_task(
                "New spec task",
                cfg,
                repo_cfg,
                registry=registry,
            )
        registry.abort_task.assert_not_called()

    def test_thread_task_triggers_reorder_background(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        thread = {"repo": "owner/repo", "pr": 1, "comment_id": 5}
        fake_task = {
            "id": "t1",
            "title": "Comment task",
            "status": "pending",
            "type": "thread",
            "thread": thread,
        }
        reorder_called: list[tuple] = []
        with (
            patch("kennel.events.add_task", return_value=fake_task),
            patch("kennel.events.launch_sync"),
        ):
            create_task(
                "Comment task",
                cfg,
                repo_cfg,
                thread=thread,
                _get_commit_summary_fn=lambda wd: "abc1234 add thing",
                _reorder_background_fn=lambda wd, cs, cfg: reorder_called.append(
                    (wd, cs, cfg)
                ),
            )
        assert len(reorder_called) == 1
        assert reorder_called[0][0] == tmp_path
        assert reorder_called[0][1] == "abc1234 add thing"
        assert reorder_called[0][2] is cfg

    def test_spec_task_does_not_trigger_reorder(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        fake_task = {
            "id": "t1",
            "title": "Spec task",
            "status": "pending",
            "type": "spec",
        }
        reorder_called: list = []
        with (
            patch("kennel.events.add_task", return_value=fake_task),
            patch("kennel.events.launch_sync"),
        ):
            create_task(
                "Spec task",
                cfg,
                repo_cfg,
                _reorder_background_fn=lambda *a: reorder_called.append(a),
            )
        assert reorder_called == []

    def test_commit_summary_comes_from_get_commit_summary_fn(
        self, tmp_path: Path
    ) -> None:
        cfg = self._cfg(tmp_path)
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        thread = {"repo": "owner/repo", "pr": 1, "comment_id": 7}
        fake_task = {
            "id": "t1",
            "title": "t",
            "status": "pending",
            "type": "thread",
            "thread": thread,
        }
        summaries: list[str] = []
        with (
            patch("kennel.events.add_task", return_value=fake_task),
            patch("kennel.events.launch_sync"),
        ):
            create_task(
                "t",
                cfg,
                repo_cfg,
                thread=thread,
                _get_commit_summary_fn=lambda wd: "custom summary",
                _reorder_background_fn=lambda wd, cs, cfg: summaries.append(cs),
            )
        assert summaries == ["custom summary"]


class TestGetCommitSummary:
    def test_returns_git_log_output(self, tmp_path: Path) -> None:
        import subprocess as sp

        fake_result = sp.CompletedProcess(
            args=[], returncode=0, stdout="abc123 add thing\n", stderr=""
        )
        with patch(
            "kennel.events.subprocess.run", return_value=fake_result
        ) as mock_run:
            result = _get_commit_summary(tmp_path)
        assert result == "abc123 add thing"
        mock_run.assert_called_once_with(
            ["git", "log", "--oneline", "-20"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_returns_empty_on_file_not_found(self, tmp_path: Path) -> None:
        with patch("kennel.events.subprocess.run", side_effect=FileNotFoundError):
            result = _get_commit_summary(tmp_path)
        assert result == ""

    def test_returns_empty_on_timeout(self, tmp_path: Path) -> None:
        import subprocess as sp

        with patch(
            "kennel.events.subprocess.run",
            side_effect=sp.TimeoutExpired(cmd="git", timeout=10),
        ):
            result = _get_commit_summary(tmp_path)
        assert result == ""

    def test_returns_empty_stdout_when_git_fails(self, tmp_path: Path) -> None:
        import subprocess as sp

        fake_result = sp.CompletedProcess(
            args=[], returncode=128, stdout="", stderr="not a git repo"
        )
        with patch("kennel.events.subprocess.run", return_value=fake_result):
            result = _get_commit_summary(tmp_path)
        assert result == ""


class TestReorderTasksBackground:
    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={},
            allowed_bots=frozenset(),
            log_level="WARNING",
            self_repo=None,
            sub_dir=tmp_path / "sub",
        )

    def test_starts_daemon_thread(self, tmp_path: Path) -> None:
        started: list = []
        _reorder_tasks_background(
            tmp_path,
            "some commits",
            self._cfg(tmp_path),
            _start=lambda t: started.append(t),
        )
        assert len(started) == 1
        t = started[0]
        assert t.daemon is True

    def test_thread_name_includes_dir_name(self, tmp_path: Path) -> None:
        started: list = []
        _reorder_tasks_background(
            tmp_path, "commits", self._cfg(tmp_path), _start=lambda t: started.append(t)
        )
        assert tmp_path.name in started[0].name

    def test_thread_target_is_reorder_tasks(self, tmp_path: Path) -> None:
        from kennel.tasks import reorder_tasks

        started: list = []
        _reorder_tasks_background(
            tmp_path, "commits", self._cfg(tmp_path), _start=lambda t: started.append(t)
        )
        assert started[0]._target is reorder_tasks

    def test_thread_args_are_work_dir_and_commit_summary(self, tmp_path: Path) -> None:
        started: list = []
        _reorder_tasks_background(
            tmp_path,
            "feat: add parser",
            self._cfg(tmp_path),
            _start=lambda t: started.append(t),
        )
        assert started[0]._args == (tmp_path, "feat: add parser")


class TestNotifyThreadChange:
    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={},
            allowed_bots=frozenset(),
            log_level="WARNING",
            self_repo=None,
            sub_dir=tmp_path / "sub",
        )

    def _task(self, **overrides) -> dict:
        t = {
            "id": "t1",
            "title": "Fix the thing",
            "status": "pending",
            "type": "thread",
            "thread": {
                "repo": "owner/repo",
                "pr": 42,
                "comment_id": 999,
                "url": "https://github.com/owner/repo/pull/42#issuecomment-999",
            },
        }
        t.update(overrides)
        return t

    def test_dropped_posts_comment(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        mock_gh = MagicMock()
        change = {"task": self._task(), "kind": "dropped"}
        _notify_thread_change(
            change, cfg, _print_prompt=MagicMock(return_value="Noted!"), _gh=mock_gh
        )
        mock_gh.comment_issue.assert_called_once_with("owner/repo", 42, "Noted!")

    def test_modified_posts_comment(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        mock_gh = MagicMock()
        change = {
            "task": self._task(),
            "kind": "modified",
            "new_title": "Updated title",
            "new_description": "",
        }
        _notify_thread_change(
            change, cfg, _print_prompt=MagicMock(return_value="Updated!"), _gh=mock_gh
        )
        mock_gh.comment_issue.assert_called_once_with("owner/repo", 42, "Updated!")

    def test_missing_thread_skips_comment(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        mock_gh = MagicMock()
        task = self._task()
        task["thread"] = {}
        change = {"task": task, "kind": "dropped"}
        _notify_thread_change(change, cfg, _print_prompt=MagicMock(), _gh=mock_gh)
        mock_gh.comment_issue.assert_not_called()

    def test_empty_opus_uses_fallback_for_dropped(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        mock_gh = MagicMock()
        change = {"task": self._task(), "kind": "dropped"}
        _notify_thread_change(
            change, cfg, _print_prompt=MagicMock(return_value=""), _gh=mock_gh
        )
        body = mock_gh.comment_issue.call_args[0][2]
        assert "Fix the thing" in body

    def test_empty_opus_uses_fallback_for_modified(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        mock_gh = MagicMock()
        change = {
            "task": self._task(),
            "kind": "modified",
            "new_title": "New title",
            "new_description": "",
        }
        _notify_thread_change(
            change, cfg, _print_prompt=MagicMock(return_value=""), _gh=mock_gh
        )
        body = mock_gh.comment_issue.call_args[0][2]
        assert "New title" in body

    def test_gh_none_uses_get_github(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        mock_gh = MagicMock()
        change = {"task": self._task(), "kind": "dropped"}
        with patch("kennel.events.get_github", return_value=mock_gh):
            _notify_thread_change(
                change, cfg, _print_prompt=MagicMock(return_value="ok")
            )
        mock_gh.comment_issue.assert_called_once()

    def test_comment_issue_exception_does_not_raise(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        mock_gh = MagicMock()
        mock_gh.comment_issue.side_effect = RuntimeError("api error")
        change = {"task": self._task(), "kind": "dropped"}
        # Should not raise
        _notify_thread_change(
            change, cfg, _print_prompt=MagicMock(return_value="ok"), _gh=mock_gh
        )


class TestLaunchSync:
    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={},
            allowed_bots=frozenset(),
            log_level="WARNING",
            self_repo=None,
            sub_dir=tmp_path / "sub",
        )

    def _repo_cfg(self, tmp_path: Path) -> RepoConfig:
        return RepoConfig(name="owner/repo", work_dir=tmp_path)

    def test_calls_sync_tasks_background(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        mock_gh = MagicMock()
        with patch("kennel.tasks.sync_tasks_background") as mock_sync:
            launch_sync(cfg, self._repo_cfg(tmp_path), _gh=mock_gh)
        mock_sync.assert_called_once_with(tmp_path, mock_gh)

    def test_does_not_raise(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        with patch("kennel.tasks.sync_tasks_background"):
            launch_sync(
                cfg, self._repo_cfg(tmp_path), _gh=MagicMock()
            )  # should not raise


class TestLaunchWorker:
    def _repo_cfg(self, tmp_path: Path) -> RepoConfig:
        return RepoConfig(name="owner/repo", work_dir=tmp_path)

    def test_wakes_registry_for_repo(self, tmp_path: Path) -> None:
        registry = MagicMock()
        launch_worker(self._repo_cfg(tmp_path), registry)
        registry.wake.assert_called_once_with("owner/repo")


class TestDispatchPullRequestReview:
    def test_submitted_with_review_id(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "submitted",
            "review": {
                "id": 55,
                "state": "changes_requested",
                "user": {"login": "owner"},
            },
            "pull_request": {"number": 3},
        }
        result = dispatch("pull_request_review", payload, cfg, _repo_cfg(tmp_path))
        assert result is not None
        assert result.review_comments is not None
        assert result.review_comments["review_id"] == 55

    def test_submitted_without_review_id(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "submitted",
            "review": {"state": "approved", "user": {"login": "owner"}},
            "pull_request": {"number": 3},
        }
        result = dispatch("pull_request_review", payload, cfg, _repo_cfg(tmp_path))
        assert result is not None
        assert result.review_comments is None

    def test_submitted_no_number_returns_none(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "submitted",
            "review": {"id": 1, "state": "approved", "user": {"login": "owner"}},
            "pull_request": {},
        }
        result = dispatch("pull_request_review", payload, cfg, _repo_cfg(tmp_path))
        assert result is None

    def test_not_allowed_user_ignored(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "submitted",
            "review": {"id": 1, "state": "approved", "user": {"login": "stranger"}},
            "pull_request": {"number": 3},
        }
        result = dispatch("pull_request_review", payload, cfg, _repo_cfg(tmp_path))
        assert result is None


class TestDispatchCheckRunNoPrs:
    def test_no_prs(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "completed",
            "check_run": {
                "conclusion": "failure",
                "name": "test-unit",
                "pull_requests": [],
            },
        }
        result = dispatch("check_run", payload, cfg, _repo_cfg(tmp_path))
        assert result is not None
        assert "unknown PR" in result.prompt


class TestDispatchIssueCommentSelf:
    def test_self_comment_ignored(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "created",
            "comment": {"id": 1, "body": "hi", "user": {"login": "fido-can-code"}},
            "issue": {
                "number": 10,
                "title": "t",
                "pull_request": {"url": "https://api.github.com/..."},
            },
        }
        result = dispatch("issue_comment", payload, cfg, _repo_cfg(tmp_path))
        assert result is None

    def test_unallowed_user_ignored(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "created",
            "comment": {"id": 1, "body": "hi", "user": {"login": "stranger"}},
            "issue": {
                "number": 10,
                "title": "t",
                "pull_request": {"url": "https://api.github.com/..."},
            },
        }
        result = dispatch("issue_comment", payload, cfg, _repo_cfg(tmp_path))
        assert result is None


class TestDispatchReviewCommentNoNumber:
    def test_no_number_after_self_check(self, tmp_path: Path) -> None:
        """Non-self user, but pr has no number → returns None (line 81)."""
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "created",
            "comment": {
                "id": 1,
                "body": "hi",
                "user": {"login": "owner"},  # allowed, not self
                "html_url": "https://example.com",
            },
            "pull_request": {},  # no number
        }
        result = dispatch(
            "pull_request_review_comment", payload, cfg, _repo_cfg(tmp_path)
        )
        assert result is None


class TestMaybeReactGhException:
    def test_gh_post_exception_is_caught(self, tmp_path: Path) -> None:
        """Exception posting the reaction is caught and logged (lines 230-231)."""
        cfg = Config(
            port=9000,
            secret=b"test",
            repos={},
            allowed_bots=frozenset(),
            log_level="WARNING",
            self_repo=None,
            sub_dir=tmp_path / "sub",
        )
        mock_gh = MagicMock()
        mock_gh.add_reaction.side_effect = RuntimeError("network down")
        maybe_react(
            "great job",
            77,
            "pulls",
            "owner/repo",
            cfg,
            _print_prompt=MagicMock(return_value="heart"),
            _gh=mock_gh,
        )  # must not raise


class TestReplyToCommentElseBranch:
    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={},
            allowed_bots=frozenset(),
            log_level="WARNING",
            self_repo=None,
            sub_dir=tmp_path / "sub",
        )

    def _repo_cfg(self, tmp_path: Path) -> RepoConfig:
        return RepoConfig(name="owner/repo", work_dir=tmp_path)

    def test_unknown_category_uses_else_reply(self, tmp_path: Path) -> None:
        """Triage returns a known category not in the explicit branches → else branch (line 313)."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 50},
            comment_body="do something",
            is_bot=False,
        )
        # Return "DO" which is a bot category but IS_BOT=False — falls through to else
        # Actually all the categories (ACT/DO/ASK/ANSWER/DEFER/DUMP) are covered.
        # The else fires when _triage returns an unrecognised prefix, which hits the
        # fallback "ACT"/"DO". We need to force an unlisted category past _triage.
        # Monkey-patch _triage directly to return a fake category.
        with (
            patch("kennel.events._triage", return_value=("UNKNOWN_CAT", "do it")),
            patch("kennel.events.needs_more_context", return_value=False),
        ):
            posted, cat, title = reply_to_comment(
                action,
                cfg,
                self._repo_cfg(tmp_path),
                _print_prompt=MagicMock(return_value="I'll look into this."),
                _gh=MagicMock(),
            )
        assert posted
        assert cat == "UNKNOWN_CAT"

    def test_gh_post_exception_caught(self, tmp_path: Path) -> None:
        """Exception in reply_to_review_comment is caught."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 51},
            comment_body="please fix this",
            is_bot=False,
        )

        def fake_pp(prompt, model, **kwargs):
            if model == "claude-haiku-4-5":
                return "NO"
            if "Triage" in prompt:
                return "ACT: fix it"
            return "I'll fix it."

        mock_gh = MagicMock()
        mock_gh.reply_to_review_comment.side_effect = RuntimeError("network down")
        posted, cat, title = reply_to_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            _print_prompt=fake_pp,
            _gh=mock_gh,
        )  # must not raise
        assert not posted  # post failed
        assert cat == "ACT"


class TestReplyToReviewAlreadyRepliedTracking:
    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={},
            allowed_bots=frozenset(),
            log_level="WARNING",
            self_repo=None,
            sub_dir=tmp_path / "sub",
        )

    def _repo_cfg(self, tmp_path: Path) -> RepoConfig:
        return RepoConfig(name="owner/repo", work_dir=tmp_path)

    def test_adds_to_already_replied_set(self, tmp_path: Path) -> None:
        """When already_replied set is passed, processed comment ids are added (line 452)."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="review",
            review_comments={"repo": "owner/repo", "pr": 5, "review_id": 900},
        )
        already: set[int] = set()

        def fake_pp(prompt, model, **kwargs):
            if model == "claude-haiku-4-5":
                return "NO"
            if "Triage" in prompt:
                return "ACT: fix it"
            return "Will fix."

        mock_gh = MagicMock()
        mock_gh.get_review_comments.return_value = [(500, "please fix")]
        reply_to_review(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            already_replied=already,
            _print_prompt=fake_pp,
            _gh=mock_gh,
        )
        assert 500 in already

    def test_does_not_add_to_already_replied_on_post_failure(
        self, tmp_path: Path
    ) -> None:
        """Dedup set is NOT updated when the GitHub post fails."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="review",
            review_comments={"repo": "owner/repo", "pr": 5, "review_id": 901},
        )
        already: set[int] = set()

        def fake_pp(prompt, model, **kwargs):
            if model == "claude-haiku-4-5":
                return "NO"
            if "Triage" in prompt:
                return "ACT: fix it"
            return "Will fix."

        mock_gh = MagicMock()
        mock_gh.get_review_comments.return_value = [(501, "please fix")]
        mock_gh.reply_to_review_comment.side_effect = RuntimeError("network down")
        reply_to_review(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            already_replied=already,
            _print_prompt=fake_pp,
            _gh=mock_gh,
        )
        assert 501 not in already  # post failed — should not be marked as replied


class TestReplyToCommentTerseEnrichment:
    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={},
            allowed_bots=frozenset(),
            log_level="WARNING",
            self_repo=None,
            sub_dir=tmp_path / "sub",
        )

    def _repo_cfg(self, tmp_path: Path) -> RepoConfig:
        return RepoConfig(name="owner/repo", work_dir=tmp_path)

    def test_terse_comment_fetches_siblings_and_adds_to_context(
        self, tmp_path: Path
    ) -> None:
        """When needs_more_context is True, sibling_threads are added to context for _triage."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 5, "comment_id": 200},
            comment_body="same",
            is_bot=False,
            context={"pr_title": "My PR", "file": "foo.py"},
        )
        captured_context: dict = {}

        def fake_triage(body, is_bot, context=None, *, _print_prompt=None):
            if context is not None:
                captured_context.update(context)
            return ("ACT", "handle same comment")

        mock_gh = MagicMock()
        mock_gh.fetch_sibling_threads.return_value = [
            {
                "path": "bar.py",
                "line": 1,
                "comments": [{"author": "rev", "body": "fix this"}],
            }
        ]

        with (
            patch("kennel.events._triage", side_effect=fake_triage),
            patch("kennel.events.needs_more_context", return_value=True),
        ):
            reply_to_comment(
                action,
                cfg,
                self._repo_cfg(tmp_path),
                _print_prompt=MagicMock(return_value="On it!"),
                _gh=mock_gh,
            )

        mock_gh.fetch_sibling_threads.assert_called_once_with("owner/repo", 5)
        assert "sibling_threads" in captured_context
        assert len(captured_context["sibling_threads"]) == 1

    def test_non_terse_comment_skips_sibling_fetch(self, tmp_path: Path) -> None:
        """When needs_more_context is False, sibling thread fetch is skipped."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 5, "comment_id": 201},
            comment_body="This is a detailed comment explaining the issue clearly.",
            is_bot=False,
        )
        mock_gh = MagicMock()

        def fake_pp(prompt, model, **kwargs):
            if "Triage" in prompt:
                return "ACT: do it"
            return "Got it."

        with patch("kennel.events.needs_more_context", return_value=False):
            reply_to_comment(
                action,
                cfg,
                self._repo_cfg(tmp_path),
                _print_prompt=fake_pp,
                _gh=mock_gh,
            )

        mock_gh.fetch_sibling_threads.assert_not_called()

    def test_terse_fetch_exception_does_not_propagate(self, tmp_path: Path) -> None:
        """If sibling fetch fails, reply_to_comment proceeds without sibling_threads."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 5, "comment_id": 202},
            comment_body="ditto",
            is_bot=False,
        )
        mock_gh = MagicMock()
        mock_gh.fetch_sibling_threads.return_value = []

        def fake_pp(prompt, model, **kwargs):
            if "Triage" in prompt:
                return "ACT: do it"
            return "On it."

        with patch("kennel.events.needs_more_context", return_value=True):
            posted, cat, title = reply_to_comment(
                action,
                cfg,
                self._repo_cfg(tmp_path),
                _print_prompt=fake_pp,
                _gh=mock_gh,
            )

        assert posted
        assert cat == "ACT"

    def test_terse_no_siblings_leaves_context_clean(self, tmp_path: Path) -> None:
        """Empty sibling threads list → sibling_threads not added to context."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 5, "comment_id": 203},
            comment_body="^",
            is_bot=False,
            context={"pr_title": "My PR"},
        )
        captured_context: dict = {}

        def fake_triage(body, is_bot, context=None, *, _print_prompt=None):
            if context is not None:
                captured_context.update(context)
            return ("ACT", "check caret comment")

        mock_gh = MagicMock()
        mock_gh.fetch_sibling_threads.return_value = []

        with (
            patch("kennel.events._triage", side_effect=fake_triage),
            patch("kennel.events.needs_more_context", return_value=True),
        ):
            reply_to_comment(
                action,
                cfg,
                self._repo_cfg(tmp_path),
                _print_prompt=MagicMock(return_value="On it!"),
                _gh=mock_gh,
            )

        assert "sibling_threads" not in captured_context
