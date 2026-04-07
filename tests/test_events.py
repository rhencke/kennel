from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from kennel.config import Config, RepoConfig
from kennel.events import (
    Action,
    _comment_lock,
    _is_allowed,
    _triage,
    create_task,
    dispatch,
    launch_sync,
    launch_worker,
    maybe_react,
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
            "comment": {"id": 456, "body": "looks good", "user": {"login": "owner"}},
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


def _make_completed_run(stdout: str = "", returncode: int = 0) -> MagicMock:
    m = MagicMock()
    m.stdout = stdout
    m.returncode = returncode
    return m


class TestCommentLock:
    def test_creates_lock_file(self, tmp_path: Path) -> None:
        path = _comment_lock(tmp_path, 42)
        assert path == tmp_path / ".git" / "fido" / "comments" / "42.lock"
        assert path.parent.is_dir()


class TestTriage:
    def test_returns_parsed_category(self, tmp_path: Path) -> None:
        with patch(
            "subprocess.run", return_value=_make_completed_run("ACT: add tests\n")
        ):
            cat, title = _triage("please add tests", is_bot=False)
        assert cat == "ACT"
        assert title == "add tests"

    def test_fallback_on_bad_response(self, tmp_path: Path) -> None:
        with patch("subprocess.run", return_value=_make_completed_run("")):
            cat, title = _triage("do stuff", is_bot=False)
        assert cat == "ACT"

    def test_fallback_for_bot(self, tmp_path: Path) -> None:
        with patch("subprocess.run", side_effect=Exception("fail")):
            cat, title = _triage("do stuff", is_bot=True)
        assert cat == "DO"

    def test_with_context(self, tmp_path: Path) -> None:
        ctx = {"pr_title": "My PR", "file": "foo.py", "diff_hunk": "@@ -1 +1 @@"}
        with patch(
            "subprocess.run", return_value=_make_completed_run("DEFER: out of scope\n")
        ):
            cat, title = _triage("nit comment", is_bot=False, context=ctx)
        assert cat == "DEFER"

    def test_unrecognized_category_falls_back(self, tmp_path: Path) -> None:
        with patch(
            "subprocess.run", return_value=_make_completed_run("WEIRD: something")
        ):
            cat, title = _triage("hi", is_bot=False)
        assert cat == "ACT"

    def test_timeout_falls_back(self, tmp_path: Path) -> None:
        with patch(
            "subprocess.run", side_effect=subprocess.TimeoutExpired("claude", 15)
        ):
            cat, title = _triage("hi", is_bot=True)
        assert cat == "DO"

    def test_bot_categories_in_prompt(self, tmp_path: Path) -> None:
        """Ensure bot-specific categories (DO/DEFER/DUMP) are used when is_bot=True."""
        captured = {}

        def fake_run(args, **kwargs):
            captured["prompt"] = args[-1]
            return _make_completed_run("DO: implement feature")

        with patch("subprocess.run", side_effect=fake_run):
            cat, _ = _triage("implement feature", is_bot=True)
        assert cat == "DO"
        assert "DO" in captured["prompt"]


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

        with (
            patch("subprocess.run", return_value=_make_completed_run("heart\n")),
            patch("kennel.github.add_reaction") as mock_react,
        ):
            maybe_react("great work!", 99, "pulls", "owner/repo", cfg)
        mock_react.assert_called_once_with("owner/repo", "pulls", 99, "heart")

    def test_no_reaction_for_none(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)

        with (
            patch("subprocess.run", return_value=_make_completed_run("NONE\n")),
            patch("kennel.github.add_reaction") as mock_react,
        ):
            maybe_react("ok", 99, "pulls", "owner/repo", cfg)
        mock_react.assert_not_called()

    def test_timeout_silently_returns(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        with patch(
            "subprocess.run", side_effect=subprocess.TimeoutExpired("claude", 15)
        ):
            maybe_react("hi", 1, "pulls", "owner/repo", cfg)  # should not raise

    def test_file_not_found_silently_returns(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        with patch("subprocess.run", side_effect=FileNotFoundError):
            maybe_react("hi", 1, "pulls", "owner/repo", cfg)  # should not raise

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

        def fake_run(args, **kwargs):
            if "claude" in args:
                captured["prompt"] = args[-1]
            return _make_completed_run("eyes\n")

        with (
            patch("subprocess.run", side_effect=fake_run),
            patch("kennel.github.add_reaction"),
        ):
            maybe_react("look at this", 1, "pulls", "owner/repo", cfg)
        assert "you are fido" in captured.get("prompt", "")


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
        cat, title = reply_to_comment(action, cfg, self._repo_cfg(tmp_path))
        assert cat == "ACT"

    def test_no_comment_body_returns_act(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="something",
            reply_to={"repo": "a/b", "pr": 1, "comment_id": 5},
        )
        cat, title = reply_to_comment(action, cfg, self._repo_cfg(tmp_path))
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

        def fake_run(args, **kwargs):
            if "claude" in args:
                text = args[-1]
                if "Triage" in text:
                    return _make_completed_run("ACT: add logging\n")
                return _make_completed_run("I will add logging.\n")
            return _make_completed_run("")

        with (
            patch("subprocess.run", side_effect=fake_run),
            patch("kennel.github.reply_to_review_comment"),
            patch("kennel.github.add_reaction"),
        ):
            cat, title = reply_to_comment(action, cfg, self._repo_cfg(tmp_path))
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

        def fake_run(args, **kwargs):
            if "claude" in args:
                text = args[-1]
                if "Triage" in text:
                    return _make_completed_run("ASK: need more info\n")
                return _make_completed_run("What specifically?\n")
            return _make_completed_run("")

        with (
            patch("subprocess.run", side_effect=fake_run),
            patch("kennel.github.reply_to_review_comment"),
            patch("kennel.github.add_reaction"),
        ):
            cat, title = reply_to_comment(action, cfg, self._repo_cfg(tmp_path))
        assert cat == "ASK"

    def test_full_flow_answer(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 12},
            comment_body="why did you do this?",
            is_bot=False,
        )

        def fake_run(args, **kwargs):
            if "claude" in args:
                text = args[-1]
                if "Triage" in text:
                    return _make_completed_run("ANSWER: explain choice\n")
                return _make_completed_run("I did this because...\n")
            return _make_completed_run("")

        with (
            patch("subprocess.run", side_effect=fake_run),
            patch("kennel.github.reply_to_review_comment"),
            patch("kennel.github.add_reaction"),
        ):
            cat, title = reply_to_comment(action, cfg, self._repo_cfg(tmp_path))
        assert cat == "ANSWER"

    def test_full_flow_defer(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 13},
            comment_body="refactor everything",
            is_bot=False,
        )

        def fake_run(args, **kwargs):
            if "claude" in args:
                text = args[-1]
                if "Triage" in text:
                    return _make_completed_run("DEFER: out of scope\n")
                return _make_completed_run("That's out of scope for this PR.\n")
            return _make_completed_run("")

        with (
            patch("subprocess.run", side_effect=fake_run),
            patch("kennel.github.reply_to_review_comment"),
            patch("kennel.github.add_reaction"),
        ):
            cat, title = reply_to_comment(action, cfg, self._repo_cfg(tmp_path))
        assert cat == "DEFER"

    def test_full_flow_dump(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 14},
            comment_body="use a different language",
            is_bot=False,
        )

        def fake_run(args, **kwargs):
            if "claude" in args:
                text = args[-1]
                if "Triage" in text:
                    return _make_completed_run("DUMP: not applicable\n")
                return _make_completed_run("Not applicable here.\n")
            return _make_completed_run("")

        with (
            patch("subprocess.run", side_effect=fake_run),
            patch("kennel.github.reply_to_review_comment"),
            patch("kennel.github.add_reaction"),
        ):
            cat, title = reply_to_comment(action, cfg, self._repo_cfg(tmp_path))
        assert cat == "DUMP"

    def test_empty_body_uses_fallback(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 15},
            comment_body="do something",
            is_bot=False,
        )

        def fake_run(args, **kwargs):
            if "claude" in args:
                text = args[-1]
                if "Triage" in text:
                    return _make_completed_run("ACT: do it\n")
                # claude returns empty body
                return _make_completed_run("")
            return _make_completed_run("")

        with (
            patch("subprocess.run", side_effect=fake_run),
            patch("kennel.github.reply_to_review_comment"),
            patch("kennel.github.add_reaction"),
        ):
            cat, title = reply_to_comment(action, cfg, self._repo_cfg(tmp_path))
        assert cat == "ACT"  # still succeeds with fallback body

    def test_claude_timeout_uses_fallback(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 16},
            comment_body="do something",
            is_bot=False,
        )
        call_count = [0]

        def fake_run(args, **kwargs):
            call_count[0] += 1
            if "claude" in args:
                text = args[-1]
                if "Triage" in text:
                    return _make_completed_run("ACT: do it\n")
                raise subprocess.TimeoutExpired("claude", 30)
            return _make_completed_run("")

        with (
            patch("subprocess.run", side_effect=fake_run),
            patch("kennel.github.reply_to_review_comment"),
            patch("kennel.github.add_reaction"),
        ):
            cat, title = reply_to_comment(action, cfg, self._repo_cfg(tmp_path))
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
            cat, title = reply_to_comment(action, cfg, self._repo_cfg(tmp_path))
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

        def fake_run(args, **kwargs):
            if "claude" in args:
                text = args[-1]
                if "Triage" in text:
                    return _make_completed_run("ACT: do it\n")
                return _make_completed_run("ok\n")
            return _make_completed_run("")

        with (
            patch("subprocess.run", side_effect=fake_run),
            patch("kennel.github.reply_to_review_comment"),
            patch("kennel.github.add_reaction"),
        ):
            cat, title = reply_to_comment(action, cfg, self._repo_cfg(tmp_path))
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

        def fake_run(args, **kwargs):
            if "claude" in args:
                text = args[-1]
                if "Triage" in text:
                    return _make_completed_run("ACT: fix it\n")
                return _make_completed_run("Will fix.\n")
            return _make_completed_run("")

        with (
            patch("subprocess.run", side_effect=fake_run),
            patch("kennel.github.get_review_comments", return_value=[100, 200]),
            patch("kennel.github.reply_to_review_comment"),
            patch("kennel.github.add_reaction"),
        ):
            reply_to_review(action, cfg, self._repo_cfg(tmp_path))

    def test_skips_already_replied(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="review",
            review_comments={"repo": "owner/repo", "pr": 5, "review_id": 778},
        )
        already = {100, 200}
        calls = []

        def fake_run(args, **kwargs):
            calls.append(args)
            return _make_completed_run("")

        with (
            patch("subprocess.run", side_effect=fake_run),
            patch("kennel.github.get_review_comments", return_value=[100, 200]),
        ):
            reply_to_review(
                action, cfg, self._repo_cfg(tmp_path), already_replied=already
            )
        # no claude calls since all comments already replied
        assert not any("claude" in a for a in calls)

    def test_fetch_exception_returns_early(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="review",
            review_comments={"repo": "owner/repo", "pr": 5, "review_id": 779},
        )
        with patch(
            "kennel.github.get_review_comments", side_effect=Exception("network fail")
        ):
            reply_to_review(action, cfg, self._repo_cfg(tmp_path))  # should not raise

    def test_no_inline_comments(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="review",
            review_comments={"repo": "owner/repo", "pr": 5, "review_id": 780},
        )
        with patch("kennel.github.get_review_comments", return_value=[]):
            reply_to_review(action, cfg, self._repo_cfg(tmp_path))  # empty → no replies


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

        def fake_run(args, **kwargs):
            if "claude" in args:
                text = args[-1]
                if "Triage" in text:
                    return _make_completed_run("ACT: fix the bug\n")
                return _make_completed_run("I'll fix that.\n")
            return _make_completed_run("owner/repo\n")

        with (
            patch("subprocess.run", side_effect=fake_run),
            patch("kennel.github.comment_issue"),
            patch("kennel.github.add_reaction"),
        ):
            cat, title = reply_to_issue_comment(
                self._action(), cfg, self._repo_cfg(tmp_path)
            )
        assert cat == "ACT"

    def test_ask_reply(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)

        def fake_run(args, **kwargs):
            if "claude" in args:
                text = args[-1]
                if "Triage" in text:
                    return _make_completed_run("ASK: unclear\n")
                return _make_completed_run("What do you mean?\n")
            return _make_completed_run("owner/repo\n")

        with (
            patch("subprocess.run", side_effect=fake_run),
            patch("kennel.github.comment_issue"),
            patch("kennel.github.add_reaction"),
        ):
            cat, title = reply_to_issue_comment(
                self._action("unclear"), cfg, self._repo_cfg(tmp_path)
            )
        assert cat == "ASK"

    def test_answer_reply(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)

        def fake_run(args, **kwargs):
            if "claude" in args:
                text = args[-1]
                if "Triage" in text:
                    return _make_completed_run("ANSWER: it works this way\n")
                return _make_completed_run("Yes, because...\n")
            return _make_completed_run("owner/repo\n")

        with (
            patch("subprocess.run", side_effect=fake_run),
            patch("kennel.github.comment_issue"),
            patch("kennel.github.add_reaction"),
        ):
            cat, title = reply_to_issue_comment(
                self._action("why?"), cfg, self._repo_cfg(tmp_path)
            )
        assert cat == "ANSWER"

    def test_dump_reply(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)

        def fake_run(args, **kwargs):
            if "claude" in args:
                text = args[-1]
                if "Triage" in text:
                    return _make_completed_run("DUMP: nope\n")
                return _make_completed_run("That won't work here.\n")
            return _make_completed_run("owner/repo\n")

        with (
            patch("subprocess.run", side_effect=fake_run),
            patch("kennel.github.comment_issue"),
            patch("kennel.github.add_reaction"),
        ):
            cat, title = reply_to_issue_comment(
                self._action("do it differently"), cfg, self._repo_cfg(tmp_path)
            )
        assert cat == "DUMP"

    def test_defer_reply(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)

        def fake_run(args, **kwargs):
            if "claude" in args:
                text = args[-1]
                if "Triage" in text:
                    return _make_completed_run("DEFER: later\n")
                return _make_completed_run("Out of scope.\n")
            return _make_completed_run("owner/repo\n")

        with (
            patch("subprocess.run", side_effect=fake_run),
            patch("kennel.github.comment_issue"),
            patch("kennel.github.add_reaction"),
        ):
            cat, title = reply_to_issue_comment(
                self._action("big refactor"), cfg, self._repo_cfg(tmp_path)
            )
        assert cat == "DEFER"

    def test_empty_body_fallback(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)

        def fake_run(args, **kwargs):
            if "claude" in args:
                text = args[-1]
                if "Triage" in text:
                    return _make_completed_run("ACT: do it\n")
                return _make_completed_run("")  # empty reply triggers fallback
            return _make_completed_run("owner/repo\n")

        with (
            patch("subprocess.run", side_effect=fake_run),
            patch("kennel.github.comment_issue"),
            patch("kennel.github.add_reaction"),
        ):
            cat, title = reply_to_issue_comment(
                self._action(), cfg, self._repo_cfg(tmp_path)
            )
        assert cat == "ACT"

    def test_timeout_fallback(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)

        def fake_run(args, **kwargs):
            if "claude" in args:
                text = args[-1]
                if "Triage" in text:
                    return _make_completed_run("ACT: do it\n")
                raise subprocess.TimeoutExpired("claude", 30)
            return _make_completed_run("owner/repo\n")

        with (
            patch("subprocess.run", side_effect=fake_run),
            patch("kennel.github.comment_issue"),
            patch("kennel.github.add_reaction"),
        ):
            cat, title = reply_to_issue_comment(
                self._action(), cfg, self._repo_cfg(tmp_path)
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

        def fake_run(args, **kwargs):
            if "claude" in args:
                text = args[-1]
                if "Triage" in text:
                    return _make_completed_run("ACT: do it\n")
                return _make_completed_run("ok\n")
            # get_repo_info still uses subprocess — raise to hit except block
            raise Exception("gh fail")

        with patch("subprocess.run", side_effect=fake_run):
            cat, title = reply_to_issue_comment(action, cfg, self._repo_cfg(tmp_path))
        assert cat == "ACT"

    def test_no_comment_id_skips_react(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="PR top-level comment on #7 by owner:\n\nhi",
            comment_body="hi",
            is_bot=False,
            context={"pr_title": "My PR"},  # no comment_id
        )

        def fake_run(args, **kwargs):
            if "claude" in args:
                text = args[-1]
                if "Triage" in text:
                    return _make_completed_run("ACT: do it\n")
                return _make_completed_run("ok\n")
            return _make_completed_run("owner/repo\n")

        with (
            patch("subprocess.run", side_effect=fake_run),
            patch("kennel.github.comment_issue"),
        ):
            cat, title = reply_to_issue_comment(action, cfg, self._repo_cfg(tmp_path))
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
        mock_add.assert_called_once_with(tmp_path, title="do something", thread=None)
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
        mock_add.assert_called_once_with(tmp_path, title="do something", thread=thread)


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

    def test_launches_popen(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        mock_proc = MagicMock()
        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            launch_sync(cfg, self._repo_cfg(tmp_path))
        assert mock_popen.called
        args = mock_popen.call_args[0][0]
        assert "bash" in args
        assert "sync-tasks.sh" in args[1]

    def test_popen_exception_does_not_raise(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        with patch("subprocess.Popen", side_effect=Exception("fail")):
            launch_sync(cfg, self._repo_cfg(tmp_path))  # should not raise


class TestLaunchWorker:
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

    def test_returns_pid(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        with patch("subprocess.Popen", return_value=mock_proc):
            pid = launch_worker(cfg, self._repo_cfg(tmp_path))
        assert pid == 12345

    def test_exception_returns_none(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        with patch("subprocess.Popen", side_effect=Exception("fail")):
            pid = launch_worker(cfg, self._repo_cfg(tmp_path))
        assert pid is None


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
        with (
            patch("subprocess.run", return_value=_make_completed_run("heart\n")),
            patch(
                "kennel.github.add_reaction",
                side_effect=RuntimeError("network down"),
            ),
        ):
            maybe_react("great job", 77, "pulls", "owner/repo", cfg)  # must not raise


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
        with patch("kennel.events._triage", return_value=("UNKNOWN_CAT", "do it")):

            def fake_run(args, **kwargs):
                if "claude" in args:
                    return _make_completed_run("I'll look into this.\n")
                return _make_completed_run("")

            with (
                patch("subprocess.run", side_effect=fake_run),
                patch("kennel.github.reply_to_review_comment"),
                patch("kennel.github.add_reaction"),
            ):
                cat, title = reply_to_comment(action, cfg, self._repo_cfg(tmp_path))
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

        def fake_run(args, **kwargs):
            if "claude" in args:
                text = args[-1]
                if "Triage" in text:
                    return _make_completed_run("ACT: fix it\n")
                return _make_completed_run("I'll fix it.\n")
            return _make_completed_run("")

        with (
            patch("subprocess.run", side_effect=fake_run),
            patch(
                "kennel.github.reply_to_review_comment",
                side_effect=RuntimeError("network down"),
            ),
            patch("kennel.github.add_reaction"),
        ):
            cat, title = reply_to_comment(
                action, cfg, self._repo_cfg(tmp_path)
            )  # must not raise
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

        def fake_run(args, **kwargs):
            if "claude" in args:
                text = args[-1]
                if "Triage" in text:
                    return _make_completed_run("ACT: fix it\n")
                return _make_completed_run("Will fix.\n")
            return _make_completed_run("")

        with (
            patch("subprocess.run", side_effect=fake_run),
            patch("kennel.github.get_review_comments", return_value=[500]),
            patch("kennel.github.reply_to_review_comment"),
            patch("kennel.github.add_reaction"),
        ):
            reply_to_review(
                action, cfg, self._repo_cfg(tmp_path), already_replied=already
            )
        assert 500 in already
