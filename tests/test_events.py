from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import pytest

from fido.claude import ClaudeClient
from fido.config import Config, RepoMembership
from fido.config import RepoConfig as _RepoConfig
from fido.events import (
    Action,
    _apply_reply_result,
    _configured_agent,
    _get_commit_summary,
    _is_allowed,
    _notify_thread_change,
    _posted_comment_id,
    _record_reply_artifact,
    _reorder_tasks_background,
    _reply_promise_ids,
    _rewrite_pr_description,
    _summarize_as_action_item,
    _task_snapshot,
    _triage,
    _try_resolve_thread,
    create_task,
    dispatch,
    launch_sync,
    launch_worker,
    maybe_react,
    needs_more_context,
    recover_reply_promises,
    reply_to_comment,
    reply_to_issue_comment,
    reply_to_review,
    review_outcome_creates_tasks,
    review_outcome_resolves_thread,
)
from fido.provider import ProviderID
from fido.rocq import replied_comment_claims as oracle
from fido.store import FidoStore, ReplyPromiseRecord


class RepoConfig(_RepoConfig):
    def __init__(self, *args, provider: ProviderID = ProviderID.CLAUDE_CODE, **kwargs):
        super().__init__(*args, provider=provider, **kwargs)


def _config(tmp_path: Path) -> Config:
    return Config(
        port=9000,
        secret=b"test",
        repos={},
        allowed_bots=frozenset({"copilot[bot]"}),
        log_level="WARNING",
        sub_dir=tmp_path / "sub",
    )


def _repo_cfg(tmp_path: Path) -> RepoConfig:
    from fido.config import RepoMembership

    return RepoConfig(
        name="owner/repo",
        work_dir=tmp_path,
        membership=RepoMembership(collaborators=frozenset({"owner"})),
    )


def _payload(repo_owner: str = "owner") -> dict:
    return {
        "repository": {
            "full_name": f"{repo_owner}/repo",
            "owner": {"login": repo_owner},
        },
    }


def _client(return_value: str = "", *, side_effect=None) -> MagicMock:
    """Build a mock ClaudeClient with run_turn configured."""
    client = MagicMock(spec=ClaudeClient)
    client.voice_model = "claude-opus-4-6"
    client.work_model = "claude-sonnet-4-6"
    client.brief_model = "claude-haiku-4-5"
    if side_effect is not None:
        client.run_turn.side_effect = side_effect
    else:
        client.run_turn.return_value = return_value
    return client


def _oracle_owner(owner: str) -> object:
    match owner:
        case "webhook":
            return oracle.OwnerWebhook()
        case "worker":
            return oracle.OwnerWorker()
        case "recovery":
            return oracle.OwnerRecovery()


def _promise_state_name(state: object) -> str:
    if isinstance(state, str):
        return {
            "prepared": "PromisePrepared",
            "posted": "PromisePosted",
            "acked": "PromiseAcked",
            "failed": "PromiseFailed",
            "in_progress": "ClaimInProgress",
            "completed": "ClaimCompleted",
            "retryable_failed": "ClaimRetryableFailed",
        }[state]
    return type(state).__name__


class TestNeedsMoreContext:
    def test_haiku_yes_returns_true(self) -> None:
        assert needs_more_context("same", agent=_client("YES"))

    def test_haiku_yes_with_explanation_returns_true(self) -> None:
        assert needs_more_context("^", agent=_client("YES, this is vague"))

    def test_haiku_no_returns_false(self) -> None:
        assert not needs_more_context(
            "This is a detailed review comment.",
            agent=_client("NO"),
        )

    def test_haiku_no_with_explanation_returns_false(self) -> None:
        assert not needs_more_context(
            "Please rename this variable to be more descriptive.",
            agent=_client("NO, it's clear"),
        )

    def test_subprocess_exception_returns_false(self) -> None:
        assert not needs_more_context("ditto", agent=_client(""))

    def test_timeout_returns_false(self) -> None:
        assert not needs_more_context("same", agent=_client(""))

    def test_empty_output_returns_false(self) -> None:
        assert not needs_more_context("here too", agent=_client(""))

    def test_uses_haiku_model(self) -> None:
        client = _client("YES")
        needs_more_context("same", agent=client)
        assert client.run_turn.call_args.kwargs["model"] == "claude-haiku-4-5"

    def test_requires_agent(self) -> None:
        with pytest.raises(ValueError, match="needs_more_context requires agent"):
            needs_more_context("some comment")

    def test_configured_agent_uses_provider_factory(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        cfg.repos["owner/repo"] = RepoConfig(
            name="owner/repo",
            work_dir=tmp_path,
            provider=ProviderID.COPILOT_CLI,
        )
        sentinel = MagicMock()
        with patch("fido.events.DefaultProviderFactory") as factory_cls:
            factory_cls.return_value.create_agent.return_value = sentinel
            assert _configured_agent(cfg, cfg.repos["owner/repo"]) is sentinel


class TestRecoverReplyPromises:
    def _prepare_promise(
        self, tmp_path: Path, comment_type: str, comment_id: int
    ) -> ReplyPromiseRecord:
        promise = FidoStore(tmp_path).prepare_reply(
            owner="recovery",
            comment_type=comment_type,
            anchor_comment_id=comment_id,
        )
        assert promise is not None
        return promise

    def _assert_recovery_matches_oracle(
        self,
        tmp_path: Path,
        promise: ReplyPromiseRecord,
        observation: object,
        *,
        covered_comment_ids: tuple[int, ...] | None = None,
    ) -> None:
        comments = list(
            covered_comment_ids
            if covered_comment_ids is not None
            else promise.covered_comment_ids[1:]
        )
        prepared = oracle.prepare_claims(
            _oracle_owner("recovery"),
            1,
            promise.anchor_comment_id,
            comments,
            {},
            {},
        )
        assert prepared is not None
        claims, promises = prepared
        claims, promises = oracle.recover_promise(1, observation, claims, promises)

        persisted = FidoStore(tmp_path).promise(promise.promise_id)
        assert persisted is not None
        assert _promise_state_name(persisted.state) == _promise_state_name(
            promises[1].promise_state
        )
        for comment_id in promise.covered_comment_ids:
            assert (
                FidoStore(tmp_path).claim_state(comment_id)
                == {
                    "ClaimInProgress": "in_progress",
                    "ClaimCompleted": "completed",
                    "ClaimRetryableFailed": "retryable_failed",
                }[_promise_state_name(claims[comment_id].claim_state)]
            )

    def test_returns_false_when_no_promises(self, tmp_path: Path) -> None:
        assert not recover_reply_promises(
            tmp_path / ".git" / "fido",
            _config(tmp_path),
            _repo_cfg(tmp_path),
            MagicMock(),
            7,
        )

    def test_recovers_issue_comment_promise(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        promise = self._prepare_promise(tmp_path, "issues", 302)
        gh = MagicMock()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}
        gh.get_issue_comment.return_value = {
            "id": 302,
            "body": "please fix",
            "html_url": "https://github.com/owner/repo/pull/7#issuecomment-302",
            "issue_url": "https://api.github.com/repos/owner/repo/issues/7",
            "user": {"login": "owner"},
        }
        with (
            patch(
                "fido.events.reply_to_issue_comment",
                return_value=("ACT", ["task one"]),
            ),
            patch("fido.events.create_task") as mock_create_task,
        ):
            result = recover_reply_promises(
                fido_dir,
                _config(tmp_path),
                _repo_cfg(tmp_path),
                gh,
                7,
            )
        assert result is True
        assert FidoStore(tmp_path).promise(promise.promise_id).state == "acked"
        mock_create_task.assert_called_once()
        assert mock_create_task.call_args.args[0] == "task one"
        assert mock_create_task.call_args.kwargs["thread"] == {
            "repo": "owner/repo",
            "pr": 7,
            "comment_id": 302,
            "url": "https://github.com/owner/repo/pull/7#issuecomment-302",
            "author": "owner",
            "comment_type": "issues",
        }

    def test_recovers_stale_issue_marker_without_reposting(
        self, tmp_path: Path
    ) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        store = FidoStore(tmp_path)
        promise = store.prepare_reply(
            owner="webhook", comment_type="issues", anchor_comment_id=303
        )
        assert promise is not None
        gh = MagicMock()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}
        gh.get_issue_comments.return_value = [
            {
                "id": 1,
                "body": f"done\n\n<!-- fido:reply-promise:{promise.promise_id} -->",
            }
        ]
        with patch("fido.events.reply_to_issue_comment") as mock_reply:
            assert recover_reply_promises(
                fido_dir,
                _config(tmp_path),
                _repo_cfg(tmp_path),
                gh,
                7,
            )
        mock_reply.assert_not_called()
        assert store.promise(promise.promise_id).state == "acked"
        self._assert_recovery_matches_oracle(
            tmp_path,
            promise,
            oracle.SeenPromiseMarker(),
        )

    def test_recovers_stale_pull_marker_without_reposting(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        store = FidoStore(tmp_path)
        promise = store.prepare_reply(
            owner="webhook", comment_type="pulls", anchor_comment_id=305
        )
        assert promise is not None
        gh = MagicMock()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}
        gh.fetch_comment_thread.return_value = [
            {
                "id": 1,
                "body": f"done\n\n<!-- fido:reply-promise:{promise.promise_id} -->",
            }
        ]
        with patch("fido.events.reply_to_comment") as mock_reply:
            assert recover_reply_promises(
                fido_dir,
                _config(tmp_path),
                _repo_cfg(tmp_path),
                gh,
                7,
            )
        mock_reply.assert_not_called()
        assert store.promise(promise.promise_id).state == "acked"
        self._assert_recovery_matches_oracle(
            tmp_path,
            promise,
            oracle.SeenPromiseMarker(),
        )

    def test_deleted_comment_promise_is_removed(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        promise = self._prepare_promise(tmp_path, "pulls", 205)
        gh = MagicMock()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}
        gh.get_pull_comment.return_value = None
        assert not recover_reply_promises(
            fido_dir,
            _config(tmp_path),
            _repo_cfg(tmp_path),
            gh,
            7,
        )
        assert FidoStore(tmp_path).promise(promise.promise_id).state == "failed"
        self._assert_recovery_matches_oracle(
            tmp_path,
            promise,
            oracle.AnchorDeleted(),
        )

    def test_deleted_issue_comment_promise_is_removed(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        promise = self._prepare_promise(tmp_path, "issues", 302)
        gh = MagicMock()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}
        gh.get_issue_comment.return_value = None
        assert not recover_reply_promises(
            fido_dir,
            _config(tmp_path),
            _repo_cfg(tmp_path),
            gh,
            7,
        )
        assert FidoStore(tmp_path).promise(promise.promise_id).state == "failed"
        self._assert_recovery_matches_oracle(
            tmp_path,
            promise,
            oracle.AnchorDeleted(),
        )

    def test_other_pr_promise_is_left_for_later(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        promise = self._prepare_promise(tmp_path, "pulls", 205)
        gh = MagicMock()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}
        gh.get_pull_comment.return_value = {
            "id": 205,
            "body": "please fix",
            "pull_request_url": "https://api.github.com/repos/owner/repo/pulls/8",
            "html_url": "https://github.com/owner/repo/pull/8#discussion_r205",
            "user": {"login": "owner"},
        }
        assert not recover_reply_promises(
            fido_dir,
            _config(tmp_path),
            _repo_cfg(tmp_path),
            gh,
            7,
        )
        assert FidoStore(tmp_path).promise(promise.promise_id).state == "prepared"
        assert [
            p.anchor_comment_id for p in FidoStore(tmp_path).recoverable_promises()
        ] == [205]
        self._assert_recovery_matches_oracle(
            tmp_path,
            promise,
            oracle.WrongPullRequest(),
        )

    def test_other_pr_issue_promise_is_left_for_later(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        promise = self._prepare_promise(tmp_path, "issues", 302)
        gh = MagicMock()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}
        gh.get_issue_comment.return_value = {
            "id": 302,
            "body": "please fix",
            "issue_url": "https://api.github.com/repos/owner/repo/issues/8",
            "html_url": "https://github.com/owner/repo/pull/8#issuecomment-302",
            "user": {"login": "owner"},
        }
        assert not recover_reply_promises(
            fido_dir,
            _config(tmp_path),
            _repo_cfg(tmp_path),
            gh,
            7,
        )
        assert FidoStore(tmp_path).promise(promise.promise_id).state == "prepared"
        assert [
            p.anchor_comment_id for p in FidoStore(tmp_path).recoverable_promises()
        ] == [302]
        self._assert_recovery_matches_oracle(
            tmp_path,
            promise,
            oracle.WrongPullRequest(),
        )

    def test_issue_comment_without_pr_url_raises(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        self._prepare_promise(tmp_path, "issues", 302)
        gh = MagicMock()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}
        gh.get_issue_comment.return_value = {
            "id": 302,
            "body": "please fix",
            "issue_url": "https://api.github.com/repos/owner/repo/not-an-issue",
            "html_url": "https://github.com/owner/repo/pull/7#issuecomment-302",
            "user": {"login": "owner"},
        }
        with (
            pytest.raises(ValueError, match="invalid GitHub API URL"),
            patch("fido.events.reply_to_issue_comment") as mock_reply,
        ):
            recover_reply_promises(
                fido_dir,
                _config(tmp_path),
                _repo_cfg(tmp_path),
                gh,
                7,
            )
        mock_reply.assert_not_called()
        assert [
            p.anchor_comment_id for p in FidoStore(tmp_path).recoverable_promises()
        ] == [302]

    def test_issue_recovery_marks_failed_when_reply_raises(
        self, tmp_path: Path
    ) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        promise = self._prepare_promise(tmp_path, "issues", 302)
        gh = MagicMock()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}
        gh.get_issue_comment.return_value = {
            "id": 302,
            "body": "please fix",
            "issue_url": "https://api.github.com/repos/owner/repo/issues/7",
            "html_url": "https://github.com/owner/repo/pull/7#issuecomment-302",
            "user": {"login": "owner"},
        }
        with (
            pytest.raises(RuntimeError, match="reply failed"),
            patch(
                "fido.events.reply_to_issue_comment",
                side_effect=RuntimeError("reply failed"),
            ),
        ):
            recover_reply_promises(
                fido_dir,
                _config(tmp_path),
                _repo_cfg(tmp_path),
                gh,
                7,
            )
        assert FidoStore(tmp_path).claim_state(302) == "retryable_failed"
        assert FidoStore(tmp_path).recoverable_promises()[0].state == "failed"
        self._assert_recovery_matches_oracle(
            tmp_path,
            promise,
            oracle.ReplayFailed(),
        )

    def test_pull_comment_without_pr_url_raises(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        self._prepare_promise(tmp_path, "pulls", 205)
        gh = MagicMock()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}
        gh.get_pull_comment.return_value = {
            "id": 205,
            "body": "please fix",
            "pull_request_url": "https://api.github.com/repos/owner/repo/not-a-pr",
            "html_url": "https://github.com/owner/repo/pull/7#discussion_r205",
            "user": {"login": "owner"},
        }
        with (
            pytest.raises(ValueError, match="invalid GitHub API URL"),
            patch("fido.events.reply_to_comment") as mock_reply,
        ):
            recover_reply_promises(
                fido_dir,
                _config(tmp_path),
                _repo_cfg(tmp_path),
                gh,
                7,
            )
        mock_reply.assert_not_called()
        assert [
            p.anchor_comment_id for p in FidoStore(tmp_path).recoverable_promises()
        ] == [205]

    def test_pull_recovery_marks_failed_when_reply_raises(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        promise = self._prepare_promise(tmp_path, "pulls", 205)
        gh = MagicMock()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}
        gh.get_pull_comment.return_value = {
            "id": 205,
            "body": "please fix",
            "path": "foo.py",
            "line": 1,
            "diff_hunk": "@@ @@",
            "pull_request_url": "https://api.github.com/repos/owner/repo/pulls/7",
            "html_url": "https://github.com/owner/repo/pull/7#discussion_r205",
            "user": {"login": "owner"},
        }
        with (
            pytest.raises(RuntimeError, match="reply failed"),
            patch(
                "fido.events.reply_to_comment",
                side_effect=RuntimeError("reply failed"),
            ),
        ):
            recover_reply_promises(
                fido_dir,
                _config(tmp_path),
                _repo_cfg(tmp_path),
                gh,
                7,
            )
        assert FidoStore(tmp_path).claim_state(205) == "retryable_failed"
        assert FidoStore(tmp_path).recoverable_promises()[0].state == "failed"
        self._assert_recovery_matches_oracle(
            tmp_path,
            promise,
            oracle.ReplayFailed(),
        )

    def test_defer_recovery_skips_task_creation(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        promise = self._prepare_promise(tmp_path, "issues", 302)
        gh = MagicMock()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}
        gh.get_issue_comment.return_value = {
            "id": 302,
            "body": "please fix",
            "html_url": "https://github.com/owner/repo/pull/7#issuecomment-302",
            "issue_url": "https://api.github.com/repos/owner/repo/issues/7",
            "user": {"login": "owner"},
        }
        with (
            patch(
                "fido.events.reply_to_issue_comment",
                return_value=("DEFER", ["later"]),
            ),
            patch("fido.events.create_task") as mock_create_task,
        ):
            result = recover_reply_promises(
                fido_dir,
                _config(tmp_path),
                _repo_cfg(tmp_path),
                gh,
                7,
            )
        assert result is True
        mock_create_task.assert_not_called()
        assert FidoStore(tmp_path).promise(promise.promise_id).state == "acked"

    def test_issue_recovery_clears_promise_before_task_creation(
        self, tmp_path: Path
    ) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        promise = self._prepare_promise(tmp_path, "issues", 302)
        gh = MagicMock()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}
        gh.get_issue_comment.return_value = {
            "id": 302,
            "body": "please fix",
            "html_url": "https://github.com/owner/repo/pull/7#issuecomment-302",
            "issue_url": "https://api.github.com/repos/owner/repo/issues/7",
            "user": {"login": "owner"},
        }

        def fail_after_reply(*args, **kwargs):
            assert FidoStore(tmp_path).promise(promise.promise_id).state == "acked"
            raise RuntimeError("task add failed")

        with (
            patch(
                "fido.events.reply_to_issue_comment",
                return_value=("ACT", ["task one"]),
            ),
            patch("fido.events.create_task", side_effect=fail_after_reply),
        ):
            with pytest.raises(RuntimeError, match="task add failed"):
                recover_reply_promises(
                    fido_dir,
                    _config(tmp_path),
                    _repo_cfg(tmp_path),
                    gh,
                    7,
                )
        assert FidoStore(tmp_path).promise(promise.promise_id).state == "acked"

    def test_coalesces_review_comment_promises_in_same_thread(
        self, tmp_path: Path
    ) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        first = self._prepare_promise(tmp_path, "pulls", 101)
        second = self._prepare_promise(tmp_path, "pulls", 102)
        gh = MagicMock()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}

        def get_pull_comment(_repo: str, comment_id: int) -> dict[str, object]:
            comments = {
                101: {
                    "id": 101,
                    "body": "first",
                    "path": "foo.py",
                    "line": 1,
                    "diff_hunk": "@@ @@",
                    "pull_request_url": "https://api.github.com/repos/owner/repo/pulls/7",
                    "html_url": "https://github.com/owner/repo/pull/7#discussion_r101",
                    "user": {"login": "owner"},
                },
                102: {
                    "id": 102,
                    "body": "second",
                    "path": "foo.py",
                    "line": 2,
                    "diff_hunk": "@@ @@",
                    "in_reply_to_id": 101,
                    "pull_request_url": "https://api.github.com/repos/owner/repo/pulls/7",
                    "html_url": "https://github.com/owner/repo/pull/7#discussion_r102",
                    "user": {"login": "owner"},
                },
            }
            return comments[comment_id]

        gh.get_pull_comment.side_effect = get_pull_comment
        with (
            patch(
                "fido.events.reply_to_comment",
                return_value=("DO", ["task a", "task b"]),
            ) as mock_reply,
            patch("fido.events.create_task") as mock_create_task,
        ):
            result = recover_reply_promises(
                fido_dir,
                _config(tmp_path),
                _repo_cfg(tmp_path),
                gh,
                7,
            )
        assert result is True
        assert mock_reply.call_args.args[0].comment_body == "first\n\n---\n\nsecond"
        assert mock_create_task.call_count == 2
        store = FidoStore(tmp_path)
        assert store.promise(first.promise_id).state == "acked"
        assert store.promise(second.promise_id).state == "acked"
        self._assert_recovery_matches_oracle(
            tmp_path,
            first,
            oracle.ReplayPosted(),
        )
        self._assert_recovery_matches_oracle(
            tmp_path,
            second,
            oracle.ReplayPosted(),
        )

    def test_coalesces_issue_comment_promises_in_same_pr_lane(
        self, tmp_path: Path
    ) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        first = self._prepare_promise(tmp_path, "issues", 301)
        second = self._prepare_promise(tmp_path, "issues", 302)
        gh = MagicMock()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}

        def get_issue_comment(_repo: str, comment_id: int) -> dict[str, object]:
            comments = {
                301: {
                    "id": 301,
                    "body": "first",
                    "html_url": "https://github.com/owner/repo/pull/7#issuecomment-301",
                    "issue_url": "https://api.github.com/repos/owner/repo/issues/7",
                    "user": {"login": "owner"},
                },
                302: {
                    "id": 302,
                    "body": "second",
                    "html_url": "https://github.com/owner/repo/pull/7#issuecomment-302",
                    "issue_url": "https://api.github.com/repos/owner/repo/issues/7",
                    "user": {"login": "owner"},
                },
            }
            return comments[comment_id]

        gh.get_issue_comment.side_effect = get_issue_comment
        with (
            patch(
                "fido.events.reply_to_issue_comment",
                return_value=("DO", ["task a"]),
            ) as mock_reply,
            patch("fido.events.create_task") as mock_create_task,
        ):
            result = recover_reply_promises(
                fido_dir,
                _config(tmp_path),
                _repo_cfg(tmp_path),
                gh,
                7,
            )
        assert result is True
        assert mock_reply.call_args.args[0].comment_body == "first\n\n---\n\nsecond"
        mock_create_task.assert_called_once()
        store = FidoStore(tmp_path)
        assert store.promise(first.promise_id).state == "acked"
        assert store.promise(second.promise_id).state == "acked"
        self._assert_recovery_matches_oracle(
            tmp_path,
            first,
            oracle.ReplayPosted(),
        )
        self._assert_recovery_matches_oracle(
            tmp_path,
            second,
            oracle.ReplayPosted(),
        )

    def test_issue_recovery_replay_records_one_artifact_for_group(
        self, tmp_path: Path
    ) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        first = self._prepare_promise(tmp_path, "issues", 301)
        second = self._prepare_promise(tmp_path, "issues", 302)
        gh = MagicMock()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}
        gh.get_repo_info.return_value = "owner/repo"
        gh.comment_issue.return_value = {"id": 9001}

        def get_issue_comment(_repo: str, comment_id: int) -> dict[str, object]:
            comments = {
                301: {
                    "id": 301,
                    "body": "first",
                    "html_url": "https://github.com/owner/repo/pull/7#issuecomment-301",
                    "issue_url": "https://api.github.com/repos/owner/repo/issues/7",
                    "user": {"login": "owner"},
                },
                302: {
                    "id": 302,
                    "body": "second",
                    "html_url": "https://github.com/owner/repo/pull/7#issuecomment-302",
                    "issue_url": "https://api.github.com/repos/owner/repo/issues/7",
                    "user": {"login": "owner"},
                },
            }
            return comments[comment_id]

        gh.get_issue_comment.side_effect = get_issue_comment

        def fake_pp(prompt, model, **kwargs):
            if "Triage" in prompt:
                return "ANSWER: yep"
            return "One combined reply."

        with patch("fido.events.maybe_react"):
            assert recover_reply_promises(
                fido_dir,
                _config(tmp_path),
                _repo_cfg(tmp_path),
                gh,
                7,
                agent=_client(side_effect=fake_pp),
            )

        store = FidoStore(tmp_path)
        first_artifact = store.artifact_for_promise(first.promise_id)
        second_artifact = store.artifact_for_promise(second.promise_id)
        assert first_artifact is not None
        assert second_artifact is not None
        assert first_artifact == second_artifact
        assert first_artifact.artifact_comment_id == 9001
        assert first_artifact.lane_key == "issues:owner/repo:7"
        assert first_artifact.promise_ids == tuple(
            sorted((first.promise_id, second.promise_id))
        )

    def test_review_recovery_clears_group_promises_before_task_creation(
        self, tmp_path: Path
    ) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        first = self._prepare_promise(tmp_path, "pulls", 101)
        second = self._prepare_promise(tmp_path, "pulls", 102)
        gh = MagicMock()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}

        def get_pull_comment(_repo: str, comment_id: int) -> dict[str, object]:
            comments = {
                101: {
                    "id": 101,
                    "body": "first",
                    "path": "foo.py",
                    "line": 1,
                    "diff_hunk": "@@ @@",
                    "pull_request_url": "https://api.github.com/repos/owner/repo/pulls/7",
                    "html_url": "https://github.com/owner/repo/pull/7#discussion_r101",
                    "user": {"login": "owner"},
                },
                102: {
                    "id": 102,
                    "body": "second",
                    "path": "foo.py",
                    "line": 2,
                    "diff_hunk": "@@ @@",
                    "in_reply_to_id": 101,
                    "pull_request_url": "https://api.github.com/repos/owner/repo/pulls/7",
                    "html_url": "https://github.com/owner/repo/pull/7#discussion_r102",
                    "user": {"login": "owner"},
                },
            }
            return comments[comment_id]

        gh.get_pull_comment.side_effect = get_pull_comment

        def fail_after_reply(*args, **kwargs):
            store = FidoStore(tmp_path)
            assert store.promise(first.promise_id).state == "acked"
            assert store.promise(second.promise_id).state == "acked"
            raise RuntimeError("task add failed")

        with (
            patch("fido.events.reply_to_comment", return_value=("DO", ["task a"])),
            patch("fido.events.create_task", side_effect=fail_after_reply),
        ):
            with pytest.raises(RuntimeError, match="task add failed"):
                recover_reply_promises(
                    fido_dir,
                    _config(tmp_path),
                    _repo_cfg(tmp_path),
                    gh,
                    7,
                )
        store = FidoStore(tmp_path)
        assert store.promise(first.promise_id).state == "acked"
        assert store.promise(second.promise_id).state == "acked"

    def test_review_recovery_replay_records_one_artifact_for_group(
        self, tmp_path: Path
    ) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        first = self._prepare_promise(tmp_path, "pulls", 101)
        second = self._prepare_promise(tmp_path, "pulls", 102)
        gh = MagicMock()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}
        gh.reply_to_review_comment.return_value = {"id": 9101}

        comments = {
            101: {
                "id": 101,
                "body": "first",
                "path": "foo.py",
                "line": 1,
                "diff_hunk": "@@ @@",
                "pull_request_url": "https://api.github.com/repos/owner/repo/pulls/7",
                "html_url": "https://github.com/owner/repo/pull/7#discussion_r101",
                "user": {"login": "owner"},
            },
            102: {
                "id": 102,
                "body": "second",
                "path": "foo.py",
                "line": 2,
                "diff_hunk": "@@ @@",
                "in_reply_to_id": 101,
                "pull_request_url": "https://api.github.com/repos/owner/repo/pulls/7",
                "html_url": "https://github.com/owner/repo/pull/7#discussion_r102",
                "user": {"login": "owner"},
            },
        }

        gh.get_pull_comment.side_effect = lambda _repo, comment_id: comments[comment_id]
        gh.fetch_comment_thread.side_effect = lambda _repo, _pr, _comment_id: [
            {
                "id": 101,
                "body": "first",
                "author": "owner",
            },
            {
                "id": 102,
                "body": "second",
                "author": "owner",
            },
        ]

        def fake_pp(prompt, model, **kwargs):
            if model == "claude-haiku-4-5":
                return "NO"
            if "Triage" in prompt:
                return "ANSWER: yep"
            return "One combined review reply."

        with patch("fido.events.maybe_react"):
            assert recover_reply_promises(
                fido_dir,
                _config(tmp_path),
                _repo_cfg(tmp_path),
                gh,
                7,
                agent=_client(side_effect=fake_pp),
            )

        store = FidoStore(tmp_path)
        first_artifact = store.artifact_for_promise(first.promise_id)
        second_artifact = store.artifact_for_promise(second.promise_id)
        assert first_artifact is not None
        assert second_artifact is not None
        assert first_artifact == second_artifact
        assert first_artifact.artifact_comment_id == 9101
        assert first_artifact.lane_key == "pulls:owner/repo:7:thread:101"
        assert first_artifact.promise_ids == tuple(
            sorted((first.promise_id, second.promise_id))
        )

    def test_recovery_raises_on_invalid_candidate_in_later_group(
        self, tmp_path: Path
    ) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        self._prepare_promise(tmp_path, "pulls", 101)
        self._prepare_promise(tmp_path, "pulls", 102)
        self._prepare_promise(tmp_path, "pulls", 201)
        self._prepare_promise(tmp_path, "pulls", 999)
        gh = MagicMock()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}

        def get_pull_comment(_repo: str, comment_id: int) -> dict[str, object]:
            comments = {
                101: {
                    "id": 101,
                    "body": "first",
                    "path": "foo.py",
                    "line": 1,
                    "diff_hunk": "@@ @@",
                    "pull_request_url": "https://api.github.com/repos/owner/repo/pulls/7",
                    "html_url": "https://github.com/owner/repo/pull/7#discussion_r101",
                    "user": {"login": "owner"},
                },
                102: {
                    "id": 102,
                    "body": "second",
                    "path": "foo.py",
                    "line": 2,
                    "diff_hunk": "@@ @@",
                    "in_reply_to_id": 101,
                    "pull_request_url": "https://api.github.com/repos/owner/repo/pulls/7",
                    "html_url": "https://github.com/owner/repo/pull/7#discussion_r102",
                    "user": {"login": "owner"},
                },
                201: {
                    "id": 201,
                    "body": "third",
                    "path": "bar.py",
                    "line": 3,
                    "diff_hunk": "@@ @@",
                    "pull_request_url": "https://api.github.com/repos/owner/repo/pulls/7",
                    "html_url": "https://github.com/owner/repo/pull/7#discussion_r201",
                    "user": {"login": "owner"},
                },
                999: {
                    "id": 999,
                    "body": "ignored",
                    "path": "zap.py",
                    "line": 9,
                    "diff_hunk": "@@ @@",
                    "pull_request_url": "https://api.github.com/repos/owner/repo/not-a-pr",
                    "html_url": "https://github.com/owner/repo/pull/7#discussion_r999",
                    "user": {"login": "owner"},
                },
            }
            return comments[comment_id]

        gh.get_pull_comment.side_effect = get_pull_comment
        with (
            patch(
                "fido.events.reply_to_comment", return_value=("ANSWER", [])
            ) as mock_reply,
            patch("fido.events.create_task") as mock_create_task,
        ):
            with pytest.raises(ValueError, match="invalid GitHub API URL"):
                recover_reply_promises(
                    fido_dir,
                    _config(tmp_path),
                    _repo_cfg(tmp_path),
                    gh,
                    7,
                )
        assert mock_reply.call_count == 0
        mock_create_task.assert_not_called()
        assert [
            p.anchor_comment_id for p in FidoStore(tmp_path).recoverable_promises()
        ] == [101, 102, 201, 999]

    def test_recovery_skips_handled_candidates_when_processing_later_groups(
        self, tmp_path: Path
    ) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        self._prepare_promise(tmp_path, "pulls", 101)
        self._prepare_promise(tmp_path, "pulls", 102)
        self._prepare_promise(tmp_path, "pulls", 201)
        gh = MagicMock()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}

        def get_pull_comment(_repo: str, comment_id: int) -> dict[str, object]:
            comments = {
                101: {
                    "id": 101,
                    "body": "first",
                    "path": "foo.py",
                    "line": 1,
                    "diff_hunk": "@@ @@",
                    "pull_request_url": "https://api.github.com/repos/owner/repo/pulls/7",
                    "html_url": "https://github.com/owner/repo/pull/7#discussion_r101",
                    "user": {"login": "owner"},
                },
                102: {
                    "id": 102,
                    "body": "second",
                    "path": "foo.py",
                    "line": 2,
                    "diff_hunk": "@@ @@",
                    "in_reply_to_id": 101,
                    "pull_request_url": "https://api.github.com/repos/owner/repo/pulls/7",
                    "html_url": "https://github.com/owner/repo/pull/7#discussion_r102",
                    "user": {"login": "owner"},
                },
                201: {
                    "id": 201,
                    "body": "third",
                    "path": "bar.py",
                    "line": 3,
                    "diff_hunk": "@@ @@",
                    "pull_request_url": "https://api.github.com/repos/owner/repo/pulls/7",
                    "html_url": "https://github.com/owner/repo/pull/7#discussion_r201",
                    "user": {"login": "owner"},
                },
            }
            return comments[comment_id]

        gh.get_pull_comment.side_effect = get_pull_comment
        with patch(
            "fido.events.reply_to_comment", return_value=("ANSWER", [])
        ) as mock_reply:
            result = recover_reply_promises(
                fido_dir,
                _config(tmp_path),
                _repo_cfg(tmp_path),
                gh,
                7,
            )
        assert result is True
        assert mock_reply.call_count == 2
        assert FidoStore(tmp_path).recoverable_promises() == []


class TestIsAllowed:
    def _repo_cfg(
        self, tmp_path: Path, collaborators: frozenset[str] = frozenset({"owner"})
    ) -> RepoConfig:
        from fido.config import RepoMembership

        return RepoConfig(
            name="owner/repo",
            work_dir=tmp_path,
            membership=RepoMembership(collaborators=collaborators),
        )

    def test_collaborator_allowed(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        rc = self._repo_cfg(tmp_path)
        assert _is_allowed("owner", rc, cfg)

    def test_any_collaborator_allowed(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        rc = self._repo_cfg(
            tmp_path, collaborators=frozenset({"alice", "bob", "rhencke"})
        )
        assert _is_allowed("rhencke", rc, cfg)

    def test_bot_allowed_even_without_collab(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        rc = self._repo_cfg(tmp_path, collaborators=frozenset())
        assert _is_allowed("copilot[bot]", rc, cfg)

    def test_random_user_denied(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        rc = self._repo_cfg(tmp_path)
        assert not _is_allowed("rando", rc, cfg)

    def test_empty_collaborators_denies_all_humans(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        rc = self._repo_cfg(tmp_path, collaborators=frozenset())
        assert not _is_allowed("anyone", rc, cfg)


class TestReplyPromiseHelpers:
    def test_reply_promise_ids_deduplicates_context_values(self) -> None:
        assert _reply_promise_ids(
            {
                "reply_promise_id": "one",
                "reply_promise_ids": ["one", "two", "", None],
            }
        ) == ("one", "two")

    def test_reply_promise_ids_handles_missing_context(self) -> None:
        assert _reply_promise_ids(None) == ()

    def test_posted_comment_id_extracts_int_only(self) -> None:
        assert _posted_comment_id({"id": 7}) == 7
        assert _posted_comment_id({"id": "7"}) is None
        assert _posted_comment_id(None) is None

    def test_record_reply_artifact_persists_and_marks_posted(
        self, tmp_path: Path
    ) -> None:
        repo_cfg = _repo_cfg(tmp_path)
        store = FidoStore(tmp_path)
        promise = store.prepare_reply(
            owner="worker", comment_type="issues", anchor_comment_id=700
        )
        assert promise is not None

        _record_reply_artifact(
            repo_cfg,
            artifact_comment_id=9007,
            comment_type="issues",
            lane_key="issues:owner/repo:7",
            promise_ids=(promise.promise_id,),
        )

        assert store.promise(promise.promise_id).state == "posted"
        artifact = store.artifact_for_promise(promise.promise_id)
        assert artifact is not None
        assert artifact.artifact_comment_id == 9007

    def test_record_reply_artifact_ignores_missing_comment_id(
        self, tmp_path: Path
    ) -> None:
        repo_cfg = _repo_cfg(tmp_path)
        store = FidoStore(tmp_path)
        promise = store.prepare_reply(
            owner="worker", comment_type="issues", anchor_comment_id=701
        )
        assert promise is not None

        _record_reply_artifact(
            repo_cfg,
            artifact_comment_id=None,
            comment_type="issues",
            lane_key="issues:owner/repo:7",
            promise_ids=(promise.promise_id,),
        )

        assert store.promise(promise.promise_id).state == "prepared"
        assert store.artifact_for_promise(promise.promise_id) is None


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

    def test_reply_to_includes_author(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "created",
            "comment": {
                "id": 124,
                "body": "nit",
                "user": {"login": "owner"},
                "html_url": "https://example.com/comment",
                "path": "test.py",
                "line": 1,
                "diff_hunk": "@@ -1 +1 @@",
            },
            "pull_request": {"number": 5, "title": "My PR", "body": ""},
        }
        result = dispatch(
            "pull_request_review_comment", payload, cfg, _repo_cfg(tmp_path)
        )
        assert result is not None
        assert result.reply_to is not None
        assert result.reply_to["author"] == "owner"
        assert result.reply_to["comment_type"] == "pulls"

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
                "name": "test",
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
        assert result.thread["author"] == "owner"
        assert result.thread["comment_type"] == "issues"

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


class TestSummarizeAsActionItem:
    def test_returns_model_result(self) -> None:
        client = _client("add logging to streamed sub-Claude output")
        result = _summarize_as_action_item(
            "Ensure we log at that level too.", agent=client
        )
        assert result == "add logging to streamed sub-Claude output"

    def test_empty_result_raises(self) -> None:
        client = _client("")
        with pytest.raises(ValueError, match="_summarize_as_action_item"):
            _summarize_as_action_item("short comment", agent=client)

    def test_strips_whitespace_from_result(self) -> None:
        client = _client("  add tests  ")
        result = _summarize_as_action_item("add tests please", agent=client)
        assert result == "add tests"

    def test_requires_agent(self) -> None:
        with pytest.raises(
            ValueError, match="_summarize_as_action_item requires agent"
        ):
            _summarize_as_action_item("add some tests")

    def test_short_result_returned_without_retry(self) -> None:
        short_title = "add unit tests"
        client = _client(short_title)
        result = _summarize_as_action_item("add some tests", agent=client)
        assert result == short_title
        client.run_turn.assert_called_once()  # no retry needed

    def test_retries_when_result_too_long(self) -> None:
        long_title = "a" * 81
        short_title = "add tests"
        client = _client(side_effect=[long_title, short_title])
        result = _summarize_as_action_item("add some tests", agent=client)
        assert result == short_title
        assert client.run_turn.call_count == 2

    def test_retries_up_to_three_times_then_truncates(self) -> None:
        long_title = "a" * 81
        client = _client(long_title)
        result = _summarize_as_action_item("add some tests", agent=client)
        assert result == long_title[:80]
        assert client.run_turn.call_count == 4  # 1 initial + 3 retries

    def test_stops_retrying_once_short_enough(self) -> None:
        titles = ["a" * 81, "b" * 81, "short title"]
        client = _client(side_effect=titles)
        result = _summarize_as_action_item("add some tests", agent=client)
        assert result == "short title"
        assert client.run_turn.call_count == 3  # 1 initial + 2 retries

    def test_uses_retry_on_preempt_via_safe_voice_turn(self) -> None:
        """safe_voice_turn always passes retry_on_preempt=True to run_turn."""
        client = _client("add tests")
        _summarize_as_action_item("add some tests", agent=client)
        _, kwargs = client.run_turn.call_args
        assert kwargs.get("retry_on_preempt") is True

    def test_shorten_empty_raises(self) -> None:
        """If shorten returns empty, safe_voice_turn raises ValueError."""
        long_title = "a" * 81
        # Initial returns long_title; shorten returns ""
        client = _client(side_effect=[long_title, ""])
        with pytest.raises(ValueError, match="run_turn returned empty"):
            _summarize_as_action_item("add some tests", agent=client)


class TestTriage:
    def test_returns_parsed_category(self, tmp_path: Path) -> None:
        cat, titles = _triage(
            "please add tests",
            is_bot=False,
            agent=_client("ACT: add tests"),
        )
        assert cat == "ACT"
        assert titles == ["add tests"]

    def test_fallback_on_bad_response(self, tmp_path: Path) -> None:
        client = _client(side_effect=["", "implement the thing"])
        cat, titles = _triage("do stuff", is_bot=False, agent=client)
        assert cat == "ACT"
        assert titles == ["implement the thing"]

    def test_fallback_for_bot(self, tmp_path: Path) -> None:
        client = _client(side_effect=["", "implement the thing"])
        cat, titles = _triage("do stuff", is_bot=True, agent=client)
        assert cat == "DO"
        assert titles == ["implement the thing"]

    def test_with_context(self, tmp_path: Path) -> None:
        ctx = {"pr_title": "My PR", "file": "foo.py", "diff_hunk": "@@ -1 +1 @@"}
        cat, titles = _triage(
            "nit comment",
            is_bot=False,
            context=ctx,
            agent=_client("DEFER: out of scope"),
        )
        assert cat == "DEFER"

    def test_unrecognized_category_falls_back(self, tmp_path: Path) -> None:
        client = _client(side_effect=["WEIRD: something", "do the thing"])
        cat, titles = _triage("hi", is_bot=False, agent=client)
        assert cat == "ACT"
        assert titles == ["do the thing"]

    def test_timeout_falls_back(self, tmp_path: Path) -> None:
        client = _client(side_effect=["", "do the thing"])
        cat, titles = _triage("hi", is_bot=True, agent=client)
        assert cat == "DO"

    def test_task_category_falls_back(self, tmp_path: Path) -> None:
        """TASK is no longer a valid bot category — falls back to DO."""
        client = _client(side_effect=["TASK: add caching", "add result caching"])
        cat, titles = _triage("cache results", is_bot=True, agent=client)
        assert cat == "DO"
        assert titles == ["add result caching"]

    def test_bot_categories_in_prompt(self, tmp_path: Path) -> None:
        """Ensure bot-specific categories (DO/DEFER/DUMP) are used when is_bot=True."""
        captured = {}

        def fake_pp(prompt, model, **kwargs):
            captured["prompt"] = prompt
            return "DO: implement feature"

        cat, _ = _triage(
            "implement feature", is_bot=True, agent=_client(side_effect=fake_pp)
        )
        assert cat == "DO"
        assert "DO" in captured["prompt"]
        assert "TASK" not in captured["prompt"]

    def test_requires_agent(self) -> None:
        with pytest.raises(ValueError, match="_triage requires agent"):
            _triage("do it", is_bot=False)

    def test_multiple_act_lines_returns_all_titles(self) -> None:
        response = "ACT: add unit tests\nACT: update documentation"
        cat, titles = _triage(
            "please add tests and docs",
            is_bot=False,
            agent=_client(response),
        )
        assert cat == "ACT"
        assert titles == ["add unit tests", "update documentation"]

    def test_mixed_categories_uses_first(self) -> None:
        """Only lines matching the first valid category are collected."""
        response = "ACT: add tests\nDEFER: out of scope"
        cat, titles = _triage(
            "comment",
            is_bot=False,
            agent=_client(response),
        )
        assert cat == "ACT"
        assert titles == ["add tests"]

    def test_zero_act_tasks_falls_back(self) -> None:
        """ACT with empty title is treated as parse failure → fallback."""
        client = _client(side_effect=["ACT: ", "do the thing"])
        cat, titles = _triage("hi", is_bot=False, agent=client)
        # empty title → stripped to "" → falsy → no titles collected → fallback
        assert cat == "ACT"
        assert titles == ["do the thing"]

    def test_lines_without_colon_are_skipped(self) -> None:
        """Preamble lines without a colon are ignored; valid lines are still parsed."""
        response = "thinking\nACT: add unit tests"
        cat, titles = _triage(
            "add tests",
            is_bot=False,
            agent=_client(response),
        )
        assert cat == "ACT"
        assert titles == ["add unit tests"]


class TestMaybeReact:
    def _repo_cfg(self, tmp_path: Path) -> RepoConfig:
        return RepoConfig(name="owner/repo", work_dir=tmp_path)

    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={"owner/repo": self._repo_cfg(tmp_path)},
            allowed_bots=frozenset(),
            log_level="WARNING",
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
            mock_gh,
            agent=_client("heart"),
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
            mock_gh,
            agent=_client("NONE"),
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
            MagicMock(),
            agent=_client(""),
        )

    def test_file_not_found_warns_and_returns(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        maybe_react(
            "hi",
            1,
            "pulls",
            "owner/repo",
            cfg,
            MagicMock(),
            agent=_client(""),
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
            MagicMock(),
            agent=_client(side_effect=fake_pp),
        )
        assert "you are fido" in captured.get("prompt", "")

    def test_defaults_to_repo_configured_agent(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        with patch("fido.events.DefaultProviderFactory") as factory_cls:
            factory_cls.return_value.create_agent.return_value = _client("NONE")
            maybe_react("hi", 1, "pulls", "owner/repo", cfg, MagicMock())
        factory_cls.return_value.create_agent.assert_called_once_with(
            cfg.repos["owner/repo"],
            work_dir=tmp_path,
            repo_name="owner/repo",
        )

    def test_missing_repo_config_raises(self, tmp_path: Path) -> None:
        cfg = Config(
            port=9000,
            secret=b"test",
            repos={},
            allowed_bots=frozenset(),
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )
        with pytest.raises(KeyError, match="owner/repo"):
            maybe_react("hi", 1, "pulls", "owner/repo", cfg, MagicMock())


class TestReplyToComment:
    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={"owner/repo": self._repo_cfg(tmp_path)},
            allowed_bots=frozenset(),
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )

    def _repo_cfg(self, tmp_path: Path) -> RepoConfig:
        return RepoConfig(name="owner/repo", work_dir=tmp_path)

    def test_no_reply_to_returns_act(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(prompt="do stuff")
        cat, titles = reply_to_comment(
            action, cfg, self._repo_cfg(tmp_path), MagicMock()
        )
        assert cat == "ACT"

    def test_no_comment_body_returns_act(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="something",
            reply_to={"repo": "a/b", "pr": 1, "comment_id": 5},
        )
        cat, titles = reply_to_comment(
            action, cfg, self._repo_cfg(tmp_path), MagicMock()
        )
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

        cat, titles = reply_to_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            MagicMock(),
            agent=_client(side_effect=fake_pp),
        )
        assert cat == "ACT"
        assert "logging" in titles[0].lower()

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

        cat, titles = reply_to_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            MagicMock(),
            agent=_client(side_effect=fake_pp),
        )
        assert cat == "ASK"

    @pytest.mark.parametrize(
        ("category", "creates_tasks", "resolves_thread"),
        [
            ("ACT", True, False),
            ("DO", True, False),
            ("ASK", False, False),
            ("ANSWER", False, False),
            ("DEFER", False, True),
            ("DUMP", False, True),
        ],
    )
    def test_review_outcome_helpers(
        self,
        category: str,
        creates_tasks: bool,
        resolves_thread: bool,
    ) -> None:
        assert review_outcome_creates_tasks(category) is creates_tasks
        assert review_outcome_resolves_thread(category) is resolves_thread

    def test_review_outcome_helpers_return_false_for_unknown_category(self) -> None:
        assert review_outcome_creates_tasks("UNKNOWN") is False
        assert review_outcome_resolves_thread("UNKNOWN") is False

    @pytest.mark.parametrize("category", ["DEFER", "DUMP"])
    def test_resolve_categories_resolve_review_thread(
        self, tmp_path: Path, category: str
    ) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 11},
            comment_body="please defer",
            is_bot=False,
        )
        gh = MagicMock()
        gh.fetch_comment_thread.return_value = [
            {"id": 11, "author": "owner", "body": "please defer"}
        ]
        gh.reply_to_review_comment.return_value = {"id": 88}
        gh.get_review_threads.return_value = [
            {
                "id": "thread-node-1",
                "isResolved": False,
                "comments": {"nodes": [{"databaseId": 11}]},
            }
        ]

        def fake_pp(prompt, model, **kwargs):
            if model == "claude-haiku-4-5":
                return "NO"
            if "Triage" in prompt:
                return f"{category}: handled"
            return "Handled."

        reply_to_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            gh,
            agent=_client(side_effect=fake_pp),
        )
        gh.resolve_thread.assert_called_once_with("thread-node-1")

    def test_try_resolve_thread_returns_early_for_missing_repo(self) -> None:
        gh = MagicMock()
        _try_resolve_thread({"pr": 1, "comment_id": 11}, gh)
        gh.get_review_threads.assert_not_called()

    def test_try_resolve_thread_returns_early_for_missing_pr(self) -> None:
        gh = MagicMock()
        _try_resolve_thread({"repo": "owner/repo", "comment_id": 11}, gh)
        gh.get_review_threads.assert_not_called()

    def test_try_resolve_thread_returns_early_for_unparseable_pr(self) -> None:
        gh = MagicMock()
        _try_resolve_thread(
            {"repo": "owner/repo", "pr": object(), "comment_id": 11}, gh
        )
        gh.get_review_threads.assert_not_called()

    def test_try_resolve_thread_skips_resolved_threads(self) -> None:
        gh = MagicMock()
        gh.get_review_threads.return_value = [
            {
                "id": "thread-node-1",
                "isResolved": True,
                "comments": {"nodes": [{"databaseId": 11}]},
            }
        ]
        _try_resolve_thread({"repo": "owner/repo", "pr": 1, "comment_id": 11}, gh)
        gh.resolve_thread.assert_not_called()

    def test_apply_reply_result_skips_non_task_issue_categories(
        self, tmp_path: Path
    ) -> None:
        cfg = self._cfg(tmp_path)
        repo_cfg = self._repo_cfg(tmp_path)
        with patch("fido.events.create_task") as mock_create_task:
            _apply_reply_result(
                "ASK",
                ["ignored"],
                cfg,
                repo_cfg,
                MagicMock(),
                thread=None,
                registry=None,
            )
        mock_create_task.assert_not_called()

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

        cat, titles = reply_to_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            MagicMock(),
            agent=_client(side_effect=fake_pp),
        )
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
            if "Convert this PR review comment" in prompt:
                return "Cache results for performance"
            return "On it!"

        mock_gh = MagicMock()
        cat, titles = reply_to_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            agent=_client(side_effect=fake_pp),
        )
        assert cat == "DO"
        assert titles == ["Cache results for performance"]
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
        cat, titles = reply_to_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            agent=_client(side_effect=fake_pp),
        )
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

        cat, titles = reply_to_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            MagicMock(),
            agent=_client(side_effect=fake_pp),
        )
        assert cat == "DUMP"

    def test_full_flow_defer_issue_creation_failure_propagates(
        self, tmp_path: Path
    ) -> None:
        """DEFER issue creation failure propagates — no reply posted with missing issue."""
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
        with pytest.raises(RuntimeError, match="network fail"):
            reply_to_comment(
                action,
                cfg,
                self._repo_cfg(tmp_path),
                mock_gh,
                agent=_client(side_effect=fake_pp),
            )
        mock_gh.reply_to_review_comment.assert_not_called()

    def test_empty_reply_body_raises(self, tmp_path: Path) -> None:
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
            if "Convert this PR review comment" in prompt:
                return "Do something"
            return ""

        with pytest.raises(ValueError, match="run_turn returned empty"):
            reply_to_comment(
                action,
                cfg,
                self._repo_cfg(tmp_path),
                MagicMock(),
                agent=_client(side_effect=fake_pp),
            )

    def test_claim_race_returns_act_with_no_titles(self, tmp_path: Path) -> None:
        """Second call with same comment_id is blocked by SQLite claim.

        Must return empty titles so the server creates no phantom tasks —
        the process that owns the claim will handle reply and task creation.
        """
        cfg = self._cfg(tmp_path)
        cid = 999
        assert FidoStore(tmp_path).prepare_reply(
            owner="webhook", comment_type="pulls", anchor_comment_id=cid
        )
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": cid},
            comment_body="competing update",
            is_bot=False,
        )
        cat, titles = reply_to_comment(
            action, cfg, self._repo_cfg(tmp_path), MagicMock()
        )
        assert cat == "ACT"
        assert titles == []

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

        cat, titles = reply_to_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            MagicMock(),
            agent=_client(side_effect=fake_pp),
        )
        assert cat == "ACT"

    def test_act_title_from_summarize_not_triage(self, tmp_path: Path) -> None:
        """ACT task title always comes from _summarize_as_action_item, not triage output."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 77},
            comment_body="add tests and update docs",
            is_bot=False,
        )

        def fake_pp(prompt, model, **kwargs):
            if model == "claude-haiku-4-5":
                return "NO"
            if "Triage" in prompt:
                return "ACT: add unit tests\nACT: update documentation"
            if "Convert this PR review comment" in prompt:
                return "Add tests and update docs"
            return "On it!"

        cat, titles = reply_to_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            MagicMock(),
            agent=_client(side_effect=fake_pp),
        )
        assert cat == "ACT"
        # Title comes from _summarize_as_action_item(root_body), not multi-item triage
        assert titles == ["Add tests and update docs"]

    def test_act_title_uses_root_comment_when_reply(self, tmp_path: Path) -> None:
        """When the triggering comment is a reply, ACT title comes from the root."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 42},
            comment_body="Woof, you're right!",
            is_bot=False,
        )

        calls: list[str] = []

        def fake_pp(prompt, model, **kwargs):
            calls.append(prompt)
            if model == "claude-haiku-4-5":
                return "NO"
            if "Triage" in prompt:
                return "ACT: do it"
            if "Convert this PR review comment" in prompt:
                return "Add null input validation"
            return "Done!"

        mock_gh = MagicMock()
        # Thread: root is reviewer feedback, second is fido's reply
        mock_gh.fetch_comment_thread.return_value = [
            {
                "id": 100,
                "author": "reviewer",
                "body": "Please add null input validation",
            },
            {"id": 101, "author": "fidocancode", "body": "Woof, you're right!"},
        ]
        cat, titles = reply_to_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            agent=_client(side_effect=fake_pp),
        )
        assert cat == "ACT"
        # Title derived from root comment, not the "Woof" reply
        assert titles == ["Add null input validation"]
        # _summarize_as_action_item was called with the root body
        summarize_calls = [p for p in calls if "Convert this PR review comment" in p]
        assert len(summarize_calls) == 1
        assert "Please add null input validation" in summarize_calls[0]
        # Posted replies are immutable; even with a prior Fido reply we post a new one.
        reply_args = mock_gh.reply_to_review_comment.call_args.args
        assert reply_args[:2] == ("owner/repo", 1)
        assert reply_args[2].startswith("Done!")
        assert "fido:reply-promise:" in reply_args[2]
        mock_gh.edit_review_comment.assert_not_called()

    def test_act_title_always_from_root_comment(self, tmp_path: Path) -> None:
        """ACT title is always derived from the root comment via _summarize_as_action_item."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 43},
            comment_body="Add error handling for null inputs",
            is_bot=False,
        )

        def fake_pp(prompt, model, **kwargs):
            if model == "claude-haiku-4-5":
                return "NO"
            if "Triage" in prompt:
                return "ACT: add error handling"
            if "Convert this PR review comment" in prompt:
                return "Add error handling for null inputs"
            return "Will do!"

        mock_gh = MagicMock()
        # Thread has only one comment — the triggering one IS the root
        mock_gh.fetch_comment_thread.return_value = [
            {
                "id": 43,
                "author": "reviewer",
                "body": "Add error handling for null inputs",
            },
        ]
        cat, titles = reply_to_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            agent=_client(side_effect=fake_pp),
        )
        assert cat == "ACT"
        # Title always comes from _summarize_as_action_item(root_body), even when
        # the triggering comment is the root itself
        assert titles == ["Add error handling for null inputs"]
        # No prior Fido reply in thread — a new reply is posted
        mock_gh.reply_to_review_comment.assert_called_once()
        mock_gh.edit_review_comment.assert_not_called()

    def test_posts_new_reply_when_human_comments_after_fido(
        self, tmp_path: Path
    ) -> None:
        """When a human posts after Fido's reply, post a new reply rather than editing."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 300},
            comment_body="We need sub issues in priority order.",
            is_bot=False,
        )

        def fake_pp(prompt, model, **kwargs):
            if model == "claude-haiku-4-5":
                return "NO"
            if "Triage" in prompt:
                return "ACT: reorder sub issues"
            if "Convert this PR review comment" in prompt:
                return "Reorder sub issues by priority"
            return "On it!"

        mock_gh = MagicMock()
        # Thread: root → fido reply → NEW human comment (fido must not edit)
        mock_gh.fetch_comment_thread.return_value = [
            {"id": 300, "author": "rhencke", "body": "Add orderBy"},
            {"id": 301, "author": "fidocancode", "body": "Got it!"},
            {
                "id": 302,
                "author": "rhencke",
                "body": "We need sub issues in priority order.",
            },
        ]
        cat, titles = reply_to_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            agent=_client(side_effect=fake_pp),
        )
        assert cat == "ACT"
        # Human spoke last — must post a fresh reply, never edit the old one
        mock_gh.reply_to_review_comment.assert_called_once()
        mock_gh.edit_review_comment.assert_not_called()

    def test_ask_title_not_rederived_from_root(self, tmp_path: Path) -> None:
        """Non-task categories (ASK) are not affected by root body re-derivation."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 44},
            comment_body="Sure, sounds good",
            is_bot=False,
        )

        summarize_called = False

        def fake_pp(prompt, model, **kwargs):
            nonlocal summarize_called
            if "Convert this PR review comment" in prompt:
                summarize_called = True
            if model == "claude-haiku-4-5":
                return "NO"
            if "Triage" in prompt:
                return "ASK: need more info"
            return "Could you clarify?"

        mock_gh = MagicMock()
        mock_gh.fetch_comment_thread.return_value = [
            {"id": 200, "author": "reviewer", "body": "What do you think?"},
            {"id": 201, "author": "fidocancode", "body": "Sure, sounds good"},
        ]
        cat, titles = reply_to_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            agent=_client(side_effect=fake_pp),
        )
        assert cat == "ASK"
        # _summarize_as_action_item must not be called for non-task categories
        assert not summarize_called
        # Posted replies are immutable; ask replies also post a new artifact.
        reply_args = mock_gh.reply_to_review_comment.call_args.args
        assert reply_args[:2] == ("owner/repo", 1)
        assert reply_args[2].startswith("Could you clarify?")
        assert "fido:reply-promise:" in reply_args[2]
        mock_gh.edit_review_comment.assert_not_called()

    def test_reply_run_turn_uses_retry_on_preempt(self, tmp_path: Path) -> None:
        """Reply generation run_turn must pass retry_on_preempt=True so a
        session preemption mid-generation retries rather than silently
        returning an empty or truncated body."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 10},
            comment_body="please add logging",
            is_bot=False,
        )
        all_run_turn_kwargs: list[dict] = []

        def fake_pp(prompt, model, **kwargs):
            all_run_turn_kwargs.append(kwargs)
            if model == "claude-haiku-4-5":
                return "NO"
            if "Triage" in prompt:
                return "ACT: add logging"
            if "Convert this PR review comment" in prompt:
                return "Add logging"
            return "I will add logging."

        reply_to_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            MagicMock(),
            agent=_client(side_effect=fake_pp),
        )
        # At least one run_turn call must carry retry_on_preempt=True — that is
        # the reply generation call, which must survive session preemption.
        assert any(kw.get("retry_on_preempt") is True for kw in all_run_turn_kwargs)


class TestReplyToReview:
    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={},
            allowed_bots=frozenset(),
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )

    def _repo_cfg(self, tmp_path: Path) -> RepoConfig:
        return RepoConfig(name="owner/repo", work_dir=tmp_path)

    def test_is_a_no_op_for_inline_comments(self, tmp_path: Path) -> None:
        """Inline comments are now exclusively handled by the per-comment
        webhook (``pull_request_review_comment``).  ``reply_to_review`` no
        longer iterates the inline comments — closes #518 (double-reply on
        review submission)."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="review",
            review_comments={"repo": "owner/repo", "pr": 5, "review_id": 777},
        )
        mock_gh = MagicMock()
        reply_to_review(action, cfg, self._repo_cfg(tmp_path), mock_gh, agent=_client())
        # Doesn't fetch, doesn't post — no GitHub side effects at all.
        mock_gh.get_review_comments.assert_not_called()
        mock_gh.reply_to_review_comment.assert_not_called()

    def test_no_op_with_no_review_comments(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(prompt="review", review_comments=None)
        # should return without error
        reply_to_review(action, cfg, self._repo_cfg(tmp_path), MagicMock())


class TestReplyToIssueComment:
    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={"owner/repo": self._repo_cfg(tmp_path)},
            allowed_bots=frozenset(),
            log_level="WARNING",
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

        cat, titles = reply_to_issue_comment(
            self._action(),
            cfg,
            self._repo_cfg(tmp_path),
            MagicMock(),
            agent=_client(side_effect=fake_pp),
        )
        assert cat == "ACT"

    def test_ask_reply(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)

        def fake_pp(prompt, model, **kwargs):
            if "Triage" in prompt:
                return "ASK: unclear"
            return "What do you mean?"

        cat, titles = reply_to_issue_comment(
            self._action("unclear"),
            cfg,
            self._repo_cfg(tmp_path),
            MagicMock(),
            agent=_client(side_effect=fake_pp),
        )
        assert cat == "ASK"

    def test_answer_reply(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)

        def fake_pp(prompt, model, **kwargs):
            if "Triage" in prompt:
                return "ANSWER: it works this way"
            return "Yes, because..."

        cat, titles = reply_to_issue_comment(
            self._action("why?"),
            cfg,
            self._repo_cfg(tmp_path),
            MagicMock(),
            agent=_client(side_effect=fake_pp),
        )
        assert cat == "ANSWER"

    def test_dump_reply(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)

        def fake_pp(prompt, model, **kwargs):
            if "Triage" in prompt:
                return "DUMP: nope"
            return "That won't work here."

        cat, titles = reply_to_issue_comment(
            self._action("do it differently"),
            cfg,
            self._repo_cfg(tmp_path),
            MagicMock(),
            agent=_client(side_effect=fake_pp),
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
        cat, titles = reply_to_issue_comment(
            self._action("big refactor"),
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            agent=_client(side_effect=fake_pp),
        )
        assert cat == "DEFER"
        mock_gh.create_issue.assert_called_once_with(
            "owner/repo",
            "later",
            "Deferred from https://github.com/owner/repo/pull/7\n\n> big refactor",
        )

    def test_defer_reply_issue_creation_failure_propagates(
        self, tmp_path: Path
    ) -> None:
        """DEFER issue creation failure propagates — no reply posted with missing issue."""
        cfg = self._cfg(tmp_path)

        def fake_pp(prompt, model, **kwargs):
            if "Triage" in prompt:
                return "DEFER: later"
            return "Out of scope."

        mock_gh = MagicMock()
        mock_gh.get_repo_info.return_value = "owner/repo"
        mock_gh.create_issue.side_effect = RuntimeError("network fail")
        with pytest.raises(RuntimeError, match="network fail"):
            reply_to_issue_comment(
                self._action("big refactor"),
                cfg,
                self._repo_cfg(tmp_path),
                mock_gh,
                agent=_client(side_effect=fake_pp),
            )
        mock_gh.comment_issue.assert_not_called()

    def test_empty_reply_body_raises(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)

        def fake_pp(prompt, model, **kwargs):
            if "Triage" in prompt:
                return "ACT: do it"
            return ""

        with pytest.raises(ValueError, match="run_turn returned empty"):
            reply_to_issue_comment(
                self._action(),
                cfg,
                self._repo_cfg(tmp_path),
                MagicMock(),
                agent=_client(side_effect=fake_pp),
            )

    def test_post_exception_propagates(self, tmp_path: Path) -> None:
        """comment_issue failure propagates so callers fail closed."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="PR top-level comment on #7 by owner:\n\nplease fix",
            comment_body="please fix",
            is_bot=False,
            context={"pr_title": "My PR"},
        )

        def fake_pp(prompt, model, **kwargs):
            if "Triage" in prompt:
                return "ACT: do it"
            return "ok"

        mock_gh = MagicMock()
        mock_gh.comment_issue.side_effect = Exception("gh fail")
        with pytest.raises(Exception, match="gh fail"):
            reply_to_issue_comment(
                action,
                cfg,
                self._repo_cfg(tmp_path),
                mock_gh,
                agent=_client(side_effect=fake_pp),
            )

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

        cat, titles = reply_to_issue_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            MagicMock(),
            agent=_client(side_effect=fake_pp),
        )
        assert cat == "ACT"

    def test_defaults_to_repo_configured_agent(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = self._action()

        def fake_pp(prompt, model, **kwargs):
            if "Triage" in prompt:
                return "ACT: do it"
            return "ok"

        with patch("fido.events.DefaultProviderFactory") as factory_cls:
            factory_cls.return_value.create_agent.return_value = _client(
                side_effect=fake_pp
            )
            cat, titles = reply_to_issue_comment(
                action, cfg, self._repo_cfg(tmp_path), MagicMock()
            )
        factory_cls.return_value.create_agent.assert_called_once_with(
            self._repo_cfg(tmp_path),
            work_dir=tmp_path,
            repo_name="owner/repo",
        )
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
            "fido.events._triage", wraps=lambda *a, **kw: ("ACT", ["do it"])
        ) as mock_triage:
            cat, titles = reply_to_issue_comment(
                action,
                cfg,
                self._repo_cfg(tmp_path),
                mock_gh,
                agent=_client(side_effect=fake_pp),
            )
        assert cat == "ACT"
        mock_gh.get_issue_comments.assert_called_once_with("owner/repo", 7)
        # Verify conversation context was built and passed to _triage
        triage_ctx = mock_triage.call_args[0][2]  # third positional arg = context
        assert "conversation" in triage_ctx
        assert "alice: first comment" in triage_ctx["conversation"]
        assert "bob: second comment" in triage_ctx["conversation"]

    def test_conversation_context_fetch_failure_logs_and_continues(
        self, tmp_path: Path
    ) -> None:
        """Conversation fetch failure logs a warning and proceeds without context.

        The reply pipeline must not be blocked by a best-effort history fetch.
        """
        cfg = self._cfg(tmp_path)
        action = self._action()
        mock_gh = MagicMock()
        mock_gh.get_issue_comments.side_effect = RuntimeError("API down")

        def fake_pp(prompt, model, **kwargs):
            if "Triage" in prompt:
                return "ACT: do it"
            return "ok"

        cat, titles = reply_to_issue_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            agent=_client(side_effect=fake_pp),
        )
        assert cat == "ACT"

    def test_multiple_tasks_from_one_comment(self, tmp_path: Path) -> None:
        """A top-level comment may produce multiple ACT tasks."""
        cfg = self._cfg(tmp_path)

        def fake_pp(prompt, model, **kwargs):
            if "Triage" in prompt:
                return "ACT: add unit tests\nACT: update documentation"
            return "On it!"

        cat, titles = reply_to_issue_comment(
            self._action("add tests and update docs"),
            cfg,
            self._repo_cfg(tmp_path),
            MagicMock(),
            agent=_client(side_effect=fake_pp),
        )
        assert cat == "ACT"
        assert titles == ["add unit tests", "update documentation"]

    def test_reply_run_turn_uses_retry_on_preempt(self, tmp_path: Path) -> None:
        """Reply generation run_turn must pass retry_on_preempt=True so a
        session preemption mid-generation retries rather than silently
        returning an empty or truncated body."""
        cfg = self._cfg(tmp_path)
        all_run_turn_kwargs: list[dict] = []

        def fake_pp(prompt, model, **kwargs):
            all_run_turn_kwargs.append(kwargs)
            if "Triage" in prompt:
                return "ACT: fix the bug"
            return "I'll fix that."

        reply_to_issue_comment(
            self._action(),
            cfg,
            self._repo_cfg(tmp_path),
            MagicMock(),
            agent=_client(side_effect=fake_pp),
        )
        # At least one run_turn call must carry retry_on_preempt=True — that is
        # the reply generation call, which must survive session preemption.
        assert any(kw.get("retry_on_preempt") is True for kw in all_run_turn_kwargs)

    def test_writes_durable_claim_after_reply(self, tmp_path: Path) -> None:
        """After posting a reply, the comment id is completed in SQLite."""
        cfg = self._cfg(tmp_path)

        def fake_pp(prompt, model, **kwargs):
            if "Triage" in prompt:
                return "ANSWER: it works this way"
            return "Yes, here is why..."

        reply_to_issue_comment(
            self._action(cid=4275080243),
            cfg,
            self._repo_cfg(tmp_path),
            MagicMock(),
            agent=_client(side_effect=fake_pp),
        )
        assert FidoStore(tmp_path).claim_state(4275080243) == "completed"

    def test_claimed_issue_comment_returns_no_titles(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        promise = FidoStore(tmp_path).prepare_reply(
            owner="webhook", comment_type="issues", anchor_comment_id=4275080244
        )
        assert promise is not None

        category, titles = reply_to_issue_comment(
            self._action(cid=4275080244),
            cfg,
            self._repo_cfg(tmp_path),
            MagicMock(),
            agent=_client("unused"),
        )

        assert category == "ACT"
        assert titles == []

    def test_no_comment_id_skips_claim_write(self, tmp_path: Path) -> None:
        """When comment_id is absent, no claim file is created (no-op)."""
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

        reply_to_issue_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            MagicMock(),
            agent=_client(side_effect=fake_pp),
        )
        claim_dir = tmp_path / ".git" / "fido" / "comments"
        assert not claim_dir.exists() or not list(claim_dir.iterdir()), (
            "no claim files should be written when comment_id is absent"
        )


class TestCreateTask:
    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={},
            allowed_bots=frozenset(),
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )

    def _fido_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / ".git" / "fido"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _mock_tasks(self, add_return: dict | None = None) -> MagicMock:
        t = MagicMock()
        t.add.return_value = add_return or {
            "id": "t1",
            "title": "task",
            "status": "pending",
            "type": "spec",
        }
        return t

    def test_calls_add_task_and_launch_sync(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        mock_gh = MagicMock()
        mock_tasks = self._mock_tasks()
        with patch("fido.events.launch_sync") as mock_sync:
            create_task("do something", cfg, repo_cfg, mock_gh, _tasks=mock_tasks)
        mock_tasks.add.assert_called_once_with(
            title="do something", task_type=ANY, thread=None
        )
        mock_sync.assert_called_once_with(cfg, repo_cfg, mock_gh)

    def test_passes_thread(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        thread = {"repo": "a/b", "pr": 1, "comment_id": 5}
        mock_tasks = self._mock_tasks()
        with patch("fido.events.launch_sync"):
            create_task(
                "do something",
                cfg,
                repo_cfg,
                MagicMock(),
                thread=thread,
                _tasks=mock_tasks,
                _reorder_background_fn=MagicMock(),
            )
        mock_tasks.add.assert_called_once_with(
            title="do something", task_type=ANY, thread=thread
        )

    def test_skips_when_thread_already_resolved(self, tmp_path: Path) -> None:
        """Late-arriving triage for a thread fido has already auto-resolved
        is dropped on the floor (closes #520)."""
        cfg = self._cfg(tmp_path)
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        thread = {"repo": "owner/repo", "pr": 5, "comment_id": 999}
        mock_tasks = self._mock_tasks()
        mock_gh = MagicMock()
        mock_gh.is_thread_resolved_for_comment.return_value = True
        with patch("fido.events.launch_sync"):
            result = create_task(
                "do something",
                cfg,
                repo_cfg,
                mock_gh,
                thread=thread,
                _tasks=mock_tasks,
                _reorder_background_fn=MagicMock(),
            )
        mock_tasks.add.assert_not_called()
        assert result["status"] == "skipped_resolved"

    def test_queues_when_thread_resolved_check_raises(self, tmp_path: Path) -> None:
        """If the GitHub thread-resolved check fails, fail open and queue
        the task — better to dedup later than drop work."""
        cfg = self._cfg(tmp_path)
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        thread = {"repo": "owner/repo", "pr": 5, "comment_id": 999}
        mock_tasks = self._mock_tasks()
        mock_gh = MagicMock()
        mock_gh.is_thread_resolved_for_comment.side_effect = RuntimeError("api down")
        with patch("fido.events.launch_sync"):
            create_task(
                "do something",
                cfg,
                repo_cfg,
                mock_gh,
                thread=thread,
                _tasks=mock_tasks,
                _reorder_background_fn=MagicMock(),
            )
        mock_tasks.add.assert_called_once()

    def test_returns_created_task(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        fake_task = {
            "id": "t1",
            "title": "do something",
            "status": "pending",
            "type": "spec",
        }
        mock_tasks = self._mock_tasks(fake_task)
        with patch("fido.events.launch_sync"):
            result = create_task(
                "do something", cfg, repo_cfg, MagicMock(), _tasks=mock_tasks
            )
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
        mock_tasks = self._mock_tasks(fake_task)
        with patch("fido.events.launch_sync"):
            create_task(
                "Comment task",
                cfg,
                repo_cfg,
                MagicMock(),
                thread=thread,
                _tasks=mock_tasks,
                _reorder_background_fn=MagicMock(),
            )  # no registry
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
        mock_tasks = self._mock_tasks(fake_task)
        with patch("fido.events.launch_sync"):
            create_task(
                "Another plain task",
                cfg,
                repo_cfg,
                MagicMock(),
                registry=registry,
                _tasks=mock_tasks,
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
        mock_tasks = self._mock_tasks(fake_task)
        with patch("fido.events.launch_sync"):
            create_task(
                "Comment task",
                cfg,
                repo_cfg,
                MagicMock(),
                thread=thread,
                registry=registry,
                _tasks=mock_tasks,
                _reorder_background_fn=MagicMock(),
            )
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
        mock_tasks = self._mock_tasks(fake_task)
        with patch("fido.events.launch_sync"):
            create_task(
                "New thread task",
                cfg,
                repo_cfg,
                MagicMock(),
                thread=new_thread,
                registry=registry,
                _tasks=mock_tasks,
                _reorder_background_fn=MagicMock(),
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
        mock_tasks = self._mock_tasks(fake_task)
        with patch("fido.events.launch_sync"):
            create_task(
                "Comment task",
                cfg,
                repo_cfg,
                MagicMock(),
                thread=thread,
                registry=registry,
                _tasks=mock_tasks,
                _reorder_background_fn=MagicMock(),
            )
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
        mock_tasks = self._mock_tasks(fake_task)
        with patch("fido.events.launch_sync"):
            create_task(
                "Comment task",
                cfg,
                repo_cfg,
                MagicMock(),
                thread=thread,
                registry=registry,
                _tasks=mock_tasks,
                _reorder_background_fn=MagicMock(),
            )
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
        mock_tasks = self._mock_tasks(fake_task)
        with patch("fido.events.launch_sync"):
            create_task(
                "Comment task",
                cfg,
                repo_cfg,
                MagicMock(),
                thread=thread,
                registry=registry,
                _tasks=mock_tasks,
                _reorder_background_fn=MagicMock(),
            )
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
        mock_tasks = self._mock_tasks(fake_task)
        with patch("fido.events.launch_sync"):
            create_task(
                "Comment task",
                cfg,
                repo_cfg,
                MagicMock(),
                thread=thread,
                registry=registry,
                _tasks=mock_tasks,
                _reorder_background_fn=MagicMock(),
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
        mock_tasks = self._mock_tasks(fake_task)
        with patch("fido.events.launch_sync"):
            create_task(
                "Comment task",
                cfg,
                repo_cfg,
                MagicMock(),
                thread=thread,
                registry=registry,
                _tasks=mock_tasks,
                _reorder_background_fn=MagicMock(),
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
        mock_tasks = self._mock_tasks(fake_task)
        with patch("fido.events.launch_sync"):
            create_task(
                "Comment task",
                cfg,
                repo_cfg,
                MagicMock(),
                thread=thread,
                registry=registry,
                _tasks=mock_tasks,
                _reorder_background_fn=MagicMock(),
            )
        registry.abort_task.assert_not_called()

    def test_ci_preempts_thread(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        import json

        registry, fido_dir = self._setup_abort_scenario(tmp_path, "thread")
        fake_task = {"id": "t-ci", "title": "CI fix", "status": "pending", "type": "ci"}
        mock_tasks = self._mock_tasks(fake_task)
        with patch("fido.events.launch_sync"):
            create_task(
                "CI fix",
                cfg,
                repo_cfg,
                MagicMock(),
                registry=registry,
                _tasks=mock_tasks,
            )
        registry.abort_task.assert_called_once_with("owner/repo")
        remaining = json.loads((fido_dir / "tasks.json").read_text())
        assert any(t["id"] == "t-current" for t in remaining)

    def test_ci_preempts_spec(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        import json

        registry, fido_dir = self._setup_abort_scenario(tmp_path, "spec")
        fake_task = {"id": "t-ci", "title": "CI fix", "status": "pending", "type": "ci"}
        mock_tasks = self._mock_tasks(fake_task)
        with patch("fido.events.launch_sync"):
            create_task(
                "CI fix",
                cfg,
                repo_cfg,
                MagicMock(),
                registry=registry,
                _tasks=mock_tasks,
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
        mock_tasks = self._mock_tasks(fake_task)
        with patch("fido.events.launch_sync"):
            create_task(
                "New spec task",
                cfg,
                repo_cfg,
                MagicMock(),
                registry=registry,
                _tasks=mock_tasks,
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
        mock_gh = MagicMock()
        reorder_called: list[tuple] = []
        mock_tasks = self._mock_tasks(fake_task)
        with patch("fido.events.launch_sync"):
            create_task(
                "Comment task",
                cfg,
                repo_cfg,
                mock_gh,
                thread=thread,
                _get_commit_summary_fn=lambda wd: "abc1234 add thing",
                _reorder_background_fn=lambda wd, cs, cfg, gh, rc, reg: (
                    reorder_called.append((wd, cs, cfg, gh, rc, reg))
                ),
                _tasks=mock_tasks,
            )
        assert len(reorder_called) == 1
        assert reorder_called[0][0] == tmp_path
        assert reorder_called[0][1] == "abc1234 add thing"
        assert reorder_called[0][2] is cfg
        assert reorder_called[0][3] is mock_gh
        assert reorder_called[0][4] is repo_cfg
        assert reorder_called[0][5] is None  # no registry passed

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
        mock_tasks = self._mock_tasks(fake_task)
        with patch("fido.events.launch_sync"):
            create_task(
                "Spec task",
                cfg,
                repo_cfg,
                MagicMock(),
                _reorder_background_fn=lambda *a: reorder_called.append(a),
                _tasks=mock_tasks,
            )
        assert reorder_called == []

    def test_spec_task_does_not_call_rewrite_pr_description(
        self, tmp_path: Path
    ) -> None:
        """Normal (non-thread) task creation never triggers a PR description rewrite."""
        cfg = self._cfg(tmp_path)
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        mock_tasks = self._mock_tasks()
        with (
            patch("fido.events.launch_sync"),
            patch("fido.events._rewrite_pr_description") as mock_rewrite,
        ):
            create_task(
                "Spec task", cfg, repo_cfg, MagicMock(), _tasks=mock_tasks
            )  # thread=None
        mock_rewrite.assert_not_called()

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
        mock_tasks = self._mock_tasks(fake_task)
        with patch("fido.events.launch_sync"):
            create_task(
                "t",
                cfg,
                repo_cfg,
                MagicMock(),
                thread=thread,
                _get_commit_summary_fn=lambda wd: "custom summary",
                _reorder_background_fn=lambda wd, cs, cfg, gh, rc, reg: (
                    summaries.append(cs)
                ),
                _tasks=mock_tasks,
            )
        assert summaries == ["custom summary"]

    def test_default_tasks_creates_task_in_file(self, tmp_path: Path) -> None:
        """When _tasks is not passed, create_task constructs Tasks(work_dir) itself."""
        cfg = self._cfg(tmp_path)
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        with patch("fido.events.launch_sync"):
            result = create_task("do a thing", cfg, repo_cfg, MagicMock())
        assert result["title"] == "do a thing"
        from fido.tasks import list_tasks

        assert any(t["title"] == "do a thing" for t in list_tasks(tmp_path))


class TestGetCommitSummary:
    def test_returns_git_log_output(self, tmp_path: Path) -> None:
        import subprocess as sp

        fake_result = sp.CompletedProcess(
            args=[], returncode=0, stdout="abc123 add thing\n", stderr=""
        )
        with patch("fido.events.subprocess.run", return_value=fake_result) as mock_run:
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
        with patch("fido.events.subprocess.run", side_effect=FileNotFoundError):
            result = _get_commit_summary(tmp_path)
        assert result == ""

    def test_returns_empty_on_timeout(self, tmp_path: Path) -> None:
        import subprocess as sp

        with patch(
            "fido.events.subprocess.run",
            side_effect=sp.TimeoutExpired(cmd="git", timeout=10),
        ):
            result = _get_commit_summary(tmp_path)
        assert result == ""

    def test_returns_empty_on_nonzero_exit(self, tmp_path: Path) -> None:
        import subprocess as sp

        fake_result = sp.CompletedProcess(
            args=[], returncode=128, stdout="", stderr="not a git repo"
        )
        with patch("fido.events.subprocess.run", return_value=fake_result):
            result = _get_commit_summary(tmp_path)
        assert result == ""

    def test_nonzero_exit_ignored_even_with_stdout(self, tmp_path: Path) -> None:
        import subprocess as sp

        # Explicit guard: returncode wins over any stdout content.
        fake_result = sp.CompletedProcess(
            args=[], returncode=1, stdout="abc123 orphan output\n", stderr=""
        )
        with patch("fido.events.subprocess.run", return_value=fake_result):
            result = _get_commit_summary(tmp_path)
        assert result == ""

    def test_returns_empty_on_oserror(self, tmp_path: Path) -> None:
        with patch(
            "fido.events.subprocess.run", side_effect=OSError("permission denied")
        ):
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
            sub_dir=tmp_path / "sub",
        )

    def _run_thread(self, started: list) -> None:
        """Run the captured thread's target synchronously."""
        started[0]._target()

    def _capture_reorder_calls(self) -> tuple[list, callable]:
        """Return (calls_list, mock_reorder_fn) that records (work_dir, cs, kwargs)."""
        calls: list = []

        def mock_reorder(work_dir, commit_summary, **kwargs):
            calls.append((work_dir, commit_summary, kwargs))

        return calls, mock_reorder

    def test_starts_daemon_thread(self, tmp_path: Path) -> None:
        started: list = []
        _, mock_reorder = self._capture_reorder_calls()
        _reorder_tasks_background(
            tmp_path,
            "some commits",
            self._cfg(tmp_path),
            MagicMock(),
            _start=lambda t: started.append(t),
            _reorder_fn=mock_reorder,
            _coalesce_state={},
        )
        assert len(started) == 1
        t = started[0]
        assert t.daemon is True

    def test_thread_name_includes_dir_name(self, tmp_path: Path) -> None:
        started: list = []
        _, mock_reorder = self._capture_reorder_calls()
        _reorder_tasks_background(
            tmp_path,
            "commits",
            self._cfg(tmp_path),
            MagicMock(),
            _start=lambda t: started.append(t),
            _reorder_fn=mock_reorder,
            _coalesce_state={},
        )
        assert tmp_path.name in started[0].name

    def test_thread_calls_reorder_with_work_dir_and_commit_summary(
        self, tmp_path: Path
    ) -> None:
        started: list = []
        calls, mock_reorder = self._capture_reorder_calls()
        _reorder_tasks_background(
            tmp_path,
            "feat: add parser",
            self._cfg(tmp_path),
            MagicMock(),
            _start=lambda t: started.append(t),
            _reorder_fn=mock_reorder,
            _coalesce_state={},
        )
        self._run_thread(started)
        assert len(calls) == 1
        assert calls[0][0] == tmp_path
        assert calls[0][1] == "feat: add parser"

    def test_on_changes_callback_notifies_thread_changes(self, tmp_path: Path) -> None:
        started: list = []
        mock_gh = MagicMock()
        calls, mock_reorder = self._capture_reorder_calls()
        _reorder_tasks_background(
            tmp_path,
            "commits",
            self._cfg(tmp_path),
            mock_gh,
            _start=lambda t: started.append(t),
            _reorder_fn=mock_reorder,
            _coalesce_state={},
        )
        self._run_thread(started)
        on_changes = calls[0][2]["_on_changes"]
        change = {
            "task": {
                "id": "t1",
                "title": "Fix it",
                "status": "pending",
                "type": "thread",
                "thread": {
                    "repo": "owner/repo",
                    "pr": 1,
                    "comment_id": 42,
                    "url": "https://github.com/owner/repo/pull/1#issuecomment-42",
                    "author": "bob",
                },
            },
            "kind": "completed",
        }
        with patch("fido.events._notify_thread_change") as mock_notify:
            on_changes([change])
        mock_notify.assert_called_once_with(
            change, self._cfg(tmp_path), mock_gh, agent=None, prompts=None
        )

    def test_on_inprogress_affected_aborts_worker_via_registry(
        self, tmp_path: Path
    ) -> None:
        started: list = []
        registry = MagicMock()
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        calls, mock_reorder = self._capture_reorder_calls()
        _reorder_tasks_background(
            tmp_path,
            "commits",
            self._cfg(tmp_path),
            MagicMock(),
            repo_cfg=repo_cfg,
            registry=registry,
            _start=lambda t: started.append(t),
            _reorder_fn=mock_reorder,
            _coalesce_state={},
        )
        self._run_thread(started)
        on_inprogress_affected = calls[0][2]["_on_inprogress_affected"]
        on_inprogress_affected()
        registry.abort_task.assert_called_once_with("owner/repo")

    def test_on_inprogress_affected_not_in_kwargs_when_no_registry(
        self, tmp_path: Path
    ) -> None:
        started: list = []
        calls, mock_reorder = self._capture_reorder_calls()
        _reorder_tasks_background(
            tmp_path,
            "commits",
            self._cfg(tmp_path),
            MagicMock(),
            _start=lambda t: started.append(t),
            _reorder_fn=mock_reorder,
            _coalesce_state={},
        )
        self._run_thread(started)
        assert "_on_inprogress_affected" not in calls[0][2]

    def test_on_done_kwarg_calls_rewrite_fn(self, tmp_path: Path) -> None:
        started: list = []
        rewrite_calls: list = []
        sync_calls: list = []
        calls, mock_reorder = self._capture_reorder_calls()

        def mock_rewrite(*a, **kw):
            rewrite_calls.append((a, kw))

        def mock_sync(*a, **kw):
            sync_calls.append((a, kw))

        _reorder_tasks_background(
            tmp_path,
            "commits",
            self._cfg(tmp_path),
            MagicMock(),
            _start=lambda t: started.append(t),
            _rewrite_fn=mock_rewrite,
            _reorder_fn=mock_reorder,
            _sync_fn=mock_sync,
            _coalesce_state={},
        )
        self._run_thread(started)
        on_done = calls[0][2]["_on_done"]
        on_done()
        assert len(sync_calls) == 1
        assert len(rewrite_calls) == 1
        args, kwargs = rewrite_calls[0]
        assert args[0] == tmp_path

    def test_on_done_passes_agent_to_rewrite_fn(self, tmp_path: Path) -> None:
        started: list = []
        rewrite_calls: list = []
        fake_client = MagicMock()
        calls, mock_reorder = self._capture_reorder_calls()

        def mock_rewrite(*a, **kw):
            rewrite_calls.append(kw)

        _reorder_tasks_background(
            tmp_path,
            "commits",
            self._cfg(tmp_path),
            MagicMock(),
            _start=lambda t: started.append(t),
            _rewrite_fn=mock_rewrite,
            _sync_fn=lambda *a, **kw: None,
            agent=fake_client,
            _reorder_fn=mock_reorder,
            _coalesce_state={},
        )
        self._run_thread(started)
        on_done = calls[0][2]["_on_done"]
        on_done()
        assert rewrite_calls[0].get("agent") is fake_client

    def test_on_done_syncs_before_rewrite(self, tmp_path: Path) -> None:
        started: list = []
        order: list[str] = []
        calls, mock_reorder = self._capture_reorder_calls()

        def mock_sync(*a, **kw):
            order.append("sync")

        def mock_rewrite(*a, **kw):
            order.append("rewrite")

        _reorder_tasks_background(
            tmp_path,
            "commits",
            self._cfg(tmp_path),
            MagicMock(),
            _start=lambda t: started.append(t),
            _rewrite_fn=mock_rewrite,
            _reorder_fn=mock_reorder,
            _sync_fn=mock_sync,
            _coalesce_state={},
        )
        self._run_thread(started)
        on_done = calls[0][2]["_on_done"]
        on_done()
        assert order == ["sync", "rewrite"]

    def test_on_done_uses_default_sync_tasks_when_no_sync_fn(
        self, tmp_path: Path
    ) -> None:
        started: list = []
        calls, mock_reorder = self._capture_reorder_calls()

        _reorder_tasks_background(
            tmp_path,
            "commits",
            self._cfg(tmp_path),
            MagicMock(),
            _start=lambda t: started.append(t),
            _rewrite_fn=lambda *a, **kw: None,
            _reorder_fn=mock_reorder,
            _coalesce_state={},
        )
        self._run_thread(started)
        on_done = calls[0][2]["_on_done"]
        with patch("fido.tasks.sync_tasks") as mock_sync:
            on_done()
        mock_sync.assert_called_once()

    def test_coalesces_when_already_running(self, tmp_path: Path) -> None:
        """Second call while first is running marks pending, does not spawn thread."""
        state: dict = {}
        started: list = []
        _, mock_reorder = self._capture_reorder_calls()

        # First call — marks running, spawns thread
        _reorder_tasks_background(
            tmp_path,
            "cs1",
            self._cfg(tmp_path),
            MagicMock(),
            _start=lambda t: started.append(t),
            _reorder_fn=mock_reorder,
            _coalesce_state=state,
        )
        assert len(started) == 1
        assert state[str(tmp_path)]["running"] is True

        # Second call while thread has not run yet — should coalesce, not spawn
        _reorder_tasks_background(
            tmp_path,
            "cs2",
            self._cfg(tmp_path),
            MagicMock(),
            _start=lambda t: started.append(t),
            _reorder_fn=mock_reorder,
            _coalesce_state=state,
        )
        assert len(started) == 1  # no second thread spawned
        assert state[str(tmp_path)]["pending"] is not None
        assert state[str(tmp_path)]["pending"][0] == "cs2"

    def test_coalesced_call_reruns_after_first_completes(self, tmp_path: Path) -> None:
        """Thread loops once for the pending coalesced call, then stops."""
        state: dict = {}
        started: list = []
        calls, mock_reorder = self._capture_reorder_calls()

        _reorder_tasks_background(
            tmp_path,
            "cs1",
            self._cfg(tmp_path),
            MagicMock(),
            _start=lambda t: started.append(t),
            _reorder_fn=mock_reorder,
            _coalesce_state=state,
        )
        # Simulate a second trigger arriving before the thread runs
        _reorder_tasks_background(
            tmp_path,
            "cs2",
            self._cfg(tmp_path),
            MagicMock(),
            _start=lambda t: started.append(t),
            _reorder_fn=mock_reorder,
            _coalesce_state=state,
        )
        # Run the single thread — should execute reorder twice (cs1 then cs2)
        self._run_thread(started)
        assert len(calls) == 2
        assert calls[0][1] == "cs1"
        assert calls[1][1] == "cs2"
        assert state[str(tmp_path)]["running"] is False
        assert state[str(tmp_path)]["pending"] is None

    def test_only_last_pending_call_is_preserved(self, tmp_path: Path) -> None:
        """Multiple coalesced callers: only the last pending commit_summary is used."""
        state: dict = {}
        started: list = []
        calls, mock_reorder = self._capture_reorder_calls()

        for cs in ("cs1", "cs2", "cs3", "cs4"):
            _reorder_tasks_background(
                tmp_path,
                cs,
                self._cfg(tmp_path),
                MagicMock(),
                _start=lambda t: started.append(t),
                _reorder_fn=mock_reorder,
                _coalesce_state=state,
            )
        # Only one thread spawned; pending holds cs4 (the latest)
        assert len(started) == 1
        assert state[str(tmp_path)]["pending"][0] == "cs4"
        self._run_thread(started)
        # Ran cs1 (first call) then cs4 (latest pending); cs2 and cs3 dropped
        assert len(calls) == 2
        assert calls[0][1] == "cs1"
        assert calls[1][1] == "cs4"

    def test_running_flag_cleared_after_no_pending(self, tmp_path: Path) -> None:
        """After a normal run with no pending call, running is set to False."""
        state: dict = {}
        started: list = []
        _, mock_reorder = self._capture_reorder_calls()
        _reorder_tasks_background(
            tmp_path,
            "cs",
            self._cfg(tmp_path),
            MagicMock(),
            _start=lambda t: started.append(t),
            _reorder_fn=mock_reorder,
            _coalesce_state=state,
        )
        self._run_thread(started)
        assert state[str(tmp_path)]["running"] is False

    def test_second_call_after_first_completes_spawns_new_thread(
        self, tmp_path: Path
    ) -> None:
        """Once the first thread finishes, a subsequent call spawns a fresh thread."""
        state: dict = {}
        started: list = []
        _, mock_reorder = self._capture_reorder_calls()

        _reorder_tasks_background(
            tmp_path,
            "cs1",
            self._cfg(tmp_path),
            MagicMock(),
            _start=lambda t: started.append(t),
            _reorder_fn=mock_reorder,
            _coalesce_state=state,
        )
        self._run_thread(started)  # first thread completes

        _reorder_tasks_background(
            tmp_path,
            "cs2",
            self._cfg(tmp_path),
            MagicMock(),
            _start=lambda t: started.append(t),
            _reorder_fn=mock_reorder,
            _coalesce_state=state,
        )
        assert len(started) == 2  # new thread spawned

    def test_different_work_dirs_do_not_interfere(self, tmp_path: Path) -> None:
        """Coalescing is per work_dir; different dirs get independent threads."""
        state: dict = {}
        started: list = []
        _, mock_reorder = self._capture_reorder_calls()
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"

        _reorder_tasks_background(
            dir_a,
            "cs",
            self._cfg(tmp_path),
            MagicMock(),
            _start=lambda t: started.append(t),
            _reorder_fn=mock_reorder,
            _coalesce_state=state,
        )
        _reorder_tasks_background(
            dir_b,
            "cs",
            self._cfg(tmp_path),
            MagicMock(),
            _start=lambda t: started.append(t),
            _reorder_fn=mock_reorder,
            _coalesce_state=state,
        )
        assert len(started) == 2  # each dir gets its own thread

    def test_sets_rescoping_true_before_reorder(self, tmp_path: Path) -> None:
        """set_rescoping(True) is called on the registry before reorder runs."""
        started: list = []
        rescoping_calls: list = []
        _, mock_reorder = self._capture_reorder_calls()
        registry = MagicMock()
        registry.set_rescoping.side_effect = lambda repo, active: (
            rescoping_calls.append(active)
        )
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        _reorder_tasks_background(
            tmp_path,
            "cs",
            self._cfg(tmp_path),
            MagicMock(),
            repo_cfg=repo_cfg,
            registry=registry,
            _start=lambda t: started.append(t),
            _reorder_fn=mock_reorder,
            _coalesce_state={},
        )
        self._run_thread(started)
        assert rescoping_calls[0] is True

    def test_clears_rescoping_false_after_reorder(self, tmp_path: Path) -> None:
        """set_rescoping(False) is called on the registry after the loop finishes."""
        started: list = []
        _, mock_reorder = self._capture_reorder_calls()
        registry = MagicMock()
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        _reorder_tasks_background(
            tmp_path,
            "cs",
            self._cfg(tmp_path),
            MagicMock(),
            repo_cfg=repo_cfg,
            registry=registry,
            _start=lambda t: started.append(t),
            _reorder_fn=mock_reorder,
            _coalesce_state={},
        )
        self._run_thread(started)
        # Last call must clear the flag
        last_call = registry.set_rescoping.call_args_list[-1]
        assert last_call[0] == ("owner/repo", False)

    def test_clears_rescoping_on_reorder_exception(self, tmp_path: Path) -> None:
        """set_rescoping(False) is called even when reorder raises."""
        started: list = []
        registry = MagicMock()
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)

        def boom(work_dir, commit_summary, **kwargs):
            raise RuntimeError("reorder exploded")

        _reorder_tasks_background(
            tmp_path,
            "cs",
            self._cfg(tmp_path),
            MagicMock(),
            repo_cfg=repo_cfg,
            registry=registry,
            _start=lambda t: started.append(t),
            _reorder_fn=boom,
            _coalesce_state={},
        )
        import pytest as _pytest

        with _pytest.raises(RuntimeError, match="reorder exploded"):
            self._run_thread(started)
        last_call = registry.set_rescoping.call_args_list[-1]
        assert last_call[0] == ("owner/repo", False)

    def test_no_rescoping_calls_when_no_registry(self, tmp_path: Path) -> None:
        """When registry is None, set_rescoping is not called."""
        started: list = []
        _, mock_reorder = self._capture_reorder_calls()
        _reorder_tasks_background(
            tmp_path,
            "cs",
            self._cfg(tmp_path),
            MagicMock(),
            _start=lambda t: started.append(t),
            _reorder_fn=mock_reorder,
            _coalesce_state={},
        )
        self._run_thread(started)
        # No registry provided — must not raise and must complete normally

    def test_sets_thread_local_repo_name_during_reorder(self, tmp_path: Path) -> None:
        """Thread-local repo_name is set to repo_cfg.name when reorder runs."""
        from fido.provider import current_repo

        started: list = []
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        seen: list = []

        def mock_reorder(work_dir, commit_summary, **kwargs):
            seen.append(current_repo())

        _reorder_tasks_background(
            tmp_path,
            "cs",
            self._cfg(tmp_path),
            MagicMock(),
            repo_cfg=repo_cfg,
            _start=lambda t: started.append(t),
            _reorder_fn=mock_reorder,
            _coalesce_state={},
        )
        self._run_thread(started)
        assert seen == ["owner/repo"]

    def test_clears_thread_local_repo_name_after_reorder(self, tmp_path: Path) -> None:
        """Thread-local repo_name is cleared in the finally block after reorder."""
        from fido.provider import current_repo, set_thread_repo

        started: list = []
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        _, mock_reorder = self._capture_reorder_calls()

        set_thread_repo("owner/repo")  # pre-set to confirm it gets cleared
        _reorder_tasks_background(
            tmp_path,
            "cs",
            self._cfg(tmp_path),
            MagicMock(),
            repo_cfg=repo_cfg,
            _start=lambda t: started.append(t),
            _reorder_fn=mock_reorder,
            _coalesce_state={},
        )
        self._run_thread(started)
        assert current_repo() is None

    def test_clears_thread_local_repo_name_on_reorder_exception(
        self, tmp_path: Path
    ) -> None:
        """Thread-local repo_name is cleared even when reorder raises."""
        from fido.provider import current_repo

        started: list = []
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)

        def boom(work_dir, commit_summary, **kwargs):
            raise RuntimeError("reorder exploded")

        _reorder_tasks_background(
            tmp_path,
            "cs",
            self._cfg(tmp_path),
            MagicMock(),
            repo_cfg=repo_cfg,
            _start=lambda t: started.append(t),
            _reorder_fn=boom,
            _coalesce_state={},
        )
        with pytest.raises(RuntimeError, match="reorder exploded"):
            self._run_thread(started)
        assert current_repo() is None

    def test_no_thread_local_set_when_no_repo_cfg(self, tmp_path: Path) -> None:
        """When repo_cfg is None, set_thread_repo is not called (no crash)."""
        from fido.provider import current_repo

        started: list = []
        seen: list = []

        def mock_reorder(work_dir, commit_summary, **kwargs):
            seen.append(current_repo())

        _reorder_tasks_background(
            tmp_path,
            "cs",
            self._cfg(tmp_path),
            MagicMock(),
            _start=lambda t: started.append(t),
            _reorder_fn=mock_reorder,
            _coalesce_state={},
        )
        self._run_thread(started)
        assert seen == [None]

    def test_sets_thread_kind_webhook_during_reorder(self, tmp_path: Path) -> None:
        """Thread kind is set to 'webhook' while the reorder loop runs (#955).

        The reorder thread must not register as 'worker' in the session talker —
        real webhooks would fire the cancel mechanism against it thinking it is
        the actual worker."""
        from fido.provider import current_thread_kind

        started: list = []
        seen: list = []

        def mock_reorder(work_dir, commit_summary, **kwargs):
            seen.append(current_thread_kind())

        _reorder_tasks_background(
            tmp_path,
            "cs",
            self._cfg(tmp_path),
            MagicMock(),
            _start=lambda t: started.append(t),
            _reorder_fn=mock_reorder,
            _coalesce_state={},
        )
        self._run_thread(started)
        assert seen == ["webhook"]

    def test_clears_thread_kind_after_reorder(self, tmp_path: Path) -> None:
        """Thread kind is cleared in the finally block after the reorder loop."""
        from fido.provider import current_thread_kind, set_thread_kind

        started: list = []
        _, mock_reorder = self._capture_reorder_calls()

        set_thread_kind("webhook")  # pre-set to confirm the finally block clears it
        _reorder_tasks_background(
            tmp_path,
            "cs",
            self._cfg(tmp_path),
            MagicMock(),
            _start=lambda t: started.append(t),
            _reorder_fn=mock_reorder,
            _coalesce_state={},
        )
        self._run_thread(started)
        # run_loop must clear kind in its finally block so the caller's
        # thread-local state is not polluted.
        assert current_thread_kind() == "worker"  # default when not set

    def test_clears_thread_kind_on_reorder_exception(self, tmp_path: Path) -> None:
        """Thread kind is cleared even when reorder raises."""
        from fido.provider import current_thread_kind

        started: list = []

        def boom(work_dir, commit_summary, **kwargs):
            raise RuntimeError("reorder exploded")

        _reorder_tasks_background(
            tmp_path,
            "cs",
            self._cfg(tmp_path),
            MagicMock(),
            _start=lambda t: started.append(t),
            _reorder_fn=boom,
            _coalesce_state={},
        )
        with pytest.raises(RuntimeError, match="reorder exploded"):
            self._run_thread(started)
        assert current_thread_kind() == "worker"  # default when not set


class TestNotifyThreadChange:
    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={"owner/repo": RepoConfig(name="owner/repo", work_dir=tmp_path)},
            allowed_bots=frozenset(),
            log_level="WARNING",
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
                "author": "alice",
                "comment_type": "issues",
            },
        }
        t.update(overrides)
        return t

    def test_completed_issue_comment_skips(self, tmp_path: Path) -> None:
        """Issue comments are silently skipped — webhook already replied."""
        cfg = self._cfg(tmp_path)
        mock_gh = MagicMock()
        change = {"task": self._task(), "kind": "completed"}
        agent = _client("Should not be called")
        _notify_thread_change(change, cfg, mock_gh, agent=agent)
        mock_gh.comment_issue.assert_not_called()
        mock_gh.reply_to_review_comment.assert_not_called()

    def test_modified_issue_comment_skips(self, tmp_path: Path) -> None:
        """Issue comments are silently skipped — webhook already replied."""
        cfg = self._cfg(tmp_path)
        mock_gh = MagicMock()
        change = {
            "task": self._task(),
            "kind": "modified",
            "new_title": "Updated title",
            "new_description": "",
        }
        agent = _client("Should not be called")
        _notify_thread_change(change, cfg, mock_gh, agent=agent)
        mock_gh.comment_issue.assert_not_called()
        mock_gh.reply_to_review_comment.assert_not_called()

    def test_missing_thread_skips_comment(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        mock_gh = MagicMock()
        task = self._task()
        task["thread"] = {}
        change = {"task": task, "kind": "completed"}
        _notify_thread_change(change, cfg, mock_gh, agent=_client())
        mock_gh.comment_issue.assert_not_called()

    def test_review_comment_run_turn_uses_retry_on_preempt(
        self, tmp_path: Path
    ) -> None:
        """run_turn must pass retry_on_preempt=True — #935.

        A webhook handler arriving while this voice turn runs preempts it,
        yielding result_len=0, cancelled=True.  Without retry_on_preempt,
        that empty string was indistinguishable from a real provider
        failure and used to raise ValueError, killing either the
        reorder-<repo> daemon or the worker's rescope_before_pick path.
        """
        cfg = self._cfg(tmp_path)
        mock_gh = MagicMock()
        task = self._task()
        task["thread"]["comment_type"] = "pulls"
        change = {"task": task, "kind": "completed"}
        agent = _client("Reply text")
        _notify_thread_change(change, cfg, mock_gh, agent=agent)
        # _client wraps a MagicMock — capture the run_turn kwargs.
        run_turn_kwargs = agent.run_turn.call_args.kwargs
        assert run_turn_kwargs.get("retry_on_preempt") is True

    def test_review_comment_empty_opus_raises(self, tmp_path: Path) -> None:
        """Empty Opus reply after retries raises — session reconnect handles
        recovery (#935)."""
        cfg = self._cfg(tmp_path)
        mock_gh = MagicMock()
        task = self._task()
        task["thread"]["comment_type"] = "pulls"
        change = {"task": task, "kind": "completed"}
        with pytest.raises(ValueError, match="run_turn returned empty"):
            _notify_thread_change(change, cfg, mock_gh, agent=_client(""))

    def test_review_comment_uses_reply_to_review_comment(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        mock_gh = MagicMock()
        task = self._task()
        task["thread"]["comment_type"] = "pulls"
        change = {"task": task, "kind": "completed"}
        _notify_thread_change(
            change,
            cfg,
            mock_gh,
            agent=_client("In-thread reply"),
        )
        mock_gh.reply_to_review_comment.assert_called_once_with(
            "owner/repo", 42, "In-thread reply", 999
        )
        mock_gh.comment_issue.assert_not_called()

    def test_review_comment_exception_does_not_raise(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        mock_gh = MagicMock()
        mock_gh.reply_to_review_comment.side_effect = RuntimeError("network")
        task = self._task()
        task["thread"]["comment_type"] = "pulls"
        change = {"task": task, "kind": "completed"}
        # Should not raise
        _notify_thread_change(change, cfg, mock_gh, agent=_client("ok"))

    def test_no_comment_type_defaults_to_skip(self, tmp_path: Path) -> None:
        """Missing comment_type defaults to the 'issues' skip path."""
        cfg = self._cfg(tmp_path)
        mock_gh = MagicMock()
        task = self._task()
        del task["thread"]["comment_type"]
        change = {"task": task, "kind": "completed"}
        _notify_thread_change(change, cfg, mock_gh, agent=_client("Fallback"))
        mock_gh.comment_issue.assert_not_called()
        mock_gh.reply_to_review_comment.assert_not_called()

    def test_author_in_opus_instruction(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        captured_prompt: list[str] = []

        def fake_pp(prompt, model, **kwargs):
            captured_prompt.append(prompt)
            return "ok"

        task = self._task()
        task["thread"]["comment_type"] = "pulls"
        change = {"task": task, "kind": "completed"}
        _notify_thread_change(
            change, cfg, MagicMock(), agent=_client(side_effect=fake_pp)
        )
        assert "alice" in captured_prompt[0]

    def test_issue_comment_skips_before_opus_call(self, tmp_path: Path) -> None:
        """Issue comments must not invoke the LLM — return before the expensive call."""
        cfg = self._cfg(tmp_path)
        mock_gh = MagicMock()
        invoked: list[bool] = []

        def should_not_be_called(prompt, model, **kwargs):
            invoked.append(True)
            return "oops"

        change = {"task": self._task(), "kind": "completed"}
        _notify_thread_change(
            change, cfg, mock_gh, agent=_client(side_effect=should_not_be_called)
        )
        assert not invoked
        mock_gh.comment_issue.assert_not_called()
        mock_gh.reply_to_review_comment.assert_not_called()

    def test_default_repo_configured_agent_used(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        mock_gh = MagicMock()
        task = self._task()
        task["thread"]["comment_type"] = "pulls"
        change = {"task": task, "kind": "completed"}
        with patch("fido.events.DefaultProviderFactory") as factory_cls:
            factory_cls.return_value.create_agent.return_value = _client("Auto reply")
            _notify_thread_change(change, cfg, mock_gh)
        factory_cls.return_value.create_agent.assert_called_once_with(
            cfg.repos["owner/repo"],
            work_dir=tmp_path,
            repo_name="owner/repo",
        )
        mock_gh.reply_to_review_comment.assert_called_once_with(
            "owner/repo", 42, "Auto reply", 999
        )
        mock_gh.comment_issue.assert_not_called()


class TestBackfillMissedPrComments:
    """Replay of issue_comment webhooks missed during fido downtime (fix #794).

    Only top-level PR comments are in scope — inline review comments and review
    threads are already scanned each iteration by ``Worker.handle_threads``.
    """

    def _cfg(
        self, tmp_path: Path, allowed_bots: frozenset[str] = frozenset()
    ) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={},
            allowed_bots=allowed_bots,
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )

    def _repo_cfg(
        self, tmp_path: Path, collaborators: frozenset[str] = frozenset({"rhencke"})
    ) -> RepoConfig:
        return RepoConfig(
            name="owner/repo",
            work_dir=tmp_path,
            membership=RepoMembership(collaborators=collaborators),
        )

    def _comment(
        self,
        comment_id: int,
        user: str = "rhencke",
        body: str = "hello",
    ) -> dict:
        return {
            "id": comment_id,
            "user": {"login": user},
            "body": body,
            "html_url": f"https://github.com/owner/repo/pull/1#issuecomment-{comment_id}",
        }

    def test_creates_task_for_allowed_collaborator_comment(
        self, tmp_path: Path
    ) -> None:
        from fido.events import backfill_missed_pr_comments

        mock_gh = MagicMock()
        mock_gh.get_issue_comments.return_value = [self._comment(100)]
        mock_gh.is_thread_resolved_for_comment.return_value = False
        cfg = self._cfg(tmp_path)
        repo_cfg = self._repo_cfg(tmp_path)
        with patch("fido.events.create_task") as mock_create:
            count = backfill_missed_pr_comments(
                cfg, repo_cfg, mock_gh, 1, gh_user="fidocancode"
            )
        assert count == 1
        mock_create.assert_called_once()
        _, kwargs = mock_create.call_args
        assert kwargs["thread"]["comment_id"] == 100
        assert kwargs["thread"]["comment_type"] == "issues"
        assert kwargs["thread"]["author"] == "rhencke"

    def test_skips_fido_own_comments(self, tmp_path: Path) -> None:
        from fido.events import backfill_missed_pr_comments

        mock_gh = MagicMock()
        mock_gh.get_issue_comments.return_value = [
            self._comment(100, user="fidocancode", body="my own reply")
        ]
        with patch("fido.events.create_task") as mock_create:
            backfill_missed_pr_comments(
                self._cfg(tmp_path),
                self._repo_cfg(tmp_path),
                mock_gh,
                1,
                gh_user="FidoCanCode",
            )
        mock_create.assert_not_called()

    def test_skips_by_gh_user_case_insensitive(self, tmp_path: Path) -> None:
        from fido.events import backfill_missed_pr_comments

        mock_gh = MagicMock()
        mock_gh.get_issue_comments.return_value = [
            self._comment(100, user="Alice", body="mine")
        ]
        with patch("fido.events.create_task") as mock_create:
            backfill_missed_pr_comments(
                self._cfg(tmp_path),
                self._repo_cfg(tmp_path),
                mock_gh,
                1,
                gh_user="alice",
            )
        mock_create.assert_not_called()

    def test_skips_fido_literal_name_even_if_gh_user_mismatch(
        self, tmp_path: Path
    ) -> None:
        """Defense in depth: even if ``gh_user`` is misconfigured, comments
        from the literal fido account must never trigger a backfill task."""
        from fido.events import backfill_missed_pr_comments

        mock_gh = MagicMock()
        mock_gh.get_issue_comments.return_value = [
            self._comment(100, user="fido-can-code", body="my reply")
        ]
        with patch("fido.events.create_task") as mock_create:
            backfill_missed_pr_comments(
                self._cfg(tmp_path),
                self._repo_cfg(tmp_path),
                mock_gh,
                1,
                gh_user="mis-configured-bot",
            )
        mock_create.assert_not_called()

    def test_skips_non_allowed_users(self, tmp_path: Path) -> None:
        from fido.events import backfill_missed_pr_comments

        mock_gh = MagicMock()
        mock_gh.get_issue_comments.return_value = [
            self._comment(100, user="random-stranger")
        ]
        with patch("fido.events.create_task") as mock_create:
            backfill_missed_pr_comments(
                self._cfg(tmp_path),
                self._repo_cfg(tmp_path, collaborators=frozenset({"rhencke"})),
                mock_gh,
                1,
                gh_user="fidocancode",
            )
        mock_create.assert_not_called()

    def test_allows_configured_bots(self, tmp_path: Path) -> None:
        from fido.events import backfill_missed_pr_comments

        mock_gh = MagicMock()
        mock_gh.get_issue_comments.return_value = [
            self._comment(100, user="dependabot[bot]", body="bump dep")
        ]
        with patch("fido.events.create_task") as mock_create:
            backfill_missed_pr_comments(
                self._cfg(tmp_path, allowed_bots=frozenset({"dependabot[bot]"})),
                self._repo_cfg(tmp_path),
                mock_gh,
                1,
                gh_user="fidocancode",
            )
        assert mock_create.call_count == 1
        _, kwargs = mock_create.call_args
        assert "bot" in kwargs["thread"]["author"]

    def test_prompt_marks_bot_vs_human(self, tmp_path: Path) -> None:
        from fido.events import backfill_missed_pr_comments

        mock_gh = MagicMock()
        mock_gh.get_issue_comments.return_value = [
            self._comment(100, user="rhencke", body="human msg"),
            self._comment(101, user="bot[bot]", body="bot msg"),
        ]
        with patch("fido.events.create_task") as mock_create:
            backfill_missed_pr_comments(
                self._cfg(tmp_path, allowed_bots=frozenset({"bot[bot]"})),
                self._repo_cfg(tmp_path),
                mock_gh,
                1,
                gh_user="fidocancode",
            )
        prompts = [c.args[0] for c in mock_create.call_args_list]
        assert any("human/owner" in p for p in prompts)
        assert any("(bot)" in p for p in prompts)

    def test_skips_empty_login_and_missing_id(self, tmp_path: Path) -> None:
        from fido.events import backfill_missed_pr_comments

        mock_gh = MagicMock()
        mock_gh.get_issue_comments.return_value = [
            {"id": 1, "user": {"login": ""}, "body": "x"},
            {"id": None, "user": {"login": "rhencke"}, "body": "x"},
            {"id": 2, "user": None, "body": "x"},
        ]
        with patch("fido.events.create_task") as mock_create:
            backfill_missed_pr_comments(
                self._cfg(tmp_path),
                self._repo_cfg(tmp_path),
                mock_gh,
                1,
                gh_user="fidocancode",
            )
        mock_create.assert_not_called()

    def test_empty_comment_list_is_noop(self, tmp_path: Path) -> None:
        from fido.events import backfill_missed_pr_comments

        mock_gh = MagicMock()
        mock_gh.get_issue_comments.return_value = []
        with patch("fido.events.create_task") as mock_create:
            count = backfill_missed_pr_comments(
                self._cfg(tmp_path),
                self._repo_cfg(tmp_path),
                mock_gh,
                1,
                gh_user="fidocancode",
            )
        assert count == 0
        mock_create.assert_not_called()

    def test_skips_already_claimed_comments(self, tmp_path: Path) -> None:
        """Comments with a durable SQLite claim are not re-queued on restart.

        reply_to_issue_comment completes comment ids in SQLite after posting;
        backfill must honour that durable claim and skip re-queueing.
        """
        from fido.events import backfill_missed_pr_comments

        mock_gh = MagicMock()
        mock_gh.get_issue_comments.return_value = [
            self._comment(100, body="already answered"),
            self._comment(200, body="not yet handled"),
        ]
        mock_gh.is_thread_resolved_for_comment.return_value = False
        promise = FidoStore(tmp_path).prepare_reply(
            owner="webhook", comment_type="issues", anchor_comment_id=100
        )
        assert promise is not None
        FidoStore(tmp_path).ack_promise(promise.promise_id)

        with patch("fido.events.create_task") as mock_create:
            backfill_missed_pr_comments(
                self._cfg(tmp_path),
                self._repo_cfg(tmp_path),
                mock_gh,
                1,
                gh_user="fidocancode",
            )

        # Only comment 200 (unclaimed) should be queued.
        assert mock_create.call_count == 1
        _, kwargs = mock_create.call_args
        assert kwargs["thread"]["comment_id"] == 200


class TestLaunchSync:
    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={},
            allowed_bots=frozenset(),
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )

    def _repo_cfg(self, tmp_path: Path) -> RepoConfig:
        return RepoConfig(name="owner/repo", work_dir=tmp_path)

    def test_calls_sync_tasks_background(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        mock_gh = MagicMock()
        with patch("fido.tasks.sync_tasks_background") as mock_sync:
            launch_sync(cfg, self._repo_cfg(tmp_path), mock_gh)
        mock_sync.assert_called_once_with(tmp_path, mock_gh)

    def test_does_not_raise(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        with patch("fido.tasks.sync_tasks_background"):
            launch_sync(cfg, self._repo_cfg(tmp_path), MagicMock())  # should not raise


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
                "name": "test",
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
            mock_gh,
            agent=_client("heart"),
        )  # must not raise


class TestReplyToCommentElseBranch:
    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={},
            allowed_bots=frozenset(),
            log_level="WARNING",
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
            patch("fido.events._triage", return_value=("UNKNOWN_CAT", ["do it"])),
            patch("fido.events.needs_more_context", return_value=False),
        ):
            cat, titles = reply_to_comment(
                action,
                cfg,
                self._repo_cfg(tmp_path),
                MagicMock(),
                agent=_client("I'll look into this."),
            )
        assert cat == "UNKNOWN_CAT"

    def test_gh_post_exception_propagates(self, tmp_path: Path) -> None:
        """Exception in reply_to_review_comment propagates so callers fail closed."""
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
        with pytest.raises(RuntimeError, match="network down"):
            reply_to_comment(
                action,
                cfg,
                self._repo_cfg(tmp_path),
                mock_gh,
                agent=_client(side_effect=fake_pp),
            )


class TestReplyToCommentTerseEnrichment:
    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={},
            allowed_bots=frozenset(),
            log_level="WARNING",
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

        def fake_triage(body, is_bot, context=None, *, agent=None, prompts=None):
            if context is not None:
                captured_context.update(context)
            return ("ACT", ["handle same comment"])

        mock_gh = MagicMock()
        mock_gh.fetch_sibling_threads.return_value = [
            {
                "path": "bar.py",
                "line": 1,
                "comments": [{"author": "rev", "body": "fix this"}],
            }
        ]

        with (
            patch("fido.events._triage", side_effect=fake_triage),
            patch("fido.events.needs_more_context", return_value=True),
        ):
            reply_to_comment(
                action,
                cfg,
                self._repo_cfg(tmp_path),
                mock_gh,
                agent=_client("On it!"),
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

        with patch("fido.events.needs_more_context", return_value=False):
            reply_to_comment(
                action,
                cfg,
                self._repo_cfg(tmp_path),
                mock_gh,
                agent=_client(side_effect=fake_pp),
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

        with patch("fido.events.needs_more_context", return_value=True):
            cat, titles = reply_to_comment(
                action,
                cfg,
                self._repo_cfg(tmp_path),
                mock_gh,
                agent=_client(side_effect=fake_pp),
            )

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

        def fake_triage(body, is_bot, context=None, *, agent=None, prompts=None):
            if context is not None:
                captured_context.update(context)
            return ("ACT", ["check caret comment"])

        mock_gh = MagicMock()
        mock_gh.fetch_sibling_threads.return_value = []

        with (
            patch("fido.events._triage", side_effect=fake_triage),
            patch("fido.events.needs_more_context", return_value=True),
        ):
            reply_to_comment(
                action,
                cfg,
                self._repo_cfg(tmp_path),
                mock_gh,
                agent=_client("On it!"),
            )

        assert "sibling_threads" not in captured_context


# ── reply_to_comment: thread re-fetch before posting ─────────────────────────


class TestReplyToCommentThreadRefetch:
    """The thread is re-fetched from GitHub right before posting so the
    edit-vs-post decision uses current state, not the stale snapshot from
    before triage.  Closes #438."""

    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={},
            allowed_bots=frozenset(),
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )

    def _repo_cfg(self, tmp_path: Path) -> RepoConfig:
        return RepoConfig(name="owner/repo", work_dir=tmp_path)

    def test_fetch_comment_thread_called_twice(self, tmp_path: Path) -> None:
        """fetch_comment_thread is called once for context (before triage) and
        once right before posting (after reply generation)."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 500},
            comment_body="please refactor this",
            is_bot=False,
        )

        def fake_pp(prompt, model, **kwargs):
            if model == "claude-haiku-4-5":
                return "NO"
            if "Triage" in prompt:
                return "ACT: refactor"
            if "Convert this PR review comment" in prompt:
                return "Refactor this module"
            return "On it!"

        mock_gh = MagicMock()
        mock_gh.fetch_comment_thread.return_value = [
            {"id": 500, "author": "reviewer", "body": "please refactor this"}
        ]

        reply_to_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            agent=_client(side_effect=fake_pp),
        )

        # Must be called exactly twice: initial context fetch + pre-post re-fetch
        assert mock_gh.fetch_comment_thread.call_count == 2
        mock_gh.fetch_comment_thread.assert_called_with("owner/repo", 1, 500)

    def test_refetch_result_used_for_edit_vs_post(self, tmp_path: Path) -> None:
        """The edit-vs-post decision uses re-fetched thread state, not the
        stale initial snapshot.  When the initial fetch shows Fido as last
        speaker (→ would edit) but the re-fetch reveals a human replied since
        (→ should post fresh), the re-fetch data wins: a new reply is posted.
        Note: the Fido reply ID is in both fetches, so the concurrent-skip
        guard is not triggered."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 501},
            comment_body="add type hints",
            is_bot=False,
        )

        def fake_pp(prompt, model, **kwargs):
            if model == "claude-haiku-4-5":
                return "NO"
            if "Triage" in prompt:
                return "ANSWER: acknowledged"
            return "Will do!"

        mock_gh = MagicMock()
        call_count = 0

        def fetch_side_effect(repo, pr, cid):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Initial fetch: Fido was last speaker (stale snapshot)
                return [
                    {"id": 501, "author": "reviewer", "body": "add type hints"},
                    {"id": 502, "author": "fidocancode", "body": "Got it!"},
                ]
            else:
                # Re-fetch: human commented after the initial fetch
                return [
                    {"id": 501, "author": "reviewer", "body": "add type hints"},
                    {"id": 502, "author": "fidocancode", "body": "Got it!"},
                    {"id": 503, "author": "reviewer", "body": "also type the return"},
                ]

        mock_gh.fetch_comment_thread.side_effect = fetch_side_effect

        reply_to_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            agent=_client(side_effect=fake_pp),
        )

        # Re-fetch shows human is last → post new reply, not edit
        # (Fido ID 502 existed in initial, so concurrent-skip is NOT triggered)
        mock_gh.reply_to_review_comment.assert_called_once()
        mock_gh.edit_review_comment.assert_not_called()

    def test_refetch_human_comment_added_during_triage_triggers_new_post(
        self, tmp_path: Path
    ) -> None:
        """If a human comments AFTER the initial fetch but BEFORE the re-fetch,
        the fresh data shows them as last speaker, so a new reply is posted
        rather than editing the old Fido reply."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 503},
            comment_body="please add tests",
            is_bot=False,
        )

        def fake_pp(prompt, model, **kwargs):
            if model == "claude-haiku-4-5":
                return "NO"
            if "Triage" in prompt:
                return "ACT: add tests"
            if "Convert this PR review comment" in prompt:
                return "Add tests"
            return "Adding tests now!"

        mock_gh = MagicMock()
        call_count = 0

        def fetch_side_effect(repo, pr, cid):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Initial fetch: Fido is last speaker
                return [
                    {"id": 503, "author": "reviewer", "body": "please add tests"},
                    {"id": 504, "author": "fidocancode", "body": "On it!"},
                ]
            else:
                # Re-fetch: human replied after the initial fetch
                return [
                    {"id": 503, "author": "reviewer", "body": "please add tests"},
                    {"id": 504, "author": "fidocancode", "body": "On it!"},
                    {
                        "id": 505,
                        "author": "reviewer",
                        "body": "also add integration tests",
                    },
                ]

        mock_gh.fetch_comment_thread.side_effect = fetch_side_effect

        reply_to_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            agent=_client(side_effect=fake_pp),
        )

        # Fresh data shows human is last speaker → post new reply, never edit
        mock_gh.reply_to_review_comment.assert_called_once()
        mock_gh.edit_review_comment.assert_not_called()

    def test_refetch_returns_empty_falls_back_to_initial(self, tmp_path: Path) -> None:
        """If the re-fetch returns empty/None (e.g. race with deletion), the
        stale initial snapshot is kept and posting proceeds normally."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 506},
            comment_body="fix the import",
            is_bot=False,
        )

        def fake_pp(prompt, model, **kwargs):
            if model == "claude-haiku-4-5":
                return "NO"
            if "Triage" in prompt:
                return "ACT: fix import"
            if "Convert this PR review comment" in prompt:
                return "Fix the import"
            return "Fixed!"

        mock_gh = MagicMock()
        call_count = 0

        def fetch_side_effect(repo, pr, cid):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [{"id": 506, "author": "reviewer", "body": "fix the import"}]
            else:
                return None  # re-fetch returned nothing

        mock_gh.fetch_comment_thread.side_effect = fetch_side_effect

        reply_to_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            agent=_client(side_effect=fake_pp),
        )

        # Falls back to initial snapshot (no Fido reply) → posts new reply
        mock_gh.reply_to_review_comment.assert_called_once()
        mock_gh.edit_review_comment.assert_not_called()

    def test_skips_post_when_concurrent_fido_reply_detected(
        self, tmp_path: Path
    ) -> None:
        """If a new Fido reply appears in the re-fetch that wasn't in the
        initial snapshot, a concurrent handler already replied — skip to
        avoid duplicates.  Closes #438."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 507},
            comment_body="please add docstrings",
            is_bot=False,
        )

        def fake_pp(prompt, model, **kwargs):
            if model == "claude-haiku-4-5":
                return "NO"
            if "Triage" in prompt:
                return "ACT: add docstrings"
            if "Convert this PR review comment" in prompt:
                return "Add docstrings"
            return "Woof, on it!"

        mock_gh = MagicMock()
        call_count = 0

        def fetch_side_effect(repo, pr, cid):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Initial fetch: no Fido reply yet
                return [
                    {"id": 507, "author": "reviewer", "body": "please add docstrings"}
                ]
            else:
                # Re-fetch: concurrent handler posted a Fido reply to THIS
                # comment (in_reply_to_id == 507) during triage.
                return [
                    {"id": 507, "author": "reviewer", "body": "please add docstrings"},
                    {
                        "id": 508,
                        "author": "fidocancode",
                        "body": "On it!",
                        "in_reply_to_id": 507,
                    },
                ]

        mock_gh.fetch_comment_thread.side_effect = fetch_side_effect

        cat, titles = reply_to_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            agent=_client(side_effect=fake_pp),
        )

        # Concurrent handler already replied — neither post nor edit is called
        mock_gh.reply_to_review_comment.assert_not_called()
        mock_gh.edit_review_comment.assert_not_called()
        # Triage result is still returned so the caller can queue tasks
        assert cat == "ACT"
        assert titles == ["Add docstrings"]

    def test_no_skip_when_concurrent_reply_is_to_sibling_comment(
        self, tmp_path: Path
    ) -> None:
        """A Fido reply that appeared during triage but targets a *different*
        comment (sibling in the same review) must NOT trip the skip — that
        was the #1004 silent-drop bug.  Closes #1004."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 507},
            comment_body="please add docstrings",
            is_bot=False,
        )

        def fake_pp(prompt, model, **kwargs):
            if model == "claude-haiku-4-5":
                return "NO"
            if "Triage" in prompt:
                return "ACT: add docstrings"
            if "Convert this PR review comment" in prompt:
                return "Add docstrings"
            return "Woof, on it!"

        mock_gh = MagicMock()
        call_count = 0

        def fetch_side_effect(repo, pr, cid):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [
                    {"id": 507, "author": "reviewer", "body": "please add docstrings"}
                ]
            # Re-fetch: a sibling comment (id 600) got a fido reply (id 601)
            # while we were triaging — that's NOT a reply to OUR comment.
            return [
                {"id": 507, "author": "reviewer", "body": "please add docstrings"},
                {
                    "id": 601,
                    "author": "fidocancode",
                    "body": "Sibling reply",
                    "in_reply_to_id": 600,
                },
            ]

        mock_gh.fetch_comment_thread.side_effect = fetch_side_effect

        reply_to_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            agent=_client(side_effect=fake_pp),
        )

        # Sibling-comment reply must NOT skip our post — we still reply.
        mock_gh.reply_to_review_comment.assert_called_once()

    def test_no_skip_when_fido_reply_was_already_in_initial_fetch(
        self, tmp_path: Path
    ) -> None:
        """A Fido reply that existed in the initial fetch is not treated as
        a concurrent duplicate — the edit-vs-post flow proceeds normally."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 509},
            comment_body="looks good to me",
            is_bot=False,
        )

        def fake_pp(prompt, model, **kwargs):
            if model == "claude-haiku-4-5":
                return "NO"
            if "Triage" in prompt:
                return "ANSWER: acknowledged"
            return "Thanks for the feedback!"

        mock_gh = MagicMock()

        def fetch_side_effect(repo, pr, cid):
            # Both fetches return the same Fido reply — it was already there
            return [
                {"id": 509, "author": "reviewer", "body": "looks good"},
                {"id": 510, "author": "fidocancode", "body": "On it!"},
            ]

        mock_gh.fetch_comment_thread.side_effect = fetch_side_effect

        reply_to_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            agent=_client(side_effect=fake_pp),
        )

        # Posted replies are immutable; Fido posts a new reply instead.
        mock_gh.reply_to_review_comment.assert_called_once()
        mock_gh.edit_review_comment.assert_not_called()

    def test_skips_post_fido_can_code_login_also_detected(self, tmp_path: Path) -> None:
        """The 'fido-can-code' login is also recognised as a Fido reply
        when checking for concurrent duplicates."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 511},
            comment_body="fix the typo",
            is_bot=False,
        )

        def fake_pp(prompt, model, **kwargs):
            if model == "claude-haiku-4-5":
                return "NO"
            if "Triage" in prompt:
                return "ACT: fix typo"
            if "Convert this PR review comment" in prompt:
                return "Fix the typo"
            return "Fixed the typo!"

        mock_gh = MagicMock()
        call_count = 0

        def fetch_side_effect(repo, pr, cid):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [{"id": 511, "author": "reviewer", "body": "fix the typo"}]
            else:
                # Concurrent reply from the fido-can-code login variant,
                # threaded under THIS comment.
                return [
                    {"id": 511, "author": "reviewer", "body": "fix the typo"},
                    {
                        "id": 512,
                        "author": "fido-can-code",
                        "body": "Fixed!",
                        "in_reply_to_id": 511,
                    },
                ]

        mock_gh.fetch_comment_thread.side_effect = fetch_side_effect

        reply_to_comment(
            action,
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            agent=_client(side_effect=fake_pp),
        )

        # Concurrent Fido reply detected (via fido-can-code) — skip
        mock_gh.reply_to_review_comment.assert_not_called()
        mock_gh.edit_review_comment.assert_not_called()


# ── _rewrite_pr_description ───────────────────────────────────────────────────


class TestRewritePrDescription:
    @pytest.fixture(autouse=True)
    def _mock_pr_body_lock(self):
        from contextlib import nullcontext

        with patch("fido.tasks.pr_body_lock", return_value=nullcontext()):
            yield

    def _pr_body(self, desc: str = "Does something useful.\n\nFixes #42.") -> str:
        return (
            f"{desc}\n\n---\n\n## Work queue\n\n"
            "<!-- WORK_QUEUE_START -->\n- [ ] do a thing\n<!-- WORK_QUEUE_END -->"
        )

    def _mock_gh(self, body: str | None = None) -> MagicMock:
        gh = MagicMock()
        gh.get_repo_info.return_value = "owner/repo"
        gh.get_user.return_value = "fido"
        gh.find_pr.return_value = {"number": 99, "state": "OPEN"}
        gh.get_pr_body.return_value = body if body is not None else self._pr_body()
        return gh

    def _mock_state(self, issue: int | None = 42) -> MagicMock:
        state = MagicMock()
        state.load.return_value = {"issue": issue} if issue else {}
        return state

    def _mock_tasks(self, task_list: list | None = None) -> MagicMock:
        tasks = MagicMock()
        tasks.list.return_value = task_list if task_list is not None else []
        return tasks

    def test_skips_when_no_issue_in_state(self, tmp_path: Path) -> None:
        mock_gh = self._mock_gh()
        _rewrite_pr_description(
            tmp_path,
            mock_gh,
            agent=_client(),
            _state=self._mock_state(issue=None),
        )
        mock_gh.edit_pr_body.assert_not_called()

    def test_raises_on_get_repo_exception(self, tmp_path: Path) -> None:
        mock_gh = self._mock_gh()
        mock_gh.get_repo_info.side_effect = RuntimeError("network error")
        with pytest.raises(RuntimeError, match="network error"):
            _rewrite_pr_description(
                tmp_path,
                mock_gh,
                agent=_client(),
                _state=self._mock_state(),
            )
        mock_gh.edit_pr_body.assert_not_called()

    def test_skips_when_no_open_pr(self, tmp_path: Path) -> None:
        mock_gh = self._mock_gh()
        mock_gh.find_pr.return_value = None
        _rewrite_pr_description(
            tmp_path,
            mock_gh,
            agent=_client(),
            _state=self._mock_state(),
        )
        mock_gh.edit_pr_body.assert_not_called()

    def test_skips_when_pr_not_open(self, tmp_path: Path) -> None:
        mock_gh = self._mock_gh()
        mock_gh.find_pr.return_value = {"number": 99, "state": "MERGED"}
        _rewrite_pr_description(
            tmp_path,
            mock_gh,
            agent=_client(),
            _state=self._mock_state(),
        )
        mock_gh.edit_pr_body.assert_not_called()

    def test_raises_on_get_pr_body_exception(self, tmp_path: Path) -> None:
        mock_gh = self._mock_gh()
        mock_gh.get_pr_body.side_effect = RuntimeError("API error")
        with pytest.raises(RuntimeError, match="API error"):
            _rewrite_pr_description(
                tmp_path,
                mock_gh,
                agent=_client(),
                _state=self._mock_state(),
                _tasks=self._mock_tasks(),
            )
        mock_gh.edit_pr_body.assert_not_called()

    def test_raises_when_no_divider_in_body(self, tmp_path: Path) -> None:
        mock_gh = self._mock_gh(
            body="No divider here. <!-- WORK_QUEUE_START -->x<!-- WORK_QUEUE_END -->"
        )
        with pytest.raises(ValueError, match="no --- divider"):
            _rewrite_pr_description(
                tmp_path,
                mock_gh,
                agent=_client(),
                _state=self._mock_state(),
                _tasks=self._mock_tasks(),
            )
        mock_gh.edit_pr_body.assert_not_called()

    def test_raises_when_opus_returns_empty(self, tmp_path: Path) -> None:
        mock_gh = self._mock_gh()
        with pytest.raises(ValueError, match="run_turn returned empty"):
            _rewrite_pr_description(
                tmp_path,
                mock_gh,
                agent=_client(""),
                _state=self._mock_state(),
                _tasks=self._mock_tasks(),
            )
        mock_gh.edit_pr_body.assert_not_called()

    def test_updates_pr_body_with_new_description(self, tmp_path: Path) -> None:
        mock_gh = self._mock_gh()
        _rewrite_pr_description(
            tmp_path,
            mock_gh,
            agent=_client("<body>New description.\n\nFixes #42.</body>"),
            _state=self._mock_state(),
            _tasks=self._mock_tasks(),
        )
        mock_gh.edit_pr_body.assert_called_once()
        new_body = mock_gh.edit_pr_body.call_args[0][2]
        assert "New description." in new_body

    def test_preserves_work_queue_section(self, tmp_path: Path) -> None:
        mock_gh = self._mock_gh()
        _rewrite_pr_description(
            tmp_path,
            mock_gh,
            agent=_client("<body>Updated description.\n\nFixes #42.</body>"),
            _state=self._mock_state(),
            _tasks=self._mock_tasks(),
        )
        new_body = mock_gh.edit_pr_body.call_args[0][2]
        assert "<!-- WORK_QUEUE_START -->" in new_body
        assert "do a thing" in new_body
        assert "<!-- WORK_QUEUE_END -->" in new_body

    def test_description_replaces_only_before_divider(self, tmp_path: Path) -> None:
        mock_gh = self._mock_gh()
        _rewrite_pr_description(
            tmp_path,
            mock_gh,
            agent=_client("<body>Fresh desc.\n\nFixes #42.</body>"),
            _state=self._mock_state(),
            _tasks=self._mock_tasks(),
        )
        new_body = mock_gh.edit_pr_body.call_args[0][2]
        assert "Does something useful." not in new_body
        assert "Fresh desc." in new_body
        assert "## Work queue" in new_body

    def test_raises_on_edit_pr_body_exception(self, tmp_path: Path) -> None:
        mock_gh = self._mock_gh()
        mock_gh.edit_pr_body.side_effect = RuntimeError("write failed")
        with pytest.raises(RuntimeError, match="write failed"):
            _rewrite_pr_description(
                tmp_path,
                mock_gh,
                agent=_client("<body>New desc.\n\nFixes #42.</body>"),
                _state=self._mock_state(),
                _tasks=self._mock_tasks(),
            )

    def test_defaults_to_none_agent(self, tmp_path: Path) -> None:
        mock_gh = self._mock_gh()
        with patch("fido.worker._write_pr_description") as mock_write:
            _rewrite_pr_description(
                tmp_path,
                mock_gh,
                _state=self._mock_state(),
                _tasks=self._mock_tasks(),
            )
        mock_write.assert_called_once()
        assert mock_write.call_args.kwargs.get("agent") is None

    def test_defaults_to_state(self, tmp_path: Path) -> None:
        mock_gh = self._mock_gh()
        mock_state = self._mock_state(issue=None)
        with patch("fido.state.State", return_value=mock_state) as mock_state_cls:
            _rewrite_pr_description(
                tmp_path,
                mock_gh,
                agent=_client(),
                _tasks=self._mock_tasks(),
            )
        mock_state_cls.assert_called_once_with(tmp_path / ".git" / "fido")
        mock_gh.edit_pr_body.assert_not_called()

    def test_does_not_retry_when_task_list_unchanged(self, tmp_path: Path) -> None:
        """When task list is stable, description is written exactly once."""
        task = {"id": "t1", "status": "pending", "title": "Do a thing"}
        tasks = MagicMock()
        tasks.list.return_value = [task]  # same list every call
        mock_gh = self._mock_gh()
        _rewrite_pr_description(
            tmp_path,
            mock_gh,
            agent=_client("<body>New desc.\n\nFixes #42.</body>"),
            _state=self._mock_state(),
            _tasks=tasks,
        )
        mock_gh.edit_pr_body.assert_called_once()

    def test_retries_when_task_list_changes_during_opus(self, tmp_path: Path) -> None:
        """If task list changes while Opus runs, the description is rewritten."""
        task_before = {"id": "t1", "status": "pending", "title": "Do a thing"}
        task_after = {"id": "t2", "status": "pending", "title": "New task"}
        tasks = MagicMock()
        # list() called: before attempt 1, after attempt 1, before attempt 2, after attempt 2
        tasks.list.side_effect = [
            [task_before],  # snapshot before attempt 1
            [task_after],  # snapshot after attempt 1 (changed → retry)
            [task_after],  # snapshot before attempt 2
            [task_after],  # snapshot after attempt 2 (stable → done)
        ]
        mock_gh = self._mock_gh()
        _rewrite_pr_description(
            tmp_path,
            mock_gh,
            agent=_client("<body>New desc.\n\nFixes #42.</body>"),
            _state=self._mock_state(),
            _tasks=tasks,
        )
        assert mock_gh.edit_pr_body.call_count == 2

    def test_stops_after_max_retries(self, tmp_path: Path) -> None:
        """Never retries more than _max_retries times even if task list keeps changing."""
        tasks = MagicMock()
        call_count = [0]

        def ever_changing():
            n = call_count[0]
            call_count[0] += 1
            return [{"id": str(n), "status": "pending", "title": f"task {n}"}]

        tasks.list.side_effect = ever_changing
        mock_gh = self._mock_gh()
        _rewrite_pr_description(
            tmp_path,
            mock_gh,
            agent=_client("<body>New desc.\n\nFixes #42.</body>"),
            _state=self._mock_state(),
            _tasks=tasks,
            _max_retries=3,
        )
        assert mock_gh.edit_pr_body.call_count == 3

    def test_no_divider_raises_before_retry(self, tmp_path: Path) -> None:
        """When the PR body has no --- divider, ValueError propagates immediately."""
        mock_gh = self._mock_gh(body="No divider here. Nothing.")
        with pytest.raises(ValueError, match="no --- divider"):
            _rewrite_pr_description(
                tmp_path,
                mock_gh,
                agent=_client("<body>New desc.\n\nFixes #42.</body>"),
                _state=self._mock_state(),
                _tasks=self._mock_tasks(),
            )
        mock_gh.edit_pr_body.assert_not_called()

    def test_refetches_pr_body_on_retry(self, tmp_path: Path) -> None:
        """PR body is re-fetched on each attempt so work-queue stays current."""
        task_before = {"id": "t1", "status": "pending", "title": "Before"}
        task_after = {"id": "t2", "status": "pending", "title": "After"}
        tasks = MagicMock()
        tasks.list.side_effect = [
            [task_before],
            [task_after],  # changed → retry
            [task_after],
            [task_after],  # stable
        ]
        mock_gh = self._mock_gh()
        _rewrite_pr_description(
            tmp_path,
            mock_gh,
            agent=_client("<body>New desc.\n\nFixes #42.</body>"),
            _state=self._mock_state(),
            _tasks=tasks,
        )
        assert mock_gh.get_pr_body.call_count == 2  # once per attempt


# ── _task_snapshot ────────────────────────────────────────────────────────────


class TestTaskSnapshot:
    def test_returns_id_status_title_tuples(self) -> None:
        tasks = [
            {"id": "a", "status": "pending", "title": "Do A"},
            {"id": "b", "status": "completed", "title": "Done B"},
        ]
        assert _task_snapshot(tasks) == [
            ("a", "pending", "Do A"),
            ("b", "completed", "Done B"),
        ]

    def test_empty_list(self) -> None:
        assert _task_snapshot([]) == []

    def test_missing_fields_default_to_empty_string(self) -> None:
        tasks = [{"id": "x"}]
        assert _task_snapshot(tasks) == [("x", "", "")]

    def test_order_is_preserved(self) -> None:
        tasks = [
            {"id": "z", "status": "pending", "title": "Z"},
            {"id": "a", "status": "pending", "title": "A"},
        ]
        result = _task_snapshot(tasks)
        assert result[0][0] == "z"
        assert result[1][0] == "a"
