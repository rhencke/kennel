"""Tests for fido.synthesis_executor — CommentResponse dispatch to GitHub effects."""

from unittest.mock import MagicMock

import pytest

from fido.rocq.replied_comment_claims import ReviewAct, ReviewAnswer
from fido.synthesis import CommentResponse
from fido.synthesis_executor import CommentTarget, SynthesisExecutor
from fido.types import RescоpeIntent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(
    reply_text: str = "My reply.",
    emoji: str | None = None,
    change_request: str | None = None,
) -> CommentResponse:
    return CommentResponse(
        reasoning="thinking",
        reply_text=reply_text,
        emoji=emoji,
        change_request=change_request,
    )


def _make_target(
    repo: str = "owner/repo",
    pr: int = 42,
    comment_id: int = 100,
    comment_type: str = "pulls",
) -> CommentTarget:
    return CommentTarget(
        repo=repo, pr=pr, comment_id=comment_id, comment_type=comment_type
    )


def _make_gh() -> MagicMock:
    gh = MagicMock()
    gh.reply_to_review_comment.return_value = {"id": 999}
    gh.comment_issue.return_value = {"id": 999}
    return gh


def _make_rescope() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# CommentTarget
# ---------------------------------------------------------------------------


class TestCommentTarget:
    def test_construction(self) -> None:
        t = _make_target()
        assert t.repo == "owner/repo"
        assert t.pr == 42
        assert t.comment_id == 100
        assert t.comment_type == "pulls"

    def test_frozen(self) -> None:
        t = _make_target()
        with pytest.raises((AttributeError, TypeError)):
            t.repo = "other"  # type: ignore[misc]

    def test_equality(self) -> None:
        assert _make_target() == _make_target()
        assert _make_target(pr=1) != _make_target(pr=2)


# ---------------------------------------------------------------------------
# SynthesisExecutor — reply posting
# ---------------------------------------------------------------------------


class TestExecutorPostsReply:
    def test_posts_review_comment_reply(self) -> None:
        gh = _make_gh()
        executor = SynthesisExecutor(gh)
        target = _make_target(comment_type="pulls")

        executor.execute(_make_response(reply_text="Good point."), target)

        gh.reply_to_review_comment.assert_called_once_with(
            "owner/repo", 42, "Good point.", 100
        )

    def test_posts_issue_comment_for_issues_type(self) -> None:
        gh = _make_gh()
        executor = SynthesisExecutor(gh)
        target = _make_target(comment_type="issues")

        executor.execute(_make_response(reply_text="Thanks!"), target)

        gh.comment_issue.assert_called_once_with("owner/repo", 42, "Thanks!")
        gh.reply_to_review_comment.assert_not_called()

    def test_always_posts_reply(self) -> None:
        gh = _make_gh()
        executor = SynthesisExecutor(gh)

        executor.execute(_make_response(), _make_target())

        assert gh.reply_to_review_comment.called or gh.comment_issue.called


# ---------------------------------------------------------------------------
# SynthesisExecutor — emoji reaction
# ---------------------------------------------------------------------------


class TestExecutorEmojiReaction:
    def test_adds_reaction_when_emoji_present(self) -> None:
        gh = _make_gh()
        executor = SynthesisExecutor(gh)

        executor.execute(_make_response(emoji="rocket"), _make_target())

        gh.add_reaction.assert_called_once_with("owner/repo", "pulls", 100, "rocket")

    def test_no_reaction_when_emoji_none(self) -> None:
        gh = _make_gh()
        executor = SynthesisExecutor(gh)

        executor.execute(_make_response(emoji=None), _make_target())

        gh.add_reaction.assert_not_called()

    def test_reaction_uses_target_comment_type(self) -> None:
        gh = _make_gh()
        executor = SynthesisExecutor(gh)
        target = _make_target(comment_type="issues", comment_id=77)

        executor.execute(_make_response(emoji="heart"), target)

        gh.add_reaction.assert_called_once_with("owner/repo", "issues", 77, "heart")


# ---------------------------------------------------------------------------
# SynthesisExecutor — rescope trigger
# ---------------------------------------------------------------------------


class TestExecutorRescope:
    def test_triggers_rescope_when_change_request_present(self) -> None:
        gh = _make_gh()
        rescope = _make_rescope()
        executor = SynthesisExecutor(gh, rescope=rescope)

        executor.execute(_make_response(change_request="Add logging"), _make_target())

        rescope.trigger_rescope.assert_called_once()
        intent = rescope.trigger_rescope.call_args[0][0]
        assert isinstance(intent, RescоpeIntent)
        assert intent.change_request == "Add logging"
        assert intent.comment_id == 100  # from _make_target()
        assert intent.timestamp  # non-empty ISO timestamp

    def test_rescope_intent_comment_id_matches_target(self) -> None:
        gh = _make_gh()
        rescope = _make_rescope()
        executor = SynthesisExecutor(gh, rescope=rescope)
        target = _make_target(comment_id=999)

        executor.execute(_make_response(change_request="Refactor"), target)

        intent = rescope.trigger_rescope.call_args[0][0]
        assert intent.comment_id == 999

    def test_no_rescope_when_change_request_none(self) -> None:
        gh = _make_gh()
        rescope = _make_rescope()
        executor = SynthesisExecutor(gh, rescope=rescope)

        executor.execute(_make_response(change_request=None), _make_target())

        rescope.trigger_rescope.assert_not_called()

    def test_no_rescope_trigger_configured_does_not_crash(self) -> None:
        gh = _make_gh()
        executor = SynthesisExecutor(gh)  # no rescope trigger

        # Should not raise — just logs a warning
        executor.execute(_make_response(change_request="Add logging"), _make_target())


# ---------------------------------------------------------------------------
# SynthesisExecutor — outcome
# ---------------------------------------------------------------------------


class TestExecutorOutcome:
    def test_returns_review_act_when_change_request_present(self) -> None:
        gh = _make_gh()
        executor = SynthesisExecutor(gh)

        outcome = executor.execute(
            _make_response(change_request="Reorder tasks"), _make_target()
        )

        assert outcome == ReviewAct()

    def test_returns_review_answer_when_no_change_request(self) -> None:
        gh = _make_gh()
        executor = SynthesisExecutor(gh)

        outcome = executor.execute(_make_response(), _make_target())

        assert outcome == ReviewAnswer()

    def test_emoji_only_returns_review_answer(self) -> None:
        gh = _make_gh()
        executor = SynthesisExecutor(gh)

        outcome = executor.execute(_make_response(emoji="rocket"), _make_target())

        assert outcome == ReviewAnswer()

    def test_both_emoji_and_change_request_returns_review_act(self) -> None:
        gh = _make_gh()
        executor = SynthesisExecutor(gh)

        outcome = executor.execute(
            _make_response(emoji="heart", change_request="Fix parser"),
            _make_target(),
        )

        assert outcome == ReviewAct()


# ---------------------------------------------------------------------------
# SynthesisExecutor — effect ordering
# ---------------------------------------------------------------------------


class TestExecutorEffectOrder:
    def test_reply_before_reaction(self) -> None:
        """Reply is always posted before adding a reaction."""
        gh = _make_gh()
        executor = SynthesisExecutor(gh)
        call_order: list[str] = []
        gh.reply_to_review_comment.side_effect = lambda *a, **kw: call_order.append(
            "reply"
        )
        gh.add_reaction.side_effect = lambda *a, **kw: call_order.append("reaction")

        executor.execute(_make_response(emoji="rocket"), _make_target())

        assert call_order == ["reply", "reaction"]

    def test_reply_before_rescope(self) -> None:
        """Reply is always posted before triggering rescope."""
        gh = _make_gh()
        rescope = _make_rescope()
        executor = SynthesisExecutor(gh, rescope=rescope)
        call_order: list[str] = []
        gh.reply_to_review_comment.side_effect = lambda *a, **kw: call_order.append(
            "reply"
        )
        rescope.trigger_rescope.side_effect = lambda *a, **kw: call_order.append(
            "rescope"
        )

        executor.execute(_make_response(change_request="Add tests"), _make_target())

        assert call_order == ["reply", "rescope"]

    def test_all_effects_in_order(self) -> None:
        """Full ordering: reply → reaction → rescope."""
        gh = _make_gh()
        rescope = _make_rescope()
        executor = SynthesisExecutor(gh, rescope=rescope)
        call_order: list[str] = []
        gh.reply_to_review_comment.side_effect = lambda *a, **kw: call_order.append(
            "reply"
        )
        gh.add_reaction.side_effect = lambda *a, **kw: call_order.append("reaction")
        rescope.trigger_rescope.side_effect = lambda *a, **kw: call_order.append(
            "rescope"
        )

        executor.execute(
            _make_response(emoji="heart", change_request="Reorder"),
            _make_target(),
        )

        assert call_order == ["reply", "reaction", "rescope"]


# ---------------------------------------------------------------------------
# SynthesisExecutor — execute_effects_only
# ---------------------------------------------------------------------------


class TestExecutorEffectsOnly:
    """execute_effects_only handles emoji + rescope but NOT reply posting."""

    def test_does_not_post_reply(self) -> None:
        gh = _make_gh()
        executor = SynthesisExecutor(gh)

        executor.execute_effects_only(_make_response("My reply."), _make_target())

        gh.reply_to_review_comment.assert_not_called()
        gh.comment_issue.assert_not_called()

    def test_removes_eyes_reaction_before_emoji(self) -> None:
        gh = _make_gh()
        gh.list_reactions.return_value = [
            {"id": 42, "content": "eyes"},
        ]
        call_order: list[str] = []
        gh.delete_reaction.side_effect = lambda *a, **kw: call_order.append(
            "remove_eyes"
        )
        gh.add_reaction.side_effect = lambda *a, **kw: call_order.append("add_emoji")
        executor = SynthesisExecutor(gh)

        executor.execute_effects_only(_make_response(emoji="rocket"), _make_target())

        assert call_order == ["remove_eyes", "add_emoji"]

    def test_removes_eyes_reaction(self) -> None:
        gh = _make_gh()
        gh.list_reactions.return_value = [
            {"id": 42, "content": "eyes"},
        ]
        executor = SynthesisExecutor(gh)

        executor.execute_effects_only(_make_response(), _make_target())

        gh.delete_reaction.assert_called_once_with("owner/repo", "pulls", 100, 42)

    def test_ignores_non_eyes_reactions(self) -> None:
        gh = _make_gh()
        gh.list_reactions.return_value = [
            {"id": 10, "content": "heart"},
            {"id": 11, "content": "+1"},
        ]
        executor = SynthesisExecutor(gh)

        executor.execute_effects_only(_make_response(), _make_target())

        gh.delete_reaction.assert_not_called()

    def test_list_reactions_error_does_not_propagate(self) -> None:
        gh = _make_gh()
        gh.list_reactions.side_effect = RuntimeError("API error")
        executor = SynthesisExecutor(gh)

        # Should not raise
        outcome = executor.execute_effects_only(_make_response(), _make_target())

        assert outcome is not None

    def test_adds_emoji_reaction(self) -> None:
        gh = _make_gh()
        gh.list_reactions.return_value = []
        executor = SynthesisExecutor(gh)

        executor.execute_effects_only(
            _make_response(emoji="heart"), _make_target(comment_type="issues")
        )

        gh.add_reaction.assert_called_once_with("owner/repo", "issues", 100, "heart")

    def test_no_emoji_no_add_reaction(self) -> None:
        gh = _make_gh()
        gh.list_reactions.return_value = []
        executor = SynthesisExecutor(gh)

        executor.execute_effects_only(_make_response(emoji=None), _make_target())

        gh.add_reaction.assert_not_called()

    def test_triggers_rescope(self) -> None:
        gh = _make_gh()
        gh.list_reactions.return_value = []
        rescope = _make_rescope()
        executor = SynthesisExecutor(gh, rescope=rescope)

        executor.execute_effects_only(
            _make_response(change_request="Add logging"), _make_target()
        )

        rescope.trigger_rescope.assert_called_once()
        intent = rescope.trigger_rescope.call_args[0][0]
        assert isinstance(intent, RescоpeIntent)
        assert intent.change_request == "Add logging"
        assert intent.comment_id == 100
        assert intent.timestamp

    def test_no_rescope_when_change_request_none(self) -> None:
        gh = _make_gh()
        gh.list_reactions.return_value = []
        rescope = _make_rescope()
        executor = SynthesisExecutor(gh, rescope=rescope)

        executor.execute_effects_only(
            _make_response(change_request=None), _make_target()
        )

        rescope.trigger_rescope.assert_not_called()

    def test_no_rescope_trigger_configured_does_not_crash(self) -> None:
        gh = _make_gh()
        gh.list_reactions.return_value = []
        executor = SynthesisExecutor(gh)  # no rescope trigger

        # Should not raise
        executor.execute_effects_only(
            _make_response(change_request="Add logging"), _make_target()
        )

    def test_returns_review_act_when_change_request_present(self) -> None:
        gh = _make_gh()
        gh.list_reactions.return_value = []
        executor = SynthesisExecutor(gh)

        outcome = executor.execute_effects_only(
            _make_response(change_request="Add logging"), _make_target()
        )

        assert outcome == ReviewAct()

    def test_returns_review_answer_when_no_change_request(self) -> None:
        gh = _make_gh()
        gh.list_reactions.return_value = []
        executor = SynthesisExecutor(gh)

        outcome = executor.execute_effects_only(
            _make_response(change_request=None), _make_target()
        )

        assert outcome == ReviewAnswer()
