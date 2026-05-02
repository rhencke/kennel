"""Tests for fido.synthesis — CommentResponse, outcome bridge, and validation."""

import pytest

from fido.rocq.replied_comment_claims import ReviewAct, ReviewAnswer
from fido.synthesis import (
    VALID_REACTIONS,
    CommentResponse,
    Insight,
    outcome_for_response,
    validate_reaction,
)

# ---------------------------------------------------------------------------
# VALID_REACTIONS constant
# ---------------------------------------------------------------------------


class TestValidReactions:
    def test_contains_expected_shortcodes(self) -> None:
        assert "+1" in VALID_REACTIONS
        assert "-1" in VALID_REACTIONS
        assert "rocket" in VALID_REACTIONS
        assert "eyes" in VALID_REACTIONS
        assert "heart" in VALID_REACTIONS
        assert "laugh" in VALID_REACTIONS
        assert "confused" in VALID_REACTIONS
        assert "hooray" in VALID_REACTIONS

    def test_is_frozenset(self) -> None:
        assert isinstance(VALID_REACTIONS, frozenset)

    def test_exactly_eight_reactions(self) -> None:
        assert len(VALID_REACTIONS) == 8


# ---------------------------------------------------------------------------
# validate_reaction
# ---------------------------------------------------------------------------


class TestValidateReaction:
    def test_valid_reaction_returns_emoji(self) -> None:
        assert validate_reaction("rocket") == "rocket"

    def test_all_valid_reactions_pass(self) -> None:
        for emoji in VALID_REACTIONS:
            assert validate_reaction(emoji) == emoji

    def test_invalid_reaction_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid reaction"):
            validate_reaction("thinking")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid reaction"):
            validate_reaction("")

    def test_error_message_includes_valid_list(self) -> None:
        with pytest.raises(ValueError, match="Valid reactions"):
            validate_reaction("notanemoji")


# ---------------------------------------------------------------------------
# CommentResponse
# ---------------------------------------------------------------------------


class TestInsight:
    def test_construction(self) -> None:
        i = Insight(title="Test insight", hook="A thing happened.", why="It matters.")
        assert i.title == "Test insight"
        assert i.hook == "A thing happened."
        assert i.why == "It matters."

    def test_frozen(self) -> None:
        i = Insight(title="T", hook="H", why="W")
        with pytest.raises((AttributeError, TypeError)):
            i.title = "changed"  # type: ignore[misc]

    def test_equality(self) -> None:
        i1 = Insight(title="T", hook="H", why="W")
        i2 = Insight(title="T", hook="H", why="W")
        assert i1 == i2

    def test_inequality(self) -> None:
        i1 = Insight(title="T", hook="H", why="W")
        i2 = Insight(title="T2", hook="H", why="W")
        assert i1 != i2


class TestCommentResponse:
    def _make(
        self,
        reasoning: str = "thought about it",
        reply_text: str = "Here is my reply.",
        emoji: str | None = None,
        change_request: str | None = None,
        insights: list[Insight] | None = None,
    ) -> CommentResponse:
        return CommentResponse(
            reasoning=reasoning,
            reply_text=reply_text,
            emoji=emoji,
            change_request=change_request,
            insights=insights if insights is not None else [],
        )

    def test_construction_minimal(self) -> None:
        r = self._make()
        assert r.reasoning == "thought about it"
        assert r.reply_text == "Here is my reply."
        assert r.emoji is None
        assert r.change_request is None
        assert r.insights == []

    def test_construction_with_emoji(self) -> None:
        r = self._make(emoji="rocket")
        assert r.emoji == "rocket"

    def test_construction_with_change_request(self) -> None:
        r = self._make(change_request="Drop the parser refactor")
        assert r.change_request == "Drop the parser refactor"

    def test_construction_with_both(self) -> None:
        r = self._make(emoji="heart", change_request="Add logging")
        assert r.emoji == "heart"
        assert r.change_request == "Add logging"

    def test_frozen(self) -> None:
        r = self._make()
        with pytest.raises((AttributeError, TypeError)):
            r.reply_text = "changed"  # type: ignore[misc]

    def test_empty_reply_text_raises(self) -> None:
        with pytest.raises(ValueError, match="reply_text must be non-empty"):
            CommentResponse(reasoning="thinking", reply_text="")

    def test_whitespace_only_reply_text_raises(self) -> None:
        with pytest.raises(ValueError, match="reply_text must be non-empty"):
            CommentResponse(reasoning="thinking", reply_text="   ")

    def test_newline_only_reply_text_raises(self) -> None:
        with pytest.raises(ValueError, match="reply_text must be non-empty"):
            CommentResponse(reasoning="thinking", reply_text="\n\t\n")

    def test_reply_text_with_leading_trailing_whitespace_accepted(self) -> None:
        r = self._make(reply_text="  actual text  ")
        assert r.reply_text == "  actual text  "

    def test_invalid_emoji_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid reaction"):
            CommentResponse(reasoning="r", reply_text="Reply.", emoji="thinking")

    def test_empty_emoji_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid reaction"):
            CommentResponse(reasoning="r", reply_text="Reply.", emoji="")

    def test_empty_change_request_raises(self) -> None:
        with pytest.raises(ValueError, match="change_request must be non-empty"):
            CommentResponse(reasoning="r", reply_text="Reply.", change_request="")

    def test_whitespace_change_request_raises(self) -> None:
        with pytest.raises(ValueError, match="change_request must be non-empty"):
            CommentResponse(reasoning="r", reply_text="Reply.", change_request="   ")

    def test_change_request_with_padding_accepted(self) -> None:
        r = self._make(change_request="  actual intent  ")
        assert r.change_request == "  actual intent  "

    def test_equality(self) -> None:
        r1 = self._make()
        r2 = self._make()
        assert r1 == r2

    def test_inequality_on_reply_text(self) -> None:
        r1 = self._make(reply_text="Hello.")
        r2 = self._make(reply_text="Goodbye.")
        assert r1 != r2

    def test_constraint_b_error_message_mentions_constraint(self) -> None:
        with pytest.raises(ValueError, match="Constraint B"):
            CommentResponse(reasoning="x", reply_text="")

    def test_default_emoji_is_none(self) -> None:
        r = CommentResponse(reasoning="r", reply_text="Reply.")
        assert r.emoji is None

    def test_default_change_request_is_none(self) -> None:
        r = CommentResponse(reasoning="r", reply_text="Reply.")
        assert r.change_request is None

    def test_default_insights_is_empty_list(self) -> None:
        r = CommentResponse(reasoning="r", reply_text="Reply.")
        assert r.insights == []

    def test_construction_with_insights(self) -> None:
        insight = Insight(title="Good point", hook="Rob favors X.", why="Because Y.")
        r = self._make(insights=[insight])
        assert r.insights == [insight]

    def test_multiple_insights(self) -> None:
        i1 = Insight(title="A", hook="H1", why="W1")
        i2 = Insight(title="B", hook="H2", why="W2")
        r = self._make(insights=[i1, i2])
        assert len(r.insights) == 2
        assert r.insights[0] == i1
        assert r.insights[1] == i2

    def test_all_valid_emojis_accepted(self) -> None:
        for emoji in VALID_REACTIONS:
            r = self._make(emoji=emoji)
            assert r.emoji == emoji


# ---------------------------------------------------------------------------
# outcome_for_response
# ---------------------------------------------------------------------------


class TestOutcomeForResponse:
    def test_change_request_present_returns_review_act(self) -> None:
        r = CommentResponse(
            reasoning="r", reply_text="Reply.", change_request="Add logging"
        )
        assert outcome_for_response(r) == ReviewAct()

    def test_change_request_absent_returns_review_answer(self) -> None:
        r = CommentResponse(reasoning="r", reply_text="Reply.")
        assert outcome_for_response(r) == ReviewAnswer()

    def test_emoji_only_returns_review_answer(self) -> None:
        r = CommentResponse(reasoning="r", reply_text="Reply.", emoji="rocket")
        assert outcome_for_response(r) == ReviewAnswer()

    def test_both_emoji_and_change_request_returns_review_act(self) -> None:
        r = CommentResponse(
            reasoning="r",
            reply_text="Reply.",
            emoji="heart",
            change_request="Reorder tasks",
        )
        assert outcome_for_response(r) == ReviewAct()
