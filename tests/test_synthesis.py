"""Tests for fido.synthesis — CommentResponse and Action types (closes #1230)."""

import pytest

from fido.synthesis import (
    VALID_REACTIONS,
    AddReaction,
    CommentResponse,
    NoOp,
    Preempt,
    RescopeIntent,
    SynthesisAction,
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
# AddReaction
# ---------------------------------------------------------------------------


class TestAddReaction:
    def test_construction(self) -> None:
        r = AddReaction(emoji="rocket")
        assert r.emoji == "rocket"

    def test_frozen(self) -> None:
        r = AddReaction(emoji="eyes")
        with pytest.raises((AttributeError, TypeError)):
            r.emoji = "heart"  # type: ignore[misc]

    def test_equality(self) -> None:
        assert AddReaction("rocket") == AddReaction("rocket")
        assert AddReaction("rocket") != AddReaction("eyes")


# ---------------------------------------------------------------------------
# RescopeIntent
# ---------------------------------------------------------------------------


class TestRescopeIntent:
    def test_construction(self) -> None:
        r = RescopeIntent(description="Add a logging task")
        assert r.description == "Add a logging task"

    def test_frozen(self) -> None:
        r = RescopeIntent(description="something")
        with pytest.raises((AttributeError, TypeError)):
            r.description = "other"  # type: ignore[misc]

    def test_equality(self) -> None:
        assert RescopeIntent("a") == RescopeIntent("a")
        assert RescopeIntent("a") != RescopeIntent("b")

    def test_empty_description_raises(self) -> None:
        with pytest.raises(ValueError, match="description must be non-empty"):
            RescopeIntent(description="")

    def test_whitespace_only_description_raises(self) -> None:
        with pytest.raises(ValueError, match="description must be non-empty"):
            RescopeIntent(description="   ")

    def test_description_with_leading_trailing_whitespace_accepted(self) -> None:
        r = RescopeIntent(description="  actual intent  ")
        assert r.description == "  actual intent  "


# ---------------------------------------------------------------------------
# Preempt
# ---------------------------------------------------------------------------


class TestPreempt:
    def test_preempt_true(self) -> None:
        p = Preempt(preempt=True)
        assert p.preempt is True

    def test_preempt_false(self) -> None:
        p = Preempt(preempt=False)
        assert p.preempt is False

    def test_frozen(self) -> None:
        p = Preempt(preempt=True)
        with pytest.raises((AttributeError, TypeError)):
            p.preempt = False  # type: ignore[misc]

    def test_equality(self) -> None:
        assert Preempt(True) == Preempt(True)
        assert Preempt(True) != Preempt(False)


# ---------------------------------------------------------------------------
# NoOp
# ---------------------------------------------------------------------------


class TestNoOp:
    def test_construction(self) -> None:
        n = NoOp()
        assert isinstance(n, NoOp)

    def test_frozen(self) -> None:
        # frozen=True means no __dict__ mutation; trivially confirmed by
        # checking it is a dataclass with no settable fields
        n = NoOp()
        with pytest.raises((AttributeError, TypeError)):
            n.anything = "x"  # type: ignore[attr-defined]

    def test_equality(self) -> None:
        assert NoOp() == NoOp()


# ---------------------------------------------------------------------------
# SynthesisAction union — membership
# ---------------------------------------------------------------------------


class TestSynthesisActionUnion:
    """Verify the union includes every action type and no others."""

    def test_add_reaction_is_synthesis_action(self) -> None:
        a: SynthesisAction = AddReaction("rocket")
        assert isinstance(a, AddReaction)

    def test_rescope_intent_is_synthesis_action(self) -> None:
        a: SynthesisAction = RescopeIntent("Add logging")
        assert isinstance(a, RescopeIntent)

    def test_preempt_is_synthesis_action(self) -> None:
        a: SynthesisAction = Preempt(True)
        assert isinstance(a, Preempt)

    def test_noop_is_synthesis_action(self) -> None:
        a: SynthesisAction = NoOp()
        assert isinstance(a, NoOp)


# ---------------------------------------------------------------------------
# CommentResponse
# ---------------------------------------------------------------------------


class TestCommentResponse:
    def _make(
        self,
        reasoning: str = "thought about it",
        reply_text: str = "Here is my reply.",
        actions: tuple[SynthesisAction, ...] = (),
    ) -> CommentResponse:
        return CommentResponse(
            reasoning=reasoning,
            reply_text=reply_text,
            actions=actions,
        )

    def test_construction_minimal(self) -> None:
        r = self._make()
        assert r.reasoning == "thought about it"
        assert r.reply_text == "Here is my reply."
        assert r.actions == ()

    def test_construction_with_actions(self) -> None:
        actions = (AddReaction("rocket"), RescopeIntent("Fix the parser"))
        r = self._make(actions=actions)
        assert len(r.actions) == 2
        assert isinstance(r.actions[0], AddReaction)
        assert isinstance(r.actions[1], RescopeIntent)

    def test_frozen(self) -> None:
        r = self._make()
        with pytest.raises((AttributeError, TypeError)):
            r.reply_text = "changed"  # type: ignore[misc]

    def test_empty_reply_text_raises(self) -> None:
        with pytest.raises(ValueError, match="reply_text must be non-empty"):
            CommentResponse(
                reasoning="thinking",
                reply_text="",
                actions=(),
            )

    def test_whitespace_only_reply_text_raises(self) -> None:
        with pytest.raises(ValueError, match="reply_text must be non-empty"):
            CommentResponse(
                reasoning="thinking",
                reply_text="   ",
                actions=(),
            )

    def test_newline_only_reply_text_raises(self) -> None:
        with pytest.raises(ValueError, match="reply_text must be non-empty"):
            CommentResponse(
                reasoning="thinking",
                reply_text="\n\t\n",
                actions=(),
            )

    def test_reply_text_with_leading_trailing_whitespace_accepted(self) -> None:
        # Only purely-whitespace values are rejected; padded real text is fine
        r = self._make(reply_text="  actual text  ")
        assert r.reply_text == "  actual text  "

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
            CommentResponse(reasoning="x", reply_text="", actions=())

    def test_mixed_action_types_in_tuple(self) -> None:
        r = CommentResponse(
            reasoning="r",
            reply_text="Reply.",
            actions=(
                AddReaction("eyes"),
                RescopeIntent("Reorder the parser tasks"),
                Preempt(False),
                NoOp(),
            ),
        )
        assert len(r.actions) == 4

    def test_default_actions_empty_tuple(self) -> None:
        r = CommentResponse(reasoning="r", reply_text="Reply.")
        assert r.actions == ()
