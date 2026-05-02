"""Tests for fido.synthesis_call — synthesis LLM call and JSON parsing."""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from fido.synthesis import AddReaction, NoOp, RescopeIntent
from fido.synthesis_call import (
    MAX_RETRIES,
    SynthesisExhaustedError,
    _extract_json_candidates,
    _parse_action,
    _parse_comment_response,
    call_synthesis,
)
from fido.types import ActiveIssue, ActivePR

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw(
    reasoning: str = "thinking",
    reply_text: str = "My reply.",
    actions: list[dict[str, Any]] | None = None,
) -> str:
    return json.dumps(
        {
            "reasoning": reasoning,
            "reply_text": reply_text,
            "actions": actions if actions is not None else [],
        }
    )


def _make_agent(return_value: str | list[str]) -> MagicMock:
    """Return a mock agent whose run_turn returns *return_value*.

    If *return_value* is a list, successive calls return successive elements.
    """
    agent = MagicMock()
    if isinstance(return_value, list):
        agent.run_turn.side_effect = return_value
    else:
        agent.run_turn.return_value = return_value
    return agent


def _make_prompts(
    system: str = "sys",
    user: str = "user",
) -> MagicMock:
    prompts = MagicMock()
    prompts.synthesis_system_prompt.return_value = system
    prompts.synthesis_prompt.return_value = user
    return prompts


# ---------------------------------------------------------------------------
# _extract_json_candidates
# ---------------------------------------------------------------------------


class TestExtractJsonCandidates:
    def test_returns_stripped_raw(self) -> None:
        candidates = _extract_json_candidates('  {"a": 1}  ')
        assert '{"a": 1}' in candidates

    def test_returns_raw_as_first_candidate(self) -> None:
        raw = '{"a": 1}'
        candidates = _extract_json_candidates(raw)
        assert candidates[0] == raw

    def test_extracts_span_when_preamble_present(self) -> None:
        raw = 'Here is the JSON: {"a": 1} done.'
        candidates = _extract_json_candidates(raw)
        assert '{"a": 1}' in candidates

    def test_no_braces_returns_single_candidate(self) -> None:
        raw = "not json at all"
        candidates = _extract_json_candidates(raw)
        assert candidates == ("not json at all",)

    def test_does_not_duplicate_when_stripped_equals_span(self) -> None:
        raw = '{"a": 1}'
        candidates = _extract_json_candidates(raw)
        # No duplicates
        assert len(candidates) == len(set(candidates))


# ---------------------------------------------------------------------------
# _parse_action
# ---------------------------------------------------------------------------


class TestParseAction:
    def test_add_reaction_valid(self) -> None:
        action = _parse_action({"type": "add_reaction", "emoji": "rocket"})
        assert action == AddReaction("rocket")

    def test_add_reaction_invalid_emoji_returns_none(self) -> None:
        action = _parse_action({"type": "add_reaction", "emoji": "thinking"})
        assert action is None

    def test_add_reaction_empty_emoji_returns_none(self) -> None:
        action = _parse_action({"type": "add_reaction", "emoji": ""})
        assert action is None

    def test_rescope_intent_valid(self) -> None:
        action = _parse_action({"type": "rescope_intent", "description": "Add logging"})
        assert action == RescopeIntent("Add logging")

    def test_rescope_intent_empty_description_returns_none(self) -> None:
        action = _parse_action({"type": "rescope_intent", "description": ""})
        assert action is None

    def test_rescope_intent_whitespace_description_returns_none(self) -> None:
        action = _parse_action({"type": "rescope_intent", "description": "   "})
        assert action is None

    def test_no_op(self) -> None:
        action = _parse_action({"type": "no_op"})
        assert action == NoOp()

    def test_unknown_type_returns_none(self) -> None:
        action = _parse_action({"type": "create_task", "title": "x"})
        assert action is None

    def test_missing_type_returns_none(self) -> None:
        action = _parse_action({"emoji": "rocket"})
        assert action is None

    def test_preempt_type_returns_none(self) -> None:
        # Preempt is no longer in the vocabulary — unknown types are skipped.
        action = _parse_action({"type": "preempt", "preempt": True})
        assert action is None


# ---------------------------------------------------------------------------
# _parse_comment_response
# ---------------------------------------------------------------------------


class TestParseCommentResponse:
    def test_valid_minimal(self) -> None:
        raw = _make_raw()
        r = _parse_comment_response(raw)
        assert r.reply_text == "My reply."
        assert r.reasoning == "thinking"
        assert r.actions == ()

    def test_valid_with_actions(self) -> None:
        raw = _make_raw(
            actions=[
                {"type": "add_reaction", "emoji": "rocket"},
                {"type": "rescope_intent", "description": "Do more tests"},
                {"type": "no_op"},
            ]
        )
        r = _parse_comment_response(raw)
        assert len(r.actions) == 3
        assert isinstance(r.actions[0], AddReaction)
        assert isinstance(r.actions[1], RescopeIntent)
        assert isinstance(r.actions[2], NoOp)

    def test_malformed_json_raises(self) -> None:
        with pytest.raises(ValueError):
            _parse_comment_response("not json")

    def test_missing_reply_text_raises(self) -> None:
        raw = json.dumps({"reasoning": "r", "actions": []})
        with pytest.raises(ValueError, match="reply_text"):
            _parse_comment_response(raw)

    def test_empty_reply_text_raises(self) -> None:
        raw = json.dumps({"reasoning": "r", "reply_text": "", "actions": []})
        with pytest.raises(ValueError, match="reply_text"):
            _parse_comment_response(raw)

    def test_whitespace_reply_text_raises(self) -> None:
        raw = json.dumps({"reasoning": "r", "reply_text": "   ", "actions": []})
        with pytest.raises(ValueError, match="reply_text"):
            _parse_comment_response(raw)

    def test_json_array_raises(self) -> None:
        with pytest.raises(ValueError):
            _parse_comment_response("[1, 2, 3]")

    def test_json_wrapped_in_preamble(self) -> None:
        inner = _make_raw()
        raw = f"Here's the JSON:\n{inner}\nDone."
        r = _parse_comment_response(raw)
        assert r.reply_text == "My reply."

    def test_unknown_action_skipped(self) -> None:
        raw = _make_raw(
            actions=[
                {"type": "create_task", "title": "x"},
                {"type": "add_reaction", "emoji": "eyes"},
            ]
        )
        r = _parse_comment_response(raw)
        # create_task is skipped; only add_reaction survives
        assert len(r.actions) == 1
        assert isinstance(r.actions[0], AddReaction)

    def test_non_dict_action_items_skipped(self) -> None:
        raw = json.dumps(
            {
                "reasoning": "r",
                "reply_text": "OK.",
                "actions": ["string_item", 42, {"type": "no_op"}],
            }
        )
        r = _parse_comment_response(raw)
        assert len(r.actions) == 1
        assert isinstance(r.actions[0], NoOp)

    def test_missing_reasoning_defaults_to_empty(self) -> None:
        raw = json.dumps({"reply_text": "Reply.", "actions": []})
        r = _parse_comment_response(raw)
        assert r.reasoning == ""

    def test_missing_actions_defaults_to_empty_tuple(self) -> None:
        raw = json.dumps({"reasoning": "r", "reply_text": "Reply."})
        r = _parse_comment_response(raw)
        assert r.actions == ()


# ---------------------------------------------------------------------------
# call_synthesis
# ---------------------------------------------------------------------------


class TestCallSynthesis:
    def test_success_on_first_attempt(self) -> None:
        raw = _make_raw(reply_text="Great feedback!")
        agent = _make_agent(raw)
        prompts = _make_prompts()

        result = call_synthesis(
            "Please fix this", is_bot=False, agent=agent, prompts=prompts
        )

        assert result.reply_text == "Great feedback!"
        assert agent.run_turn.call_count == 1

    def test_passes_system_prompt_to_agent(self) -> None:
        agent = _make_agent(_make_raw())
        prompts = _make_prompts(system="my-system-prompt")

        call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        _, kwargs = agent.run_turn.call_args
        assert kwargs["system_prompt"] == "my-system-prompt"

    def test_passes_user_prompt_to_agent(self) -> None:
        agent = _make_agent(_make_raw())
        prompts = _make_prompts(user="my-user-prompt")

        call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        args, _ = agent.run_turn.call_args
        assert args[0] == "my-user-prompt"

    def test_retry_on_parse_failure_then_success(self) -> None:
        raw_bad = "not json"
        raw_good = _make_raw(reply_text="Fixed!")
        agent = _make_agent([raw_bad, raw_good])
        prompts = _make_prompts()

        result = call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        assert result.reply_text == "Fixed!"
        assert agent.run_turn.call_count == 2

    def test_retry_appends_suffix_to_prompt(self) -> None:
        raw_bad = "not json"
        raw_good = _make_raw()
        agent = _make_agent([raw_bad, raw_good])
        prompts = _make_prompts(user="base-prompt")

        call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        first_call_prompt = agent.run_turn.call_args_list[0][0][0]
        second_call_prompt = agent.run_turn.call_args_list[1][0][0]
        assert first_call_prompt == "base-prompt"
        assert second_call_prompt != "base-prompt"
        assert second_call_prompt.startswith("base-prompt")

    def test_exhausts_all_retries_and_raises(self) -> None:
        agent = _make_agent(["bad"] * MAX_RETRIES)
        prompts = _make_prompts()

        with pytest.raises(SynthesisExhaustedError, match="exhausted"):
            call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        assert agent.run_turn.call_count == MAX_RETRIES

    def test_exhaustion_error_mentions_constraint_b(self) -> None:
        agent = _make_agent(["bad"] * MAX_RETRIES)
        prompts = _make_prompts()

        with pytest.raises(SynthesisExhaustedError, match="Constraint B"):
            call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

    def test_passes_issue_to_system_prompt(self) -> None:
        agent = _make_agent(_make_raw())
        prompts = _make_prompts()
        issue = ActiveIssue(number=7, title="Fix crash", body="It crashes.")

        call_synthesis(
            "comment", is_bot=False, agent=agent, prompts=prompts, issue=issue
        )

        prompts.synthesis_system_prompt.assert_called_once_with(issue=issue, pr=None)

    def test_passes_pr_to_system_prompt(self) -> None:
        agent = _make_agent(_make_raw())
        prompts = _make_prompts()
        issue = ActiveIssue(number=7, title="T", body="")
        pr = ActivePR(
            number=42, title="My PR", url="https://github.com/a/b/pull/42", body=""
        )

        call_synthesis(
            "comment", is_bot=False, agent=agent, prompts=prompts, issue=issue, pr=pr
        )

        prompts.synthesis_system_prompt.assert_called_once_with(issue=issue, pr=pr)

    def test_passes_is_bot_to_synthesis_prompt(self) -> None:
        agent = _make_agent(_make_raw())
        prompts = _make_prompts()

        call_synthesis("comment", is_bot=True, agent=agent, prompts=prompts)

        call_kwargs = prompts.synthesis_prompt.call_args
        assert call_kwargs[1]["is_bot"] is True

    def test_passes_context_to_synthesis_prompt(self) -> None:
        agent = _make_agent(_make_raw())
        prompts = _make_prompts()
        ctx = {"pr_title": "My PR"}

        call_synthesis(
            "comment", is_bot=False, context=ctx, agent=agent, prompts=prompts
        )

        call_kwargs = prompts.synthesis_prompt.call_args
        assert call_kwargs[1]["context"] == ctx

    def test_provider_error_propagates_immediately(self) -> None:
        agent = MagicMock()
        agent.run_turn.side_effect = RuntimeError("provider down")
        prompts = _make_prompts()

        with pytest.raises(RuntimeError, match="provider down"):
            call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        # No retries — provider errors propagate immediately
        assert agent.run_turn.call_count == 1

    def test_success_with_all_action_types(self) -> None:
        raw = _make_raw(
            reply_text="Here's what I think.",
            actions=[
                {"type": "add_reaction", "emoji": "heart"},
                {"type": "rescope_intent", "description": "Drop the parser refactor"},
                {"type": "no_op"},
            ],
        )
        agent = _make_agent(raw)
        prompts = _make_prompts()

        result = call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        assert len(result.actions) == 3
        assert isinstance(result.actions[0], AddReaction)
        assert result.actions[0].emoji == "heart"
        assert isinstance(result.actions[1], RescopeIntent)
        assert result.actions[1].description == "Drop the parser refactor"
        assert isinstance(result.actions[2], NoOp)

    def test_default_context_is_none(self) -> None:
        agent = _make_agent(_make_raw())
        prompts = _make_prompts()

        call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        call_kwargs = prompts.synthesis_prompt.call_args
        assert call_kwargs[1].get("context") is None

    def test_max_retries_constant_is_three(self) -> None:
        assert MAX_RETRIES == 3
