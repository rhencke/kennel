"""Tests for fido.synthesis_call — synthesis LLM call and JSON parsing."""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from fido.synthesis import Insight
from fido.synthesis_call import (
    MAX_RETRIES,
    SynthesisExhaustedError,
    _extract_json_objects,
    _parse_comment_response,
    call_failure_explanation,
    call_synthesis,
)
from fido.types import ActiveIssue, ActivePR

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw(
    reasoning: str = "thinking",
    reply_text: str = "My reply.",
    emoji: str | None = None,
    change_request: str | None = None,
    insights: list[dict[str, str]] | None = None,
) -> str:
    obj: dict[str, Any] = {
        "reasoning": reasoning,
        "reply_text": reply_text,
    }
    if emoji is not None:
        obj["emoji"] = emoji
    if change_request is not None:
        obj["change_request"] = change_request
    if insights is not None:
        obj["insights"] = insights
    return json.dumps(obj)


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
# _extract_json_objects
# ---------------------------------------------------------------------------


class TestExtractJsonObjects:
    def test_returns_parsed_dict_for_clean_json(self) -> None:
        result = _extract_json_objects('{"a": 1}')
        assert result == [{"a": 1}]

    def test_returns_empty_for_no_braces(self) -> None:
        result = _extract_json_objects("not json at all")
        assert result == []

    def test_finds_object_when_preamble_present(self) -> None:
        result = _extract_json_objects('Here is the JSON: {"a": 1} done.')
        assert result == [{"a": 1}]

    def test_skips_invalid_brace_and_continues(self) -> None:
        result = _extract_json_objects('{ not json, then {"a": 1}')
        assert result == [{"a": 1}]

    def test_returns_all_objects_in_order(self) -> None:
        result = _extract_json_objects('{"a": 1} then {"b": 2}')
        assert result == [{"a": 1}, {"b": 2}]

    def test_handles_nested_objects(self) -> None:
        result = _extract_json_objects('{"outer": {"inner": 42}}')
        assert result == [{"outer": {"inner": 42}}]

    def test_skips_non_dict_json_values(self) -> None:
        # A JSON array starting with [ has no {, and a bare number has no {.
        # A JSON string has no {. Confirm arrays are skipped.
        result = _extract_json_objects("[1, 2, 3]")
        assert result == []

    def test_handles_leading_and_trailing_whitespace(self) -> None:
        result = _extract_json_objects('  {"a": 1}  ')
        assert result == [{"a": 1}]

    def test_returns_empty_for_empty_string(self) -> None:
        result = _extract_json_objects("")
        assert result == []


# ---------------------------------------------------------------------------
# _parse_comment_response
# ---------------------------------------------------------------------------


class TestParseCommentResponse:
    def test_valid_minimal(self) -> None:
        raw = _make_raw()
        r = _parse_comment_response(raw)
        assert r.reply_text == "My reply."
        assert r.reasoning == "thinking"
        assert r.emoji is None
        assert r.change_request is None
        assert r.insights == []

    def test_valid_with_emoji(self) -> None:
        raw = _make_raw(emoji="rocket")
        r = _parse_comment_response(raw)
        assert r.emoji == "rocket"

    def test_valid_with_change_request(self) -> None:
        raw = _make_raw(change_request="Add more tests")
        r = _parse_comment_response(raw)
        assert r.change_request == "Add more tests"

    def test_valid_with_both(self) -> None:
        raw = _make_raw(emoji="heart", change_request="Reorder the tasks")
        r = _parse_comment_response(raw)
        assert r.emoji == "heart"
        assert r.change_request == "Reorder the tasks"

    def test_malformed_json_raises(self) -> None:
        with pytest.raises(ValueError):
            _parse_comment_response("not json")

    def test_missing_reply_text_raises(self) -> None:
        raw = json.dumps({"reasoning": "r"})
        with pytest.raises(ValueError, match="reply_text"):
            _parse_comment_response(raw)

    def test_empty_reply_text_raises(self) -> None:
        raw = json.dumps({"reasoning": "r", "reply_text": ""})
        with pytest.raises(ValueError, match="reply_text"):
            _parse_comment_response(raw)

    def test_whitespace_reply_text_raises(self) -> None:
        raw = json.dumps({"reasoning": "r", "reply_text": "   "})
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

    def test_invalid_emoji_dropped(self) -> None:
        raw = json.dumps({"reasoning": "r", "reply_text": "OK.", "emoji": "thinking"})
        r = _parse_comment_response(raw)
        assert r.emoji is None

    def test_empty_emoji_dropped(self) -> None:
        raw = json.dumps({"reasoning": "r", "reply_text": "OK.", "emoji": ""})
        r = _parse_comment_response(raw)
        assert r.emoji is None

    def test_null_emoji_stays_none(self) -> None:
        raw = json.dumps({"reasoning": "r", "reply_text": "OK.", "emoji": None})
        r = _parse_comment_response(raw)
        assert r.emoji is None

    def test_whitespace_change_request_dropped(self) -> None:
        raw = json.dumps(
            {"reasoning": "r", "reply_text": "OK.", "change_request": "   "}
        )
        r = _parse_comment_response(raw)
        assert r.change_request is None

    def test_null_change_request_stays_none(self) -> None:
        raw = json.dumps(
            {"reasoning": "r", "reply_text": "OK.", "change_request": None}
        )
        r = _parse_comment_response(raw)
        assert r.change_request is None

    def test_missing_reasoning_defaults_to_empty(self) -> None:
        raw = json.dumps({"reply_text": "Reply."})
        r = _parse_comment_response(raw)
        assert r.reasoning == ""

    def test_non_string_emoji_dropped(self) -> None:
        raw = json.dumps({"reasoning": "r", "reply_text": "OK.", "emoji": 42})
        r = _parse_comment_response(raw)
        assert r.emoji is None

    def test_non_string_change_request_dropped(self) -> None:
        raw = json.dumps({"reasoning": "r", "reply_text": "OK.", "change_request": 42})
        r = _parse_comment_response(raw)
        assert r.change_request is None

    def test_valid_insights_parsed(self) -> None:
        raw = _make_raw(
            insights=[{"title": "Good catch", "hook": "Rob prefers X.", "why": "Y."}]
        )
        r = _parse_comment_response(raw)
        assert len(r.insights) == 1
        assert r.insights[0] == Insight(
            title="Good catch", hook="Rob prefers X.", why="Y."
        )

    def test_multiple_insights_parsed(self) -> None:
        raw = _make_raw(
            insights=[
                {"title": "A", "hook": "H1", "why": "W1"},
                {"title": "B", "hook": "H2", "why": "W2"},
            ]
        )
        r = _parse_comment_response(raw)
        assert len(r.insights) == 2
        assert r.insights[0].title == "A"
        assert r.insights[1].title == "B"

    def test_missing_insights_defaults_to_empty(self) -> None:
        raw = json.dumps({"reasoning": "r", "reply_text": "OK."})
        r = _parse_comment_response(raw)
        assert r.insights == []

    def test_null_insights_defaults_to_empty(self) -> None:
        raw = json.dumps({"reasoning": "r", "reply_text": "OK.", "insights": None})
        r = _parse_comment_response(raw)
        assert r.insights == []

    def test_non_list_insights_defaults_to_empty(self) -> None:
        raw = json.dumps({"reasoning": "r", "reply_text": "OK.", "insights": "bad"})
        r = _parse_comment_response(raw)
        assert r.insights == []

    def test_malformed_insight_entry_dropped(self) -> None:
        raw = _make_raw(
            insights=[
                {"title": "", "hook": "H", "why": "W"},  # empty title
                {"title": "Good one", "hook": "H", "why": "W"},
            ]
        )
        r = _parse_comment_response(raw)
        assert len(r.insights) == 1
        assert r.insights[0].title == "Good one"

    def test_non_dict_insight_entry_dropped(self) -> None:
        raw = json.dumps(
            {"reasoning": "r", "reply_text": "OK.", "insights": ["not a dict"]}
        )
        r = _parse_comment_response(raw)
        assert r.insights == []


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

    def test_success_with_emoji_and_change_request(self) -> None:
        raw = _make_raw(
            reply_text="Here's what I think.",
            emoji="heart",
            change_request="Drop the parser refactor",
        )
        agent = _make_agent(raw)
        prompts = _make_prompts()

        result = call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        assert result.emoji == "heart"
        assert result.change_request == "Drop the parser refactor"

    def test_default_context_is_none(self) -> None:
        agent = _make_agent(_make_raw())
        prompts = _make_prompts()

        call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        call_kwargs = prompts.synthesis_prompt.call_args
        assert call_kwargs[1].get("context") is None

    def test_max_retries_constant_is_three(self) -> None:
        assert MAX_RETRIES == 3


# ---------------------------------------------------------------------------
# call_failure_explanation
# ---------------------------------------------------------------------------


def _make_failure_prompts(failure_text: str = "fallback prompt") -> MagicMock:
    prompts = MagicMock()
    prompts.synthesis_failure_explanation_prompt.return_value = failure_text
    return prompts


class TestCallFailureExplanation:
    def test_success_on_first_attempt(self) -> None:
        agent = _make_agent("Sorry, I couldn't reply — please rephrase.")
        prompts = _make_failure_prompts("explain failure")

        result = call_failure_explanation(
            "please fix this", agent=agent, prompts=prompts
        )

        assert result.reply_text == "Sorry, I couldn't reply — please rephrase."
        assert result.emoji is None
        assert result.change_request is None
        assert result.insights == []
        # Prompt builder was given the original comment so the LLM can
        # reference it in the explanation.
        prompts.synthesis_failure_explanation_prompt.assert_called_once_with(
            "please fix this"
        )
        assert agent.run_turn.call_count == 1

    def test_strips_whitespace_from_reply(self) -> None:
        agent = _make_agent("   Plain reply.\n\n")
        prompts = _make_failure_prompts()

        result = call_failure_explanation("comment", agent=agent, prompts=prompts)

        assert result.reply_text == "Plain reply."

    def test_retries_with_nudge_on_empty_response(self) -> None:
        # First attempt empty, second succeeds — exercises the retry-with-nudge
        # loop AND the ``if attempt > 0`` success-after-retry log branch.
        agent = _make_agent(["", "Eventually a real reply."])
        prompts = _make_failure_prompts("base")

        result = call_failure_explanation("comment", agent=agent, prompts=prompts)

        assert result.reply_text == "Eventually a real reply."
        assert agent.run_turn.call_count == 2
        # The retry attempt's prompt has the nudge suffix appended.
        first_call_arg = agent.run_turn.call_args_list[0][0][0]
        second_call_arg = agent.run_turn.call_args_list[1][0][0]
        assert first_call_arg == "base"
        assert second_call_arg.startswith("base")
        assert second_call_arg != first_call_arg  # nudge was appended

    def test_raises_synthesis_exhausted_when_all_attempts_empty(self) -> None:
        agent = _make_agent([""] * MAX_RETRIES)
        prompts = _make_failure_prompts()

        with pytest.raises(SynthesisExhaustedError, match="failure-explanation"):
            call_failure_explanation("comment", agent=agent, prompts=prompts)

        assert agent.run_turn.call_count == MAX_RETRIES

    def test_handles_none_response_as_empty(self) -> None:
        # Defensive: agent.run_turn returning None should be treated as empty,
        # not crash on .strip().
        agent = _make_agent([None, "real reply"])  # type: ignore[list-item]
        prompts = _make_failure_prompts()

        result = call_failure_explanation("comment", agent=agent, prompts=prompts)

        assert result.reply_text == "real reply"
