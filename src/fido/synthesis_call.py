"""Synthesis LLM call — unified comment-handling turn.

Wraps the two-prompt synthesis exchange into a single typed call that
returns a :class:`~fido.synthesis.CommentResponse`.  Retries on
malformed JSON up to :data:`MAX_RETRIES` times with a stricter
instruction appended, then raises :class:`SynthesisExhaustedError`
(fail-closed per Constraint B: reply text is always required, never
defaulted).
"""

import json
import logging
from typing import Any

from fido.prompts import Prompts
from fido.provider import ProviderAgent
from fido.synthesis import (
    AddReaction,
    CommentResponse,
    NoOp,
    RescopeIntent,
    SynthesisAction,
    validate_reaction,
)
from fido.types import ActiveIssue, ActivePR

log = logging.getLogger(__name__)

#: Maximum number of synthesis LLM attempts before raising.
MAX_RETRIES: int = 3

_RETRY_SUFFIX = (
    "\n\n---\n"
    "Your previous response was not valid JSON matching the required schema.  "
    "Respond with ONLY a JSON object — no preamble, no trailing text, no markdown "
    "code fences.  The reply_text field must be a non-empty string."
)


class SynthesisExhaustedError(Exception):
    """All synthesis retries exhausted without a valid :class:`~fido.synthesis.CommentResponse`.

    Signals a Constraint B violation: the synthesis call never defaults or
    returns an empty reply — it either succeeds or fails loudly.
    """


def _extract_json_candidates(raw: str) -> tuple[str, ...]:
    """Return candidate strings to attempt JSON parsing against.

    Tries the raw output (stripped) first, then the largest ``{…}``
    span from the first ``{`` to the last ``}``.  This handles the two
    most common model failure modes: leading/trailing whitespace and
    preamble or trailing explanation text around the JSON object.
    """
    stripped = raw.strip()
    candidates: list[str] = [stripped]
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        span = raw[start : end + 1]
        if span != stripped:
            candidates.append(span)
    return tuple(candidates)


def _parse_action(obj: dict[str, Any]) -> SynthesisAction | None:
    """Parse one action dict into a :class:`~fido.synthesis.SynthesisAction`.

    Returns ``None`` for unknown or invalid entries — individual bad
    actions are skipped with a warning rather than triggering a retry.
    """
    action_type = obj.get("type")
    if action_type == "add_reaction":
        emoji = obj.get("emoji", "")
        try:
            return AddReaction(emoji=validate_reaction(emoji))
        except ValueError:
            log.warning("synthesis: invalid reaction shortcode %r — skipping", emoji)
            return None
    if action_type == "rescope_intent":
        description = obj.get("description", "")
        try:
            return RescopeIntent(description=description)
        except ValueError:
            log.warning("synthesis: empty rescope_intent description — skipping")
            return None
    if action_type == "no_op":
        return NoOp()
    log.warning("synthesis: unknown action type %r — skipping", action_type)
    return None


def _parse_comment_response(raw: str) -> CommentResponse:
    """Parse *raw* model output into a :class:`~fido.synthesis.CommentResponse`.

    Tries each candidate from :func:`_extract_json_candidates` in order.
    Raises :exc:`ValueError` if none yields a valid ``CommentResponse``;
    the caller (:func:`call_synthesis`) catches this and retries.
    """
    last_error: Exception | None = None
    for candidate in _extract_json_candidates(raw):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue

        if not isinstance(obj, dict):
            last_error = ValueError(f"expected JSON object, got {type(obj).__name__}")
            continue

        reasoning = obj.get("reasoning", "")
        reply_text = obj.get("reply_text", "")
        actions_raw = obj.get("actions", [])

        if not isinstance(reply_text, str) or not reply_text.strip():
            last_error = ValueError(
                f"reply_text absent or empty (Constraint B): {reply_text!r}"
            )
            continue

        actions: list[SynthesisAction] = []
        if isinstance(actions_raw, list):
            for item in actions_raw:
                if isinstance(item, dict):
                    action = _parse_action(item)
                    if action is not None:
                        actions.append(action)

        try:
            return CommentResponse(
                reasoning=str(reasoning),
                reply_text=reply_text,
                actions=tuple(actions),
            )
        except ValueError as exc:
            last_error = exc
            continue

    raise ValueError(
        f"failed to parse CommentResponse from model output: {last_error!r}"
        f"\nraw output: {raw!r}"
    )


def call_synthesis(
    comment_body: str,
    *,
    is_bot: bool,
    context: dict[str, Any] | None = None,
    issue: ActiveIssue | None = None,
    pr: ActivePR | None = None,
    agent: ProviderAgent,
    prompts: Prompts,
) -> CommentResponse:
    """Run the unified comment-handling synthesis turn.

    Makes up to :data:`MAX_RETRIES` LLM calls.  The first attempt uses
    the base prompt; each subsequent attempt appends a stricter JSON-output
    instruction.  Raises :class:`SynthesisExhaustedError` if all attempts
    fail (Constraint B: reply text is always required, never silently
    defaulted).

    Parameters
    ----------
    comment_body:
        The text of the PR comment to respond to.
    is_bot:
        Whether the comment came from an automated tool (adjusts voice
        guidance in the prompt).
    context:
        Optional triage context dict (e.g. ``{"pr_title": ...}``) passed
        to the prompt builder.
    issue:
        Active issue, injected into the system prompt for ground-truth
        context.
    pr:
        Active PR, injected into the system prompt alongside *issue*.
    agent:
        LLM agent with a ``run_turn(content, *, system_prompt)`` method.
    prompts:
        Prompt builder (``Prompts`` instance).
    """
    system_prompt = prompts.synthesis_system_prompt(issue=issue, pr=pr)
    base_user_prompt = prompts.synthesis_prompt(
        comment_body, is_bot=is_bot, context=context
    )

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        user_prompt = (
            base_user_prompt if attempt == 0 else base_user_prompt + _RETRY_SUFFIX
        )
        raw = agent.run_turn(user_prompt, system_prompt=system_prompt)
        log.debug(
            "synthesis attempt %d/%d raw output: %r", attempt + 1, MAX_RETRIES, raw
        )

        try:
            response = _parse_comment_response(raw)
        except ValueError as exc:
            last_error = exc
            log.warning(
                "synthesis attempt %d/%d parse failure — %s",
                attempt + 1,
                MAX_RETRIES,
                exc,
            )
            continue

        if attempt > 0:
            log.info("synthesis: succeeded on attempt %d/%d", attempt + 1, MAX_RETRIES)
        return response

    raise SynthesisExhaustedError(
        f"synthesis exhausted {MAX_RETRIES} retries without a valid CommentResponse "
        f"(Constraint B violation) — last error: {last_error}"
    )
