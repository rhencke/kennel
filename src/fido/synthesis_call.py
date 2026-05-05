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
from fido.provider import READ_ONLY_ALLOWED_TOOLS, ProviderAgent
from fido.synthesis import (
    VALID_REACTIONS,
    CommentResponse,
    Insight,
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


def _extract_json_objects(raw: str) -> list[dict[str, Any]]:
    """Return all JSON objects found in *raw* using a consume loop.

    Scans *raw* for ``{``, attempts :meth:`json.JSONDecoder.raw_decode`
    from that position, advances past the decoded span on success, or
    skips the character and continues on failure.  Returns a list of
    every successfully decoded dict in order of appearance.

    This is more robust than a ``first-{-to-last-}`` span heuristic: it
    handles preamble prose, trailing explanation text, stray braces, and
    nested objects correctly because the decoder itself determines the
    exact end of each JSON document.
    """
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    pos = 0
    while pos < len(raw):
        brace = raw.find("{", pos)
        if brace == -1:
            break
        try:
            obj, end = decoder.raw_decode(raw, brace)
        except json.JSONDecodeError:
            pos = brace + 1
            continue
        if isinstance(obj, dict):
            objects.append(obj)
        pos = end
    return objects


def _parse_comment_response(raw: str) -> CommentResponse:
    """Parse *raw* model output into a :class:`~fido.synthesis.CommentResponse`.

    Extracts all JSON objects from *raw* via :func:`_extract_json_objects`
    and returns the first that validates as a ``CommentResponse``.
    Raises :exc:`ValueError` if none does; the caller
    (:func:`call_synthesis`) catches this and retries.
    """
    last_error: Exception = ValueError("no JSON objects found in model output")
    for obj in _extract_json_objects(raw):
        reasoning = obj.get("reasoning", "")
        reply_text = obj.get("reply_text", "")

        if not isinstance(reply_text, str) or not reply_text.strip():
            last_error = ValueError(
                f"reply_text absent or empty (Constraint B): {reply_text!r}"
            )
            continue

        # Parse optional emoji — invalid shortcodes are warned and dropped.
        emoji_raw = obj.get("emoji")
        emoji: str | None = None
        if isinstance(emoji_raw, str) and emoji_raw:
            if emoji_raw in VALID_REACTIONS:
                emoji = emoji_raw
            else:
                log.warning(
                    "synthesis: invalid reaction shortcode %r — dropping", emoji_raw
                )

        # Parse optional change_request — must be a non-empty string or null.
        change_request_raw = obj.get("change_request")
        change_request: str | None = None
        if isinstance(change_request_raw, str) and change_request_raw.strip():
            change_request = change_request_raw

        # Parse optional insights list — each entry must have title, hook, why.
        insights: list[Insight] = []
        insights_raw = obj.get("insights")
        if isinstance(insights_raw, list):
            for entry in insights_raw:
                if not isinstance(entry, dict):
                    continue
                title = entry.get("title", "")
                hook = entry.get("hook", "")
                why = entry.get("why", "")
                if (
                    isinstance(title, str)
                    and title.strip()
                    and isinstance(hook, str)
                    and hook.strip()
                    and isinstance(why, str)
                    and why.strip()
                ):
                    insights.append(Insight(title=title, hook=hook, why=why))
                else:
                    log.warning("synthesis: dropping malformed insight entry %r", entry)

        try:
            return CommentResponse(
                reasoning=str(reasoning),
                reply_text=reply_text,
                emoji=emoji,
                change_request=change_request,
                insights=insights,
            )
        except ValueError as exc:  # pragma: no cover - defensive
            # All ``CommentResponse`` invariants (non-empty reply_text,
            # valid emoji shortcode, non-empty change_request when set) are
            # pre-checked above, so construction here cannot fail through
            # the normal parser path.  Kept as a defensive catch in case
            # the dataclass adds new validation that the parser hasn't
            # learned about yet.
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
        raw = agent.run_turn(
            user_prompt,
            allowed_tools=READ_ONLY_ALLOWED_TOOLS,
            system_prompt=system_prompt,
        )
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


_FAILURE_EXPLANATION_RETRY_SUFFIX = (
    "\n\n---\n"
    "Your previous response was empty.  Output the reply text now — plain prose only, "
    "no JSON, no markdown fences, no preamble.  At least one full sentence."
)


def call_failure_explanation(
    comment_body: str,
    *,
    agent: ProviderAgent,
    prompts: Prompts,
) -> CommentResponse:
    """Generate a fallback reply when :func:`call_synthesis` exhausted retries.

    Asks the LLM, via the same retry-with-nudge loop as :func:`call_synthesis`,
    to write a short reply acknowledging the failure and asking the commenter
    to rephrase.  Returns a :class:`CommentResponse` with only ``reply_text``
    populated — no emoji, no change_request, no insights — so the executor's
    success-path effects (post reply, clear eyes) handle it identically to a
    normal synthesis result.

    Raises :exc:`SynthesisExhaustedError` if the fallback also exhausts
    retries.  Caller is responsible for any further cleanup.
    """
    user_prompt = prompts.synthesis_failure_explanation_prompt(comment_body)

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        suffix = _FAILURE_EXPLANATION_RETRY_SUFFIX if attempt > 0 else ""
        raw = agent.run_turn(
            user_prompt + suffix,
            allowed_tools=READ_ONLY_ALLOWED_TOOLS,
        )
        log.debug(
            "failure-explanation attempt %d/%d raw output: %r",
            attempt + 1,
            MAX_RETRIES,
            raw,
        )
        text = (raw or "").strip()
        if not text:
            last_error = ValueError("empty reply from failure-explanation turn")
            log.warning(
                "failure-explanation attempt %d/%d returned empty text — retrying",
                attempt + 1,
                MAX_RETRIES,
            )
            continue
        if attempt > 0:
            log.info(
                "failure-explanation: succeeded on attempt %d/%d",
                attempt + 1,
                MAX_RETRIES,
            )
        return CommentResponse(
            reasoning=(
                "(synthesis exhausted retries; this reply was generated by the "
                "fallback failure-explanation turn)"
            ),
            reply_text=text,
        )

    raise SynthesisExhaustedError(
        f"failure-explanation exhausted {MAX_RETRIES} retries without a usable reply "
        f"— last error: {last_error}"
    )
