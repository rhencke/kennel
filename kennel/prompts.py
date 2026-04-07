"""Prompt builders — pure functions that assemble text for Claude."""

from __future__ import annotations

from typing import Any

# ── Triage ────────────────────────────────────────────────────────────────────


def triage_categories(is_bot: bool) -> str:
    """Return the category list string for a triage prompt."""
    if is_bot:
        return "DO (worth implementing), DEFER (out of scope), DUMP (not applicable)"
    return (
        "ACT (code change needed), ASK (unclear what code change is needed), "
        "ANSWER (question, casual/playful comment, or anything that isn't a code change request"
        " — just respond naturally)"
    )


def triage_context_block(context: dict[str, Any] | None) -> str:
    """Build the PR/file/diff context block from a context dict."""
    ctx = context or {}
    parts: list[str] = []
    if ctx.get("pr_title"):
        parts.append(f"PR: {ctx['pr_title']}")
    if ctx.get("file"):
        parts.append(f"File: {ctx['file']}")
    if ctx.get("diff_hunk"):
        parts.append(f"Diff:\n{ctx['diff_hunk']}")
    return "\n".join(parts)


def triage_prompt(
    comment_body: str,
    is_bot: bool,
    context: dict[str, Any] | None = None,
) -> str:
    """Build a triage prompt for Haiku/Opus.

    Returns a prompt that asks the model to classify the comment and return a
    category + short task title in the form ``CATEGORY: title``.
    """
    categories = triage_categories(is_bot)
    ctx_str = triage_context_block(context)
    return (
        f"Triage this PR comment into exactly one category: {categories}\n\n"
        f"{ctx_str}\n\nComment: {comment_body}\n\n"
        "Reply with ONLY the category word (e.g. ACT or DEFER), then a colon, then a short task title. "
        "Example: ACT: add unit tests for parser"
    )


# ── Reply instructions ────────────────────────────────────────────────────────


def reply_context_block(
    context: dict[str, Any] | None,
    comment: str,
    title: str,
) -> str:
    """Build the rich context block used inside a reply instruction."""
    ctx = context or {}
    parts: list[str] = []
    if ctx.get("pr_title"):
        parts.append(f"PR: {ctx['pr_title']}")
    if ctx.get("file"):
        parts.append(f"File: {ctx['file']}")
        if ctx.get("line"):
            parts.append(f"Line: {ctx['line']}")
    if ctx.get("diff_hunk"):
        parts.append(f"Diff:\n```\n{ctx['diff_hunk']}\n```")
    parts.append(f"Comment: {comment}")
    parts.append(f"Your plan: {title}")
    return "\n\n".join(parts)


def reply_instruction(
    category: str,
    comment_body: str,
    title: str,
    context: dict[str, Any] | None = None,
) -> str:
    """Build the instruction text for a review-comment reply.

    Used by ``reply_to_comment`` in events.py.  Returns a plain instruction
    string (no persona wrapper) so the caller can compose it with
    :func:`persona_wrap`.
    """
    ctx = reply_context_block(context, comment_body, title)
    if category in ("ACT", "DO"):
        return (
            f"Write a short GitHub PR reply to this comment. Acknowledge what they're asking for "
            f"and briefly explain your approach.\n\n{ctx}"
        )
    if category == "ASK":
        return (
            f"Write a short GitHub PR reply asking a focused clarifying question. "
            f"You need more information before you can act.\n\n{ctx}"
        )
    if category == "ANSWER":
        return (
            f"Write a short GitHub PR reply directly answering this question. "
            f"Be helpful and specific. Do NOT say you'll make code changes.\n\nQuestion: {comment_body}"
        )
    if category == "DEFER":
        return (
            f"Write a short GitHub PR reply acknowledging this suggestion but explaining it's "
            f"out of scope for this PR.\n\n{ctx}"
        )
    if category == "DUMP":
        return (
            f"Write a short GitHub PR reply politely declining this suggestion and briefly "
            f"explaining why it's not applicable.\n\n{ctx}"
        )
    return f"Write a short GitHub PR reply to this comment.\n\n{ctx}"


def issue_reply_instruction(
    category: str,
    comment_body: str,
    title: str,
    context: dict[str, Any] | None = None,
) -> str:
    """Build the instruction text for a top-level issue/PR comment reply.

    Used by ``reply_to_issue_comment`` in events.py.
    """
    ctx = context or {}
    parts: list[str] = []
    if ctx.get("pr_title"):
        parts.append(f"PR: {ctx['pr_title']}")
    parts.append(f"Comment: {comment_body}")
    parts.append(f"Your plan: {title}")
    context_str = "\n\n".join(parts)

    if category in ("ACT", "DO"):
        return f"Write a short GitHub PR reply acknowledging and explaining your approach.\n\n{context_str}"
    if category == "ASK":
        return f"Write a short GitHub PR reply asking a clarifying question.\n\n{context_str}"
    if category == "ANSWER":
        return f"Write a short GitHub PR reply directly answering the question.\n\nQuestion: {comment_body}"
    if category == "DUMP":
        return f"Write a short polite decline.\n\n{context_str}"
    return f"Write a short GitHub PR reply.\n\n{context_str}"


# ── Persona wrap ──────────────────────────────────────────────────────────────


def persona_wrap(persona: str, instruction: str) -> str:
    """Wrap an instruction with the Fido persona and output constraint.

    The result is ready to pass as the ``-p`` argument to ``claude --print``.
    """
    return (
        f"{persona}\n\n{instruction}\n\n"
        "Output only the comment text, no quotes, no explanation. Keep it brief."
    )


# ── Reaction ──────────────────────────────────────────────────────────────────


def status_system_prompt() -> str:
    """Return the system prompt for GitHub status generation."""
    return (
        "You are writing your GitHub profile status as Fido the dog. "
        "Output exactly two lines. "
        "Line 1: a single emoji for the status icon. "
        "Line 2: the status text (under 80 chars, no quotes, no preamble)."
    )


def status_prompt(persona: str, what: str) -> str:
    """Build the user prompt for GitHub status generation."""
    return f"{persona}\n\nWhat you're doing right now: {what}"


def react_prompt(persona: str, comment_body: str) -> str:
    """Build the reaction-decision prompt for Fido.

    Asks the model whether to react to *comment_body* and which emoji to use.
    """
    return (
        f"{persona}\n\n"
        f"You just saw this comment on a PR:\n\n{comment_body}\n\n"
        "Would you react to this with a GitHub emoji reaction? Not every comment needs one — "
        "use your dog instincts. Pick from: 👍 (+1), 👎 (-1), 😄 (laugh), 😕 (confused), "
        "❤️ (heart), 🎉 (hooray), 🚀 (rocket), 👀 (eyes). "
        "Reply with JUST the reaction keyword (e.g. heart, rocket, eyes). "
        "If you wouldn't react, reply NONE."
    )
