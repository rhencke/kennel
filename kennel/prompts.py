"""Prompt builders — pure functions and a DI class that assembles text for Claude."""

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
    if ctx.get("sibling_threads"):
        thread_parts: list[str] = []
        for thread in ctx["sibling_threads"]:
            path = thread.get("path", "")
            line = thread.get("line")
            header = f"{path}:{line}" if line else path
            comment_lines = [
                f"  {c.get('author', '')}: {c.get('body', '')}"
                for c in thread.get("comments", [])
            ]
            thread_parts.append(header + "\n" + "\n".join(comment_lines))
        parts.append("Sibling threads:\n" + "\n\n".join(thread_parts))
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
    issue_url: str | None = None,
) -> str:
    """Build the instruction text for a review-comment reply.

    Used by ``reply_to_comment`` in events.py.  Returns a plain instruction
    string (no persona wrapper) so the caller can compose it with
    :meth:`Prompts.persona_wrap`.
    """
    ctx = reply_context_block(context, comment_body, title)
    if category in ("ACT", "DO"):
        return (
            f"Write a short GitHub PR reply to this comment. Acknowledge what they're asking for "
            f"and briefly explain your approach. "
            f"Do NOT promise to open issues or do anything outside of code changes in this PR.\n\n{ctx}"
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
        issue_line = (
            f"An issue has been opened to track this: {issue_url}"
            if issue_url
            else "An issue will be opened to track this"
        )
        return (
            f"Write a short GitHub PR reply acknowledging this suggestion but explaining it's "
            f"out of scope for this PR. "
            f"{issue_line} — mention it in your reply.\n\n{ctx}"
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
    issue_url: str | None = None,
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
        return (
            f"Write a short GitHub PR reply acknowledging and explaining your approach. "
            f"Do NOT promise to open issues or do anything outside of code changes in this PR.\n\n{context_str}"
        )
    if category == "ASK":
        return f"Write a short GitHub PR reply asking a clarifying question.\n\n{context_str}"
    if category == "ANSWER":
        return f"Write a short GitHub PR reply directly answering the question.\n\nQuestion: {comment_body}"
    if category == "DEFER":
        issue_line = (
            f"An issue has been opened to track this: {issue_url}"
            if issue_url
            else "An issue will be opened to track this"
        )
        return (
            f"Write a short GitHub PR reply acknowledging this suggestion but explaining it's "
            f"out of scope for this PR. "
            f"{issue_line} — mention it in your reply.\n\n{context_str}"
        )
    if category == "DUMP":
        return f"Write a short polite decline.\n\n{context_str}"
    return f"Write a short GitHub PR reply.\n\n{context_str}"


# ── Prompts DI class ──────────────────────────────────────────────────────────


class Prompts:
    """Persona-aware prompt builder.

    Accepts a ``persona`` string via the constructor so callers need only read
    the persona file once and inject it — rather than re-reading it inside
    every prompt function.  Follows the dependency injection pattern described
    in CLAUDE.md.

    Stateless helpers that do not depend on the persona remain as module-level
    functions above (e.g. :func:`triage_prompt`, :func:`reply_instruction`).

    Usage::

        p = Prompts(persona)
        prompt = p.persona_wrap(instruction)
        prompt = p.react_prompt(comment_body)
        prompt = p.pickup_comment_prompt(issue_title)
        prompt = p.status_prompt(what)
    """

    def __init__(self, persona: str) -> None:
        self.persona = persona

    def persona_wrap(self, instruction: str) -> str:
        """Wrap an instruction with the Fido persona and output constraint.

        The result is ready to pass as the ``-p`` argument to ``claude --print``.
        """
        return (
            f"{self.persona}\n\n{instruction}\n\n"
            "Output only the comment text, no quotes, no explanation. Keep it brief."
        )

    def pickup_comment_prompt(self, issue_title: str) -> str:
        """Build the prompt for generating a Fido-flavoured pickup comment on an issue.

        The plain text ``"Picking up issue: <title>"`` is rewritten in character
        by Claude using the stored persona.
        """
        plain = f"Picking up issue: {issue_title}"
        return (
            f"{self.persona}\n\n"
            "Rewrite the following GitHub issue comment in character as Fido. "
            "Keep it to 1-2 sentences. "
            "Output only the comment text, no quotes, no explanation.\n\n"
            f"{plain}"
        )

    def status_prompt(self, what: str) -> str:
        """Build the user prompt for GitHub status generation."""
        return f"{self.persona}\n\nWhat you're doing right now: {what}"

    def status_system_prompt(self) -> str:
        """Return the system prompt for GitHub status generation."""
        return (
            "You are writing your GitHub profile status as Fido the dog. "
            "Output exactly two lines. "
            "Line 1: a single emoji for the status icon. "
            "Line 2: the status text (under 80 chars, no quotes, no preamble)."
        )

    def react_prompt(self, comment_body: str) -> str:
        """Build the reaction-decision prompt for Fido.

        Asks the model whether to react to *comment_body* and which emoji to use.
        """
        return (
            f"{self.persona}\n\n"
            f"You just saw this comment on a PR:\n\n{comment_body}\n\n"
            "Would you react to this with a GitHub emoji reaction? Not every comment needs one — "
            "use your dog instincts. Pick from: 👍 (+1), 👎 (-1), 😄 (laugh), 😕 (confused), "
            "❤️ (heart), 🎉 (hooray), 🚀 (rocket), 👀 (eyes). "
            "Reply with JUST the reaction keyword (e.g. heart, rocket, eyes). "
            "If you wouldn't react, reply NONE."
        )
