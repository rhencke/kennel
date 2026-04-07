"""Unit tests for kennel/prompts.py — all prompt-building functions."""

from __future__ import annotations

import pytest

from kennel.prompts import (
    issue_reply_instruction,
    persona_wrap,
    react_prompt,
    reply_context_block,
    reply_instruction,
    status_prompt,
    status_system_prompt,
    triage_categories,
    triage_context_block,
    triage_prompt,
)

# ── triage_categories ─────────────────────────────────────────────────────────


class TestTriageCategories:
    def test_human_categories(self) -> None:
        result = triage_categories(is_bot=False)
        assert "ACT" in result
        assert "ASK" in result
        assert "ANSWER" in result
        assert "DO" not in result
        assert "DEFER" not in result

    def test_bot_categories(self) -> None:
        result = triage_categories(is_bot=True)
        assert "DO" in result
        assert "DEFER" in result
        assert "DUMP" in result
        assert "ACT" not in result


# ── triage_context_block ──────────────────────────────────────────────────────


class TestTriageContextBlock:
    def test_empty_context(self) -> None:
        assert triage_context_block(None) == ""
        assert triage_context_block({}) == ""

    def test_pr_title_only(self) -> None:
        result = triage_context_block({"pr_title": "Fix bug"})
        assert "PR: Fix bug" in result

    def test_file_only(self) -> None:
        result = triage_context_block({"file": "src/foo.py"})
        assert "File: src/foo.py" in result

    def test_diff_hunk_only(self) -> None:
        result = triage_context_block({"diff_hunk": "@@ -1,2 +1,3 @@"})
        assert "Diff:" in result
        assert "@@ -1,2 +1,3 @@" in result

    def test_all_fields(self) -> None:
        result = triage_context_block(
            {
                "pr_title": "Refactor",
                "file": "app.py",
                "diff_hunk": "- old\n+ new",
            }
        )
        assert "PR: Refactor" in result
        assert "File: app.py" in result
        assert "Diff:" in result
        assert "- old\n+ new" in result

    def test_ignores_unknown_keys(self) -> None:
        result = triage_context_block({"unknown_key": "value", "pr_title": "hi"})
        assert "unknown_key" not in result
        assert "PR: hi" in result


# ── triage_prompt ─────────────────────────────────────────────────────────────


class TestTriagePrompt:
    def test_includes_comment(self) -> None:
        result = triage_prompt("please fix the bug", is_bot=False)
        assert "please fix the bug" in result

    def test_includes_categories(self) -> None:
        result = triage_prompt("fix this", is_bot=False)
        assert "ACT" in result

    def test_includes_bot_categories(self) -> None:
        result = triage_prompt("suggestion", is_bot=True)
        assert "DO" in result
        assert "DEFER" in result

    def test_includes_context(self) -> None:
        result = triage_prompt("comment", is_bot=False, context={"pr_title": "My PR"})
        assert "PR: My PR" in result

    def test_includes_example(self) -> None:
        result = triage_prompt("x", is_bot=False)
        assert "Example:" in result

    def test_no_context(self) -> None:
        # Prompt with empty context still works — just has an empty ctx_str
        result = triage_prompt("hello", is_bot=False, context=None)
        assert "hello" in result


# ── reply_context_block ───────────────────────────────────────────────────────


class TestReplyContextBlock:
    def test_always_includes_comment_and_plan(self) -> None:
        result = reply_context_block(None, "the comment", "my plan")
        assert "Comment: the comment" in result
        assert "Your plan: my plan" in result

    def test_pr_title(self) -> None:
        result = reply_context_block({"pr_title": "Big change"}, "c", "p")
        assert "PR: Big change" in result

    def test_file_without_line(self) -> None:
        result = reply_context_block({"file": "foo.py"}, "c", "p")
        assert "File: foo.py" in result
        assert "Line:" not in result

    def test_file_with_line(self) -> None:
        result = reply_context_block({"file": "foo.py", "line": 42}, "c", "p")
        assert "File: foo.py" in result
        assert "Line: 42" in result

    def test_diff_hunk(self) -> None:
        result = reply_context_block({"diff_hunk": "+ new line"}, "c", "p")
        assert "Diff:" in result
        assert "```" in result
        assert "+ new line" in result

    def test_empty_context(self) -> None:
        result = reply_context_block({}, "c", "p")
        assert "Comment: c" in result
        assert "Your plan: p" in result


# ── reply_instruction ─────────────────────────────────────────────────────────


class TestReplyInstruction:
    @pytest.mark.parametrize("category", ["ACT", "DO"])
    def test_act_do_acknowledges(self, category: str) -> None:
        result = reply_instruction(category, "fix this", "will fix", {})
        assert "Acknowledge" in result or "acknowledge" in result
        assert "approach" in result

    def test_ask_asks_question(self) -> None:
        result = reply_instruction("ASK", "unclear", "need info", {})
        assert "clarifying question" in result

    def test_answer_no_code_changes(self) -> None:
        result = reply_instruction("ANSWER", "what is X?", "explain X", {})
        assert "Do NOT say you'll make code changes" in result
        assert "Question: what is X?" in result

    def test_defer_out_of_scope(self) -> None:
        result = reply_instruction("DEFER", "big refactor", "defer", {})
        assert "out of scope" in result

    def test_dump_politely_declines(self) -> None:
        result = reply_instruction("DUMP", "bad idea", "decline", {})
        assert "politely declining" in result or "politely" in result

    def test_unknown_category_fallback(self) -> None:
        result = reply_instruction("UNKNOWN", "comment", "title", {})
        assert "Write a short GitHub PR reply" in result
        assert "Comment: comment" in result

    def test_passes_context_to_act(self) -> None:
        result = reply_instruction("ACT", "fix it", "patch", {"pr_title": "Bugfix PR"})
        assert "PR: Bugfix PR" in result


# ── issue_reply_instruction ───────────────────────────────────────────────────


class TestIssueReplyInstruction:
    @pytest.mark.parametrize("category", ["ACT", "DO"])
    def test_act_do_acknowledging(self, category: str) -> None:
        result = issue_reply_instruction(category, "fix it", "will fix", {})
        assert "acknowledging" in result

    def test_ask_clarifying(self) -> None:
        result = issue_reply_instruction("ASK", "unclear", "need more info", {})
        assert "clarifying question" in result

    def test_answer_direct(self) -> None:
        result = issue_reply_instruction("ANSWER", "what is X?", "explain", {})
        assert "Question: what is X?" in result

    def test_dump_decline(self) -> None:
        result = issue_reply_instruction("DUMP", "bad idea", "decline", {})
        assert "decline" in result

    def test_unknown_fallback(self) -> None:
        result = issue_reply_instruction("MYSTERY", "hello", "hmm", {})
        assert "short GitHub PR reply" in result

    def test_includes_pr_title_in_context(self) -> None:
        result = issue_reply_instruction("ACT", "fix it", "fix", {"pr_title": "My PR"})
        assert "PR: My PR" in result

    def test_no_context(self) -> None:
        result = issue_reply_instruction("ACT", "do something", "will do")
        assert "Comment: do something" in result


# ── persona_wrap ──────────────────────────────────────────────────────────────


class TestPersonaWrap:
    def test_includes_persona(self) -> None:
        result = persona_wrap("I am Fido.", "Write a reply.")
        assert "I am Fido." in result

    def test_includes_instruction(self) -> None:
        result = persona_wrap("persona", "do the thing")
        assert "do the thing" in result

    def test_includes_output_constraint(self) -> None:
        result = persona_wrap("persona", "instruction")
        assert "Output only the comment text" in result
        assert "no quotes" in result

    def test_empty_persona(self) -> None:
        result = persona_wrap("", "instruct")
        assert "instruct" in result
        assert "Output only" in result


# ── react_prompt ──────────────────────────────────────────────────────────────


class TestReactPrompt:
    def test_includes_persona(self) -> None:
        result = react_prompt("I am Fido.", "great work!")
        assert "I am Fido." in result

    def test_includes_comment(self) -> None:
        result = react_prompt("persona", "looks good!")
        assert "looks good!" in result

    def test_includes_emoji_options(self) -> None:
        result = react_prompt("persona", "comment")
        assert "rocket" in result
        assert "heart" in result

    def test_includes_none_option(self) -> None:
        result = react_prompt("persona", "comment")
        assert "NONE" in result

    def test_empty_persona(self) -> None:
        result = react_prompt("", "hi")
        assert "hi" in result
        assert "emoji" in result


# ── status_system_prompt ──────────────────────────────────────────────────────


class TestStatusSystemPrompt:
    def test_returns_string(self) -> None:
        result = status_system_prompt()
        assert isinstance(result, str)

    def test_mentions_two_lines(self) -> None:
        result = status_system_prompt()
        assert "two lines" in result

    def test_mentions_emoji(self) -> None:
        result = status_system_prompt()
        assert "emoji" in result

    def test_mentions_fido(self) -> None:
        result = status_system_prompt()
        assert "Fido" in result


# ── status_prompt ─────────────────────────────────────────────────────────────


class TestStatusPrompt:
    def test_includes_persona(self) -> None:
        result = status_prompt("I am Fido.", "writing tests")
        assert "I am Fido." in result

    def test_includes_what(self) -> None:
        result = status_prompt("persona", "reviewing PRs")
        assert "reviewing PRs" in result

    def test_what_is_framed_as_doing(self) -> None:
        result = status_prompt("persona", "fixing a bug")
        assert "What you're doing right now" in result
        assert "fixing a bug" in result

    def test_empty_persona(self) -> None:
        result = status_prompt("", "napping")
        assert "napping" in result
