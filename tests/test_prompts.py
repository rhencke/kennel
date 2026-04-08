"""Unit tests for kennel/prompts.py — prompt-building functions and Prompts class."""

from __future__ import annotations

import pytest

from kennel.prompts import (
    Prompts,
    issue_reply_instruction,
    reply_context_block,
    reply_instruction,
    triage_categories,
    triage_context_block,
    triage_prompt,
)

# ── triage_categories ─────────────────────────────────────────────────────────


class TestTriageCategories:
    def test_human_categories(self) -> None:
        result = triage_categories(is_bot=False)
        assert "ACT" in result
        assert "DEFER" in result
        assert "ASK" in result
        assert "ANSWER" in result
        assert "DO" not in result

    def test_bot_categories(self) -> None:
        result = triage_categories(is_bot=True)
        assert "DO" in result
        assert "DEFER" in result
        assert "DUMP" in result
        assert "TASK" not in result
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

    def test_sibling_threads_rendered(self) -> None:
        result = triage_context_block(
            {
                "sibling_threads": [
                    {
                        "path": "src/foo.py",
                        "line": 10,
                        "comments": [
                            {"author": "alice", "body": "why is this here?"},
                            {"author": "fido", "body": "good catch!"},
                        ],
                    }
                ]
            }
        )
        assert "Sibling threads:" in result
        assert "src/foo.py:10" in result
        assert "alice: why is this here?" in result
        assert "fido: good catch!" in result

    def test_sibling_threads_no_line(self) -> None:
        result = triage_context_block(
            {
                "sibling_threads": [
                    {
                        "path": "README.md",
                        "line": None,
                        "comments": [{"author": "bob", "body": "typo"}],
                    }
                ]
            }
        )
        assert "README.md" in result
        assert "bob: typo" in result

    def test_sibling_threads_multiple(self) -> None:
        result = triage_context_block(
            {
                "sibling_threads": [
                    {
                        "path": "a.py",
                        "line": 1,
                        "comments": [{"author": "x", "body": "first"}],
                    },
                    {
                        "path": "b.py",
                        "line": 2,
                        "comments": [{"author": "y", "body": "second"}],
                    },
                ]
            }
        )
        assert "a.py:1" in result
        assert "b.py:2" in result
        assert "x: first" in result
        assert "y: second" in result

    def test_empty_sibling_threads_omitted(self) -> None:
        result = triage_context_block({"sibling_threads": []})
        assert "Sibling threads:" not in result


# ── triage_prompt ─────────────────────────────────────────────────────────────


class TestTriagePrompt:
    def test_includes_comment(self) -> None:
        result = triage_prompt("please fix the bug", is_bot=False)
        assert "please fix the bug" in result

    def test_includes_categories(self) -> None:
        result = triage_prompt("fix this", is_bot=False)
        assert "ACT" in result
        assert "DEFER" in result

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
        assert "Do NOT promise" in result

    @pytest.mark.parametrize("category", ["ACT", "DO"])
    def test_act_do_no_promises_does_not_restrict_task_creation(
        self, category: str
    ) -> None:
        """No-promises constraint governs reply *text* only.

        ACT/DO triage always results in a task being created by server.py —
        the constraint must not say 'create tasks' or it would misrepresent
        what the system actually does.
        """
        result = reply_instruction(category, "please fix this", "fix: edge case", {})
        assert "Do NOT promise" in result
        assert "create tasks" not in result

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

    def test_defer_issue_opened_with_url(self) -> None:
        result = reply_instruction(
            "DEFER",
            "big refactor",
            "defer",
            {},
            issue_url="https://github.com/x/y/issues/1",
        )
        assert "An issue has been opened" in result
        assert "https://github.com/x/y/issues/1" in result

    def test_defer_issue_will_be_opened_without_url(self) -> None:
        result = reply_instruction("DEFER", "big refactor", "defer", {})
        assert "An issue will be opened" in result

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

    @pytest.mark.parametrize("category", ["ACT", "DO"])
    def test_act_do_no_promises(self, category: str) -> None:
        result = issue_reply_instruction(category, "fix it", "will fix", {})
        assert "Do NOT promise to open issues" in result

    @pytest.mark.parametrize("category", ["ACT", "DO"])
    def test_act_do_no_promises_does_not_restrict_task_creation(
        self, category: str
    ) -> None:
        """No-promises constraint governs reply *text* only.

        ACT/DO triage always results in a task being created by server.py —
        the constraint must not say 'create tasks' or it would misrepresent
        what the system actually does.
        """
        result = issue_reply_instruction(
            category, "please fix this", "fix: edge case", {}
        )
        assert "Do NOT promise" in result
        assert "create tasks" not in result

    def test_ask_clarifying(self) -> None:
        result = issue_reply_instruction("ASK", "unclear", "need more info", {})
        assert "clarifying question" in result

    def test_answer_direct(self) -> None:
        result = issue_reply_instruction("ANSWER", "what is X?", "explain", {})
        assert "Question: what is X?" in result

    def test_defer_out_of_scope(self) -> None:
        result = issue_reply_instruction("DEFER", "add feature", "defer", {})
        assert "out of scope" in result

    def test_defer_issue_opened_with_url(self) -> None:
        result = issue_reply_instruction(
            "DEFER",
            "add feature",
            "defer",
            {},
            issue_url="https://github.com/x/y/issues/2",
        )
        assert "An issue has been opened" in result
        assert "https://github.com/x/y/issues/2" in result

    def test_defer_issue_will_be_opened_without_url(self) -> None:
        result = issue_reply_instruction("DEFER", "add feature", "defer", {})
        assert "An issue will be opened" in result

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


# ── Prompts.status_system_prompt ─────────────────────────────────────────────


class TestStatusTextSystemPrompt:
    def test_returns_string(self) -> None:
        result = Prompts("persona").status_text_system_prompt()
        assert isinstance(result, str)

    def test_no_emoji_instruction(self) -> None:
        result = Prompts("persona").status_text_system_prompt()
        assert "no emoji" in result.lower() or "ONLY the status text" in result

    def test_mentions_80_chars(self) -> None:
        result = Prompts("persona").status_text_system_prompt()
        assert "80" in result

    def test_mentions_fido(self) -> None:
        result = Prompts("persona").status_text_system_prompt()
        assert "Fido" in result


class TestStatusEmojiSystemPrompt:
    def test_returns_string(self) -> None:
        result = Prompts("persona").status_emoji_system_prompt()
        assert isinstance(result, str)

    def test_mentions_emoji(self) -> None:
        result = Prompts("persona").status_emoji_system_prompt()
        assert "emoji" in result

    def test_mentions_fido(self) -> None:
        result = Prompts("persona").status_emoji_system_prompt()
        assert "Fido" in result


class TestStatusEmojiPrompt:
    def test_includes_persona(self) -> None:
        result = Prompts("I am Fido.").status_emoji_prompt("working hard")
        assert "I am Fido." in result

    def test_includes_text(self) -> None:
        result = Prompts("persona").status_emoji_prompt("chasing bugs")
        assert "chasing bugs" in result


# ── Prompts class ─────────────────────────────────────────────────────────────


class TestPromptsPersonaWrap:
    def test_includes_persona(self) -> None:
        result = Prompts("I am Fido.").persona_wrap("Write a reply.")
        assert "I am Fido." in result

    def test_includes_instruction(self) -> None:
        result = Prompts("persona").persona_wrap("do the thing")
        assert "do the thing" in result

    def test_includes_output_constraint(self) -> None:
        result = Prompts("persona").persona_wrap("instruction")
        assert "Output only the comment text" in result
        assert "no quotes" in result

    def test_empty_persona(self) -> None:
        result = Prompts("").persona_wrap("instruct")
        assert "instruct" in result
        assert "Output only" in result


class TestPromptsReactPrompt:
    def test_includes_persona(self) -> None:
        result = Prompts("I am Fido.").react_prompt("great work!")
        assert "I am Fido." in result

    def test_includes_comment(self) -> None:
        result = Prompts("persona").react_prompt("looks good!")
        assert "looks good!" in result

    def test_includes_emoji_options(self) -> None:
        result = Prompts("persona").react_prompt("comment")
        assert "rocket" in result
        assert "heart" in result

    def test_includes_none_option(self) -> None:
        result = Prompts("persona").react_prompt("comment")
        assert "NONE" in result

    def test_empty_persona(self) -> None:
        result = Prompts("").react_prompt("hi")
        assert "hi" in result
        assert "emoji" in result


class TestPromptsStatusTextPrompt:
    def test_includes_persona(self) -> None:
        result = Prompts("I am Fido.").status_text_prompt("writing tests")
        assert "I am Fido." in result

    def test_includes_what(self) -> None:
        result = Prompts("persona").status_text_prompt("reviewing PRs")
        assert "reviewing PRs" in result

    def test_what_is_framed_as_doing(self) -> None:
        result = Prompts("persona").status_text_prompt("fixing a bug")
        assert "What you're doing right now" in result
        assert "fixing a bug" in result


class TestPromptsPickupCommentPrompt:
    def test_includes_persona(self) -> None:
        result = Prompts("I am Fido.").pickup_comment_prompt("Fix the thing")
        assert "I am Fido." in result

    def test_includes_issue_title(self) -> None:
        result = Prompts("persona").pickup_comment_prompt("Refactor auth module")
        assert "Refactor auth module" in result

    def test_includes_plain_text(self) -> None:
        result = Prompts("persona").pickup_comment_prompt("Add caching")
        assert "Picking up issue: Add caching" in result

    def test_instructs_fido_character(self) -> None:
        result = Prompts("persona").pickup_comment_prompt("title")
        assert "Fido" in result

    def test_requests_short_output(self) -> None:
        result = Prompts("persona").pickup_comment_prompt("title")
        assert "1-2 sentences" in result

    def test_output_constraint_present(self) -> None:
        result = Prompts("persona").pickup_comment_prompt("title")
        assert "Output only the comment text" in result

    def test_empty_persona(self) -> None:
        result = Prompts("").pickup_comment_prompt("Some issue")
        assert "Picking up issue: Some issue" in result

    def test_returns_string(self) -> None:
        assert isinstance(Prompts("persona").pickup_comment_prompt("title"), str)


class TestPromptsStoresPersona:
    def test_persona_stored(self) -> None:
        p = Prompts("my persona")
        assert p.persona == "my persona"

    def test_different_personas_independent(self) -> None:
        p1 = Prompts("persona A")
        p2 = Prompts("persona B")
        assert "persona A" in p1.status_text_prompt("working")
        assert "persona B" in p2.status_text_prompt("working")
        assert "persona A" not in p2.status_text_prompt("working")
