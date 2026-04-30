"""Unit tests for fido/prompts.py — prompt-building functions and Prompts class."""

import pytest

from fido.prompts import (
    NO_TOOLS_CLAUSE,
    TRIAGE_CLAUSE,
    Prompts,
    reply_context_block,
    triage_categories,
    triage_context_block,
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
        assert "DEFER" not in result
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

    def test_pr_body(self) -> None:
        result = triage_context_block({"pr_body": "Adds caching to the parser."})
        assert "PR description:" in result
        assert "Adds caching to the parser." in result

    def test_empty_pr_body_omitted(self) -> None:
        result = triage_context_block({"pr_body": ""})
        assert "PR description:" not in result

    def test_all_fields(self) -> None:
        result = triage_context_block(
            {
                "pr_title": "Refactor",
                "pr_body": "Refactors the parser.",
                "file": "app.py",
                "diff_hunk": "- old\n+ new",
            }
        )
        assert "PR: Refactor" in result
        assert "PR description:" in result
        assert "Refactors the parser." in result
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

    def test_comment_thread_rendered(self) -> None:
        result = triage_context_block(
            {
                "comment_thread": [
                    {"author": "alice", "body": "fix this please"},
                    {"author": "fido", "body": "done in latest commit"},
                ]
            }
        )
        assert "Comment thread:" in result
        assert "alice: fix this please" in result
        assert "fido: done in latest commit" in result

    def test_empty_comment_thread_omitted(self) -> None:
        result = triage_context_block({"comment_thread": []})
        assert "Comment thread:" not in result

    def test_conversation_rendered(self) -> None:
        result = triage_context_block(
            {"conversation": "\n\nFull conversation:\nalice: hi"}
        )
        assert "Full conversation:" in result
        assert "alice: hi" in result


# ── Prompts.triage_prompt ────────────────────────────────────────────────────


class TestTriagePrompt:
    def test_includes_comment(self) -> None:
        result = Prompts("").triage_prompt("please fix the bug", is_bot=False)
        assert "please fix the bug" in result

    def test_includes_categories(self) -> None:
        result = Prompts("").triage_prompt("fix this", is_bot=False)
        assert "ACT" in result
        assert "DEFER" in result

    def test_includes_bot_categories(self) -> None:
        result = Prompts("").triage_prompt("suggestion", is_bot=True)
        assert "DO" in result
        assert "DEFER" in result

    def test_includes_context(self) -> None:
        result = Prompts("").triage_prompt(
            "comment", is_bot=False, context={"pr_title": "My PR"}
        )
        assert "PR: My PR" in result

    def test_includes_example(self) -> None:
        result = Prompts("").triage_prompt("x", is_bot=False)
        assert "Example" in result

    def test_requires_imperative_action_item_title(self) -> None:
        result = Prompts("").triage_prompt("x", is_bot=False)
        assert "imperative" in result
        assert "verb" in result
        assert "never quote" in result.lower()

    def test_no_context(self) -> None:
        # Prompt with empty context still works — just has an empty ctx_str
        result = Prompts("").triage_prompt("hello", is_bot=False, context=None)
        assert "hello" in result

    def test_with_context(self) -> None:
        p = Prompts("")
        ctx = {"pr_title": "Refactor"}
        result = p.triage_prompt("x", is_bot=True, context=ctx)
        assert "Refactor" in result


class TestFreshSessionRetryPrompt:
    def test_includes_issue_context_details(self) -> None:
        result = Prompts("").fresh_session_retry_prompt(
            task_title="Fix parser",
            task_id="task-1",
            work_dir="/repo",
            pr_number=42,
            branch="fix-parser",
            issue_number=271,
            issue_title="Parser crash",
            issue_body="It blows up on empty input.",
            pr_title="Fix parser crash",
            pr_body="Implements the parser fix.",
        )
        assert "- Issue: #271 Parser crash" in result
        assert "Issue description:\nIt blows up on empty input." in result


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


# ── Prompts.reply_instruction ────────────────────────────────────────────────


class TestReplyInstruction:
    @pytest.mark.parametrize("category", ["ACT", "DO"])
    def test_act_do_acknowledges(self, category: str) -> None:
        result = Prompts("").reply_instruction(category, "fix this", "will fix", {})
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
        result = Prompts("").reply_instruction(
            category, "please fix this", "fix: edge case", {}
        )
        assert "Do NOT promise" in result
        assert "create tasks" not in result

    def test_ask_asks_question(self) -> None:
        result = Prompts("").reply_instruction("ASK", "unclear", "need info", {})
        assert "clarifying question" in result

    def test_answer_no_code_changes(self) -> None:
        result = Prompts("").reply_instruction("ANSWER", "what is X?", "explain X", {})
        assert "Do NOT say you'll make code changes" in result
        assert "Question: what is X?" in result

    def test_defer_out_of_scope(self) -> None:
        result = Prompts("").reply_instruction("DEFER", "big refactor", "defer", {})
        assert "out of scope" in result

    def test_defer_issue_opened_with_url(self) -> None:
        result = Prompts("").reply_instruction(
            "DEFER",
            "big refactor",
            "defer",
            {},
            issue_url="https://github.com/x/y/issues/1",
        )
        assert "An issue has been opened" in result
        assert "https://github.com/x/y/issues/1" in result

    def test_defer_issue_will_be_opened_without_url(self) -> None:
        result = Prompts("").reply_instruction("DEFER", "big refactor", "defer", {})
        assert "An issue will be opened" in result

    def test_dump_politely_declines(self) -> None:
        result = Prompts("").reply_instruction("DUMP", "bad idea", "decline", {})
        assert "politely declining" in result or "politely" in result

    def test_unknown_category_fallback(self) -> None:
        result = Prompts("").reply_instruction("UNKNOWN", "comment", "title", {})
        assert "Write a short GitHub PR reply" in result
        assert "Comment: comment" in result

    def test_passes_context_to_act(self) -> None:
        result = Prompts("").reply_instruction(
            "ACT", "fix it", "patch", {"pr_title": "Bugfix PR"}
        )
        assert "PR: Bugfix PR" in result

    def test_with_issue_url(self) -> None:
        url = "https://github.com/x/y/issues/1"
        result = Prompts("").reply_instruction(
            "DEFER", "big refactor", "defer", {}, issue_url=url
        )
        assert url in result


# ── Prompts.issue_reply_instruction ──────────────────────────────────────────


class TestIssueReplyInstruction:
    @pytest.mark.parametrize("category", ["ACT", "DO"])
    def test_act_do_acknowledging(self, category: str) -> None:
        result = Prompts("").issue_reply_instruction(category, "fix it", "will fix", {})
        assert "acknowledging" in result

    @pytest.mark.parametrize("category", ["ACT", "DO"])
    def test_act_do_no_promises(self, category: str) -> None:
        result = Prompts("").issue_reply_instruction(category, "fix it", "will fix", {})
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
        result = Prompts("").issue_reply_instruction(
            category, "please fix this", "fix: edge case", {}
        )
        assert "Do NOT promise" in result
        assert "create tasks" not in result

    def test_ask_clarifying(self) -> None:
        result = Prompts("").issue_reply_instruction(
            "ASK", "unclear", "need more info", {}
        )
        assert "clarifying question" in result

    def test_answer_direct(self) -> None:
        result = Prompts("").issue_reply_instruction(
            "ANSWER", "what is X?", "explain", {}
        )
        assert "Question: what is X?" in result

    def test_defer_out_of_scope(self) -> None:
        result = Prompts("").issue_reply_instruction(
            "DEFER", "add feature", "defer", {}
        )
        assert "out of scope" in result

    def test_defer_issue_opened_with_url(self) -> None:
        result = Prompts("").issue_reply_instruction(
            "DEFER",
            "add feature",
            "defer",
            {},
            issue_url="https://github.com/x/y/issues/2",
        )
        assert "An issue has been opened" in result
        assert "https://github.com/x/y/issues/2" in result

    def test_defer_issue_will_be_opened_without_url(self) -> None:
        result = Prompts("").issue_reply_instruction(
            "DEFER", "add feature", "defer", {}
        )
        assert "An issue will be opened" in result

    def test_dump_decline(self) -> None:
        result = Prompts("").issue_reply_instruction("DUMP", "bad idea", "decline", {})
        assert "decline" in result

    def test_unknown_fallback(self) -> None:
        result = Prompts("").issue_reply_instruction("MYSTERY", "hello", "hmm", {})
        assert "short GitHub PR reply" in result

    def test_includes_pr_title_in_context(self) -> None:
        result = Prompts("").issue_reply_instruction(
            "ACT", "fix it", "fix", {"pr_title": "My PR"}
        )
        assert "PR: My PR" in result

    def test_no_context(self) -> None:
        result = Prompts("").issue_reply_instruction("ACT", "do something", "will do")
        assert "Comment: do something" in result

    def test_with_issue_url(self) -> None:
        url = "https://github.com/x/y/issues/2"
        result = Prompts("").issue_reply_instruction(
            "DEFER", "feature", "defer", {}, issue_url=url
        )
        assert url in result


# ── Prompts.status_system_prompt ─────────────────────────────────────────────


class TestStatusSystemPrompt:
    def test_returns_string(self) -> None:
        result = Prompts("persona").status_system_prompt()
        assert isinstance(result, str)

    def test_mentions_json(self) -> None:
        result = Prompts("persona").status_system_prompt()
        assert "JSON" in result

    def test_mentions_status_and_emoji_fields(self) -> None:
        result = Prompts("persona").status_system_prompt()
        assert '"status"' in result
        assert '"emoji"' in result

    def test_mentions_80_chars(self) -> None:
        result = Prompts("persona").status_system_prompt()
        assert "80" in result

    def test_mentions_fido(self) -> None:
        result = Prompts("persona").status_system_prompt()
        assert "Fido" in result

    def test_instructs_busy_priority(self) -> None:
        result = Prompts("persona").status_system_prompt()
        assert "busy" in result

    def test_instructs_idle_napping(self) -> None:
        result = Prompts("persona").status_system_prompt()
        assert "idle" in result or "napping" in result.lower()


# ── Prompts class ─────────────────────────────────────────────────────────────


class TestPromptsReplySystemPrompt:
    def test_includes_persona(self) -> None:
        result = Prompts("I am Fido.").reply_system_prompt()
        assert "I am Fido." in result

    def test_prohibits_preamble_phrases(self) -> None:
        result = Prompts("persona").reply_system_prompt()
        assert "Here's" in result or "preamble" in result

    def test_output_only_instruction(self) -> None:
        result = Prompts("persona").reply_system_prompt()
        assert "ONLY" in result

    def test_no_meta_commentary(self) -> None:
        result = Prompts("persona").reply_system_prompt()
        assert "meta-commentary" in result or "Here's the reply" in result


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

    def test_includes_no_tools_clause(self) -> None:
        # react_prompt is pure text — must include NO_TOOLS_CLAUSE (not the broader
        # TRIAGE_CLAUSE) so a comment that looks like a directive doesn't cause
        # Opus to fire Edit/Write calls during what should be a one-shot reaction
        # decision.
        result = Prompts("persona").react_prompt("fix this please")
        assert NO_TOOLS_CLAUSE in result
        assert TRIAGE_CLAUSE not in result


class TestPromptsStatusPrompt:
    def test_includes_persona(self) -> None:
        result = Prompts("I am Fido.").status_prompt(
            [("owner/repo", "writing tests", True)]
        )
        assert "I am Fido." in result

    def test_includes_what(self) -> None:
        result = Prompts("persona").status_prompt(
            [("owner/repo", "reviewing PRs", True)]
        )
        assert "reviewing PRs" in result

    def test_includes_repo_name(self) -> None:
        result = Prompts("persona").status_prompt(
            [("FidoCanCode/home", "fixing a bug", True)]
        )
        assert "FidoCanCode/home" in result

    def test_busy_worker_labeled(self) -> None:
        result = Prompts("persona").status_prompt(
            [("owner/repo", "working hard", True)]
        )
        assert "busy" in result

    def test_idle_worker_labeled(self) -> None:
        result = Prompts("persona").status_prompt([("owner/repo", "napping", False)])
        assert "idle" in result

    def test_multiple_repos_all_present(self) -> None:
        result = Prompts("persona").status_prompt(
            [
                ("a/busy", "Writing code", True),
                ("b/idle", "Napping", False),
            ]
        )
        assert "a/busy" in result
        assert "Writing code" in result
        assert "b/idle" in result
        assert "Napping" in result

    def test_empty_activities(self) -> None:
        result = Prompts("persona").status_prompt([])
        assert isinstance(result, str)
        assert "No active workers" in result


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


class TestPromptsPickupRetryCommentPrompt:
    """Fresh-retry pickup-ack prompt (fix for FidoCanCode/home#802)."""

    def test_includes_persona(self) -> None:
        result = Prompts("I am Fido.").pickup_retry_comment_prompt("t", [10])
        assert "I am Fido." in result

    def test_includes_issue_title_and_prs(self) -> None:
        result = Prompts("p").pickup_retry_comment_prompt("Migrate Gitea", [215, 210])
        assert "Migrate Gitea" in result
        assert "#215" in result
        assert "#210" in result

    def test_instructs_acknowledgement(self) -> None:
        result = Prompts("p").pickup_retry_comment_prompt("t", [7])
        assert "starting genuinely fresh" in result
        assert "closed PR" in result

    def test_requests_fresh_start_commitment(self) -> None:
        result = Prompts("p").pickup_retry_comment_prompt("t", [1])
        assert "not reusing anything" in result

    def test_output_constraint_present(self) -> None:
        result = Prompts("p").pickup_retry_comment_prompt("t", [1])
        assert "Output only the comment text" in result

    def test_single_pr_formatted(self) -> None:
        result = Prompts("p").pickup_retry_comment_prompt("t", [42])
        assert "#42" in result

    def test_returns_string(self) -> None:
        assert isinstance(Prompts("p").pickup_retry_comment_prompt("t", [1]), str)


# ── Prompts.rescope_prompt ───────────────────────────────────────────────────


class TestRescopePrompt:
    def _task(
        self,
        title: str,
        task_id: str = "1",
        task_type: str = "spec",
        status: str = "pending",
        description: str = "",
    ) -> dict:
        return {
            "id": task_id,
            "title": title,
            "type": task_type,
            "status": status,
            "description": description,
        }

    def test_includes_pending_tasks_json(self) -> None:
        tasks = [self._task("Add feature", task_id="1")]
        result = Prompts("").rescope_prompt(tasks, "")
        assert "Add feature" in result
        assert '"id": "1"' in result

    def test_excludes_completed_from_pending_json(self) -> None:
        tasks = [
            self._task("Done task", task_id="1", status="completed"),
            self._task("Todo task", task_id="2"),
        ]
        result = Prompts("").rescope_prompt(tasks, "")
        # Completed appears in the completed block, not the pending JSON
        assert '"id": "2"' in result
        assert '"id": "1"' not in result.split("Pending tasks")[1]

    def test_completed_titles_listed_in_completed_block(self) -> None:
        tasks = [
            self._task("Already done", task_id="1", status="completed"),
            self._task("Still pending", task_id="2"),
        ]
        result = Prompts("").rescope_prompt(tasks, "")
        assert "Already done" in result.split("Pending tasks")[0]

    def test_no_completed_tasks_shows_none(self) -> None:
        tasks = [self._task("Only pending", task_id="1")]
        result = Prompts("").rescope_prompt(tasks, "")
        assert "(none)" in result.split("Pending tasks")[0]

    def test_commit_summary_included(self) -> None:
        tasks = [self._task("Add tests", task_id="1")]
        result = Prompts("").rescope_prompt(tasks, "feat: add parser method")
        assert "feat: add parser method" in result

    def test_empty_commit_summary_shows_none(self) -> None:
        tasks = [self._task("Add tests", task_id="1")]
        result = Prompts("").rescope_prompt(tasks, "")
        assert "(none)" in result

    def test_ci_tasks_must_come_first_rule_stated(self) -> None:
        tasks = [self._task("Fix CI", task_id="1", task_type="ci")]
        result = Prompts("").rescope_prompt(tasks, "")
        assert "ci" in result.lower()
        assert "first" in result

    def test_json_output_format_instructed(self) -> None:
        tasks = [self._task("Do something", task_id="1")]
        result = Prompts("").rescope_prompt(tasks, "")
        assert '{"tasks": [...]}' in result

    def test_preserve_ids_rule_stated(self) -> None:
        tasks = [self._task("Task A", task_id="abc-123")]
        result = Prompts("").rescope_prompt(tasks, "")
        assert "ID" in result or "id" in result
        assert "preserve" in result.lower() or "never change" in result.lower()

    def test_remove_covered_tasks_rule_stated(self) -> None:
        tasks = [self._task("Task A", task_id="1")]
        result = Prompts("").rescope_prompt(tasks, "commit covering this")
        assert "commit" in result.lower() or "covered" in result.lower()

    def test_rewrite_spec_on_thread_conflict_rule_stated(self) -> None:
        tasks = [
            self._task("Old spec title", task_id="1", task_type="spec"),
            self._task("New comment task", task_id="2", task_type="thread"),
        ]
        result = Prompts("").rescope_prompt(tasks, "")
        assert "thread" in result.lower() or "rewrite" in result.lower()

    def test_no_other_text_instruction_present(self) -> None:
        tasks = [self._task("X", task_id="1")]
        result = Prompts("").rescope_prompt(tasks, "")
        assert "No other text" in result

    def test_in_progress_tasks_included_in_pending(self) -> None:
        tasks = [
            self._task("Running task", task_id="1", status="in_progress"),
        ]
        result = Prompts("").rescope_prompt(tasks, "")
        assert '"id": "1"' in result

    def test_description_included_in_task_json(self) -> None:
        tasks = [self._task("X", task_id="1", description="important details")]
        result = Prompts("").rescope_prompt(tasks, "")
        assert "important details" in result

    def test_empty_task_list(self) -> None:
        result = Prompts("").rescope_prompt([], "")
        assert isinstance(result, str)
        assert "(none)" in result  # both completed and commit summary


# ── Prompts.rescope_duplicate_nudge ──────────────────────────────────────────


class TestRescopeDuplicateNudge:
    def test_includes_duplicate_titles(self) -> None:
        result = Prompts("").rescope_duplicate_nudge(
            ["Same name"], attempts_remaining=0
        )
        assert "Same name" in result

    def test_includes_multiple_duplicate_titles(self) -> None:
        result = Prompts("").rescope_duplicate_nudge(
            ["Title A", "Title B"], attempts_remaining=0
        )
        assert "Title A" in result
        assert "Title B" in result

    def test_asks_for_unique_titles(self) -> None:
        result = Prompts("").rescope_duplicate_nudge(["Dup"], attempts_remaining=0)
        assert "unique" in result.lower()

    def test_includes_json_format_instruction(self) -> None:
        result = Prompts("").rescope_duplicate_nudge(["Dup"], attempts_remaining=0)
        assert '{"tasks": [...]}' in result

    def test_no_other_text_instruction_present(self) -> None:
        result = Prompts("").rescope_duplicate_nudge(["Dup"], attempts_remaining=0)
        assert "No other text" in result

    def test_final_attempt_message_when_zero_remaining(self) -> None:
        result = Prompts("").rescope_duplicate_nudge(["Dup"], attempts_remaining=0)
        assert "final attempt" in result.lower()

    def test_remaining_count_when_nonzero(self) -> None:
        result = Prompts("").rescope_duplicate_nudge(["Dup"], attempts_remaining=2)
        assert "2" in result


# ── Prompts.stores_persona ────────────────────────────────────────────────────


class TestPromptsStoresPersona:
    def test_persona_stored(self) -> None:
        p = Prompts("my persona")
        assert p.persona == "my persona"

    def test_different_personas_independent(self) -> None:
        p1 = Prompts("persona A")
        p2 = Prompts("persona B")
        activities = [("owner/repo", "working", True)]
        assert "persona A" in p1.status_prompt(activities)
        assert "persona B" in p2.status_prompt(activities)
        assert "persona A" not in p2.status_prompt(activities)


# ── Prompts.rewrite_description_prompt ───────────────────────────────────────


class TestRewriteDescriptionPrompt:
    def _task(
        self,
        title: str,
        task_id: str = "1",
        status: str = "pending",
        description: str = "",
    ) -> dict:
        return {
            "id": task_id,
            "title": title,
            "status": status,
            "description": description,
        }

    def _body(self, desc: str = "Does something useful.\n\nFixes #5.") -> str:
        return (
            f"{desc}\n\n---\n\n## Work queue\n\n"
            "<!-- WORK_QUEUE_START -->\n- [ ] do a thing\n<!-- WORK_QUEUE_END -->"
        )

    def test_includes_current_description(self) -> None:
        result = Prompts("").rewrite_description_prompt(
            self._body("Implements the feature.\n\nFixes #7."),
            [self._task("New task")],
        )
        assert "Implements the feature." in result

    def test_excludes_work_queue_section_from_context(self) -> None:
        result = Prompts("").rewrite_description_prompt(
            self._body(), [self._task("A task")]
        )
        assert "WORK_QUEUE_START" not in result
        assert "do a thing" not in result

    def test_includes_pending_tasks(self) -> None:
        result = Prompts("").rewrite_description_prompt(
            self._body(),
            [self._task("Add caching layer")],
        )
        assert "Add caching layer" in result

    def test_excludes_completed_tasks(self) -> None:
        result = Prompts("").rewrite_description_prompt(
            self._body(),
            [
                self._task("Done already", status="completed"),
                self._task("Still pending"),
            ],
        )
        assert "Done already" not in result

    def test_empty_pending_shows_none(self) -> None:
        result = Prompts("").rewrite_description_prompt(
            self._body(),
            [self._task("Finished", status="completed")],
        )
        assert "(none)" in result

    def test_task_description_included(self) -> None:
        result = Prompts("").rewrite_description_prompt(
            self._body(),
            [self._task("Cache results", description="use Redis")],
        )
        assert "use Redis" in result

    def test_fixes_line_preservation_rule_stated(self) -> None:
        result = Prompts("").rewrite_description_prompt(self._body(), [])
        assert "Fixes #N" in result or "Fixes #" in result
        assert "preserve" in result.lower() or "exactly" in result.lower()

    def test_no_work_queue_content_rule_stated(self) -> None:
        result = Prompts("").rewrite_description_prompt(self._body(), [])
        assert "work queue" in result.lower()

    def test_body_tag_contract_stated(self) -> None:
        """Prompt must instruct Opus to wrap output in <body> tags so we can
        reliably strip preamble and trailing chatter."""
        result = Prompts("").rewrite_description_prompt(self._body(), [])
        assert "<body>" in result
        assert "</body>" in result

    def test_extracts_description_at_divider(self) -> None:
        body = "My description.\n\nFixes #3.\n\n---\n\nStuff below divider."
        result = Prompts("").rewrite_description_prompt(body, [])
        assert "My description." in result
        assert "Stuff below divider." not in result

    def test_fallback_to_wq_marker_when_no_divider(self) -> None:
        body = (
            "Short desc.\n<!-- WORK_QUEUE_START -->"
            "\n- [ ] task\n<!-- WORK_QUEUE_END -->"
        )
        result = Prompts("").rewrite_description_prompt(body, [])
        assert "Short desc." in result
        assert "WORK_QUEUE_START" not in result

    def test_fallback_to_full_body_when_no_markers(self) -> None:
        body = "Plain description with no markers."
        result = Prompts("").rewrite_description_prompt(body, [])
        assert "Plain description with no markers." in result

    def test_empty_task_list(self) -> None:
        result = Prompts("").rewrite_description_prompt(self._body(), [])
        assert isinstance(result, str)
        assert "(none)" in result
