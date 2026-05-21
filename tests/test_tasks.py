import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from fido.claude import ClaudeClient
from fido.prompts import Prompts
from fido.rocq import pr_body_task_store as oracle
from fido.rocq import thread_auto_resolve as thread_oracle
from fido.tasks import (
    Tasks,
    _apply_reorder,
    _assert_merge_lineage_preserved,
    _assert_rescope_matches_oracle,
    _build_task_list_snapshot,
    _compute_thread_changes,
    _find_cross_op_errors,
    _find_duplicate_titles,
    _format_work_queue,
    _make_new_tasks_from_opus,
    _operations_to_items,
    _parse_rescope_operations,
    _parse_rescope_verdicts,
    _rescope_releases_for_oracle,
    _rescope_snapshot_order_for_oracle,
    _rescope_state_for_oracle,
    _rescope_task_kind_for_oracle,
    _rescope_task_source_comment_for_oracle,
    _rescope_task_status_for_oracle,
    _task_kind_for_oracle,
    _task_source_comment_for_oracle,
    _task_status_for_oracle,
    _task_store_for_oracle,
    _validate_rescope_batch,
    reorder_tasks,
    review_thread_for_auto_resolve_oracle,
    thread_tasks_for_auto_resolve_oracle,
)
from fido.types import RescopeIntent, TaskStatus, TaskType


def _client(run_turn_return: str = "") -> MagicMock:
    """Create a mock ClaudeClient with a configurable run_turn return."""
    client = MagicMock(spec=ClaudeClient)
    client.voice_model = "claude-opus-4-6"
    client.work_model = "claude-sonnet-4-6"
    client.brief_model = "claude-haiku-4-5"
    client.run_turn.return_value = run_turn_return
    return client


def _task_file(tmp_path: Path) -> Path:
    git_dir = tmp_path / ".git" / "fido"
    git_dir.mkdir(parents=True)
    return git_dir / "tasks.json"


class TestThreadAutoResolveOracleAdapter:
    def test_maps_thread_comment_actor_classes(self) -> None:
        thread = review_thread_for_auto_resolve_oracle(
            {
                "isResolved": False,
                "comments": {
                    "nodes": [
                        {"databaseId": 1, "author": {"login": "fido-bot"}},
                        {"databaseId": 2, "author": {"login": "owner"}},
                        {"databaseId": 3, "author": {"login": "collab"}},
                        {"databaseId": 4, "author": {"login": "ci-helper"}},
                        {"databaseId": 5, "author": {"login": "copilot[bot]"}},
                        {"databaseId": 6, "author": {"login": "drive-by"}},
                    ]
                },
            },
            "fido-bot",
            owner="owner",
            collaborators=frozenset({"collab"}),
            allowed_bots=frozenset({"ci-helper"}),
        )

        authors = [
            comment.thread_comment_author for comment in thread.review_thread_comments
        ]
        assert isinstance(authors[0], thread_oracle.CommentByFido)
        assert isinstance(authors[1], thread_oracle.CommentByActionable)
        assert isinstance(authors[2], thread_oracle.CommentByActionable)
        assert isinstance(authors[3], thread_oracle.CommentByBot)
        assert isinstance(authors[4], thread_oracle.CommentByBot)
        assert isinstance(authors[5], thread_oracle.CommentIgnored)


class TestTaskStoreOracleAdapter:
    def test_task_kind_maps_ci_title_before_type(self) -> None:
        task = {"title": "CI failure: lint", "type": "spec"}

        assert isinstance(_task_kind_for_oracle(task), oracle.TaskCI)

    def test_task_kind_maps_thread_ask_defer_and_spec(self) -> None:
        assert isinstance(
            _task_kind_for_oracle({"title": "Review", "type": "thread"}),
            oracle.TaskThread,
        )
        assert isinstance(
            _task_kind_for_oracle({"title": "ASK: clarify", "type": "spec"}),
            oracle.TaskAsk,
        )
        assert isinstance(
            _task_kind_for_oracle({"title": "DEFER: later", "type": "spec"}),
            oracle.TaskDefer,
        )
        assert isinstance(
            _task_kind_for_oracle({"title": "Build it", "type": "spec"}),
            oracle.TaskSpec,
        )

    def test_task_status_maps_visible_states(self) -> None:
        assert isinstance(
            _task_status_for_oracle({"status": TaskStatus.COMPLETED}),
            oracle.StatusCompleted,
        )
        assert isinstance(
            _task_status_for_oracle({"status": TaskStatus.BLOCKED}),
            oracle.StatusBlocked,
        )
        assert isinstance(
            _task_status_for_oracle({"status": TaskStatus.IN_PROGRESS}),
            oracle.StatusPending,
        )
        assert isinstance(_task_status_for_oracle({}), oracle.StatusPending)

    def test_task_source_comment_maps_thread_metadata(self) -> None:
        assert _task_source_comment_for_oracle({"thread": {"comment_id": "42"}}) == 42
        assert _task_source_comment_for_oracle({"thread": {}}) is None

    def test_task_store_preserves_order_and_rows(self) -> None:
        task_list = [
            {
                "title": "First",
                "description": "one",
                "type": "spec",
                "status": "pending",
            },
            {
                "title": "Second",
                "description": "two",
                "type": "thread",
                "status": "completed",
                "thread": {"comment_id": 9},
            },
        ]

        store, by_oracle_id = _task_store_for_oracle(task_list)

        assert store.task_store_order == [1, 2]
        assert by_oracle_id == {1: task_list[0], 2: task_list[1]}
        assert store.task_store_rows[1].description == "one"
        assert store.task_store_rows[2].source_comment == 9

    def test_format_work_queue_uses_oracle_projection(self) -> None:
        task_list = [
            {"title": "Spec", "status": "pending", "type": "spec"},
            {"title": "CI failure: lint", "status": "pending", "type": "spec"},
            {"title": "Blocked", "status": "blocked", "type": "spec"},
            {"title": "Done", "status": "completed", "type": "thread"},
        ]

        queue = _format_work_queue(task_list)
        lines = queue.splitlines()

        assert lines[0] == "- [ ] CI failure: lint **→ next** <!-- type:spec -->"
        assert lines[1] == "- [ ] Spec <!-- type:spec -->"
        assert "Blocked" not in queue
        assert "- [x] Done <!-- type:thread -->" in queue


class TestRescopeOracleAdapter:
    def test_task_kind_maps_all_runtime_kinds(self) -> None:
        assert (
            type(
                _rescope_task_kind_for_oracle({"title": "ASK: clarify", "type": "spec"})
            ).__name__
            == "TaskAsk"
        )
        assert (
            type(
                _rescope_task_kind_for_oracle({"title": "DEFER: later", "type": "spec"})
            ).__name__
            == "TaskDefer"
        )
        assert (
            type(
                _rescope_task_kind_for_oracle(
                    {"title": "CI failure: lint", "type": "spec"}
                )
            ).__name__
            == "TaskCI"
        )
        assert (
            type(
                _rescope_task_kind_for_oracle({"title": "Review", "type": "thread"})
            ).__name__
            == "TaskThread"
        )
        assert (
            type(
                _rescope_task_kind_for_oracle({"title": "Build it", "type": "spec"})
            ).__name__
            == "TaskSpec"
        )

    def test_task_status_and_source_comment_map_runtime_values(self) -> None:
        assert (
            type(
                _rescope_task_status_for_oracle({"status": TaskStatus.COMPLETED})
            ).__name__
            == "StatusCompleted"
        )
        assert (
            type(
                _rescope_task_status_for_oracle({"status": TaskStatus.BLOCKED})
            ).__name__
            == "StatusBlocked"
        )
        assert type(_rescope_task_status_for_oracle({})).__name__ == "StatusPending"
        assert (
            _rescope_task_source_comment_for_oracle({"thread": {"comment_id": "42"}})
            == 42
        )
        assert _rescope_task_source_comment_for_oracle({}) is None

    def test_snapshot_order_includes_snapshot_non_completed_tasks(
        self,
    ) -> None:
        current = [
            {"id": "a", "title": "A", "status": "pending", "type": "spec"},
            {"id": "b", "title": "B", "status": "completed", "type": "spec"},
            {"id": "c", "title": "C", "status": "pending", "type": "spec"},
        ]
        ids_by_task_id, _tasks_by_oracle_id, _order, _rows = _rescope_state_for_oracle(
            current
        )

        assert _rescope_snapshot_order_for_oracle(
            current, frozenset({"a", "b"}), ids_by_task_id
        ) == [1]

    def test_releases_encode_description_updates_and_keeps(self) -> None:
        """#1357: omitted tasks get KeepTask, not CompleteTask.  Completion
        is a worker-turn decision; the rescope reducer can only Keep,
        Rewrite, or KeepTask (for omitted)."""
        current = [
            {
                "id": "a",
                "title": "A",
                "description": "old",
                "status": "pending",
                "type": "thread",
                "thread": {"comment_id": "1"},
            },
            {
                "id": "b",
                "title": "B",
                "description": "",
                "status": "pending",
                "type": "spec",
            },
            {
                "id": "c",
                "title": "C",
                "description": "",
                "status": "pending",
                "type": "ci",
            },
        ]
        ids_by_task_id, tasks_by_oracle_id, _order, _rows = _rescope_state_for_oracle(
            current
        )
        releases = _rescope_releases_for_oracle(
            current,
            [
                {"id": "a", "title": "Ignored title", "description": "new"},
                {"id": "c", "title": "C"},
            ],
            frozenset({"a", "b", "c"}),
            ids_by_task_id,
            split_child_ids={},
            tasks_by_oracle_id=tasks_by_oracle_id,
        )

        assert [type(release.release_decision).__name__ for release in releases] == [
            "RewriteTask",
            "KeepTask",
            "KeepTask",
        ]


class TestAddTask:
    def test_creates_file(self, tmp_path: Path) -> None:
        task = Tasks(tmp_path).add(title="test task", task_type=TaskType.SPEC)
        assert task["title"] == "test task"
        assert task["status"] == "pending"
        assert task["type"] == "spec"
        tasks = Tasks(tmp_path).list()
        assert len(tasks) == 1

    def test_appends(self, tmp_path: Path) -> None:
        Tasks(tmp_path).add(title="one", task_type=TaskType.SPEC)
        Tasks(tmp_path).add(title="two", task_type=TaskType.SPEC)
        tasks = Tasks(tmp_path).list()
        assert len(tasks) == 2

    def test_with_thread(self, tmp_path: Path) -> None:
        thread = {"repo": "a/b", "pr": 1, "comment_id": 123}
        task = Tasks(tmp_path).add(title="t", task_type=TaskType.THREAD, thread=thread)
        assert task["thread"] == thread

    def test_id_is_unique(self, tmp_path: Path) -> None:
        t1 = Tasks(tmp_path).add(title="a", task_type=TaskType.SPEC)
        t2 = Tasks(tmp_path).add(title="b", task_type=TaskType.SPEC)
        assert t1["id"] != t2["id"]

    def test_appends_at_end(self, tmp_path: Path) -> None:
        Tasks(tmp_path).add(title="first", task_type=TaskType.SPEC)
        Tasks(tmp_path).add(title="second", task_type=TaskType.SPEC)
        tasks = Tasks(tmp_path).list()
        assert tasks[0]["title"] == "first"
        assert tasks[1]["title"] == "second"

    def test_returns_existing_pending_task_if_title_matches(
        self, tmp_path: Path
    ) -> None:
        t1 = Tasks(tmp_path).add(title="duplicate task", task_type=TaskType.SPEC)
        t2 = Tasks(tmp_path).add(title="duplicate task", task_type=TaskType.SPEC)
        assert t1["id"] == t2["id"]
        assert len(Tasks(tmp_path).list()) == 1

    def test_sanitizes_multiline_title(self, tmp_path: Path) -> None:
        task = Tasks(tmp_path).add(
            title="first line\n\nsecond paragraph\n- bullet\n- another",
            task_type=TaskType.SPEC,
        )
        assert task["title"] == "first line second paragraph - bullet - another"
        assert "\n" not in task["title"]

    def test_collapses_whitespace_in_title(self, tmp_path: Path) -> None:
        task = Tasks(tmp_path).add(
            title="  too   many    spaces  \t\there  ",
            task_type=TaskType.SPEC,
        )
        assert task["title"] == "too many spaces here"

    def test_does_not_deduplicate_completed_tasks(self, tmp_path: Path) -> None:
        t1 = Tasks(tmp_path).add(title="done task", task_type=TaskType.SPEC)
        Tasks(tmp_path).complete_by_id(t1["id"])
        t2 = Tasks(tmp_path).add(title="done task", task_type=TaskType.SPEC)
        assert t1["id"] != t2["id"]
        assert len(Tasks(tmp_path).list()) == 2

    def test_thread_task_appends_at_end(self, tmp_path: Path) -> None:
        Tasks(tmp_path).add(title="existing", task_type=TaskType.SPEC)
        thread = {"repo": "r/r", "pr": 1, "comment_id": 42}
        Tasks(tmp_path).add(
            title="comment task", task_type=TaskType.THREAD, thread=thread
        )
        tasks = Tasks(tmp_path).list()
        assert tasks[0]["title"] == "existing"
        assert tasks[1]["title"] == "comment task"

    def test_deduplicates_by_comment_id(self, tmp_path: Path) -> None:
        thread = {"repo": "r/r", "pr": 1, "comment_id": 99}
        t1 = Tasks(tmp_path).add(
            title="first title", task_type=TaskType.THREAD, thread=thread
        )
        t2 = Tasks(tmp_path).add(
            title="second title", task_type=TaskType.THREAD, thread=thread
        )
        assert t1["id"] == t2["id"]
        assert len(Tasks(tmp_path).list()) == 1

    def test_deduplicates_by_comment_id_even_when_completed(
        self, tmp_path: Path
    ) -> None:
        thread = {"repo": "r/r", "pr": 1, "comment_id": 55}
        t1 = Tasks(tmp_path).add(
            title="handle feedback", task_type=TaskType.THREAD, thread=thread
        )
        Tasks(tmp_path).complete_by_id(t1["id"])
        t2 = Tasks(tmp_path).add(
            title="handle feedback again",
            task_type=TaskType.THREAD,
            thread=thread,
        )
        assert t1["id"] == t2["id"]
        assert len(Tasks(tmp_path).list()) == 1

    def test_distinct_comments_in_same_lineage_produce_distinct_tasks(
        self, tmp_path: Path
    ) -> None:
        """#1665: lineage stops being a join key.  Three distinct
        comments on the same PR thread produce three distinct tasks
        — combine / split decisions move to the rescope reducer
        (#1666 / #1667), not the add boundary."""
        first_thread = {
            "repo": "r/r",
            "pr": 1,
            "comment_id": 101,
            "lineage_comment_ids": [100, 101],
        }
        second_thread = {
            "repo": "r/r",
            "pr": 1,
            "comment_id": 102,
            "lineage_comment_ids": [100, 102],
        }

        first = Tasks(tmp_path).add(
            title="handle first comment",
            task_type=TaskType.THREAD,
            thread=first_thread,
        )
        second = Tasks(tmp_path).add(
            title="handle follow-up comment",
            task_type=TaskType.THREAD,
            thread=second_thread,
        )

        assert first["id"] != second["id"]
        tasks = Tasks(tmp_path).list()
        assert len(tasks) == 2
        assert tasks[0]["thread"]["comment_id"] == 101
        assert tasks[1]["thread"]["comment_id"] == 102

    def test_different_comment_ids_are_not_deduplicated(self, tmp_path: Path) -> None:
        t1 = Tasks(tmp_path).add(
            title="t",
            task_type=TaskType.THREAD,
            thread={"repo": "r/r", "pr": 1, "comment_id": 1},
        )
        t2 = Tasks(tmp_path).add(
            title="t",
            task_type=TaskType.THREAD,
            thread={"repo": "r/r", "pr": 1, "comment_id": 2},
        )
        assert t1["id"] != t2["id"]
        assert len(Tasks(tmp_path).list()) == 2

    def test_completed_lineage_does_not_block_new_task_with_different_comment(
        self, tmp_path: Path
    ) -> None:
        """Regression: completed task with shared lineage_key must not dedup
        a new task from a different comment (#1246, sibling of #1188)."""
        thread_a = {
            "repo": "r/r",
            "pr": 1,
            "comment_id": 100,
            "lineage_key": "issues:r/r:1",
        }
        thread_b = {
            "repo": "r/r",
            "pr": 1,
            "comment_id": 200,
            "lineage_key": "issues:r/r:1",
        }

        t1 = Tasks(tmp_path).add(
            title="do first thing",
            task_type=TaskType.THREAD,
            thread=thread_a,
        )
        Tasks(tmp_path).complete_by_id(t1["id"])

        t2 = Tasks(tmp_path).add(
            title="do second thing",
            task_type=TaskType.THREAD,
            thread=thread_b,
        )

        assert t1["id"] != t2["id"], "new comment must produce a new task"
        tasks = Tasks(tmp_path).list()
        assert len(tasks) == 2

    def test_in_progress_sibling_does_not_block_new_distinct_comment(
        self, tmp_path: Path
    ) -> None:
        """#1665: even an in-progress sibling task in the same lineage
        does not coalesce a new distinct comment.  The combine
        decision moves to the rescope reducer (#1667) — at the add
        boundary every distinct comment yields its own task."""
        thread_a = {
            "repo": "r/r",
            "pr": 1,
            "comment_id": 100,
        }
        thread_b = {
            "repo": "r/r",
            "pr": 1,
            "comment_id": 200,
        }

        t1 = Tasks(tmp_path).add(
            title="do first thing",
            task_type=TaskType.THREAD,
            thread=thread_a,
        )
        Tasks(tmp_path).update(t1["id"], TaskStatus.IN_PROGRESS)

        t2 = Tasks(tmp_path).add(
            title="do second thing",
            task_type=TaskType.THREAD,
            thread=thread_b,
        )

        assert t1["id"] != t2["id"]
        assert len(Tasks(tmp_path).list()) == 2

    def test_task_type_required(self, tmp_path: Path) -> None:
        import pytest

        with pytest.raises(TypeError, match="task_type must be TaskType"):
            Tasks(tmp_path).add(title="t", task_type="spec")  # type: ignore[arg-type]


class TestUpdateTask:
    def test_updates_status(self, tmp_path: Path) -> None:
        task = Tasks(tmp_path).add(title="t", task_type=TaskType.SPEC)
        assert Tasks(tmp_path).update(task["id"], TaskStatus.COMPLETED)
        tasks = Tasks(tmp_path).list()
        assert tasks[0]["status"] == "completed"

    def test_returns_false_if_not_found(self, tmp_path: Path) -> None:
        Tasks(tmp_path).add(title="t", task_type=TaskType.SPEC)
        assert not Tasks(tmp_path).update("nonexistent", TaskStatus.COMPLETED)


class TestListTasks:
    def test_empty(self, tmp_path: Path) -> None:
        _task_file(tmp_path).write_text("[]")
        assert Tasks(tmp_path).list() == []

    def test_corrupt_json(self, tmp_path: Path) -> None:
        _task_file(tmp_path).write_text("not json")
        with pytest.raises(ValueError, match="corrupt tasks.json"):
            Tasks(tmp_path).list()

    def test_raises_on_missing_type_field(self, tmp_path: Path) -> None:
        tf = _task_file(tmp_path)
        tf.write_text('[{"id": "bad", "title": "no type", "status": "pending"}]')
        with pytest.raises(ValueError, match="missing required type field"):
            Tasks(tmp_path).list()


class TestCompleteById:
    def test_marks_completed_returns_none_when_no_thread(self, tmp_path: Path) -> None:
        task = Tasks(tmp_path).add(title="do something", task_type=TaskType.SPEC)
        result = Tasks(tmp_path).complete_by_id(task["id"])
        assert result is None
        assert Tasks(tmp_path).list()[0]["status"] == "completed"

    def test_returns_thread_when_present(self, tmp_path: Path) -> None:
        thread = {"repo": "a/b", "pr": 1, "comment_id": 99}
        task = Tasks(tmp_path).add(title="t", task_type=TaskType.THREAD, thread=thread)
        result = Tasks(tmp_path).complete_by_id(task["id"])
        assert result == thread

    def test_returns_none_when_not_found(self, tmp_path: Path) -> None:
        Tasks(tmp_path).add(title="other", task_type=TaskType.SPEC)
        assert Tasks(tmp_path).complete_by_id("missing") is None
        assert Tasks(tmp_path).list()[0]["status"] == "pending"

    def test_skips_already_completed(self, tmp_path: Path) -> None:
        task = Tasks(tmp_path).add(title="t", task_type=TaskType.SPEC)
        Tasks(tmp_path).complete_by_id(task["id"])
        # second call on already-completed task returns None
        assert Tasks(tmp_path).complete_by_id(task["id"]) is None

    def test_completes_correct_task_by_id(self, tmp_path: Path) -> None:
        t1 = Tasks(tmp_path).add(title="a", task_type=TaskType.SPEC)
        Tasks(tmp_path).add(title="b", task_type=TaskType.SPEC)
        Tasks(tmp_path).complete_by_id(t1["id"])
        tasks = Tasks(tmp_path).list()
        assert tasks[0]["status"] == "completed"
        assert tasks[1]["status"] == "pending"


class TestHasPendingTasksForComment:
    def test_returns_true_when_pending_task_exists(self, tmp_path: Path) -> None:
        Tasks(tmp_path).add(
            title="t",
            task_type=TaskType.THREAD,
            thread={"repo": "r/r", "pr": 1, "comment_id": 42},
        )
        assert Tasks(tmp_path).has_pending_for_comment(42)

    def test_returns_false_when_no_tasks(self, tmp_path: Path) -> None:
        assert not Tasks(tmp_path).has_pending_for_comment(42)

    def test_returns_false_when_task_completed(self, tmp_path: Path) -> None:
        task = Tasks(tmp_path).add(
            title="t",
            task_type=TaskType.THREAD,
            thread={"repo": "r/r", "pr": 1, "comment_id": 42},
        )
        Tasks(tmp_path).complete_by_id(task["id"])
        assert not Tasks(tmp_path).has_pending_for_comment(42)

    def test_returns_false_for_different_comment_id(self, tmp_path: Path) -> None:
        Tasks(tmp_path).add(
            title="t",
            task_type=TaskType.THREAD,
            thread={"repo": "r/r", "pr": 1, "comment_id": 99},
        )
        assert not Tasks(tmp_path).has_pending_for_comment(42)

    def test_accepts_string_comment_id(self, tmp_path: Path) -> None:
        Tasks(tmp_path).add(
            title="t",
            task_type=TaskType.THREAD,
            thread={"repo": "r/r", "pr": 1, "comment_id": 42},
        )
        assert Tasks(tmp_path).has_pending_for_comment("42")

    def test_only_pending_tasks_count(self, tmp_path: Path) -> None:
        import json

        # Write two tasks with same comment_id directly (bypassing dedup)
        task_file = _task_file(tmp_path)
        task_file.write_text(
            json.dumps(
                [
                    {
                        "id": "1",
                        "title": "first",
                        "type": "thread",
                        "status": "completed",
                        "thread": {"repo": "r/r", "pr": 1, "comment_id": 42},
                    },
                    {
                        "id": "2",
                        "title": "second",
                        "type": "thread",
                        "status": "pending",
                        "thread": {"repo": "r/r", "pr": 1, "comment_id": 42},
                    },
                ]
            )
        )
        assert Tasks(tmp_path).has_pending_for_comment(42)


class TestRemoveTask:
    def test_removes(self, tmp_path: Path) -> None:
        task = Tasks(tmp_path).add(title="t", task_type=TaskType.SPEC)
        assert Tasks(tmp_path).remove(task["id"])
        assert Tasks(tmp_path).list() == []

    def test_returns_false_if_not_found(self, tmp_path: Path) -> None:
        Tasks(tmp_path).add(title="t", task_type=TaskType.SPEC)
        assert not Tasks(tmp_path).remove("nonexistent")


class TestResetToPending:
    """Tests for Tasks.reset_to_pending — the abort-cleanup helper added
    for #1357 case B (aborted tasks must survive in the queue, not vanish)."""

    def test_resets_in_progress_to_pending(self, tmp_path: Path) -> None:
        task = Tasks(tmp_path).add(title="t", task_type=TaskType.SPEC)
        Tasks(tmp_path).update(task["id"], TaskStatus.IN_PROGRESS)
        assert Tasks(tmp_path).reset_to_pending(task["id"]) is True
        assert Tasks(tmp_path).list()[0]["status"] == str(TaskStatus.PENDING)

    def test_no_op_when_already_pending(self, tmp_path: Path) -> None:
        task = Tasks(tmp_path).add(title="t", task_type=TaskType.SPEC)
        # Idempotent: returns True (found) without rewriting the file.
        assert Tasks(tmp_path).reset_to_pending(task["id"]) is True
        assert Tasks(tmp_path).list()[0]["status"] == str(TaskStatus.PENDING)

    def test_returns_false_if_not_found(self, tmp_path: Path) -> None:
        Tasks(tmp_path).add(title="t", task_type=TaskType.SPEC)
        assert Tasks(tmp_path).reset_to_pending("nonexistent") is False


class TestUnblockTasks:
    def test_unblocks_blocked_tasks(self, tmp_path: Path) -> None:
        t1 = Tasks(tmp_path).add(title="first", task_type=TaskType.SPEC)
        t2 = Tasks(tmp_path).add(title="second", task_type=TaskType.SPEC)
        Tasks(tmp_path).update(t1["id"], TaskStatus.BLOCKED)
        Tasks(tmp_path).update(t2["id"], TaskStatus.BLOCKED)
        count = Tasks(tmp_path).unblock_tasks()
        assert count == 2
        tasks = Tasks(tmp_path).list()
        assert all(t["status"] == str(TaskStatus.PENDING) for t in tasks)

    def test_returns_zero_when_nothing_blocked(self, tmp_path: Path) -> None:
        Tasks(tmp_path).add(title="t", task_type=TaskType.SPEC)
        assert Tasks(tmp_path).unblock_tasks() == 0

    def test_does_not_touch_non_blocked_tasks(self, tmp_path: Path) -> None:
        t1 = Tasks(tmp_path).add(title="pending", task_type=TaskType.SPEC)
        t2 = Tasks(tmp_path).add(title="done", task_type=TaskType.SPEC)
        t3 = Tasks(tmp_path).add(title="blocked", task_type=TaskType.SPEC)
        Tasks(tmp_path).update(t2["id"], TaskStatus.COMPLETED)
        Tasks(tmp_path).update(t3["id"], TaskStatus.BLOCKED)
        count = Tasks(tmp_path).unblock_tasks()
        assert count == 1
        tasks = {t["id"]: t for t in Tasks(tmp_path).list()}
        assert tasks[t1["id"]]["status"] == str(TaskStatus.PENDING)
        assert tasks[t2["id"]]["status"] == str(TaskStatus.COMPLETED)
        assert tasks[t3["id"]]["status"] == str(TaskStatus.PENDING)

    def test_returns_zero_on_empty_file(self, tmp_path: Path) -> None:
        assert Tasks(tmp_path).unblock_tasks() == 0


# ── _build_task_list_snapshot ─────────────────────────────────────────────────


class TestBuildTaskListSnapshot:
    """The :class:`TaskListSnapshot` projection published from
    :meth:`Tasks.on_mutate` after every tasks.json write (#1696)."""

    def test_counts_pending_and_completed(self) -> None:
        snap = _build_task_list_snapshot(
            [
                {"status": "pending", "title": "a"},
                {"status": "pending", "title": "b"},
                {"status": "completed", "title": "c"},
                {"status": "in_progress", "title": "d"},
            ]
        )
        assert snap.pending_task_count == 2
        assert snap.completed_task_count == 1

    def test_current_task_prefers_in_progress(self) -> None:
        """When something is in_progress, ``current_task`` is its title and
        ``task_number`` points at its 1-based position in non-completed."""
        snap = _build_task_list_snapshot(
            [
                {"status": "completed", "title": "done"},
                {"status": "pending", "title": "next"},
                {"status": "in_progress", "title": "active"},
                {"status": "pending", "title": "later"},
            ]
        )
        assert snap.current_task == "active"
        # non-completed = [next, active, later]; active is index 2.
        assert snap.task_number == 2
        assert snap.task_total == 3

    def test_current_task_falls_back_to_first_pending(self) -> None:
        snap = _build_task_list_snapshot(
            [
                {"status": "completed", "title": "done"},
                {"status": "pending", "title": "do this"},
                {"status": "pending", "title": "then that"},
            ]
        )
        assert snap.current_task == "do this"
        assert snap.task_number == 1
        assert snap.task_total == 2

    def test_all_completed_yields_zero_counters(self) -> None:
        """Per #1696, the absent ``current_task`` is the empty string and
        the absent ``task_number`` / ``task_total`` are 0 — uniform
        zero sentinels, no None on the SCADA snapshot."""
        snap = _build_task_list_snapshot(
            [
                {"status": "completed", "title": "a"},
                {"status": "completed", "title": "b"},
            ]
        )
        assert snap.current_task == ""
        assert snap.task_number == 0
        assert snap.task_total == 0


# ── _parse_rescope_operations (#1719) ─────────────────────────────────────────


class TestParseRescopeOperations:
    def test_parses_keep(self) -> None:
        raw = '{"operations": [{"op": "keep", "id": "abc"}]}'
        ops, errors = _parse_rescope_operations(raw)
        assert errors == []
        assert _operations_to_items(ops) == [{"id": "abc", "contributing_intents": []}]

    def test_parses_rewrite(self) -> None:
        raw = (
            '{"operations": [{"op": "rewrite", "id": "abc", '
            '"title": "New", "description": "scope"}]}'
        )
        ops, errors = _parse_rescope_operations(raw)
        assert errors == []
        assert _operations_to_items(ops) == [
            {
                "id": "abc",
                "title": "New",
                "description": "scope",
                "contributing_intents": [],
            }
        ]

    def test_parses_rewrite_anchor(self) -> None:
        raw = (
            '{"operations": [{"op": "rewrite_anchor", "id": "abc", '
            '"anchor_comment_id": 12345}]}'
        )
        ops, errors = _parse_rescope_operations(raw)
        assert errors == []
        assert _operations_to_items(ops) == [
            {
                "id": "abc",
                "anchor_comment_id": 12345,
                "contributing_intents": [],
            }
        ]

    def test_parses_remove(self) -> None:
        raw = '{"operations": [{"op": "remove", "id": "abc"}]}'
        ops, errors = _parse_rescope_operations(raw)
        assert errors == []
        assert _operations_to_items(ops) == [
            {"id": "abc", "status": "completed", "contributing_intents": []}
        ]

    def test_parses_merge_with_source_completion_expansion(self) -> None:
        # A single merge op lowers to one target item (with merge_sources)
        # plus N source-completion items so the rocq per-task coverage
        # invariant still holds.
        raw = (
            '{"operations": [{"op": "merge", "target_id": "a", '
            '"sources": ["b", "c"], "title": "Merged", '
            '"description": "scope"}]}'
        )
        ops, errors = _parse_rescope_operations(raw)
        assert errors == []
        assert _operations_to_items(ops) == [
            {
                "id": "a",
                "title": "Merged",
                "description": "scope",
                "merge_sources": ["b", "c"],
                "contributing_intents": [],
            },
            {"id": "b", "status": "completed", "contributing_intents": []},
            {"id": "c", "status": "completed", "contributing_intents": []},
        ]

    def test_parses_split(self) -> None:
        raw = (
            '{"operations": [{"op": "split", "id": "src", '
            '"children": [{"title": "A", "description": "first"}, '
            '{"title": "B", "description": "second"}]}]}'
        )
        ops, errors = _parse_rescope_operations(raw)
        assert errors == []
        assert _operations_to_items(ops) == [
            {
                "id": "src",
                "split_targets": [
                    {"title": "A", "description": "first"},
                    {"title": "B", "description": "second"},
                ],
                "contributing_intents": [],
            }
        ]

    def test_parses_new(self) -> None:
        raw = (
            '{"operations": [{"op": "new", "title": "T", '
            '"description": "d", "type": "spec"}]}'
        )
        ops, errors = _parse_rescope_operations(raw)
        assert errors == []
        assert _operations_to_items(ops) == [
            {
                "id": None,
                "title": "T",
                "description": "d",
                "type": "spec",
                "contributing_intents": [],
            }
        ]

    def test_parses_json_with_preamble(self) -> None:
        raw = 'Reordered:\n{"operations": [{"op": "keep", "id": "1"}]}'
        ops, errors = _parse_rescope_operations(raw)
        assert errors == []
        assert len(ops) == 1

    def test_no_json_at_all_yields_one_error(self) -> None:
        ops, errors = _parse_rescope_operations("not json at all")
        assert ops == []
        assert errors == ["response: no JSON object found"]

    def test_missing_operations_key(self) -> None:
        ops, errors = _parse_rescope_operations('{"tasks": []}')
        assert ops == []
        assert errors == ['response: missing top-level "operations" array']

    def test_operations_must_be_list(self) -> None:
        ops, errors = _parse_rescope_operations('{"operations": "nope"}')
        assert ops == []
        assert any("must be a list" in e for e in errors)

    def test_unknown_op_name(self) -> None:
        raw = '{"operations": [{"op": "bogus", "id": "x"}]}'
        ops, errors = _parse_rescope_operations(raw)
        assert ops == []
        assert any("unknown operation 'bogus'" in e for e in errors)

    def test_collects_every_error_in_one_pass(self) -> None:
        # Rob's directive: useful retries that detail what was wrong,
        # as many things as we can find at once.  This batch combines
        # five distinct defects across five operations — the parser
        # must report them all, not bail on the first.
        raw = (
            '{"operations": ['
            '{"op": "keep"},'  # missing id
            '{"op": "rewrite", "id": "x", "title": ""},'  # blank title, missing description
            '{"op": "rewrite_anchor", "id": "y", "anchor_comment_id": -1},'  # bad anchor
            '{"op": "merge", "target_id": "z", "sources": [], '
            '"title": "T", "description": "d"},'  # empty sources
            '{"op": "split", "id": "s", "children": [{"title": ""}]}'  # bad child
            "]}"
        )
        ops, errors = _parse_rescope_operations(raw)
        assert ops == []
        # Each defect produces at least one error message; assert the
        # full set without pinning exact wording.
        joined = " | ".join(errors)
        assert "operations[0].id" in joined
        assert "operations[1].title" in joined
        assert "operations[2].anchor_comment_id" in joined
        assert "operations[3].sources" in joined
        assert "operations[4].children" in joined

    def test_op_must_be_string(self) -> None:
        raw = '{"operations": [{"op": 42}]}'
        ops, errors = _parse_rescope_operations(raw)
        assert ops == []
        assert any("operations[0].op" in e and "must be a string" in e for e in errors)

    def test_op_must_be_dict(self) -> None:
        raw = '{"operations": ["not a dict"]}'
        ops, errors = _parse_rescope_operations(raw)
        assert ops == []
        assert any("operations[0]: must be a dict" in e for e in errors)

    def test_empty_operations_list_is_valid(self) -> None:
        ops, errors = _parse_rescope_operations('{"operations": []}')
        assert ops == []
        assert errors == []

    def test_skips_junk_brace_and_decodes_real_envelope(self) -> None:
        raw = '{ this is { junk } before {"operations": [{"op": "keep", "id": "1"}]}'
        ops, errors = _parse_rescope_operations(raw)
        assert errors == []
        assert _operations_to_items(ops) == [{"id": "1", "contributing_intents": []}]

    def test_remove_missing_id(self) -> None:
        ops, errors = _parse_rescope_operations('{"operations": [{"op": "remove"}]}')
        assert ops == []
        assert any("operations[0].id" in e for e in errors)

    def test_new_missing_title_description_type(self) -> None:
        ops, errors = _parse_rescope_operations('{"operations": [{"op": "new"}]}')
        assert ops == []
        joined = " | ".join(errors)
        assert "operations[0].title" in joined
        assert "operations[0].description" in joined
        assert "operations[0].type" in joined

    def test_merge_with_non_string_source_entries(self) -> None:
        # The error-collecting parser flags every bad entry in the
        # sources list — not just the first.
        raw = (
            '{"operations": [{"op": "merge", "target_id": "a", '
            '"sources": ["b", 42, ""], "title": "M", "description": ""}]}'
        )
        _ops, errors = _parse_rescope_operations(raw)
        assert any("operations[0].sources[1]" in e for e in errors)
        assert any("operations[0].sources[2]" in e for e in errors)

    def test_merge_with_all_invalid_sources(self) -> None:
        # Empty after filtering bad entries — distinct error from the
        # "list is structurally empty" branch above.
        raw = (
            '{"operations": [{"op": "merge", "target_id": "a", '
            '"sources": ["", 0], "title": "M", "description": ""}]}'
        )
        ops, errors = _parse_rescope_operations(raw)
        assert ops == []
        assert any("every entry was malformed" in e for e in errors)

    def test_split_with_empty_children_list(self) -> None:
        raw = '{"operations": [{"op": "split", "id": "src", "children": []}]}'
        ops, errors = _parse_rescope_operations(raw)
        assert ops == []
        assert any(
            "operations[0].children" in e and "non-empty list" in e for e in errors
        )

    def test_split_with_non_dict_child(self) -> None:
        raw = (
            '{"operations": [{"op": "split", "id": "src", "children": ["not a dict"]}]}'
        )
        ops, errors = _parse_rescope_operations(raw)
        assert ops == []
        assert any(
            "operations[0].children[0]" in e and "must be a dict" in e for e in errors
        )

    def test_keep_carries_contributing_intents(self) -> None:
        # #1722: every op may attribute itself to one or more
        # originating RescopeIntent comment ids.  The translator
        # stamps the resulting item dict so the materializer can
        # persist them on the task.
        raw = (
            '{"operations": [{"op": "keep", "id": "abc", '
            '"contributing_intents": [101, 202]}]}'
        )
        ops, errors = _parse_rescope_operations(raw)
        assert errors == []
        items = _operations_to_items(ops)
        assert items == [{"id": "abc", "contributing_intents": [101, 202]}]

    def test_contributing_intents_default_empty(self) -> None:
        # Missing field = no attribution; translator emits an empty
        # list so callers don't need to special-case ``None``.
        raw = '{"operations": [{"op": "keep", "id": "abc"}]}'
        ops, errors = _parse_rescope_operations(raw)
        assert errors == []
        items = _operations_to_items(ops)
        assert items == [{"id": "abc", "contributing_intents": []}]

    def test_contributing_intents_dedup_in_arrival_order(self) -> None:
        # Duplicate entries in a single op's intents list are
        # collapsed in arrival order so the persisted set is clean.
        raw = (
            '{"operations": [{"op": "keep", "id": "x", '
            '"contributing_intents": [5, 5, 7, 5, 7]}]}'
        )
        ops, errors = _parse_rescope_operations(raw)
        assert errors == []
        items = _operations_to_items(ops)
        assert items[0]["contributing_intents"] == [5, 7]

    def test_contributing_intents_must_be_list(self) -> None:
        raw = (
            '{"operations": [{"op": "keep", "id": "x", '
            '"contributing_intents": "not a list"}]}'
        )
        ops, errors = _parse_rescope_operations(raw)
        assert ops == []
        assert any(
            "operations[0].contributing_intents" in e and "must be a list" in e
            for e in errors
        )

    def test_contributing_intents_entries_must_be_positive_int(self) -> None:
        # Every malformed entry is reported in one pass — booleans,
        # zero, negatives, and non-ints all fail.  Rob's directive:
        # useful retries that detail every problem at once.
        raw = (
            '{"operations": [{"op": "keep", "id": "x", '
            '"contributing_intents": [0, -1, "str", true, 42]}]}'
        )
        ops, errors = _parse_rescope_operations(raw)
        assert ops == []
        for index in (0, 1, 2, 3):
            assert any(
                f"operations[0].contributing_intents[{index}]" in e for e in errors
            )

    def test_merge_op_carries_contributing_intents_on_target_and_sources(
        self,
    ) -> None:
        # #1722: merge target item carries the op's intents AND each
        # synthesised source-completion item carries them too — those
        # intents drove the source's closure as well as the target's
        # mutation.
        raw = (
            '{"operations": [{"op": "merge", "target_id": "a", '
            '"sources": ["b"], "title": "Merged", '
            '"description": "scope", "contributing_intents": [99]}]}'
        )
        ops, errors = _parse_rescope_operations(raw)
        assert errors == []
        items = _operations_to_items(ops)
        target = next(i for i in items if i["id"] == "a")
        source_completion = next(i for i in items if i["id"] == "b")
        assert target["contributing_intents"] == [99]
        assert source_completion["contributing_intents"] == [99]

    def test_decode_skips_non_dict_first_value(self) -> None:
        # First decodable JSON value at the first ``{`` could be a
        # nested non-dict shape; the decoder advances past it and
        # finds the real envelope.  Two sibling objects in a row
        # exercise the ``pos = end`` continue-and-keep-scanning path.
        raw = '{"unrelated": []} {"operations": [{"op": "keep", "id": "1"}]}'
        ops, errors = _parse_rescope_operations(raw)
        assert ops == []  # first decoded object lacks "operations" key
        assert errors == ['response: missing top-level "operations" array']


def _intent(
    cid: int,
    text: str = "do thing",
    *,
    timestamp: str = "2024-01-15T10:00:00+00:00",
    author: str = "alice",
) -> RescopeIntent:
    return RescopeIntent(
        change_request=text,
        comment_id=cid,
        timestamp=timestamp,
        author=author,
    )


class TestParseRescopeVerdicts:
    def test_minimal_honored_verdict(self) -> None:
        raw = '{"verdicts": [{"intent_comment_id": 1, "outcome": "honored"}]}'
        verdicts, errors = _parse_rescope_verdicts(raw, [_intent(1)])
        assert errors == []
        assert len(verdicts) == 1
        assert verdicts[0].outcome == "honored"

    def test_full_verdict_with_all_fields(self) -> None:
        raw = (
            '{"verdicts": [{'
            '"intent_comment_id": 1, '
            '"outcome": "reshaped", '
            '"ops": [{"op": "rewrite", "id": "T1", "title": "x"}], '
            '"affected_task_ids": ["T1"], '
            '"by_intent_comment_id": null, '
            '"narrative": "folded into existing task"'
            "}]}"
        )
        verdicts, errors = _parse_rescope_verdicts(raw, [_intent(1)])
        assert errors == []
        assert verdicts[0].outcome == "reshaped"
        assert verdicts[0].affected_task_ids == ("T1",)
        assert verdicts[0].narrative == "folded into existing task"

    def test_supersedence_in_batch(self) -> None:
        # "red" → "no, green" — red's verdict points at green.
        raw = (
            '{"verdicts": ['
            '{"intent_comment_id": 1, "outcome": "superseded", '
            '"by_intent_comment_id": 2, '
            '"narrative": "color overridden"},'
            '{"intent_comment_id": 2, "outcome": "honored"}'
            "]}"
        )
        verdicts, errors = _parse_rescope_verdicts(raw, [_intent(1), _intent(2)])
        assert errors == []
        assert verdicts[0].by_intent_comment_id == 2

    def test_no_json_object(self) -> None:
        _, errors = _parse_rescope_verdicts("not even json", [_intent(1)])
        assert errors == ["response: no JSON object found"]

    def test_missing_verdicts_key(self) -> None:
        _, errors = _parse_rescope_verdicts('{"foo": []}', [_intent(1)])
        assert errors == ['response: missing top-level "verdicts" array']

    def test_verdicts_not_a_list(self) -> None:
        _, errors = _parse_rescope_verdicts('{"verdicts": {}}', [_intent(1)])
        assert errors == ["response.verdicts: must be a list, got dict"]

    def test_verdict_not_a_dict(self) -> None:
        _, errors = _parse_rescope_verdicts('{"verdicts": ["nope"]}', [_intent(1)])
        # Plus the missing-verdict error for intent 1 since we couldn't parse it.
        assert "verdicts[0]: must be a dict" in errors[0]

    def test_missing_intent_comment_id(self) -> None:
        raw = '{"verdicts": [{"outcome": "honored"}]}'
        _, errors = _parse_rescope_verdicts(raw, [_intent(1)])
        assert any("missing required field 'intent_comment_id'" in e for e in errors)

    def test_missing_outcome(self) -> None:
        raw = '{"verdicts": [{"intent_comment_id": 1}]}'
        _, errors = _parse_rescope_verdicts(raw, [_intent(1)])
        assert any("missing required field 'outcome'" in e for e in errors)

    def test_unknown_intent_comment_id(self) -> None:
        raw = '{"verdicts": [{"intent_comment_id": 999, "outcome": "honored"}]}'
        _, errors = _parse_rescope_verdicts(raw, [_intent(1)])
        assert any("999 not in batch" in e for e in errors)

    def test_duplicate_intent_comment_id_in_verdicts(self) -> None:
        raw = (
            '{"verdicts": ['
            '{"intent_comment_id": 1, "outcome": "honored"},'
            '{"intent_comment_id": 1, "outcome": "no_op"}'
            "]}"
        )
        _, errors = _parse_rescope_verdicts(raw, [_intent(1)])
        assert any("already covered by an earlier verdict" in e for e in errors)

    def test_missing_verdict_for_intent(self) -> None:
        # Intent 2 is in the batch but has no verdict → error.
        raw = '{"verdicts": [{"intent_comment_id": 1, "outcome": "honored"}]}'
        _, errors = _parse_rescope_verdicts(raw, [_intent(1), _intent(2)])
        assert any("missing verdict for intent_comment_id 2" in e for e in errors)

    def test_multiple_missing_verdicts_all_reported(self) -> None:
        # All-errors-at-once contract — both missing ids listed.
        raw = '{"verdicts": []}'
        _, errors = _parse_rescope_verdicts(raw, [_intent(1), _intent(2)])
        assert any("missing verdict for intent_comment_id 1" in e for e in errors)
        assert any("missing verdict for intent_comment_id 2" in e for e in errors)

    def test_by_intent_comment_id_outside_batch(self) -> None:
        raw = (
            '{"verdicts": [{"intent_comment_id": 1, "outcome": "superseded", '
            '"by_intent_comment_id": 999, "narrative": "x"}]}'
        )
        _, errors = _parse_rescope_verdicts(raw, [_intent(1)])
        assert any("by_intent_comment_id: 999 not in batch" in e for e in errors)

    def test_intent_verdict_construction_error_captured(self) -> None:
        # Outcome typo → IntentVerdict ctor raises ValueError → captured as parse error.
        raw = '{"verdicts": [{"intent_comment_id": 1, "outcome": "supersede"}]}'
        _, errors = _parse_rescope_verdicts(raw, [_intent(1)])
        assert any("outcome must be one of" in e for e in errors)

    def test_self_supersedence_captured_via_ctor(self) -> None:
        raw = (
            '{"verdicts": [{"intent_comment_id": 1, "outcome": "superseded", '
            '"by_intent_comment_id": 1, "narrative": "x"}]}'
        )
        _, errors = _parse_rescope_verdicts(raw, [_intent(1)])
        # ctor rejects self-supersedence with "must not reference"
        assert any("must not reference" in e for e in errors)

    def test_supersedence_cycle_detected(self) -> None:
        # 1 ← superseded by 2; 2 ← superseded by 1 — cycle.
        raw = (
            '{"verdicts": ['
            '{"intent_comment_id": 1, "outcome": "superseded", '
            ' "by_intent_comment_id": 2, "narrative": "x"},'
            '{"intent_comment_id": 2, "outcome": "superseded", '
            ' "by_intent_comment_id": 1, "narrative": "y"}'
            "]}"
        )
        _, errors = _parse_rescope_verdicts(raw, [_intent(1), _intent(2)])
        assert any("supersedence graph has a cycle" in e for e in errors)

    def test_longer_supersedence_cycle_detected(self) -> None:
        # 1 → 2 → 3 → 1
        raw = (
            '{"verdicts": ['
            '{"intent_comment_id": 1, "outcome": "superseded", '
            ' "by_intent_comment_id": 2, "narrative": "x"},'
            '{"intent_comment_id": 2, "outcome": "superseded", '
            ' "by_intent_comment_id": 3, "narrative": "y"},'
            '{"intent_comment_id": 3, "outcome": "superseded", '
            ' "by_intent_comment_id": 1, "narrative": "z"}'
            "]}"
        )
        _, errors = _parse_rescope_verdicts(raw, [_intent(1), _intent(2), _intent(3)])
        assert any("supersedence graph has a cycle" in e for e in errors)

    def test_long_chain_no_cycle_ok(self) -> None:
        # 1 → 2 → 3 (no cycle)
        raw = (
            '{"verdicts": ['
            '{"intent_comment_id": 1, "outcome": "superseded", '
            ' "by_intent_comment_id": 2, "narrative": "x"},'
            '{"intent_comment_id": 2, "outcome": "superseded", '
            ' "by_intent_comment_id": 3, "narrative": "y"},'
            '{"intent_comment_id": 3, "outcome": "honored"}'
            "]}"
        )
        verdicts, errors = _parse_rescope_verdicts(
            raw, [_intent(1), _intent(2), _intent(3)]
        )
        assert errors == []
        assert len(verdicts) == 3

    def test_empty_intents_and_empty_verdicts(self) -> None:
        # Degenerate: no intents, no verdicts → ok.
        verdicts, errors = _parse_rescope_verdicts('{"verdicts": []}', [])
        assert errors == []
        assert verdicts == []

    def test_falsy_ops_field_not_silently_coerced(self) -> None:
        # codex P2 on slice 2: ``"ops": {}`` is malformed (a bare
        # mapping, not a sequence).  Earlier ``or ()`` would have
        # collapsed to () and bypassed the ctor check — now the
        # mapping reaches IntentVerdict's __post_init__ which
        # rejects ``isinstance(ops, Mapping)``.
        raw = (
            '{"verdicts": [{"intent_comment_id": 1, "outcome": "honored", "ops": {}}]}'
        )
        _, errors = _parse_rescope_verdicts(raw, [_intent(1)])
        assert any("sequence of op mappings" in e for e in errors)

    def test_falsy_affected_task_ids_bare_string_rejected(self) -> None:
        # codex P2 on slice 2: ``"affected_task_ids": ""`` is a bare
        # string; was silently collapsed to () before, now rejected.
        raw = (
            '{"verdicts": [{"intent_comment_id": 1, "outcome": "honored", '
            '"affected_task_ids": ""}]}'
        )
        _, errors = _parse_rescope_verdicts(raw, [_intent(1)])
        assert any("not a bare str" in e for e in errors)

    def test_null_ops_captured_as_parse_error(self) -> None:
        # JSON null reaches the ctor as None — ctor's tuple(None)
        # raises TypeError, which we capture into errors rather than
        # propagating.
        raw = (
            '{"verdicts": [{"intent_comment_id": 1, "outcome": "honored", '
            '"ops": null}]}'
        )
        _, errors = _parse_rescope_verdicts(raw, [_intent(1)])
        assert len(errors) > 0
        # The exact message comes from the ctor's tuple() call —
        # don't pin its phrasing, just verify the path is captured.
        assert any("verdicts[0]" in e for e in errors)

    def test_absent_ops_defaults_to_empty(self) -> None:
        # Absent ``ops`` field is the documented "no ops" case —
        # default ``()`` is the right value, not an error.
        raw = '{"verdicts": [{"intent_comment_id": 1, "outcome": "honored"}]}'
        verdicts, errors = _parse_rescope_verdicts(raw, [_intent(1)])
        assert errors == []
        assert verdicts[0].ops == ()
        assert verdicts[0].affected_task_ids == ()

    def test_affected_task_ids_mapping_rejected_by_ctor(self) -> None:
        # codex P2 on PR #1809: ``{"T1": ...}`` iterates as ("T1",)
        # and would pass the per-entry str check; reject mappings
        # explicitly at the ctor boundary.
        raw = (
            '{"verdicts": [{"intent_comment_id": 1, "outcome": "honored", '
            '"affected_task_ids": {"T1": null}}]}'
        )
        _, errors = _parse_rescope_verdicts(raw, [_intent(1)])
        assert any("not a mapping" in e for e in errors)

    def test_duplicate_intent_ids_in_input_batch_rejected(self) -> None:
        # codex P2 on PR #1809: collapsing intents to a set hides
        # duplicate comment_ids from coverage checking — a single
        # verdict could "cover" both silently.  Fail closed at the
        # parser boundary so the upstream coalescer bug is visible.
        raw = '{"verdicts": [{"intent_comment_id": 1, "outcome": "honored"}]}'
        _, errors = _parse_rescope_verdicts(raw, [_intent(1), _intent(1)])
        assert any("duplicate comment_id" in e and "[1]" in e for e in errors)

    def test_multiple_duplicate_intent_ids_all_reported(self) -> None:
        raw = '{"verdicts": []}'
        _, errors = _parse_rescope_verdicts(
            raw,
            [_intent(1), _intent(1), _intent(2), _intent(2), _intent(2)],
        )
        # Single error line lists ALL duplicates so the coalescer
        # bug is debuggable on sight.
        assert any(
            "duplicate comment_id" in e and "1" in e and "2" in e for e in errors
        )


class TestFindCrossOpErrors:
    def test_unknown_id_rejected(self) -> None:
        from fido.tasks import _RescopeOpKeep

        errors = _find_cross_op_errors(
            [_RescopeOpKeep(id="ghost", contributing_intents=[])],
            frozenset({"a"}),
        )
        assert any(
            "'ghost'" in e and "not in the pending snapshot" in e for e in errors
        )

    def test_duplicate_id_across_ops_rejected(self) -> None:
        from fido.tasks import _RescopeOpKeep, _RescopeOpRemove

        ops: list = [
            _RescopeOpKeep(id="a", contributing_intents=[]),
            _RescopeOpRemove(id="a", contributing_intents=[]),
        ]
        errors = _find_cross_op_errors(ops, frozenset({"a"}))
        assert any("claimed by 2 operations" in e for e in errors)

    def test_overlapping_merge_sources_rejected(self) -> None:
        # Source 'b' folded into TWO different targets — lineage would
        # land in both, contradicting "each source merges into at most
        # one target".  cross-op layer catches it the same way the
        # validator does (#1738 codex Medium).
        from fido.tasks import _RescopeOpMerge

        ops: list = [
            _RescopeOpMerge(
                target_id="a",
                sources=["b"],
                title="A",
                description="",
                contributing_intents=[],
            ),
            _RescopeOpMerge(
                target_id="c",
                sources=["b"],
                title="C",
                description="",
                contributing_intents=[],
            ),
        ]
        errors = _find_cross_op_errors(ops, frozenset({"a", "b", "c"}))
        assert any("'b'" in e and "claimed by 2 operations" in e for e in errors)

    def test_merge_source_id_validated(self) -> None:
        from fido.tasks import _RescopeOpMerge

        ops: list = [
            _RescopeOpMerge(
                target_id="a",
                sources=["ghost"],
                title="A",
                description="",
                contributing_intents=[],
            )
        ]
        errors = _find_cross_op_errors(ops, frozenset({"a"}))
        assert any(
            "'ghost'" in e and "not in the pending snapshot" in e for e in errors
        )

    def test_new_op_does_not_claim_any_id(self) -> None:
        # `new` ops mint fresh ids at apply time — they don't reference
        # the snapshot, so the cross-op claim count must skip them.
        from fido.tasks import _RescopeOpNew

        ops: list = [
            _RescopeOpNew(
                title="A", description="", type="spec", contributing_intents=[]
            )
        ]
        assert _find_cross_op_errors(ops, frozenset()) == []

    def test_split_op_claims_source_id(self) -> None:
        from fido.tasks import _RescopeOpSplit, _RescopeOpSplitChild

        ops: list = [
            _RescopeOpSplit(
                id="src",
                children=[_RescopeOpSplitChild(title="C", description="")],
                contributing_intents=[],
            )
        ]
        # Unknown source id.
        errors = _find_cross_op_errors(ops, frozenset())
        assert any("'src'" in e and "not in the pending snapshot" in e for e in errors)


# ── _apply_reorder ────────────────────────────────────────────────────────────


class TestApplyReorder:
    def _t(
        self,
        task_id: str,
        title: str,
        task_type: str = "spec",
        status: str = "pending",
        description: str = "",
    ) -> dict:
        t: dict = {
            "id": task_id,
            "title": title,
            "type": task_type,
            "status": status,
            "description": description,
        }
        return t

    def _item(
        self, task_id: str, title: str = "", description: str | None = None
    ) -> dict:
        d: dict = {"id": task_id, "title": title}
        if description is not None:
            d["description"] = description
        return d

    def test_reorders_two_tasks(self) -> None:
        current = [self._t("1", "First"), self._t("2", "Second")]
        items = [self._item("2", "Second"), self._item("1", "First")]
        result = _apply_reorder(current, items)
        assert [t["id"] for t in result] == ["1", "2"]

    def test_applies_title_from_opus_for_existing_id(self) -> None:
        # #1713: title is mutable metadata for an existing task id.
        current = [self._t("1", "Old title")]
        items = [self._item("1", "New title")]
        result = _apply_reorder(current, items)
        assert result[0]["title"] == "New title"

    def test_preserves_title_when_opus_returns_empty(self) -> None:
        # An empty/missing title is treated as "Opus didn't supply a rename"
        # — preserves the existing title rather than overwriting with "".
        current = [self._t("1", "Original title")]
        items = [self._item("1", "")]
        result = _apply_reorder(current, items)
        assert result[0]["title"] == "Original title"

    def test_preserves_title_when_opus_returns_non_string(self) -> None:
        # System-boundary guard: a non-string title from the LLM (number,
        # object, null, list) must not be persisted, since downstream code
        # calls .upper() / .startswith() on the title and would crash on the
        # next rescope.  Preserve the existing title instead.
        current = [self._t("1", "Original title")]
        for bad_title in (123, None, {"oops": True}, ["list"]):
            items = [{"id": "1", "title": bad_title}]
            result = _apply_reorder(current, items)
            assert result[0]["title"] == "Original title", (
                f"non-string title {bad_title!r} should not overwrite"
            )

    def test_preserves_title_when_opus_returns_whitespace_only(self) -> None:
        # Whitespace-only titles ("   ", "\t", "\n") normalize to empty
        # and so preserve the existing title — same semantic as a missing
        # rename.  Without this guard, Opus could blank out a task title.
        current = [self._t("1", "Original title")]
        for blank in ("   ", "\t\t", "\n", " \n\t "):
            items = [self._item("1", blank)]
            result = _apply_reorder(current, items)
            assert result[0]["title"] == "Original title", (
                f"whitespace title {blank!r} should not overwrite"
            )

    def test_normalizes_multiline_title_rewrite(self) -> None:
        # Opus can return a multiline title (containing \n) which would
        # break PR-body round-tripping (one task per markdown checkbox
        # line, parsed by seed_tasks_from_pr_body).  Rewrites go through
        # the same whitespace normalization as Tasks.add — all whitespace
        # runs collapse to single spaces.
        current = [self._t("1", "Original title")]
        items = [self._item("1", "First line\n  second line\twith\ttabs")]
        result = _apply_reorder(current, items)
        assert result[0]["title"] == "First line second line with tabs"

    def test_updates_description_from_opus(self) -> None:
        current = [self._t("1", "Task", description="old desc")]
        items = [self._item("1", "Task", description="new desc")]
        result = _apply_reorder(current, items)
        assert result[0]["description"] == "new desc"

    def test_clears_description_when_opus_sets_empty(self) -> None:
        current = [self._t("1", "Task", description="something")]
        items = [self._item("1", "Task", description="")]
        result = _apply_reorder(current, items)
        assert result[0]["description"] == ""

    def test_preserves_description_when_key_absent(self) -> None:
        current = [self._t("1", "Task", description="keep this")]
        items = [{"id": "1", "title": "Task"}]  # no description key
        result = _apply_reorder(current, items)
        assert result[0]["description"] == "keep this"

    def test_ignores_unknown_id_from_opus(self) -> None:
        current = [self._t("1", "Real task")]
        items = [self._item("999", "Ghost task"), self._item("1", "Real task")]
        result = _apply_reorder(current, items)
        assert len(result) == 1
        assert result[0]["id"] == "1"

    def test_ignores_duplicate_id_from_opus(self) -> None:
        # First occurrence wins; the second item with the same id is ignored.
        # The title from the first item is applied (#1713).
        current = [self._t("1", "Task")]
        items = [self._item("1", "Task v1"), self._item("1", "Task v2")]
        result = _apply_reorder(current, items)
        assert len([t for t in result if t["id"] == "1"]) == 1
        assert result[0]["title"] == "Task v1"

    def test_ci_tasks_always_first(self) -> None:
        current = [
            self._t("1", "Spec task"),
            self._t("2", "CI failure", task_type="ci"),
        ]
        items = [self._item("1", "Spec task"), self._item("2", "CI failure")]
        result = _apply_reorder(current, items)
        assert result[0]["id"] == "2"
        assert result[1]["id"] == "1"

    def test_in_progress_task_kept_when_opus_excludes_it(self) -> None:
        """#1357: Opus-omitting an in-progress task must NOT mark it
        completed.  The omission is treated as keep-as-is — only an
        explicit worker-turn outcome may complete a task."""
        current = [
            self._t("1", "Active task", status="in_progress"),
            self._t("2", "Spec task"),
        ]
        original_ids = frozenset({"1", "2"})
        items = [self._item("2", "Spec task")]
        result = _apply_reorder(current, items, original_ids)
        task1 = next(t for t in result if t["id"] == "1")
        assert task1["status"] == "in_progress"
        assert task1["title"] == "Active task"

    def test_completed_tasks_preserved_at_end(self) -> None:
        current = [
            self._t("1", "Pending"),
            self._t("2", "Done", status="completed"),
        ]
        items = [self._item("1", "Pending")]
        result = _apply_reorder(current, items)
        assert result[-1]["id"] == "2"
        assert result[-1]["status"] == "completed"

    def test_newly_added_tasks_preserved(self) -> None:
        current = [self._t("1", "Original"), self._t("2", "New arrival")]
        original_ids = frozenset({"1"})  # task "2" added after snapshot
        items = [self._item("1", "Original")]
        result = _apply_reorder(current, items, original_ids)
        ids = [t["id"] for t in result]
        assert "1" in ids
        assert "2" in ids

    def test_ignores_opus_returned_id_added_after_snapshot(self) -> None:
        current = [self._t("1", "Original"), self._t("2", "New arrival")]
        original_ids = frozenset({"1"})
        items = [
            self._item("2", "Reordered new arrival", description="changed"),
            self._item("1", "Original"),
        ]

        result = _apply_reorder(current, items, original_ids)

        assert [t["id"] for t in result] == ["1", "2"]
        assert result[1]["title"] == "New arrival"
        assert result[1]["description"] == ""

    def test_keeps_pending_task_when_opus_excludes_it(self) -> None:
        """#1357: Opus-omitting a pending task must NOT mark it completed.
        Completion is a worker-turn decision; the rescope reducer can't
        infer it from omission."""
        current = [self._t("1", "Keep"), self._t("2", "Still pending")]
        original_ids = frozenset({"1", "2"})
        items = [self._item("1", "Keep")]
        result = _apply_reorder(current, items, original_ids)
        task2 = next(t for t in result if t["id"] == "2")
        assert task2["status"] == "pending"
        assert task2["title"] == "Still pending"

    def test_empty_ordered_items_preserves_completed_and_new(self) -> None:
        current = [
            self._t("1", "Completed", status="completed"),
            self._t("2", "New"),
        ]
        original_ids = frozenset({"1"})  # "2" is new
        result = _apply_reorder(current, [], original_ids)
        ids = [t["id"] for t in result]
        assert "1" in ids
        assert "2" in ids

    def test_preserves_thread_metadata(self) -> None:
        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        t = self._t("1", "Thread task", task_type="thread")
        t["thread"] = thread
        items = [self._item("1", "Thread task")]
        result = _apply_reorder([t], items)
        assert result[0]["thread"] == thread

    def test_applies_anchor_change_and_preserves_old_in_lineage(self) -> None:
        # #1714: rescope can rewrite a task's source-comment anchor for an
        # existing task id.  The previous anchor moves into
        # lineage_comment_ids so reply-back paths can still walk back to
        # the original commenter; the new anchor becomes the primary
        # comment_id reply/resolve paths read.
        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        t = self._t("1", "Thread task", task_type="thread")
        t["thread"] = thread
        items = [
            {"id": "1", "title": "Thread task", "anchor_comment_id": 99},
        ]
        result = _apply_reorder([t], items)
        new_thread = result[0]["thread"]
        assert new_thread["comment_id"] == 99
        # Old anchor preserved as origin metadata; new anchor also present.
        lineage = new_thread["lineage_comment_ids"]
        assert 42 in lineage
        assert 99 in lineage
        # Identity is the durable id, unchanged.
        assert result[0]["id"] == "1"

    def test_anchor_change_keeps_existing_lineage_intact(self) -> None:
        # If lineage already lists earlier related comments, the anchor
        # change extends rather than replaces.
        thread = {
            "repo": "a/b",
            "pr": 1,
            "comment_id": 42,
            "lineage_comment_ids": [10, 42],
        }
        t = self._t("1", "Thread task", task_type="thread")
        t["thread"] = thread
        items = [{"id": "1", "title": "Thread task", "anchor_comment_id": 99}]
        result = _apply_reorder([t], items)
        lineage = result[0]["thread"]["lineage_comment_ids"]
        assert lineage == [10, 42, 99]

    def test_anchor_change_takes_precedence_over_text_rewrite(self) -> None:
        # The model carries one op per task per batch.  When item asks for
        # both anchor and text changes, the adapter emits RewriteAnchor
        # (anchor is structural — it re-targets the reply destination).
        # Title/description changes ride the next rescope iteration.
        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        t = self._t("1", "Old title", task_type="thread", description="old desc")
        t["thread"] = thread
        items = [
            {
                "id": "1",
                "title": "New title",
                "description": "new desc",
                "anchor_comment_id": 99,
            },
        ]
        result = _apply_reorder([t], items)
        assert result[0]["thread"]["comment_id"] == 99
        # Title/description deferred; existing values still in place.
        assert result[0]["title"] == "Old title"
        assert result[0]["description"] == "old desc"

    def test_anchor_unchanged_skips_lineage_mutation(self) -> None:
        # If the proposed anchor equals the existing one, no thread mutation
        # happens — the apply path stays a KeepTask / RewriteTask only.
        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        t = self._t("1", "Thread task", task_type="thread")
        t["thread"] = thread
        items = [{"id": "1", "title": "Thread task", "anchor_comment_id": 42}]
        result = _apply_reorder([t], items)
        assert result[0]["thread"] == thread

    def test_anchor_on_non_thread_task_falls_through_to_text_rewrite(self) -> None:
        # codex on #1731: a spec task has no thread metadata; an
        # anchor_comment_id from Opus on it is garbage and must not
        # suppress a legitimate title/description rewrite.  Without this
        # gate the elif chain would emit RewriteAnchor against a task
        # with no thread to update, dropping the text change silently.
        current = [self._t("1", "Old title", description="old desc")]  # spec, no thread
        items = [
            {
                "id": "1",
                "title": "New title",
                "description": "new desc",
                "anchor_comment_id": 99,
            },
        ]
        result = _apply_reorder(current, items)
        assert result[0]["title"] == "New title"
        assert result[0]["description"] == "new desc"
        assert "thread" not in result[0]

    def test_invalid_anchor_id_falls_through_to_text_rewrite(self) -> None:
        # codex on #1731: GitHub comment ids are positive non-bool ints.
        # Reject 0, negatives, booleans (which inherit from int in Python),
        # and non-int values.  Falling through means the text rewrite still
        # applies and a bogus anchor is never persisted.
        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        t = self._t("1", "Old title", task_type="thread", description="old")
        t["thread"] = thread
        for bad_anchor in (True, False, 0, -1, "99", 1.0, None):
            items = [
                {
                    "id": "1",
                    "title": "New title",
                    "description": "new",
                    "anchor_comment_id": bad_anchor,
                },
            ]
            result = _apply_reorder([dict(t)], items)
            assert result[0]["thread"]["comment_id"] == 42, (
                f"bogus anchor {bad_anchor!r} should not overwrite"
            )
            assert result[0]["title"] == "New title", (
                f"bogus anchor {bad_anchor!r} should not suppress text rewrite"
            )

    def test_anchor_change_drops_stale_per_comment_thread_fields(self) -> None:
        # codex on #1731: url, author, path, line, diff_hunk, lineage_key
        # all describe the OLD anchor.  Once the anchor moves they no
        # longer apply — drop them so worker code can't read stale data
        # (wrong URL, mis-attributed author).  Lane-level fields (repo,
        # pr, comment_type) and the lineage list survive.  comment_type
        # in particular is preserved because _notify_thread_change reads
        # it to choose the GitHub API; defaulting it to 'issues' would
        # silently drop review-thread notifications (codex #2 on #1731).
        thread = {
            "repo": "a/b",
            "pr": 1,
            "comment_id": 42,
            "comment_type": "pulls",
            "url": "https://github.com/a/b/pull/1#discussion_r42",
            "author": "old-commenter",
            "path": "x.py",
            "line": 10,
            "diff_hunk": "@@",
            "lineage_key": "pulls:a/b:1:thread:42",
            "lineage_comment_ids": [42],
        }
        t = self._t("1", "Thread task", task_type="thread")
        t["thread"] = thread
        items = [{"id": "1", "title": "Thread task", "anchor_comment_id": 99}]
        result = _apply_reorder([t], items)
        new_thread = result[0]["thread"]
        assert new_thread["comment_id"] == 99
        assert new_thread["repo"] == "a/b"
        assert new_thread["pr"] == 1
        assert new_thread["comment_type"] == "pulls"
        assert new_thread["lineage_comment_ids"] == [42, 99]
        for stale in ("url", "author", "path", "line", "diff_hunk", "lineage_key"):
            assert stale not in new_thread, (
                f"{stale} described the old anchor and must be dropped"
            )

    def test_materializer_normalizes_anchor_comparison(self) -> None:
        # codex on #1731: a no-op rescope (Opus's anchor matches the
        # existing one, expressed as int) against a legacy '42'-string
        # anchor in tasks.json must NOT trip the re-anchor path; without
        # int-normalized comparison we'd drop url/author/etc. metadata
        # for an anchor change that didn't actually happen.
        thread = {
            "repo": "a/b",
            "pr": 1,
            "comment_id": "42",  # legacy string form
            "url": "https://github.com/a/b/pull/1#discussion_r42",
            "author": "commenter",
        }
        t = self._t("1", "Thread task", task_type="thread")
        t["thread"] = thread
        items = [{"id": "1", "title": "Thread task", "anchor_comment_id": 42}]
        result = _apply_reorder([t], items)
        # Thread metadata stays put — no anchor change happened.
        assert result[0]["thread"] == thread

    def test_merge_folds_source_lineage_into_target(self) -> None:
        # #1717: MergeTasks(target, sources, new_title, new_description)
        # folds every source's lineage_comments + source_comment into
        # target's lineage_comments — no origin lost.  Sources are closed
        # by their own CompleteTask items in the same batch.
        thread_a = {"repo": "r/r", "pr": 1, "comment_id": 100}
        thread_b = {
            "repo": "r/r",
            "pr": 1,
            "comment_id": 200,
            "lineage_comment_ids": [50, 200],
        }
        thread_c = {
            "repo": "r/r",
            "pr": 1,
            "comment_id": 300,
            "lineage_comment_ids": [300, 250],
        }
        a = self._t("a", "Task A", task_type="thread")
        a["thread"] = thread_a
        b = self._t("b", "Task B", task_type="thread")
        b["thread"] = thread_b
        c = self._t("c", "Task C", task_type="thread")
        c["thread"] = thread_c
        items = [
            {
                "id": "a",
                "title": "Merged title",
                "description": "merged scope",
                "merge_sources": ["b", "c"],
            },
            {"id": "b", "title": "Task B", "status": "completed"},
            {"id": "c", "title": "Task C", "status": "completed"},
        ]
        result = _apply_reorder([a, b, c], items)
        target = next(t for t in result if t["id"] == "a")
        # Target keeps its own anchor and pending status; gets merged
        # title/desc from the explicit payload; absorbs sources' lineage.
        assert target["status"] == str(TaskStatus.PENDING)
        assert target["thread"]["comment_id"] == 100
        assert target["title"] == "Merged title"
        assert target["description"] == "merged scope"
        # Ordered union: a's [100] (default), then b's [50, 200], then
        # c's [300, 250] (b/c lineage_comment_ids fields explicitly).
        assert target["thread"]["lineage_comment_ids"] == [100, 50, 200, 300, 250]
        # Sources are closed by their own CompleteTask items.
        for src_id in ("b", "c"):
            src = next(t for t in result if t["id"] == src_id)
            assert src["status"] == str(TaskStatus.COMPLETED)

    def test_merge_lineage_assertion_raises_on_dropped_source(self) -> None:
        # The runtime assertion catches a (hypothetical) oracle/adapter
        # divergence that drops source lineage from the target row.
        # Build a synthetic before/after pair where the target row's
        # lineage_comments doesn't contain the source's anchor.
        from fido.rocq import task_queue_rescope as oracle

        src_row = oracle.TaskRow(
            title="src",
            description="",
            kind=oracle.TaskThread(),
            status=oracle.StatusPending(),
            source_comment=200,
            lineage_comments=[200],
        )
        target_after = oracle.TaskRow(
            title="tgt",
            description="",
            kind=oracle.TaskThread(),
            status=oracle.StatusPending(),
            source_comment=100,
            lineage_comments=[100],  # missing 200 — source lineage dropped
        )
        rows_before = {1: target_after, 2: src_row}
        rows_after = {1: target_after, 2: src_row}
        merge_release = oracle.RescopeRelease(
            oracle.ReleaseACT(), oracle.MergeTasks(1, [2], "tgt", "")
        )
        with pytest.raises(AssertionError, match="dropped source lineage"):
            _assert_merge_lineage_preserved([merge_release], rows_before, rows_after)

    def test_merge_lineage_assertion_raises_on_missing_target(self) -> None:
        # If the target row is somehow missing from the oracle output,
        # fail fast rather than silently produce a half-applied merge.
        from fido.rocq import task_queue_rescope as oracle

        src_row = oracle.TaskRow(
            title="src",
            description="",
            kind=oracle.TaskThread(),
            status=oracle.StatusPending(),
            source_comment=200,
            lineage_comments=[200],
        )
        rows_before = {2: src_row}
        rows_after: dict[int, oracle.TaskRow] = {2: src_row}  # no target
        merge_release = oracle.RescopeRelease(
            oracle.ReleaseACT(), oracle.MergeTasks(1, [2], "tgt", "")
        )
        with pytest.raises(AssertionError, match="missing from rescope output"):
            _assert_merge_lineage_preserved([merge_release], rows_before, rows_after)

    def test_auto_resolve_projection_skips_non_thread_tasks(self) -> None:
        # A task with no thread contributes zero ThreadTask entries —
        # the auto-resolve oracle only cares about review-thread
        # comments, not spec/CI work.
        spec = self._t("spec", "Spec")  # no thread dict
        assert thread_tasks_for_auto_resolve_oracle([spec]) == []

    def test_merged_target_lineage_blocks_source_thread_auto_resolve(self) -> None:
        # codex on #1738 (high): the auto-resolve oracle's projection
        # must cover every comment in a pending merged target's
        # lineage, not just the primary anchor.  Otherwise after a
        # merge the source thread auto-resolves before the merged
        # target is actually done, because no pending task carries the
        # source's comment_id as primary anymore (the source is
        # COMPLETED, the target's primary is its own anchor).
        thread_a = {"repo": "r/r", "pr": 1, "comment_id": 100}
        thread_b = {"repo": "r/r", "pr": 1, "comment_id": 200}
        a = self._t("a", "Task A", task_type="thread")
        a["thread"] = thread_a
        b = self._t("b", "Task B", task_type="thread")
        b["thread"] = thread_b
        items = [
            {"id": "a", "title": "Merged", "merge_sources": ["b"]},
            {"id": "b", "title": "Task B", "status": "completed"},
        ]
        result = _apply_reorder([a, b], items)
        oracle_tasks = thread_tasks_for_auto_resolve_oracle(result)
        # Comment 200 (the source's anchor) appears as a PENDING
        # ThreadTask via the merged target's lineage — the auto-resolve
        # oracle will see it and refuse to resolve thread 200 until the
        # merged target completes.
        from fido.rocq import thread_auto_resolve as resolve_oracle

        pending_for_200 = [
            t
            for t in oracle_tasks
            if t.thread_task_comment == 200
            and isinstance(t.thread_task_status, resolve_oracle.StatusPending)
        ]
        assert pending_for_200, (
            "merged target's lineage must surface as a pending ThreadTask "
            "for each absorbed source's anchor — otherwise auto-resolve "
            "fires prematurely on the source thread"
        )

    def test_merge_apply_runs_lineage_preservation_assertion(self) -> None:
        # codex on #1738 (medium): the Rocq predicate
        # merge_preserves_source_lineage is asserted at runtime for
        # every MergeTasks op the adapter emits — so any divergence
        # between the executable model predicate and the materialized
        # output fails closed instead of silently dropping a source
        # comment.  This test exercises the happy path; the predicate
        # itself is exercised by test_merge_uses_oracle_predicate_to_*.
        thread_a = {"repo": "r/r", "pr": 1, "comment_id": 100}
        thread_b = {
            "repo": "r/r",
            "pr": 1,
            "comment_id": 200,
            "lineage_comment_ids": [50, 200],
        }
        a = self._t("a", "Task A", task_type="thread")
        a["thread"] = thread_a
        b = self._t("b", "Task B", task_type="thread")
        b["thread"] = thread_b
        items = [
            {"id": "a", "title": "Merged", "merge_sources": ["b"]},
            {"id": "b", "title": "Task B", "status": "completed"},
        ]
        # Should not raise; if assertion fired we'd get AssertionError.
        result = _apply_reorder([a, b], items)
        target = next(t for t in result if t["id"] == "a")
        assert 50 in target["thread"]["lineage_comment_ids"]
        assert 200 in target["thread"]["lineage_comment_ids"]

    def test_merge_with_unhashable_source_does_not_crash(self) -> None:
        # codex on #1738: _apply_reorder must not raise TypeError when
        # called with a malformed merge_sources list (e.g. nested list /
        # dict).  In production the validator rejects this atomically;
        # tests bypass the validator, so the adapter has its own
        # isinstance(str) guard before the dict membership check.
        current = [self._t("a", "Task A"), self._t("b", "Task B")]
        items = [
            {
                "id": "a",
                "title": "Merged",
                "merge_sources": [["nested"], {"d": 1}, "b"],
            },
            {"id": "b", "title": "B", "status": "completed"},
        ]
        # Should fall through to the merge with only the valid source.
        result = _apply_reorder(current, items)
        target = next(t for t in result if t["id"] == "a")
        b_completed = next(t for t in result if t["id"] == "b")
        assert target["title"] == "Merged"
        assert b_completed["status"] == str(TaskStatus.COMPLETED)

    def test_merge_uses_oracle_predicate_to_prove_no_lineage_lost(self) -> None:
        # The Rocq model's merge_preserves_source_lineage predicate is
        # extracted to Python; check it returns True after a real merge.
        from fido.rocq import task_queue_rescope as oracle

        a_row = oracle.TaskRow(
            title="A",
            description="",
            kind=oracle.TaskThread(),
            status=oracle.StatusPending(),
            source_comment=100,
            lineage_comments=[100],
        )
        b_row = oracle.TaskRow(
            title="B",
            description="",
            kind=oracle.TaskThread(),
            status=oracle.StatusPending(),
            source_comment=200,
            lineage_comments=[50, 200],
        )
        rows_before = {1: a_row, 2: b_row}
        merge_op = oracle.MergeTasks(1, [2], "Merged", "")
        # Extracted return shape is the Coq triple ((rows, pending), completed).
        ((rows_after, _), _) = oracle.apply_rescope_op(
            merge_op, 1, a_row, rows_before, [], []
        )
        target_after = rows_after[1]
        # Predicate confirms sources' lineage + anchor are present in target.
        assert oracle.merge_preserves_source_lineage([2], rows_before, target_after)
        # Sanity: the merged lineage matches what we expect.
        assert target_after.lineage_comments == [100, 50, 200]

    def test_split_closes_source_and_spawns_children_inheriting_lineage(self) -> None:
        # #1718: SplitTask(source, [SplitChild(...)]) closes the source
        # row and spawns N children that inherit the source's
        # lineage_comments + source_comment verbatim — reply paths still
        # reach every original commenter via the children's threads.
        from fido.tasks import _apply_reorder as apply_reorder

        thread_src = {
            "repo": "r/r",
            "pr": 1,
            "comment_id": 100,
            "comment_type": "pulls",
            "url": "https://example/100",
            "lineage_comment_ids": [100, 50],
        }
        src = self._t("src", "Original task", task_type="thread")
        src["thread"] = thread_src
        items = [
            {
                "id": "src",
                "title": "Original task",
                "split_targets": [
                    {"title": "Child A", "description": "first half"},
                    {"title": "Child B", "description": "second half"},
                ],
            }
        ]
        result = apply_reorder([src], items)
        # Source closed; two new pending children inherit thread metadata.
        source_after = next(t for t in result if t["id"] == "src")
        assert source_after["status"] == str(TaskStatus.COMPLETED)
        children = [t for t in result if t["id"] != "src"]
        assert len(children) == 2
        titles = [c["title"] for c in children]
        assert titles == ["Child A", "Child B"]
        for child, expected_desc in zip(children, ["first half", "second half"]):
            assert child["status"] == str(TaskStatus.PENDING)
            assert child["description"] == expected_desc
            assert child["type"] == "thread"
            assert child["thread"]["comment_id"] == 100
            assert child["thread"]["comment_type"] == "pulls"
            # Lineage comments inherited verbatim — every original
            # commenter is still reachable from each child's thread.
            assert child["thread"]["lineage_comment_ids"] == [100, 50]

    def test_split_lineage_assertion_raises_on_missing_child(self) -> None:
        # Mirror of test_merge_lineage_assertion_raises_on_missing_target:
        # if a SplitTask op's children aren't materialised in rows_after,
        # fail closed instead of silently producing an incomplete split.
        from fido.rocq import task_queue_rescope as oracle
        from fido.tasks import _assert_split_lineage_preserved

        src_row = oracle.TaskRow(
            title="src",
            description="",
            kind=oracle.TaskThread(),
            status=oracle.StatusPending(),
            source_comment=100,
            lineage_comments=[100, 50],
        )
        rows_before = {1: src_row}
        # rows_after deliberately missing the child row keyed at 2.
        rows_after: dict[int, oracle.TaskRow] = {1: src_row}
        split_release = oracle.RescopeRelease(
            oracle.ReleaseACT(),
            oracle.SplitTask(
                1,
                [
                    oracle.SplitChild(
                        child_task=2, child_title="A", child_description=""
                    )
                ],
            ),
        )
        with pytest.raises(AssertionError, match="dropped child lineage"):
            _assert_split_lineage_preserved([split_release], rows_before, rows_after)

    def test_split_lineage_assertion_raises_on_missing_source(self) -> None:
        # Mirror of merge missing-target: if the SplitTask source row
        # vanished from rows_before, the predicate has no template to
        # compare children against — fail fast.
        from fido.rocq import task_queue_rescope as oracle
        from fido.tasks import _assert_split_lineage_preserved

        rows_before: dict[int, oracle.TaskRow] = {}
        rows_after: dict[int, oracle.TaskRow] = {}
        split_release = oracle.RescopeRelease(
            oracle.ReleaseACT(),
            oracle.SplitTask(
                1,
                [
                    oracle.SplitChild(
                        child_task=2, child_title="A", child_description=""
                    )
                ],
            ),
        )
        with pytest.raises(AssertionError, match="missing from rescope input"):
            _assert_split_lineage_preserved([split_release], rows_before, rows_after)

    def test_split_uses_oracle_predicate_to_prove_no_lineage_lost(self) -> None:
        # The Rocq model's split_preserves_source_lineage predicate is
        # extracted to Python; check it returns True after a real split.
        from fido.rocq import task_queue_rescope as oracle

        src_row = oracle.TaskRow(
            title="src",
            description="",
            kind=oracle.TaskThread(),
            status=oracle.StatusPending(),
            source_comment=100,
            lineage_comments=[100, 50],
        )
        rows_before = {1: src_row}
        split_op = oracle.SplitTask(
            1,
            [
                oracle.SplitChild(child_task=2, child_title="A", child_description=""),
                oracle.SplitChild(child_task=3, child_title="B", child_description=""),
            ],
        )
        ((rows_after, _), _) = oracle.apply_rescope_op(
            split_op, 1, src_row, rows_before, [], []
        )
        # Predicate confirms every child carries source's lineage + anchor.
        assert oracle.split_preserves_source_lineage([2, 3], src_row, rows_after)
        # Sanity: the children's lineage matches the source verbatim.
        assert rows_after[2].lineage_comments == [100, 50]
        assert rows_after[2].source_comment == 100
        assert rows_after[3].lineage_comments == [100, 50]

    def test_split_source_completion_notification_is_suppressed(self) -> None:
        # Mirror of the merge-source suppression: a split source's
        # status flips to COMPLETED but the work moved into the
        # children, not "Fido finished your work" — the auto reply
        # would mislead the original commenter.  Per-source decisions
        # are owned by the reply-back filter epic (#1256 / #1723 / #1724).
        from fido.tasks import _split_source_ids

        items = [
            {
                "id": "src",
                "split_targets": [
                    {"title": "Child A"},
                    {"title": "Child B"},
                ],
            },
            # Sources without split_targets aren't suppressed.
            {"id": "other", "title": "unrelated"},
        ]
        assert _split_source_ids(items) == {"src"}

    def test_split_with_post_snapshot_child_ids_distinct_from_existing(self) -> None:
        # Newly-allocated child task ids must not collide with the
        # source's id or any other task id in the queue.  Allocation is
        # timestamp-random per call.
        from fido.tasks import _apply_reorder as apply_reorder

        src = self._t("src", "Original")
        other = self._t("other-1234567890-0001", "Sibling")
        items = [
            {
                "id": "src",
                "split_targets": [{"title": "Child"}],
            },
        ]
        result = apply_reorder([src, other], items)
        ids = [t["id"] for t in result]
        # Source kept its id (now closed); sibling preserved; child
        # received a fresh id distinct from both.
        assert "src" in ids
        assert "other-1234567890-0001" in ids
        child_ids = [i for i in ids if i not in {"src", "other-1234567890-0001"}]
        assert len(child_ids) == 1
        assert child_ids[0] not in {"src", "other-1234567890-0001"}

    def test_split_children_and_new_tasks_have_distinct_ids(self) -> None:
        # Split-child ids and ``_make_new_tasks_from_opus`` ids both
        # come from ``uuid.uuid7()`` (74 bits of entropy per ms), so
        # collisions across the two generators — and with existing
        # on-disk ids — are statistically impossible.  Smoke-check
        # that a batch combining a split with a null-id new task
        # produces all-distinct ids.
        from fido.tasks import _apply_reorder as apply_reorder

        src = self._t("src", "Original")
        items = [
            {"id": "src", "split_targets": [{"title": "Child A"}]},
            {"id": None, "title": "Brand new"},
        ]
        result = apply_reorder([src], items)
        ids = [t["id"] for t in result]
        assert len(ids) == len(set(ids)), f"duplicate ids in result: {ids}"
        # The split source kept its id (now closed); two fresh ids
        # were minted (one for the child, one for the new task).
        assert "src" in ids
        fresh_ids = {i for i in ids if i != "src"}
        assert len(fresh_ids) == 2

    def test_allocate_split_child_ids_shares_created_at_across_children(self) -> None:
        # created_at is captured once per call so the apply and
        # verify passes (which both hit this allocator output)
        # produce identical materializations even if real-time
        # crosses a second boundary between them.  (Ids themselves
        # are UUIDv7, distinct by construction.)
        from fido.tasks import _allocate_split_child_ids

        items = [
            {
                "id": "src",
                "split_targets": [{"title": "A"}, {"title": "B"}, {"title": "C"}],
            }
        ]
        allocated = _allocate_split_child_ids(items)
        timestamps = {created_at for _id, created_at in allocated["src"]}
        assert len(timestamps) == 1, (
            "every child of one batch must share the same created_at — "
            "otherwise the divergence verifier would raise on a batch "
            "straddling a wall-clock second"
        )

    def test_rewrite_carries_contributing_intents_to_persisted_task(self) -> None:
        # #1722: contributing_intents on the op flows through the
        # apply path and lands on the resulting task dict so a future
        # classifier can read which intents drove the rewrite.
        current = [self._t("1", "Old")]
        items = [
            {
                "id": "1",
                "title": "New",
                "description": "scoped",
                "contributing_intents": [42, 99],
            }
        ]
        result = _apply_reorder(current, items)
        assert result[0]["contributing_intents"] == [42, 99]

    def test_multiple_intents_can_contribute_to_one_task(self) -> None:
        # Acceptance criteria from #1722: tests cover multiple intents
        # contributing to one task.  Two intents both attributed to a
        # single rewrite op land on the resulting task as a list.
        current = [self._t("1", "Original")]
        items = [
            {
                "id": "1",
                "title": "Combined ask",
                "description": "...",
                "contributing_intents": [101, 202, 303],
            }
        ]
        result = _apply_reorder(current, items)
        assert result[0]["contributing_intents"] == [101, 202, 303]

    def test_split_propagates_contributing_intents_to_every_child(self) -> None:
        # Acceptance criteria from #1722: split records associate the
        # source intent ids with every split child unless explicitly
        # narrowed (this leaf does not implement narrowing — children
        # always inherit).  The split source's pre-existing intents
        # also flow to every child.
        src = self._t("src", "Original", task_type="thread")
        src["thread"] = {"repo": "r/r", "pr": 1, "comment_id": 100}
        src["contributing_intents"] = [50]  # pre-existing intent on source
        items = [
            {
                "id": "src",
                "split_targets": [
                    {"title": "Child A", "description": "first"},
                    {"title": "Child B", "description": "second"},
                ],
                "contributing_intents": [777],  # split-trigger intent
            }
        ]
        result = _apply_reorder([src], items)
        children = [t for t in result if t["id"] != "src"]
        assert len(children) == 2
        for child in children:
            # Source's [50] + op's [777] union = [50, 777] in insertion
            # order.
            assert child["contributing_intents"] == [50, 777]

    def test_merge_unions_source_intents_into_target(self) -> None:
        # Sources' pre-existing contributing_intents fold into the
        # merged target alongside the merge op's own intents.
        a = self._t("a", "Task A")
        a["contributing_intents"] = [10]
        b = self._t("b", "Task B")
        b["contributing_intents"] = [20, 30]
        items = [
            {
                "id": "a",
                "title": "Merged",
                "description": "",
                "merge_sources": ["b"],
                "contributing_intents": [40],
            },
            {"id": "b", "status": "completed"},
        ]
        result = _apply_reorder([a, b], items)
        target = next(t for t in result if t["id"] == "a")
        # Target's prior [10] + op's [40] + source b's [20, 30] union.
        assert target["contributing_intents"] == [10, 40, 20, 30]

    def test_new_op_carries_contributing_intents_to_new_task(self) -> None:
        # Brand-new tasks created via `new` ops keep the originating
        # intent ids so a downstream classifier can route the eventual
        # completion notification to the right commenter(s).
        current: list = []
        items = [
            {
                "id": None,
                "title": "Brand new",
                "description": "scope",
                "type": "spec",
                "contributing_intents": [555],
            }
        ]
        result = _apply_reorder(current, items)
        assert len(result) == 1
        assert result[0]["title"] == "Brand new"
        assert result[0]["contributing_intents"] == [555]

    def test_explicit_completion_marks_task_completed(self) -> None:
        # #1716: an item with status="completed" emits CompleteTask, which
        # the reducer applies as a status flip to COMPLETED.  This is the
        # only way rescope can remove a snapped task — omission still means
        # keep-as-is (#1357).
        current = [self._t("1", "Done work")]
        items = [{"id": "1", "title": "Done work", "status": "completed"}]
        result = _apply_reorder(current, items)
        assert result[0]["status"] == str(TaskStatus.COMPLETED)

    def test_explicit_completion_takes_precedence_over_text_and_anchor(
        self,
    ) -> None:
        # CompleteTask is the most structural op — it removes the task.
        # When an item asks for both completion and text/anchor changes,
        # the adapter emits CompleteTask; the metadata changes are moot
        # because the task is done.
        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        t = self._t("1", "Old title", task_type="thread", description="old desc")
        t["thread"] = thread
        items = [
            {
                "id": "1",
                "title": "New title",
                "description": "new desc",
                "anchor_comment_id": 99,
                "status": "completed",
            },
        ]
        result = _apply_reorder([t], items)
        assert result[0]["status"] == str(TaskStatus.COMPLETED)
        # Title/desc/anchor at the moment of completion are immaterial —
        # the row is done.  Existing thread metadata isn't re-anchored.
        assert result[0]["thread"]["comment_id"] == 42

    def test_omitted_task_is_kept_not_completed(self) -> None:
        # #1357 / #1716 invariant: omission ≠ deletion.  A pending task
        # Opus omits from its output survives at its current status.
        # Only an explicit status="completed" item triggers removal.
        current = [self._t("1", "Keep me"), self._t("2", "Also keep")]
        original_ids = frozenset({"1", "2"})
        items = [self._item("1", "Keep me")]  # task "2" omitted
        result = _apply_reorder(current, items, original_ids)
        task2 = next(t for t in result if t["id"] == "2")
        assert task2["status"] == str(TaskStatus.PENDING)
        assert task2["title"] == "Also keep"

    def test_explicit_completion_of_thread_task_fires_completed_change_record(
        self,
    ) -> None:
        # _compute_thread_changes already keys completion records on
        # result.status == COMPLETED, so the explicit completion path
        # naturally produces "kind=completed" change records that
        # reply-back consumers (#1256) will eventually filter and post.
        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        t = self._t("1", "Thread task", task_type="thread")
        t["thread"] = thread
        items = [{"id": "1", "title": "Thread task", "status": "completed"}]
        result = _apply_reorder([t], items)
        changes = _compute_thread_changes([t], result, frozenset({"1"}))
        assert len(changes) == 1
        assert changes[0]["kind"] == "completed"
        assert changes[0]["task"]["id"] == "1"

    def test_applies_duplicate_titles_at_apply_reorder_layer(self) -> None:
        # _apply_reorder is the low-level reducer; uniqueness is enforced
        # upstream by the rescope nudge loop in reorder_tasks(), not here
        # (#1713).  At this layer, whatever titles Opus proposed are applied.
        current = [self._t("1", "Alpha task"), self._t("2", "Beta task")]
        items = [self._item("1", "Shared name"), self._item("2", "Shared name")]
        result = _apply_reorder(current, items)
        assert next(t for t in result if t["id"] == "1")["title"] == "Shared name"
        assert next(t for t in result if t["id"] == "2")["title"] == "Shared name"

    def test_unique_title_rewrites_are_applied(self) -> None:
        # #1713: title rewrites for existing task ids flow through.
        current = [self._t("1", "Old A"), self._t("2", "Old B"), self._t("3", "Old C")]
        items = [
            self._item("1", "New A"),
            self._item("2", "New B"),
            self._item("3", "New C"),
        ]
        result = _apply_reorder(current, items)
        titles = {t["id"]: t["title"] for t in result}
        assert titles == {"1": "New A", "2": "New B", "3": "New C"}

    def test_fails_closed_when_runtime_result_diverges_from_oracle(self) -> None:
        # The oracle now applies title rewrites, so the "wrong" runtime
        # result must differ on something it doesn't apply — use status.
        current = [self._t("1", "Original")]
        diverged = [self._t("1", "Original", status="completed")]

        with pytest.raises(AssertionError, match="diverged"):
            _assert_rescope_matches_oracle(
                current,
                [self._item("1", "Original")],
                {"1"},
                diverged,
            )

    def test_creates_new_task_from_opus_null_id(self) -> None:
        current = [self._t("1", "Existing")]
        items = [
            self._item("1", "Existing"),
            {"title": "Brand new", "description": "details"},
        ]
        result = _apply_reorder(current, items)
        pending = [t for t in result if t.get("status") != "completed"]
        assert len(pending) == 2
        new_task = next(t for t in pending if t["id"] != "1")
        assert new_task["title"] == "Brand new"
        assert new_task["description"] == "details"
        assert new_task["status"] == "pending"

    def test_creates_new_task_from_opus_explicit_null_id(self) -> None:
        current = [self._t("1", "Existing")]
        items = [
            self._item("1", "Existing"),
            {"id": None, "title": "Null-id task", "description": ""},
        ]
        result = _apply_reorder(current, items)
        pending = [t for t in result if t.get("status") != "completed"]
        assert len(pending) == 2
        new_task = next(t for t in pending if t["id"] != "1")
        assert new_task["title"] == "Null-id task"

    def test_new_task_receives_fresh_id(self) -> None:
        current = [self._t("1", "Existing")]
        items = [{"title": "Fresh task", "description": ""}]
        result = _apply_reorder(current, items)
        new_task = next(
            t for t in result if t["id"] != "1" and t.get("status") != "completed"
        )
        assert new_task["id"] != "1"
        assert len(new_task["id"]) > 0

    def test_new_ci_task_sorted_before_non_ci(self) -> None:
        current = [self._t("1", "Spec task")]
        items = [
            {"title": "New CI fix", "type": "ci", "description": ""},
            self._item("1", "Spec task"),
        ]
        result = _apply_reorder(current, items)
        pending = [t for t in result if t.get("status") != "completed"]
        assert pending[0]["title"] == "New CI fix"
        assert pending[0]["type"] == "ci"

    def test_new_task_blank_title_ignored(self) -> None:
        current = [self._t("1", "Existing")]
        items = [
            {"title": "", "description": "empty title"},
            self._item("1", "Existing"),
        ]
        result = _apply_reorder(current, items)
        pending = [t for t in result if t.get("status") != "completed"]
        assert len(pending) == 1
        assert pending[0]["id"] == "1"

    def test_string_id_not_in_snapshot_still_ignored(self) -> None:
        # Existing test: Opus returning a made-up string ID is still ignored.
        current = [self._t("1", "Real task")]
        items = [{"id": "made-up-id", "title": "Ghost"}, self._item("1", "Real task")]
        result = _apply_reorder(current, items)
        pending = [t for t in result if t.get("status") != "completed"]
        assert len(pending) == 1
        assert pending[0]["id"] == "1"


# ── _make_new_tasks_from_opus ─────────────────────────────────────────────────


class TestMakeNewTasksFromOpus:
    def test_returns_empty_for_all_known_ids(self) -> None:
        items = [{"id": "1", "title": "Existing"}]
        result = _make_new_tasks_from_opus(items, frozenset({"1"}))
        assert result == []

    def test_returns_empty_for_unknown_string_id(self) -> None:
        items = [{"id": "unknown", "title": "Ghost"}]
        result = _make_new_tasks_from_opus(items, frozenset({"1"}))
        assert result == []

    def test_creates_task_for_absent_id(self) -> None:
        items = [{"title": "New task", "description": "detail"}]
        result = _make_new_tasks_from_opus(items, frozenset())
        assert len(result) == 1
        assert result[0]["title"] == "New task"
        assert result[0]["description"] == "detail"
        assert result[0]["status"] == "pending"

    def test_creates_task_for_null_id(self) -> None:
        items = [{"id": None, "title": "Null id task", "description": ""}]
        result = _make_new_tasks_from_opus(items, frozenset())
        assert len(result) == 1
        assert result[0]["title"] == "Null id task"

    def test_assigns_fresh_id(self) -> None:
        items = [{"title": "Task A", "description": ""}]
        result = _make_new_tasks_from_opus(items, frozenset())
        assert result[0]["id"]
        assert result[0]["id"] not in ("", None)

    def test_uses_spec_type_by_default(self) -> None:
        items = [{"title": "Task", "description": ""}]
        result = _make_new_tasks_from_opus(items, frozenset())
        assert result[0]["type"] == "spec"

    def test_respects_specified_type(self) -> None:
        items = [{"title": "CI task", "type": "ci", "description": ""}]
        result = _make_new_tasks_from_opus(items, frozenset())
        assert result[0]["type"] == "ci"

    def test_skips_blank_title(self) -> None:
        # #1844 codex P2: slot is preserved at null-id position so the
        # interleave doesn't shift later items into earlier positions.
        # Blank-title items materialize as ``None`` rather than vanishing.
        items = [{"title": "", "description": "no title"}]
        result = _make_new_tasks_from_opus(items, frozenset())
        assert result == [None]

    def test_skips_whitespace_only_title(self) -> None:
        items = [{"title": "   ", "description": "no title"}]
        result = _make_new_tasks_from_opus(items, frozenset())
        assert result == [None]

    def test_multiple_new_tasks(self) -> None:
        items = [
            {"title": "First new", "description": ""},
            {"title": "Second new", "description": ""},
        ]
        result = _make_new_tasks_from_opus(items, frozenset())
        assert len(result) == 2
        assert result[0]["title"] == "First new"
        assert result[1]["title"] == "Second new"

    def test_mixed_known_and_new(self) -> None:
        items = [
            {"id": "1", "title": "Known"},
            {"title": "New", "description": ""},
        ]
        result = _make_new_tasks_from_opus(items, frozenset({"1"}))
        assert len(result) == 1
        assert result[0]["title"] == "New"

    def test_dedups_against_post_snapshot_thread_task(self) -> None:
        """#1337 regression: when create_task added a thread task during the
        Opus call (a new id appearing in current that's NOT in the snapshot
        and whose lineage_comment_ids overlap a rescope intent), Opus's null-id
        item for that same intent is a duplicate and must be dropped.
        """
        snapshot_ids = frozenset({"orig-1"})
        current = [
            {"id": "orig-1", "title": "Original", "status": "pending"},
            {
                "id": "post-snapshot-2",
                "title": "Replace MagicMock with _FakeDispatcher",
                "status": "in_progress",
                "type": "thread",
                "thread": {
                    "comment_id": 4371338003,
                    "lineage_comment_ids": [4371338003],
                },
            },
        ]
        intents = [
            RescopeIntent(
                comment_id=4371338003,
                change_request="Replace MagicMock(spec=Dispatcher) with hand-rolled _FakeDispatcher",
                timestamp="2026-05-04T13:15:44Z",
            ),
        ]
        items = [
            {"id": "orig-1", "title": "Original"},
            {
                "title": "Replace MagicMock(spec=Dispatcher) with _FakeDispatcher",
                "description": "Step-by-step rewrite",
            },
        ]
        result = _make_new_tasks_from_opus(
            items, snapshot_ids, current=current, intents=intents
        )
        # #1844 codex P2: the dropped null-id slot is represented as
        # ``None`` rather than absent so the interleave preserves
        # positional alignment for later new tasks.
        assert result == [None], "covered intent must drop the duplicate (#1337)"

    def test_does_not_dedup_when_lineage_does_not_match_intent(self) -> None:
        """When no post-snapshot thread task covers the intent, Opus's null-id
        item is genuinely new and must be kept."""
        snapshot_ids = frozenset({"orig-1"})
        current = [
            {"id": "orig-1", "title": "Original", "status": "pending"},
            {
                "id": "post-2",
                "title": "Some unrelated thread task",
                "status": "pending",
                "thread": {"comment_id": 999, "lineage_comment_ids": [999]},
            },
        ]
        intents = [
            RescopeIntent(
                comment_id=4371338003,
                change_request="other intent",
                timestamp="2026-05-04T13:15:44Z",
            ),
        ]
        items = [{"title": "Genuinely new spec", "description": ""}]
        result = _make_new_tasks_from_opus(
            items, snapshot_ids, current=current, intents=intents
        )
        assert len(result) == 1
        assert result[0] is not None
        assert result[0]["title"] == "Genuinely new spec"

    def test_thread_anchor_derived_from_contributing_intent(self) -> None:
        # #1843: a brand-new task spawned by rescope carries the
        # originating intent's PR-context as its ``thread`` anchor so
        # reply-back can route follow-ups back to the source comment.
        from fido.tasks import _make_new_tasks_from_opus

        intent = RescopeIntent(
            change_request="add hello",
            comment_id=4489194431,
            timestamp="2026-05-19T15:13:00+00:00",
            repo="FidoCanCode/home",
            pr_number=1842,
        )
        ordered = [
            {
                "title": "Add 'hello' to the readme",
                "description": "exercise",
                "type": "spec",
                "contributing_intents": [4489194431],
            }
        ]
        result = _make_new_tasks_from_opus(ordered, frozenset(), intents=[intent])
        assert len(result) == 1
        assert result[0] is not None
        assert result[0]["thread"] == {
            "repo": "FidoCanCode/home",
            "pr": 1842,
            "comment_id": 4489194431,
        }

    def test_thread_anchor_absent_when_intent_lacks_pr_context(self) -> None:
        # Legacy intents (pre-#1843) don't carry repo/pr_number — the
        # task gets created without a thread anchor.  Better than a
        # half-populated thread that breaks downstream lookups.
        from fido.tasks import _make_new_tasks_from_opus

        intent = RescopeIntent(
            change_request="orphan",
            comment_id=99,
            timestamp="2026-05-19T15:13:00+00:00",
        )
        ordered = [
            {
                "title": "Orphan task",
                "description": "no pr context",
                "type": "spec",
                "contributing_intents": [99],
            }
        ]
        result = _make_new_tasks_from_opus(ordered, frozenset(), intents=[intent])
        assert len(result) == 1
        assert result[0] is not None
        assert "thread" not in result[0]

    def test_dropped_slot_preserves_alignment_for_later_new_tasks(self) -> None:
        # #1844 codex P2: a dropped duplicate slot at the front must
        # not shift later kept new tasks into the dropped position.
        # ``_apply_reorder`` walks the aligned list positionally and
        # this test pins the per-slot alignment so the interleave can
        # rely on it.
        snapshot_ids = frozenset({"orig-1"})
        current = [
            {"id": "orig-1", "title": "Original", "status": "pending"},
            {
                "id": "post-snapshot-2",
                "title": "Duplicate of the covered intent",
                "status": "in_progress",
                "type": "thread",
                "thread": {
                    "comment_id": 4371338003,
                    "lineage_comment_ids": [4371338003],
                },
            },
        ]
        intents = [
            RescopeIntent(
                comment_id=4371338003,
                change_request="covered by the post-snapshot task",
                timestamp="2026-05-04T13:15:44Z",
            ),
        ]
        # Two null-id items: the first matches the covered intent and
        # is dropped, the second is a genuine new task.  Result must
        # be [None, dict] — same length as the count of null-id items.
        items = [
            {"title": "Duplicate"},
            {"title": "Genuine new"},
        ]
        result = _make_new_tasks_from_opus(
            items, snapshot_ids, current=current, intents=intents
        )
        assert len(result) == 2
        assert result[0] is None
        assert result[1] is not None
        assert result[1]["title"] == "Genuine new"


# ── _find_duplicate_titles ────────────────────────────────────────────────────


class TestFindDuplicateTitles:
    def _item(self, title: str) -> dict:
        # Null-id items (new tasks) — no existing-title fallback applies, so
        # the effective title is just the normalized proposed value.
        return {"id": None, "title": title}

    def test_returns_empty_when_all_unique(self) -> None:
        items = [self._item("A"), self._item("B"), self._item("C")]
        assert _find_duplicate_titles(items, {}) == []

    def test_returns_duplicate_title(self) -> None:
        items = [self._item("Same"), self._item("Other"), self._item("Same")]
        assert _find_duplicate_titles(items, {}) == ["Same"]

    def test_each_duplicate_listed_once(self) -> None:
        items = [self._item("X"), self._item("X"), self._item("X")]
        assert _find_duplicate_titles(items, {}) == ["X"]

    def test_multiple_distinct_duplicates(self) -> None:
        items = [
            self._item("A"),
            self._item("B"),
            self._item("A"),
            self._item("B"),
        ]
        assert _find_duplicate_titles(items, {}) == ["A", "B"]

    def test_ignores_empty_titles(self) -> None:
        items = [{"id": None, "title": ""}, {"id": None, "title": ""}]
        assert _find_duplicate_titles(items, {}) == []

    def test_ignores_missing_title_key(self) -> None:
        items = [{"id": None}, {"id": None}]
        assert _find_duplicate_titles(items, {}) == []

    def test_returns_empty_list_when_no_items(self) -> None:
        assert _find_duplicate_titles([], {}) == []

    def test_dedups_on_normalized_form_not_raw(self) -> None:
        # codex on #1729: "A\nB" and "A B" both collapse to "A B" at apply
        # time; the dedup check has to see the same normalized value or
        # they slip through the nudge as distinct and then collide on disk.
        items = [self._item("A\nB"), self._item("A B")]
        assert _find_duplicate_titles(items, {}) == ["A B"]

    def test_dedups_on_existing_title_fallback_for_existing_ids(self) -> None:
        # codex on #1729: {id:1,title:""} and {id:2,title:"Alpha"} pass the
        # raw nudge as distinct, then both land as "Alpha" once the blank
        # falls back to the existing title for id 1.  Effective-title dedup
        # catches the collision before the nudge accepts the response.
        existing_by_id = {"1": "Alpha", "2": "Beta"}
        items = [{"id": "1", "title": ""}, {"id": "2", "title": "Alpha"}]
        assert _find_duplicate_titles(items, existing_by_id) == ["Alpha"]


# ── _validate_rescope_batch ───────────────────────────────────────────────────


class TestValidateRescopeBatch:
    """#1715: rescope batch validation runs before any commit.  Errors
    reject the whole batch atomically — partial commits would leave
    tasks.json in a state Opus didn't propose."""

    def _t(self, task_id: str, title: str = "") -> dict:
        return {
            "id": task_id,
            "title": title,
            "type": "spec",
            "status": "pending",
            "description": "",
        }

    def test_empty_batch_is_valid(self) -> None:
        assert _validate_rescope_batch([self._t("1")], []) == []

    def test_known_id_with_changes_is_valid(self) -> None:
        current = [self._t("1", "A")]
        items = [{"id": "1", "title": "A renamed", "description": "x"}]
        assert _validate_rescope_batch(current, items) == []

    def test_null_id_is_treated_as_new_task_not_rejected(self) -> None:
        current = [self._t("1")]
        items = [{"id": None, "title": "Brand new", "description": ""}]
        assert _validate_rescope_batch(current, items) == []

    def test_unknown_source_id_is_rejected(self) -> None:
        current = [self._t("1")]
        items = [{"id": "999", "title": "Hallucinated"}]
        errors = _validate_rescope_batch(current, items)
        assert len(errors) == 1
        assert "999" in errors[0]
        assert "unknown source id" in errors[0]

    def test_duplicate_item_id_is_rejected(self) -> None:
        current = [self._t("1")]
        items = [
            {"id": "1", "title": "first"},
            {"id": "1", "title": "second"},
        ]
        errors = _validate_rescope_batch(current, items)
        assert len(errors) == 1
        assert "duplicate" in errors[0]

    def test_blank_id_is_rejected(self) -> None:
        current = [self._t("1")]
        items = [{"id": "", "title": "blank id"}]
        errors = _validate_rescope_batch(current, items)
        assert any("non-empty string" in e for e in errors)

    def test_non_string_id_is_rejected(self) -> None:
        current = [self._t("1")]
        items = [{"id": 42, "title": "int id"}]
        errors = _validate_rescope_batch(current, items)
        assert any("non-empty string" in e for e in errors)

    def test_non_dict_item_is_rejected(self) -> None:
        current = [self._t("1")]
        items = ["not a dict"]
        errors = _validate_rescope_batch(current, items)  # type: ignore[arg-type]
        assert any("not a dict" in e for e in errors)

    def test_mixed_valid_and_invalid_collects_all_errors(self) -> None:
        # Validator runs to completion so the operator sees every problem
        # in one log scan, not just the first.
        current = [self._t("1"), self._t("2")]
        items = [
            {"id": "1", "title": "OK"},  # valid
            {"id": "999", "title": "ghost"},  # unknown source
            {"id": "1", "title": "dup"},  # duplicate of OK
            {"id": None, "title": "new"},  # valid (null = new)
        ]
        errors = _validate_rescope_batch(current, items)
        assert len(errors) == 2
        assert any("999" in e and "unknown" in e for e in errors)
        assert any("duplicate" in e for e in errors)

    def test_valid_merge_batch_is_accepted(self) -> None:
        # #1717: a target with merge_sources is valid when every source
        # is a known id AND appears in the same batch with status=completed.
        current = [self._t("a"), self._t("b"), self._t("c")]
        items = [
            {"id": "a", "title": "Merged", "merge_sources": ["b", "c"]},
            {"id": "b", "title": "B", "status": "completed"},
            {"id": "c", "title": "C", "status": "completed"},
        ]
        assert _validate_rescope_batch(current, items) == []

    def test_merge_with_unknown_source_is_rejected(self) -> None:
        current = [self._t("a"), self._t("b")]
        items = [
            {"id": "a", "title": "Merged", "merge_sources": ["b", "ghost"]},
            {"id": "b", "title": "B", "status": "completed"},
        ]
        errors = _validate_rescope_batch(current, items)
        assert any("ghost" in e and "unknown" in e for e in errors)

    def test_merge_into_self_is_rejected(self) -> None:
        current = [self._t("a"), self._t("b")]
        items = [
            {"id": "a", "title": "Merged", "merge_sources": ["a", "b"]},
            {"id": "b", "title": "B", "status": "completed"},
        ]
        errors = _validate_rescope_batch(current, items)
        assert any("merge a task into itself" in e for e in errors)

    def test_merge_source_without_completion_is_rejected(self) -> None:
        # If the source isn't marked completed in the batch, the per-task
        # coverage invariant breaks: the source has no op of its own.
        current = [self._t("a"), self._t("b")]
        items = [
            {"id": "a", "title": "Merged", "merge_sources": ["b"]},
            {"id": "b", "title": "B"},  # missing status="completed"
        ]
        errors = _validate_rescope_batch(current, items)
        assert any(
            'must also appear in the batch with status="completed"' in e for e in errors
        )

    def test_merge_sources_must_be_list(self) -> None:
        current = [self._t("a")]
        items = [{"id": "a", "title": "x", "merge_sources": "b"}]
        errors = _validate_rescope_batch(current, items)
        assert any("must be a list" in e for e in errors)

    def test_merge_source_must_be_non_empty_string(self) -> None:
        current = [self._t("a"), self._t("b")]
        items = [
            {"id": "a", "title": "Merged", "merge_sources": ["b", 42, ""]},
            {"id": "b", "title": "B", "status": "completed"},
        ]
        errors = _validate_rescope_batch(current, items)
        assert any("must be non-empty string" in e and "42" in e for e in errors)
        assert any("must be non-empty string" in e and "''" in e for e in errors)

    def test_merge_sources_on_completed_target_is_rejected(self) -> None:
        # codex on #1738: explicit-completion precedence (#1716) wins
        # over merge in _rescope_releases_for_oracle, so a batch with
        # both status="completed" and merge_sources on the same target
        # would silently drop the merge AND suppress the source's
        # completion notification — the source vanishes without lineage
        # being preserved anywhere.  Reject the contradictory shape.
        current = [self._t("target"), self._t("source")]
        items = [
            {
                "id": "target",
                "title": "T",
                "status": "completed",
                "merge_sources": ["source"],
            },
            {"id": "source", "title": "S", "status": "completed"},
        ]
        errors = _validate_rescope_batch(current, items)
        assert any(
            "merging into a completed task is contradictory" in e
            and "via this batch" in e
            for e in errors
        )

    def test_merge_sources_on_null_id_target_is_rejected(self) -> None:
        # codex on #1738: a null-id item carrying merge_sources slips
        # past every merge check today.  _make_new_tasks_from_opus
        # creates the new task without folding the source's lineage,
        # and _merge_source_ids still suppresses the source's
        # completion notification — the source vanishes without
        # lineage preservation, violating the no-origin-loss invariant.
        current = [self._t("source")]
        items = [
            {
                "id": None,
                "title": "New merged task",
                "merge_sources": ["source"],
            },
            {"id": "source", "title": "S", "status": "completed"},
        ]
        errors = _validate_rescope_batch(current, items)
        assert any("null/missing id" in e for e in errors)

    def test_null_id_without_merge_sources_is_still_valid(self) -> None:
        # Negative regression: ordinary new-task items (null id, no
        # merge_sources) stay valid.
        current = [self._t("a")]
        items = [{"id": None, "title": "Brand new"}]
        assert _validate_rescope_batch(current, items) == []

    def test_null_id_with_falsy_non_list_merge_sources_is_rejected(self) -> None:
        # codex follow-up on #1738: presence-and-value check catches
        # malformed-but-falsy values (``""``, ``0``, ``False``) that
        # truthiness would let slip through.  Empty list stays accepted
        # as the documented "no merge" sentinel.
        current = [self._t("a")]
        for bad in ("", 0, False, "string-not-list", 42):
            items = [{"id": None, "title": "new", "merge_sources": bad}]
            errors = _validate_rescope_batch(current, items)
            assert any("null/missing id" in e for e in errors), (
                f"merge_sources={bad!r} on null id should be rejected"
            )

    def test_null_id_with_empty_list_merge_sources_is_accepted(self) -> None:
        # The empty-list sentinel is the documented "no merge" no-op
        # and stays accepted on a null-id new task.
        current = [self._t("a")]
        items = [{"id": None, "title": "Brand new", "merge_sources": []}]
        assert _validate_rescope_batch(current, items) == []

    def test_same_source_into_multiple_targets_is_rejected(self) -> None:
        # codex Medium on #1738: a source feeding multiple targets
        # duplicates its lineage into each, which is split/rebuild
        # semantics — not a merge.  Split lands under #1718.  This leaf
        # rejects the contradictory shape: each source may merge into
        # at most one target.
        current = [self._t("a"), self._t("b"), self._t("c")]
        items = [
            {"id": "a", "title": "A", "merge_sources": ["c"]},
            {"id": "b", "title": "B", "merge_sources": ["c"]},
            {"id": "c", "title": "C", "status": "completed"},
        ]
        errors = _validate_rescope_batch(current, items)
        assert any("may merge into at most one target" in e for e in errors)

    def test_merge_into_blocked_target_is_rejected(self) -> None:
        # codex on #1738: a blocked target accepts merge_sources today,
        # but the worker picker skips blocked tasks — the source flips
        # to completed and its notification is suppressed while the
        # merged work parks indefinitely on a row Fido won't pick up.
        # Same shape of silent drop as the completed-target case.
        target = self._t("target")
        target["status"] = "blocked"
        items = [
            {"id": "target", "title": "T", "merge_sources": ["source"]},
            {"id": "source", "title": "S", "status": "completed"},
        ]
        current = [target, self._t("source")]
        errors = _validate_rescope_batch(current, items)
        assert any("target") and any("blocked" in e for e in errors)

    def test_merge_into_already_completed_target_on_disk_is_rejected(
        self,
    ) -> None:
        # codex on #1738: even when the item doesn't restate
        # status="completed", merging into a target whose CURRENT
        # status is already COMPLETED gets silently dropped by
        # _rescope_releases_for_oracle (it skips completed snapshot
        # tasks) — but _merge_source_ids still suppresses the source's
        # completion notification.  Reject the shape at the validator.
        target = self._t("target")
        target["status"] = "completed"
        items = [
            {"id": "target", "title": "T", "merge_sources": ["source"]},
            {"id": "source", "title": "S", "status": "completed"},
        ]
        current = [target, self._t("source")]
        errors = _validate_rescope_batch(current, items)
        assert any(
            "merging into a completed task is contradictory" in e and "already on" in e
            for e in errors
        )

    def test_empty_merge_sources_on_completed_target_is_harmless(self) -> None:
        # codex on #1738 (low): an empty merge_sources is the
        # documented "no merge" sentinel and shouldn't be rejected on a
        # completed target — there's no contradiction to fail closed on.
        current = [self._t("target")]
        items = [
            {
                "id": "target",
                "title": "T",
                "status": "completed",
                "merge_sources": [],
            },
        ]
        assert _validate_rescope_batch(current, items) == []

    def test_thread_source_into_non_thread_target_is_rejected(self) -> None:
        # codex P1 on #1738: a thread source contributes comment lineage
        # via thread.lineage_comment_ids on disk; a non-thread target has
        # nowhere to store that lineage, so the materializer would
        # silently drop it.  Reject the merge atomically.
        spec_target = self._t("spec", "Spec target")
        thread_source = self._t("thread", "Thread source")
        thread_source["type"] = "thread"
        thread_source["thread"] = {"repo": "r/r", "pr": 1, "comment_id": 99}
        items = [
            {"id": "spec", "title": "Merged", "merge_sources": ["thread"]},
            {"id": "thread", "title": "Thread source", "status": "completed"},
        ]
        errors = _validate_rescope_batch([spec_target, thread_source], items)
        assert any("thread source" in e and "non-thread target" in e for e in errors)

    def test_spec_to_spec_merge_is_accepted(self) -> None:
        # Negative regression: same-kind merges (no thread anywhere) stay
        # valid — the lineage-loss rule only fires when a thread source
        # would feed a non-thread target.
        current = [self._t("a"), self._t("b")]
        items = [
            {"id": "a", "title": "Merged", "merge_sources": ["b"]},
            {"id": "b", "title": "B", "status": "completed"},
        ]
        assert _validate_rescope_batch(current, items) == []

    # ── split validation (#1718) ──────────────────────────────────────────

    def test_split_targets_must_be_a_list(self) -> None:
        current = [self._t("src")]
        items = [{"id": "src", "title": "S", "split_targets": "Child A"}]
        errors = _validate_rescope_batch(current, items)
        assert any("split_targets" in e and "must be a list" in e for e in errors)

    def test_empty_split_targets_is_accepted_as_no_op(self) -> None:
        # Mirror of the empty merge_sources sentinel: ``[]`` documents
        # "no split", not "broken split" — accept silently.
        current = [self._t("src")]
        items = [{"id": "src", "title": "S", "split_targets": []}]
        assert _validate_rescope_batch(current, items) == []

    def test_split_on_null_id_is_rejected(self) -> None:
        current = [self._t("src")]
        items = [{"id": None, "title": "x", "split_targets": [{"title": "A"}]}]
        errors = _validate_rescope_batch(current, items)
        assert any("split_targets" in e and "null/missing id" in e for e in errors)

    def test_split_child_must_be_dict_with_non_empty_title(self) -> None:
        current = [self._t("src")]
        items = [
            {
                "id": "src",
                "title": "S",
                "split_targets": [
                    "not a dict",
                    {"title": ""},
                    {"title": 42},
                    {},
                ],
            }
        ]
        errors = _validate_rescope_batch(current, items)
        # First entry rejected for shape; rest rejected for title.
        assert any("[0]" in e and "must be a dict" in e for e in errors)
        assert any("[1].title" in e for e in errors)
        assert any("[2].title" in e for e in errors)
        assert any("[3].title" in e for e in errors)

    def test_split_combined_with_status_completed_is_rejected(self) -> None:
        current = [self._t("src")]
        items = [
            {
                "id": "src",
                "status": "completed",
                "split_targets": [{"title": "A"}],
            }
        ]
        errors = _validate_rescope_batch(current, items)
        assert any(
            "split_targets" in e and "structural ops are mutually exclusive" in e
            for e in errors
        )

    def test_split_combined_with_merge_sources_is_rejected(self) -> None:
        current = [self._t("src"), self._t("other")]
        items = [
            {
                "id": "src",
                "title": "S",
                "merge_sources": ["other"],
                "split_targets": [{"title": "A"}],
            },
            {"id": "other", "title": "O", "status": "completed"},
        ]
        errors = _validate_rescope_batch(current, items)
        assert any("combined with merge_sources" in e for e in errors)

    def test_split_on_source_thats_also_a_merge_source_is_rejected(self) -> None:
        # Lineage duplication: source folds into merge target AND is
        # decomposed into children — the lineage lands in two places.
        current = [self._t("src"), self._t("merger")]
        items = [
            {"id": "merger", "title": "M", "merge_sources": ["src"]},
            {
                "id": "src",
                "title": "S",
                "split_targets": [{"title": "A"}],
            },
        ]
        errors = _validate_rescope_batch(current, items)
        assert any(
            "split_targets" in e and "appears in another item's merge_sources" in e
            for e in errors
        )

    def test_split_on_completed_target_on_disk_is_rejected(self) -> None:
        src = self._t("src")
        src["status"] = "completed"
        items = [
            {"id": "src", "title": "S", "split_targets": [{"title": "A"}]},
        ]
        errors = _validate_rescope_batch([src], items)
        assert any(
            "split_targets" in e and "already completed on disk" in e for e in errors
        )

    def test_split_on_blocked_target_on_disk_is_rejected(self) -> None:
        src = self._t("src")
        src["status"] = "blocked"
        items = [
            {"id": "src", "title": "S", "split_targets": [{"title": "A"}]},
        ]
        errors = _validate_rescope_batch([src], items)
        assert any("split_targets" in e and "is blocked" in e for e in errors)

    def test_valid_split_passes_validation(self) -> None:
        current = [self._t("src")]
        items = [
            {
                "id": "src",
                "title": "S",
                "split_targets": [
                    {"title": "Child A"},
                    {"title": "Child B", "description": "scope"},
                ],
            }
        ]
        assert _validate_rescope_batch(current, items) == []

    def test_split_on_unknown_id_skips_split_specific_checks(self) -> None:
        # When the item carries an id Opus invented (not in known_ids),
        # the upstream id-existence check already records "unknown
        # source id" — the split block then early-skips its own checks
        # rather than spamming duplicate error messages on a target
        # that doesn't exist.
        current = [self._t("real")]
        items = [
            {
                "id": "ghost",
                "split_targets": [{"title": "Child"}],
            }
        ]
        errors = _validate_rescope_batch(current, items)
        assert any("unknown source id" in e for e in errors)
        # No split-specific error piles on top of the unknown-id error.
        assert not any("split_targets on 'ghost'" in e for e in errors)

    def test_split_on_ask_source_is_rejected(self) -> None:
        # codex P1: kind classification is title-prefix driven, so a
        # split whose children carry plain titles silently reclassifies
        # an ASK source's children to executable spec tasks.  Reject.
        ask = self._t("ask", "ASK: how do I X?")
        items = [{"id": "ask", "split_targets": [{"title": "Do X"}]}]
        errors = _validate_rescope_batch([ask], items)
        assert any("ASK/DEFER/CI" in e for e in errors)

    def test_split_on_defer_source_is_rejected(self) -> None:
        defer = self._t("d", "DEFER: do later")
        items = [{"id": "d", "split_targets": [{"title": "Now"}]}]
        errors = _validate_rescope_batch([defer], items)
        assert any("ASK/DEFER/CI" in e for e in errors)

    def test_split_on_ci_failure_source_is_rejected(self) -> None:
        ci = self._t("c", "CI FAILURE: lint broke")
        ci["type"] = "ci"
        items = [{"id": "c", "split_targets": [{"title": "fix half"}]}]
        errors = _validate_rescope_batch([ci], items)
        assert any("ASK/DEFER/CI" in e for e in errors)


# ── reorder_tasks ─────────────────────────────────────────────────────────────


class TestReorderTasks:
    def _add(
        self, tmp_path: Path, title: str, task_type: TaskType = TaskType.SPEC
    ) -> dict:
        return Tasks(tmp_path).add(title=title, task_type=task_type)

    def _response(self, items: list[dict]) -> str:
        """Translate legacy-shape item dicts into the new operations envelope.

        Lets the existing TestReorderTasks suite keep passing the
        familiar item dicts while the rescope I/O contract upgrades to
        explicit ``{"operations": [...]}`` (#1719).  Mapping:

        * ``{"id": x, "status": "completed"}``   → ``remove``
        * ``{"id": x, "merge_sources": [...]}``  → ``merge``
        * ``{"id": x, "split_targets": [...]}``  → ``split``
        * ``{"id": x, "anchor_comment_id": n}``  → ``rewrite_anchor``
        * ``{"id": x, "title": ..., "description": ...}`` → ``rewrite``
        * ``{"id": x}`` (id only)                → ``keep``
        * ``{"id": null, ...}``                  → ``new``

        Per-source ``{"id": src, "status": "completed"}`` items that
        accompany a ``merge`` are dropped here because the new ``merge``
        op auto-expands its source closures (the rocq per-task coverage
        invariant runs in the translator below, not in the test
        fixture).
        """
        merge_source_ids: set[str] = set()
        for item in items:
            if isinstance(item.get("merge_sources"), list):
                for src in item["merge_sources"]:
                    if isinstance(src, str):
                        merge_source_ids.add(src)
        operations: list[dict] = []
        for item in items:
            if item.get("status") == "completed" and item.get("id") in merge_source_ids:
                continue
            item_id = item.get("id")
            if item_id is None or item_id == "":
                operations.append(
                    {
                        "op": "new",
                        "title": item.get("title", ""),
                        "description": item.get("description", ""),
                        "type": item.get("type", "spec"),
                    }
                )
                continue
            if item.get("status") == "completed":
                operations.append({"op": "remove", "id": item_id})
                continue
            if isinstance(item.get("merge_sources"), list) and item["merge_sources"]:
                operations.append(
                    {
                        "op": "merge",
                        "target_id": item_id,
                        "sources": list(item["merge_sources"]),
                        "title": item.get("title", ""),
                        "description": item.get("description", ""),
                    }
                )
                continue
            if isinstance(item.get("split_targets"), list) and item["split_targets"]:
                operations.append(
                    {
                        "op": "split",
                        "id": item_id,
                        "children": list(item["split_targets"]),
                    }
                )
                continue
            if "anchor_comment_id" in item:
                operations.append(
                    {
                        "op": "rewrite_anchor",
                        "id": item_id,
                        "anchor_comment_id": item["anchor_comment_id"],
                    }
                )
                continue
            if "title" in item or "description" in item:
                operations.append(
                    {
                        "op": "rewrite",
                        "id": item_id,
                        "title": item.get("title", ""),
                        "description": item.get("description", ""),
                    }
                )
                continue
            operations.append({"op": "keep", "id": item_id})
        return json.dumps({"operations": operations})

    def test_skips_when_no_tasks(self, tmp_path: Path) -> None:
        client = _client("")
        reorder_tasks(Tasks(tmp_path), "", agent=client)
        client.run_turn.assert_not_called()

    def test_creates_default_client_when_none(self, tmp_path: Path) -> None:
        self._add(tmp_path, "Task A")

        factory_calls: list[None] = []

        def fake_factory() -> object:
            factory_calls.append(None)
            return _client("")

        reorder_tasks(Tasks(tmp_path), "", _client_factory=fake_factory)
        assert factory_calls == [None]

    def test_skips_on_empty_opus_response(self, tmp_path: Path) -> None:
        self._add(tmp_path, "Task A")
        result_before = Tasks(tmp_path).list()
        reorder_tasks(Tasks(tmp_path), "", agent=_client(""))
        assert Tasks(tmp_path).list() == result_before

    def test_skips_on_unparseable_response(self, tmp_path: Path) -> None:
        self._add(tmp_path, "Task A")
        result_before = Tasks(tmp_path).list()
        reorder_tasks(Tasks(tmp_path), "", agent=_client("not json"))
        assert Tasks(tmp_path).list() == result_before

    def test_invalid_batch_leaves_durable_list_unchanged(self, tmp_path: Path) -> None:
        # #1715: a mixed batch — one valid item and one referencing a
        # hallucinated id — must reject the WHOLE batch.  The valid
        # item's rename does not partially commit; tasks.json stays as
        # it was before reorder_tasks ran.
        t1 = self._add(tmp_path, "Original")
        before = Tasks(tmp_path).list()
        raw = self._response(
            [
                {"id": t1["id"], "title": "Renamed", "description": ""},
                {"id": "ghost-999", "title": "hallucinated"},
            ]
        )
        reorder_tasks(Tasks(tmp_path), "", agent=_client(raw))
        # Durable list is byte-for-byte identical to pre-call state.
        assert Tasks(tmp_path).list() == before

    def test_invalid_batch_skips_on_done(self, tmp_path: Path) -> None:
        # codex on #1733: _on_done's contract is "post-successful reorder."
        # Production wires it to sync_tasks (git push) and
        # _rewrite_pr_description (GitHub API write); firing on rejection
        # would cause unnecessary churn for invalid Opus output.
        t1 = self._add(tmp_path, "Original")
        raw = self._response([{"id": "ghost-999", "title": "hallucinated"}])
        done_calls: list[int] = []
        reorder_tasks(
            Tasks(tmp_path),
            "",
            agent=_client(raw),
            _on_done=lambda: done_calls.append(1),
        )
        assert done_calls == []
        assert Tasks(tmp_path).list()[0]["title"] == t1["title"]

    def test_invalid_batch_skips_on_changes_callback(self, tmp_path: Path) -> None:
        # _on_changes is post-commit notification machinery; rejected
        # batches mustn't fire it (there's no rescope to report on).
        thread = {"repo": "r/r", "pr": 1, "comment_id": 99}
        t1 = Tasks(tmp_path).add(
            title="Thread task", task_type=TaskType.THREAD, thread=thread
        )
        raw = self._response(
            [
                {"id": t1["id"], "title": "Renamed", "description": ""},
                {"id": "ghost-999", "title": "hallucinated"},
            ]
        )
        received: list = []
        reorder_tasks(
            Tasks(tmp_path),
            "",
            agent=_client(raw),
            _on_changes=lambda changes: received.extend(changes),
        )
        assert received == []

    def test_rescope_apply_callback_fires_with_op_inputs(self, tmp_path: Path) -> None:
        # INV-F (#1804): when intents are provided, the
        # ``_on_rescope_apply`` callback fires with the result task
        # list, the oracle's OpInput list (typed), the task-id →
        # positive map, and the parsed verdicts so the notifier can
        # ask the oracle which intents need a reply-back.
        t1 = self._add(tmp_path, "Original")
        intents = [
            RescopeIntent(
                change_request="rename it",
                comment_id=101,
                timestamp="2024-01-15T10:00:00+00:00",
            )
        ]
        raw = json.dumps(
            {
                "verdicts": [
                    {
                        "intent_comment_id": 101,
                        "outcome": "reshaped",
                        "ops": [
                            {
                                "op": "rewrite",
                                "id": t1["id"],
                                "title": "Renamed",
                                "description": "",
                            }
                        ],
                        "affected_task_ids": [t1["id"]],
                        "narrative": "renamed per ask",
                    }
                ]
            }
        )
        captured: list[tuple[list, list, dict, list]] = []
        reorder_tasks(
            Tasks(tmp_path),
            "",
            agent=_client(raw),
            intents=intents,
            _on_rescope_apply=lambda r, o, m, v: captured.append((r, o, m, v)),
        )
        assert len(captured) == 1
        result_tasks, op_inputs, ids_map, verdicts = captured[0]
        assert any(t["id"] == t1["id"] for t in result_tasks)
        assert ids_map[t1["id"]] >= 1
        assert any(101 in op.oi_intents for op in op_inputs)
        assert any(v.intent_comment_id == 101 for v in verdicts)

    def test_rescope_apply_callback_skipped_without_intents(
        self, tmp_path: Path
    ) -> None:
        # CI-triggered rescope (no intents) → callback doesn't fire.
        self._add(tmp_path, "Task")
        raw = json.dumps({"operations": []})
        called: list[int] = []
        reorder_tasks(
            Tasks(tmp_path),
            "",
            agent=_client(raw),
            _on_rescope_apply=lambda _r, _o, _m, _v: called.append(1),
        )
        assert called == []

    def test_validator_rejection_after_parse_success_drops_batch(
        self, tmp_path: Path
    ) -> None:
        # Parse + cross-op succeed, but the disk-aware validator
        # rejects (ASK source can't be split — kind is title-prefix
        # driven, splitting would silently reclassify children).  This
        # exercises the validator-rejection logging path and the
        # post-rejection early-return that skips _on_done / _on_changes.
        ask = Tasks(tmp_path).add(title="ASK: how do I X?", task_type=TaskType.SPEC)
        before = Tasks(tmp_path).list()
        # Build the operations envelope by hand so we can emit the
        # split shape that the legacy-shape translator wouldn't produce.
        raw = json.dumps(
            {
                "operations": [
                    {
                        "op": "split",
                        "id": ask["id"],
                        "children": [{"title": "Do X", "description": ""}],
                    }
                ]
            }
        )
        done_calls: list[int] = []
        reorder_tasks(
            Tasks(tmp_path),
            "",
            agent=_client(raw),
            _on_done=lambda: done_calls.append(1),
        )
        # Durable list unchanged; _on_done not fired.
        assert Tasks(tmp_path).list() == before
        assert done_calls == []

    def test_parse_error_nudge_with_empty_response_drops_batch(
        self, tmp_path: Path
    ) -> None:
        # Initial response has parse errors → nudge fires → empty
        # response from Opus on the nudge → batch dropped.  Exercises
        # the empty-after-parse-error-nudge log + early-return.
        from unittest.mock import MagicMock

        Tasks(tmp_path).add(title="Original", task_type=TaskType.SPEC)
        before = Tasks(tmp_path).list()
        bad_raw = json.dumps({"operations": [{"op": "bogus", "id": "x"}]})
        client = MagicMock(spec=ClaudeClient)
        client.voice_model = "claude-opus-4-6"
        client.work_model = "claude-sonnet-4-6"
        client.brief_model = "claude-haiku-4-5"
        # First call returns the unparseable response; nudge call
        # returns empty so the loop drops the batch.
        client.run_turn.side_effect = [bad_raw, ""]
        reorder_tasks(Tasks(tmp_path), "", agent=client)
        assert Tasks(tmp_path).list() == before
        assert client.run_turn.call_count == 2

    def test_empty_operations_array_preserves_all_tasks(self, tmp_path: Path) -> None:
        # #1721: an explicit `{"operations": []}` response means "keep
        # everything as-is", not "wipe the queue".  Omission has the
        # same semantics — the only way to remove a task is an
        # explicit `remove` op (#1357).
        t1 = self._add(tmp_path, "Keep me A")
        t2 = self._add(tmp_path, "Keep me B")
        raw = json.dumps({"operations": []})
        reorder_tasks(Tasks(tmp_path), "", agent=_client(raw))
        result = Tasks(tmp_path).list()
        assert [t["id"] for t in result] == [t1["id"], t2["id"]]
        assert all(t["status"] != str(TaskStatus.COMPLETED) for t in result)

    def test_partial_response_preserves_omitted_tasks(self, tmp_path: Path) -> None:
        # #1721: a partial response — Opus mentions some snapped ids
        # but not others — must not silently delete the omitted ones.
        # The unmentioned task stays pending, identical to how it was.
        t1 = self._add(tmp_path, "Renamed")
        t2 = self._add(tmp_path, "Untouched")
        raw = self._response(
            [{"id": t1["id"], "title": "Renamed", "description": "new"}]
        )
        reorder_tasks(Tasks(tmp_path), "", agent=_client(raw))
        result = Tasks(tmp_path).list()
        ids = [t["id"] for t in result]
        assert t1["id"] in ids
        assert t2["id"] in ids
        untouched = next(t for t in result if t["id"] == t2["id"])
        assert untouched["title"] == "Untouched"
        assert untouched["status"] != str(TaskStatus.COMPLETED)

    def test_post_snapshot_tasks_preserved_across_rescope(self, tmp_path: Path) -> None:
        # #1721: tasks added AFTER the snapshot Opus saw (a comment
        # arriving while reorder_tasks ran) must survive — they're
        # not in the snapshot, so they cannot have been claimed by an
        # operation, and therefore are passed through unchanged.
        t1 = self._add(tmp_path, "Snapped task")
        original_ids = frozenset({t1["id"]})
        # Now add a post-snapshot task BEFORE reorder_tasks runs.  The
        # snapshot Opus will be told about is just t1; its rescope
        # response operates on t1 alone.
        t2 = self._add(tmp_path, "Post-snapshot task")
        raw = self._response(
            [{"id": t1["id"], "title": "Renamed snap", "description": ""}]
        )
        # Pass the original snapshot id frozenset by patching the
        # internal frozenset construction via the public reorder_tasks
        # entry point — production passes the comment-time snapshot
        # via task_list.  We force the same effect by ensuring t2's
        # id is NOT in the snapshot frozenset that _apply_reorder
        # synthesizes.  Since reorder_tasks builds the snapshot from
        # tasks.list() at call time, both t1 and t2 are in the
        # snapshot — to truly exercise the post-snapshot path we have
        # to call _apply_reorder directly with a snapshot that
        # excludes t2.
        from fido.tasks import _apply_reorder

        ordered_items = [{"id": t1["id"], "title": "Renamed snap", "description": ""}]
        result = _apply_reorder(
            Tasks(tmp_path).list(), ordered_items, original_ids=original_ids
        )
        ids = [t["id"] for t in result]
        assert t1["id"] in ids
        assert t2["id"] in ids, "post-snapshot task must survive rescope"
        post_snap = next(t for t in result if t["id"] == t2["id"])
        assert post_snap["title"] == "Post-snapshot task"
        # Sanity: reorder_tasks doesn't drop t2 either when called the
        # normal way (snapshot includes both, no op claims t2 →
        # KeepTask emitted under the hood by the omission path).
        del raw  # unused once we drop down to _apply_reorder
        del t2

    def test_explicit_full_rebuild_removes_snapped_and_creates_new(
        self, tmp_path: Path
    ) -> None:
        # #1721: a full rebuild is `remove` + `new` ops, not an
        # implicit wipe.  Verify the path end-to-end.
        t1 = self._add(tmp_path, "Old A")
        t2 = self._add(tmp_path, "Old B")
        raw = json.dumps(
            {
                "operations": [
                    {"op": "remove", "id": t1["id"]},
                    {"op": "remove", "id": t2["id"]},
                    {
                        "op": "new",
                        "title": "Replacement",
                        "description": "fresh plan",
                        "type": "spec",
                    },
                ]
            }
        )
        reorder_tasks(Tasks(tmp_path), "", agent=_client(raw))
        result = Tasks(tmp_path).list()
        old_a = next(t for t in result if t["id"] == t1["id"])
        old_b = next(t for t in result if t["id"] == t2["id"])
        assert old_a["status"] == str(TaskStatus.COMPLETED)
        assert old_b["status"] == str(TaskStatus.COMPLETED)
        new_tasks = [
            t
            for t in result
            if t["id"] not in {t1["id"], t2["id"]}
            and t["status"] != str(TaskStatus.COMPLETED)
        ]
        assert len(new_tasks) == 1
        assert new_tasks[0]["title"] == "Replacement"
        assert new_tasks[0]["description"] == "fresh plan"

    def test_preserves_snapshot_order_for_non_ci_tasks(self, tmp_path: Path) -> None:
        t1 = self._add(tmp_path, "First")
        t2 = self._add(tmp_path, "Second")
        # Opus returns them reversed
        raw = self._response(
            [
                {"id": t2["id"], "title": "Second", "description": ""},
                {"id": t1["id"], "title": "First", "description": ""},
            ]
        )
        reorder_tasks(Tasks(tmp_path), "", agent=_client(raw))
        result = Tasks(tmp_path).list()
        assert result[0]["id"] == t1["id"]
        assert result[1]["id"] == t2["id"]

    def test_applies_title_from_opus_for_existing_id(self, tmp_path: Path) -> None:
        # #1713: title is mutable for an existing task id; the rewrite
        # flows end-to-end through reorder_tasks.
        t1 = self._add(tmp_path, "Old title")
        raw = self._response(
            [{"id": t1["id"], "title": "New title", "description": ""}]
        )
        reorder_tasks(Tasks(tmp_path), "", agent=_client(raw))
        assert Tasks(tmp_path).list()[0]["title"] == "New title"

    def test_keeps_task_opus_excludes(self, tmp_path: Path) -> None:
        """#1357: end-to-end — reorder_tasks must not mark a task completed
        because Opus omitted it.  Status survives the rescope unchanged."""
        t1 = self._add(tmp_path, "Keep")
        t2 = self._add(tmp_path, "Still pending")
        raw = self._response([{"id": t1["id"], "title": "Keep", "description": ""}])
        reorder_tasks(Tasks(tmp_path), "", agent=_client(raw))
        result = Tasks(tmp_path).list()
        task2 = next(t for t in result if t["id"] == t2["id"])
        assert task2["status"] == "pending"
        assert task2["title"] == "Still pending"

    def test_preserves_completed_tasks(self, tmp_path: Path) -> None:
        t1 = self._add(tmp_path, "Done")
        Tasks(tmp_path).complete_by_id(t1["id"])
        t2 = self._add(tmp_path, "Pending")
        raw = self._response([{"id": t2["id"], "title": "Pending", "description": ""}])
        reorder_tasks(Tasks(tmp_path), "", agent=_client(raw))
        result = Tasks(tmp_path).list()
        statuses = {t["id"]: t["status"] for t in result}
        assert statuses[t1["id"]] == "completed"

    def test_rescope_prompt_fn_receives_task_list_and_commit_summary(
        self, tmp_path: Path
    ) -> None:
        self._add(tmp_path, "Task A")
        mock_prompts = MagicMock(spec=Prompts)
        mock_prompts.rescope_prompt.return_value = "prompt text"
        reorder_tasks(
            Tasks(tmp_path),
            "feat: added thing",
            agent=_client(""),
            prompts=mock_prompts,
        )
        mock_prompts.rescope_prompt.assert_called_once()
        call_kwargs = mock_prompts.rescope_prompt.call_args
        assert call_kwargs[0][1] == "feat: added thing"
        assert len(call_kwargs[0][0]) == 1

    def test_picks_up_task_added_while_opus_was_thinking(self, tmp_path: Path) -> None:
        t1 = self._add(tmp_path, "Original task")
        new_task_id: list[str] = []

        def slow_run_turn(prompt: str, *, model: object = None, **kw: object) -> object:
            # Simulate a new task arriving while Opus is running
            t2 = Tasks(tmp_path).add(
                title="Arrived mid-reorder", task_type=TaskType.SPEC
            )
            new_task_id.append(t2["id"])
            return json.dumps(
                {
                    "tasks": [
                        {
                            "id": t2["id"],
                            "title": "Reordered mid-reorder task",
                            "description": "should be ignored",
                        },
                        {"id": t1["id"], "title": "Original task", "description": ""},
                    ]
                }
            )

        client = _client()
        client.run_turn.side_effect = slow_run_turn
        reorder_tasks(Tasks(tmp_path), "", agent=client)
        result = Tasks(tmp_path).list()
        ids = [t["id"] for t in result]
        assert new_task_id[0] in ids  # not silently dropped
        assert ids == [t1["id"], new_task_id[0]]
        new_task = next(t for t in result if t["id"] == new_task_id[0])
        assert new_task["title"] == "Arrived mid-reorder"
        assert new_task["description"] == ""

    def test_on_changes_not_fired_when_opus_omits_thread_task(
        self, tmp_path: Path
    ) -> None:
        """#1357: omission ⇒ keep-as-is, so no thread-completion change to
        report.  Opus omitting a thread task no longer counts as a
        completion event."""
        thread = {
            "repo": "r/r",
            "pr": 1,
            "comment_id": 99,
            "url": "https://example.com",
        }
        t1 = Tasks(tmp_path).add(
            title="Thread task", task_type=TaskType.THREAD, thread=thread
        )
        t2 = self._add(tmp_path, "Keep this")
        received: list = []
        raw = self._response(
            [{"id": t2["id"], "title": "Keep this", "description": ""}]
        )
        reorder_tasks(
            Tasks(tmp_path),
            "",
            agent=_client(raw),
            _on_changes=lambda changes: received.extend(changes),
        )
        assert received == []
        task1 = next(t for t in Tasks(tmp_path).list() if t["id"] == t1["id"])
        assert task1["status"] == "pending"

    def test_on_changes_skipped_when_only_description_changes(
        self, tmp_path: Path
    ) -> None:
        """Pure description rewrites (title preserved) do not fire a
        reply-back to the commenter — internal rephrasing isn't a contract
        change worth notifying the reviewer about (#1388)."""
        thread = {
            "repo": "r/r",
            "pr": 1,
            "comment_id": 99,
            "url": "https://example.com",
        }
        t1 = Tasks(tmp_path).add(
            title="Stable title",
            task_type=TaskType.THREAD,
            thread=thread,
        )
        received: list = []
        raw = self._response(
            [{"id": t1["id"], "title": "Stable title", "description": "new"}]
        )
        reorder_tasks(
            Tasks(tmp_path),
            "",
            agent=_client(raw),
            _on_changes=lambda changes: received.extend(changes),
        )
        assert received == []
        task1 = next(t for t in Tasks(tmp_path).list() if t["id"] == t1["id"])
        assert task1["description"] == "new"
        assert task1["title"] == "Stable title"

    def test_on_changes_fires_when_rescope_rewrites_title(self, tmp_path: Path) -> None:
        """#1713: title is mutable; a title rewrite by Opus is a contract
        change for thread tasks and notifies the original commenter."""
        thread = {
            "repo": "r/r",
            "pr": 1,
            "comment_id": 99,
            "url": "https://example.com",
        }
        t1 = Tasks(tmp_path).add(
            title="Old title",
            task_type=TaskType.THREAD,
            thread=thread,
        )
        received: list = []
        raw = self._response(
            [{"id": t1["id"], "title": "New title", "description": ""}]
        )
        reorder_tasks(
            Tasks(tmp_path),
            "",
            agent=_client(raw),
            _on_changes=lambda changes: received.extend(changes),
        )
        assert len(received) == 1
        assert received[0]["kind"] == "modified"
        assert received[0]["new_title"] == "New title"
        task1 = next(t for t in Tasks(tmp_path).list() if t["id"] == t1["id"])
        assert task1["title"] == "New title"

    def test_on_changes_not_called_when_no_thread_tasks_changed(
        self, tmp_path: Path
    ) -> None:
        t1 = self._add(tmp_path, "Spec task")
        received: list = []
        raw = self._response(
            [{"id": t1["id"], "title": "Spec task", "description": ""}]
        )
        reorder_tasks(
            Tasks(tmp_path),
            "",
            agent=_client(raw),
            _on_changes=lambda changes: received.extend(changes),
        )
        assert received == []

    def test_on_changes_none_does_not_error(self, tmp_path: Path) -> None:
        """A None callback must not raise when the rescope rewrites a thread
        task's description (the change-emitting path under #1357)."""
        thread = {"repo": "r/r", "pr": 1, "comment_id": 42, "url": "https://x.com"}
        t1 = Tasks(tmp_path).add(
            title="Thread task", task_type=TaskType.THREAD, thread=thread
        )
        raw = self._response(
            [{"id": t1["id"], "title": "Thread task", "description": "rewritten"}]
        )
        reorder_tasks(Tasks(tmp_path), "", agent=_client(raw), _on_changes=None)
        task1 = next(t for t in Tasks(tmp_path).list() if t["id"] == t1["id"])
        assert task1["description"] == "rewritten"

    def test_on_inprogress_affected_not_called_when_opus_omits_inprogress_task(
        self, tmp_path: Path
    ) -> None:
        """#1357: omitting an in-progress task no longer auto-completes it,
        so the affected-callback no longer fires from omission.  The task
        survives in the queue at in_progress; only an explicit modification
        by Opus triggers _on_inprogress_affected (covered by the modify
        test)."""
        t1 = self._add(tmp_path, "In-progress task")
        Tasks(tmp_path).update(t1["id"], TaskStatus.IN_PROGRESS)
        t2 = self._add(tmp_path, "Keep this")
        raw = self._response(
            [{"id": t2["id"], "title": "Keep this", "description": ""}]
        )
        affected: list[int] = []
        reorder_tasks(
            Tasks(tmp_path),
            "",
            agent=_client(raw),
            _on_inprogress_affected=lambda _task_id: affected.append(1),
        )
        assert affected == []
        task1 = next(t for t in Tasks(tmp_path).list() if t["id"] == t1["id"])
        assert task1["status"] == "in_progress"

    def test_on_inprogress_affected_called_when_inprogress_task_modified(
        self, tmp_path: Path
    ) -> None:
        # #1713: title is mutable; the rewrite is applied alongside the
        # in-progress reset-to-pending + abort signal.
        t1 = self._add(tmp_path, "Stable title")
        Tasks(tmp_path).update(t1["id"], TaskStatus.IN_PROGRESS)
        raw = self._response(
            [{"id": t1["id"], "title": "Changed title", "description": "new"}]
        )
        affected: list[int] = []
        reorder_tasks(
            Tasks(tmp_path),
            "",
            agent=_client(raw),
            _on_inprogress_affected=lambda _task_id: affected.append(1),
        )
        assert affected == [1]
        result = Tasks(tmp_path).list()
        assert result[0]["title"] == "Changed title"
        assert result[0]["description"] == "new"
        assert result[0]["status"] == str(TaskStatus.PENDING)

    def test_on_inprogress_affected_not_called_when_inprogress_task_unchanged(
        self, tmp_path: Path
    ) -> None:
        t1 = self._add(tmp_path, "Stable task")
        Tasks(tmp_path).update(t1["id"], TaskStatus.IN_PROGRESS)
        raw = self._response(
            [{"id": t1["id"], "title": "Stable task", "description": ""}]
        )
        affected: list[int] = []
        reorder_tasks(
            Tasks(tmp_path),
            "",
            agent=_client(raw),
            _on_inprogress_affected=lambda _task_id: affected.append(1),
        )
        assert affected == []
        # task still in_progress (unchanged by Opus)
        result = Tasks(tmp_path).list()
        assert result[0]["status"] == str(TaskStatus.IN_PROGRESS)

    def test_on_inprogress_affected_not_called_when_anchor_id_type_only_differs(
        self, tmp_path: Path
    ) -> None:
        # codex on #1731: anchor comparison runs through
        # _task_source_comment_for_oracle (int(comment_id)) so a legacy
        # str id ('42') in tasks.json compares equal to the
        # oracle-materialized 42.  Without normalization, a spurious
        # type-mismatch would abort the in-progress turn.
        thread = {
            "repo": "r/r",
            "pr": 1,
            "comment_id": "42",  # legacy string form
        }
        t1 = Tasks(tmp_path).add(
            title="Thread task", task_type=TaskType.THREAD, thread=thread
        )
        Tasks(tmp_path).update(t1["id"], TaskStatus.IN_PROGRESS)
        # Same anchor, expressed as int — must compare equal.
        raw = self._response(
            [
                {
                    "id": t1["id"],
                    "title": "Thread task",
                    "description": "",
                    "anchor_comment_id": 42,
                },
            ]
        )
        affected: list[str] = []
        reorder_tasks(
            Tasks(tmp_path),
            "",
            agent=_client(raw),
            _on_inprogress_affected=lambda task_id: affected.append(task_id),
        )
        assert affected == []
        result = Tasks(tmp_path).list()
        assert result[0]["status"] == str(TaskStatus.IN_PROGRESS)

    def test_on_inprogress_affected_called_when_new_task_demotes_inprogress(
        self, tmp_path: Path
    ) -> None:
        # #1844: a comment-driven new task lands at position 0 in Opus's
        # ordered_items, ahead of the in-progress task.  The reorder
        # interleaves the new task at Opus's position; the in-progress
        # is no longer first in the pending block, so it's demoted to
        # pending and the on_inprogress_affected callback fires so the
        # worker aborts its turn and re-picks against the new front.
        t1 = self._add(tmp_path, "Current task")
        Tasks(tmp_path).update(t1["id"], TaskStatus.IN_PROGRESS)
        raw = self._response(
            [
                # null-id at position 0 = new task placed ahead of t1.
                {"id": None, "title": "Higher-priority new ask"},
                {"id": t1["id"]},
            ]
        )
        affected: list[str] = []
        reorder_tasks(
            Tasks(tmp_path),
            "",
            agent=_client(raw),
            _on_inprogress_affected=lambda task_id: affected.append(task_id),
        )
        assert affected == [t1["id"]]
        result = Tasks(tmp_path).list()
        assert result[0]["title"] == "Higher-priority new ask"
        # t1 is demoted to pending; worker will re-pick it after the
        # new task completes.
        t1_after = next(t for t in result if t["id"] == t1["id"])
        assert t1_after["status"] == str(TaskStatus.PENDING)

    def test_inprogress_not_demoted_when_new_task_lands_after(
        self, tmp_path: Path
    ) -> None:
        # Inverse: the new task lands after the in-progress one.  No
        # demotion, no preempt callback.
        t1 = self._add(tmp_path, "Current task")
        Tasks(tmp_path).update(t1["id"], TaskStatus.IN_PROGRESS)
        raw = self._response(
            [
                {"id": t1["id"]},
                {"id": None, "title": "Later ask"},
            ]
        )
        affected: list[str] = []
        reorder_tasks(
            Tasks(tmp_path),
            "",
            agent=_client(raw),
            _on_inprogress_affected=lambda task_id: affected.append(task_id),
        )
        assert affected == []
        result = Tasks(tmp_path).list()
        assert result[0]["id"] == t1["id"]
        assert result[0]["status"] == str(TaskStatus.IN_PROGRESS)

    def test_inprogress_demoted_helper_skips_completed_at_front(self) -> None:
        # #1844: a completed task at the front of result doesn't count
        # as a demotion (the picker skips completed).
        from fido.tasks import _inprogress_was_demoted

        ip = {"id": "ip", "status": "in_progress", "type": "spec"}
        result = [
            {"id": "done", "status": "completed", "type": "spec"},
            ip,
        ]
        assert _inprogress_was_demoted(ip, result) is False

    def test_inprogress_demoted_helper_skips_ask_and_defer_prefixed(
        self,
    ) -> None:
        # codex P2 on PR #1847: ASK:/DEFER: title-prefixed tasks aren't
        # runnable (the picker filters them out), so a leading ASK: or
        # DEFER: row doesn't count as demotion.
        from fido.tasks import _inprogress_was_demoted

        ip = {"id": "ip", "status": "in_progress", "type": "spec", "title": "Real work"}
        result = [
            {
                "id": "ask1",
                "status": "pending",
                "type": "spec",
                "title": "ASK: should we...",
            },
            {
                "id": "defer1",
                "status": "pending",
                "type": "spec",
                "title": "DEFER: maybe later",
            },
            ip,
        ]
        assert _inprogress_was_demoted(ip, result) is False

    def test_inprogress_demoted_helper_returns_false_when_only_ci_remain(
        self,
    ) -> None:
        # #1844: edge case — in-progress task IS the only non-CI item
        # but it doesn't appear in result (defensive; shouldn't happen
        # in practice).  The helper walks past every CI / completed
        # entry and returns False rather than crashing.
        from fido.tasks import _inprogress_was_demoted

        ip = {"id": "ip", "status": "in_progress", "type": "ci"}
        result = [
            {"id": "ci1", "status": "pending", "type": "ci"},
            {"id": "done", "status": "completed", "type": "spec"},
        ]
        assert _inprogress_was_demoted(ip, result) is False

    def test_inprogress_demotion_skips_intervening_ci_and_completed(
        self, tmp_path: Path
    ) -> None:
        # The demotion check only looks at non-CI, non-completed tasks
        # ahead of the in-progress one.  A completed task at position 0
        # doesn't count as a demotion (it's a no-op for the picker), and
        # CI tasks are intentionally skipped (pre-#1846 they always come
        # first via the structural invariant + the existing CI preempt
        # path handles them).
        ci = self._add(tmp_path, "Fix CI", task_type=TaskType.CI)
        done = self._add(tmp_path, "Already done")
        Tasks(tmp_path).update(done["id"], TaskStatus.COMPLETED)
        t1 = self._add(tmp_path, "Current task")
        Tasks(tmp_path).update(t1["id"], TaskStatus.IN_PROGRESS)
        raw = self._response(
            [
                {"id": ci["id"]},
                {"id": t1["id"]},
                {"id": done["id"], "status": "completed"},
            ]
        )
        affected: list[str] = []
        reorder_tasks(
            Tasks(tmp_path),
            "",
            agent=_client(raw),
            _on_inprogress_affected=lambda task_id: affected.append(task_id),
        )
        # CI at front + completed at end is the existing invariant; t1
        # is first non-CI pending so NOT demoted.
        assert affected == []

    def test_omitted_snapshot_task_keeps_position_when_new_task_added(
        self, tmp_path: Path
    ) -> None:
        # codex P1 round 3 on #1847: a snapshot task Opus didn't
        # mention (omitted ⇒ KeepTask) must keep its snapshot position
        # even when another item in ordered_items spawns a new task.
        # Earlier interleave appended every "unseen" oracle task at
        # the end of the list, silently demoting omitted tasks behind
        # every explicit item.
        t1 = self._add(tmp_path, "First")
        self._add(tmp_path, "Second — Opus doesn't mention")
        t3 = self._add(tmp_path, "Third")
        # Opus emits only t1 and t3, plus a new task after t3.
        # t2 is omitted ⇒ KeepTask ⇒ should stay at position 1.
        raw = self._response(
            [
                {"id": t1["id"]},
                {"id": t3["id"]},
                {"id": None, "title": "Brand new"},
            ]
        )
        reorder_tasks(Tasks(tmp_path), "", agent=_client(raw))
        result = Tasks(tmp_path).list()
        titles = [t["title"] for t in result]
        # t2 stays at position 1 (between t1 and t3), new task goes
        # after t3 (its anchor in ordered_items).
        assert titles == [
            "First",
            "Second — Opus doesn't mention",
            "Third",
            "Brand new",
        ]

    def test_inprogress_demoted_by_ci_failure_titled_spec_task(
        self, tmp_path: Path
    ) -> None:
        # codex P2 round 3 on #1847: ``"CI FAILURE:"`` title with
        # ``type: "spec"`` is treated as a normal runnable pending task
        # by the picker (CI prioritisation looks at ``type`` only).
        # The demotion check must match — a new such row landing ahead
        # of the in-progress one must demote.
        t1 = self._add(tmp_path, "Current task")
        Tasks(tmp_path).update(t1["id"], TaskStatus.IN_PROGRESS)
        raw = self._response(
            [
                # spec-typed new task with a CI-flavored title prefix
                # — picker sees runnable, so we must too.
                {
                    "id": None,
                    "title": "CI FAILURE: looks like CI but spec type",
                    "type": "spec",
                },
                {"id": t1["id"]},
            ]
        )
        affected: list[str] = []
        reorder_tasks(
            Tasks(tmp_path),
            "",
            agent=_client(raw),
            _on_inprogress_affected=lambda task_id: affected.append(task_id),
        )
        assert affected == [t1["id"]]

    def test_on_inprogress_affected_called_when_anchor_changes(
        self, tmp_path: Path
    ) -> None:
        # codex on #1731: anchor-only rewrites must trigger the
        # in-progress affected callback too.  A worker turn that began
        # under the old anchor must not finish under the new one — the
        # task gets reset to pending and the worker re-picks it under
        # the new anchor.
        thread = {
            "repo": "r/r",
            "pr": 1,
            "comment_id": 42,
            "url": "https://example.com/c42",
            "author": "old-commenter",
        }
        t1 = Tasks(tmp_path).add(
            title="Thread task", task_type=TaskType.THREAD, thread=thread
        )
        Tasks(tmp_path).update(t1["id"], TaskStatus.IN_PROGRESS)
        # Same title and description; only anchor differs.
        raw = self._response(
            [
                {
                    "id": t1["id"],
                    "title": "Thread task",
                    "description": "",
                    "anchor_comment_id": 99,
                },
            ]
        )
        affected: list[str] = []
        reorder_tasks(
            Tasks(tmp_path),
            "",
            agent=_client(raw),
            _on_inprogress_affected=lambda task_id: affected.append(task_id),
        )
        assert affected == [t1["id"]]
        result = Tasks(tmp_path).list()
        assert result[0]["status"] == str(TaskStatus.PENDING)
        assert result[0]["thread"]["comment_id"] == 99

    def test_on_inprogress_affected_called_when_merge_grows_lineage(
        self, tmp_path: Path
    ) -> None:
        # codex on #1738: a MergeTasks that folds source lineage into
        # the in-progress target only grows thread.lineage_comment_ids;
        # title/desc/anchor/status stay the same.  The worker prompt
        # captured the old lineage, so the callback must still fire and
        # reset the task to pending.
        t_target = Tasks(tmp_path).add(
            title="Target",
            task_type=TaskType.THREAD,
            thread={"repo": "r/r", "pr": 1, "comment_id": 100},
        )
        t_source = Tasks(tmp_path).add(
            title="Source",
            task_type=TaskType.THREAD,
            thread={"repo": "r/r", "pr": 1, "comment_id": 200},
        )
        Tasks(tmp_path).update(t_target["id"], TaskStatus.IN_PROGRESS)
        raw = self._response(
            [
                {
                    "id": t_target["id"],
                    "title": "Target",
                    "description": "",
                    "merge_sources": [t_source["id"]],
                },
                {"id": t_source["id"], "title": "Source", "status": "completed"},
            ]
        )
        affected: list[str] = []
        reorder_tasks(
            Tasks(tmp_path),
            "",
            agent=_client(raw),
            _on_inprogress_affected=lambda task_id: affected.append(task_id),
        )
        assert affected == [t_target["id"]]
        result = Tasks(tmp_path).list()
        target = next(t for t in result if t["id"] == t_target["id"])
        assert target["status"] == str(TaskStatus.PENDING)
        # Lineage grew with the source's anchor.
        assert 200 in target["thread"]["lineage_comment_ids"]

    def test_on_changes_skips_merge_source_completion_notifications(
        self, tmp_path: Path
    ) -> None:
        # codex on #1738: a merge source's status flip to COMPLETED is
        # bookkeeping for the rescope reducer's per-task coverage —
        # the work was MOVED to the target, not finished by commits.
        # _compute_thread_changes must NOT fire a "covered by recent
        # commits" change record for it; the merged target will fire
        # its own record when it eventually completes.
        t_target = Tasks(tmp_path).add(
            title="Target",
            task_type=TaskType.THREAD,
            thread={"repo": "r/r", "pr": 1, "comment_id": 100, "url": "x"},
        )
        t_source = Tasks(tmp_path).add(
            title="Source",
            task_type=TaskType.THREAD,
            thread={"repo": "r/r", "pr": 1, "comment_id": 200, "url": "y"},
        )
        raw = self._response(
            [
                {
                    "id": t_target["id"],
                    "title": "Target",
                    "description": "",
                    "merge_sources": [t_source["id"]],
                },
                {"id": t_source["id"], "title": "Source", "status": "completed"},
            ]
        )
        received: list[dict] = []
        reorder_tasks(
            Tasks(tmp_path),
            "",
            agent=_client(raw),
            _on_changes=lambda changes: received.extend(changes),
        )
        # Source's COMPLETED transition is suppressed (it was merged,
        # not completed).  Only the target — still pending — would
        # generate a future change record when it completes.
        for change in received:
            assert change["task"]["id"] != t_source["id"], (
                "merge-source completion must not fire on_changes — "
                "the work moved to the target, not 'covered by commits'"
            )

    def test_on_inprogress_affected_called_when_explicitly_completed(
        self, tmp_path: Path
    ) -> None:
        # #1716: when Opus explicitly completes the in-progress task, the
        # worker turn must abort.  Unlike text/anchor changes, the task is
        # NOT reset to pending — it's done.  The callback fires so the
        # worker can stop running on a now-completed task.
        t1 = self._add(tmp_path, "Active task")
        Tasks(tmp_path).update(t1["id"], TaskStatus.IN_PROGRESS)
        raw = self._response(
            [
                {
                    "id": t1["id"],
                    "title": "Active task",
                    "description": "",
                    "status": "completed",
                },
            ]
        )
        affected: list[str] = []
        reorder_tasks(
            Tasks(tmp_path),
            "",
            agent=_client(raw),
            _on_inprogress_affected=lambda task_id: affected.append(task_id),
        )
        assert affected == [t1["id"]]
        # Task stays completed — not reset to pending.
        result = Tasks(tmp_path).list()
        assert result[0]["status"] == str(TaskStatus.COMPLETED)

    def test_on_inprogress_affected_not_called_when_no_inprogress_task(
        self, tmp_path: Path
    ) -> None:
        self._add(tmp_path, "Pending task")
        t2 = self._add(tmp_path, "Keep this")
        raw = self._response(
            [{"id": t2["id"], "title": "Keep this", "description": ""}]
        )
        affected: list[int] = []
        reorder_tasks(
            Tasks(tmp_path),
            "",
            agent=_client(raw),
            _on_inprogress_affected=lambda _task_id: affected.append(1),
        )
        assert affected == []

    def test_on_inprogress_affected_none_does_not_error(self, tmp_path: Path) -> None:
        """A None callback must not raise when Opus modifies the
        in-progress task — the reset-to-pending + abort path under #1357."""
        t1 = self._add(tmp_path, "In-progress task")
        Tasks(tmp_path).update(t1["id"], TaskStatus.IN_PROGRESS)
        raw = self._response(
            [
                {
                    "id": t1["id"],
                    "title": "In-progress task",
                    "description": "rewritten scope",
                }
            ]
        )
        reorder_tasks(
            Tasks(tmp_path),
            "",
            agent=_client(raw),
            _on_inprogress_affected=None,
        )
        task1 = next(t for t in Tasks(tmp_path).list() if t["id"] == t1["id"])
        assert task1["status"] == str(TaskStatus.PENDING)
        assert task1["description"] == "rewritten scope"

    def test_on_done_called_after_successful_reorder(self, tmp_path: Path) -> None:
        t1 = self._add(tmp_path, "Task A")
        raw = self._response([{"id": t1["id"], "title": "Task A", "description": ""}])
        done_calls: list = []
        reorder_tasks(
            Tasks(tmp_path),
            "",
            agent=_client(raw),
            _on_done=lambda: done_calls.append(1),
        )
        assert done_calls == [1]

    def test_on_done_not_called_when_no_tasks(self, tmp_path: Path) -> None:
        done_calls: list = []
        reorder_tasks(
            Tasks(tmp_path),
            "",
            agent=_client("{}"),
            _on_done=lambda: done_calls.append(1),
        )
        assert done_calls == []

    def test_on_done_not_called_on_empty_opus_response(self, tmp_path: Path) -> None:
        self._add(tmp_path, "Task A")
        done_calls: list = []
        reorder_tasks(
            Tasks(tmp_path),
            "",
            agent=_client(""),
            _on_done=lambda: done_calls.append(1),
        )
        assert done_calls == []

    def test_on_done_not_called_on_unparseable_response(self, tmp_path: Path) -> None:
        self._add(tmp_path, "Task A")
        done_calls: list = []
        reorder_tasks(
            Tasks(tmp_path),
            "",
            agent=_client("not json at all"),
            _on_done=lambda: done_calls.append(1),
        )
        assert done_calls == []

    def test_nudges_when_opus_proposes_duplicate_titles(self, tmp_path: Path) -> None:
        # The nudge succeeds and produces unique titles, which then flow
        # through (#1713 made title mutable for an existing task id).
        t1 = self._add(tmp_path, "Alpha")
        t2 = self._add(tmp_path, "Beta")
        dup_response = self._response(
            [
                {"id": t1["id"], "title": "Shared name", "description": ""},
                {"id": t2["id"], "title": "Shared name", "description": ""},
            ]
        )
        fixed_response = self._response(
            [
                {"id": t1["id"], "title": "Fixed Alpha", "description": ""},
                {"id": t2["id"], "title": "Fixed Beta", "description": ""},
            ]
        )
        client = _client()
        client.run_turn.side_effect = [dup_response, fixed_response]
        mock_prompts = MagicMock(spec=Prompts)
        mock_prompts.rescope_prompt.return_value = "prompt"
        mock_prompts.rescope_duplicate_nudge.return_value = "nudge"
        reorder_tasks(Tasks(tmp_path), "", agent=client, prompts=mock_prompts)
        mock_prompts.rescope_duplicate_nudge.assert_called_once_with(
            ["Shared name"], attempts_remaining=2
        )
        assert client.run_turn.call_count == 2
        tasks = Tasks(tmp_path).list()
        assert next(t for t in tasks if t["id"] == t1["id"])["title"] == "Fixed Alpha"
        assert next(t for t in tasks if t["id"] == t2["id"])["title"] == "Fixed Beta"

    def test_falls_back_to_proposed_titles_when_nudge_still_has_duplicates(
        self, tmp_path: Path
    ) -> None:
        # When all nudge attempts still yield duplicate titles, the proposed
        # titles are applied anyway (#1713) — uniqueness during rewrites is
        # best-effort via the nudge, not a hard constraint.
        t1 = self._add(tmp_path, "Alpha")
        t2 = self._add(tmp_path, "Beta")
        dup_response = self._response(
            [
                {"id": t1["id"], "title": "Shared name", "description": ""},
                {"id": t2["id"], "title": "Shared name", "description": ""},
            ]
        )
        client = _client()
        client.run_turn.side_effect = [dup_response] * 4
        mock_prompts = MagicMock(spec=Prompts)
        mock_prompts.rescope_prompt.return_value = "prompt"
        mock_prompts.rescope_duplicate_nudge.return_value = "nudge"
        reorder_tasks(Tasks(tmp_path), "", agent=client, prompts=mock_prompts)
        assert client.run_turn.call_count == 4
        tasks = Tasks(tmp_path).list()
        assert next(t for t in tasks if t["id"] == t1["id"])["title"] == "Shared name"
        assert next(t for t in tasks if t["id"] == t2["id"])["title"] == "Shared name"

    def test_attempts_remaining_decrements_across_nudges(self, tmp_path: Path) -> None:
        from unittest.mock import call

        t1 = self._add(tmp_path, "Alpha")
        t2 = self._add(tmp_path, "Beta")
        dup_response = self._response(
            [
                {"id": t1["id"], "title": "Shared name", "description": ""},
                {"id": t2["id"], "title": "Shared name", "description": ""},
            ]
        )
        client = _client()
        client.run_turn.side_effect = [dup_response] * 4
        mock_prompts = MagicMock(spec=Prompts)
        mock_prompts.rescope_prompt.return_value = "prompt"
        mock_prompts.rescope_duplicate_nudge.return_value = "nudge"
        reorder_tasks(Tasks(tmp_path), "", agent=client, prompts=mock_prompts)
        # Nudge calls: attempts_remaining 2 → 1 → 0
        assert mock_prompts.rescope_duplicate_nudge.call_count == 3
        calls = mock_prompts.rescope_duplicate_nudge.call_args_list
        assert calls[0] == call(["Shared name"], attempts_remaining=2)
        assert calls[1] == call(["Shared name"], attempts_remaining=1)
        assert calls[2] == call(["Shared name"], attempts_remaining=0)

    def test_proceeds_with_pre_nudge_titles_when_nudge_returns_empty(
        self, tmp_path: Path
    ) -> None:
        # An empty nudge response stops the retry loop and the most recent
        # parseable proposal (the duplicate one) is applied (#1713).
        t1 = self._add(tmp_path, "Alpha")
        t2 = self._add(tmp_path, "Beta")
        dup_response = self._response(
            [
                {"id": t1["id"], "title": "Shared name", "description": ""},
                {"id": t2["id"], "title": "Shared name", "description": ""},
            ]
        )
        client = _client()
        client.run_turn.side_effect = [dup_response, ""]
        mock_prompts = MagicMock(spec=Prompts)
        mock_prompts.rescope_prompt.return_value = "prompt"
        mock_prompts.rescope_duplicate_nudge.return_value = "nudge"
        reorder_tasks(Tasks(tmp_path), "", agent=client, prompts=mock_prompts)
        tasks = Tasks(tmp_path).list()
        assert next(t for t in tasks if t["id"] == t1["id"])["title"] == "Shared name"
        assert next(t for t in tasks if t["id"] == t2["id"])["title"] == "Shared name"

    def test_proceeds_with_pre_nudge_titles_when_nudge_response_unparseable(
        self, tmp_path: Path
    ) -> None:
        # Unparseable nudge response: same fallback as empty (#1713).
        t1 = self._add(tmp_path, "Alpha")
        t2 = self._add(tmp_path, "Beta")
        dup_response = self._response(
            [
                {"id": t1["id"], "title": "Shared name", "description": ""},
                {"id": t2["id"], "title": "Shared name", "description": ""},
            ]
        )
        client = _client()
        client.run_turn.side_effect = [dup_response, "not json"]
        mock_prompts = MagicMock(spec=Prompts)
        mock_prompts.rescope_prompt.return_value = "prompt"
        mock_prompts.rescope_duplicate_nudge.return_value = "nudge"
        reorder_tasks(Tasks(tmp_path), "", agent=client, prompts=mock_prompts)
        tasks = Tasks(tmp_path).list()
        assert next(t for t in tasks if t["id"] == t1["id"])["title"] == "Shared name"
        assert next(t for t in tasks if t["id"] == t2["id"])["title"] == "Shared name"

    def test_no_nudge_when_titles_all_unique(self, tmp_path: Path) -> None:
        t1 = self._add(tmp_path, "Alpha")
        t2 = self._add(tmp_path, "Beta")
        raw = self._response(
            [
                {"id": t1["id"], "title": "New Alpha", "description": ""},
                {"id": t2["id"], "title": "New Beta", "description": ""},
            ]
        )
        client = _client(raw)
        mock_prompts = MagicMock(spec=Prompts)
        mock_prompts.rescope_prompt.return_value = "prompt"
        reorder_tasks(Tasks(tmp_path), "", agent=client, prompts=mock_prompts)
        mock_prompts.rescope_duplicate_nudge.assert_not_called()
        assert client.run_turn.call_count == 1


class TestReorderTasksVerdictWiring:
    """INV-D #1812: reorder_tasks uses the verdict envelope when intents present."""

    def _add(self, tmp_path: Path, title: str) -> dict[str, Any]:
        return Tasks(tmp_path).add(title=title, task_type=TaskType.SPEC)

    def test_uses_verdict_prompt_when_intents_present(self, tmp_path: Path) -> None:
        self._add(tmp_path, "A")
        intents = [
            RescopeIntent(
                change_request="add B",
                comment_id=101,
                timestamp="2024-01-15T10:00:00+00:00",
            )
        ]
        raw = json.dumps({"verdicts": [{"intent_comment_id": 101, "outcome": "no_op"}]})
        mock_prompts = MagicMock(spec=Prompts)
        mock_prompts.rescope_prompt_verdicts.return_value = "VERDICT PROMPT"
        # legacy ops prompt MUST NOT be invoked
        mock_prompts.rescope_prompt = MagicMock()
        reorder_tasks(
            Tasks(tmp_path),
            "",
            agent=_client(raw),
            prompts=mock_prompts,
            intents=intents,
        )
        mock_prompts.rescope_prompt_verdicts.assert_called_once()
        mock_prompts.rescope_prompt.assert_not_called()

    def test_uses_ops_prompt_when_no_intents(self, tmp_path: Path) -> None:
        self._add(tmp_path, "A")
        raw = json.dumps({"operations": []})
        mock_prompts = MagicMock(spec=Prompts)
        mock_prompts.rescope_prompt.return_value = "OPS PROMPT"
        mock_prompts.rescope_prompt_verdicts = MagicMock()
        reorder_tasks(Tasks(tmp_path), "", agent=_client(raw), prompts=mock_prompts)
        mock_prompts.rescope_prompt.assert_called_once()
        mock_prompts.rescope_prompt_verdicts.assert_not_called()

    def test_verdict_ops_flattened_and_applied(self, tmp_path: Path) -> None:
        # The op inside the verdict reaches the apply path and the
        # task gets renamed.
        t1 = self._add(tmp_path, "Original")
        intents = [
            RescopeIntent(
                change_request="rename",
                comment_id=101,
                timestamp="2024-01-15T10:00:00+00:00",
            )
        ]
        raw = json.dumps(
            {
                "verdicts": [
                    {
                        "intent_comment_id": 101,
                        "outcome": "reshaped",
                        "ops": [
                            {
                                "op": "rewrite",
                                "id": t1["id"],
                                "title": "Renamed",
                                "description": "",
                            }
                        ],
                        "affected_task_ids": [t1["id"]],
                        "narrative": "renamed",
                    }
                ]
            }
        )
        reorder_tasks(Tasks(tmp_path), "", agent=_client(raw), intents=intents)
        result = Tasks(tmp_path).list()
        assert any(t["id"] == t1["id"] and t["title"] == "Renamed" for t in result)

    def test_verdict_parse_error_nudges_with_verdict_nudge(
        self, tmp_path: Path
    ) -> None:
        # When the response isn't a valid verdict envelope, the
        # verdict-shaped parse nudge fires (not the ops-shape nudge).
        self._add(tmp_path, "A")
        intents = [
            RescopeIntent(
                change_request="x",
                comment_id=101,
                timestamp="2024-01-15T10:00:00+00:00",
            )
        ]
        bad_raw = '{"operations": []}'  # ops envelope, not verdicts
        valid_raw = json.dumps(
            {"verdicts": [{"intent_comment_id": 101, "outcome": "no_op"}]}
        )
        agent = _client()
        agent.run_turn.side_effect = [bad_raw, valid_raw]
        mock_prompts = MagicMock(spec=Prompts)
        mock_prompts.rescope_prompt_verdicts.return_value = "VERDICT PROMPT"
        mock_prompts.rescope_verdicts_parse_nudge.return_value = "VERDICT NUDGE"
        reorder_tasks(
            Tasks(tmp_path),
            "",
            agent=agent,
            prompts=mock_prompts,
            intents=intents,
        )
        mock_prompts.rescope_verdicts_parse_nudge.assert_called_once()
        # ops-shape nudge MUST NOT be invoked in verdict mode.
        assert (
            not mock_prompts.rescope_parse_nudge.called
            if hasattr(mock_prompts, "rescope_parse_nudge")
            else True
        )

    def test_contributing_intents_attributed_from_verdict(self, tmp_path: Path) -> None:
        # INV-F (#1804): the OpInput list the ``_on_rescope_apply``
        # callback receives carries the contributing_intents derived
        # from the verdict's intent_comment_id so the oracle's
        # later-cross-author rule sees correct attribution.
        t1 = self._add(tmp_path, "Original")
        intents = [
            RescopeIntent(
                change_request="rename",
                comment_id=42,
                timestamp="2024-01-15T10:00:00+00:00",
            )
        ]
        raw = json.dumps(
            {
                "verdicts": [
                    {
                        "intent_comment_id": 42,
                        "outcome": "reshaped",
                        "ops": [
                            {
                                "op": "rewrite",
                                "id": t1["id"],
                                "title": "Renamed",
                                "description": "",
                            }
                        ],
                        "affected_task_ids": [t1["id"]],
                        "narrative": "x",
                    }
                ]
            }
        )
        captured: list[tuple[list, list, dict, list]] = []
        reorder_tasks(
            Tasks(tmp_path),
            "",
            agent=_client(raw),
            intents=intents,
            _on_rescope_apply=lambda r, o, m, v: captured.append((r, o, m, v)),
        )
        assert len(captured) == 1
        _, op_inputs, _, _ = captured[0]
        # The op_inputs carry contributing_intents threaded from the
        # verdict — at least one op should list intent 42.
        assert any(42 in op.oi_intents for op in op_inputs)

    def test_explicit_op_attribution_unioned_with_verdict_attribution(
        self, tmp_path: Path
    ) -> None:
        # Per-op ``contributing_intents`` provided by Opus is preserved
        # alongside the verdict's own ``intent_comment_id`` (union, not
        # replacement) so a hand-attributed op doesn't lose its
        # verdict-level provenance.
        t1 = self._add(tmp_path, "Original")
        intents = [
            RescopeIntent(
                change_request="x",
                comment_id=10,
                timestamp="2024-01-15T10:00:00+00:00",
            ),
            RescopeIntent(
                change_request="y",
                comment_id=20,
                timestamp="2024-01-15T10:01:00+00:00",
            ),
        ]
        # Opus attributes the rewrite to intent 20 explicitly, and
        # the verdict pertains to intent 10 (joint-honor case).
        raw = json.dumps(
            {
                "verdicts": [
                    {
                        "intent_comment_id": 10,
                        "outcome": "reshaped",
                        "ops": [
                            {
                                "op": "rewrite",
                                "id": t1["id"],
                                "title": "Renamed",
                                "description": "",
                                "contributing_intents": [20],
                            }
                        ],
                        "affected_task_ids": [t1["id"]],
                        "narrative": "joint-honored",
                    },
                    {
                        "intent_comment_id": 20,
                        "outcome": "honored",
                        "affected_task_ids": [t1["id"]],
                    },
                ]
            }
        )
        captured: list[tuple[list, list, dict, list]] = []
        reorder_tasks(
            Tasks(tmp_path),
            "",
            agent=_client(raw),
            intents=intents,
            _on_rescope_apply=lambda r, o, m, v: captured.append((r, o, m, v)),
        )
        _, op_inputs, _, _ = captured[0]
        # Both intents (10 from verdict, 20 from explicit attribution)
        # appear on the rewrite op via the union.
        union_intents = {i for op in op_inputs for i in op.oi_intents}
        assert {10, 20}.issubset(union_intents)

    def test_verdict_mode_skips_duplicate_title_nudge(self, tmp_path: Path) -> None:
        # In verdict mode, the duplicate-title nudge loop is skipped
        # entirely (its nudge text would shift Opus mid-conversation
        # from the verdict envelope back to ops).  Verify by mocking
        # rescope_duplicate_nudge and asserting it's never invoked,
        # even when the verdict produces a duplicate title in the
        # batch.
        intents = [
            RescopeIntent(
                change_request="x",
                comment_id=101,
                timestamp="2024-01-15T10:00:00+00:00",
            )
        ]
        raw = json.dumps(
            {
                "verdicts": [
                    {
                        "intent_comment_id": 101,
                        "outcome": "honored",
                        "ops": [
                            {
                                "op": "new",
                                "title": "Same",
                                "description": "first",
                                "type": "spec",
                            },
                            {
                                "op": "new",
                                "title": "Same",
                                "description": "second",
                                "type": "spec",
                            },
                        ],
                        "affected_task_ids": [],
                    }
                ]
            }
        )
        mock_prompts = MagicMock(spec=Prompts)
        mock_prompts.rescope_prompt_verdicts.return_value = "VERDICT PROMPT"
        reorder_tasks(
            Tasks(tmp_path),
            "",
            agent=_client(raw),
            prompts=mock_prompts,
            intents=intents,
        )
        # The duplicate-title nudge MUST NOT have been invoked —
        # that's the invariant for verdict mode regardless of what
        # ``_make_new_tasks_from_opus`` decides downstream.
        mock_prompts.rescope_duplicate_nudge.assert_not_called()


class TestFlattenVerdictsToOps:
    """Pure helper used by reorder_tasks's verdict path (#1812 INV-D)."""

    def test_empty_verdicts_yields_empty_list(self) -> None:
        from fido.tasks import _flatten_verdicts_to_ops

        assert _flatten_verdicts_to_ops([]) == []

    def test_single_verdict_op_inherits_intent_attribution(self) -> None:
        from fido.tasks import _flatten_verdicts_to_ops
        from fido.types import IntentVerdict

        v = IntentVerdict(
            intent_comment_id=5,
            outcome="reshaped",
            ops=({"op": "rewrite", "id": "T1", "title": "x"},),
            affected_task_ids=("T1",),
            narrative="x",
        )
        flat = _flatten_verdicts_to_ops([v])
        assert len(flat) == 1
        assert flat[0]["contributing_intents"] == [5]
        assert flat[0]["op"] == "rewrite"

    def test_explicit_attribution_unioned_with_verdict_id(self) -> None:
        from fido.tasks import _flatten_verdicts_to_ops
        from fido.types import IntentVerdict

        v = IntentVerdict(
            intent_comment_id=5,
            outcome="reshaped",
            ops=(
                {
                    "op": "rewrite",
                    "id": "T1",
                    "title": "x",
                    "contributing_intents": [9, 11],
                },
            ),
            affected_task_ids=("T1",),
            narrative="x",
        )
        flat = _flatten_verdicts_to_ops([v])
        assert flat[0]["contributing_intents"] == [5, 9, 11]

    def test_duplicate_attribution_deduped(self) -> None:
        from fido.tasks import _flatten_verdicts_to_ops
        from fido.types import IntentVerdict

        v = IntentVerdict(
            intent_comment_id=5,
            outcome="reshaped",
            ops=(
                {
                    "op": "rewrite",
                    "id": "T1",
                    "title": "x",
                    "contributing_intents": [5],
                },
            ),
            affected_task_ids=("T1",),
            narrative="x",
        )
        flat = _flatten_verdicts_to_ops([v])
        # Verdict id 5 already in explicit attribution; result has it
        # once, not twice.
        assert flat[0]["contributing_intents"] == [5]

    def test_multiple_verdicts_flatten_in_order(self) -> None:
        from fido.tasks import _flatten_verdicts_to_ops
        from fido.types import IntentVerdict

        verdicts = [
            IntentVerdict(
                intent_comment_id=1,
                outcome="honored",
                ops=({"op": "keep", "id": "A"},),
                affected_task_ids=("A",),
            ),
            IntentVerdict(
                intent_comment_id=2,
                outcome="honored",
                ops=(
                    {"op": "rewrite", "id": "B", "title": "x"},
                    {"op": "remove", "id": "C"},
                ),
                affected_task_ids=("B", "C"),
            ),
        ]
        flat = _flatten_verdicts_to_ops(verdicts)
        # Three ops total, in verdict order then op order.
        assert [op["op"] for op in flat] == ["keep", "rewrite", "remove"]
        assert flat[0]["contributing_intents"] == [1]
        assert flat[1]["contributing_intents"] == [2]
        assert flat[2]["contributing_intents"] == [2]

    def test_malformed_int_contributing_intents_passed_through(self) -> None:
        # codex P1 on PR #1813: ``contributing_intents: 42`` (bare int)
        # would have crashed ``{*existing, ...}`` with TypeError.  Pass
        # through untouched so the op parser raises a clean schema
        # error.
        from fido.tasks import _flatten_verdicts_to_ops
        from fido.types import IntentVerdict

        v = IntentVerdict(
            intent_comment_id=5,
            outcome="honored",
            ops=(
                {
                    "op": "keep",
                    "id": "T1",
                    "contributing_intents": 42,  # malformed: int not list
                },
            ),
            affected_task_ids=("T1",),
        )
        flat = _flatten_verdicts_to_ops([v])
        # Malformed value passed through; verdict id NOT injected.
        assert flat[0]["contributing_intents"] == 42

    def test_malformed_string_contributing_intents_passed_through(self) -> None:
        # codex P2 on PR #1813: ``""`` was previously collapsed to []
        # via ``or []``; now passed through.
        from fido.tasks import _flatten_verdicts_to_ops
        from fido.types import IntentVerdict

        v = IntentVerdict(
            intent_comment_id=5,
            outcome="honored",
            ops=({"op": "keep", "id": "T1", "contributing_intents": ""},),
            affected_task_ids=("T1",),
        )
        flat = _flatten_verdicts_to_ops([v])
        assert flat[0]["contributing_intents"] == ""

    def test_malformed_dict_contributing_intents_passed_through(self) -> None:
        # codex P2 on PR #1813: ``{10: "x"}`` was previously coerced
        # to ``[10, 5]`` via set-union over dict keys.
        from fido.tasks import _flatten_verdicts_to_ops
        from fido.types import IntentVerdict

        v = IntentVerdict(
            intent_comment_id=5,
            outcome="honored",
            ops=(
                {
                    "op": "keep",
                    "id": "T1",
                    "contributing_intents": {"10": "x"},
                },
            ),
            affected_task_ids=("T1",),
        )
        flat = _flatten_verdicts_to_ops([v])
        assert flat[0]["contributing_intents"] == {"10": "x"}

    def test_mixed_type_list_contributing_intents_passed_through(self) -> None:
        # codex P1 on PR #1813: ``[10, "bad"]`` would have raised in
        # ``sorted({...})`` because mixed types aren't orderable.  Pass
        # through.
        from fido.tasks import _flatten_verdicts_to_ops
        from fido.types import IntentVerdict

        v = IntentVerdict(
            intent_comment_id=5,
            outcome="honored",
            ops=(
                {
                    "op": "keep",
                    "id": "T1",
                    "contributing_intents": [10, "bad"],
                },
            ),
            affected_task_ids=("T1",),
        )
        flat = _flatten_verdicts_to_ops([v])
        # After IntentVerdict's deep_freeze, the list became a tuple.
        # Pass-through preserves it as-is.
        assert flat[0]["contributing_intents"] == (10, "bad")

    def test_bool_in_list_contributing_intents_passed_through(self) -> None:
        # ``bool`` is an ``int`` subclass; reject explicitly so
        # ``True``/``False`` can't slip in as attribution.
        from fido.tasks import _flatten_verdicts_to_ops
        from fido.types import IntentVerdict

        v = IntentVerdict(
            intent_comment_id=5,
            outcome="honored",
            ops=(
                {
                    "op": "keep",
                    "id": "T1",
                    "contributing_intents": [True, 10],
                },
            ),
            affected_task_ids=("T1",),
        )
        flat = _flatten_verdicts_to_ops([v])
        # After deep_freeze, list → tuple; bool-containing → passed through.
        assert flat[0]["contributing_intents"] == (True, 10)

    def test_falsy_zero_contributing_intents_passed_through(self) -> None:
        # ``0`` is falsy AND not a list; the old ``or []`` would have
        # collapsed it.  Passed through now.
        from fido.tasks import _flatten_verdicts_to_ops
        from fido.types import IntentVerdict

        v = IntentVerdict(
            intent_comment_id=5,
            outcome="honored",
            ops=({"op": "keep", "id": "T1", "contributing_intents": 0},),
            affected_task_ids=("T1",),
        )
        flat = _flatten_verdicts_to_ops([v])
        assert flat[0]["contributing_intents"] == 0


# ── _build_op_inputs (INV-F adapter glue) ─────────────────────────────────────


class TestBuildOpInputs:
    """Adapter glue covering every ``_RescopeOp`` variant the
    INV-F notifier hands to the oracle."""

    def test_each_op_kind_round_trips_with_contributing_intents(self) -> None:
        from fido.rocq import task_queue_rescope as oracle
        from fido.tasks import (
            _build_op_inputs,
            _RescopeOpKeep,
            _RescopeOpMerge,
            _RescopeOpNew,
            _RescopeOpRemove,
            _RescopeOpRewrite,
            _RescopeOpRewriteAnchor,
            _RescopeOpSplit,
            _RescopeOpSplitChild,
        )

        ids = {"k": 1, "rw": 2, "ra": 3, "rm": 4, "merge": 5, "src1": 6, "split": 7}
        ops = [
            _RescopeOpKeep(id="k", contributing_intents=(101,)),
            _RescopeOpRewrite(
                id="rw",
                title="new",
                description="d",
                contributing_intents=(102,),
            ),
            _RescopeOpRewriteAnchor(
                id="ra", anchor_comment_id=999, contributing_intents=(103,)
            ),
            _RescopeOpRemove(id="rm", contributing_intents=(104,)),
            _RescopeOpMerge(
                target_id="merge",
                sources=["src1"],
                title="m",
                description="d",
                contributing_intents=(105,),
            ),
            _RescopeOpSplit(
                id="split",
                children=[_RescopeOpSplitChild(title="c", description="")],
                contributing_intents=(106,),
            ),
            _RescopeOpNew(
                title="brand new",
                description="d",
                type="spec",
                contributing_intents=(107,),
            ),
        ]
        op_inputs = _build_op_inputs(ops, ids)
        # _RescopeOpNew is skipped (no existing task id), so six entries.
        assert len(op_inputs) == 6
        kinds = [type(oi.oi_op).__name__ for oi in op_inputs]
        assert kinds == [
            "KeepTask",
            "RewriteTask",
            "RewriteAnchor",
            "CompleteTask",
            "MergeTasks",
            "SplitTask",
        ]
        # Contributing intents thread through unchanged.
        intents = [oi.oi_intents for oi in op_inputs]
        assert intents == [[101], [102], [103], [104], [105], [106]]
        # Split children carry through with placeholder ids (the oracle
        # treats SplitTask as EffectReorganize on the source regardless).
        split_op = op_inputs[5].oi_op
        assert isinstance(split_op, oracle.SplitTask)
        assert len(split_op.children) == 1


# ── _compute_thread_changes ───────────────────────────────────────────────────


class TestComputeThreadChanges:
    def _t(
        self,
        task_id: str,
        title: str,
        status: str = "pending",
        thread: dict | None = None,
        description: str = "",
    ) -> dict:
        t: dict = {
            "id": task_id,
            "title": title,
            "type": "thread" if thread else "spec",
            "status": status,
            "description": description,
        }
        if thread:
            t["thread"] = thread
        return t

    def _thread(self) -> dict:
        return {"repo": "r/r", "pr": 1, "comment_id": 42, "url": "https://x.com"}

    def test_completed_thread_task(self) -> None:
        original = [self._t("1", "Thread task", thread=self._thread())]
        result: list = []
        changes = _compute_thread_changes(original, result, frozenset({"1"}))
        assert len(changes) == 1
        assert changes[0]["kind"] == "completed"
        assert changes[0]["task"]["id"] == "1"

    def test_modified_title(self) -> None:
        original = [self._t("1", "Old title", thread=self._thread())]
        result = [self._t("1", "New title", thread=self._thread())]
        changes = _compute_thread_changes(original, result, frozenset({"1"}))
        assert len(changes) == 1
        assert changes[0]["kind"] == "modified"
        assert changes[0]["new_title"] == "New title"

    def test_description_only_change_is_silent(self) -> None:
        """Pure description rewrites (title unchanged) are internal
        rescope rephrasing — no change record, no reply-back to the
        commenter.  Only title changes signal a material scope shift
        worth notifying the reviewer about (#1388)."""
        original = [self._t("1", "Task", thread=self._thread(), description="old")]
        result = [self._t("1", "Task", thread=self._thread(), description="new")]
        changes = _compute_thread_changes(original, result, frozenset({"1"}))
        assert changes == []

    def test_unchanged_thread_task_not_reported(self) -> None:
        t = self._t("1", "Task", thread=self._thread())
        changes = _compute_thread_changes([t], [t], frozenset({"1"}))
        assert changes == []

    def test_change_record_carries_post_rescope_contributing_intents(self) -> None:
        # #1722: change records expose contributing_intents from the
        # post-rescope task so the future classifier (#1723) and
        # per-intent notifier (#1724) can route replies.  Both
        # "completed" and "modified" records carry the field.
        thread = self._thread()
        original = [self._t("1", "Old", thread=thread)]
        result_modified = [self._t("1", "New", thread=thread)]
        result_modified[0]["contributing_intents"] = [42, 99]
        modified = _compute_thread_changes(original, result_modified, frozenset({"1"}))
        assert modified[0]["kind"] == "modified"
        assert modified[0]["contributing_intents"] == [42, 99]
        # Same for completion records.
        result_completed: list = []
        completed = _compute_thread_changes(
            original, result_completed, frozenset({"1"})
        )
        assert completed[0]["kind"] == "completed"
        # The original task had no contributing_intents and the result
        # is missing entirely — empty list, not None.
        assert completed[0]["contributing_intents"] == []

    def test_spec_task_not_reported_even_if_dropped(self) -> None:
        t = self._t("1", "Spec task")  # no thread
        changes = _compute_thread_changes([t], [], frozenset({"1"}))
        assert changes == []

    def test_completed_task_not_reported(self) -> None:
        t = self._t("1", "Done", status="completed", thread=self._thread())
        changes = _compute_thread_changes([t], [], frozenset({"1"}))
        assert changes == []

    def test_task_not_in_original_ids_not_reported(self) -> None:
        t = self._t("1", "New task", thread=self._thread())
        changes = _compute_thread_changes([t], [], frozenset())
        assert changes == []

    def test_multiple_changes_returned(self) -> None:
        thread = self._thread()
        t1 = self._t("1", "Covered by commit", thread=thread)
        t2 = self._t("2", "Old", thread=thread)
        r2 = self._t("2", "New", thread=thread)
        changes = _compute_thread_changes([t1, t2], [r2], frozenset({"1", "2"}))
        kinds = {c["kind"] for c in changes}
        assert "completed" in kinds
        assert "modified" in kinds


class TestTasks:
    def test_modify_yields_empty_list_when_no_tasks(self, tmp_path: Path) -> None:
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        with Tasks(work_dir).modify() as tasks:
            assert tasks == []

    def test_modify_persists_changes(self, tmp_path: Path) -> None:
        from fido.types import TaskType

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        Tasks(work_dir).add("Existing task", TaskType.SPEC)
        with Tasks(work_dir).modify() as tasks:
            tasks[0]["title"] = "Modified task"
        assert Tasks(work_dir).list()[0]["title"] == "Modified task"

    def test_modify_raises_on_corrupt_json(self, tmp_path: Path) -> None:
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        _task_file(work_dir).write_text("not json")
        with pytest.raises(ValueError, match="corrupt tasks.json"):
            with Tasks(work_dir).modify() as _:
                pass

    def test_modify_raises_on_missing_type_field(self, tmp_path: Path) -> None:
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        _task_file(work_dir).write_text(
            '[{"id": "bad", "title": "no type", "status": "pending"}]'
        )
        with pytest.raises(ValueError, match="missing required type field"):
            with Tasks(work_dir).modify() as _:
                pass


class TestTasksCompleteWithResolve:
    """Tests for Tasks.complete_with_resolve — the unified complete+thread-resolve path."""

    def _work_dir(self, tmp_path: Path) -> Path:
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        return work_dir

    def test_marks_task_completed(self, tmp_path: Path) -> None:
        work_dir = self._work_dir(tmp_path)
        task = Tasks(work_dir).add("task", TaskType.SPEC)
        Tasks(work_dir).complete_with_resolve(
            task["id"], MagicMock(), syncer=lambda _w, _g: None
        )
        assert Tasks(work_dir).list()[0]["status"] == "completed"

    def test_no_thread_does_not_call_github(self, tmp_path: Path) -> None:
        work_dir = self._work_dir(tmp_path)
        task = Tasks(work_dir).add("task", TaskType.SPEC)
        gh = MagicMock()
        Tasks(work_dir).complete_with_resolve(
            task["id"], gh, syncer=lambda _w, _g: None
        )
        gh.get_user.assert_not_called()
        gh.resolve_thread.assert_not_called()

    def test_thread_missing_fields_skips_resolve(self, tmp_path: Path) -> None:
        """Thread dict with missing pr/comment_id skips resolution."""
        work_dir = self._work_dir(tmp_path)
        task = Tasks(work_dir).add("task", TaskType.THREAD, thread={"repo": "a/b"})
        gh = MagicMock()
        Tasks(work_dir).complete_with_resolve(
            task["id"], gh, syncer=lambda _w, _g: None
        )
        gh.resolve_thread.assert_not_called()

    def test_nonexistent_id_does_not_raise(self, tmp_path: Path) -> None:
        work_dir = self._work_dir(tmp_path)
        Tasks(work_dir).complete_with_resolve(
            "nonexistent-id", MagicMock(), syncer=lambda _w, _g: None
        )

    def test_triggers_background_sync(self, tmp_path: Path) -> None:
        """Every completion fires sync_tasks_background so the PR body
        checkbox flips even when the worker loop doesn't sync between
        the completion and the PR-ready/merge step (#988)."""
        work_dir = self._work_dir(tmp_path)
        task = Tasks(work_dir).add("task", TaskType.SPEC)
        gh = MagicMock()
        sync_calls: list[tuple[Any, Any]] = []
        Tasks(work_dir).complete_with_resolve(
            task["id"],
            gh,
            syncer=lambda wd, g: sync_calls.append((wd, g)),
        )
        assert sync_calls == [(work_dir, gh)]

    def test_triggers_background_sync_even_for_unknown_id(self, tmp_path: Path) -> None:
        """Sync still fires even when the task_id isn't found — caller
        intent ('I just tried to complete something') is the trigger."""
        work_dir = self._work_dir(tmp_path)
        gh = MagicMock()
        sync_calls: list[tuple[Any, Any]] = []
        Tasks(work_dir).complete_with_resolve(
            "nonexistent-id",
            gh,
            syncer=lambda wd, g: sync_calls.append((wd, g)),
        )
        assert sync_calls == [(work_dir, gh)]

    def test_resolves_thread_when_we_are_last(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        work_dir = self._work_dir(tmp_path)
        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        task = Tasks(work_dir).add("threaded task", TaskType.THREAD, thread=thread)

        gh = MagicMock()
        gh.get_user.return_value = "fido-bot"
        gh.get_review_threads.return_value = [
            {
                "id": "thread_node_abc",
                "isResolved": False,
                "comments": {
                    "nodes": [
                        {"databaseId": 42, "author": {"login": "a"}},
                        {"databaseId": 99, "author": {"login": "fido-bot"}},
                    ]
                },
            }
        ]

        with caplog.at_level(logging.INFO, logger="fido"):
            Tasks(work_dir).complete_with_resolve(
                task["id"], gh, syncer=lambda _w, _g: None
            )

        gh.resolve_thread.assert_called_once_with("thread_node_abc")
        assert "thread resolved: thread_node_abc" in caplog.text

    def test_skips_resolve_when_not_last_commenter(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        work_dir = self._work_dir(tmp_path)
        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        task = Tasks(work_dir).add("threaded task", TaskType.THREAD, thread=thread)

        gh = MagicMock()
        gh.get_user.return_value = "fido-bot"
        gh.get_review_threads.return_value = [
            {
                "id": "thread_node_abc",
                "isResolved": False,
                "comments": {
                    "nodes": [
                        {"databaseId": 42, "author": {"login": "fido-bot"}},
                        {"databaseId": 99, "author": {"login": "a"}},
                    ]
                },
            }
        ]

        with caplog.at_level(logging.INFO, logger="fido"):
            Tasks(work_dir).complete_with_resolve(
                task["id"], gh, syncer=lambda _w, _g: None
            )

        gh.resolve_thread.assert_not_called()
        assert "not resolving" in caplog.text

    def test_skips_resolve_when_no_matching_comments(self, tmp_path: Path) -> None:
        work_dir = self._work_dir(tmp_path)
        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        task = Tasks(work_dir).add("threaded task", TaskType.THREAD, thread=thread)

        gh = MagicMock()
        gh.get_user.return_value = "fido-bot"
        gh.get_review_threads.return_value = []

        Tasks(work_dir).complete_with_resolve(
            task["id"], gh, syncer=lambda _w, _g: None
        )

        gh.resolve_thread.assert_not_called()

    def test_resolves_thread_when_ignored_outsider_replied_after_us(
        self, tmp_path: Path
    ) -> None:
        work_dir = self._work_dir(tmp_path)
        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        task = Tasks(work_dir).add("threaded task", TaskType.THREAD, thread=thread)

        gh = MagicMock()
        gh.get_user.return_value = "fido-bot"
        gh.get_review_threads.return_value = [
            {
                "id": "thread_node_abc",
                "isResolved": False,
                "comments": {
                    "nodes": [
                        {"databaseId": 42, "author": {"login": "a"}},
                        {"databaseId": 99, "author": {"login": "fido-bot"}},
                        {"databaseId": 100, "author": {"login": "drive-by"}},
                    ]
                },
            }
        ]

        Tasks(work_dir).complete_with_resolve(
            task["id"], gh, syncer=lambda _w, _g: None
        )

        gh.resolve_thread.assert_called_once_with("thread_node_abc")

    def test_skips_resolve_when_thread_already_resolved(self, tmp_path: Path) -> None:
        work_dir = self._work_dir(tmp_path)
        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        task = Tasks(work_dir).add("threaded task", TaskType.THREAD, thread=thread)

        gh = MagicMock()
        gh.get_user.return_value = "fido-bot"
        gh.get_review_threads.return_value = [
            {
                "id": "thread_node_abc",
                "isResolved": True,
                "comments": {
                    "nodes": [{"databaseId": 42, "author": {"login": "fido-bot"}}]
                },
            }
        ]

        Tasks(work_dir).complete_with_resolve(
            task["id"], gh, syncer=lambda _w, _g: None
        )

        gh.resolve_thread.assert_not_called()

    def test_skips_resolve_when_pending_sibling_task_remains(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        work_dir = self._work_dir(tmp_path)
        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        task = Tasks(work_dir).add("threaded task", TaskType.THREAD, thread=thread)
        Tasks(work_dir).add(
            "pending sibling",
            TaskType.THREAD,
            thread={"repo": "a/b", "pr": 1, "comment_id": 77},
        )

        gh = MagicMock()
        gh.get_user.return_value = "fido-bot"
        gh.get_review_threads.return_value = [
            {
                "id": "thread_node_abc",
                "isResolved": False,
                "comments": {
                    "nodes": [
                        {"databaseId": 42, "author": {"login": "a"}},
                        {"databaseId": 77, "author": {"login": "a"}},
                        {"databaseId": 99, "author": {"login": "fido-bot"}},
                    ]
                },
            }
        ]

        with caplog.at_level(logging.INFO, logger="fido"):
            Tasks(work_dir).complete_with_resolve(
                task["id"], gh, syncer=lambda _w, _g: None
            )

        gh.resolve_thread.assert_not_called()
        assert "pending same-thread work" in caplog.text

    def test_exception_silenced_and_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        work_dir = self._work_dir(tmp_path)
        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        task = Tasks(work_dir).add("threaded task", TaskType.THREAD, thread=thread)

        gh = MagicMock()
        gh.get_user.side_effect = RuntimeError("network error")

        with caplog.at_level(logging.WARNING, logger="fido"):
            Tasks(work_dir).complete_with_resolve(
                task["id"], gh, syncer=lambda _w, _g: None
            )
        assert "thread resolution skipped" in caplog.text
