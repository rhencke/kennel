import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fido.claude import ClaudeClient
from fido.prompts import Prompts
from fido.rocq import pr_body_task_store as oracle
from fido.tasks import (
    Tasks,
    _apply_reorder,
    _assert_rescope_matches_oracle,
    _compute_thread_changes,
    _find_duplicate_titles,
    _format_work_queue,
    _parse_reorder_response,
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
    add_task,
    complete_by_id,
    has_pending_tasks_for_comment,
    list_tasks,
    remove_task,
    reorder_tasks,
    unblock_tasks,
    update_task,
)
from fido.types import TaskStatus, TaskType


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
        assert store.task_store_rows[1].task_description == "one"
        assert store.task_store_rows[2].task_source_comment == 9

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

    def test_releases_encode_completion_description_updates_and_keeps(self) -> None:
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
        ids_by_task_id, _tasks_by_oracle_id, _order, _rows = _rescope_state_for_oracle(
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
        )

        assert [type(release.release_decision).__name__ for release in releases] == [
            "RewriteTask",
            "CompleteTask",
            "KeepTask",
        ]


class TestAddTask:
    def test_creates_file(self, tmp_path: Path) -> None:
        task = add_task(tmp_path, title="test task", task_type=TaskType.SPEC)
        assert task["title"] == "test task"
        assert task["status"] == "pending"
        assert task["type"] == "spec"
        tasks = list_tasks(tmp_path)
        assert len(tasks) == 1

    def test_appends(self, tmp_path: Path) -> None:
        add_task(tmp_path, title="one", task_type=TaskType.SPEC)
        add_task(tmp_path, title="two", task_type=TaskType.SPEC)
        tasks = list_tasks(tmp_path)
        assert len(tasks) == 2

    def test_with_thread(self, tmp_path: Path) -> None:
        thread = {"repo": "a/b", "pr": 1, "comment_id": 123}
        task = add_task(tmp_path, title="t", task_type=TaskType.THREAD, thread=thread)
        assert task["thread"] == thread

    def test_id_is_unique(self, tmp_path: Path) -> None:
        t1 = add_task(tmp_path, title="a", task_type=TaskType.SPEC)
        t2 = add_task(tmp_path, title="b", task_type=TaskType.SPEC)
        assert t1["id"] != t2["id"]

    def test_appends_at_end(self, tmp_path: Path) -> None:
        add_task(tmp_path, title="first", task_type=TaskType.SPEC)
        add_task(tmp_path, title="second", task_type=TaskType.SPEC)
        tasks = list_tasks(tmp_path)
        assert tasks[0]["title"] == "first"
        assert tasks[1]["title"] == "second"

    def test_returns_existing_pending_task_if_title_matches(
        self, tmp_path: Path
    ) -> None:
        t1 = add_task(tmp_path, title="duplicate task", task_type=TaskType.SPEC)
        t2 = add_task(tmp_path, title="duplicate task", task_type=TaskType.SPEC)
        assert t1["id"] == t2["id"]
        assert len(list_tasks(tmp_path)) == 1

    def test_sanitizes_multiline_title(self, tmp_path: Path) -> None:
        task = add_task(
            tmp_path,
            title="first line\n\nsecond paragraph\n- bullet\n- another",
            task_type=TaskType.SPEC,
        )
        assert task["title"] == "first line second paragraph - bullet - another"
        assert "\n" not in task["title"]

    def test_collapses_whitespace_in_title(self, tmp_path: Path) -> None:
        task = add_task(
            tmp_path,
            title="  too   many    spaces  \t\there  ",
            task_type=TaskType.SPEC,
        )
        assert task["title"] == "too many spaces here"

    def test_does_not_deduplicate_completed_tasks(self, tmp_path: Path) -> None:
        t1 = add_task(tmp_path, title="done task", task_type=TaskType.SPEC)
        complete_by_id(tmp_path, t1["id"])
        t2 = add_task(tmp_path, title="done task", task_type=TaskType.SPEC)
        assert t1["id"] != t2["id"]
        assert len(list_tasks(tmp_path)) == 2

    def test_thread_task_appends_at_end(self, tmp_path: Path) -> None:
        add_task(tmp_path, title="existing", task_type=TaskType.SPEC)
        thread = {"repo": "r/r", "pr": 1, "comment_id": 42}
        add_task(
            tmp_path, title="comment task", task_type=TaskType.THREAD, thread=thread
        )
        tasks = list_tasks(tmp_path)
        assert tasks[0]["title"] == "existing"
        assert tasks[1]["title"] == "comment task"

    def test_deduplicates_by_comment_id(self, tmp_path: Path) -> None:
        thread = {"repo": "r/r", "pr": 1, "comment_id": 99}
        t1 = add_task(
            tmp_path, title="first title", task_type=TaskType.THREAD, thread=thread
        )
        t2 = add_task(
            tmp_path, title="second title", task_type=TaskType.THREAD, thread=thread
        )
        assert t1["id"] == t2["id"]
        assert len(list_tasks(tmp_path)) == 1

    def test_deduplicates_by_comment_id_even_when_completed(
        self, tmp_path: Path
    ) -> None:
        thread = {"repo": "r/r", "pr": 1, "comment_id": 55}
        t1 = add_task(
            tmp_path, title="handle feedback", task_type=TaskType.THREAD, thread=thread
        )
        complete_by_id(tmp_path, t1["id"])
        t2 = add_task(
            tmp_path,
            title="handle feedback again",
            task_type=TaskType.THREAD,
            thread=thread,
        )
        assert t1["id"] == t2["id"]
        assert len(list_tasks(tmp_path)) == 1

    def test_different_comment_ids_are_not_deduplicated(self, tmp_path: Path) -> None:
        t1 = add_task(
            tmp_path,
            title="t",
            task_type=TaskType.THREAD,
            thread={"repo": "r/r", "pr": 1, "comment_id": 1},
        )
        t2 = add_task(
            tmp_path,
            title="t",
            task_type=TaskType.THREAD,
            thread={"repo": "r/r", "pr": 1, "comment_id": 2},
        )
        assert t1["id"] != t2["id"]
        assert len(list_tasks(tmp_path)) == 2

    def test_task_type_required(self, tmp_path: Path) -> None:
        import pytest

        with pytest.raises(TypeError, match="task_type must be TaskType"):
            add_task(tmp_path, title="t", task_type="spec")  # type: ignore[arg-type]


class TestUpdateTask:
    def test_updates_status(self, tmp_path: Path) -> None:
        task = add_task(tmp_path, title="t", task_type=TaskType.SPEC)
        assert update_task(tmp_path, task["id"], TaskStatus.COMPLETED)
        tasks = list_tasks(tmp_path)
        assert tasks[0]["status"] == "completed"

    def test_returns_false_if_not_found(self, tmp_path: Path) -> None:
        add_task(tmp_path, title="t", task_type=TaskType.SPEC)
        assert not update_task(tmp_path, "nonexistent", TaskStatus.COMPLETED)


class TestListTasks:
    def test_empty(self, tmp_path: Path) -> None:
        _task_file(tmp_path).write_text("[]")
        assert list_tasks(tmp_path) == []

    def test_corrupt_json(self, tmp_path: Path) -> None:
        _task_file(tmp_path).write_text("not json")
        with pytest.raises(ValueError, match="corrupt tasks.json"):
            list_tasks(tmp_path)

    def test_raises_on_missing_type_field(self, tmp_path: Path) -> None:
        tf = _task_file(tmp_path)
        tf.write_text('[{"id": "bad", "title": "no type", "status": "pending"}]')
        with pytest.raises(ValueError, match="missing required type field"):
            list_tasks(tmp_path)


class TestCompleteById:
    def test_marks_completed_returns_none_when_no_thread(self, tmp_path: Path) -> None:
        task = add_task(tmp_path, title="do something", task_type=TaskType.SPEC)
        result = complete_by_id(tmp_path, task["id"])
        assert result is None
        assert list_tasks(tmp_path)[0]["status"] == "completed"

    def test_returns_thread_when_present(self, tmp_path: Path) -> None:
        thread = {"repo": "a/b", "pr": 1, "comment_id": 99}
        task = add_task(tmp_path, title="t", task_type=TaskType.THREAD, thread=thread)
        result = complete_by_id(tmp_path, task["id"])
        assert result == thread

    def test_returns_none_when_not_found(self, tmp_path: Path) -> None:
        add_task(tmp_path, title="other", task_type=TaskType.SPEC)
        assert complete_by_id(tmp_path, "missing") is None
        assert list_tasks(tmp_path)[0]["status"] == "pending"

    def test_skips_already_completed(self, tmp_path: Path) -> None:
        task = add_task(tmp_path, title="t", task_type=TaskType.SPEC)
        complete_by_id(tmp_path, task["id"])
        # second call on already-completed task returns None
        assert complete_by_id(tmp_path, task["id"]) is None

    def test_completes_correct_task_by_id(self, tmp_path: Path) -> None:
        t1 = add_task(tmp_path, title="a", task_type=TaskType.SPEC)
        add_task(tmp_path, title="b", task_type=TaskType.SPEC)
        complete_by_id(tmp_path, t1["id"])
        tasks = list_tasks(tmp_path)
        assert tasks[0]["status"] == "completed"
        assert tasks[1]["status"] == "pending"


class TestHasPendingTasksForComment:
    def test_returns_true_when_pending_task_exists(self, tmp_path: Path) -> None:
        add_task(
            tmp_path,
            title="t",
            task_type=TaskType.THREAD,
            thread={"repo": "r/r", "pr": 1, "comment_id": 42},
        )
        assert has_pending_tasks_for_comment(tmp_path, 42)

    def test_returns_false_when_no_tasks(self, tmp_path: Path) -> None:
        assert not has_pending_tasks_for_comment(tmp_path, 42)

    def test_returns_false_when_task_completed(self, tmp_path: Path) -> None:
        task = add_task(
            tmp_path,
            title="t",
            task_type=TaskType.THREAD,
            thread={"repo": "r/r", "pr": 1, "comment_id": 42},
        )
        complete_by_id(tmp_path, task["id"])
        assert not has_pending_tasks_for_comment(tmp_path, 42)

    def test_returns_false_for_different_comment_id(self, tmp_path: Path) -> None:
        add_task(
            tmp_path,
            title="t",
            task_type=TaskType.THREAD,
            thread={"repo": "r/r", "pr": 1, "comment_id": 99},
        )
        assert not has_pending_tasks_for_comment(tmp_path, 42)

    def test_accepts_string_comment_id(self, tmp_path: Path) -> None:
        add_task(
            tmp_path,
            title="t",
            task_type=TaskType.THREAD,
            thread={"repo": "r/r", "pr": 1, "comment_id": 42},
        )
        assert has_pending_tasks_for_comment(tmp_path, "42")

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
        assert has_pending_tasks_for_comment(tmp_path, 42)


class TestRemoveTask:
    def test_removes(self, tmp_path: Path) -> None:
        task = add_task(tmp_path, title="t", task_type=TaskType.SPEC)
        assert remove_task(tmp_path, task["id"])
        assert list_tasks(tmp_path) == []

    def test_returns_false_if_not_found(self, tmp_path: Path) -> None:
        add_task(tmp_path, title="t", task_type=TaskType.SPEC)
        assert not remove_task(tmp_path, "nonexistent")


class TestUnblockTasks:
    def test_unblocks_blocked_tasks(self, tmp_path: Path) -> None:
        t1 = add_task(tmp_path, title="first", task_type=TaskType.SPEC)
        t2 = add_task(tmp_path, title="second", task_type=TaskType.SPEC)
        update_task(tmp_path, t1["id"], TaskStatus.BLOCKED)
        update_task(tmp_path, t2["id"], TaskStatus.BLOCKED)
        count = unblock_tasks(tmp_path)
        assert count == 2
        tasks = list_tasks(tmp_path)
        assert all(t["status"] == str(TaskStatus.PENDING) for t in tasks)

    def test_returns_zero_when_nothing_blocked(self, tmp_path: Path) -> None:
        add_task(tmp_path, title="t", task_type=TaskType.SPEC)
        assert unblock_tasks(tmp_path) == 0

    def test_does_not_touch_non_blocked_tasks(self, tmp_path: Path) -> None:
        t1 = add_task(tmp_path, title="pending", task_type=TaskType.SPEC)
        t2 = add_task(tmp_path, title="done", task_type=TaskType.SPEC)
        t3 = add_task(tmp_path, title="blocked", task_type=TaskType.SPEC)
        update_task(tmp_path, t2["id"], TaskStatus.COMPLETED)
        update_task(tmp_path, t3["id"], TaskStatus.BLOCKED)
        count = unblock_tasks(tmp_path)
        assert count == 1
        tasks = {t["id"]: t for t in list_tasks(tmp_path)}
        assert tasks[t1["id"]]["status"] == str(TaskStatus.PENDING)
        assert tasks[t2["id"]]["status"] == str(TaskStatus.COMPLETED)
        assert tasks[t3["id"]]["status"] == str(TaskStatus.PENDING)

    def test_returns_zero_on_empty_file(self, tmp_path: Path) -> None:
        assert unblock_tasks(tmp_path) == 0

    def test_tasks_unblock_tasks_method(self, tmp_path: Path) -> None:
        task = add_task(tmp_path, title="t", task_type=TaskType.SPEC)
        update_task(tmp_path, task["id"], TaskStatus.BLOCKED)
        count = Tasks(tmp_path).unblock_tasks()
        assert count == 1
        assert list_tasks(tmp_path)[0]["status"] == str(TaskStatus.PENDING)


# ── _parse_reorder_response ───────────────────────────────────────────────────


class TestParseReorderResponse:
    def test_parses_valid_json(self) -> None:
        raw = '{"tasks": [{"id": "1", "title": "Task A", "description": ""}]}'
        result = _parse_reorder_response(raw)
        assert result == [{"id": "1", "title": "Task A", "description": ""}]

    def test_parses_json_with_preamble(self) -> None:
        raw = 'Here are the reordered tasks:\n\n{"tasks": [{"id": "1", "title": "Task A", "description": ""}]}'
        result = _parse_reorder_response(raw)
        assert result is not None
        assert result[0]["id"] == "1"

    def test_returns_none_for_invalid_json(self) -> None:
        assert _parse_reorder_response("not json at all") is None

    def test_returns_none_when_no_tasks_key(self) -> None:
        assert _parse_reorder_response('{"other": []}') is None

    def test_returns_none_when_tasks_not_list(self) -> None:
        assert _parse_reorder_response('{"tasks": "not a list"}') is None

    def test_returns_none_for_json_non_dict(self) -> None:
        # json.loads("null") → None → None.get("tasks") → AttributeError
        assert _parse_reorder_response("null") is None

    def test_returns_empty_list_when_tasks_is_empty(self) -> None:
        result = _parse_reorder_response('{"tasks": []}')
        assert result == []

    def test_parses_multiple_tasks(self) -> None:
        raw = '{"tasks": [{"id": "1", "title": "A", "description": ""}, {"id": "2", "title": "B", "description": ""}]}'
        result = _parse_reorder_response(raw)
        assert result is not None
        assert len(result) == 2


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

    def test_preserves_title_from_opus(self) -> None:
        current = [self._t("1", "Old title")]
        items = [self._item("1", "New title")]
        result = _apply_reorder(current, items)
        assert result[0]["title"] == "Old title"

    def test_preserves_title_when_opus_returns_empty(self) -> None:
        current = [self._t("1", "Original title")]
        items = [self._item("1", "")]  # empty title → don't overwrite
        result = _apply_reorder(current, items)
        assert result[0]["title"] == "Original title"

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
        current = [self._t("1", "Task")]
        items = [self._item("1", "Task v1"), self._item("1", "Task v2")]
        result = _apply_reorder(current, items)
        assert len([t for t in result if t["id"] == "1"]) == 1
        assert result[0]["title"] == "Task"

    def test_ci_tasks_always_first(self) -> None:
        current = [
            self._t("1", "Spec task"),
            self._t("2", "CI failure", task_type="ci"),
        ]
        items = [self._item("1", "Spec task"), self._item("2", "CI failure")]
        result = _apply_reorder(current, items)
        assert result[0]["id"] == "2"
        assert result[1]["id"] == "1"

    def test_in_progress_task_marked_completed_when_opus_excludes_it(self) -> None:
        current = [
            self._t("1", "Active task", status="in_progress"),
            self._t("2", "Spec task"),
        ]
        original_ids = frozenset({"1", "2"})
        # Opus omitted task "1" (in_progress) — marked completed; caller aborts worker
        items = [self._item("2", "Spec task")]
        result = _apply_reorder(current, items, original_ids)
        task1 = next(t for t in result if t["id"] == "1")
        assert task1["status"] == "completed"

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

    def test_marks_pending_task_completed_when_opus_excludes_it(self) -> None:
        current = [self._t("1", "Keep"), self._t("2", "No longer needed")]
        original_ids = frozenset({"1", "2"})
        items = [self._item("1", "Keep")]  # Opus omitted "2"
        result = _apply_reorder(current, items, original_ids)
        task2 = next(t for t in result if t["id"] == "2")
        assert task2["status"] == "completed"

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

    def test_preserves_titles_when_opus_proposes_duplicates(self) -> None:
        current = [self._t("1", "Alpha task"), self._t("2", "Beta task")]
        items = [self._item("1", "Shared name"), self._item("2", "Shared name")]
        result = _apply_reorder(current, items)
        assert next(t for t in result if t["id"] == "1")["title"] == "Alpha task"
        assert next(t for t in result if t["id"] == "2")["title"] == "Beta task"

    def test_unique_title_rewrites_are_ignored(self) -> None:
        current = [self._t("1", "Old A"), self._t("2", "Old B"), self._t("3", "Old C")]
        items = [
            self._item("1", "New A"),
            self._item("2", "New B"),
            self._item("3", "New C"),
        ]
        result = _apply_reorder(current, items)
        titles = {t["id"]: t["title"] for t in result}
        assert titles == {"1": "Old A", "2": "Old B", "3": "Old C"}

    def test_duplicate_title_rewrites_do_not_log_warning(self, caplog) -> None:
        current = [self._t("1", "Alpha task"), self._t("2", "Beta task")]
        items = [self._item("1", "Shared name"), self._item("2", "Shared name")]
        _apply_reorder(current, items)
        assert "rejecting rewrite" not in caplog.text

    def test_fails_closed_when_runtime_result_diverges_from_oracle(self) -> None:
        current = [self._t("1", "Original")]
        bad_result = [self._t("1", "Changed")]

        with pytest.raises(AssertionError, match="diverged"):
            _assert_rescope_matches_oracle(
                current,
                [self._item("1", "Changed")],
                {"1"},
                bad_result,
            )


# ── _find_duplicate_titles ────────────────────────────────────────────────────


class TestFindDuplicateTitles:
    def _item(self, title: str) -> dict:
        return {"id": "x", "title": title}

    def test_returns_empty_when_all_unique(self) -> None:
        items = [self._item("A"), self._item("B"), self._item("C")]
        assert _find_duplicate_titles(items) == []

    def test_returns_duplicate_title(self) -> None:
        items = [self._item("Same"), self._item("Other"), self._item("Same")]
        assert _find_duplicate_titles(items) == ["Same"]

    def test_each_duplicate_listed_once(self) -> None:
        items = [self._item("X"), self._item("X"), self._item("X")]
        assert _find_duplicate_titles(items) == ["X"]

    def test_multiple_distinct_duplicates(self) -> None:
        items = [
            self._item("A"),
            self._item("B"),
            self._item("A"),
            self._item("B"),
        ]
        assert _find_duplicate_titles(items) == ["A", "B"]

    def test_ignores_empty_titles(self) -> None:
        items = [{"id": "1", "title": ""}, {"id": "2", "title": ""}]
        assert _find_duplicate_titles(items) == []

    def test_ignores_missing_title_key(self) -> None:
        items = [{"id": "1"}, {"id": "2"}]
        assert _find_duplicate_titles(items) == []

    def test_returns_empty_list_when_no_items(self) -> None:
        assert _find_duplicate_titles([]) == []


# ── reorder_tasks ─────────────────────────────────────────────────────────────


class TestReorderTasks:
    def _add(self, tmp_path: Path, title: str, task_type=TaskType.SPEC) -> dict:
        return add_task(tmp_path, title=title, task_type=task_type)

    def _response(self, items: list[dict]) -> str:
        return json.dumps({"tasks": items})

    def test_skips_when_no_tasks(self, tmp_path: Path) -> None:
        client = _client("")
        reorder_tasks(tmp_path, "", agent=client)
        client.run_turn.assert_not_called()

    def test_creates_default_client_when_none(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        self._add(tmp_path, "Task A")
        with patch(
            "fido.tasks.ClaudeClient",
            return_value=_client(""),
        ) as mock_cls:
            reorder_tasks(tmp_path, "")
            mock_cls.assert_called_once_with()

    def test_skips_on_empty_opus_response(self, tmp_path: Path) -> None:
        self._add(tmp_path, "Task A")
        result_before = list_tasks(tmp_path)
        reorder_tasks(tmp_path, "", agent=_client(""))
        assert list_tasks(tmp_path) == result_before

    def test_skips_on_unparseable_response(self, tmp_path: Path) -> None:
        self._add(tmp_path, "Task A")
        result_before = list_tasks(tmp_path)
        reorder_tasks(tmp_path, "", agent=_client("not json"))
        assert list_tasks(tmp_path) == result_before

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
        reorder_tasks(tmp_path, "", agent=_client(raw))
        result = list_tasks(tmp_path)
        assert result[0]["id"] == t1["id"]
        assert result[1]["id"] == t2["id"]

    def test_preserves_title_from_opus(self, tmp_path: Path) -> None:
        t1 = self._add(tmp_path, "Old title")
        raw = self._response(
            [{"id": t1["id"], "title": "New title", "description": ""}]
        )
        reorder_tasks(tmp_path, "", agent=_client(raw))
        assert list_tasks(tmp_path)[0]["title"] == "Old title"

    def test_marks_completed_task_opus_excludes(self, tmp_path: Path) -> None:
        t1 = self._add(tmp_path, "Keep")
        t2 = self._add(tmp_path, "No longer needed")
        raw = self._response([{"id": t1["id"], "title": "Keep", "description": ""}])
        reorder_tasks(tmp_path, "", agent=_client(raw))
        result = list_tasks(tmp_path)
        task2 = next(t for t in result if t["id"] == t2["id"])
        assert task2["status"] == "completed"

    def test_preserves_completed_tasks(self, tmp_path: Path) -> None:
        t1 = self._add(tmp_path, "Done")
        complete_by_id(tmp_path, t1["id"])
        t2 = self._add(tmp_path, "Pending")
        raw = self._response([{"id": t2["id"], "title": "Pending", "description": ""}])
        reorder_tasks(tmp_path, "", agent=_client(raw))
        result = list_tasks(tmp_path)
        statuses = {t["id"]: t["status"] for t in result}
        assert statuses[t1["id"]] == "completed"

    def test_rescope_prompt_fn_receives_task_list_and_commit_summary(
        self, tmp_path: Path
    ) -> None:
        self._add(tmp_path, "Task A")
        mock_prompts = MagicMock(spec=Prompts)
        mock_prompts.rescope_prompt.return_value = "prompt text"
        reorder_tasks(
            tmp_path,
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

        def slow_run_turn(prompt, *, model=None, **kw):
            # Simulate a new task arriving while Opus is running
            t2 = add_task(
                tmp_path, title="Arrived mid-reorder", task_type=TaskType.SPEC
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
        reorder_tasks(tmp_path, "", agent=client)
        result = list_tasks(tmp_path)
        ids = [t["id"] for t in result]
        assert new_task_id[0] in ids  # not silently dropped
        assert ids == [t1["id"], new_task_id[0]]
        new_task = next(t for t in result if t["id"] == new_task_id[0])
        assert new_task["title"] == "Arrived mid-reorder"
        assert new_task["description"] == ""

    def test_on_changes_called_when_thread_task_completed_by_reorder(
        self, tmp_path: Path
    ) -> None:
        thread = {
            "repo": "r/r",
            "pr": 1,
            "comment_id": 99,
            "url": "https://example.com",
        }
        t1 = add_task(
            tmp_path, title="Thread task", task_type=TaskType.THREAD, thread=thread
        )
        t2 = self._add(tmp_path, "Keep this")
        received: list = []
        raw = self._response(
            [{"id": t2["id"], "title": "Keep this", "description": ""}]
        )
        reorder_tasks(
            tmp_path,
            "",
            agent=_client(raw),
            _on_changes=lambda changes: received.extend(changes),
        )
        assert len(received) == 1
        assert received[0]["kind"] == "completed"
        assert received[0]["task"]["id"] == t1["id"]

    def test_on_changes_called_when_thread_task_modified(self, tmp_path: Path) -> None:
        thread = {
            "repo": "r/r",
            "pr": 1,
            "comment_id": 99,
            "url": "https://example.com",
        }
        t1 = add_task(
            tmp_path,
            title="Stable title",
            task_type=TaskType.THREAD,
            thread=thread,
        )
        received: list = []
        raw = self._response(
            [{"id": t1["id"], "title": "Changed title", "description": "new"}]
        )
        reorder_tasks(
            tmp_path,
            "",
            agent=_client(raw),
            _on_changes=lambda changes: received.extend(changes),
        )
        assert len(received) == 1
        assert received[0]["kind"] == "modified"
        assert received[0]["new_title"] == "Stable title"
        assert received[0]["new_description"] == "new"

    def test_on_changes_not_called_when_no_thread_tasks_changed(
        self, tmp_path: Path
    ) -> None:
        t1 = self._add(tmp_path, "Spec task")
        received: list = []
        raw = self._response(
            [{"id": t1["id"], "title": "Spec task", "description": ""}]
        )
        reorder_tasks(
            tmp_path,
            "",
            agent=_client(raw),
            _on_changes=lambda changes: received.extend(changes),
        )
        assert received == []

    def test_on_changes_none_does_not_error(self, tmp_path: Path) -> None:
        thread = {"repo": "r/r", "pr": 1, "comment_id": 42, "url": "https://x.com"}
        t1 = add_task(
            tmp_path, title="Thread task", task_type=TaskType.THREAD, thread=thread
        )
        t2 = self._add(tmp_path, "Keep")
        raw = self._response([{"id": t2["id"], "title": "Keep", "description": ""}])
        # Should not raise even though t1 is completed and _on_changes is None
        reorder_tasks(tmp_path, "", agent=_client(raw), _on_changes=None)
        # t1 should be marked completed
        task1 = next(t for t in list_tasks(tmp_path) if t["id"] == t1["id"])
        assert task1["status"] == "completed"

    def test_on_inprogress_affected_called_when_inprogress_task_completed(
        self, tmp_path: Path
    ) -> None:
        t1 = self._add(tmp_path, "In-progress task")
        update_task(tmp_path, t1["id"], TaskStatus.IN_PROGRESS)
        t2 = self._add(tmp_path, "Keep this")
        raw = self._response(
            [{"id": t2["id"], "title": "Keep this", "description": ""}]
        )
        affected: list[int] = []
        reorder_tasks(
            tmp_path,
            "",
            agent=_client(raw),
            _on_inprogress_affected=lambda: affected.append(1),
        )
        assert affected == [1]
        # in-progress task is marked completed
        task1 = next(t for t in list_tasks(tmp_path) if t["id"] == t1["id"])
        assert task1["status"] == "completed"

    def test_on_inprogress_affected_called_when_inprogress_task_modified(
        self, tmp_path: Path
    ) -> None:
        t1 = self._add(tmp_path, "Stable title")
        update_task(tmp_path, t1["id"], TaskStatus.IN_PROGRESS)
        raw = self._response(
            [{"id": t1["id"], "title": "Changed title", "description": "new"}]
        )
        affected: list[int] = []
        reorder_tasks(
            tmp_path,
            "",
            agent=_client(raw),
            _on_inprogress_affected=lambda: affected.append(1),
        )
        assert affected == [1]
        # task reset to pending with preserved title and updated description
        result = list_tasks(tmp_path)
        assert result[0]["title"] == "Stable title"
        assert result[0]["description"] == "new"
        assert result[0]["status"] == str(TaskStatus.PENDING)

    def test_on_inprogress_affected_not_called_when_inprogress_task_unchanged(
        self, tmp_path: Path
    ) -> None:
        t1 = self._add(tmp_path, "Stable task")
        update_task(tmp_path, t1["id"], TaskStatus.IN_PROGRESS)
        raw = self._response(
            [{"id": t1["id"], "title": "Stable task", "description": ""}]
        )
        affected: list[int] = []
        reorder_tasks(
            tmp_path,
            "",
            agent=_client(raw),
            _on_inprogress_affected=lambda: affected.append(1),
        )
        assert affected == []
        # task still in_progress (unchanged by Opus)
        result = list_tasks(tmp_path)
        assert result[0]["status"] == str(TaskStatus.IN_PROGRESS)

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
            tmp_path,
            "",
            agent=_client(raw),
            _on_inprogress_affected=lambda: affected.append(1),
        )
        assert affected == []

    def test_on_inprogress_affected_none_does_not_error(self, tmp_path: Path) -> None:
        t1 = self._add(tmp_path, "In-progress task")
        update_task(tmp_path, t1["id"], TaskStatus.IN_PROGRESS)
        t2 = self._add(tmp_path, "Keep this")
        raw = self._response(
            [{"id": t2["id"], "title": "Keep this", "description": ""}]
        )
        # Should not raise even though in-progress task is completed and callback is None
        reorder_tasks(
            tmp_path,
            "",
            agent=_client(raw),
            _on_inprogress_affected=None,
        )
        task1 = next(t for t in list_tasks(tmp_path) if t["id"] == t1["id"])
        assert task1["status"] == "completed"

    def test_on_done_called_after_successful_reorder(self, tmp_path: Path) -> None:
        t1 = self._add(tmp_path, "Task A")
        raw = self._response([{"id": t1["id"], "title": "Task A", "description": ""}])
        done_calls: list = []
        reorder_tasks(
            tmp_path,
            "",
            agent=_client(raw),
            _on_done=lambda: done_calls.append(1),
        )
        assert done_calls == [1]

    def test_on_done_not_called_when_no_tasks(self, tmp_path: Path) -> None:
        done_calls: list = []
        reorder_tasks(
            tmp_path,
            "",
            agent=_client("{}"),
            _on_done=lambda: done_calls.append(1),
        )
        assert done_calls == []

    def test_on_done_not_called_on_empty_opus_response(self, tmp_path: Path) -> None:
        self._add(tmp_path, "Task A")
        done_calls: list = []
        reorder_tasks(
            tmp_path,
            "",
            agent=_client(""),
            _on_done=lambda: done_calls.append(1),
        )
        assert done_calls == []

    def test_on_done_not_called_on_unparseable_response(self, tmp_path: Path) -> None:
        self._add(tmp_path, "Task A")
        done_calls: list = []
        reorder_tasks(
            tmp_path,
            "",
            agent=_client("not json at all"),
            _on_done=lambda: done_calls.append(1),
        )
        assert done_calls == []

    def test_nudges_when_opus_proposes_duplicate_titles(self, tmp_path: Path) -> None:
        t1 = self._add(tmp_path, "Alpha")
        t2 = self._add(tmp_path, "Beta")
        # First response has duplicate titles
        dup_response = self._response(
            [
                {"id": t1["id"], "title": "Shared name", "description": ""},
                {"id": t2["id"], "title": "Shared name", "description": ""},
            ]
        )
        # Nudge response has unique titles
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
        reorder_tasks(tmp_path, "", agent=client, prompts=mock_prompts)
        mock_prompts.rescope_duplicate_nudge.assert_called_once_with(
            ["Shared name"], attempts_remaining=2
        )
        assert client.run_turn.call_count == 2
        tasks = list_tasks(tmp_path)
        assert next(t for t in tasks if t["id"] == t1["id"])["title"] == "Alpha"
        assert next(t for t in tasks if t["id"] == t2["id"])["title"] == "Beta"

    def test_falls_back_silently_when_nudge_still_has_duplicates(
        self, tmp_path: Path
    ) -> None:
        t1 = self._add(tmp_path, "Alpha")
        t2 = self._add(tmp_path, "Beta")
        dup_response = self._response(
            [
                {"id": t1["id"], "title": "Shared name", "description": ""},
                {"id": t2["id"], "title": "Shared name", "description": ""},
            ]
        )
        client = _client()
        # All 3 nudge responses still have duplicates → exhaust all retries
        client.run_turn.side_effect = [dup_response] * 4
        mock_prompts = MagicMock(spec=Prompts)
        mock_prompts.rescope_prompt.return_value = "prompt"
        mock_prompts.rescope_duplicate_nudge.return_value = "nudge"
        reorder_tasks(tmp_path, "", agent=client, prompts=mock_prompts)
        # All 3 nudges fired (1 initial + 3 nudge calls = 4 total)
        assert client.run_turn.call_count == 4
        tasks = list_tasks(tmp_path)
        assert next(t for t in tasks if t["id"] == t1["id"])["title"] == "Alpha"
        assert next(t for t in tasks if t["id"] == t2["id"])["title"] == "Beta"

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
        reorder_tasks(tmp_path, "", agent=client, prompts=mock_prompts)
        # Nudge calls: attempts_remaining 2 → 1 → 0
        assert mock_prompts.rescope_duplicate_nudge.call_count == 3
        calls = mock_prompts.rescope_duplicate_nudge.call_args_list
        assert calls[0] == call(["Shared name"], attempts_remaining=2)
        assert calls[1] == call(["Shared name"], attempts_remaining=1)
        assert calls[2] == call(["Shared name"], attempts_remaining=0)

    def test_proceeds_with_original_when_nudge_returns_empty(
        self, tmp_path: Path
    ) -> None:
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
        reorder_tasks(tmp_path, "", agent=client, prompts=mock_prompts)
        tasks = list_tasks(tmp_path)
        assert next(t for t in tasks if t["id"] == t1["id"])["title"] == "Alpha"
        assert next(t for t in tasks if t["id"] == t2["id"])["title"] == "Beta"

    def test_proceeds_with_original_when_nudge_response_unparseable(
        self, tmp_path: Path
    ) -> None:
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
        reorder_tasks(tmp_path, "", agent=client, prompts=mock_prompts)
        tasks = list_tasks(tmp_path)
        assert next(t for t in tasks if t["id"] == t1["id"])["title"] == "Alpha"
        assert next(t for t in tasks if t["id"] == t2["id"])["title"] == "Beta"

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
        reorder_tasks(tmp_path, "", agent=client, prompts=mock_prompts)
        mock_prompts.rescope_duplicate_nudge.assert_not_called()
        assert client.run_turn.call_count == 1


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

    def test_modified_description(self) -> None:
        original = [self._t("1", "Task", thread=self._thread(), description="old")]
        result = [self._t("1", "Task", thread=self._thread(), description="new")]
        changes = _compute_thread_changes(original, result, frozenset({"1"}))
        assert len(changes) == 1
        assert changes[0]["kind"] == "modified"
        assert changes[0]["new_description"] == "new"

    def test_unchanged_thread_task_not_reported(self) -> None:
        t = self._t("1", "Task", thread=self._thread())
        changes = _compute_thread_changes([t], [t], frozenset({"1"}))
        assert changes == []

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
    def test_list_returns_all_tasks(self, tmp_path: Path) -> None:
        from fido.types import TaskType

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        add_task(work_dir, "Task A", TaskType.SPEC)
        result = Tasks(work_dir).list()
        assert len(result) == 1
        assert result[0]["title"] == "Task A"

    def test_add_creates_task(self, tmp_path: Path) -> None:
        from fido.types import TaskType

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        task = Tasks(work_dir).add("Task B", TaskType.CI)
        assert task["title"] == "Task B"
        assert task["type"] == "ci"

    def test_complete_by_id_marks_task_completed(self, tmp_path: Path) -> None:
        from fido.types import TaskType

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        task = add_task(work_dir, "Task C", TaskType.SPEC)
        Tasks(work_dir).complete_by_id(task["id"])
        assert list_tasks(work_dir)[0]["status"] == "completed"

    def test_has_pending_for_comment_returns_bool(self, tmp_path: Path) -> None:
        from fido.types import TaskType

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        add_task(work_dir, "Task D", TaskType.THREAD, thread={"comment_id": 7})
        assert Tasks(work_dir).has_pending_for_comment(7) is True
        assert Tasks(work_dir).has_pending_for_comment(99) is False

    def test_remove_deletes_task(self, tmp_path: Path) -> None:
        from fido.types import TaskType

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        task = add_task(work_dir, "Task E", TaskType.SPEC)
        assert Tasks(work_dir).remove(task["id"]) is True
        assert list_tasks(work_dir) == []

    def test_update_changes_status(self, tmp_path: Path) -> None:
        from fido.types import TaskStatus, TaskType

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        task = add_task(work_dir, "Task F", TaskType.SPEC)
        Tasks(work_dir).update(task["id"], TaskStatus.IN_PROGRESS)
        assert list_tasks(work_dir)[0]["status"] == "in_progress"

    def test_modify_yields_empty_list_when_no_tasks(self, tmp_path: Path) -> None:
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        with Tasks(work_dir).modify() as tasks:
            assert tasks == []

    def test_modify_persists_changes(self, tmp_path: Path) -> None:
        from fido.types import TaskType

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        add_task(work_dir, "Existing task", TaskType.SPEC)
        with Tasks(work_dir).modify() as tasks:
            tasks[0]["title"] = "Modified task"
        assert list_tasks(work_dir)[0]["title"] == "Modified task"

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
        task = add_task(work_dir, "task", TaskType.SPEC)
        Tasks(work_dir).complete_with_resolve(task["id"], MagicMock())
        assert list_tasks(work_dir)[0]["status"] == "completed"

    def test_no_thread_does_not_call_github(self, tmp_path: Path) -> None:
        work_dir = self._work_dir(tmp_path)
        task = add_task(work_dir, "task", TaskType.SPEC)
        gh = MagicMock()
        Tasks(work_dir).complete_with_resolve(task["id"], gh)
        gh.get_user.assert_not_called()
        gh.resolve_thread.assert_not_called()

    def test_thread_missing_fields_skips_resolve(self, tmp_path: Path) -> None:
        """Thread dict with missing pr/comment_id skips resolution."""
        work_dir = self._work_dir(tmp_path)
        task = add_task(work_dir, "task", TaskType.THREAD, thread={"repo": "a/b"})
        gh = MagicMock()
        Tasks(work_dir).complete_with_resolve(task["id"], gh)
        gh.resolve_thread.assert_not_called()

    def test_nonexistent_id_does_not_raise(self, tmp_path: Path) -> None:
        work_dir = self._work_dir(tmp_path)
        Tasks(work_dir).complete_with_resolve("nonexistent-id", MagicMock())

    def test_triggers_background_sync(self, tmp_path: Path) -> None:
        """Every completion fires sync_tasks_background so the PR body
        checkbox flips even when the worker loop doesn't sync between
        the completion and the PR-ready/merge step (#988)."""
        work_dir = self._work_dir(tmp_path)
        task = add_task(work_dir, "task", TaskType.SPEC)
        gh = MagicMock()
        with patch("fido.tasks.sync_tasks_background") as mock_sync:
            Tasks(work_dir).complete_with_resolve(task["id"], gh)
        mock_sync.assert_called_once_with(work_dir, gh)

    def test_triggers_background_sync_even_for_unknown_id(self, tmp_path: Path) -> None:
        """Sync still fires even when the task_id isn't found — caller
        intent ('I just tried to complete something') is the trigger."""
        work_dir = self._work_dir(tmp_path)
        gh = MagicMock()
        with patch("fido.tasks.sync_tasks_background") as mock_sync:
            Tasks(work_dir).complete_with_resolve("nonexistent-id", gh)
        mock_sync.assert_called_once_with(work_dir, gh)

    def test_resolves_thread_when_we_are_last(self, tmp_path: Path, caplog) -> None:
        import logging

        work_dir = self._work_dir(tmp_path)
        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        task = add_task(work_dir, "threaded task", TaskType.THREAD, thread=thread)

        gh = MagicMock()
        gh.get_user.return_value = "fido-bot"
        gh.get_pull_comments.return_value = [
            {
                "id": 42,
                "in_reply_to_id": None,
                "user": {"login": "reviewer"},
                "created_at": "2024-01-01T00:00:00Z",
            },
            {
                "id": 99,
                "in_reply_to_id": 42,
                "user": {"login": "fido-bot"},
                "created_at": "2024-01-02T00:00:00Z",
            },
        ]
        gh.get_review_threads.return_value = [
            {
                "id": "thread_node_abc",
                "isResolved": False,
                "comments": {"nodes": [{"databaseId": 42}]},
            }
        ]

        with caplog.at_level(logging.INFO, logger="fido"):
            Tasks(work_dir).complete_with_resolve(task["id"], gh)

        gh.resolve_thread.assert_called_once_with("thread_node_abc")
        assert "thread resolved: thread_node_abc" in caplog.text

    def test_skips_resolve_when_not_last_commenter(
        self, tmp_path: Path, caplog
    ) -> None:
        import logging

        work_dir = self._work_dir(tmp_path)
        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        task = add_task(work_dir, "threaded task", TaskType.THREAD, thread=thread)

        gh = MagicMock()
        gh.get_user.return_value = "fido-bot"
        gh.get_pull_comments.return_value = [
            {
                "id": 42,
                "in_reply_to_id": None,
                "user": {"login": "fido-bot"},
                "created_at": "2024-01-01T00:00:00Z",
            },
            {
                "id": 99,
                "in_reply_to_id": 42,
                "user": {"login": "reviewer"},
                "created_at": "2024-01-02T00:00:00Z",
            },
        ]

        with caplog.at_level(logging.INFO, logger="fido"):
            Tasks(work_dir).complete_with_resolve(task["id"], gh)

        gh.resolve_thread.assert_not_called()
        assert "not resolving" in caplog.text

    def test_skips_resolve_when_no_matching_comments(self, tmp_path: Path) -> None:
        work_dir = self._work_dir(tmp_path)
        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        task = add_task(work_dir, "threaded task", TaskType.THREAD, thread=thread)

        gh = MagicMock()
        gh.get_user.return_value = "fido-bot"
        gh.get_pull_comments.return_value = []

        Tasks(work_dir).complete_with_resolve(task["id"], gh)

        gh.resolve_thread.assert_not_called()

    def test_skips_resolve_when_thread_already_resolved(self, tmp_path: Path) -> None:
        work_dir = self._work_dir(tmp_path)
        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        task = add_task(work_dir, "threaded task", TaskType.THREAD, thread=thread)

        gh = MagicMock()
        gh.get_user.return_value = "fido-bot"
        gh.get_pull_comments.return_value = [
            {
                "id": 42,
                "in_reply_to_id": None,
                "user": {"login": "fido-bot"},
                "created_at": "2024-01-01T00:00:00Z",
            },
        ]
        gh.get_review_threads.return_value = [
            {
                "id": "thread_node_abc",
                "isResolved": True,
                "comments": {"nodes": [{"databaseId": 42}]},
            }
        ]

        Tasks(work_dir).complete_with_resolve(task["id"], gh)

        gh.resolve_thread.assert_not_called()

    def test_exception_silenced_and_logged(self, tmp_path: Path, caplog) -> None:
        import logging

        work_dir = self._work_dir(tmp_path)
        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        task = add_task(work_dir, "threaded task", TaskType.THREAD, thread=thread)

        gh = MagicMock()
        gh.get_user.side_effect = RuntimeError("network error")

        with caplog.at_level(logging.WARNING, logger="fido"):
            Tasks(work_dir).complete_with_resolve(task["id"], gh)
        assert "thread resolution skipped" in caplog.text
