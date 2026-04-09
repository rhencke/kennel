from __future__ import annotations

from pathlib import Path

import pytest

from kennel.tasks import (
    add_task,
    complete_by_id,
    has_pending_tasks_for_comment,
    list_tasks,
    remove_task,
    update_task,
)
from kennel.types import TaskStatus, TaskType


def _task_file(tmp_path: Path) -> Path:
    git_dir = tmp_path / ".git" / "fido"
    git_dir.mkdir(parents=True)
    return git_dir / "tasks.json"


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
        assert list_tasks(tmp_path) == []

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
