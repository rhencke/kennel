from __future__ import annotations

from pathlib import Path

from kennel.tasks import (
    add_task,
    complete_by_title,
    list_tasks,
    remove_task,
    update_task,
)


def _task_file(tmp_path: Path) -> Path:
    git_dir = tmp_path / ".git" / "fido"
    git_dir.mkdir(parents=True)
    return git_dir / "tasks.json"


class TestAddTask:
    def test_creates_file(self, tmp_path: Path) -> None:
        task = add_task(tmp_path, title="test task")
        assert task["title"] == "test task"
        assert task["status"] == "pending"
        tasks = list_tasks(tmp_path)
        assert len(tasks) == 1

    def test_appends(self, tmp_path: Path) -> None:
        add_task(tmp_path, title="one")
        add_task(tmp_path, title="two")
        tasks = list_tasks(tmp_path)
        assert len(tasks) == 2

    def test_with_thread(self, tmp_path: Path) -> None:
        thread = {"repo": "a/b", "pr": 1, "comment_id": 123}
        task = add_task(tmp_path, title="t", thread=thread)
        assert task["thread"] == thread

    def test_id_is_unique(self, tmp_path: Path) -> None:
        t1 = add_task(tmp_path, title="a")
        t2 = add_task(tmp_path, title="b")
        assert t1["id"] != t2["id"]

    def test_priority_inserts_at_front(self, tmp_path: Path) -> None:
        add_task(tmp_path, title="existing")
        add_task(tmp_path, title="urgent", priority=True)
        tasks = list_tasks(tmp_path)
        assert tasks[0]["title"] == "urgent"
        assert tasks[1]["title"] == "existing"

    def test_default_appends_at_end(self, tmp_path: Path) -> None:
        add_task(tmp_path, title="first")
        add_task(tmp_path, title="second")
        tasks = list_tasks(tmp_path)
        assert tasks[0]["title"] == "first"
        assert tasks[1]["title"] == "second"


class TestUpdateTask:
    def test_updates_status(self, tmp_path: Path) -> None:
        task = add_task(tmp_path, title="t")
        assert update_task(tmp_path, task["id"], "completed")
        tasks = list_tasks(tmp_path)
        assert tasks[0]["status"] == "completed"

    def test_returns_false_if_not_found(self, tmp_path: Path) -> None:
        add_task(tmp_path, title="t")
        assert not update_task(tmp_path, "nonexistent", "completed")


class TestListTasks:
    def test_empty(self, tmp_path: Path) -> None:
        _task_file(tmp_path).write_text("[]")
        assert list_tasks(tmp_path) == []

    def test_corrupt_json(self, tmp_path: Path) -> None:
        _task_file(tmp_path).write_text("not json")
        assert list_tasks(tmp_path) == []


class TestCompleteByTitle:
    def test_marks_completed_returns_none_when_no_thread(self, tmp_path: Path) -> None:
        add_task(tmp_path, title="do something")
        result = complete_by_title(tmp_path, "do something")
        assert result is None
        assert list_tasks(tmp_path)[0]["status"] == "completed"

    def test_returns_thread_when_present(self, tmp_path: Path) -> None:
        thread = {"repo": "a/b", "pr": 1, "comment_id": 99}
        add_task(tmp_path, title="t", thread=thread)
        result = complete_by_title(tmp_path, "t")
        assert result == thread

    def test_returns_none_when_not_found(self, tmp_path: Path) -> None:
        add_task(tmp_path, title="other")
        assert complete_by_title(tmp_path, "missing") is None
        assert list_tasks(tmp_path)[0]["status"] == "pending"

    def test_skips_already_completed(self, tmp_path: Path) -> None:
        add_task(tmp_path, title="t")
        complete_by_title(tmp_path, "t")
        # second call on already-completed task returns None
        assert complete_by_title(tmp_path, "t") is None

    def test_completes_first_pending_when_duplicate_titles(
        self, tmp_path: Path
    ) -> None:
        add_task(tmp_path, title="t")
        add_task(tmp_path, title="t")
        complete_by_title(tmp_path, "t")
        tasks = list_tasks(tmp_path)
        assert tasks[0]["status"] == "completed"
        assert tasks[1]["status"] == "pending"


class TestRemoveTask:
    def test_removes(self, tmp_path: Path) -> None:
        task = add_task(tmp_path, title="t")
        assert remove_task(tmp_path, task["id"])
        assert list_tasks(tmp_path) == []

    def test_returns_false_if_not_found(self, tmp_path: Path) -> None:
        add_task(tmp_path, title="t")
        assert not remove_task(tmp_path, "nonexistent")
