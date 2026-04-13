from __future__ import annotations

import json
from pathlib import Path

import pytest

from kennel.tasks import (
    Tasks,
    _apply_reorder,
    _compute_thread_changes,
    _parse_reorder_response,
    add_task,
    complete_by_id,
    has_pending_tasks_for_comment,
    list_tasks,
    remove_task,
    reorder_tasks,
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
        assert [t["id"] for t in result] == ["2", "1"]

    def test_updates_title_from_opus(self) -> None:
        current = [self._t("1", "Old title")]
        items = [self._item("1", "New title")]
        result = _apply_reorder(current, items)
        assert result[0]["title"] == "New title"

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


# ── reorder_tasks ─────────────────────────────────────────────────────────────


class TestReorderTasks:
    def _add(self, tmp_path: Path, title: str, task_type=TaskType.SPEC) -> dict:
        return add_task(tmp_path, title=title, task_type=task_type)

    def _response(self, items: list[dict]) -> str:
        return json.dumps({"tasks": items})

    def test_skips_when_no_tasks(self, tmp_path: Path) -> None:
        called = []
        reorder_tasks(
            tmp_path, "", _print_prompt=lambda *a, **k: called.append(1) or ""
        )
        assert called == []

    def test_skips_on_empty_opus_response(self, tmp_path: Path) -> None:
        self._add(tmp_path, "Task A")
        result_before = list_tasks(tmp_path)
        reorder_tasks(tmp_path, "", _print_prompt=lambda *a, **k: "")
        assert list_tasks(tmp_path) == result_before

    def test_skips_on_unparseable_response(self, tmp_path: Path) -> None:
        self._add(tmp_path, "Task A")
        result_before = list_tasks(tmp_path)
        reorder_tasks(tmp_path, "", _print_prompt=lambda *a, **k: "not json")
        assert list_tasks(tmp_path) == result_before

    def test_reorders_tasks(self, tmp_path: Path) -> None:
        t1 = self._add(tmp_path, "First")
        t2 = self._add(tmp_path, "Second")
        # Opus returns them reversed
        raw = self._response(
            [
                {"id": t2["id"], "title": "Second", "description": ""},
                {"id": t1["id"], "title": "First", "description": ""},
            ]
        )
        reorder_tasks(tmp_path, "", _print_prompt=lambda *a, **k: raw)
        result = list_tasks(tmp_path)
        assert result[0]["id"] == t2["id"]
        assert result[1]["id"] == t1["id"]

    def test_updates_title_from_opus(self, tmp_path: Path) -> None:
        t1 = self._add(tmp_path, "Old title")
        raw = self._response(
            [{"id": t1["id"], "title": "New title", "description": ""}]
        )
        reorder_tasks(tmp_path, "", _print_prompt=lambda *a, **k: raw)
        assert list_tasks(tmp_path)[0]["title"] == "New title"

    def test_marks_completed_task_opus_excludes(self, tmp_path: Path) -> None:
        t1 = self._add(tmp_path, "Keep")
        t2 = self._add(tmp_path, "No longer needed")
        raw = self._response([{"id": t1["id"], "title": "Keep", "description": ""}])
        reorder_tasks(tmp_path, "", _print_prompt=lambda *a, **k: raw)
        result = list_tasks(tmp_path)
        task2 = next(t for t in result if t["id"] == t2["id"])
        assert task2["status"] == "completed"

    def test_preserves_completed_tasks(self, tmp_path: Path) -> None:
        t1 = self._add(tmp_path, "Done")
        complete_by_id(tmp_path, t1["id"])
        t2 = self._add(tmp_path, "Pending")
        raw = self._response([{"id": t2["id"], "title": "Pending", "description": ""}])
        reorder_tasks(tmp_path, "", _print_prompt=lambda *a, **k: raw)
        result = list_tasks(tmp_path)
        statuses = {t["id"]: t["status"] for t in result}
        assert statuses[t1["id"]] == "completed"

    def test_rescope_prompt_fn_receives_task_list_and_commit_summary(
        self, tmp_path: Path
    ) -> None:
        self._add(tmp_path, "Task A")
        captured = {}

        def fake_rescope(task_list, commit_summary):
            captured["task_list"] = task_list
            captured["commit_summary"] = commit_summary
            return "prompt text"

        reorder_tasks(
            tmp_path,
            "feat: added thing",
            _print_prompt=lambda *a, **k: "",
            _rescope_prompt_fn=fake_rescope,
        )
        assert captured["commit_summary"] == "feat: added thing"
        assert len(captured["task_list"]) == 1

    def test_picks_up_task_added_while_opus_was_thinking(self, tmp_path: Path) -> None:
        t1 = self._add(tmp_path, "Original task")
        new_task_id: list[str] = []

        def slow_print_prompt(prompt, model, **kw):
            # Simulate a new task arriving while Opus is running
            t2 = add_task(
                tmp_path, title="Arrived mid-reorder", task_type=TaskType.SPEC
            )
            new_task_id.append(t2["id"])
            return json.dumps(
                {
                    "tasks": [
                        {"id": t1["id"], "title": "Original task", "description": ""}
                    ]
                }
            )

        reorder_tasks(tmp_path, "", _print_prompt=slow_print_prompt)
        result = list_tasks(tmp_path)
        ids = [t["id"] for t in result]
        assert new_task_id[0] in ids  # not silently dropped

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
            _print_prompt=lambda *a, **k: raw,
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
            tmp_path, title="Old title", task_type=TaskType.THREAD, thread=thread
        )
        received: list = []
        raw = self._response(
            [{"id": t1["id"], "title": "New title", "description": ""}]
        )
        reorder_tasks(
            tmp_path,
            "",
            _print_prompt=lambda *a, **k: raw,
            _on_changes=lambda changes: received.extend(changes),
        )
        assert len(received) == 1
        assert received[0]["kind"] == "modified"
        assert received[0]["new_title"] == "New title"

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
            _print_prompt=lambda *a, **k: raw,
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
        reorder_tasks(tmp_path, "", _print_prompt=lambda *a, **k: raw, _on_changes=None)
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
            _print_prompt=lambda *a, **k: raw,
            _on_inprogress_affected=lambda: affected.append(1),
        )
        assert affected == [1]
        # in-progress task is marked completed
        task1 = next(t for t in list_tasks(tmp_path) if t["id"] == t1["id"])
        assert task1["status"] == "completed"

    def test_on_inprogress_affected_called_when_inprogress_task_modified(
        self, tmp_path: Path
    ) -> None:
        t1 = self._add(tmp_path, "Old title")
        update_task(tmp_path, t1["id"], TaskStatus.IN_PROGRESS)
        raw = self._response(
            [{"id": t1["id"], "title": "New title", "description": ""}]
        )
        affected: list[int] = []
        reorder_tasks(
            tmp_path,
            "",
            _print_prompt=lambda *a, **k: raw,
            _on_inprogress_affected=lambda: affected.append(1),
        )
        assert affected == [1]
        # task reset to pending with new title
        result = list_tasks(tmp_path)
        assert result[0]["title"] == "New title"
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
            _print_prompt=lambda *a, **k: raw,
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
            _print_prompt=lambda *a, **k: raw,
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
            _print_prompt=lambda *a, **k: raw,
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
            _print_prompt=lambda *a, **k: raw,
            _on_done=lambda: done_calls.append(1),
        )
        assert done_calls == [1]

    def test_on_done_not_called_when_no_tasks(self, tmp_path: Path) -> None:
        done_calls: list = []
        reorder_tasks(
            tmp_path,
            "",
            _print_prompt=lambda *a, **k: "{}",
            _on_done=lambda: done_calls.append(1),
        )
        assert done_calls == []

    def test_on_done_not_called_on_empty_opus_response(self, tmp_path: Path) -> None:
        self._add(tmp_path, "Task A")
        done_calls: list = []
        reorder_tasks(
            tmp_path,
            "",
            _print_prompt=lambda *a, **k: "",
            _on_done=lambda: done_calls.append(1),
        )
        assert done_calls == []

    def test_on_done_not_called_on_unparseable_response(self, tmp_path: Path) -> None:
        self._add(tmp_path, "Task A")
        done_calls: list = []
        reorder_tasks(
            tmp_path,
            "",
            _print_prompt=lambda *a, **k: "not json at all",
            _on_done=lambda: done_calls.append(1),
        )
        assert done_calls == []


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
    def test_list_delegates_to_list_tasks(self, tmp_path: Path) -> None:
        from kennel.types import TaskType

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        add_task(work_dir, "Task A", TaskType.SPEC)
        result = Tasks(work_dir).list()
        assert len(result) == 1
        assert result[0]["title"] == "Task A"

    def test_add_delegates_to_add_task(self, tmp_path: Path) -> None:
        from kennel.types import TaskType

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        task = Tasks(work_dir).add("Task B", TaskType.CI)
        assert task["title"] == "Task B"
        assert task["type"] == "ci"

    def test_complete_by_id_delegates(self, tmp_path: Path) -> None:
        from kennel.types import TaskType

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        task = add_task(work_dir, "Task C", TaskType.SPEC)
        Tasks(work_dir).complete_by_id(task["id"])
        assert list_tasks(work_dir)[0]["status"] == "completed"

    def test_has_pending_for_comment_delegates(self, tmp_path: Path) -> None:
        from kennel.types import TaskType

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        add_task(work_dir, "Task D", TaskType.THREAD, thread={"comment_id": 7})
        assert Tasks(work_dir).has_pending_for_comment(7) is True
        assert Tasks(work_dir).has_pending_for_comment(99) is False

    def test_remove_delegates(self, tmp_path: Path) -> None:
        from kennel.types import TaskType

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        task = add_task(work_dir, "Task E", TaskType.SPEC)
        assert Tasks(work_dir).remove(task["id"]) is True
        assert list_tasks(work_dir) == []

    def test_update_delegates(self, tmp_path: Path) -> None:
        from kennel.types import TaskStatus, TaskType

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
        from kennel.types import TaskType

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        add_task(work_dir, "Existing task", TaskType.SPEC)
        with Tasks(work_dir).modify() as tasks:
            tasks[0]["title"] = "Modified task"
        assert list_tasks(work_dir)[0]["title"] == "Modified task"
