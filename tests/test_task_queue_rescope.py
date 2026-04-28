from pathlib import Path
from unittest.mock import MagicMock

from fido.events import _maybe_abort_for_new_task
from fido.rocq import task_queue_rescope as oracle
from fido.state import State
from fido.tasks import Tasks, _apply_reorder
from fido.types import TaskStatus, TaskType
from fido.worker import _pick_next_task


def _kind(task_type: str, title: str) -> object:
    upper = title.upper()
    if upper.startswith("ASK:"):
        return oracle.TaskAsk()
    if upper.startswith("DEFER:"):
        return oracle.TaskDefer()
    match task_type:
        case "ci":
            return oracle.TaskCI()
        case "thread":
            return oracle.TaskThread()
        case _:
            return oracle.TaskSpec()


def _status(status: str) -> object:
    match status:
        case "completed":
            return oracle.StatusCompleted()
        case "blocked":
            return oracle.StatusBlocked()
        case _:
            return oracle.StatusPending()


def _row(task: dict) -> object:
    thread = task.get("thread") or {}
    comment_id = thread.get("comment_id")
    return oracle.TaskRow(
        task_title=task["title"],
        task_description=task.get("description", ""),
        task_kind=_kind(task.get("type", "spec"), task["title"]),
        task_status=_status(task.get("status", "pending")),
        task_source_comment=comment_id,
    )


def _oracle_state(
    tasks: list[dict],
) -> tuple[dict[str, int], list[int], dict[int, object]]:
    ids = {task["id"]: index for index, task in enumerate(tasks, 1)}
    order = [ids[task["id"]] for task in tasks]
    rows = {ids[task["id"]]: _row(task) for task in tasks}
    return ids, order, rows


def _materialize(
    order: list[int], rows: dict[int, object], inverse: dict[int, str]
) -> list[dict]:
    materialized: list[dict] = []
    for task_id in order:
        row = rows[task_id]
        materialized.append(
            {
                "id": inverse[task_id],
                "title": row.task_title,
                "description": row.task_description,
                "status": type(row.task_status).__name__.removeprefix("Status").lower(),
            }
        )
    return materialized


def _rescope_ops(
    snapshot: list[dict], ordered_items: list[dict]
) -> tuple[list[int], list[object]]:
    ids = {task["id"]: index for index, task in enumerate(snapshot, 1)}
    ordered_by_id = {item["id"]: item for item in ordered_items}
    snapshot_order: list[int] = []
    ops: list[object] = []
    for task in snapshot:
        if task.get("status") == "completed":
            continue
        task_id = ids[task["id"]]
        snapshot_order.append(task_id)
        item = ordered_by_id.get(task["id"])
        if item is None:
            ops.append(oracle.CompleteTask(task_id))
            continue
        if item.get("title", task["title"]) != task["title"] or item.get(
            "description", task.get("description", "")
        ) != task.get("description", ""):
            ops.append(
                oracle.RewriteTask(
                    task_id,
                    item.get("title", task["title"]),
                    item.get("description", task.get("description", "")),
                )
            )
        else:
            ops.append(oracle.KeepTask(task_id))
    return snapshot_order, ops


def _enqueue(
    task_id: int,
    row: object,
    order: list[int],
    rows: dict[int, object],
) -> tuple[list[int], dict[int, object], int]:
    pair, created = oracle.enqueue_task(task_id, row, order, rows)
    next_order, next_rows = pair
    return next_order, next_rows, created


def test_model_enqueue_pick_and_lease_lifecycle() -> None:
    row1 = oracle.TaskRow(
        task_title="Implement feature",
        task_description="",
        task_kind=oracle.TaskSpec(),
        task_status=oracle.StatusPending(),
        task_source_comment=None,
    )
    row2 = oracle.TaskRow(
        task_title="Fix CI",
        task_description="",
        task_kind=oracle.TaskCI(),
        task_status=oracle.StatusPending(),
        task_source_comment=None,
    )
    row3 = oracle.TaskRow(
        task_title="ASK: should we widen this?",
        task_description="",
        task_kind=oracle.TaskAsk(),
        task_status=oracle.StatusPending(),
        task_source_comment=None,
    )

    order: list[int] = []
    rows: dict[int, object] = {}
    order, rows, created = _enqueue(1, row1, order, rows)
    assert created == 1
    order, rows, created = _enqueue(2, row2, order, rows)
    assert created == 2
    order, rows, created = _enqueue(3, row3, order, rows)
    assert created == 3

    assert oracle.pick_next_task(order, rows) == 2
    lease = oracle.begin_task(2, None, rows)
    assert lease == 2
    lease, rows = oracle.complete_task(2, lease, rows)
    assert lease is None
    assert type(rows[2].task_status).__name__ == "StatusCompleted"


def test_model_enqueue_dedups_by_comment_and_pending_title() -> None:
    thread_row = oracle.TaskRow(
        task_title="Review request",
        task_description="",
        task_kind=oracle.TaskThread(),
        task_status=oracle.StatusPending(),
        task_source_comment=77,
    )
    spec_row = oracle.TaskRow(
        task_title="Implement feature",
        task_description="",
        task_kind=oracle.TaskSpec(),
        task_status=oracle.StatusPending(),
        task_source_comment=None,
    )
    order, rows, first = _enqueue(1, thread_row, [], {})
    order, rows, duplicate = _enqueue(2, thread_row, order, rows)
    assert duplicate == first
    assert order == [1]

    order, rows, first_spec = _enqueue(3, spec_row, order, rows)
    order, rows, duplicate_spec = _enqueue(4, spec_row, order, rows)
    assert duplicate_spec == first_spec


def test_model_rescope_and_abort_helpers() -> None:
    rows = {
        1: oracle.TaskRow(
            "spec one", "old", oracle.TaskSpec(), oracle.StatusPending(), None
        ),
        2: oracle.TaskRow(
            "thread two", "", oracle.TaskThread(), oracle.StatusPending(), 42
        ),
        3: oracle.TaskRow(
            "done", "", oracle.TaskSpec(), oracle.StatusCompleted(), None
        ),
        4: oracle.TaskRow(
            "late arrival", "", oracle.TaskSpec(), oracle.StatusPending(), None
        ),
        5: oracle.TaskRow("ci fix", "", oracle.TaskCI(), oracle.StatusPending(), None),
    }
    snapshot_order = [1, 2]
    current_order = [1, 2, 3, 4, 5]
    ops = [
        oracle.RewriteTask(2, "thread two rewritten", "new"),
        oracle.CompleteTask(1),
    ]
    assert oracle.rescope_ops_cover_snapshot(snapshot_order, ops)

    order, updated_rows = oracle.apply_rescope(snapshot_order, current_order, rows, ops)
    assert order == [2, 3, 1, 4, 5]
    assert updated_rows[2].task_title == "thread two"
    assert type(updated_rows[1].task_status).__name__ == "StatusCompleted"
    assert updated_rows[2].task_description == "new"
    assert updated_rows[2].task_source_comment == 42
    assert oracle.rescope_preserves_task_identity(snapshot_order, rows, updated_rows)

    lease = 2
    assert oracle.rescope_affects_active_task(lease, rows, updated_rows)
    assert oracle.should_abort_for_new_task(5, lease, rows)
    assert not oracle.should_abort_for_new_task(4, lease, rows)
    assert oracle.abort_task(2, lease) is None


def test_model_unblock_tasks() -> None:
    rows = {
        1: oracle.TaskRow(
            "blocked", "", oracle.TaskSpec(), oracle.StatusBlocked(), None
        ),
        2: oracle.TaskRow(
            "done", "", oracle.TaskSpec(), oracle.StatusCompleted(), None
        ),
    }
    unblocked = oracle.unblock_tasks([1, 2], rows)
    assert type(unblocked[1].task_status).__name__ == "StatusPending"
    assert type(unblocked[2].task_status).__name__ == "StatusCompleted"


def test_pick_next_task_matches_oracle() -> None:
    tasks = [
        {"id": "t1", "title": "ASK: need detail", "status": "pending", "type": "spec"},
        {"id": "t2", "title": "Implement feature", "status": "pending", "type": "spec"},
        {"id": "t3", "title": "Fix CI", "status": "pending", "type": "ci"},
        {"id": "t4", "title": "blocked", "status": "blocked", "type": "thread"},
    ]
    ids, order, rows = _oracle_state(tasks)
    selected = oracle.pick_next_task(order, rows)
    picked = _pick_next_task(tasks)
    assert picked is not None
    assert picked["id"] == "t3"
    assert selected == ids["t3"]


def test_apply_reorder_matches_oracle() -> None:
    current = [
        {
            "id": "t1",
            "title": "Implement feature",
            "description": "old",
            "status": "pending",
            "type": "spec",
        },
        {
            "id": "t2",
            "title": "Comment follow-up",
            "description": "",
            "status": "pending",
            "type": "thread",
            "thread": {"comment_id": 55},
        },
        {
            "id": "t3",
            "title": "Already done",
            "description": "",
            "status": "completed",
            "type": "spec",
        },
        {
            "id": "t4",
            "title": "Late task",
            "description": "",
            "status": "pending",
            "type": "spec",
        },
        {
            "id": "t5",
            "title": "Fix CI",
            "description": "",
            "status": "pending",
            "type": "ci",
        },
    ]
    ordered_items = [
        {"id": "t2", "title": "Comment follow-up", "description": "new"},
        {"id": "t5", "title": "Fix CI", "description": ""},
    ]

    runtime = _apply_reorder(
        current, ordered_items, frozenset(t["id"] for t in current)
    )
    ids, current_order, rows = _oracle_state(current)
    inverse = {value: key for key, value in ids.items()}
    snapshot_order, ops = _rescope_ops(current, ordered_items)
    oracle_order, oracle_rows = oracle.apply_rescope(
        snapshot_order, current_order, rows, ops
    )
    materialized = _materialize(oracle_order, oracle_rows, inverse)

    assert [(task["id"], task["title"], task["status"]) for task in runtime] == [
        (task["id"], task["title"], task["status"]) for task in materialized
    ]


def test_rescope_identity_rejects_title_and_comment_drift() -> None:
    before = oracle.TaskRow(
        task_title="Thread follow-up",
        task_description="old",
        task_kind=oracle.TaskThread(),
        task_status=oracle.StatusPending(),
        task_source_comment=55,
    )
    rows_before = {1: before}

    _order, rows_after = oracle.apply_rescope(
        [1],
        [1],
        rows_before,
        [oracle.RewriteTask(1, "Different title", "new")],
    )
    assert rows_after[1].task_title == "Thread follow-up"
    assert rows_after[1].task_source_comment == 55
    assert rows_after[1].task_description == "new"
    assert not before.identity_changed(rows_after[1])
    assert oracle.rescope_preserves_task_identity([1], rows_before, rows_after)

    changed_title = oracle.TaskRow(
        task_title="Different title",
        task_description="new",
        task_kind=oracle.TaskThread(),
        task_status=oracle.StatusPending(),
        task_source_comment=55,
    )
    changed_comment = oracle.TaskRow(
        task_title="Thread follow-up",
        task_description="new",
        task_kind=oracle.TaskThread(),
        task_status=oracle.StatusPending(),
        task_source_comment=77,
    )
    assert before.identity_changed(changed_title)
    assert before.identity_changed(changed_comment)
    assert not oracle.rescope_preserves_task_identity(
        [1], rows_before, {1: changed_title}
    )
    assert not oracle.rescope_preserves_task_identity(
        [1], rows_before, {1: changed_comment}
    )


def test_batched_rescope_converges_for_permuted_releases() -> None:
    rows = {
        1: oracle.TaskRow(
            "Thread follow-up",
            "old",
            oracle.TaskThread(),
            oracle.StatusPending(),
            55,
        ),
        2: oracle.TaskRow(
            "Spec task",
            "",
            oracle.TaskSpec(),
            oracle.StatusPending(),
            None,
        ),
        3: oracle.TaskRow(
            "Fix CI",
            "",
            oracle.TaskCI(),
            oracle.StatusPending(),
            None,
        ),
        4: oracle.TaskRow(
            "Already done",
            "",
            oracle.TaskSpec(),
            oracle.StatusCompleted(),
            None,
        ),
        5: oracle.TaskRow(
            "Late arrival",
            "",
            oracle.TaskSpec(),
            oracle.StatusPending(),
            None,
        ),
    }
    snapshot_order = [1, 2, 3]
    current_order = [1, 2, 3, 4, 5]
    act_rewrite = oracle.RescopeRelease(
        oracle.ReleaseACT(),
        oracle.RewriteTask(1, "Title drift ignored", "new detail"),
    )
    do_complete = oracle.RescopeRelease(oracle.ReleaseDO(), oracle.CompleteTask(2))
    act_keep = oracle.RescopeRelease(oracle.ReleaseACT(), oracle.KeepTask(3))

    left_releases = [act_rewrite, do_complete, act_keep]
    right_releases = [act_keep, act_rewrite, do_complete]

    assert oracle.normalize_rescope_batch(snapshot_order, left_releases) == (
        oracle.normalize_rescope_batch(snapshot_order, right_releases)
    )

    left_order, left_rows = oracle.apply_batched_rescope(
        snapshot_order, current_order, rows, left_releases
    )
    right_order, right_rows = oracle.apply_batched_rescope(
        snapshot_order, current_order, rows, right_releases
    )

    assert left_order == [3, 1, 4, 2, 5]
    assert left_order == right_order
    assert left_rows == right_rows
    assert left_rows[1].task_title == "Thread follow-up"
    assert left_rows[1].task_description == "new detail"
    assert type(left_rows[2].task_status).__name__ == "StatusCompleted"
    assert oracle.rescope_preserves_task_identity(snapshot_order, rows, left_rows)
    assert oracle.batched_rescope_materially_significant(
        snapshot_order, current_order, rows, left_releases
    )
    assert oracle.batched_rescope_materially_significant(
        snapshot_order, current_order, rows, right_releases
    )


def test_abort_decision_matches_oracle(tmp_path: Path) -> None:
    repo_cfg = MagicMock()
    repo_cfg.work_dir = tmp_path
    repo_cfg.name = "owner/repo"
    State(tmp_path / ".git" / "fido").save({"current_task_id": "t-current"})
    task_list = [
        {
            "id": "t-current",
            "title": "Current spec",
            "status": "pending",
            "type": "spec",
        },
        {
            "id": "t-thread",
            "title": "Review follow-up",
            "status": "pending",
            "type": "thread",
            "thread": {"comment_id": 55},
        },
    ]
    new_task = task_list[1]
    registry = MagicMock()
    ids, _order, rows = _oracle_state(task_list)
    lease = ids["t-current"]
    expected = oracle.should_abort_for_new_task(ids["t-thread"], lease, rows)

    _maybe_abort_for_new_task(
        repo_cfg,
        new_task,
        registry,
        _state=State(tmp_path / ".git" / "fido"),
        _tasks=MagicMock(list=MagicMock(return_value=task_list)),
    )

    assert expected is True
    registry.abort_task.assert_called_once_with("owner/repo")


def test_tasks_unblock_matches_oracle(tmp_path: Path) -> None:
    tasks = Tasks(tmp_path)
    blocked = tasks.add("blocked", TaskType.SPEC, status=TaskStatus.BLOCKED)
    done = tasks.add("done", TaskType.SPEC, status=TaskStatus.COMPLETED)

    ids, order, rows = _oracle_state(tasks.list())
    oracle_rows = oracle.unblock_tasks(order, rows)
    tasks.unblock_tasks()
    runtime = tasks.list()

    assert next(t for t in runtime if t["id"] == blocked["id"])["status"] == "pending"
    assert next(t for t in runtime if t["id"] == done["id"])["status"] == "completed"
    assert type(oracle_rows[ids[blocked["id"]]].task_status).__name__ == "StatusPending"
    assert type(oracle_rows[ids[done["id"]]].task_status).__name__ == "StatusCompleted"


def test_complete_task_visible_idempotency() -> None:
    # Thread task with source comment: first call marks completed, returns comment id
    thread_row = oracle.TaskRow(
        task_title="Review follow-up",
        task_description="",
        task_kind=oracle.TaskThread(),
        task_status=oracle.StatusPending(),
        task_source_comment=99,
    )
    rows: dict[int, object] = {1: thread_row}
    rows_after, comment_id = oracle.complete_task_visible(1, rows)
    assert comment_id == 99
    assert type(rows_after[1].task_status).__name__ == "StatusCompleted"

    # Second call is idempotent: already completed → returns None, rows unchanged
    rows_again, comment_id2 = oracle.complete_task_visible(1, rows_after)
    assert comment_id2 is None
    assert rows_again is rows_after

    # Spec task (no source comment): completes but returns None on first call too
    spec_row = oracle.TaskRow(
        task_title="Implement feature",
        task_description="",
        task_kind=oracle.TaskSpec(),
        task_status=oracle.StatusPending(),
        task_source_comment=None,
    )
    rows2: dict[int, object] = {2: spec_row}
    rows2_after, comment_id3 = oracle.complete_task_visible(2, rows2)
    assert comment_id3 is None
    assert type(rows2_after[2].task_status).__name__ == "StatusCompleted"

    # Blocked thread task: completes and returns comment id on first call
    blocked_thread_row = oracle.TaskRow(
        task_title="Blocked review",
        task_description="",
        task_kind=oracle.TaskThread(),
        task_status=oracle.StatusBlocked(),
        task_source_comment=77,
    )
    rows3: dict[int, object] = {3: blocked_thread_row}
    rows3_after, comment_id4 = oracle.complete_task_visible(3, rows3)
    assert comment_id4 == 77
    assert type(rows3_after[3].task_status).__name__ == "StatusCompleted"

    # Re-call on the now-completed blocked task returns None
    rows3_again, comment_id5 = oracle.complete_task_visible(3, rows3_after)
    assert comment_id5 is None


def test_task_change_detection() -> None:
    # Baseline rows before rescope: two thread tasks, one spec task
    thread1_row = oracle.TaskRow(
        task_title="First review",
        task_description="old desc",
        task_kind=oracle.TaskThread(),
        task_status=oracle.StatusPending(),
        task_source_comment=101,
    )
    thread2_row = oracle.TaskRow(
        task_title="Second review",
        task_description="",
        task_kind=oracle.TaskThread(),
        task_status=oracle.StatusPending(),
        task_source_comment=202,
    )
    spec_row = oracle.TaskRow(
        task_title="Spec task",
        task_description="",
        task_kind=oracle.TaskSpec(),
        task_status=oracle.StatusPending(),
        task_source_comment=None,
    )
    rows_before: dict[int, object] = {1: thread1_row, 2: thread2_row, 3: spec_row}

    # Case 1: task rewritten → TaskModified with new title/description
    rewritten_row = oracle.TaskRow(
        task_title="First review (updated)",
        task_description="new desc",
        task_kind=oracle.TaskThread(),
        task_status=oracle.StatusPending(),
        task_source_comment=101,
    )
    rows_rewritten = {1: rewritten_row, 2: thread2_row, 3: spec_row}
    change = oracle.task_change(1, rows_before, rows_rewritten)
    assert isinstance(change, oracle.TaskModified)
    assert change.task == 1
    assert change.new_title == "First review (updated)"
    assert change.new_description == "new desc"

    # Case 2: task completed → TaskCompleted
    completed_row = oracle.TaskRow(
        task_title="First review",
        task_description="old desc",
        task_kind=oracle.TaskThread(),
        task_status=oracle.StatusCompleted(),
        task_source_comment=101,
    )
    rows_completed = {1: completed_row, 2: thread2_row, 3: spec_row}
    change2 = oracle.task_change(1, rows_before, rows_completed)
    assert isinstance(change2, oracle.TaskCompleted)
    assert change2.task == 1

    # Case 3: task absent from rows_after → TaskCancelled
    rows_cancelled = {2: thread2_row, 3: spec_row}
    change3 = oracle.task_change(1, rows_before, rows_cancelled)
    assert isinstance(change3, oracle.TaskCancelled)
    assert change3.task == 1

    # Case 4: spec task (no source comment) → None regardless of changes
    spec_rewritten = oracle.TaskRow(
        task_title="Spec task new title",
        task_description="updated",
        task_kind=oracle.TaskSpec(),
        task_status=oracle.StatusPending(),
        task_source_comment=None,
    )
    rows_spec_changed = {1: thread1_row, 2: thread2_row, 3: spec_rewritten}
    change4 = oracle.task_change(3, rows_before, rows_spec_changed)
    assert change4 is None

    # Case 5: no change to thread task → None
    change5 = oracle.task_change(2, rows_before, rows_before)
    assert change5 is None


def test_compute_task_changes_aggregates_in_snapshot_order() -> None:
    thread1_row = oracle.TaskRow(
        task_title="First review",
        task_description="",
        task_kind=oracle.TaskThread(),
        task_status=oracle.StatusPending(),
        task_source_comment=101,
    )
    thread2_row = oracle.TaskRow(
        task_title="Second review",
        task_description="",
        task_kind=oracle.TaskThread(),
        task_status=oracle.StatusPending(),
        task_source_comment=202,
    )
    spec_row = oracle.TaskRow(
        task_title="Spec task",
        task_description="",
        task_kind=oracle.TaskSpec(),
        task_status=oracle.StatusPending(),
        task_source_comment=None,
    )
    late_row = oracle.TaskRow(
        task_title="Late thread task",
        task_description="",
        task_kind=oracle.TaskThread(),
        task_status=oracle.StatusPending(),
        task_source_comment=303,
    )
    rows_before: dict[int, object] = {
        1: thread1_row,
        2: thread2_row,
        3: spec_row,
        4: late_row,
    }
    snapshot_order = [1, 2, 3]  # task 4 arrived after snapshot

    # After rescope: task 1 completed, task 2 rewritten, task 3 unchanged (spec)
    completed_row = oracle.TaskRow(
        task_title="First review",
        task_description="",
        task_kind=oracle.TaskThread(),
        task_status=oracle.StatusCompleted(),
        task_source_comment=101,
    )
    rewritten_row = oracle.TaskRow(
        task_title="Second review (updated)",
        task_description="now with detail",
        task_kind=oracle.TaskThread(),
        task_status=oracle.StatusPending(),
        task_source_comment=202,
    )
    rows_after: dict[int, object] = {
        1: completed_row,
        2: rewritten_row,
        3: spec_row,
        4: late_row,
    }

    changes = oracle.compute_task_changes(snapshot_order, rows_before, rows_after)

    # Only thread tasks with changes appear, in snapshot order
    assert len(changes) == 2
    assert isinstance(changes[0], oracle.TaskCompleted)
    assert changes[0].task == 1
    assert isinstance(changes[1], oracle.TaskModified)
    assert changes[1].task == 2
    assert changes[1].new_title == "Second review (updated)"
    assert changes[1].new_description == "now with detail"

    # Task 4 (post-snapshot) not included even though it's a thread task
    task4_change = oracle.task_change(4, rows_before, rows_after)
    assert task4_change is None  # task 4 wasn't in rows_before with a status change


def test_active_task_ownership_lifecycle() -> None:
    spec_row = oracle.TaskRow(
        task_title="Implement feature",
        task_description="",
        task_kind=oracle.TaskSpec(),
        task_status=oracle.StatusPending(),
        task_source_comment=None,
    )
    ask_row = oracle.TaskRow(
        task_title="ASK: should we widen?",
        task_description="",
        task_kind=oracle.TaskAsk(),
        task_status=oracle.StatusPending(),
        task_source_comment=None,
    )
    rows: dict[int, object] = {1: spec_row, 2: ask_row}

    # No lease held — begin_task acquires lease for an executable task
    lease = oracle.begin_task(1, None, rows)
    assert lease == 1

    # Lease already held — begin_task returns None even for a different task
    second_attempt = oracle.begin_task(2, lease, rows)
    assert second_attempt is None

    # Non-executable task (ASK kind) — begin_task returns None even with no lease
    ask_lease = oracle.begin_task(2, None, rows)
    assert ask_lease is None

    # Task not in rows — begin_task returns None
    missing_lease = oracle.begin_task(99, None, rows)
    assert missing_lease is None

    # abort_task clears matching lease
    after_abort = oracle.abort_task(1, lease)
    assert after_abort is None

    # abort_task is a no-op when task doesn't match lease
    other_lease: object = 2
    after_wrong_abort = oracle.abort_task(1, other_lease)
    assert after_wrong_abort == other_lease

    # abort_task is a no-op when no lease held
    after_no_lease_abort = oracle.abort_task(1, None)
    assert after_no_lease_abort is None

    # complete_task clears matching lease and marks task completed
    lease = oracle.begin_task(1, None, rows)
    new_lease, rows_after = oracle.complete_task(1, lease, rows)
    assert new_lease is None
    assert type(rows_after[1].task_status).__name__ == "StatusCompleted"
    # Other tasks unchanged
    assert type(rows_after[2].task_status).__name__ == "StatusPending"

    # complete_task with non-matching lease leaves lease intact
    other_lease2: object = 2
    new_lease2, rows_after2 = oracle.complete_task(1, other_lease2, rows)
    assert new_lease2 == other_lease2
    assert type(rows_after2[1].task_status).__name__ == "StatusCompleted"

    # task_still_pending returns True for pending, False for completed or missing
    assert oracle.task_still_pending(2, rows)
    assert not oracle.task_still_pending(1, rows_after)
    assert not oracle.task_still_pending(99, rows)


def test_cleanup_aborted_task() -> None:
    spec_row = oracle.TaskRow(
        task_title="Spec task",
        task_description="",
        task_kind=oracle.TaskSpec(),
        task_status=oracle.StatusPending(),
        task_source_comment=None,
    )
    other_row = oracle.TaskRow(
        task_title="Other task",
        task_description="",
        task_kind=oracle.TaskSpec(),
        task_status=oracle.StatusPending(),
        task_source_comment=None,
    )
    rows: dict[int, object] = {1: spec_row, 2: other_row}
    order = [1, 2]
    lease: object = 1

    # cleanup_aborted_task clears the matching lease, removes task from order and rows
    (new_lease, new_order), new_rows = oracle.cleanup_aborted_task(
        1, lease, order, rows
    )
    assert new_lease is None
    assert new_order == [2]
    assert 1 not in new_rows
    assert 2 in new_rows

    # cleanup_aborted_task with non-matching lease leaves lease intact
    other_lease: object = 2
    (new_lease2, new_order2), new_rows2 = oracle.cleanup_aborted_task(
        1, other_lease, order, rows
    )
    assert new_lease2 == other_lease  # non-matching lease preserved
    assert new_order2 == [2]
    assert 1 not in new_rows2

    # cleanup_aborted_task with None lease leaves lease None
    (new_lease3, new_order3), new_rows3 = oracle.cleanup_aborted_task(
        1, None, order, rows
    )
    assert new_lease3 is None
    assert new_order3 == [2]
    assert 1 not in new_rows3

    # cleanup_aborted_task on task not in order/rows is a no-op for those structures
    (new_lease4, new_order4), new_rows4 = oracle.cleanup_aborted_task(
        99, lease, order, rows
    )
    assert new_lease4 == lease  # task 99 doesn't match task 1's lease
    assert new_order4 == [1, 2]
    assert 1 in new_rows4 and 2 in new_rows4

    # remove_from_order removes only the target, preserves duplicates of other ids
    order_with_extras = [3, 1, 3, 2, 1]
    stripped = oracle.remove_from_order(1, order_with_extras)
    assert stripped == [3, 3, 2]


def test_create_task_dedup_and_abort_decision_integration() -> None:
    """Full integration: enqueue tasks with dedup, acquire a lease, then
    enqueue higher/lower/equal priority tasks and check abort decisions."""
    order: list[int] = []
    rows: dict[int, object] = {}

    # --- Phase 1: build a realistic queue via enqueue_task ---

    spec_row = oracle.TaskRow(
        task_title="Implement feature",
        task_description="",
        task_kind=oracle.TaskSpec(),
        task_status=oracle.StatusPending(),
        task_source_comment=None,
    )
    thread_row = oracle.TaskRow(
        task_title="Review follow-up",
        task_description="",
        task_kind=oracle.TaskThread(),
        task_status=oracle.StatusPending(),
        task_source_comment=42,
    )
    ci_row = oracle.TaskRow(
        task_title="Fix CI",
        task_description="",
        task_kind=oracle.TaskCI(),
        task_status=oracle.StatusPending(),
        task_source_comment=None,
    )

    # Enqueue spec task (id 1)
    order, rows, created_spec = _enqueue(1, spec_row, order, rows)
    assert created_spec == 1

    # Enqueue thread task (id 2)
    order, rows, created_thread = _enqueue(2, thread_row, order, rows)
    assert created_thread == 2
    assert order == [1, 2]

    # Dedup: re-enqueue same comment_id → returns existing thread task
    thread_row_dup = oracle.TaskRow(
        task_title="Different title same comment",
        task_description="",
        task_kind=oracle.TaskThread(),
        task_status=oracle.StatusPending(),
        task_source_comment=42,
    )
    order, rows, dup_thread = _enqueue(3, thread_row_dup, order, rows)
    assert dup_thread == 2  # deduped to existing
    assert 3 not in [rows.get(k) for k in rows]  # task 3 never added to rows

    # Dedup: re-enqueue same spec title while pending → returns existing
    spec_row_dup = oracle.TaskRow(
        task_title="Implement feature",
        task_description="different desc",
        task_kind=oracle.TaskSpec(),
        task_status=oracle.StatusPending(),
        task_source_comment=None,
    )
    order, rows, dup_spec = _enqueue(4, spec_row_dup, order, rows)
    assert dup_spec == 1  # deduped to existing
    assert order == [1, 2]

    # Complete spec task, then re-enqueue same title → no dedup (original completed)
    lease = oracle.begin_task(1, None, rows)
    _, rows = oracle.complete_task(1, lease, rows)
    order, rows, new_spec = _enqueue(5, spec_row, order, rows)
    assert new_spec == 5  # fresh entry, not deduped
    assert order == [1, 2, 5]

    # --- Phase 2: acquire lease on spec task 5, test abort decisions ---

    lease = oracle.begin_task(5, None, rows)
    assert lease == 5

    # CI task (rank 0) preempts spec (rank 2) → should abort
    order, rows, created_ci = _enqueue(6, ci_row, order, rows)
    assert created_ci == 6
    assert oracle.should_abort_for_new_task(6, lease, rows) is True

    # Thread task (rank 1) preempts spec (rank 2) → should abort
    thread_row2 = oracle.TaskRow(
        task_title="Another review",
        task_description="",
        task_kind=oracle.TaskThread(),
        task_status=oracle.StatusPending(),
        task_source_comment=99,
    )
    order, rows, created_thread2 = _enqueue(7, thread_row2, order, rows)
    assert oracle.should_abort_for_new_task(7, lease, rows) is True

    # Spec task (rank 2) does NOT preempt spec (rank 2) → no abort
    spec_row2 = oracle.TaskRow(
        task_title="Another feature",
        task_description="",
        task_kind=oracle.TaskSpec(),
        task_status=oracle.StatusPending(),
        task_source_comment=None,
    )
    order, rows, created_spec2 = _enqueue(8, spec_row2, order, rows)
    assert oracle.should_abort_for_new_task(8, lease, rows) is False

    # ASK task (no rank) does NOT cause abort
    ask_row = oracle.TaskRow(
        task_title="ASK: expand scope?",
        task_description="",
        task_kind=oracle.TaskAsk(),
        task_status=oracle.StatusPending(),
        task_source_comment=None,
    )
    order, rows, created_ask = _enqueue(9, ask_row, order, rows)
    assert oracle.should_abort_for_new_task(9, lease, rows) is False

    # DEFER task (no rank) does NOT cause abort
    defer_row = oracle.TaskRow(
        task_title="DEFER: out of scope",
        task_description="",
        task_kind=oracle.TaskDefer(),
        task_status=oracle.StatusPending(),
        task_source_comment=None,
    )
    order, rows, created_defer = _enqueue(10, defer_row, order, rows)
    assert oracle.should_abort_for_new_task(10, lease, rows) is False

    # Same task as active lease → no abort (self-reference)
    assert oracle.should_abort_for_new_task(5, lease, rows) is False

    # No lease held → no abort regardless of priority
    assert oracle.should_abort_for_new_task(6, None, rows) is False

    # --- Phase 3: abort, pick next, verify CI-first ordering ---

    aborted_lease = oracle.abort_task(5, lease)
    assert aborted_lease is None

    # pick_next_task should select CI (rank 0) first
    next_task = oracle.pick_next_task(order, rows)
    assert next_task == 6  # CI task

    # --- Phase 4: switch lease to CI, thread no longer preempts ---

    ci_lease = oracle.begin_task(6, None, rows)
    assert ci_lease == 6
    # Thread (rank 1) does NOT preempt CI (rank 0)
    assert oracle.should_abort_for_new_task(7, ci_lease, rows) is False
    # Another CI (rank 0) does NOT preempt CI (rank 0)
    ci_row2 = oracle.TaskRow(
        task_title="Fix CI again",
        task_description="",
        task_kind=oracle.TaskCI(),
        task_status=oracle.StatusPending(),
        task_source_comment=None,
    )
    order, rows, created_ci2 = _enqueue(11, ci_row2, order, rows)
    assert oracle.should_abort_for_new_task(11, ci_lease, rows) is False
