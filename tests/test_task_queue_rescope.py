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
    assert updated_rows[2].task_title == "thread two rewritten"
    assert type(updated_rows[1].task_status).__name__ == "StatusCompleted"

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
        {"id": "t2", "title": "Comment follow-up rewritten", "description": "new"},
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
