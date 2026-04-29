from typing import Any

from fido.rocq import pr_body_task_store as oracle


def _row(
    title: str,
    kind: Any,
    status: Any = oracle.StatusPending(),
    description: str = "",
) -> Any:
    return oracle.TaskRow(
        title=title,
        description=description,
        kind=kind,
        status=status,
        source_comment=None,
    )


def _status_names(rows: list[Any]) -> list[str]:
    return [type(row.pr_body_status).__name__ for row in rows]


def test_projection_matches_rendered_pr_body_order() -> None:
    store = oracle.TaskStore(
        task_store_order=[1, 2, 3, 4],
        task_store_rows={
            1: _row("spec work", oracle.TaskSpec()),
            2: _row("completed work", oracle.TaskThread(), oracle.StatusCompleted()),
            3: _row("ci fix", oracle.TaskCI()),
            4: _row("blocked question", oracle.TaskAsk(), oracle.StatusBlocked()),
        },
    )

    projected = oracle.project_task_store(store)

    assert [row.pr_body_task for row in projected] == [3, 1, 2]
    assert [row.pr_body_title for row in projected] == [
        "ci fix",
        "spec work",
        "completed work",
    ]
    assert _status_names(projected) == ["PRPending", "PRPending", "PRCompleted"]


def test_transition_resyncs_after_add_complete_and_rescope() -> None:
    initial_store = oracle.TaskStore(
        task_store_order=[1],
        task_store_rows={1: _row("spec work", oracle.TaskSpec())},
    )
    state = oracle.synced_state(initial_store)
    added_row = _row("ci fix", oracle.TaskCI())

    after_add = oracle.transition(state, oracle.WriteTaskAdd(2, added_row))
    assert after_add is not None
    assert oracle.pr_body_matches_store_bool(after_add)
    assert [row.pr_body_task for row in after_add.visible_pr_body] == [2, 1]

    after_complete = oracle.transition(after_add, oracle.WriteTaskComplete(2))
    assert after_complete is not None
    assert oracle.pr_body_matches_store_bool(after_complete)
    assert _status_names(after_complete.visible_pr_body) == ["PRPending", "PRCompleted"]

    after_rescope = oracle.transition(
        after_complete,
        oracle.WriteTaskRescope([1], [oracle.RewriteTask(1, "renamed spec", "done")]),
    )
    assert after_rescope is not None
    assert oracle.pr_body_matches_store_bool(after_rescope)
    assert [row.pr_body_title for row in after_rescope.visible_pr_body] == [
        "renamed spec",
        "ci fix",
    ]


def test_transition_rejects_stale_pr_body_before_next_write() -> None:
    store = oracle.TaskStore(
        task_store_order=[1],
        task_store_rows={1: _row("spec work", oracle.TaskSpec())},
    )
    stale_state = oracle.SystemState(durable_task_store=store, visible_pr_body=[])

    assert not oracle.pr_body_matches_store_bool(stale_state)
    assert oracle.transition(stale_state, oracle.WriteTaskComplete(1)) is None
