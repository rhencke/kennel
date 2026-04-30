from pathlib import Path
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
    return [type(row.status).__name__ for row in rows]


def _function_source(source: str, name: str) -> str:
    start = source.index(f"def {name}(")
    next_function = source.find("\ndef ", start + 1)
    if next_function == -1:
        return source[start:]
    return source[start:next_function]


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

    assert [row.task for row in projected] == [3, 1, 2]
    assert [row.title for row in projected] == [
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
    assert [row.task for row in after_add.visible_pr_body] == [2, 1]

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
    assert [row.title for row in after_rescope.visible_pr_body] == [
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


def test_pr_body_list_equality_lowers_to_native_equality() -> None:
    source = Path(oracle.__file__).read_text()

    assert "def pr_body_eqb(" not in source
    assert "return visible == projected" in source


def test_positive_membership_lowers_to_native_membership() -> None:
    source = Path(oracle.__file__).read_text()

    assert "def positive_mem(" not in source
    assert "if task in snapshot_order:" in source


def test_imported_record_fields_lower_to_attribute_access() -> None:
    source = Path(oracle.__file__).read_text()

    assert "status(row)" not in source
    assert "kind(row)" not in source
    assert "title(row)" not in source
    assert "description(row)" not in source
    assert "match row.status:" in source
    assert "row.kind" in source


def test_pr_body_row_fields_drop_redundant_record_prefix() -> None:
    source = Path(oracle.__file__).read_text()
    pr_body_row = source[
        source.index("class PRBodyRow:") : source.index(
            "\ndef task_kind_matches_ci_filter",
        )
    ]

    assert "    task: int" in pr_body_row
    assert "    title: str" in pr_body_row
    assert "    description: str" in pr_body_row
    assert "    kind: TaskKind" in pr_body_row
    assert "    status: PRBodyStatus" in pr_body_row
    assert "pr_body_" not in pr_body_row


def test_tail_recursive_duplicate_scans_lower_to_for_loops() -> None:
    source = Path(oracle.__file__).read_text()
    comment_duplicate = _function_source(source, "find_comment_duplicate")
    pending_title_duplicate = _function_source(source, "find_pending_title_duplicate")

    assert "while True:" not in comment_duplicate
    assert "for task in order:" in comment_duplicate
    assert "while True:" not in pending_title_duplicate
    assert "for task in order:" in pending_title_duplicate
