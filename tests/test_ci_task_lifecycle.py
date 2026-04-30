from fido.rocq import ci_task_lifecycle as oracle


def _snapshot(
    run: int,
    name: str = "ci / test",
    conclusion: object | None = None,
) -> oracle.CIFailureSnapshot:
    return oracle.CIFailureSnapshot(
        ci_run=run,
        ci_check_name=name,
        ci_conclusion=conclusion or oracle.CIConclusionFailure(),
    )


def _unpack_record_failure(
    result: tuple[tuple[tuple[dict[int, object], list[int]], dict[int, object]], int],
) -> tuple[dict[int, object], list[int], dict[int, object], int]:
    pair, task = result
    store_order, rows = pair
    store, order = store_order
    return store, order, rows, task


def _unpack_resolved(
    result: tuple[tuple[dict[int, object], dict[int, object]], int | None],
) -> tuple[dict[int, object], dict[int, object], int | None]:
    pair, lease = result
    store, rows = pair
    return store, rows, lease


def _unpack_paused(
    result: tuple[tuple[dict[int, object], dict[int, object]], int | None],
) -> tuple[dict[int, object], dict[int, object], int | None]:
    pair, lease = result
    store, rows = pair
    return store, rows, lease


def test_latest_failure_reuses_live_ci_task() -> None:
    old_snapshot = _snapshot(10)
    new_snapshot = _snapshot(11, conclusion=oracle.CIConclusionTimedOut())
    row = old_snapshot.new_live_row(7)
    ci_store = {1: row}
    task_rows = {7: old_snapshot.task_row()}

    next_store, next_order, next_rows, task = _unpack_record_failure(
        oracle.record_ci_failure(1, new_snapshot, 8, ci_store, [7], task_rows)
    )

    assert task == 7
    assert next_order == [7]
    assert next_rows == task_rows
    assert next_store[1].ci_snapshot == new_snapshot
    assert isinstance(next_store[1].ci_phase, oracle.CIFailing)
    assert next_store[1].ci_task == 7


def test_fresh_ci_failure_creates_first_pickup_task() -> None:
    spec_row = oracle.TaskRow(
        title="Implement feature",
        description="",
        kind=oracle.TaskSpec(),
        status=oracle.StatusPending(),
        source_comment=None,
    )
    snapshot = _snapshot(10)

    ci_store, order, rows, task = _unpack_record_failure(
        oracle.record_ci_failure(1, snapshot, 2, {}, [1], {1: spec_row})
    )

    assert task == 2
    assert order == [1, 2]
    assert isinstance(ci_store[1].ci_phase, oracle.CIFailing)
    assert rows[2].title == "ci / test"
    assert isinstance(rows[2].kind, oracle.TaskCI)
    assert oracle.pick_next_task(order, rows) == 2


def test_failed_attempt_retries_until_fixed() -> None:
    row = oracle.CIRow(
        ci_snapshot=_snapshot(10),
        ci_phase=oracle.CIFixing(),
        ci_task=7,
        ci_attempts=4,
    )

    next_store = oracle.record_ci_attempt_failed(1, {1: row})

    assert isinstance(next_store[1].ci_phase, oracle.CIFailing)
    assert next_store[1].ci_task == 7
    assert next_store[1].ci_attempts == 5
    assert next_store[1].live_task() == 7


def test_human_pause_blocks_live_task_and_clears_lease() -> None:
    snapshot = _snapshot(10)
    row = oracle.CIRow(
        ci_snapshot=snapshot,
        ci_phase=oracle.CIFixing(),
        ci_task=7,
        ci_attempts=1,
    )
    task_rows = {7: snapshot.task_row()}

    next_store, next_rows, lease = _unpack_paused(
        oracle.pause_ci_for_human(1, {1: row}, task_rows, 7)
    )

    assert isinstance(next_store[1].ci_phase, oracle.CIPaused)
    assert next_store[1].ci_task == 7
    assert next_store[1].ci_attempts == 1
    assert next_store[1].live_task() is None
    assert isinstance(next_rows[7].status, oracle.StatusBlocked)
    assert lease is None


def test_human_resume_restores_retry_task() -> None:
    snapshot = _snapshot(10)
    row = oracle.CIRow(
        ci_snapshot=snapshot,
        ci_phase=oracle.CIPaused(),
        ci_task=7,
        ci_attempts=1,
    )
    task_rows = {
        7: oracle.TaskRow(
            title="ci / test",
            description="",
            kind=oracle.TaskCI(),
            status=oracle.StatusBlocked(),
            source_comment=None,
        )
    }

    next_store, next_rows = oracle.resume_ci_after_human(1, {1: row}, task_rows)

    assert isinstance(next_store[1].ci_phase, oracle.CIFailing)
    assert next_store[1].ci_task == 7
    assert next_store[1].ci_attempts == 1
    assert next_store[1].live_task() == 7
    assert isinstance(next_rows[7].status, oracle.StatusPending)


def test_resolution_completes_live_task_and_clears_lease() -> None:
    snapshot = _snapshot(10)
    row = snapshot.new_live_row(7)
    task_rows = {7: snapshot.task_row()}

    next_store, next_rows, lease = _unpack_resolved(
        oracle.record_ci_resolved(1, {1: row}, task_rows, 7)
    )

    assert isinstance(next_store[1].ci_phase, oracle.CIResolved)
    assert next_store[1].ci_task is None
    assert next_store[1].ci_attempts == 0
    assert lease is None
    assert isinstance(next_rows[7].status, oracle.StatusCompleted)
