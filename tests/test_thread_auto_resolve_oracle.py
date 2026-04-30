from pathlib import Path

from fido.rocq import thread_auto_resolve as oracle


def _comment(
    comment_id: int, author: oracle.ThreadCommentAuthor
) -> oracle.ThreadComment:
    return oracle.ThreadComment(
        thread_comment_id=comment_id,
        thread_comment_author=author,
    )


def _thread(
    comments: list[oracle.ThreadComment],
    *,
    resolved: bool = False,
) -> oracle.ReviewThread:
    return oracle.ReviewThread(
        review_thread_resolved=resolved,
        review_thread_comments=comments,
    )


def _task(
    comment_id: int,
    status: oracle.TaskStatus = oracle.StatusPending(),
) -> oracle.ThreadTask:
    return oracle.ThreadTask(
        thread_task_comment=comment_id,
        thread_task_status=status,
    )


def test_resolution_requires_fido_last_comment() -> None:
    thread = _thread(
        [
            _comment(1, oracle.CommentByHuman()),
            _comment(2, oracle.CommentByFido()),
            _comment(3, oracle.CommentByHuman()),
        ],
    )

    assert not oracle.should_resolve_thread(thread, [])
    assert isinstance(
        oracle.resolution_decision(thread, []),
        oracle.KeepReviewThreadOpen,
    )


def test_resolution_requires_no_pending_work_anywhere_on_thread() -> None:
    thread = _thread(
        [
            _comment(10, oracle.CommentByHuman()),
            _comment(11, oracle.CommentByHuman()),
            _comment(12, oracle.CommentByFido()),
        ],
    )

    assert oracle.thread_comment_ids(thread.review_thread_comments) == [10, 11, 12]
    assert oracle.has_pending_thread_task([10, 11, 12], [_task(11)])
    assert not oracle.should_resolve_thread(thread, [_task(11)])
    assert isinstance(
        oracle.resolution_decision(thread, [_task(11)]),
        oracle.KeepReviewThreadOpen,
    )


def test_resolution_ignores_pending_work_outside_thread() -> None:
    thread = _thread(
        [
            _comment(20, oracle.CommentByHuman()),
            _comment(21, oracle.CommentByFido()),
        ],
    )

    assert oracle.should_resolve_thread(thread, [_task(999)])
    assert isinstance(
        oracle.resolution_decision(thread, [_task(999)]),
        oracle.ResolveReviewThread,
    )


def test_completed_or_blocked_same_thread_work_allows_resolution() -> None:
    thread = _thread(
        [
            _comment(30, oracle.CommentByHuman()),
            _comment(31, oracle.CommentByFido()),
        ],
    )
    tasks = [
        _task(30, oracle.StatusCompleted()),
        _task(31, oracle.StatusBlocked()),
    ]

    assert not oracle.has_pending_thread_task([30, 31], tasks)
    assert oracle.should_resolve_thread(thread, tasks)


def test_already_resolved_thread_stays_open_for_oracle() -> None:
    thread = _thread(
        [
            _comment(40, oracle.CommentByHuman()),
            _comment(41, oracle.CommentByFido()),
        ],
        resolved=True,
    )

    assert not oracle.should_resolve_thread(thread, [])
    assert isinstance(
        oracle.resolution_decision(thread, []),
        oracle.KeepReviewThreadOpen,
    )


def test_resolved_thread_latest_human_comment_queues_work() -> None:
    thread = _thread(
        [
            _comment(50, oracle.CommentByHuman()),
            _comment(51, oracle.CommentByFido()),
            _comment(52, oracle.CommentByHuman()),
        ],
        resolved=True,
    )

    assert oracle.latest_human_comment(thread.review_thread_comments) == 52
    assert isinstance(
        oracle.resolved_thread_queue_decision(thread, 52),
        oracle.QueueThreadTask,
    )


def test_resolved_thread_stale_human_comment_is_dismissed() -> None:
    thread = _thread(
        [
            _comment(60, oracle.CommentByHuman()),
            _comment(61, oracle.CommentByFido()),
            _comment(62, oracle.CommentByHuman()),
        ],
        resolved=True,
    )

    assert isinstance(
        oracle.resolved_thread_queue_decision(thread, 60),
        oracle.DismissStaleResolvedThread,
    )


def test_unresolved_thread_always_queues_comment_task() -> None:
    thread = _thread(
        [
            _comment(70, oracle.CommentByHuman()),
            _comment(71, oracle.CommentByFido()),
        ],
    )

    assert isinstance(
        oracle.resolved_thread_queue_decision(thread, 70),
        oracle.QueueThreadTask,
    )


def test_membership_helper_lowers_to_native_membership() -> None:
    source = Path(oracle.__file__).read_text()

    assert "def positive_mem(" not in source
    assert "task.thread_task_comment in comment_ids" in source
