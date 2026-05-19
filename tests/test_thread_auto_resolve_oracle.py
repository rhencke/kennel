from pathlib import Path

from fido.rocq import thread_auto_resolve as oracle

_PENDING: oracle.TaskStatus = oracle.StatusPending()


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
    status: oracle.TaskStatus = _PENDING,
) -> oracle.ThreadTask:
    return oracle.ThreadTask(
        thread_task_comment=comment_id,
        thread_task_status=status,
    )


def test_resolution_requires_fido_last_comment() -> None:
    thread = _thread(
        [
            _comment(1, oracle.CommentByActionable()),
            _comment(2, oracle.CommentByFido()),
            _comment(3, oracle.CommentByActionable()),
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
            _comment(10, oracle.CommentByActionable()),
            _comment(11, oracle.CommentByActionable()),
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


def test_resolution_ignores_outside_commenters() -> None:
    thread = _thread(
        [
            _comment(13, oracle.CommentByActionable()),
            _comment(14, oracle.CommentByFido()),
            _comment(15, oracle.CommentIgnored()),
        ],
    )

    assert oracle.thread_comment_ids(thread.review_thread_comments) == [13, 14, 15]
    assert oracle.modeled_thread_comment_ids(thread.review_thread_comments) == [
        13,
        14,
    ]
    assert oracle.should_resolve_thread(thread, [])
    assert oracle.should_resolve_thread(thread, [_task(15)])
    assert isinstance(
        oracle.resolution_decision(thread, []),
        oracle.ResolveReviewThread,
    )


def test_bot_comment_after_fido_blocks_resolution() -> None:
    thread = _thread(
        [
            _comment(16, oracle.CommentByActionable()),
            _comment(17, oracle.CommentByFido()),
            _comment(18, oracle.CommentByBot()),
        ],
    )

    assert oracle.modeled_thread_comment_ids(thread.review_thread_comments) == [
        16,
        17,
        18,
    ]
    assert not oracle.should_resolve_thread(thread, [])
    assert isinstance(
        oracle.resolution_decision(thread, []),
        oracle.KeepReviewThreadOpen,
    )


def test_resolution_ignores_pending_work_outside_thread() -> None:
    thread = _thread(
        [
            _comment(20, oracle.CommentByActionable()),
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
            _comment(30, oracle.CommentByActionable()),
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
            _comment(40, oracle.CommentByActionable()),
            _comment(41, oracle.CommentByFido()),
        ],
        resolved=True,
    )

    assert not oracle.should_resolve_thread(thread, [])
    assert isinstance(
        oracle.resolution_decision(thread, []),
        oracle.KeepReviewThreadOpen,
    )


def test_resolved_thread_latest_queueable_comment_queues_work() -> None:
    thread = _thread(
        [
            _comment(50, oracle.CommentByActionable()),
            _comment(51, oracle.CommentByFido()),
            _comment(52, oracle.CommentByActionable()),
        ],
        resolved=True,
    )

    assert oracle.latest_queueable_comment(thread.review_thread_comments) == 52
    assert isinstance(
        oracle.resolved_thread_queue_decision(thread, 52),
        oracle.QueueThreadTask,
    )


def test_resolved_thread_stale_queueable_comment_is_dismissed() -> None:
    thread = _thread(
        [
            _comment(60, oracle.CommentByActionable()),
            _comment(61, oracle.CommentByFido()),
            _comment(62, oracle.CommentByActionable()),
        ],
        resolved=True,
    )

    assert isinstance(
        oracle.resolved_thread_queue_decision(thread, 60),
        oracle.DismissStaleResolvedThread,
    )


def test_resolved_thread_ignored_comment_is_dismissed() -> None:
    thread = _thread(
        [
            _comment(63, oracle.CommentByActionable()),
            _comment(64, oracle.CommentByFido()),
            _comment(65, oracle.CommentIgnored()),
        ],
        resolved=True,
    )

    assert oracle.latest_queueable_comment(thread.review_thread_comments) == 63
    assert isinstance(
        oracle.resolved_thread_queue_decision(thread, 65),
        oracle.DismissStaleResolvedThread,
    )


def test_resolved_thread_latest_bot_comment_queues_work() -> None:
    thread = _thread(
        [
            _comment(66, oracle.CommentByActionable()),
            _comment(67, oracle.CommentByFido()),
            _comment(68, oracle.CommentByBot()),
        ],
        resolved=True,
    )

    assert oracle.latest_queueable_comment(thread.review_thread_comments) == 68
    assert isinstance(
        oracle.resolved_thread_queue_decision(thread, 68),
        oracle.QueueThreadTask,
    )


def test_bot_only_thread_resolves_when_no_pending() -> None:
    # #1856: a bot-only thread (bot → fido) resolves once Fido is last
    # and no pending task references the thread.  Bots are treated as
    # actionable for resolution purposes — every bot comment becomes a
    # DO or DUMP that Fido has already acted on by the time the resolve
    # sweep runs.
    thread = _thread(
        [
            _comment(5, oracle.CommentByBot()),
            _comment(6, oracle.CommentByFido()),
        ],
    )

    assert isinstance(
        oracle.resolution_decision(thread, []),
        oracle.ResolveReviewThread,
    )


def test_bot_only_thread_blocked_by_pending_task() -> None:
    # The DO half of #1856: while the derived task is still pending,
    # the thread stays open — followup_done gates the close just as it
    # does for human-driven threads.
    thread = _thread(
        [
            _comment(5, oracle.CommentByBot()),
            _comment(6, oracle.CommentByFido()),
        ],
    )
    pending = oracle.ThreadTask(
        thread_task_comment=5,
        thread_task_status=oracle.StatusPending(),
    )

    assert isinstance(
        oracle.resolution_decision(thread, [pending]),
        oracle.KeepReviewThreadOpen,
    )


def test_bot_only_thread_resolves_after_completed_task() -> None:
    # The DO-completed half of #1856: once the derived task completes,
    # the thread resolves.
    thread = _thread(
        [
            _comment(5, oracle.CommentByBot()),
            _comment(6, oracle.CommentByFido()),
        ],
    )
    completed = oracle.ThreadTask(
        thread_task_comment=5,
        thread_task_status=oracle.StatusCompleted(),
    )

    assert isinstance(
        oracle.resolution_decision(thread, [completed]),
        oracle.ResolveReviewThread,
    )


def test_unresolved_thread_always_queues_comment_task() -> None:
    thread = _thread(
        [
            _comment(70, oracle.CommentByActionable()),
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
