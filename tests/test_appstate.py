"""Tests for fido.appstate — pure data shapes for FidoState."""

from fido.appstate import IssueSnapshot, make_issue_snapshot


class TestMakeIssueSnapshot:
    """make_issue_snapshot is the pure transform that the worker uses to
    publish issue/PR/task state into FidoState (#1690).  Every field the
    legacy disk-reading ``_collect_fido_state`` produced is computed
    here from the raw state dict and tasks list."""

    def test_empty_inputs_produce_default_snapshot(self) -> None:
        snapshot = make_issue_snapshot({}, [])
        assert snapshot == IssueSnapshot(
            issue=None,
            issue_title=None,
            issue_started_at=None,
            pr_number=None,
            pr_title=None,
            pending_task_count=0,
            completed_task_count=0,
            current_task=None,
            task_number=None,
            task_total=None,
        )

    def test_state_fields_passed_through(self) -> None:
        snapshot = make_issue_snapshot(
            {
                "issue": 7,
                "issue_title": "Fix it",
                "issue_started_at": "2026-04-19T12:00:00+00:00",
                "pr_number": 13,
                "pr_title": "Fix it (closes #7)",
            },
            [],
        )
        assert snapshot.issue == 7
        assert snapshot.issue_title == "Fix it"
        assert snapshot.issue_started_at == "2026-04-19T12:00:00+00:00"
        assert snapshot.pr_number == 13
        assert snapshot.pr_title == "Fix it (closes #7)"

    def test_pending_and_completed_counts(self) -> None:
        tasks = [
            {"status": "pending", "title": "a"},
            {"status": "pending", "title": "b"},
            {"status": "completed", "title": "c"},
            {"status": "in_progress", "title": "d"},
        ]
        snapshot = make_issue_snapshot({}, tasks)
        assert snapshot.pending_task_count == 2
        assert snapshot.completed_task_count == 1

    def test_current_task_prefers_in_progress(self) -> None:
        tasks = [
            {"status": "pending", "title": "first pending"},
            {"status": "in_progress", "title": "active"},
            {"status": "pending", "title": "later"},
        ]
        snapshot = make_issue_snapshot({}, tasks)
        assert snapshot.current_task == "active"

    def test_current_task_falls_back_to_first_pending(self) -> None:
        tasks = [
            {"status": "completed", "title": "done"},
            {"status": "pending", "title": "do this"},
            {"status": "pending", "title": "then that"},
        ]
        snapshot = make_issue_snapshot({}, tasks)
        assert snapshot.current_task == "do this"

    def test_task_number_reflects_in_progress_position_in_non_completed(self) -> None:
        tasks = [
            {"status": "completed", "title": "done"},
            {"status": "pending", "title": "first pending"},
            {"status": "in_progress", "title": "active"},
            {"status": "pending", "title": "later"},
        ]
        snapshot = make_issue_snapshot({}, tasks)
        assert snapshot.task_number == 2  # active is the 2nd non-completed
        assert snapshot.task_total == 3

    def test_task_number_defaults_to_one_when_no_in_progress(self) -> None:
        tasks = [
            {"status": "pending", "title": "a"},
            {"status": "pending", "title": "b"},
        ]
        snapshot = make_issue_snapshot({}, tasks)
        assert snapshot.task_number == 1
        assert snapshot.task_total == 2

    def test_task_number_and_total_are_none_when_all_completed(self) -> None:
        tasks = [
            {"status": "completed", "title": "a"},
            {"status": "completed", "title": "b"},
        ]
        snapshot = make_issue_snapshot({}, tasks)
        assert snapshot.task_number is None
        assert snapshot.task_total is None
