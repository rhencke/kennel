"""Tests for fido.nudges — Nudges collaborator."""

from fido.nudges import Nudges
from fido.rocq.commit_result import CommitHookFailure


class TestNudgesContextHeader:
    """The shared context header is embedded in every nudge prompt."""

    def test_missing_sentinel_includes_header(self) -> None:
        nudges = Nudges()
        result = nudges.missing_sentinel(
            task_title="Fix bug",
            task_id="t1",
            work_dir="/repo",
            pr_number=42,
            parse_error="no sentinel found",
        )
        assert "Task: Fix bug (id: t1)" in result
        assert "PR: 42" in result
        assert "Work dir: /repo" in result

    def test_nothing_staged_includes_header(self) -> None:
        nudges = Nudges()
        result = nudges.nothing_staged(
            task_title="Add tests",
            task_id="t2",
            work_dir="/repo",
            pr_number=7,
        )
        assert "Task: Add tests (id: t2)" in result
        assert "PR: 7" in result
        assert "Work dir: /repo" in result

    def test_hook_failure_includes_header(self) -> None:
        nudges = Nudges()
        result = nudges.hook_failure(
            task_title="Fix lint",
            task_id="t3",
            work_dir="/repo",
            pr_number=99,
            failure=CommitHookFailure(output="ruff: 2 errors"),
        )
        assert "Task: Fix lint (id: t3)" in result
        assert "PR: 99" in result
        assert "Work dir: /repo" in result


class TestNudgesMissingSentinel:
    def test_contains_parse_error(self) -> None:
        nudges = Nudges()
        result = nudges.missing_sentinel(
            task_title="x",
            task_id="t1",
            work_dir="/r",
            pr_number=1,
            parse_error="expected JSON on last line",
        )
        assert "expected JSON on last line" in result

    def test_contains_turn_outcome_instruction(self) -> None:
        nudges = Nudges()
        result = nudges.missing_sentinel(
            task_title="x",
            task_id="t1",
            work_dir="/r",
            pr_number=1,
            parse_error="bad",
        )
        assert "turn_outcome" in result
        assert "commit-task-complete" in result
        assert "skip-task-with-reason" in result
        assert "stuck-on-task" in result


class TestNudgesNothingStaged:
    def test_contains_nothing_staged_instruction(self) -> None:
        nudges = Nudges()
        result = nudges.nothing_staged(
            task_title="x",
            task_id="t1",
            work_dir="/r",
            pr_number=1,
        )
        assert "nothing staged" in result.lower() or "nothing" in result.lower()
        assert "skip-task-with-reason" in result


class TestNudgesHookFailure:
    def test_contains_hook_output(self) -> None:
        nudges = Nudges()
        result = nudges.hook_failure(
            task_title="x",
            task_id="t1",
            work_dir="/r",
            pr_number=1,
            failure=CommitHookFailure(output="ruff found 3 errors"),
        )
        assert "ruff found 3 errors" in result

    def test_contains_fix_instruction(self) -> None:
        nudges = Nudges()
        result = nudges.hook_failure(
            task_title="x",
            task_id="t1",
            work_dir="/r",
            pr_number=1,
            failure=CommitHookFailure(output="x"),
        )
        assert "pre-commit hook rejected" in result.lower()
        assert "fix" in result.lower()
        assert "turn_outcome" in result

    def test_header_not_duplicated(self) -> None:
        """hook_failure must compose header+body internally — no double header."""
        nudges = Nudges()
        result = nudges.hook_failure(
            task_title="Fix lint",
            task_id="t1",
            work_dir="/repo",
            pr_number=5,
            failure=CommitHookFailure(output="err"),
        )
        assert result.count("Task: Fix lint") == 1


class TestNudgesForKind:
    """for_kind dispatches correctly to each method."""

    def test_dispatch_missing_sentinel(self) -> None:
        from fido.rocq import nudge_kind as nudge_oracle

        nudges = Nudges()
        result = nudges.for_kind(
            nudge_oracle.NudgeMissingSentinel(),
            "title",
            "t1",
            "/r",
            1,
            parse_error="oops",
        )
        assert "oops" in result
        assert "turn_outcome" in result

    def test_dispatch_nothing_staged(self) -> None:
        from fido.rocq import nudge_kind as nudge_oracle

        nudges = Nudges()
        result = nudges.for_kind(
            nudge_oracle.NudgeNothingStaged(),
            "title",
            "t1",
            "/r",
            1,
        )
        assert "skip-task-with-reason" in result

    def test_dispatch_hook_failure(self) -> None:
        from fido.rocq import nudge_kind as nudge_oracle

        nudges = Nudges()
        result = nudges.for_kind(
            nudge_oracle.NudgeHookFailure(),
            "title",
            "t1",
            "/r",
            1,
            failure=CommitHookFailure(output="hook err"),
        )
        assert "hook err" in result

    def test_dispatch_hook_failure_asserts_failure_present(self) -> None:
        import pytest

        from fido.rocq import nudge_kind as nudge_oracle

        nudges = Nudges()
        with pytest.raises(AssertionError, match="hook_failure nudge requires"):
            nudges.for_kind(
                nudge_oracle.NudgeHookFailure(),
                "title",
                "t1",
                "/r",
                1,
            )
