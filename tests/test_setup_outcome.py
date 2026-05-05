"""Tests for fido.setup_outcome — setup-phase sentinel parser."""

import pytest

from fido.setup_outcome import (
    NoTasksNeeded,
    PlannedTask,
    TasksPlanned,
    parse_setup_outcome,
)


class TestParseSetupOutcomeEmpty:
    def test_empty_string(self) -> None:
        with pytest.raises(ValueError, match="No non-empty lines"):
            parse_setup_outcome("")

    def test_whitespace_only(self) -> None:
        with pytest.raises(ValueError, match="No non-empty lines"):
            parse_setup_outcome("   \n  \n  ")


class TestParseSetupOutcomeInvalidJson:
    def test_not_json(self) -> None:
        with pytest.raises(ValueError, match="not valid JSON"):
            parse_setup_outcome("Done planning, see above for tasks.")

    def test_json_array(self) -> None:
        with pytest.raises(ValueError, match="not an object"):
            parse_setup_outcome('[{"title": "x"}]')

    def test_json_string(self) -> None:
        with pytest.raises(ValueError, match="not an object"):
            parse_setup_outcome('"tasks-planned"')


class TestParseSetupOutcomeUnrecognized:
    def test_no_setup_outcome_key(self) -> None:
        with pytest.raises(ValueError, match='no "setup_outcome" key'):
            parse_setup_outcome('{"status": "done"}')

    def test_unknown_setup_outcome_value(self) -> None:
        with pytest.raises(ValueError, match="Unrecognised setup_outcome"):
            parse_setup_outcome('{"setup_outcome": "do-the-thing"}')


class TestParseSetupOutcomeTasksPlanned:
    def test_single_task_title_only(self) -> None:
        line = '{"setup_outcome": "tasks-planned", "tasks": [{"title": "Add foo"}]}'
        result = parse_setup_outcome(line)
        assert result == TasksPlanned(
            tasks=(PlannedTask(title="Add foo", description=""),),
            pr_description="",
        )

    def test_multiple_tasks_with_descriptions(self) -> None:
        line = (
            '{"setup_outcome": "tasks-planned", "tasks": ['
            '{"title": "First"}, '
            '{"title": "Second", "description": "details"}'
            "]}"
        )
        result = parse_setup_outcome(line)
        assert result == TasksPlanned(
            tasks=(
                PlannedTask(title="First", description=""),
                PlannedTask(title="Second", description="details"),
            ),
            pr_description="",
        )

    def test_title_strips_whitespace(self) -> None:
        line = '{"setup_outcome": "tasks-planned", "tasks": [{"title": "  Trim me  "}]}'
        result = parse_setup_outcome(line)
        assert isinstance(result, TasksPlanned)
        assert result.tasks[0].title == "Trim me"

    def test_empty_tasks_array_rejected(self) -> None:
        line = '{"setup_outcome": "tasks-planned", "tasks": []}'
        with pytest.raises(ValueError, match="at least one task"):
            parse_setup_outcome(line)

    def test_tasks_not_array(self) -> None:
        line = '{"setup_outcome": "tasks-planned", "tasks": "nope"}'
        with pytest.raises(ValueError, match="must be a JSON array"):
            parse_setup_outcome(line)

    def test_tasks_missing(self) -> None:
        line = '{"setup_outcome": "tasks-planned"}'
        with pytest.raises(ValueError, match="must be a JSON array"):
            parse_setup_outcome(line)

    def test_task_not_object(self) -> None:
        line = '{"setup_outcome": "tasks-planned", "tasks": ["just a string"]}'
        with pytest.raises(ValueError, match=r"tasks\[0\] is not a JSON object"):
            parse_setup_outcome(line)

    def test_task_missing_title(self) -> None:
        line = '{"setup_outcome": "tasks-planned", "tasks": [{}]}'
        with pytest.raises(ValueError, match=r'tasks\[0\].*"title"'):
            parse_setup_outcome(line)

    def test_task_title_empty(self) -> None:
        line = '{"setup_outcome": "tasks-planned", "tasks": [{"title": ""}]}'
        with pytest.raises(ValueError, match=r'tasks\[0\].*"title"'):
            parse_setup_outcome(line)

    def test_task_title_whitespace_only(self) -> None:
        line = '{"setup_outcome": "tasks-planned", "tasks": [{"title": "   "}]}'
        with pytest.raises(ValueError, match=r'tasks\[0\].*"title"'):
            parse_setup_outcome(line)

    def test_task_title_not_string(self) -> None:
        line = '{"setup_outcome": "tasks-planned", "tasks": [{"title": 42}]}'
        with pytest.raises(ValueError, match=r'tasks\[0\].*"title"'):
            parse_setup_outcome(line)

    def test_task_description_not_string(self) -> None:
        line = (
            '{"setup_outcome": "tasks-planned", "tasks": '
            '[{"title": "x", "description": 99}]}'
        )
        with pytest.raises(ValueError, match='"description" must be a string'):
            parse_setup_outcome(line)

    def test_second_task_invalid(self) -> None:
        line = (
            '{"setup_outcome": "tasks-planned", "tasks": ['
            '{"title": "ok"}, {"foo": "no title"}'
            "]}"
        )
        with pytest.raises(ValueError, match=r'tasks\[1\].*"title"'):
            parse_setup_outcome(line)

    def test_pr_description_captured(self) -> None:
        line = (
            '{"setup_outcome": "tasks-planned", '
            '"pr_description": "## Summary\\n\\n- thing\\n\\nFixes #42.", '
            '"tasks": [{"title": "x"}]}'
        )
        result = parse_setup_outcome(line)
        assert isinstance(result, TasksPlanned)
        assert result.pr_description == "## Summary\n\n- thing\n\nFixes #42."

    def test_pr_description_must_be_string(self) -> None:
        line = (
            '{"setup_outcome": "tasks-planned", '
            '"pr_description": 42, '
            '"tasks": [{"title": "x"}]}'
        )
        with pytest.raises(ValueError, match='"pr_description" must be a string'):
            parse_setup_outcome(line)

    def test_pr_description_null_treated_as_absent(self) -> None:
        line = (
            '{"setup_outcome": "tasks-planned", '
            '"pr_description": null, '
            '"tasks": [{"title": "x"}]}'
        )
        result = parse_setup_outcome(line)
        assert isinstance(result, TasksPlanned)
        assert result.pr_description == ""


class TestParseSetupOutcomeNoTasksNeeded:
    def test_valid(self) -> None:
        line = (
            '{"setup_outcome": "no-tasks-needed", '
            '"reason": "Already covered by abc1234"}'
        )
        result = parse_setup_outcome(line)
        assert result == NoTasksNeeded(
            reason="Already covered by abc1234", pr_description=""
        )

    def test_with_pr_description(self) -> None:
        line = (
            '{"setup_outcome": "no-tasks-needed", '
            '"reason": "no-op", '
            '"pr_description": "## Why\\n\\nNothing to do here.\\n\\nFixes #1."}'
        )
        result = parse_setup_outcome(line)
        assert result == NoTasksNeeded(
            reason="no-op",
            pr_description="## Why\n\nNothing to do here.\n\nFixes #1.",
        )

    def test_missing_reason(self) -> None:
        line = '{"setup_outcome": "no-tasks-needed"}'
        with pytest.raises(ValueError, match="non-empty.*reason"):
            parse_setup_outcome(line)

    def test_reason_empty(self) -> None:
        line = '{"setup_outcome": "no-tasks-needed", "reason": ""}'
        with pytest.raises(ValueError, match="non-empty.*reason"):
            parse_setup_outcome(line)

    def test_reason_whitespace_only(self) -> None:
        line = '{"setup_outcome": "no-tasks-needed", "reason": "   "}'
        with pytest.raises(ValueError, match="non-empty.*reason"):
            parse_setup_outcome(line)

    def test_reason_not_string(self) -> None:
        line = '{"setup_outcome": "no-tasks-needed", "reason": 42}'
        with pytest.raises(ValueError, match="non-empty.*reason"):
            parse_setup_outcome(line)


class TestParseSetupOutcomeMultiLine:
    def test_sentinel_on_last_line(self) -> None:
        text = (
            "I've reviewed the issue.\n"
            "Three tasks should cover it.\n"
            '{"setup_outcome": "tasks-planned", "tasks": [{"title": "A"}]}'
        )
        result = parse_setup_outcome(text)
        assert isinstance(result, TasksPlanned)
        assert result.tasks[0].title == "A"

    def test_stale_sentinel_in_middle_non_json_at_end(self) -> None:
        """An earlier-line sentinel is invisible — only the last non-empty
        line counts."""
        text = (
            '{"setup_outcome": "tasks-planned", "tasks": [{"title": "stale"}]}\n'
            "Some narration after"
        )
        with pytest.raises(ValueError, match="not valid JSON"):
            parse_setup_outcome(text)

    def test_trailing_blank_lines_ignored(self) -> None:
        text = '{"setup_outcome": "no-tasks-needed", "reason": "no-op"}\n\n\n  \n'
        result = parse_setup_outcome(text)
        assert result == NoTasksNeeded(reason="no-op", pr_description="")
