"""Tests for fido.turn_outcome — sentinel parser."""

from unittest.mock import patch

import pytest

import fido.turn_outcome as _to_mod
from fido.rocq.turn_outcome import (
    CommitTaskComplete,
    CommitTaskInProgress,
    SkipTaskWithReason,
    StuckOnTask,
)
from fido.turn_outcome import parse_turn_outcome


class TestParseTurnOutcomeEmpty:
    def test_empty_string(self) -> None:
        with pytest.raises(ValueError, match="No non-empty lines"):
            parse_turn_outcome("")

    def test_whitespace_only(self) -> None:
        with pytest.raises(ValueError, match="No non-empty lines"):
            parse_turn_outcome("   \n  \n  ")


class TestParseTurnOutcomeInvalidJson:
    def test_not_json(self) -> None:
        with pytest.raises(ValueError, match="not valid JSON"):
            parse_turn_outcome("Done, all tests pass!")

    def test_json_array(self) -> None:
        with pytest.raises(ValueError, match="not an object"):
            parse_turn_outcome('["commit-task-complete"]')


class TestParseTurnOutcomeUnrecognized:
    def test_no_turn_outcome_key(self) -> None:
        with pytest.raises(ValueError, match='no "turn_outcome" key'):
            parse_turn_outcome('{"status": "done"}')

    def test_unknown_turn_outcome_value(self) -> None:
        with pytest.raises(ValueError, match="Unrecognised turn_outcome"):
            parse_turn_outcome('{"turn_outcome": "do-the-thing"}')


class TestParseTurnOutcomeCommitTaskComplete:
    def test_valid(self) -> None:
        line = '{"turn_outcome": "commit-task-complete", "summary": "Add foo"}'
        assert parse_turn_outcome(line) == CommitTaskComplete(summary="Add foo")

    def test_missing_summary(self) -> None:
        with pytest.raises(ValueError, match="non-empty.*summary"):
            parse_turn_outcome('{"turn_outcome": "commit-task-complete"}')

    def test_summary_not_string(self) -> None:
        line = '{"turn_outcome": "commit-task-complete", "summary": 42}'
        with pytest.raises(ValueError, match="non-empty.*summary"):
            parse_turn_outcome(line)

    def test_summary_empty(self) -> None:
        line = '{"turn_outcome": "commit-task-complete", "summary": ""}'
        with pytest.raises(ValueError, match="non-empty.*summary"):
            parse_turn_outcome(line)

    def test_summary_whitespace_only(self) -> None:
        line = '{"turn_outcome": "commit-task-complete", "summary": "   "}'
        with pytest.raises(ValueError, match="non-empty.*summary"):
            parse_turn_outcome(line)


class TestParseTurnOutcomeCommitTaskInProgress:
    def test_valid(self) -> None:
        line = '{"turn_outcome": "commit-task-in-progress", "summary": "WIP: Add bar"}'
        assert parse_turn_outcome(line) == CommitTaskInProgress(summary="WIP: Add bar")

    def test_missing_summary(self) -> None:
        with pytest.raises(ValueError, match="non-empty.*summary"):
            parse_turn_outcome('{"turn_outcome": "commit-task-in-progress"}')

    def test_summary_not_string(self) -> None:
        line = '{"turn_outcome": "commit-task-in-progress", "summary": null}'
        with pytest.raises(ValueError, match="non-empty.*summary"):
            parse_turn_outcome(line)

    def test_summary_empty(self) -> None:
        line = '{"turn_outcome": "commit-task-in-progress", "summary": ""}'
        with pytest.raises(ValueError, match="non-empty.*summary"):
            parse_turn_outcome(line)

    def test_summary_whitespace_only(self) -> None:
        line = '{"turn_outcome": "commit-task-in-progress", "summary": "\\t\\n  "}'
        with pytest.raises(ValueError, match="non-empty.*summary"):
            parse_turn_outcome(line)


class TestParseTurnOutcomeSkipTaskWithReason:
    def test_valid(self) -> None:
        line = '{"turn_outcome": "skip-task-with-reason", "reason": "already done in abc1234"}'
        assert parse_turn_outcome(line) == SkipTaskWithReason(
            reason="already done in abc1234"
        )

    def test_missing_reason(self) -> None:
        with pytest.raises(ValueError, match="non-empty.*reason"):
            parse_turn_outcome('{"turn_outcome": "skip-task-with-reason"}')

    def test_reason_not_string(self) -> None:
        line = '{"turn_outcome": "skip-task-with-reason", "reason": false}'
        with pytest.raises(ValueError, match="non-empty.*reason"):
            parse_turn_outcome(line)

    def test_reason_empty(self) -> None:
        line = '{"turn_outcome": "skip-task-with-reason", "reason": ""}'
        with pytest.raises(ValueError, match="non-empty.*reason"):
            parse_turn_outcome(line)

    def test_reason_whitespace_only(self) -> None:
        line = '{"turn_outcome": "skip-task-with-reason", "reason": "  \\n "}'
        with pytest.raises(ValueError, match="non-empty.*reason"):
            parse_turn_outcome(line)


class TestParseTurnOutcomeStuckOnTask:
    def test_valid(self) -> None:
        line = '{"turn_outcome": "stuck-on-task", "reason": "need API credentials"}'
        assert parse_turn_outcome(line) == StuckOnTask(reason="need API credentials")

    def test_missing_reason(self) -> None:
        with pytest.raises(ValueError, match="non-empty.*reason"):
            parse_turn_outcome('{"turn_outcome": "stuck-on-task"}')

    def test_reason_not_string(self) -> None:
        line = '{"turn_outcome": "stuck-on-task", "reason": 123}'
        with pytest.raises(ValueError, match="non-empty.*reason"):
            parse_turn_outcome(line)

    def test_reason_empty(self) -> None:
        line = '{"turn_outcome": "stuck-on-task", "reason": ""}'
        with pytest.raises(ValueError, match="non-empty.*reason"):
            parse_turn_outcome(line)

    def test_reason_whitespace_only(self) -> None:
        line = '{"turn_outcome": "stuck-on-task", "reason": " "}'
        with pytest.raises(ValueError, match="non-empty.*reason"):
            parse_turn_outcome(line)


class TestParseTurnOutcomeMultiLine:
    def test_sentinel_on_last_line(self) -> None:
        text = (
            "I've implemented the change.\n"
            "All tests pass.\n"
            '{"turn_outcome": "commit-task-complete", "summary": "Implement thing"}'
        )
        assert parse_turn_outcome(text) == CommitTaskComplete(summary="Implement thing")

    def test_stale_sentinel_in_middle_non_json_at_end(self) -> None:
        """A valid-looking sentinel buried in the middle is invisible — only the
        last non-empty line matters, and here that line is plain text."""
        text = (
            '{"turn_outcome": "commit-task-complete", "summary": "stale"}\n'
            "Some more text after"
        )
        with pytest.raises(ValueError, match="not valid JSON"):
            parse_turn_outcome(text)

    def test_trailing_blank_lines_ignored(self) -> None:
        """Trailing blank lines are filtered out; the sentinel still parses."""
        text = '{"turn_outcome": "skip-task-with-reason", "reason": "no-op"}\n\n\n  \n'
        assert parse_turn_outcome(text) == SkipTaskWithReason(reason="no-op")

    def test_in_progress_on_last_line(self) -> None:
        text = (
            "Staged the first batch of changes.\n"
            '{"turn_outcome": "commit-task-in-progress", "summary": "wip: part 1 of 3"}'
        )
        assert parse_turn_outcome(text) == CommitTaskInProgress(
            summary="wip: part 1 of 3"
        )


class TestOracleDivergenceDetection:
    """Verify that oracle divergences crash loudly rather than silently passing."""

    def test_parse_oracle_returns_none_raises(self) -> None:
        """If parse_sentinel returns None for a valid kind/payload, AssertionError fires."""

        def bad_sentinel(_kind: str, _payload: str) -> None:
            return None

        with patch.object(_to_mod, "parse_sentinel", bad_sentinel):
            with pytest.raises(AssertionError, match="oracle returned None"):
                parse_turn_outcome(
                    '{"turn_outcome": "commit-task-complete", "summary": "Add foo"}'
                )

    def test_parse_oracle_mismatch_raises(self) -> None:
        """If parse_sentinel returns a different outcome than the parser built, AssertionError fires."""

        def bad_sentinel(_kind: str, _payload: str) -> CommitTaskInProgress:
            # Oracle disagrees: claims the outcome is in-progress, not complete
            return CommitTaskInProgress(summary="wrong")

        with patch.object(_to_mod, "parse_sentinel", bad_sentinel):
            with pytest.raises(AssertionError, match="oracle mismatch"):
                parse_turn_outcome(
                    '{"turn_outcome": "commit-task-complete", "summary": "Add foo"}'
                )

    def test_reject_oracle_accepts_unknown_kind_raises(self) -> None:
        """If parse_sentinel accepts an input the parser rejects (unknown kind), AssertionError fires."""

        def bad_sentinel(_kind: str, _payload: str) -> CommitTaskComplete:
            # Oracle wrongly accepts an unrecognised kind
            return CommitTaskComplete(summary="should not parse")

        with patch.object(_to_mod, "parse_sentinel", bad_sentinel):
            with pytest.raises(
                AssertionError, match="oracle accepted input the parser rejects"
            ):
                parse_turn_outcome('{"turn_outcome": "do-the-thing"}')

    def test_reject_oracle_accepts_empty_payload_raises(self) -> None:
        """If parse_sentinel accepts an empty-payload input the parser rejects, AssertionError fires."""

        def bad_sentinel(_kind: str, _payload: str) -> CommitTaskComplete:
            # Oracle wrongly accepts an empty summary
            return CommitTaskComplete(summary="")

        with patch.object(_to_mod, "parse_sentinel", bad_sentinel):
            with pytest.raises(
                AssertionError, match="oracle accepted input the parser rejects"
            ):
                parse_turn_outcome(
                    '{"turn_outcome": "commit-task-complete", "summary": ""}'
                )
