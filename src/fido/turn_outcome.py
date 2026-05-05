"""Turn-outcome sentinel parser for the worker task protocol.

Every provider turn must end with a JSON sentinel on its final non-empty line.
The sentinel declares what the harness should do next: stage and commit the
working tree then mark the task completed, stage and commit but keep the task
pending for another turn, or skip the commit and record a reason.

The LLM declares intent; Python acts on it.  Git operations are never the
LLM's responsibility.

Type definitions (``CommitTaskComplete``, ``CommitTaskInProgress``,
``SkipTaskWithReason``, ``TurnOutcome``) live in the Rocq-extracted module
:mod:`fido.rocq.turn_outcome`.  Importers should get types directly from
that module.  This module owns only the parser boundary adapter.
"""

import json

from fido.rocq.turn_outcome import (
    CommitTaskComplete,
    CommitTaskInProgress,
    SkipTaskWithReason,
    StuckOnTask,
    TurnOutcome,
    parse_sentinel,
)

__all__ = [
    "parse_turn_outcome",
]


def _assert_parse_oracle(kind: str, payload: str, result: TurnOutcome) -> None:
    """Assert that the Rocq-proven parse_sentinel agrees with our dispatch."""
    oracle = parse_sentinel(kind, payload)
    if oracle is None:
        raise AssertionError(
            f"parse_sentinel oracle returned None for kind={kind!r} "
            f"payload={payload!r}, but parser produced {result!r}"
        )
    if result != oracle:
        raise AssertionError(
            f"parse_sentinel oracle mismatch: oracle={oracle!r}, actual={result!r}"
        )


def parse_turn_outcome(text: str) -> TurnOutcome:
    """Parse the turn_outcome sentinel from the last non-empty line of *text*.

    Only the final non-empty line is examined — stale sentinels earlier in
    the response are ignored.

    Raises:
        ValueError: If the line is absent, not valid JSON, not a recognised
            shape, or missing required fields.  The message describes *why*
            parsing failed so the nudge loop can relay it to the LLM.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError(
            "No non-empty lines in output — expected a turn_outcome JSON "
            "sentinel on the final line"
        )
    last = lines[-1].strip()
    try:
        obj = json.loads(last)
    except json.JSONDecodeError:
        raise ValueError(f"Last non-empty line is not valid JSON: {last!r}") from None
    if not isinstance(obj, dict):
        raise ValueError(f"Last line parsed as JSON but is not an object: {last!r}")
    kind = obj.get("turn_outcome")
    if kind == "commit-task-complete":
        summary = obj.get("summary")
        if not isinstance(summary, str) or not summary:
            raise ValueError(
                'turn_outcome "commit-task-complete" requires a non-empty '
                f'"summary" string, got: {obj.get("summary")!r}'
            )
        result = CommitTaskComplete(summary=summary)
        _assert_parse_oracle(kind, summary, result)
        return result
    if kind == "commit-task-in-progress":
        summary = obj.get("summary")
        if not isinstance(summary, str) or not summary:
            raise ValueError(
                'turn_outcome "commit-task-in-progress" requires a non-empty '
                f'"summary" string, got: {obj.get("summary")!r}'
            )
        result = CommitTaskInProgress(summary=summary)
        _assert_parse_oracle(kind, summary, result)
        return result
    if kind == "skip-task-with-reason":
        reason = obj.get("reason")
        if not isinstance(reason, str) or not reason:
            raise ValueError(
                'turn_outcome "skip-task-with-reason" requires a non-empty '
                f'"reason" string, got: {obj.get("reason")!r}'
            )
        result = SkipTaskWithReason(reason=reason)
        _assert_parse_oracle(kind, reason, result)
        return result
    if kind == "stuck-on-task":
        reason = obj.get("reason")
        if not isinstance(reason, str) or not reason:
            raise ValueError(
                'turn_outcome "stuck-on-task" requires a non-empty '
                f'"reason" string, got: {obj.get("reason")!r}'
            )
        result = StuckOnTask(reason=reason)
        _assert_parse_oracle(kind, reason, result)
        return result
    if kind is None:
        raise ValueError(f'JSON object has no "turn_outcome" key: {last!r}')
    raise ValueError(f"Unrecognised turn_outcome value: {kind!r}")
