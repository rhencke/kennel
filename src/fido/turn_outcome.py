"""Turn-outcome sentinel parser for the worker task protocol.

Every provider turn must end with a JSON sentinel on its final non-empty line.
The sentinel declares what the harness should do next: stage and commit the
working tree then mark the task completed, stage and commit but keep the task
pending for another turn, or skip the commit and record a reason.

The LLM declares intent; Python acts on it.  Git operations are never the
LLM's responsibility.

Type definitions (``CommitTaskComplete``, ``CommitTaskInProgress``,
``SkipTaskWithReason``, ``TurnOutcome``) live in the Rocq-extracted module
:mod:`fido.rocq.turn_outcome` and are re-exported here so importers get a
single canonical source.  The parser (``parse_turn_outcome``) is a Python
boundary adapter that stays in this module.
"""

import json

from fido.rocq.turn_outcome import (
    CommitTaskComplete,
    CommitTaskInProgress,
    SkipTaskWithReason,
    TurnOutcome,
    TurnOutcomeT,
)

__all__ = [
    "CommitTaskComplete",
    "CommitTaskInProgress",
    "SkipTaskWithReason",
    "TurnOutcome",
    "TurnOutcomeT",
    "parse_turn_outcome",
]


def parse_turn_outcome(text: str) -> TurnOutcome | None:
    """Parse the turn_outcome sentinel from the last non-empty line of *text*.

    Only the final non-empty line is examined — stale sentinels earlier in
    the response are ignored.  Returns ``None`` if the line is absent, not
    valid JSON, not a recognised shape, or missing required fields.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    last = lines[-1].strip()
    try:
        obj = json.loads(last)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    kind = obj.get("turn_outcome")
    if kind == "commit-task-complete":
        summary = obj.get("summary")
        if not isinstance(summary, str) or not summary:
            return None
        return CommitTaskComplete(summary=summary)
    if kind == "commit-task-in-progress":
        summary = obj.get("summary")
        if not isinstance(summary, str) or not summary:
            return None
        return CommitTaskInProgress(summary=summary)
    if kind == "skip-task-with-reason":
        reason = obj.get("reason")
        if not isinstance(reason, str) or not reason:
            return None
        return SkipTaskWithReason(reason=reason)
    return None
