"""Turn-outcome sentinel types and parser for the worker task protocol.

Every provider turn must end with a JSON sentinel on its final non-empty line.
The sentinel declares what the harness should do next: stage and commit the
working tree then mark the task completed, stage and commit but keep the task
pending for another turn, or skip the commit and record a reason.

The LLM declares intent; Python acts on it.  Git operations are never the
LLM's responsibility.
"""

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class CommitTaskComplete:
    """LLM declares: stage + commit, then mark the task completed.

    *summary* becomes the git commit message.
    """

    summary: str


@dataclass(frozen=True)
class CommitTaskInProgress:
    """LLM declares: stage + commit, but keep the task pending for another turn.

    *summary* becomes the git commit message.  Use this when the task spans
    multiple provider turns — the harness commits the partial work so progress
    is durable, then re-enters the task on the next iteration.
    """

    summary: str


@dataclass(frozen=True)
class SkipTaskWithReason:
    """LLM declares: do not commit; record *reason* instead.

    If the task is genuinely complete (already covered by a prior commit,
    consolidated into another task, or proved a no-op), the reason is
    sufficient grounding for completion.  Otherwise the task remains pending
    with the reason logged.
    """

    reason: str


TurnOutcome = CommitTaskComplete | CommitTaskInProgress | SkipTaskWithReason


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
