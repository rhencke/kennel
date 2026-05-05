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
from collections.abc import Callable

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


def _assert_reject_oracle(kind: str, payload: str) -> None:
    """Assert that the Rocq-proven parse_sentinel also rejects this input.

    Called on ValueError paths where the parser refuses to produce a
    TurnOutcome.  If the model accepts an input the parser rejects,
    this crashes — surfacing the divergence immediately.
    """
    oracle = parse_sentinel(kind, payload)
    if oracle is not None:
        raise AssertionError(
            f"parse_sentinel oracle accepted input the parser rejects: "
            f"kind={kind!r} payload={payload!r} => {oracle!r}"
        )


def _require_nonempty_str(obj: dict[str, object], field: str, kind: str) -> str:
    """Extract and validate a required non-empty string field from *obj*.

    Raises ValueError with a descriptive message if the field is missing,
    not a string, empty, or whitespace-only.
    """
    value = obj.get(field)
    if not isinstance(value, str) or not value.strip():
        # When the value IS a string (but empty/whitespace), assert the model
        # also rejects.  Python normalizes by stripping — the model rejects
        # empty payloads — so we pass the stripped form.
        if isinstance(value, str):
            _assert_reject_oracle(kind, value.strip())
        raise ValueError(
            f'turn_outcome "{kind}" requires a non-empty '
            f'"{field}" string, got: {value!r}'
        )
    return value


def _parse_outcome_field(
    obj: dict[str, object],
    kind: str,
    field: str,
    make: Callable[[str], TurnOutcome],
) -> TurnOutcome:
    """Validate *field* in *obj*, construct the outcome via *make*, assert the oracle."""
    payload = _require_nonempty_str(obj, field, kind)
    result = make(payload)
    _assert_parse_oracle(kind, payload, result)
    return result


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
    match kind:
        case "commit-task-complete":
            return _parse_outcome_field(
                obj, kind, "summary", lambda s: CommitTaskComplete(summary=s)
            )
        case "commit-task-in-progress":
            return _parse_outcome_field(
                obj, kind, "summary", lambda s: CommitTaskInProgress(summary=s)
            )
        case "skip-task-with-reason":
            return _parse_outcome_field(
                obj, kind, "reason", lambda s: SkipTaskWithReason(reason=s)
            )
        case "stuck-on-task":
            return _parse_outcome_field(
                obj, kind, "reason", lambda s: StuckOnTask(reason=s)
            )
        case None:
            raise ValueError(f'JSON object has no "turn_outcome" key: {last!r}')
        case _:
            # The kind is unrecognised — assert the model also rejects it.
            # Use a non-empty probe payload: the model returns None for any
            # unknown kind regardless of payload content.
            _assert_reject_oracle(kind, "x")
            raise ValueError(f"Unrecognised turn_outcome value: {kind!r}")
