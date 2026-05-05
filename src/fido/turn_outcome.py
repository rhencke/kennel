"""Turn-outcome sentinel parser for the worker task protocol.

Every provider turn must end with a JSON sentinel on its final non-empty line.
The sentinel declares what the harness should do next: stage and commit the
working tree then mark the task completed, stage and commit but keep the task
pending for another turn, or skip the commit and record a reason.

The LLM declares intent; Python acts on it.  Git operations are never the
LLM's responsibility.

Optional ``insights`` and ``out_of_scope_asks`` arrays let the LLM declare
auxiliary issues for the harness to file on its behalf.  Both default to
empty arrays and may appear alongside any ``turn_outcome`` value.

Type definitions (``CommitTaskComplete``, ``CommitTaskInProgress``,
``SkipTaskWithReason``, ``TurnOutcome``) live in the Rocq-extracted module
:mod:`fido.rocq.turn_outcome`.  Importers should get types directly from
that module.  This module owns only the parser boundary adapter.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass

from fido.rocq.turn_outcome import (
    CommitTaskComplete,
    CommitTaskInProgress,
    SkipTaskWithReason,
    StuckOnTask,
    TurnOutcome,
    parse_sentinel,
)

__all__ = [
    "Insight",
    "OutOfScopeAsk",
    "TurnOutcomeBundle",
    "parse_turn_outcome",
]


@dataclass(frozen=True)
class Insight:
    """A bug-or-design observation the LLM declared in its turn output;
    filed by the harness as a GitHub issue with the ``Insight`` label.

    Mirrors :class:`fido.synthesis.Insight` but is the task-phase variant
    consumed via the turn_outcome sentinel."""

    title: str
    body: str


@dataclass(frozen=True)
class OutOfScopeAsk:
    """A request that arrived during task work but is out of scope for
    the current task; filed by the harness as a tracked GitHub issue so
    the work isn't lost but doesn't bloat the current PR."""

    title: str
    body: str


@dataclass(frozen=True)
class TurnOutcomeBundle:
    """Full result of parsing a turn_outcome sentinel: the dispatch outcome
    plus any auxiliary issues the LLM declared for harness-side filing."""

    outcome: TurnOutcome
    insights: tuple[Insight, ...]
    out_of_scope_asks: tuple[OutOfScopeAsk, ...]


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


def _parse_aux_issues(
    obj: dict[str, object], field: str, factory: Callable[[str, str], object]
) -> tuple[object, ...]:
    """Parse an optional auxiliary-issue array (``insights`` or
    ``out_of_scope_asks``) from *obj*.

    Returns an empty tuple when the field is absent or ``None``.  Raises
    ``ValueError`` when present but malformed (non-list, items missing
    required fields, etc.).
    """
    raw = obj.get(field)
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f'turn_outcome "{field}" must be a JSON array, got: {raw!r}')
    out: list[object] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"{field}[{i}] is not a JSON object: {item!r}")
        title = item.get("title")
        body = item.get("body")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f'{field}[{i}] requires a non-empty "title" string')
        if not isinstance(body, str) or not body.strip():
            raise ValueError(f'{field}[{i}] requires a non-empty "body" string')
        out.append(factory(title.strip(), body))
    return tuple(out)


def parse_turn_outcome(text: str) -> TurnOutcomeBundle:
    """Parse the turn_outcome sentinel from the last non-empty line of *text*.

    Only the final non-empty line is examined — stale sentinels earlier in
    the response are ignored.

    Returns a :class:`TurnOutcomeBundle` with the dispatch ``outcome`` plus
    any optional ``insights`` and ``out_of_scope_asks`` declared by the LLM.

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
            outcome: TurnOutcome = _parse_outcome_field(
                obj, kind, "summary", lambda s: CommitTaskComplete(summary=s)
            )
        case "commit-task-in-progress":
            outcome = _parse_outcome_field(
                obj, kind, "summary", lambda s: CommitTaskInProgress(summary=s)
            )
        case "skip-task-with-reason":
            outcome = _parse_outcome_field(
                obj, kind, "reason", lambda s: SkipTaskWithReason(reason=s)
            )
        case "stuck-on-task":
            outcome = _parse_outcome_field(
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
    insights = _parse_aux_issues(
        obj, "insights", lambda title, body: Insight(title=title, body=body)
    )
    asks = _parse_aux_issues(
        obj,
        "out_of_scope_asks",
        lambda title, body: OutOfScopeAsk(title=title, body=body),
    )
    # mypy/pyright: tuple element types are object[]; cast the homogeneous
    # tuples for the bundle type signature.
    return TurnOutcomeBundle(
        outcome=outcome,
        insights=tuple(i for i in insights if isinstance(i, Insight)),
        out_of_scope_asks=tuple(a for a in asks if isinstance(a, OutOfScopeAsk)),
    )
