"""Setup-outcome sentinel parser for the worker setup phase.

The setup sub-agent (sub/setup.md) plans the task list AND drafts the PR
description in a single turn.  Under the post-#1363 protocol the LLM declares
intent via a JSON sentinel on the final non-empty line of its output; the
harness reads it and CRUDs the task list and PR body itself.
``Bash(./fido task *)`` is blocked at the permissions layer so the LLM cannot
write to the task store directly, and PR-body edits go through the harness's
``sync.lock``-guarded path.

Two outcomes:

* ``tasks-planned`` — the LLM proposes a list of new tasks AND the PR
  description body.  Each task entry must carry a non-empty ``title``;
  ``description`` is optional and defaults to empty.  ``pr_description`` is
  optional too (when omitted, the harness falls back to a separate LLM call
  via :func:`fido.worker._write_pr_description`).  The harness creates one
  ``spec``-type pending task per entry and writes the PR body directly.
* ``no-tasks-needed`` — the LLM judged that no further work is required on
  this branch.  The harness falls through to the "setup produced no tasks"
  finalize path (post a Fido-voice comment, mark ready, request review).
  ``pr_description`` is also accepted here in case the LLM wants to seed the
  PR body before the no-tasks finalize path runs.

The LLM declares intent; Python acts on it.  Task creation and PR-body edits
are never the LLM's responsibility under this protocol.
"""

import json
from dataclasses import dataclass

__all__ = [
    "PlannedTask",
    "SetupOutcome",
    "TasksPlanned",
    "NoTasksNeeded",
    "parse_setup_outcome",
]


@dataclass(frozen=True)
class PlannedTask:
    title: str
    description: str = ""


@dataclass(frozen=True)
class TasksPlanned:
    tasks: tuple[PlannedTask, ...]
    pr_description: str = ""


@dataclass(frozen=True)
class NoTasksNeeded:
    reason: str
    pr_description: str = ""


SetupOutcome = TasksPlanned | NoTasksNeeded


def _last_json_object(text: str) -> dict[str, object]:
    """Decode the last non-empty line of *text* as a JSON object.

    Raises ValueError if the line is absent, not valid JSON, or not an object.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError(
            "No non-empty lines in output — expected a setup_outcome JSON "
            "sentinel on the final line"
        )
    last = lines[-1].strip()
    try:
        obj = json.loads(last)
    except json.JSONDecodeError:
        raise ValueError(f"Last non-empty line is not valid JSON: {last!r}") from None
    if not isinstance(obj, dict):
        raise ValueError(f"Last line parsed as JSON but is not an object: {last!r}")
    return obj


def _require_str(obj: dict[str, object], field: str, context: str) -> str:
    """Extract a required non-empty string field from *obj*."""
    value = obj.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f'{context} requires a non-empty "{field}" string, got: {value!r}'
        )
    return value.strip()


def _parse_tasks(raw: object) -> tuple[PlannedTask, ...]:
    """Validate the ``tasks`` array and convert each entry."""
    if not isinstance(raw, list):
        raise ValueError(f'"tasks" must be a JSON array, got: {raw!r}')
    out: list[PlannedTask] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"tasks[{i}] is not a JSON object: {item!r}")
        title = _require_str(item, "title", f"tasks[{i}]")
        desc_raw = item.get("description", "")
        if not isinstance(desc_raw, str):
            raise ValueError(
                f'tasks[{i}] "description" must be a string when present, '
                f"got: {desc_raw!r}"
            )
        out.append(PlannedTask(title=title, description=desc_raw))
    return tuple(out)


def _optional_str(obj: dict[str, object], field: str, context: str) -> str:
    """Extract an optional string field; absent or null → ``""``.

    A non-string value (int, list, etc.) is rejected with ValueError so a
    typo'd field type doesn't silently degrade to empty.
    """
    raw = obj.get(field)
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raise ValueError(
            f'{context} "{field}" must be a string when present, got: {raw!r}'
        )
    return raw


def parse_setup_outcome(text: str) -> SetupOutcome:
    """Parse the setup_outcome sentinel from the last non-empty line of *text*.

    Recognised shapes:

    * ``{"setup_outcome": "tasks-planned", "tasks": [{"title": "...",
      "description": "..."}, ...], "pr_description": "..."}``
    * ``{"setup_outcome": "no-tasks-needed", "reason": "...",
      "pr_description": "..."}``

    ``pr_description`` is optional in both shapes.  When provided, the harness
    writes it to the PR body directly and skips the separate description-rewrite
    LLM call.

    Raises ValueError on any malformed shape so the worker can log the
    parse failure and fall through to the no-tasks finalize path.
    """
    obj = _last_json_object(text)
    kind = obj.get("setup_outcome")
    match kind:
        case "tasks-planned":
            tasks = _parse_tasks(obj.get("tasks"))
            if not tasks:
                raise ValueError(
                    'setup_outcome "tasks-planned" requires at least one task '
                    '— use "no-tasks-needed" instead'
                )
            pr_description = _optional_str(
                obj, "pr_description", 'setup_outcome "tasks-planned"'
            )
            return TasksPlanned(tasks=tasks, pr_description=pr_description)
        case "no-tasks-needed":
            reason = _require_str(obj, "reason", 'setup_outcome "no-tasks-needed"')
            pr_description = _optional_str(
                obj, "pr_description", 'setup_outcome "no-tasks-needed"'
            )
            return NoTasksNeeded(reason=reason, pr_description=pr_description)
        case None:
            raise ValueError(f'JSON object has no "setup_outcome" key: {obj!r}')
        case _:
            raise ValueError(f"Unrecognised setup_outcome value: {kind!r}")
