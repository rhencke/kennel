"""Translate raw GitHub webhook payloads into :class:`IssueTreeCache`
event tuples (closes #812, sub-issue half closes #817).

Pure value-only helper — no I/O, no cache mutation.  Caller passes the
result to ``cache.apply_event(*translation)`` if non-None.

Required webhook subscriptions (verify per repo at startup):

- ``issues`` — opened, closed, reopened, assigned, unassigned, edited,
  milestoned, demilestoned, transferred, deleted
- ``sub_issues`` — sub_issue_added, sub_issue_removed, parent_issue_added,
  parent_issue_removed.  Without this subscription the cache misses every
  sub-issue link change and the picker descends a stale tree (#817).
- ``pull_request`` — closed, reopened (used by other code paths; the
  cache itself does not currently track PRs)
"""

from datetime import datetime, timezone
from typing import Any


def _parse_ts(value: str | None) -> datetime:
    """Parse an ISO-8601 GitHub webhook timestamp to a tz-aware
    datetime.  Falls back to "now" when missing — webhooks always carry
    a timestamp on the relevant object so this branch is a guard rail.
    """
    if not value:
        return datetime.now(tz=timezone.utc)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _issue_assignees(issue: dict[str, Any]) -> list[str]:
    return [
        a.get("login", "") for a in (issue.get("assignees") or []) if a.get("login")
    ]


def _issue_milestone(issue: dict[str, Any]) -> str | None:
    milestone = issue.get("milestone")
    if not milestone:
        return None
    title = milestone.get("title")
    return title if isinstance(title, str) else None


def translate(
    event_type: str, payload: dict[str, Any]
) -> tuple[str, dict[str, Any]] | None:
    """Translate a webhook event to ``(cache_event_type, payload)`` for
    :meth:`IssueTreeCache.apply_event`, or ``None`` if the event isn't
    relevant to the cache.

    Webhook source-of-truth shape per GitHub docs:

    - ``issues``: payload has ``action``, ``issue`` (full issue
      snapshot), and ``assignee`` / ``milestone`` keys depending on action.
    - ``sub_issues``: payload has ``action`` plus ``parent_issue`` and
      ``sub_issue`` (full issue snapshots for both ends of the link).
      Both ``sub_issue_added`` / ``parent_issue_added`` and the
      corresponding ``_removed`` actions describe the same logical
      mutation from opposite ends — both map to the same cache event.
    - ``pull_request``: not currently translated (PRs aren't in the
      tree cache).
    """
    if event_type == "sub_issues":
        return _translate_sub_issues(payload)
    if event_type != "issues":
        return None
    action = payload.get("action", "")
    issue = payload.get("issue") or {}
    number = issue.get("number")
    if number is None:
        return None
    timestamp = _parse_ts(issue.get("updated_at") or issue.get("created_at"))

    base: dict[str, Any] = {
        "issue_number": number,
        "timestamp": timestamp,
    }

    match action:
        case "opened" | "reopened":
            return (
                "opened" if action == "opened" else "reopened",
                {
                    **base,
                    "title": issue.get("title", ""),
                    "assignees": _issue_assignees(issue),
                    "parent": (issue.get("parent") or {}).get("number"),
                    "sub_issues": [],
                    "milestone": _issue_milestone(issue),
                    "created_at": _parse_ts(issue.get("created_at")),
                },
            )
        case "closed" | "transferred" | "deleted":
            return ("closed", base)
        case "assigned":
            assignee = (payload.get("assignee") or {}).get("login")
            if not assignee:
                return None
            return ("assigned", {**base, "login": assignee})
        case "unassigned":
            assignee = (payload.get("assignee") or {}).get("login")
            if not assignee:
                return None
            return ("unassigned", {**base, "login": assignee})
        case "milestoned" | "demilestoned":
            return (
                "milestoned",
                {**base, "milestone": _issue_milestone(issue)},
            )
        case "edited":
            # Only re-emit when the title changed; other edits (body,
            # state) are no-ops for the picker tree.
            changes = payload.get("changes") or {}
            if "title" not in changes:
                return None
            return ("edited_title", {**base, "title": issue.get("title", "")})
        case "sub_issue_added":
            sub_issue = payload.get("sub_issue") or {}
            child = sub_issue.get("number")
            if child is None:
                return None
            return ("sub_issue_added", {**base, "child": child})
        case "sub_issue_removed":
            sub_issue = payload.get("sub_issue") or {}
            child = sub_issue.get("number")
            if child is None:
                return None
            return ("sub_issue_removed", {**base, "child": child})
        case _:
            return None


def _translate_sub_issues(
    payload: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    """Translate a ``sub_issues`` webhook payload (closes #817).

    GitHub fires this event from both sides of the link:

    - ``sub_issue_added`` / ``sub_issue_removed`` — fired on the parent
    - ``parent_issue_added`` / ``parent_issue_removed`` — fired on the child

    Both shapes carry the same ``parent_issue`` + ``sub_issue`` snapshots,
    so we map all four actions to the existing
    ``sub_issue_added`` / ``sub_issue_removed`` cache events keyed by the
    parent's number.  ``IssueTreeCache._handle_sub_issue_added`` already
    updates *both* nodes (parent.sub_issues and child.parent), so handling
    either side once is enough — and idempotent if both sides arrive.
    """
    action = payload.get("action", "")
    parent = payload.get("parent_issue") or {}
    sub = payload.get("sub_issue") or {}
    parent_number = parent.get("number")
    child_number = sub.get("number")
    if parent_number is None or child_number is None:
        return None
    # Sub-issue link mutations don't bump either end's ``updated_at`` —
    # that field tracks issue-body edits, which are unrelated to
    # relationship changes.  Using it as the event timestamp caused the
    # cache to drop legit mutations as stale whenever the parent or
    # child hadn't been edited recently (#819).  The mutation is
    # happening now from our perspective, so use now() — correct for
    # the cache's last_applied_at monotonicity check.
    timestamp = datetime.now(tz=timezone.utc)
    base = {
        "issue_number": parent_number,
        "child": child_number,
        "timestamp": timestamp,
    }
    match action:
        case "sub_issue_added" | "parent_issue_added":
            return ("sub_issue_added", base)
        case "sub_issue_removed" | "parent_issue_removed":
            return ("sub_issue_removed", base)
        case _:
            return None


__all__ = ["translate"]
