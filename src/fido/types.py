"""Shared type definitions for fido."""

from dataclasses import dataclass
from enum import StrEnum


class TaskType(StrEnum):
    CI = "ci"
    THREAD = "thread"
    SPEC = "spec"


class TaskStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ActiveIssue:
    """Snapshot of the GitHub issue being worked on.

    Injected into the system prompt at every LLM call site so the agent
    always has the spec in front of it, regardless of which turn it is on.
    """

    number: int
    title: str
    body: str


@dataclass(frozen=True)
class ActivePR:
    """Snapshot of the pull request associated with the current work session."""

    number: int
    title: str
    url: str
    body: str


@dataclass(frozen=True)
class ClosedPR:
    """A prior closed (not merged) PR that referenced the same issue.

    Surfaced in the system prompt so the agent can learn from earlier
    attempts and avoid repeating the same mistakes.
    """

    number: int
    title: str
    body: str
    close_reason: str


@dataclass(frozen=True)
class ClosedSubIssue:
    """A closed direct sub-issue of the parent issue being worked on.

    Fetched during setup so the agent can subtract already-covered scope
    from the parent's task list (or declare ``NoTasksNeeded`` when all
    scope is covered).

    Attributes
    ----------
    number:
        GitHub issue number of the sub-issue.
    title:
        Title of the sub-issue.
    body:
        Body text of the sub-issue.
    close_state:
        One of ``"merged"`` (a linked PR was merged), ``"closed_unmerged"``
        (a linked PR was closed without merging), or ``"closed_no_pr"``
        (no linked PR — cancelled, won't-fix, or deferred).
    state_reason:
        GitHub's ``state_reason`` for the sub-issue (``"completed"``,
        ``"not_planned"``, ``"reopened"``, or ``None`` when absent).  Only
        meaningful when ``close_state`` is ``"closed_no_pr"`` — used to
        distinguish "completed without a PR" from "won't fix / deferred".
    pr_number:
        GitHub PR number of the linked pull request, or ``None`` when
        ``close_state`` is ``"closed_no_pr"``.
    pr_repo:
        ``"owner/repo"`` of the linked pull request, or ``None`` when
        ``close_state`` is ``"closed_no_pr"``.  May differ from the parent
        issue's repo for cross-repo sub-issue PRs.
    pr_body:
        Body text of the linked PR, or ``""`` when no PR exists.
    """

    number: int
    title: str
    body: str
    close_state: str
    state_reason: str | None = None
    pr_number: int | None = None
    pr_repo: str | None = None
    pr_body: str = ""


@dataclass(frozen=True)
class TaskSnapshot:
    """Projection of a task dict for LLM context rendering.

    Captures only the fields needed by :func:`~fido.prompts.render_active_context`
    so the renderer does not depend on the full task dict shape from
    ``tasks.json``.
    """

    title: str
    type: str
    status: str
    description: str = ""


@dataclass(frozen=True)
class RescopeIntent:
    """Origin metadata for a rescope trigger from comment synthesis.

    Carries the plain-English change request alongside the originating
    comment identity and a stable ordering timestamp so the rescoper can
    process intents in arrival order and reply back to the right commenter.

    Attributes
    ----------
    change_request:
        Plain-English description of the requested scope change, as written
        by the synthesis LLM from the comment's content.
    comment_id:
        GitHub ID of the comment that triggered the rescope.
    timestamp:
        ISO-8601 UTC timestamp of when the rescope was triggered — used for
        ordering when multiple intents are coalesced into a single Opus call.
    """

    change_request: str
    comment_id: int
    timestamp: str
    comment_type: str = "pulls"
    """GitHub comment namespace: ``"pulls"`` for review-thread comments
    (the rescope reply path posts via ``reply_to_review_comment``);
    ``"issues"`` for top-level PR/issue comments where the webhook
    handler already posted a triage reply (rescope notifier silently
    skips per #1724 codex P2 — same policy as ``_notify_thread_change``)."""
    author: str = ""
    """GitHub login of the commenter.  Rendered alongside the change
    request in the rescope prompt so Opus can apply per-author
    supersedence semantics (later #1803 / INV-E: suppress reply-back
    when an intent was superseded by another from the same author —
    the commenter already knows they overrode themselves).  Empty
    string when the source path didn't populate it (legacy / synthetic
    test fixtures pre-INV-C)."""


@dataclass(frozen=True)
class GitIdentity:
    """GitHub-derived git commit identity.

    Name comes from the authenticated user's display name (falling back to
    login).  Email is always the GitHub noreply form
    ``{id}+{login}@users.noreply.github.com`` — the user's real email is
    never used, even if exposed via the API.
    """

    name: str
    email: str

    def __str__(self) -> str:
        return f"{self.name} <{self.email}>"
