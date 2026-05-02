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
