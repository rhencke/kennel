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
