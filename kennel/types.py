"""Shared type definitions for kennel."""

from __future__ import annotations

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
