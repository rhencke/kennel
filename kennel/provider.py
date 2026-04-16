"""Provider contracts and normalized provider-limit state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol


class ProviderID(StrEnum):
    """Supported LLM providers for kennel."""

    CLAUDE_CODE = "claude-code"
    COPILOT_CLI = "copilot-cli"
    CODEX = "codex"
    GEMINI = "gemini"


@dataclass(frozen=True)
class ProviderLimitWindow:
    """One provider-specific limit window normalized for shared policy/UI use."""

    name: str
    used: int | None = None
    limit: int | None = None
    resets_at: datetime | None = None
    unit: str | None = None

    @property
    def pressure(self) -> float | None:
        """Return ``used / limit`` when both sides are known and limit is positive."""
        if self.used is None or self.limit is None or self.limit <= 0:
            return None
        return self.used / self.limit


@dataclass(frozen=True)
class ProviderLimitSnapshot:
    """Normalized limit state for one provider at one point in time."""

    provider: ProviderID
    windows: tuple[ProviderLimitWindow, ...] = ()
    unavailable_reason: str | None = None

    def closest_to_exhaustion(self) -> ProviderLimitWindow | None:
        """Return the highest-pressure window, or the first window when pressure is unknown."""
        pressured = [window for window in self.windows if window.pressure is not None]
        if pressured:
            return max(pressured, key=lambda window: window.pressure or 0.0)
        return self.windows[0] if self.windows else None


class PromptSession(Protocol):
    """Persistent prompt/session collaborator owned by a repo worker thread."""

    @property
    def owner(self) -> str | None: ...

    @property
    def pid(self) -> int | None: ...

    def prompt(
        self,
        content: str,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> str: ...

    def send(self, content: str) -> None: ...

    def consume_until_result(self) -> str: ...

    @property
    def last_turn_cancelled(self) -> bool: ...

    def wait_for_pending_preempt(self, timeout: float = 30.0) -> bool: ...

    def switch_model(self, model: str) -> None: ...

    def restart(self) -> None: ...

    def is_alive(self) -> bool: ...

    def stop(self) -> None: ...

    def __enter__(self) -> "PromptSession": ...

    def __exit__(self, *args: object) -> None: ...


class Provider(Protocol):
    """Shared provider boundary for runner LLM operations."""

    @property
    def provider_id(self) -> ProviderID: ...

    @property
    def session(self) -> PromptSession | None: ...

    @property
    def session_owner(self) -> str | None: ...

    @property
    def session_alive(self) -> bool: ...

    @property
    def session_pid(self) -> int | None: ...

    def attach_session(self, session: PromptSession | None) -> None: ...

    def detach_session(self) -> PromptSession | None: ...

    def get_limit_snapshot(self) -> ProviderLimitSnapshot | None: ...

    def print_prompt(
        self,
        prompt: str,
        model: str,
        system_prompt: str | None = None,
    ) -> str: ...

    def print_prompt_json(
        self,
        prompt: str,
        key: str,
        model: str,
        system_prompt: str | None = None,
    ) -> str: ...

    def print_prompt_from_file(
        self,
        system_file: Path,
        prompt_file: Path,
        model: str,
        timeout: int = 30,
        idle_timeout: float = 1800.0,
        cwd: Path | str | None = None,
    ) -> str: ...

    def resume_session(
        self,
        session_id: str,
        prompt_file: Path,
        model: str,
        timeout: int = 300,
        idle_timeout: float = 1800.0,
        cwd: Path | str | None = None,
    ) -> str: ...

    def generate_reply(
        self,
        prompt: str,
        model: str = "claude-opus-4-6",
        timeout: int = 30,
    ) -> str: ...

    def generate_branch_name(
        self,
        prompt: str,
        model: str = "claude-haiku-4-5-20251001",
        timeout: int = 15,
    ) -> str: ...

    def generate_status(
        self,
        prompt: str,
        system_prompt: str,
        model: str = "claude-opus-4-6",
    ) -> str: ...

    def generate_status_emoji(
        self,
        prompt: str,
        system_prompt: str,
        model: str = "claude-opus-4-6",
    ) -> str: ...

    def generate_status_with_session(
        self,
        prompt: str,
        system_prompt: str,
        model: str = "claude-opus-4-6",
        timeout: int = 15,
    ) -> tuple[str, str]: ...

    def resume_status(
        self,
        session_id: str,
        prompt: str,
        model: str = "claude-opus-4-6",
        timeout: int = 15,
    ) -> str: ...
