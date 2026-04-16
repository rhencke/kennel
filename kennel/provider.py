"""Provider contracts and normalized provider-limit state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol, TypeAlias


class ProviderID(StrEnum):
    """Supported LLM providers for kennel."""

    CLAUDE_CODE = "claude-code"
    COPILOT_CLI = "copilot-cli"
    CODEX = "codex"
    GEMINI = "gemini"


ReasoningEffort: TypeAlias = Literal["low", "medium", "high"]
ReasoningEffortSpec: TypeAlias = ReasoningEffort | tuple[ReasoningEffort, ...] | None


@dataclass(frozen=True, eq=False)
class ProviderModel:
    """Explicit provider model selection, including reasoning effort when supported."""

    model: str
    effort: ReasoningEffortSpec = None

    @property
    def efforts(self) -> tuple[ReasoningEffort, ...]:
        """Return configured effort levels in preference order."""
        if self.effort is None:
            return ()
        if isinstance(self.effort, tuple):
            return self.effort
        return (self.effort,)

    def __str__(self) -> str:
        return self.model

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ProviderModel):
            return (self.model, self.effort) == (other.model, other.effort)
        if isinstance(other, str):
            return self.model == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.model, self.effort))


def coerce_provider_model(model: ProviderModel | str) -> ProviderModel:
    """Return *model* as a :class:`ProviderModel`."""
    if isinstance(model, ProviderModel):
        return model
    return ProviderModel(model)


def model_name(model: ProviderModel | str) -> str:
    """Return the provider-native model id for *model*."""
    return coerce_provider_model(model).model


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
    def owner(self) -> str | None:
        """Return the logical owner label for the session, if any."""
        ...

    @property
    def pid(self) -> int | None:
        """Return the subprocess PID backing the session, if any."""
        ...

    @property
    def session_id(self) -> str | None:
        """Return the provider-native persistent session identifier, if any."""
        ...

    def prompt(
        self,
        content: str,
        model: ProviderModel | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Run a single prompt turn and return the final assistant text."""
        ...

    def send(self, content: str) -> None:
        """Send a turn to the session without waiting for the final result."""
        ...

    def consume_until_result(self) -> str:
        """Block until the current in-flight turn produces its final text result."""
        ...

    @property
    def last_turn_cancelled(self) -> bool:
        """Report whether the most recent turn was cancelled by preemption."""
        ...

    def wait_for_pending_preempt(self, timeout: float = 30.0) -> bool:
        """Wait for any queued preemption work to drain before retrying a turn."""
        ...

    def switch_model(self, model: ProviderModel) -> None:
        """Switch the live session to *model* without resetting session state."""
        ...

    def recover(self) -> None:
        """Reattach or revive the underlying session after an interruption."""
        ...

    def reset(self, model: ProviderModel | None = None) -> None:
        """Reset the session, optionally selecting a fresh initial *model*."""
        ...

    def is_alive(self) -> bool:
        """Return whether the underlying session transport is still alive."""
        ...

    def stop(self) -> None:
        """Shut down the session and release any underlying resources."""
        ...

    def __enter__(self) -> "PromptSession":
        """Enter the session's turn-serialization context."""
        ...

    def __exit__(self, *args: object) -> None:
        """Exit the session's turn-serialization context."""
        ...


class ProviderAgent(Protocol):
    """Runtime/session boundary for a provider's interactive agent."""

    @property
    def provider_id(self) -> ProviderID:
        """Return the stable provider identifier for this agent."""
        ...

    @property
    def session(self) -> PromptSession | None:
        """Return the currently attached persistent session, if any."""
        ...

    @property
    def session_owner(self) -> str | None:
        """Return the attached session owner label, if available."""
        ...

    @property
    def session_alive(self) -> bool:
        """Return whether the attached session transport is currently alive."""
        ...

    @property
    def session_pid(self) -> int | None:
        """Return the attached session PID, if the provider exposes one."""
        ...

    @property
    def session_id(self) -> str | None:
        """Return the provider-native persistent session identifier, if any."""
        ...

    voice_model: ProviderModel
    work_model: ProviderModel
    brief_model: ProviderModel

    def attach_session(self, session: PromptSession | None) -> None:
        """Attach *session* as the agent's persistent session."""
        ...

    def detach_session(self) -> PromptSession | None:
        """Detach and return the current persistent session, if any."""
        ...

    def ensure_session(self, model: ProviderModel | None = None) -> None:
        """Ensure that a persistent session exists, optionally seeded with *model*."""
        ...

    def stop_session(self) -> None:
        """Stop and detach any currently attached persistent session."""
        ...

    def run_turn(
        self,
        content: str,
        *,
        model: ProviderModel | None = None,
        system_prompt: str | None = None,
        retry_on_preempt: bool = False,
        fresh_session: bool = False,
    ) -> str:
        """Run one interactive turn through the persistent session and return text."""
        ...

    def print_prompt_from_file(
        self,
        system_file: Path,
        prompt_file: Path,
        model: ProviderModel,
        timeout: int = 30,
        idle_timeout: float = 1800.0,
        cwd: Path | str = ".",
    ) -> str:
        """Run a one-shot prompt sourced from files and return raw provider output."""
        ...

    def resume_session(
        self,
        session_id: str,
        prompt_file: Path,
        model: ProviderModel,
        timeout: int = 300,
        idle_timeout: float = 1800.0,
        cwd: Path | str = ".",
    ) -> str:
        """Resume a prior one-shot provider session and return raw output."""
        ...

    def generate_reply(
        self,
        prompt: str,
        model: ProviderModel | None = None,
        timeout: int = 30,
    ) -> str:
        """Generate a short natural-language reply for a GitHub comment flow."""
        ...

    def generate_branch_name(
        self,
        prompt: str,
        model: ProviderModel | None = None,
        timeout: int = 15,
    ) -> str:
        """Generate a git branch-name slug from *prompt*."""
        ...

    def generate_status(
        self,
        prompt: str,
        system_prompt: str,
        model: ProviderModel | None = None,
    ) -> str:
        """Generate a two-line GitHub status message."""
        ...

    def generate_status_emoji(
        self,
        prompt: str,
        system_prompt: str,
        model: ProviderModel | None = None,
    ) -> str:
        """Generate a single emoji suitable for a GitHub status."""
        ...

    def generate_status_with_session(
        self,
        prompt: str,
        system_prompt: str,
        model: ProviderModel | None = None,
        timeout: int = 15,
    ) -> tuple[str, str]:
        """Generate a status and return ``(status_text, resumable_session_id)``."""
        ...

    def resume_status(
        self,
        session_id: str,
        prompt: str,
        model: ProviderModel | None = None,
        timeout: int = 15,
    ) -> str:
        """Resume a prior status-generation session and return the refined text."""
        ...

    def extract_session_id(self, output: str) -> str:
        """Extract a session id from provider-specific raw one-shot output."""
        ...


class ProviderAPI(Protocol):
    """Read-only account/service API surface for a provider."""

    @property
    def provider_id(self) -> ProviderID:
        """Return the stable provider identifier for this API client."""
        ...

    def get_limit_snapshot(self) -> ProviderLimitSnapshot:
        """Return the provider's current normalized rate-limit snapshot."""
        ...


class Provider(Protocol):
    """Composite provider with separate API and agent collaborators."""

    @property
    def provider_id(self) -> ProviderID:
        """Return the stable provider identifier for this composite provider."""
        ...

    @property
    def api(self) -> ProviderAPI:
        """Return the provider's read-only API collaborator."""
        ...

    @property
    def agent(self) -> ProviderAgent:
        """Return the provider's interactive runtime collaborator."""
        ...
