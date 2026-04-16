"""Shared session-backed provider agent behavior."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path

from kennel.provider import PromptSession, ProviderModel, TurnSessionMode


class SessionBackedAgent:
    """Common session attachment and lifecycle logic for provider agents."""

    voice_model: ProviderModel
    brief_model: ProviderModel

    def __init__(
        self,
        *,
        session_fn: Callable[[], PromptSession],
        session_system_file: Path | None,
        work_dir: Path | str | None,
        repo_name: str | None,
        session: PromptSession | None,
    ) -> None:
        self._session_fn = session_fn
        self._session_system_file = session_system_file
        self._work_dir = work_dir
        self._repo_name = repo_name
        self._session_lock = threading.Lock()
        self._session: PromptSession | None = session

    @property
    def session(self) -> PromptSession | None:
        with self._session_lock:
            return self._session

    def attach_session(self, session: PromptSession | None) -> None:
        with self._session_lock:
            self._session = session

    def detach_session(self) -> PromptSession | None:
        with self._session_lock:
            session = self._session
            self._session = None
            return session

    @property
    def session_owner(self) -> str | None:
        with self._session_lock:
            session = self._session
        return session.owner if session is not None else None

    @property
    def session_alive(self) -> bool:
        with self._session_lock:
            session = self._session
        return session is not None and session.is_alive()

    @property
    def session_pid(self) -> int | None:
        with self._session_lock:
            session = self._session
        return session.pid if session is not None else None

    @property
    def session_id(self) -> str | None:
        with self._session_lock:
            session = self._session
        if session is None or not hasattr(session, "session_id"):
            return None
        session_id = getattr(session, "session_id")
        return session_id if isinstance(session_id, str) and session_id else None

    def ensure_session(self, model: ProviderModel | None = None) -> None:
        with self._session_lock:
            session = self._session
            if session is None:
                if self._session_system_file is None or self._work_dir is None:
                    raise ValueError(
                        f"{type(self).__name__}.ensure_session requires session_system_file and work_dir"
                    )
                if model is None:
                    raise ValueError(
                        f"{type(self).__name__}.ensure_session requires model when creating a session"
                    )
                self._session = self._spawn_owned_session(model)
                return
        if model is not None:
            session.switch_model(model)

    def stop_session(self) -> None:
        with self._session_lock:
            session = self._session
            self._session = None
        if session is not None:
            session.stop()

    def _can_spawn_owned_session(self) -> bool:
        return self._session_system_file is not None and self._work_dir is not None

    def _resolve_turn_session(
        self,
        *,
        model: ProviderModel | None,
        session_mode: TurnSessionMode,
    ) -> PromptSession:
        resolver_error: RuntimeError | None = None
        with self._session_lock:
            session = self._session
        if session is None:
            try:
                session = self._session_fn()
            except RuntimeError as exc:
                resolver_error = exc
                session = None
        if session is None and self._can_spawn_owned_session():
            if model is None:
                raise ValueError(
                    f"{type(self).__name__}.run_turn requires model when creating a session"
                )
            session = self._spawn_owned_session(model)
            with self._session_lock:
                self._session = session
            return session
        if session is None and resolver_error is not None:
            raise resolver_error
        if session is None:
            raise RuntimeError(
                f"{type(self).__name__}.run_turn could not resolve a session"
            )
        if session_mode == TurnSessionMode.FRESH:
            reset = getattr(session, "reset", None)
            if not callable(reset):
                raise ValueError(
                    f"{type(self).__name__}.run_turn session_mode=fresh requires resettable session"
                )
            reset(model)
        return session

    def run_turn(
        self,
        content: str,
        *,
        model: ProviderModel | None = None,
        system_prompt: str | None = None,
        retry_on_preempt: bool = False,
        session_mode: TurnSessionMode = TurnSessionMode.REUSE,
    ) -> str:
        del content, model, system_prompt, retry_on_preempt, session_mode
        raise NotImplementedError

    def _spawn_owned_session(self, model: ProviderModel) -> PromptSession:
        raise NotImplementedError

    def _run_turn_json_value(
        self,
        prompt: str,
        key: str,
        model: ProviderModel,
        system_prompt: str | None = None,
    ) -> str:
        json_instruction = (
            f'Respond with ONLY a JSON object in the form {{"{key}": "your answer"}}.'
            " No other text before or after the JSON."
        )
        full_system = (
            f"{system_prompt}\n\n{json_instruction}"
            if system_prompt
            else json_instruction
        )
        raw = self.run_turn(prompt, model=model, system_prompt=full_system)
        if not raw:
            return ""
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return ""
        return (
            obj[key] if isinstance(obj, dict) and isinstance(obj.get(key), str) else ""
        )

    def _run_shared_turn(
        self,
        content: str,
        *,
        model: ProviderModel,
        system_prompt: str | None = None,
    ) -> tuple[str, str]:
        session = self._resolve_turn_session(
            model=model,
            session_mode=TurnSessionMode.REUSE,
        )
        text = session.prompt(content, model=model, system_prompt=system_prompt)
        session_id = getattr(session, "session_id", None)
        return text, session_id if isinstance(session_id, str) else ""

    def _require_matching_session(self, session_id: str) -> PromptSession:
        with self._session_lock:
            session = self._session
        if session is None:
            session = self._session_fn()
        live_session_id = getattr(session, "session_id", None)
        if live_session_id != session_id:
            raise RuntimeError(
                f"{type(self).__name__} resume helpers require the matching live session"
            )
        return session

    def generate_status(
        self,
        prompt: str,
        system_prompt: str,
        model: ProviderModel | None = None,
    ) -> str:
        return self.run_turn(
            prompt,
            model=self.voice_model if model is None else model,
            system_prompt=system_prompt,
        )

    def generate_status_emoji(
        self,
        prompt: str,
        system_prompt: str,
        model: ProviderModel | None = None,
    ) -> str:
        return self._run_turn_json_value(
            prompt,
            "emoji",
            self.voice_model if model is None else model,
            system_prompt=system_prompt,
        )

    def generate_reply(
        self,
        prompt: str,
        model: ProviderModel | None = None,
        timeout: int = 30,
    ) -> str:
        del timeout
        text, _ = self._run_shared_turn(
            prompt,
            model=self.voice_model if model is None else model,
        )
        return text.strip()

    def generate_branch_name(
        self,
        prompt: str,
        model: ProviderModel | None = None,
        timeout: int = 15,
    ) -> str:
        del timeout
        text, _ = self._run_shared_turn(
            prompt,
            model=self.brief_model if model is None else model,
        )
        stripped = text.strip()
        return stripped.splitlines()[0] if stripped else ""

    def generate_status_with_session(
        self,
        prompt: str,
        system_prompt: str,
        model: ProviderModel | None = None,
        timeout: int = 15,
    ) -> tuple[str, str]:
        del timeout
        text, session_id = self._run_shared_turn(
            prompt,
            model=self.voice_model if model is None else model,
            system_prompt=system_prompt,
        )
        return text.strip(), session_id

    def resume_status(
        self,
        session_id: str,
        prompt: str,
        model: ProviderModel | None = None,
        timeout: int = 15,
    ) -> str:
        del timeout
        session = self._require_matching_session(session_id)
        return session.prompt(
            prompt,
            model=self.voice_model if model is None else model,
        ).strip()
