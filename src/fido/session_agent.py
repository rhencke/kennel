"""Shared session-backed provider agent behavior."""

import json
import logging
import threading
from collections.abc import Callable
from pathlib import Path

from fido.appstate import FidoState, ProviderSnapshot
from fido.atomic import AtomicUpdater
from fido.provider import (
    READ_ONLY_ALLOWED_TOOLS,
    PromptOutcome,
    PromptSession,
    ProviderModel,
    SnapshotPublisher,
    TurnSessionMode,
)
from fido.rocq import cancel_survives_respawn as cancel_fsm

log = logging.getLogger(__name__)


class SessionBackedAgent(SnapshotPublisher):
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
        state_updater: AtomicUpdater[FidoState] | None = None,
    ) -> None:
        self._session_fn = session_fn
        self._session_system_file = session_system_file
        self._work_dir = work_dir
        self._repo_name = repo_name
        self._session_lock = threading.Lock()
        self._session: PromptSession | None = session
        self._state_updater: AtomicUpdater[FidoState] | None = state_updater

    @property
    def state_updater(self) -> AtomicUpdater[FidoState] | None:
        """Return the injected :class:`~fido.atomic.AtomicUpdater`, if any."""
        return self._state_updater

    def publish_metrics(
        self,
        *,
        owner: str | None,
        alive: bool,
        pid: int | None,
        dropped_count: int,
        sent_count: int,
        received_count: int,
    ) -> None:
        """Publish a fresh :class:`~fido.appstate.ProviderSnapshot`.

        Called by provider sessions after incrementing counters.  Installs a
        new :class:`~fido.appstate.ProviderSnapshot` at
        ``repos[repo_name].provider`` in the atomic state cell via
        :attr:`_state_updater`.

        No-op when :attr:`_state_updater` or :attr:`_repo_name` is ``None``
        — the agent was either constructed without state-publish wiring
        (tests, standalone use) or without a known repo identity.
        """
        if self._state_updater is None or self._repo_name is None:
            return
        snapshot = ProviderSnapshot(
            session_owner=owner or "",
            session_alive=alive,
            session_pid=pid or 0,
            session_dropped_count=dropped_count,
            session_sent_count=sent_count,
            session_received_count=received_count,
        )
        _name = self._repo_name
        self._state_updater.update(lambda root: root.repos[_name].provider, snapshot)

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
        session_id = session.session_id
        return session_id if isinstance(session_id, str) and session_id else None

    @property
    def session_dropped_count(self) -> int:
        with self._session_lock:
            session = self._session
        if session is None or not hasattr(session, "dropped_session_count"):
            return 0
        dropped = session.dropped_session_count
        return dropped if dropped >= 0 else 0

    @property
    def session_sent_count(self) -> int:
        """Cumulative number of messages sent to the provider since boot."""
        with self._session_lock:
            session = self._session
        return session.sent_count if session is not None else 0

    @property
    def session_received_count(self) -> int:
        """Cumulative number of responses/events received from the provider since boot."""
        with self._session_lock:
            session = self._session
        return session.received_count if session is not None else 0

    def ensure_session(
        self,
        model: ProviderModel | None = None,
        *,
        session_id: str | None = None,
    ) -> None:
        """Ensure a persistent session exists, optionally resuming *session_id*.

        When *session_id* is provided and no session has been created yet,
        the new session spawns with that id so the provider can resume the
        prior conversation (claude ``--resume``, Copilot ACP ``load_session``).
        Ignored when the session already exists.  Fix for #649 — without
        this, every fido self-restart drops the conversation context.
        """
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
                self._session = self._spawn_owned_session(model, session_id=session_id)
                return
        if model is not None:
            session.switch_model(model)

    def stop_session(self) -> None:
        with self._session_lock:
            session = self._session
            self._session = None
        if session is not None:
            session.stop()

    def recover_session(self) -> bool:
        """Recover the attached persistent session if one exists."""
        with self._session_lock:
            session = self._session
        if session is None:
            return False
        session.recover()
        return True

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
        allowed_tools: str | None = READ_ONLY_ALLOWED_TOOLS,
        system_prompt: str | None = None,
        retry_on_preempt: bool = False,
        session_mode: TurnSessionMode = TurnSessionMode.REUSE,
    ) -> PromptOutcome:
        session = self._resolve_turn_session(
            model=model,
            session_mode=session_mode,
        )
        attempt = 0
        while True:
            outcome = self._prompt_with_recovery(
                session,
                content,
                model=model,
                allowed_tools=allowed_tools,
                system_prompt=system_prompt,
            )
            if not retry_on_preempt or not outcome.cancelled:
                return outcome
            attempt += 1
            log.info(
                "%s.run_turn: preempted mid-flight — retry %d",
                type(self).__name__,
                attempt,
            )

    def _spawn_owned_session(
        self, model: ProviderModel, *, session_id: str | None = None
    ) -> PromptSession:
        raise NotImplementedError

    def _run_turn_json_value(
        self,
        prompt: str,
        key: str,
        model: ProviderModel,
        *,
        allowed_tools: str | None = READ_ONLY_ALLOWED_TOOLS,
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
        raw = self.run_turn(
            prompt,
            model=model,
            allowed_tools=allowed_tools,
            system_prompt=full_system,
        )
        if not raw:
            return ""
        for candidate in self._json_parse_candidates(raw):
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and isinstance(obj.get(key), str):
                return obj[key]
        return ""

    def _json_parse_candidates(self, raw: str) -> tuple[str, ...]:
        return (raw,)

    def _session_is_dead(self, session: PromptSession) -> bool:
        return session.is_alive() is False

    def _recover_prompt_session(self, session: PromptSession) -> bool:
        recover = getattr(session, "recover", None)
        if not callable(recover):
            return False
        recover()
        return True

    def _prompt_failure_is_passthrough(self, exc: Exception) -> bool:
        del exc
        return False

    def _should_retry_prompt_failure(
        self,
        exc: Exception,
        session: PromptSession,
    ) -> bool:
        del exc
        return self._session_is_dead(session)

    def _dead_prompt_error_message(self) -> str:
        return "session died during prompt"

    def _prompt_with_recovery(
        self,
        session: PromptSession,
        content: str,
        *,
        model: ProviderModel | None,
        allowed_tools: str | None = READ_ONLY_ALLOWED_TOOLS,
        system_prompt: str | None,
    ) -> PromptOutcome:
        # Cancel-survives-respawn FSM oracle (closes #1792).  Backed by the
        # proof in ``models/cancel_survives_respawn.v``: once a cancel is
        # observed in any cycle of this recovery loop, the function must
        # return with the session reporting ``last_turn_cancelled = True``
        # and must NOT retry the prompt on a fresh subprocess (the retry
        # would silently consume the cancel intent and the caller would
        # never see the preemption it requested).  Every transition below
        # is fed through ``cancel_fsm.transition`` so a runtime divergence
        # from the proved FSM crashes loudly instead of leaking a queued
        # PR comment.
        fsm_state = cancel_fsm.transition(cancel_fsm.initial_state, cancel_fsm.Prompt())
        assert fsm_state is not None, (
            "cancel_survives_respawn FSM rejected Prompt from Initial"
        )
        recovered = False
        while True:
            try:
                raw = session.prompt(
                    content,
                    model=model,
                    allowed_tools=allowed_tools,
                    system_prompt=system_prompt,
                )
                # Test doubles may return a plain ``str`` rather than a
                # :class:`PromptOutcome`; normalize by reading the
                # session's cancel bit only as the fallback path.
                if hasattr(raw, "cancelled"):
                    result = raw
                else:
                    result = PromptOutcome(
                        raw,
                        cancelled=getattr(session, "last_turn_cancelled", False)
                        is True,
                    )
            except Exception as exc:
                if self._prompt_failure_is_passthrough(exc):
                    raise
                # Exception path: read cancellation from the exception
                # itself.  ``session.prompt`` captures the sticky bit
                # INSIDE its own lock at the moment of failure and
                # attaches ``cancel_observed`` to the raised exception
                # (codex P1 round on commit 3a9cd09).  This avoids
                # reading mutable session state after the lock is
                # released — that read raced with the next thread
                # acquiring the session and clearing the bit.  Falls
                # back to ``consume_pending_cancel`` and then to a
                # one-shot ``last_turn_cancelled`` read for test
                # doubles that don't propagate ``cancel_observed``.
                cancel_attr = getattr(exc, "cancel_observed", None)
                if cancel_attr is not None:
                    cancel_observed = cancel_attr is True
                else:
                    consume = getattr(session, "consume_pending_cancel", None)
                    if callable(consume):
                        cancel_observed = consume() is True
                    else:
                        cancel_observed = (
                            getattr(session, "last_turn_cancelled", False) is True
                        )
                if cancel_observed:
                    fsm_state = cancel_fsm.transition(
                        fsm_state, cancel_fsm.CancelFire()
                    )
                    assert fsm_state is not None
                fsm_state = cancel_fsm.transition(
                    fsm_state, cancel_fsm.SubprocessExit()
                )
                assert fsm_state is not None
                if cancel_observed:
                    # The prompt failed because a peer thread fired a cancel
                    # during this turn (the subprocess crashed mid-drain).
                    # Retrying would respawn the session in NoCancel state
                    # and silently drop the preemption (codex P1 / #1792).
                    # Recover the session BEFORE returning so the next
                    # prompt call doesn't loop on the same dead-session
                    # + sticky-cancel-bit state (codex P1 follow-up on
                    # #1793: ``retry_on_preempt=True`` callers and the
                    # next worker turn both need a live session waiting).
                    # The respawn clears the sticky bit naturally — see
                    # :meth:`ClaudeSession.iter_events` and the
                    # ``self._cancel.clear()`` boundary.
                    self._recover_prompt_session(session)
                    fsm_state = cancel_fsm.transition(fsm_state, cancel_fsm.Recover())
                    assert fsm_state is not None
                    fsm_state = cancel_fsm.transition(fsm_state, cancel_fsm.Return())
                    assert fsm_state == cancel_fsm.returned_cancelled, (
                        f"cancel_survives_respawn FSM: expected returned_cancelled, "
                        f"got {fsm_state!r}"
                    )
                    log.info(
                        "%s: prompt failed mid-cancel — recovered session, "
                        "returning empty so the caller observes cancellation",
                        type(self).__name__,
                    )
                    return PromptOutcome("", cancelled=True)
                should_retry = self._should_retry_prompt_failure(exc, session)
                if (
                    recovered
                    or not should_retry
                    or not self._recover_prompt_session(session)
                ):
                    raise
                fsm_state = cancel_fsm.transition(fsm_state, cancel_fsm.Recover())
                assert fsm_state is not None
                recovered = True
                log_name = type(self).__name__
                log.warning(
                    "%s: recovered session after prompt failure: %s", log_name, exc
                )
                continue
            cancel_observed = result.cancelled
            if cancel_observed:
                fsm_state = cancel_fsm.transition(fsm_state, cancel_fsm.CancelFire())
                assert fsm_state is not None
            if result or cancel_observed or not self._session_is_dead(session):
                fsm_state = cancel_fsm.transition(fsm_state, cancel_fsm.Return())
                assert fsm_state is not None
                return result
            fsm_state = cancel_fsm.transition(fsm_state, cancel_fsm.SubprocessExit())
            assert fsm_state is not None
            if recovered or not self._recover_prompt_session(session):
                raise RuntimeError(self._dead_prompt_error_message())
            fsm_state = cancel_fsm.transition(fsm_state, cancel_fsm.Recover())
            assert fsm_state is not None
            recovered = True
            log.warning(
                "%s: recovered session after empty dead prompt", type(self).__name__
            )

    def _run_shared_turn(
        self,
        content: str,
        *,
        model: ProviderModel,
        allowed_tools: str | None = READ_ONLY_ALLOWED_TOOLS,
        system_prompt: str | None = None,
    ) -> tuple[str, str]:
        # Route through _prompt_with_recovery so a stale subprocess (BrokenPipe
        # on stdin write) recovers and retries instead of killing the worker
        # and leaving the persistent ClaudeSession FSM stuck in Sending.
        session = self._resolve_turn_session(
            model=model,
            session_mode=TurnSessionMode.REUSE,
        )
        text = self._prompt_with_recovery(
            session,
            content,
            model=model,
            allowed_tools=allowed_tools,
            system_prompt=system_prompt,
        )
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
        *,
        allowed_tools: str | None = READ_ONLY_ALLOWED_TOOLS,
    ) -> str:
        return self.run_turn(
            prompt,
            model=self.voice_model if model is None else model,
            allowed_tools=allowed_tools,
            system_prompt=system_prompt,
        )

    def generate_status_emoji(
        self,
        prompt: str,
        system_prompt: str,
        model: ProviderModel | None = None,
        *,
        allowed_tools: str | None = READ_ONLY_ALLOWED_TOOLS,
    ) -> str:
        return self._run_turn_json_value(
            prompt,
            "emoji",
            self.voice_model if model is None else model,
            allowed_tools=allowed_tools,
            system_prompt=system_prompt,
        )

    def generate_reply(
        self,
        prompt: str,
        model: ProviderModel | None = None,
        timeout: int = 30,
        *,
        allowed_tools: str | None = READ_ONLY_ALLOWED_TOOLS,
    ) -> str:
        del timeout
        text, _ = self._run_shared_turn(
            prompt,
            model=self.voice_model if model is None else model,
            allowed_tools=allowed_tools,
        )
        return text.strip()

    def generate_branch_name(
        self,
        prompt: str,
        model: ProviderModel | None = None,
        timeout: int = 15,
        *,
        allowed_tools: str | None = READ_ONLY_ALLOWED_TOOLS,
    ) -> str:
        del timeout
        text, _ = self._run_shared_turn(
            prompt,
            model=self.brief_model if model is None else model,
            allowed_tools=allowed_tools,
        )
        stripped = text.strip()
        return stripped.splitlines()[0] if stripped else ""

    def generate_status_with_session(
        self,
        prompt: str,
        system_prompt: str,
        model: ProviderModel | None = None,
        timeout: int = 15,
        *,
        allowed_tools: str | None = READ_ONLY_ALLOWED_TOOLS,
    ) -> tuple[str, str]:
        del timeout
        text, session_id = self._run_shared_turn(
            prompt,
            model=self.voice_model if model is None else model,
            allowed_tools=allowed_tools,
            system_prompt=system_prompt,
        )
        return text.strip(), session_id

    def resume_status(
        self,
        session_id: str,
        prompt: str,
        model: ProviderModel | None = None,
        timeout: int = 15,
        *,
        allowed_tools: str | None = READ_ONLY_ALLOWED_TOOLS,
    ) -> str:
        del timeout
        session = self._require_matching_session(session_id)
        return session.prompt(
            prompt,
            model=self.voice_model if model is None else model,
            allowed_tools=allowed_tools,
        ).strip()
