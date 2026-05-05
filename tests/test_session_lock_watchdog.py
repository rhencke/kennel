"""Tests for ``SessionLockWatchdog`` — the runtime driver that fires
``ForceRelease`` on FSM lock holders that have held the lock past a
deadline (closes #1377).

The watchdog delegates the actual eviction to
:meth:`~fido.provider.OwnedSession.force_release`, which is itself
covered by ``test_session_lock_oracle.py``.  These tests exercise
*just the watchdog's decision logic*: which repos it considers, what
hold-age threshold triggers eviction, and how it behaves when no
holder is present.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from fido import session_lock_watchdog
from fido.config import RepoConfig
from fido.provider import (
    OwnedSession,
    ProviderID,
    SessionTalker,
    register_talker,
    unregister_talker,
)
from fido.session_lock_watchdog import SessionLockWatchdog


class _RecordingSession(OwnedSession):
    """OwnedSession stub that records every ``force_release`` call.

    The watchdog calls :meth:`force_release` rather than poking FSM
    state directly, so a stub that records the calls is enough to
    assert "eviction fired with the expected reason" without spinning
    up a real provider session.
    """

    _repo_name: str | None

    def __init__(self, repo_name: str) -> None:
        self._repo_name = repo_name
        self._init_handler_reentry()
        self.force_release_calls: list[str] = []

    def _fire_worker_cancel(self) -> None:  # pragma: no cover — abstract hook
        pass

    def force_release(self, *, reason: str) -> bool:  # type: ignore[override]
        self.force_release_calls.append(reason)
        return True


class _StubRegistry:
    """Hand-rolled :class:`WorkerRegistry` substitute.

    Exposes only :meth:`get_session` because that is the sole
    registry method the watchdog touches — keeps the fake's surface
    area minimal so it can't drift away from the production contract
    silently.
    """

    def __init__(self, sessions: dict[str, OwnedSession | None]) -> None:
        self._sessions = sessions

    def get_session(self, repo_name: str) -> OwnedSession | None:
        return self._sessions.get(repo_name)


def _repo_config(name: str) -> RepoConfig:
    """Minimal RepoConfig for watchdog tests — values don't matter,
    only the dict key (the repo name) is consulted."""
    return RepoConfig(
        name=name,
        work_dir=Path(f"/tmp/{name.replace('/', '-')}"),
        provider=ProviderID.CLAUDE_CODE,
    )


# ---------------------------------------------------------------------------
# No-op paths: nothing to evict
# ---------------------------------------------------------------------------


def test_run_skips_repo_without_session() -> None:
    """Repos whose registry has no session attached are silently skipped."""
    registry = _StubRegistry({"FidoCanCode/home": None})
    watchdog = SessionLockWatchdog(
        registry,  # type: ignore[arg-type]
        {"FidoCanCode/home": _repo_config("FidoCanCode/home")},
        hold_deadline_seconds=0.01,
    )
    assert watchdog.run() == 0


def test_run_skips_repo_without_active_holder() -> None:
    """A session with no registered talker (FSM Free) is left alone."""
    session = _RecordingSession("FidoCanCode/home")
    registry = _StubRegistry({"FidoCanCode/home": session})
    watchdog = SessionLockWatchdog(
        registry,  # type: ignore[arg-type]
        {"FidoCanCode/home": _repo_config("FidoCanCode/home")},
        hold_deadline_seconds=0.01,
    )
    # No register_talker call → get_talker returns None.
    watchdog.run()
    assert session.force_release_calls == []


def test_run_skips_holder_within_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Holders that have not exceeded the deadline are left alone."""
    repo_name = "FidoCanCode/home"
    session = _RecordingSession(repo_name)
    registry = _StubRegistry({repo_name: session})
    started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    register_talker(
        SessionTalker(
            repo_name=repo_name,
            thread_id=42,
            kind="worker",
            description="recent turn",
            claude_pid=1,
            started_at=started_at,
        )
    )
    try:
        # ``talker_now`` is the seam the watchdog uses to compute hold age.
        # Set it to "5 seconds after started_at" so held_for = 5 s.
        monkeypatch.setattr(
            session_lock_watchdog,
            "talker_now",
            lambda: started_at + timedelta(seconds=5),
        )
        watchdog = SessionLockWatchdog(
            registry,  # type: ignore[arg-type]
            {repo_name: _repo_config(repo_name)},
            hold_deadline_seconds=10.0,
        )
        watchdog.run()
        assert session.force_release_calls == []
    finally:
        unregister_talker(repo_name, 42)


# ---------------------------------------------------------------------------
# Eviction path
# ---------------------------------------------------------------------------


def test_run_evicts_holder_past_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """A holder whose age exceeds the deadline triggers ``force_release``.

    The watchdog passes a reason string identifying the evicted tid
    and the actual hold time so the production log makes the
    eviction unambiguous in a postmortem.
    """
    repo_name = "FidoCanCode/home"
    session = _RecordingSession(repo_name)
    registry = _StubRegistry({repo_name: session})
    started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    register_talker(
        SessionTalker(
            repo_name=repo_name,
            thread_id=99999,
            kind="webhook",
            description="wedged synthesis turn",
            claude_pid=1,
            started_at=started_at,
        )
    )
    try:
        monkeypatch.setattr(
            session_lock_watchdog,
            "talker_now",
            lambda: started_at + timedelta(seconds=120),
        )
        watchdog = SessionLockWatchdog(
            registry,  # type: ignore[arg-type]
            {repo_name: _repo_config(repo_name)},
            hold_deadline_seconds=60.0,
        )
        watchdog.run()
    finally:
        unregister_talker(repo_name, 99999)

    assert len(session.force_release_calls) == 1
    reason = session.force_release_calls[0]
    assert "tid=99999" in reason
    assert "kind=webhook" in reason
    assert "wedged synthesis turn" in reason
    # Reason carries both the threshold (60s) and the observed hold (120s)
    # so a postmortem can tell what the watchdog actually saw.
    assert "60s" in reason
    assert "120s" in reason


def test_run_handles_multiple_repos_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Watchdog evaluates each configured repo separately on the same tick."""
    started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    home = _RecordingSession("FidoCanCode/home")
    confusio = _RecordingSession("rhencke/confusio")
    registry = _StubRegistry({"FidoCanCode/home": home, "rhencke/confusio": confusio})

    register_talker(
        SessionTalker(
            repo_name="FidoCanCode/home",
            thread_id=1,
            kind="webhook",
            description="long",
            claude_pid=1,
            started_at=started_at,
        )
    )
    register_talker(
        SessionTalker(
            repo_name="rhencke/confusio",
            thread_id=2,
            kind="worker",
            description="short",
            claude_pid=1,
            started_at=started_at + timedelta(seconds=55),
        )
    )
    try:
        monkeypatch.setattr(
            session_lock_watchdog,
            "talker_now",
            lambda: started_at + timedelta(seconds=60),
        )
        watchdog = SessionLockWatchdog(
            registry,  # type: ignore[arg-type]
            {
                "FidoCanCode/home": _repo_config("FidoCanCode/home"),
                "rhencke/confusio": _repo_config("rhencke/confusio"),
            },
            hold_deadline_seconds=10.0,
        )
        watchdog.run()
    finally:
        unregister_talker("FidoCanCode/home", 1)
        unregister_talker("rhencke/confusio", 2)

    # home's holder has been held 60 s (> 10 s) → evicted.
    assert len(home.force_release_calls) == 1
    # confusio's holder has only been held 5 s → spared.
    assert confusio.force_release_calls == []


# ---------------------------------------------------------------------------
# Session-type filtering
# ---------------------------------------------------------------------------


def test_run_skips_non_owned_session_returns() -> None:
    """The defensive ``isinstance(session, OwnedSession)`` filter prevents
    eviction attempts against a session type that does not implement
    :meth:`force_release` — important for forward compatibility with
    future provider sessions whose lifecycle differs from
    :class:`OwnedSession`.
    """

    class _NotAnOwnedSession:
        pass

    registry = _StubRegistry({"FidoCanCode/home": _NotAnOwnedSession()})  # type: ignore[dict-item]
    watchdog = SessionLockWatchdog(
        registry,  # type: ignore[arg-type]
        {"FidoCanCode/home": _repo_config("FidoCanCode/home")},
        hold_deadline_seconds=0.01,
    )
    assert watchdog.run() == 0


# ---------------------------------------------------------------------------
# Daemon thread
# ---------------------------------------------------------------------------


def test_start_thread_returns_running_daemon_that_invokes_run() -> None:
    """``start_thread`` returns a daemon thread whose loop body invokes
    :meth:`run` every *_interval* seconds.  Uses a tiny interval and an
    event-recording ``run`` override so the test can confirm the loop
    actually fires (covers the daemon's ``self.run()`` call site).
    """
    invoked = threading.Event()

    class _SignalingWatchdog(SessionLockWatchdog):
        def run(self) -> int:  # type: ignore[override]
            invoked.set()
            return 0

    watchdog = _SignalingWatchdog(
        _StubRegistry({}),  # type: ignore[arg-type]
        {},
        hold_deadline_seconds=999.0,
    )
    # Tiny interval so the daemon hits its first ``run`` quickly.  The
    # thread is a daemon — no join needed; process exit will reap it.
    t = watchdog.start_thread(_interval=0.01)
    assert t.daemon is True
    assert t.name == "session-lock-watchdog"
    assert invoked.wait(timeout=2.0), "watchdog daemon never invoked run()"


# ---------------------------------------------------------------------------
# Module-level convenience entry point
# ---------------------------------------------------------------------------


def test_module_level_run_evicts_past_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The module-level ``run`` helper composes ``SessionLockWatchdog`` and
    runs one tick — convenience entry point parallels ``watchdog.run``.
    """
    from fido.session_lock_watchdog import run as run_watchdog

    repo_name = "FidoCanCode/home"
    session = _RecordingSession(repo_name)
    registry = _StubRegistry({repo_name: session})
    started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    register_talker(
        SessionTalker(
            repo_name=repo_name,
            thread_id=7,
            kind="worker",
            description="wedged",
            claude_pid=1,
            started_at=started_at,
        )
    )
    try:
        monkeypatch.setattr(
            session_lock_watchdog,
            "talker_now",
            lambda: started_at + timedelta(seconds=2),
        )
        result = run_watchdog(
            registry,  # type: ignore[arg-type]
            {repo_name: _repo_config(repo_name)},
            hold_deadline_seconds=1.0,
        )
    finally:
        unregister_talker(repo_name, 7)

    assert result == 0
    assert len(session.force_release_calls) == 1
