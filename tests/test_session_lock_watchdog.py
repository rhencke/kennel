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
import time
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

    *outstanding_send_at* lets tests pin the no-reply clock; defaults
    to ``None`` (idle, no send awaiting reply).  The production
    session arms it on send and clears it on receive (#1709).
    """

    _repo_name: str | None

    def __init__(
        self,
        repo_name: str,
        *,
        outstanding_send_at: datetime | None = None,
    ) -> None:
        self._repo_name = repo_name
        self._init_handler_reentry()
        self._outstanding_send_at = outstanding_send_at
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
        no_reply_seconds=0.01,
    )
    assert watchdog.run() == 0


def test_run_skips_repo_without_active_holder() -> None:
    """A session with no registered talker (FSM Free) is left alone."""
    session = _RecordingSession("FidoCanCode/home")
    registry = _StubRegistry({"FidoCanCode/home": session})
    watchdog = SessionLockWatchdog(
        registry,  # type: ignore[arg-type]
        {"FidoCanCode/home": _repo_config("FidoCanCode/home")},
        no_reply_seconds=0.01,
    )
    # No register_talker call → get_talker returns None.
    watchdog.run()
    assert session.force_release_calls == []


def test_run_skips_holder_within_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A send that's still within the no-reply deadline is left alone."""
    repo_name = "FidoCanCode/home"
    sent_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    session = _RecordingSession(repo_name, outstanding_send_at=sent_at)
    registry = _StubRegistry({repo_name: session})
    register_talker(
        SessionTalker(
            repo_name=repo_name,
            thread_id=42,
            kind="worker",
            description="recent turn",
            subprocess_pid=1,
            started_at=sent_at,
        )
    )
    try:
        # ``talker_now`` is the seam the watchdog uses to compute the
        # silence duration.  Set it to "5 seconds after sent_at" so
        # the outstanding send has been waiting only 5 s.
        monkeypatch.setattr(
            session_lock_watchdog,
            "talker_now",
            lambda: sent_at + timedelta(seconds=5),
        )
        watchdog = SessionLockWatchdog(
            registry,  # type: ignore[arg-type]
            {repo_name: _repo_config(repo_name)},
            no_reply_seconds=10.0,
        )
        watchdog.run()
        assert session.force_release_calls == []
    finally:
        unregister_talker(repo_name, 42)


# ---------------------------------------------------------------------------
# Eviction path
# ---------------------------------------------------------------------------


def test_run_evicts_holder_past_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """A send whose silence exceeds the deadline triggers ``force_release``.

    The watchdog's reason string identifies the evicted tid and the
    silence duration so the production log makes the eviction
    unambiguous in a postmortem.
    """
    repo_name = "FidoCanCode/home"
    sent_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    session = _RecordingSession(repo_name, outstanding_send_at=sent_at)
    registry = _StubRegistry({repo_name: session})
    register_talker(
        SessionTalker(
            repo_name=repo_name,
            thread_id=99999,
            kind="webhook",
            description="wedged synthesis turn",
            subprocess_pid=1,
            started_at=sent_at,
        )
    )
    try:
        monkeypatch.setattr(
            session_lock_watchdog,
            "talker_now",
            lambda: sent_at + timedelta(seconds=120),
        )
        watchdog = SessionLockWatchdog(
            registry,  # type: ignore[arg-type]
            {repo_name: _repo_config(repo_name)},
            no_reply_seconds=60.0,
        )
        watchdog.run()
    finally:
        unregister_talker(repo_name, 99999)

    assert len(session.force_release_calls) == 1
    reason = session.force_release_calls[0]
    assert "tid=99999" in reason
    assert "kind=webhook" in reason
    assert "wedged synthesis turn" in reason
    # Reason carries both the threshold (60s) and the observed silence
    # (120s) so a postmortem can tell what the watchdog actually saw.
    assert "60s" in reason
    assert "120s" in reason


def test_run_handles_multiple_repos_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Watchdog evaluates each configured repo separately on the same tick."""
    sent_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    home = _RecordingSession("FidoCanCode/home", outstanding_send_at=sent_at)
    confusio = _RecordingSession(
        "rhencke/confusio", outstanding_send_at=sent_at + timedelta(seconds=55)
    )
    registry = _StubRegistry({"FidoCanCode/home": home, "rhencke/confusio": confusio})

    register_talker(
        SessionTalker(
            repo_name="FidoCanCode/home",
            thread_id=1,
            kind="webhook",
            description="long",
            subprocess_pid=1,
            started_at=sent_at,
        )
    )
    register_talker(
        SessionTalker(
            repo_name="rhencke/confusio",
            thread_id=2,
            kind="worker",
            description="short",
            subprocess_pid=1,
            started_at=sent_at + timedelta(seconds=55),
        )
    )
    try:
        monkeypatch.setattr(
            session_lock_watchdog,
            "talker_now",
            lambda: sent_at + timedelta(seconds=60),
        )
        watchdog = SessionLockWatchdog(
            registry,  # type: ignore[arg-type]
            {
                "FidoCanCode/home": _repo_config("FidoCanCode/home"),
                "rhencke/confusio": _repo_config("rhencke/confusio"),
            },
            no_reply_seconds=10.0,
        )
        watchdog.run()
    finally:
        unregister_talker("FidoCanCode/home", 1)
        unregister_talker("rhencke/confusio", 2)

    # home's send has been outstanding 60 s (> 10 s) → evicted.
    assert len(home.force_release_calls) == 1
    # confusio's send has only been outstanding 5 s → spared.
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
        no_reply_seconds=0.01,
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
        no_reply_seconds=999.0,
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
    sent_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    session = _RecordingSession(repo_name, outstanding_send_at=sent_at)
    registry = _StubRegistry({repo_name: session})
    register_talker(
        SessionTalker(
            repo_name=repo_name,
            thread_id=7,
            kind="worker",
            description="wedged",
            subprocess_pid=1,
            started_at=sent_at,
        )
    )
    try:
        monkeypatch.setattr(
            session_lock_watchdog,
            "talker_now",
            lambda: sent_at + timedelta(seconds=2),
        )
        result = run_watchdog(
            registry,  # type: ignore[arg-type]
            {repo_name: _repo_config(repo_name)},
            no_reply_seconds=1.0,
        )
    finally:
        unregister_talker(repo_name, 7)

    assert result == 0
    assert len(session.force_release_calls) == 1


# ---------------------------------------------------------------------------
# Send/receive semantics (#1709): receive disarms the clock; multi-send
# restarts it; idle never trips.
# ---------------------------------------------------------------------------


def test_idle_session_with_no_outstanding_send_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session whose ``outstanding_send_at`` is ``None`` (idle, or its
    last send already received an ACK) is left alone forever — even with
    a registered talker present and ``talker_now`` arbitrarily far ahead.
    """
    repo_name = "FidoCanCode/home"
    session = _RecordingSession(repo_name)  # outstanding_send_at=None
    registry = _StubRegistry({repo_name: session})
    started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    register_talker(
        SessionTalker(
            repo_name=repo_name,
            thread_id=42,
            kind="worker",
            description="lock held but no send pending",
            subprocess_pid=1,
            started_at=started_at,
        )
    )
    try:
        monkeypatch.setattr(
            session_lock_watchdog,
            "talker_now",
            lambda: started_at + timedelta(hours=24),
        )
        watchdog = SessionLockWatchdog(
            registry,  # type: ignore[arg-type]
            {repo_name: _repo_config(repo_name)},
            no_reply_seconds=60.0,
        )
        watchdog.run()
    finally:
        unregister_talker(repo_name, 42)

    assert session.force_release_calls == [], (
        "watchdog must not fire when no send is awaiting a reply"
    )


def test_mark_received_clears_outstanding_send() -> None:
    """The session's send / receive helpers compose:

    1. ``_mark_send_outstanding()`` arms the clock.
    2. ``_mark_received()`` disarms it.

    After a receive the watchdog sees ``outstanding_send_at == None``
    and skips, even though a send was outstanding moments earlier.
    """
    session = _RecordingSession("FidoCanCode/home")
    assert session.outstanding_send_at is None
    session._mark_send_outstanding()  # noqa: SLF001
    assert session.outstanding_send_at is not None
    session._mark_received()  # noqa: SLF001
    assert session.outstanding_send_at is None


def test_no_eviction_when_outstanding_set_but_no_talker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stale ``outstanding_send_at`` without a current talker must not
    fire eviction (#1710 codex P2).  The release path normally clears
    the timestamp, but this gate is the safety net for any future
    race where a stale armed value survives lock release."""
    repo_name = "FidoCanCode/home"
    sent_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    session = _RecordingSession(repo_name, outstanding_send_at=sent_at)
    registry = _StubRegistry({repo_name: session})
    # No register_talker call — get_talker(repo_name) returns None.
    monkeypatch.setattr(
        session_lock_watchdog,
        "talker_now",
        lambda: sent_at + timedelta(seconds=3600),
    )
    watchdog = SessionLockWatchdog(
        registry,  # type: ignore[arg-type]
        {repo_name: _repo_config(repo_name)},
        no_reply_seconds=60.0,
    )
    watchdog.run()
    assert session.force_release_calls == [], (
        "watchdog must not evict when no talker is currently registered"
    )


def test_force_release_clears_outstanding_send_at() -> None:
    """``force_release`` clears ``outstanding_send_at`` before any new
    acquire can happen (#1710 codex round 2 P1).

    The evicted holder's eventual ``__exit__`` skips the normal
    ``_fsm_release`` via the ``_evicted_tids`` guard, so the clear
    inside ``_fsm_release`` doesn't run for evicted holders.  Without
    a clear in ``force_release`` itself, a new acquirer would inherit
    the prior turn's stale timestamp and the watchdog could evict
    them before they sent anything.
    """
    # Build a real OwnedSession in OwnedByWorker so ForceRelease has
    # somewhere to evict from.
    session = _RecordingSession("FidoCanCode/home")
    session._fsm_acquire_worker()  # noqa: SLF001
    session._mark_send_outstanding()  # noqa: SLF001
    assert session.outstanding_send_at is not None
    # Override the recording stub's force_release to call the real one
    # (the recorder otherwise short-circuits to just append to its log).
    OwnedSession.force_release(session, reason="test wedge")  # type: ignore[arg-type]
    assert session.outstanding_send_at is None, (
        "force_release must disarm the no-reply clock so a "
        "freshly-acquired successor isn't evicted on stale silence"
    )


def test_release_clears_outstanding_send_at(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_fsm_release`` clears ``outstanding_send_at`` so a stale armed
    timestamp from an aborted turn never survives lock release (#1710
    codex P2)."""
    # Build a real OwnedSession with the FSM primed for a worker hold.
    session = _RecordingSession("FidoCanCode/home")
    session._mark_send_outstanding()  # noqa: SLF001
    assert session.outstanding_send_at is not None
    # Drive the FSM to OwnedByWorker so _fsm_release accepts WorkerRelease.
    session._fsm_acquire_worker()  # noqa: SLF001
    session._fsm_release()  # noqa: SLF001
    assert session.outstanding_send_at is None, "release must disarm the no-reply clock"


def test_multi_send_restarts_no_reply_clock() -> None:
    """Calling ``_mark_send_outstanding`` twice in a row resets the
    clock — only the most recent unanswered send matters."""
    session = _RecordingSession("FidoCanCode/home")
    session._mark_send_outstanding()  # noqa: SLF001
    first = session.outstanding_send_at
    assert first is not None
    # Force a measurable gap so the second timestamp is strictly later.
    time.sleep(0.001)
    session._mark_send_outstanding()  # noqa: SLF001
    second = session.outstanding_send_at
    assert second is not None
    assert second > first
