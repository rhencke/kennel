"""Regression tests for kind-aware preempt in ClaudeSession.prompt (#637).

Uses the prompt's log output to distinguish "preempting worker" (cancel +
early control_request) from "queuing behind X holder" (no cancel).  Direct
inspection of stdin writes would conflate the early control_request with
:meth:`ClaudeSession._drain_to_boundary`'s own control_request from the
regular :meth:`send` path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

from kennel import claude as claude_mod
from kennel.claude import ClaudeSession, ClaudeTalker, _talker_now


def _make_session_proc(lines: list[str]) -> MagicMock:
    proc = MagicMock()
    proc.poll = MagicMock(return_value=None)
    proc.wait = MagicMock(return_value=0)
    proc.returncode = 0
    proc.stdin = MagicMock()
    proc.stdin.closed = False
    stdout = MagicMock()
    stdout.readline = MagicMock(side_effect=list(lines) + [""])
    proc.stdout = stdout
    proc.stderr = MagicMock()
    return proc


def _setup_session(tmp_path: Path) -> tuple[ClaudeSession, MagicMock]:
    system_file = tmp_path / "system.md"
    system_file.write_text("sys")
    proc = _make_session_proc(['{"type":"result","result":"reply"}\n'])
    proc.pid = 55555
    fake_popen = MagicMock(return_value=proc)
    fake_selector = MagicMock(return_value=([proc.stdout], [], []))
    session = ClaudeSession(
        system_file,
        work_dir=tmp_path,
        popen=fake_popen,
        selector=fake_selector,
        repo_name="owner/repo",
        model="claude-opus-4-6",
    )
    return session, proc


def _fake_talker(kind: str) -> ClaudeTalker:
    return ClaudeTalker(
        repo_name="owner/repo",
        thread_id=999_999,
        kind=kind,  # type: ignore[arg-type]
        description="fake",
        claude_pid=55555,
        started_at=_talker_now(),
    )


def test_webhook_preempting_worker_logs_preempting_and_fires_interrupt(
    tmp_path: Path, caplog, monkeypatch
) -> None:
    """Webhook caller + worker holds the lock + in-flight turn →
    logs ``preempting worker`` and calls ``_send_control_interrupt`` early
    (before entering the lock)."""
    session, _proc = _setup_session(tmp_path)
    session._in_turn = True
    monkeypatch.setattr(claude_mod, "get_talker", lambda _repo: _fake_talker("worker"))
    # Wrap _send_control_interrupt so we can see when it fired.
    called_before_lock = []
    real_interrupt = session._send_control_interrupt

    def tracking_interrupt() -> None:
        # When the prompt-level interrupt fires, the lock isn't held yet.
        called_before_lock.append(not session._lock.locked())
        real_interrupt()

    session._send_control_interrupt = tracking_interrupt  # type: ignore[method-assign]
    claude_mod.set_thread_kind("webhook")
    try:
        with caplog.at_level(logging.INFO, logger="kennel"):
            session.prompt("reply plz")
    finally:
        claude_mod.set_thread_kind(None)
        session.stop()
    assert "preempting worker" in caplog.text
    # At least one control_request must have fired while the lock was free
    # (the "early" interrupt from the preempt path, not the drain inside send).
    assert any(called_before_lock), (
        "webhook preempting worker must send control_request BEFORE "
        "acquiring the lock so claude aborts mid-tool"
    )


def test_webhook_does_not_preempt_another_webhook(
    tmp_path: Path, caplog, monkeypatch
) -> None:
    """Webhook caller + webhook currently holds the lock → no cancel, no
    early interrupt.  Logs ``queuing behind webhook`` instead."""
    session, _proc = _setup_session(tmp_path)
    session._in_turn = False  # avoid the send() drain path muddying the test
    monkeypatch.setattr(claude_mod, "get_talker", lambda _repo: _fake_talker("webhook"))
    interrupts_before_lock = []
    real_interrupt = session._send_control_interrupt

    def tracking_interrupt() -> None:
        interrupts_before_lock.append(not session._lock.locked())
        real_interrupt()

    session._send_control_interrupt = tracking_interrupt  # type: ignore[method-assign]
    claude_mod.set_thread_kind("webhook")
    try:
        with caplog.at_level(logging.INFO, logger="kennel"):
            session.prompt("reply plz")
    finally:
        claude_mod.set_thread_kind(None)
        session.stop()
    assert "queuing behind webhook" in caplog.text
    assert not any(interrupts_before_lock)


def test_worker_caller_does_not_preempt(tmp_path: Path, caplog, monkeypatch) -> None:
    """Worker caller (its own retry) never sets cancel or sends early
    control_request, even if another webhook currently holds the lock."""
    session, _proc = _setup_session(tmp_path)
    session._in_turn = False
    monkeypatch.setattr(claude_mod, "get_talker", lambda _repo: _fake_talker("webhook"))
    interrupts_before_lock = []
    real_interrupt = session._send_control_interrupt

    def tracking_interrupt() -> None:
        interrupts_before_lock.append(not session._lock.locked())
        real_interrupt()

    session._send_control_interrupt = tracking_interrupt  # type: ignore[method-assign]
    claude_mod.set_thread_kind("worker")
    try:
        with caplog.at_level(logging.INFO, logger="kennel"):
            session.prompt("keep working")
    finally:
        claude_mod.set_thread_kind(None)
        session.stop()
    assert "queuing behind webhook" in caplog.text
    assert not any(interrupts_before_lock)


def test_webhook_with_idle_session_skips_early_control_request(
    tmp_path: Path, caplog, monkeypatch
) -> None:
    """When no turn is in flight (_in_turn=False), webhook skips the early
    control_request but still logs ``preempting worker`` (cancel flag is
    still set so iter_events bails out on its next poll)."""
    session, _proc = _setup_session(tmp_path)
    session._in_turn = False
    monkeypatch.setattr(claude_mod, "get_talker", lambda _repo: _fake_talker("worker"))
    interrupts_before_lock = []
    real_interrupt = session._send_control_interrupt

    def tracking_interrupt() -> None:
        interrupts_before_lock.append(not session._lock.locked())
        real_interrupt()

    session._send_control_interrupt = tracking_interrupt  # type: ignore[method-assign]
    claude_mod.set_thread_kind("webhook")
    try:
        with caplog.at_level(logging.INFO, logger="kennel"):
            session.prompt("reply plz")
    finally:
        claude_mod.set_thread_kind(None)
        session.stop()
    assert "preempting worker" in caplog.text
    assert not any(interrupts_before_lock), (
        "no in-flight turn → no early control_request (would hang on idle "
        "subprocess waiting for a type=result that never comes)"
    )


def test_early_control_request_error_is_logged_not_fatal(
    tmp_path: Path, caplog, monkeypatch
) -> None:
    """If the early control_request write fails (BrokenPipe etc.), prompt
    logs a warning and still enters the lock for its own turn."""
    session, proc = _setup_session(tmp_path)
    session._in_turn = True
    monkeypatch.setattr(claude_mod, "get_talker", lambda _repo: _fake_talker("worker"))
    # First stdin.write (the early control_request) raises; later writes
    # succeed so the drain / send path can proceed.
    proc.stdin.write.side_effect = [
        BrokenPipeError("pipe closed"),
        None,
        None,
        None,
    ]
    claude_mod.set_thread_kind("webhook")
    try:
        with caplog.at_level(logging.WARNING, logger="kennel"):
            try:
                session.prompt("reply plz")
            except Exception:
                pass  # downstream send may raise once stdin is broken
    finally:
        claude_mod.set_thread_kind(None)
        session.stop()
    assert "early control_request failed" in caplog.text


def test_queuing_log_reports_no_holder_when_session_is_free(
    tmp_path: Path, caplog, monkeypatch
) -> None:
    """Webhook caller + no one currently holds the lock → logs
    ``queuing behind none`` (not ``preempting worker``)."""
    session, _proc = _setup_session(tmp_path)
    session._in_turn = False
    monkeypatch.setattr(claude_mod, "get_talker", lambda _repo: None)
    claude_mod.set_thread_kind("webhook")
    try:
        with caplog.at_level(logging.INFO, logger="kennel"):
            session.prompt("hi")
    finally:
        claude_mod.set_thread_kind(None)
        session.stop()
    assert "queuing behind none" in caplog.text


def test_set_thread_kind_roundtrip() -> None:
    claude_mod.set_thread_kind(None)
    assert claude_mod.current_thread_kind() == "worker"
    claude_mod.set_thread_kind("webhook")
    assert claude_mod.current_thread_kind() == "webhook"
    claude_mod.set_thread_kind("worker")
    assert claude_mod.current_thread_kind() == "worker"
    claude_mod.set_thread_kind(None)
    assert claude_mod.current_thread_kind() == "worker"
