"""Tests for ClaudeSession.hold_for_handler (#658)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

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


def _setup_session(tmp_path: Path, repo: str = "owner/repo") -> ClaudeSession:
    system_file = tmp_path / "system.md"
    system_file.write_text("sys")
    proc = _make_session_proc(['{"type":"result","result":"reply"}\n'])
    proc.pid = 55555
    return ClaudeSession(
        system_file,
        work_dir=tmp_path,
        popen=MagicMock(return_value=proc),
        selector=MagicMock(return_value=([proc.stdout], [], [])),
        repo_name=repo,
        model="claude-opus-4-6",
    )


def test_hold_acquires_lock_and_registers_talker(tmp_path: Path) -> None:
    session = _setup_session(tmp_path)
    claude_mod.set_thread_kind("webhook")
    try:
        with session.hold_for_handler():
            talker = claude_mod.get_talker("owner/repo")
            assert talker is not None
            assert talker.kind == "webhook"
            assert session._lock._is_owned()  # type: ignore[attr-defined]
        # After exit, talker unregistered, lock released.
        assert claude_mod.get_talker("owner/repo") is None
        assert not session._lock._is_owned()  # type: ignore[attr-defined]
    finally:
        claude_mod.set_thread_kind(None)
        session.stop()


def test_nested_with_inside_hold_does_not_double_register(
    tmp_path: Path, monkeypatch
) -> None:
    """Re-entering ``with session:`` inside ``hold_for_handler`` must not
    attempt a second talker registration (would raise ClaudeLeakError)."""
    session = _setup_session(tmp_path)
    register_calls = []
    real_register = claude_mod.register_talker

    def counting_register(talker):
        register_calls.append(talker.kind)
        real_register(talker)

    monkeypatch.setattr(claude_mod, "register_talker", counting_register)
    claude_mod.set_thread_kind("webhook")
    try:
        with session.hold_for_handler():
            assert len(register_calls) == 1  # outer entry registered once
            with session:  # nested re-entry
                assert len(register_calls) == 1  # not re-registered
            assert len(register_calls) == 1  # still registered after inner exit
        # After outer exit, unregistered.
        assert claude_mod.get_talker("owner/repo") is None
    finally:
        claude_mod.set_thread_kind(None)
        session.stop()


def test_hold_preempt_fires_cancel_when_worker_holds(
    tmp_path: Path, monkeypatch
) -> None:
    """hold_for_handler(preempt_worker=True) fires _fire_worker_cancel iff
    the current lock holder is a worker and the caller is a webhook."""
    session = _setup_session(tmp_path)

    def fake_talker(kind: str) -> ClaudeTalker:
        return ClaudeTalker(
            repo_name="owner/repo",
            thread_id=999_999,
            kind=kind,  # type: ignore[arg-type]
            description="fake",
            claude_pid=55555,
            started_at=_talker_now(),
        )

    monkeypatch.setattr(
        claude_mod, "get_talker", lambda _repo: fake_talker("worker")
    )
    cancel_calls = []
    monkeypatch.setattr(
        session, "_fire_worker_cancel", lambda: cancel_calls.append(1)
    )
    claude_mod.set_thread_kind("webhook")
    try:
        with session.hold_for_handler(preempt_worker=True):
            pass
    finally:
        claude_mod.set_thread_kind(None)
        session.stop()
    assert cancel_calls == [1]


def test_hold_preempt_skipped_when_no_preempt_worker_flag(
    tmp_path: Path, monkeypatch
) -> None:
    """Default preempt_worker=False — no cancel fires even with a worker
    holder."""
    session = _setup_session(tmp_path)

    def fake_talker(kind: str) -> ClaudeTalker:
        return ClaudeTalker(
            repo_name="owner/repo",
            thread_id=999_999,
            kind=kind,  # type: ignore[arg-type]
            description="fake",
            claude_pid=55555,
            started_at=_talker_now(),
        )

    monkeypatch.setattr(
        claude_mod, "get_talker", lambda _repo: fake_talker("worker")
    )
    cancel_calls = []
    monkeypatch.setattr(
        session, "_fire_worker_cancel", lambda: cancel_calls.append(1)
    )
    try:
        with session.hold_for_handler():  # preempt_worker default False
            pass
    finally:
        session.stop()
    assert cancel_calls == []


def test_other_thread_blocks_while_held(tmp_path: Path) -> None:
    """A different thread trying to ``with session:`` while another thread
    is inside hold_for_handler must block until the hold exits (#658 —
    that is the whole point of holding the lock across turns)."""
    session = _setup_session(tmp_path)
    holder_entered = threading.Event()
    release_holder = threading.Event()
    other_acquired = threading.Event()
    other_finished = threading.Event()

    def holder() -> None:
        claude_mod.set_thread_kind("webhook")
        try:
            with session.hold_for_handler():
                holder_entered.set()
                release_holder.wait(timeout=5.0)
        finally:
            claude_mod.set_thread_kind(None)

    def other() -> None:
        claude_mod.set_thread_kind("worker")
        try:
            with session:
                other_acquired.set()
            other_finished.set()
        finally:
            claude_mod.set_thread_kind(None)

    t1 = threading.Thread(target=holder, daemon=True)
    t1.start()
    holder_entered.wait(timeout=2.0)
    t2 = threading.Thread(target=other, daemon=True)
    t2.start()
    # Give t2 a chance — it must NOT have acquired the lock yet.
    assert not other_acquired.wait(timeout=0.1)
    # Release the holder; t2 should now acquire.
    release_holder.set()
    t1.join(timeout=2.0)
    assert other_acquired.wait(timeout=2.0), "other thread never acquired"
    assert other_finished.wait(timeout=2.0)
    session.stop()


def test_hold_reraises_leak_error_and_releases_lock(
    tmp_path: Path, monkeypatch
) -> None:
    """If register_talker raises ClaudeLeakError inside hold, the lock must
    be released before the exception propagates so we don't deadlock."""
    session = _setup_session(tmp_path)
    # Pre-register a talker for the same repo from a different thread id so
    # the hold's register_talker collides.
    claude_mod.register_talker(
        ClaudeTalker(
            repo_name="owner/repo",
            thread_id=111_111,  # different tid — triggers leak
            kind="worker",
            description="squatter",
            claude_pid=0,
            started_at=_talker_now(),
        )
    )
    claude_mod.set_thread_kind("webhook")
    try:
        with pytest.raises(claude_mod.ClaudeLeakError):
            with session.hold_for_handler():
                pass  # should not reach here
        # Lock must be released so other threads can acquire.
        acquired = session._lock.acquire(blocking=False)
        assert acquired
        session._lock.release()
    finally:
        claude_mod.set_thread_kind(None)
        claude_mod.unregister_talker("owner/repo", 111_111)
        session.stop()
