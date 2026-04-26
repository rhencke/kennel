"""Tests for ClaudeSession.hold_for_handler (#658)."""

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fido import provider
from fido.claude import ClaudeSession
from fido.provider import SessionTalker, talker_now


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
    provider.set_thread_kind("webhook")
    try:
        with session.hold_for_handler():
            talker = provider.get_talker("owner/repo")
            assert talker is not None
            assert talker.kind == "webhook"
            assert session._lock._is_owned()  # type: ignore[attr-defined]
        # After exit, talker unregistered, lock released.
        assert provider.get_talker("owner/repo") is None
        assert not session._lock._is_owned()  # type: ignore[attr-defined]
    finally:
        provider.set_thread_kind(None)
        session.stop()


def test_nested_with_inside_hold_does_not_double_register(
    tmp_path: Path, monkeypatch
) -> None:
    """Re-entering ``with session:`` inside ``hold_for_handler`` must not
    attempt a second talker registration (would raise SessionLeakError)."""
    session = _setup_session(tmp_path)
    register_calls = []
    real_register = provider.register_talker

    def counting_register(talker):
        register_calls.append(talker.kind)
        real_register(talker)

    monkeypatch.setattr(provider, "register_talker", counting_register)
    provider.set_thread_kind("webhook")
    try:
        with session.hold_for_handler():
            assert len(register_calls) == 1  # outer entry registered once
            with session:  # nested re-entry
                assert len(register_calls) == 1  # not re-registered
            assert len(register_calls) == 1  # still registered after inner exit
        # After outer exit, unregistered.
        assert provider.get_talker("owner/repo") is None
    finally:
        provider.set_thread_kind(None)
        session.stop()


def test_hold_preempt_fires_cancel_when_worker_holds(
    tmp_path: Path, monkeypatch
) -> None:
    """hold_for_handler(preempt_worker=True) fires _fire_worker_cancel iff
    the current lock holder is a worker and the caller is a webhook."""
    session = _setup_session(tmp_path)

    def fake_talker(kind: str) -> SessionTalker:
        return SessionTalker(
            repo_name="owner/repo",
            thread_id=999_999,
            kind=kind,  # type: ignore[arg-type]
            description="fake",
            claude_pid=55555,
            started_at=talker_now(),
        )

    monkeypatch.setattr(provider, "get_talker", lambda _repo: fake_talker("worker"))
    cancel_calls = []
    monkeypatch.setattr(session, "_fire_worker_cancel", lambda: cancel_calls.append(1))
    provider.set_thread_kind("webhook")
    try:
        with session.hold_for_handler(preempt_worker=True):
            pass
    finally:
        provider.set_thread_kind(None)
        session.stop()
    assert cancel_calls == [1]


def test_hold_preempt_no_fire_when_no_worker_holder(
    tmp_path: Path, monkeypatch
) -> None:
    """preempt_worker=True with no current holder — try_preempt_worker returns
    (False, None) and no cancel fires.  Exercises the ``else`` branch of the
    new preempt outcome logging in hold_for_handler (#955)."""
    session = _setup_session(tmp_path)
    # No holder registered — try_preempt_worker sees current_kind=None.
    monkeypatch.setattr(provider, "get_talker", lambda _repo: None)
    cancel_calls = []
    monkeypatch.setattr(session, "_fire_worker_cancel", lambda: cancel_calls.append(1))
    provider.set_thread_kind("webhook")
    try:
        with session.hold_for_handler(preempt_worker=True):
            pass
    finally:
        provider.set_thread_kind(None)
        session.stop()
    assert cancel_calls == []


def test_hold_preempt_skipped_when_no_preempt_worker_flag(
    tmp_path: Path, monkeypatch
) -> None:
    """Default preempt_worker=False — no cancel fires even with a worker
    holder."""
    session = _setup_session(tmp_path)

    def fake_talker(kind: str) -> SessionTalker:
        return SessionTalker(
            repo_name="owner/repo",
            thread_id=999_999,
            kind=kind,  # type: ignore[arg-type]
            description="fake",
            claude_pid=55555,
            started_at=talker_now(),
        )

    monkeypatch.setattr(provider, "get_talker", lambda _repo: fake_talker("worker"))
    cancel_calls = []
    monkeypatch.setattr(session, "_fire_worker_cancel", lambda: cancel_calls.append(1))
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
        provider.set_thread_kind("webhook")
        try:
            with session.hold_for_handler():
                holder_entered.set()
                release_holder.wait(timeout=5.0)
        finally:
            provider.set_thread_kind(None)

    def other() -> None:
        provider.set_thread_kind("worker")
        try:
            with session:
                other_acquired.set()
            other_finished.set()
        finally:
            provider.set_thread_kind(None)

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


def test_webhook_preempts_worker_mid_turn(tmp_path: Path) -> None:
    """End-to-end: webhook calling hold_for_handler(preempt_worker=True)
    wakes a worker that is blocked inside iter_events, causing the worker to
    exit its turn and release the lock so the webhook can acquire it (#955)."""
    import time

    system_file = tmp_path / "system.md"
    system_file.write_text("sys")
    proc = MagicMock()
    proc.pid = 55555
    proc.poll = MagicMock(return_value=None)
    proc.wait = MagicMock(return_value=0)
    proc.returncode = 0
    proc.stdin = MagicMock()
    proc.stdin.closed = False
    proc.stdout = MagicMock()
    proc.stderr = MagicMock()

    # Worker turn: readline blocks until cancel fires, then returns EOF.
    worker_blocked = threading.Event()
    cancel_received = threading.Event()

    def blocking_readline() -> str:
        worker_blocked.set()
        cancel_received.wait(timeout=5.0)
        return ""  # EOF — worker exits iter_events

    proc.stdout.readline = MagicMock(side_effect=blocking_readline)

    # Selector: immediately returns stdout as ready so iter_events calls readline.
    session = ClaudeSession(
        system_file,
        work_dir=tmp_path,
        popen=MagicMock(return_value=proc),
        selector=MagicMock(return_value=([proc.stdout], [], [])),
        repo_name="owner/repo",
        model="claude-opus-4-6",
    )

    worker_in_turn = threading.Event()
    worker_done = threading.Event()
    webhook_acquired = threading.Event()
    webhook_done = threading.Event()

    def worker() -> None:
        provider.set_thread_kind("worker")
        try:
            with session:
                worker_in_turn.set()
                # consume_until_result drives iter_events; readline blocks
                session.consume_until_result()
            worker_done.set()
        finally:
            provider.set_thread_kind(None)

    def webhook() -> None:
        provider.set_thread_kind("webhook")
        try:
            # Wait until worker is actually blocked, then preempt.
            worker_blocked.wait(timeout=2.0)
            # Signal readline to unblock after cancel fires.
            original_fire = session._fire_worker_cancel

            def fire_and_unblock() -> None:
                original_fire()
                cancel_received.set()

            session._fire_worker_cancel = fire_and_unblock  # type: ignore[method-assign]
            t_start = time.monotonic()
            with session.hold_for_handler(preempt_worker=True):
                webhook_acquired.set()
                elapsed = time.monotonic() - t_start
                assert elapsed < 2.0, (
                    f"webhook took too long to acquire: {elapsed:.2f}s"
                )
            webhook_done.set()
        finally:
            provider.set_thread_kind(None)

    t_worker = threading.Thread(target=worker, daemon=True)
    t_worker.start()
    worker_in_turn.wait(timeout=2.0)

    t_webhook = threading.Thread(target=webhook, daemon=True)
    t_webhook.start()

    assert webhook_acquired.wait(timeout=5.0), "webhook never acquired lock"
    assert webhook_done.wait(timeout=5.0)
    assert worker_done.wait(timeout=5.0)
    t_worker.join(timeout=2.0)
    t_webhook.join(timeout=2.0)
    session.stop()


def test_handler_prompt_runs_after_preempt_does_not_inherit_cancel(
    tmp_path: Path, monkeypatch
) -> None:
    """Post-#979: after hold_for_handler(preempt_worker=True) fires the
    cancel signal, the handler's own prompt() must run to completion.  In
    the new design the prior turn's boundary is drained inside iter_events
    itself (cancel no longer breaks early), so by the time the handler
    enters the stream is clean.  The cancel signal that was set for the
    previous holder is consumed at the start of the handler's iter_events
    via the existing ``_preempt_pending`` gate."""
    session = _setup_session(tmp_path)

    def fake_talker(kind: str) -> SessionTalker:
        return SessionTalker(
            repo_name="owner/repo",
            thread_id=999_999,
            kind=kind,  # type: ignore[arg-type]
            description="fake",
            claude_pid=55555,
            started_at=talker_now(),
        )

    monkeypatch.setattr(provider, "get_talker", lambda _repo: fake_talker("worker"))

    # Pipe contains exactly the handler's own response — no stale events
    # from the prior turn (those were drained inside iter_events when the
    # worker turn closed cleanly on type=result).
    proc = _make_session_proc(['{"type":"result","result":"triage-reply"}\n'])
    proc.pid = 55555
    monkeypatch.setattr(session, "_proc", proc)
    monkeypatch.setattr(
        session, "_selector", MagicMock(return_value=([proc.stdout], [], []))
    )

    provider.set_thread_kind("webhook")
    try:
        with session.hold_for_handler(preempt_worker=True):
            # _fire_worker_cancel set _cancel + _preempt_pending.  Handler's
            # first prompt() must actually send and read its own response.
            result = session.prompt("triage this please")
    finally:
        provider.set_thread_kind(None)
        session.stop()

    # The handler must have written its user message to stdin (atomicity
    # guaranteed by _stdin_lock — see #979).
    write_calls = [c.args[0] for c in proc.stdin.write.call_args_list]
    user_writes = [w for w in write_calls if "triage this please" in w]
    assert user_writes, f"handler prompt never wrote its message — writes={write_calls}"
    assert result == "triage-reply", f"handler prompt got wrong result — got {result!r}"


def test_inner_prompt_preserves_queued_webhook_pending_flag(
    tmp_path: Path,
) -> None:
    """Regression for #1017: _preempt_pending set by a second webhook
    queuing behind an active hold_for_handler must survive inner prompt()
    calls made by the holder.

    The old code had two paths that unconditionally cleared _preempt_pending:
    (1) reentrant __enter__ inside hold_for_handler, and (2) prompt()'s
    finally block.  Either path wiped the second webhook's pending signal
    while the first webhook's hold was still active.  When the hold released,
    the worker saw _preempt_pending=False and won the lock over the queued
    webhook, leaving the reviewer's comment unanswered."""
    session = _setup_session(tmp_path)
    provider.set_thread_kind("webhook")
    try:
        with session.hold_for_handler(preempt_worker=False):
            # Simulate webhook B queuing behind this hold_for_handler.
            session._signal_pending_preempt()
            assert session._preempt_pending.is_set(), "sanity: pending set"
            # Inner prompt() — old code cleared _preempt_pending here.
            session.prompt("triage this issue")
            # After returning, pending must still be set so the worker
            # blocks in __enter__'s wait_for and yields to webhook B.
            assert session._preempt_pending.is_set(), (
                "_preempt_pending cleared by inner prompt() inside "
                "hold_for_handler — worker would win lock over queued "
                "webhook (bug #1017 regression)"
            )
    finally:
        provider.set_thread_kind(None)
        session.stop()


def test_queued_webhook_acquires_lock_before_worker_after_inner_prompt(
    tmp_path: Path,
) -> None:
    """Regression for #1017 (threading): webhook B queuing behind an active
    hold_for_handler acquires the lock before the worker, even after the
    holder makes inner prompt() calls while B's pending signal is live.

    Timing is made deterministic by ensuring _preempt_pending is already set
    before the worker thread enters __enter__ — this forces the worker into
    _preempt_cond.wait_for() rather than racing directly on _lock.acquire()."""
    session = _setup_session(tmp_path)

    webhook_a_holding = threading.Event()
    webhook_b_queued = threading.Event()
    order: list[str] = []
    errors: list[Exception] = []

    def webhook_a() -> None:
        provider.set_thread_kind("webhook")
        try:
            with session.hold_for_handler(preempt_worker=False):
                webhook_a_holding.set()
                # Wait until webhook B has queued (pending set) before
                # making the inner prompt — this is the ordering that
                # triggers the bug: B sets pending, then A's inner prompt
                # used to clear it.
                assert webhook_b_queued.wait(timeout=2.0), "webhook_b_queued timed out"
                session.prompt("inner turn from webhook A")
        except Exception as exc:
            errors.append(exc)
        finally:
            provider.set_thread_kind(None)

    def webhook_b() -> None:
        provider.set_thread_kind("webhook")
        try:
            assert webhook_a_holding.wait(timeout=2.0), "webhook_a_holding timed out"
            # Queue behind webhook A — sets _preempt_pending so the worker
            # yields priority.
            session._signal_pending_preempt()
            webhook_b_queued.set()
            # Now block waiting for webhook A's hold to release.
            with session:
                order.append("webhook_b")
        except Exception as exc:
            errors.append(exc)
        finally:
            provider.set_thread_kind(None)

    def worker() -> None:
        provider.set_thread_kind("worker")
        try:
            # Wait until B has set _preempt_pending before entering __enter__.
            # This guarantees the worker hits _preempt_cond.wait_for() instead
            # of racing on _lock.acquire() — making the handoff deterministic.
            assert webhook_b_queued.wait(timeout=2.0), "webhook_b_queued timed out"
            with session:
                order.append("worker")
        except Exception as exc:
            errors.append(exc)
        finally:
            provider.set_thread_kind(None)

    t_a = threading.Thread(target=webhook_a, daemon=True)
    t_b = threading.Thread(target=webhook_b, daemon=True)
    t_w = threading.Thread(target=worker, daemon=True)

    t_a.start()
    t_b.start()
    t_w.start()

    t_a.join(timeout=5.0)
    t_b.join(timeout=5.0)
    t_w.join(timeout=5.0)

    assert not errors, f"thread errors: {errors}"
    assert len(order) == 2, f"not all threads completed: {order}"
    assert order[0] == "webhook_b", (
        f"worker won lock over queued webhook — got order {order} "
        f"(bug #1017 regression)"
    )
    session.stop()


def test_hold_reraises_leak_error_and_releases_lock(
    tmp_path: Path, monkeypatch
) -> None:
    """If register_talker raises SessionLeakError inside hold, the lock must
    be released before the exception propagates so we don't deadlock."""
    session = _setup_session(tmp_path)
    # Pre-register a talker for the same repo from a different thread id so
    # the hold's register_talker collides.
    provider.register_talker(
        SessionTalker(
            repo_name="owner/repo",
            thread_id=111_111,  # different tid — triggers leak
            kind="worker",
            description="squatter",
            claude_pid=0,
            started_at=talker_now(),
        )
    )
    provider.set_thread_kind("webhook")
    try:
        with pytest.raises(provider.SessionLeakError):
            with session.hold_for_handler():
                pass  # should not reach here
        # Lock must be released so other threads can acquire.
        acquired = session._lock.acquire(blocking=False)
        assert acquired
        session._lock.release()
    finally:
        provider.set_thread_kind(None)
        provider.unregister_talker("owner/repo", 111_111)
        session.stop()
