"""Tests for the FSM-driven lock coordination in OwnedSession.

Covers the FSM coordination methods (``_fsm_acquire_worker`` /
``_fsm_acquire_handler`` / ``_fsm_release`` / ``force_release``) —
the authoritative lock coordinator extracted from
``models/session_lock.v``.  Workers block until Free with no queued
handlers; handlers block until Free (FIFO queue for handler-on-handler
ordering); ``force_release`` evicts a wedged holder regardless of
state (liveness escape hatch added for #1377).

Proved theorems being guarded:
  ``no_dual_ownership``        — acquisition rejected when session already owned
  ``release_only_by_owner``    — release rejected when wrong owner or Free
  ``force_release_to_free``    — ForceRelease always lands in Free
  ``unconditional_liveness``   — a *single* event (ForceRelease) drives every
                                 state to Free (∃ev. ∀s. ...) — strong form
                                 the watchdog relies on, fired without first
                                 inspecting FSM state
  ``every_state_reaches_free`` — weaker form (∀s. ∃ev. ...) — each state has
                                 at least one event reaching Free, citing
                                 voluntary releases for owned states
  ``owned_state_exit_paths``   — only WorkerRelease/Preempt/ForceRelease exit
                                 OwnedByWorker; only HandlerRelease/ForceRelease
                                 exit OwnedByHandler
"""

import threading

import pytest

from fido.provider import (
    OwnedSession,
    SessionTalker,
    get_talker,
    register_talker,
    talker_now,
    unregister_talker,
)
from fido.rocq.transition import (
    Free,
    OwnedByHandler,
    OwnedByWorker,
)


class _FakeSession(OwnedSession):
    """Minimal OwnedSession subclass for unit-testing coordination behaviour.

    The FSM logic lives entirely in the base class; this stub exists
    only to satisfy the class hierarchy and the ``_repo_name`` attribute
    that ``OwnedSession`` requires.  No real lock or subprocess needed.
    """

    _repo_name: str | None = None

    def __init__(self) -> None:
        self._init_handler_reentry()

    def _fire_worker_cancel(self) -> None:  # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def test_fsm_state_starts_free() -> None:
    """FSM initialises to Free — nobody holds the session."""
    session = _FakeSession()
    assert isinstance(session._fsm_state, Free)


# ---------------------------------------------------------------------------
# FSM coordination: _fsm_acquire_worker
# ---------------------------------------------------------------------------


def test_fsm_acquire_worker_immediate_when_free() -> None:
    """Worker acquires immediately when state is Free and queue is empty."""
    session = _FakeSession()
    session._fsm_acquire_worker()
    assert isinstance(session._fsm_state, OwnedByWorker)


def test_fsm_acquire_worker_waits_until_handler_releases() -> None:
    """Worker blocks while OwnedByHandler and acquires after release."""
    session = _FakeSession()
    session._fsm_acquire_handler()  # handler holds; worker will block

    worker_acquired = threading.Event()

    def worker() -> None:
        session._fsm_acquire_worker()
        worker_acquired.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    # Worker must be waiting — it cannot acquire while handler holds.
    assert not worker_acquired.wait(timeout=0.1), "worker should still be waiting"
    # Release the handler → worker can now acquire.
    session._fsm_release()
    assert worker_acquired.wait(timeout=5.0), "worker never acquired"
    assert isinstance(session._fsm_state, OwnedByWorker)
    session._fsm_release()
    t.join(timeout=1.0)


def test_fsm_acquire_worker_yields_to_queued_handler() -> None:
    """Worker waits even when Free if a handler is in the queue.

    Uses direct queue injection so the ordering is deterministic.
    """
    session = _FakeSession()
    # Inject a fake handler waiter directly so we control the queue.
    fake_handler_waiter = threading.Event()
    with session._fsm_cond:
        session._handler_queue.append(fake_handler_waiter)

    # Worker should not be able to acquire (queue non-empty even though Free).
    worker_acquired = threading.Event()

    def worker() -> None:
        session._fsm_acquire_worker()
        worker_acquired.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    assert not worker_acquired.wait(timeout=0.1), (
        "worker should yield to queued handler"
    )

    # Remove the fake handler from the queue and notify so the worker re-checks.
    with session._fsm_cond:
        session._handler_queue.clear()
        session._fsm_cond.notify_all()

    assert worker_acquired.wait(timeout=5.0), (
        "worker never acquired after queue drained"
    )
    assert isinstance(session._fsm_state, OwnedByWorker)
    session._fsm_release()
    t.join(timeout=1.0)


# ---------------------------------------------------------------------------
# FSM coordination: _fsm_acquire_handler
# ---------------------------------------------------------------------------


def test_fsm_acquire_handler_immediate_when_free() -> None:
    """Handler acquires immediately when state is Free."""
    session = _FakeSession()
    session._fsm_acquire_handler()
    assert isinstance(session._fsm_state, OwnedByHandler)


def test_fsm_acquire_handler_queues_and_acquires_after_release() -> None:
    """Handler queues when occupied and acquires when the holder releases."""
    session = _FakeSession()
    session._fsm_acquire_handler()  # first handler holds

    acquired = threading.Event()

    def handler2() -> None:
        session._fsm_acquire_handler()  # will queue
        acquired.set()

    t = threading.Thread(target=handler2, daemon=True)
    t.start()
    # Second handler must not acquire until first releases.
    assert not acquired.wait(timeout=0.1), "handler2 should be queued"
    # Release first handler → handler2 acquires.
    session._fsm_release()
    assert acquired.wait(timeout=5.0), "handler2 never acquired"
    assert isinstance(session._fsm_state, OwnedByHandler)
    session._fsm_release()
    t.join(timeout=1.0)


# ---------------------------------------------------------------------------
# FSM coordination: _fsm_release
# ---------------------------------------------------------------------------


def test_fsm_release_worker_no_queue() -> None:
    """Worker release with no queued handlers transitions to Free."""
    session = _FakeSession()
    session._fsm_acquire_worker()
    session._fsm_release()
    assert isinstance(session._fsm_state, Free)


def test_fsm_release_handler_no_queue() -> None:
    """Handler release with no queued handlers transitions to Free."""
    session = _FakeSession()
    session._fsm_acquire_handler()
    session._fsm_release()
    assert isinstance(session._fsm_state, Free)


def test_fsm_release_from_free_raises() -> None:
    """Releasing from Free raises RuntimeError (release_only_by_owner)."""
    session = _FakeSession()
    with pytest.raises(RuntimeError, match="release_only_by_owner"):
        session._fsm_release()


def test_fsm_release_handler_with_queued_handler_skips_free() -> None:
    """Handler release with a queued handler hands ownership directly."""
    session = _FakeSession()
    session._fsm_acquire_handler()  # first handler holds

    fake_waiter = threading.Event()
    with session._fsm_cond:
        session._handler_queue.append(fake_waiter)

    session._fsm_release()  # should hand to fake_waiter, not go through Free

    assert fake_waiter.is_set(), "queued handler was not signalled"
    assert isinstance(session._fsm_state, OwnedByHandler)
    # Clean up by releasing the "fake" handler's ownership.
    session._fsm_release()
    assert isinstance(session._fsm_state, Free)


def test_fsm_release_worker_with_queued_handler() -> None:
    """Worker release with a queued handler hands ownership to the handler."""
    session = _FakeSession()
    session._fsm_acquire_worker()

    fake_waiter = threading.Event()
    with session._fsm_cond:
        session._handler_queue.append(fake_waiter)

    session._fsm_release()

    assert fake_waiter.is_set()
    assert isinstance(session._fsm_state, OwnedByHandler)
    session._fsm_release()
    assert isinstance(session._fsm_state, Free)


# ---------------------------------------------------------------------------
# FSM coordination: handler FIFO ordering
# ---------------------------------------------------------------------------


def test_fsm_handler_fifo_order() -> None:
    """Handlers acquire in the order they registered (FIFO).

    Uses direct queue injection so the ordering is deterministic and
    not subject to OS thread scheduling races.
    """
    session = _FakeSession()
    session._fsm_acquire_handler()  # initial holder

    # Inject two waiters in known order.
    waiter1 = threading.Event()
    waiter2 = threading.Event()
    with session._fsm_cond:
        session._handler_queue.append(waiter1)
        session._handler_queue.append(waiter2)

    # First release hands to waiter1.
    session._fsm_release()
    assert waiter1.is_set()
    assert not waiter2.is_set()
    assert isinstance(session._fsm_state, OwnedByHandler)

    # Second release (waiter1's handler done) hands to waiter2.
    session._fsm_release()
    assert waiter2.is_set()
    assert isinstance(session._fsm_state, OwnedByHandler)

    # Final release clears to Free.
    session._fsm_release()
    assert isinstance(session._fsm_state, Free)


# ---------------------------------------------------------------------------
# FSM coordination: force_release (liveness escape — #1377)
# ---------------------------------------------------------------------------


class _RecordingForceReleaseSession(_FakeSession):
    """``_FakeSession`` that records every ``_on_force_release`` call.

    Lets tests assert the subclass hook fired with the expected reason
    after the FSM transition committed, mirroring the production
    ``ClaudeSession._on_force_release`` override that kills the
    subprocess.
    """

    def __init__(self) -> None:
        super().__init__()
        self.force_release_reasons: list[str] = []

    def _on_force_release(self, *, reason: str) -> None:
        self.force_release_reasons.append(reason)


def test_force_release_from_free_is_noop() -> None:
    """``force_release`` returns False when the FSM is already Free.

    Watchdog can race a holder that just released voluntarily — the
    eviction must be a no-op rather than corrupting state or raising.
    """
    session = _RecordingForceReleaseSession()
    assert session.force_release(reason="watchdog tick") is False
    assert isinstance(session._fsm_state, Free)
    # Subclass hook is *not* fired on the no-op path — there is no
    # holder to knock out of any blocking call.
    assert session.force_release_reasons == []


def test_force_release_from_owned_worker_lands_in_free() -> None:
    """``force_release`` from OwnedByWorker → Free (force_release_to_free)."""
    session = _RecordingForceReleaseSession()
    session._fsm_acquire_worker()
    assert isinstance(session._fsm_state, OwnedByWorker)

    assert session.force_release(reason="worker wedge") is True

    assert isinstance(session._fsm_state, Free)
    assert session.force_release_reasons == ["worker wedge"]


def test_force_release_from_owned_handler_lands_in_free() -> None:
    """``force_release`` from OwnedByHandler → Free (force_release_to_free)."""
    session = _RecordingForceReleaseSession()
    session._fsm_acquire_handler()
    assert isinstance(session._fsm_state, OwnedByHandler)

    assert session.force_release(reason="handler wedge") is True

    assert isinstance(session._fsm_state, Free)
    assert session.force_release_reasons == ["handler wedge"]


def test_force_release_with_queued_handler_transfers_ownership() -> None:
    """``force_release`` with a queued handler hands ownership directly.

    The first oracle call (ForceRelease) takes the FSM through Free;
    the second (HandlerAcquire from Free) lands in OwnedByHandler.
    The queued waiter's event is set so the parked handler thread
    unblocks.
    """
    session = _RecordingForceReleaseSession()
    session._fsm_acquire_worker()  # wedged worker holds the lock

    fake_waiter = threading.Event()
    with session._fsm_cond:
        session._handler_queue.append(fake_waiter)

    assert session.force_release(reason="worker wedge with queue") is True

    assert fake_waiter.is_set(), "queued handler was not signalled"
    assert isinstance(session._fsm_state, OwnedByHandler)
    assert session._handler_queue == []
    assert session.force_release_reasons == ["worker wedge with queue"]


def test_evicted_holder_release_is_skipped() -> None:
    """After ``force_release``, the evicted holder's ``_fsm_release`` is a no-op.

    The evicted holder thread eventually escapes its wedged IO call
    (typically because the subclass hook killed the subprocess and
    its ``select()`` returned EOF).  Its ``__exit__`` calls
    ``_fsm_release`` — without the ``_evicted_tids`` guard, that would
    crash on ``release_only_by_owner`` because the FSM has already
    moved past the holder's role.
    """
    session = _RecordingForceReleaseSession()
    session._fsm_acquire_worker()
    # Simulate the watchdog evicting *this* tid — there is no per-repo
    # ``SessionTalker`` registered (``_FakeSession._repo_name = None``)
    # so we mark the eviction directly the way ``force_release`` would
    # for this thread's tid.
    with session._fsm_cond:
        session._evicted_tids.add(threading.get_ident())
    # Advance the FSM the way ``force_release`` would.
    session._fsm_state = Free()

    # The evicted holder's eventual release: must NOT raise even though
    # the FSM state is Free (the normal release_only_by_owner check
    # would reject WorkerRelease/HandlerRelease here).
    session._fsm_release()  # should be a silent no-op

    # The eviction marker is consumed exactly once.
    assert threading.get_ident() not in session._evicted_tids


def test_force_release_unregisters_evicted_talker_and_records_tid() -> None:
    """``force_release`` records the evicted holder's tid via the global
    talker registry and unregisters the talker so the next acquire's
    ``register_talker`` does not raise :class:`SessionLeakError` on a
    stale entry.
    """

    class _ReposSession(_FakeSession):
        _repo_name: str | None = "test/force-release-talker"

    session = _ReposSession()
    session._fsm_acquire_worker()

    holder_tid = 424242
    register_talker(
        SessionTalker(
            repo_name="test/force-release-talker",
            thread_id=holder_tid,
            kind="worker",
            description="test holder",
            claude_pid=1,
            started_at=talker_now(),
        )
    )
    try:
        assert get_talker("test/force-release-talker") is not None, (
            "talker setup failed"
        )

        assert session.force_release(reason="talker test") is True

        assert isinstance(session._fsm_state, Free)
        assert holder_tid in session._evicted_tids
        assert get_talker("test/force-release-talker") is None, (
            "stale talker should have been unregistered"
        )
    finally:
        # Defensive cleanup: if the production path unregistered the
        # talker (the assertion above), this is a no-op.  If the test
        # failed before that point, this prevents the global registry
        # from leaking into subsequent tests.
        unregister_talker("test/force-release-talker", holder_tid)


def test_force_release_oracle_assertion_path_is_unreachable_in_practice() -> None:
    """``force_release_to_free`` proves ForceRelease is accepted in every
    state; the runtime ``RuntimeError`` branch in ``force_release`` is
    therefore unreachable through normal operation.

    This test guards the *invariant statement*: from each of the three
    states, ``transition(s, ForceRelease())`` returns Some Free.
    Mirrors the proof in ``models/session_lock.v`` so a regression in
    the extracted Python is caught at the test layer too.
    """
    from fido.rocq.transition import ForceRelease, transition

    for s in (Free(), OwnedByWorker(), OwnedByHandler()):
        result = transition(s, ForceRelease())
        assert isinstance(result, Free), (
            f"force_release_to_free violated: transition({s!r}, "
            f"ForceRelease()) = {result!r}"
        )


def test_unconditional_liveness_witness_is_force_release() -> None:
    """Runtime mirror of ``unconditional_liveness`` (``models/session_lock.v``).

    The Rocq theorem says ``∃ev. ∀s. transition s ev = Some Free``.
    ``ForceRelease`` is the witness — strictly stronger than
    ``every_state_reaches_free`` (``∀s. ∃ev. …``), which would leave
    open the possibility of no single event working universally.

    This test pins down ForceRelease *as* that witness so the
    watchdog's "fire ForceRelease without inspecting FSM state"
    guarantee is asserted on the extracted Python and not just in
    the model.  Mirroring the model proof keeps the runtime honest
    about the strong form rather than the weak one.
    """
    from fido.rocq.transition import ForceRelease, transition

    witness = ForceRelease()
    # The witness works for *every* state — the strong ∃∀ shape.
    assert all(
        isinstance(transition(s, witness), Free)
        for s in (Free(), OwnedByWorker(), OwnedByHandler())
    ), (
        "unconditional_liveness violated: ForceRelease should reach Free "
        "from every state; the watchdog relies on this without state inspection."
    )
