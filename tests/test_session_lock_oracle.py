"""Tests for the FSM-driven lock coordination in OwnedSession.

Covers the FSM coordination methods (``_fsm_acquire_worker`` /
``_fsm_acquire_handler`` / ``_fsm_release``) — the authoritative
lock coordinator extracted from ``models/session_lock.v``.  Workers
block until Free with no queued handlers; handlers block until Free
(FIFO queue for handler-on-handler ordering).

Proved theorems being guarded:
  ``no_dual_ownership``     — acquisition rejected when session already owned
  ``release_only_by_owner`` — release rejected when wrong owner or Free
"""

import threading

import pytest

from fido.provider import OwnedSession
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
