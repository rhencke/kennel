"""Tests for the session-lock FSM oracle wired into OwnedSession.

The oracle runs the Rocq-extracted ``transition`` function from
``src/fido/rocq/transition.py`` on every outermost lock
acquire and release.  Divergence from the proved model raises a
``RuntimeError`` naming the violated theorem so bugs are immediately
identifiable from the traceback.

Proved theorems being guarded:
  ``no_dual_ownership``     — acquisition rejected when session already owned
  ``release_only_by_owner`` — release rejected when wrong owner or Free
"""

import pytest

from fido.provider import OwnedSession
from fido.rocq.transition import (
    Free,
    OwnedByHandler,
    OwnedByWorker,
)


class _FakeSession(OwnedSession):
    """Minimal OwnedSession subclass for unit-testing oracle behaviour.

    The oracle logic lives entirely in the base class; this stub exists
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


def test_oracle_starts_free() -> None:
    """Oracle initialises to Free — nobody holds the session."""
    session = _FakeSession()
    assert isinstance(session._oracle_state, Free)


# ---------------------------------------------------------------------------
# Happy-path: worker acquire → release
# ---------------------------------------------------------------------------


def test_oracle_worker_acquire_and_release() -> None:
    """Worker acquire transitions Free → OwnedByWorker; release → Free."""
    session = _FakeSession()

    session._oracle_on_acquire("worker")
    assert isinstance(session._oracle_state, OwnedByWorker)

    session._oracle_on_release()
    assert isinstance(session._oracle_state, Free)


# ---------------------------------------------------------------------------
# Happy-path: handler (webhook) acquire → release
# ---------------------------------------------------------------------------


def test_oracle_handler_acquire_and_release() -> None:
    """Handler acquire transitions Free → OwnedByHandler; release → Free."""
    session = _FakeSession()

    session._oracle_on_acquire("webhook")
    assert isinstance(session._oracle_state, OwnedByHandler)

    session._oracle_on_release()
    assert isinstance(session._oracle_state, Free)


# ---------------------------------------------------------------------------
# Theorem: no_dual_ownership
# ---------------------------------------------------------------------------


def test_oracle_no_dual_ownership_worker_then_worker() -> None:
    """Worker cannot acquire an already worker-owned session.

    Proved by ``no_dual_ownership`` in ``models/session_lock.v``.
    """
    session = _FakeSession()
    session._oracle_on_acquire("worker")  # Free → OwnedByWorker

    with pytest.raises(RuntimeError, match="no_dual_ownership"):
        session._oracle_on_acquire("worker")  # OwnedByWorker + WorkerAcquire → None


def test_oracle_no_dual_ownership_worker_then_handler() -> None:
    """Handler cannot acquire a worker-owned session without preemption.

    Proved by ``no_dual_ownership`` in ``models/session_lock.v``.
    """
    session = _FakeSession()
    session._oracle_on_acquire("worker")  # Free → OwnedByWorker

    with pytest.raises(RuntimeError, match="no_dual_ownership"):
        session._oracle_on_acquire("webhook")  # OwnedByWorker + HandlerAcquire → None


def test_oracle_no_dual_ownership_handler_then_worker() -> None:
    """Worker cannot acquire a handler-owned session.

    Proved by ``no_dual_ownership`` in ``models/session_lock.v``.
    """
    session = _FakeSession()
    session._oracle_on_acquire("webhook")  # Free → OwnedByHandler

    with pytest.raises(RuntimeError, match="no_dual_ownership"):
        session._oracle_on_acquire("worker")  # OwnedByHandler + WorkerAcquire → None


def test_oracle_no_dual_ownership_handler_then_handler() -> None:
    """Handler cannot acquire an already handler-owned session.

    Proved by ``no_dual_ownership`` in ``models/session_lock.v``.
    """
    session = _FakeSession()
    session._oracle_on_acquire("webhook")  # Free → OwnedByHandler

    with pytest.raises(RuntimeError, match="no_dual_ownership"):
        session._oracle_on_acquire("webhook")  # OwnedByHandler + HandlerAcquire → None


# ---------------------------------------------------------------------------
# Theorem: release_only_by_owner
# ---------------------------------------------------------------------------


def test_oracle_release_only_by_owner_spurious_from_free() -> None:
    """Releasing from Free is rejected (spurious release).

    Covered by the Free cases in ``release_only_by_owner``.
    """
    session = _FakeSession()
    # State is Free; _oracle_on_release() will send HandlerRelease because
    # Free is not OwnedByWorker — transition(Free, HandlerRelease) = None.
    with pytest.raises(RuntimeError, match="release_only_by_owner"):
        session._oracle_on_release()
