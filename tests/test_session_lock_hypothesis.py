"""Hypothesis property tests for the session-lock FSM.

The extracted ``transition`` function in
``kennel/models_generated/transition.py`` is the runtime oracle for
``OwnedSession``.  These tests drive it with randomly generated event
sequences to confirm that the proved invariants hold in the extracted
Python — not just for the specific cases enumerated in the unit tests,
but for *every* input Hypothesis can construct.

Proved invariants exercised:
  ``no_dual_ownership``     — any acquire from an already-owned state → None
  ``release_only_by_owner`` — cross-release or spurious release → None

A third "closure" property checks that every successful transition lands on
a valid state and that the FSM never silently returns garbage.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from kennel.models_generated.transition import (
    Free,
    HandlerAcquire,
    HandlerRelease,
    OwnedByHandler,
    OwnedByWorker,
    Preempt,
    WorkerAcquire,
    WorkerRelease,
    transition,
)
from kennel.models_generated.transition import (
    State as FsmState,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_all_states = st.one_of(
    st.just(Free()),
    st.just(OwnedByWorker()),
    st.just(OwnedByHandler()),
)

_acquire_events = st.one_of(
    st.just(WorkerAcquire()),
    st.just(HandlerAcquire()),
)

_release_events = st.one_of(
    st.just(WorkerRelease()),
    st.just(HandlerRelease()),
)

_all_events = st.one_of(
    _acquire_events,
    _release_events,
    st.just(Preempt()),
)


# ---------------------------------------------------------------------------
# Property: no_dual_ownership
# ---------------------------------------------------------------------------


@given(
    owned_state=st.one_of(st.just(OwnedByWorker()), st.just(OwnedByHandler())),
    acquire_event=_acquire_events,
)
def test_no_dual_ownership(owned_state: FsmState, acquire_event: object) -> None:
    """Any acquire event is rejected when the session is already owned.

    Corresponds to ``no_dual_ownership`` in ``models/session_lock.v``.
    Hypothesis exhausts all (owned_state × acquire_event) pairs — there are
    only four, so every case is covered on the first run.
    """
    result = transition(owned_state, acquire_event)
    assert result is None, (
        f"no_dual_ownership violated: transition({type(owned_state).__name__}, "
        f"{type(acquire_event).__name__}) = {result!r}, expected None"
    )


# ---------------------------------------------------------------------------
# Property: release_only_by_owner (cross-release cases)
# ---------------------------------------------------------------------------


def test_cross_release_worker_release_from_handler() -> None:
    """WorkerRelease from OwnedByHandler is always rejected.

    Proved by ``release_only_by_owner``.
    """
    result = transition(OwnedByHandler(), WorkerRelease())
    assert result is None


def test_cross_release_handler_release_from_worker() -> None:
    """HandlerRelease from OwnedByWorker is always rejected.

    Proved by ``release_only_by_owner``.
    """
    result = transition(OwnedByWorker(), HandlerRelease())
    assert result is None


@given(release_event=_release_events)
def test_spurious_release_from_free(release_event: object) -> None:
    """Any release event is rejected from the Free state.

    Proved by ``release_only_by_owner``.
    """
    result = transition(Free(), release_event)
    assert result is None, (
        f"release_only_by_owner violated: transition(Free, "
        f"{type(release_event).__name__}) = {result!r}, expected None"
    )


# ---------------------------------------------------------------------------
# Property: FSM closure — valid transitions land on valid states
# ---------------------------------------------------------------------------


@given(state=_all_states, event=_all_events)
def test_transition_returns_state_or_none(state: FsmState, event: object) -> None:
    """transition() returns a FsmState instance or None — never garbage.

    Guards against extraction bugs that might produce an unexpected type.
    """
    result = transition(state, event)
    assert result is None or isinstance(result, FsmState), (
        f"transition({type(state).__name__}, {type(event).__name__}) "
        f"returned unexpected type {type(result).__name__!r}: {result!r}"
    )


# ---------------------------------------------------------------------------
# Property: random event sequences maintain FSM consistency
# ---------------------------------------------------------------------------


@given(events=st.lists(_all_events, min_size=1, max_size=50))
def test_random_event_sequence_consistency(events: list[object]) -> None:
    """Random event sequences never leave the FSM in an invalid state.

    Walk a sequence of events through the FSM, applying only the
    transitions that succeed (None → skip).  After every successful step
    the resulting state must be a recognised FsmState, and the FSM must
    remain reachable from Free (i.e. no phantom states are created).
    """
    current: FsmState = Free()
    valid_states = (Free, OwnedByWorker, OwnedByHandler)

    for ev in events:
        result = transition(current, ev)
        if result is not None:
            assert isinstance(result, valid_states), (
                f"FSM produced invalid state {type(result).__name__!r} "
                f"after transition({type(current).__name__}, {type(ev).__name__})"
            )
            current = result

    # After all events the FSM must still be in a recognised state.
    assert isinstance(current, valid_states)


# ---------------------------------------------------------------------------
# Property: Preempt is the only way handler can take a worker-held session
# ---------------------------------------------------------------------------


@given(
    event=st.one_of(
        st.just(WorkerAcquire()),
        st.just(HandlerAcquire()),
        st.just(WorkerRelease()),
        st.just(HandlerRelease()),
    )
)
def test_preempt_is_only_worker_to_handler_path(event: object) -> None:
    """Only Preempt can move OwnedByWorker → OwnedByHandler.

    All other events either keep the worker in the owned state (via
    WorkerRelease → Free, which is not OwnedByHandler) or are rejected.
    Confirms the model's exclusive-takeover design.
    """
    result = transition(OwnedByWorker(), event)
    assert not isinstance(result, OwnedByHandler), (
        f"Unexpected OwnedByHandler from non-Preempt event "
        f"{type(event).__name__!r}: transition returned {result!r}"
    )
