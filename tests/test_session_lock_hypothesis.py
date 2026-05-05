"""Hypothesis property tests for the session-lock FSM.

The extracted ``transition`` function in
``src/fido/rocq/transition.py`` is the runtime oracle for
``OwnedSession``.  These tests drive it with randomly generated event
sequences to confirm that the proved invariants hold in the extracted
Python — not just for the specific cases enumerated in the unit tests,
but for *every* input Hypothesis can construct.

Proved invariants exercised:
  ``no_dual_ownership``        — any acquire from an already-owned state → None
  ``release_only_by_owner``    — cross-release or spurious release → None
  ``force_release_to_free``    — ForceRelease always lands in Free regardless
                                 of starting state (proved on every random
                                 reachable state, not just the three roots)
  ``unconditional_liveness``   — at the trace level: from any state reached
                                 by an arbitrary event sequence, firing
                                 ForceRelease lands in Free (the watchdog
                                 needs no priors about state)

A "closure" property checks that every successful transition lands on
a valid state and that the FSM never silently returns garbage.
"""

from hypothesis import given
from hypothesis import strategies as st

from fido.rocq.transition import (
    ForceRelease,
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
from fido.rocq.transition import (
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
    st.just(ForceRelease()),
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

    Note: ForceRelease is excluded from the strategy — it lands in
    Free, not OwnedByHandler, so it does not violate the property
    being asserted (and we test ForceRelease's target state explicitly
    below).
    """
    result = transition(OwnedByWorker(), event)
    assert not isinstance(result, OwnedByHandler), (
        f"Unexpected OwnedByHandler from non-Preempt event "
        f"{type(event).__name__!r}: transition returned {result!r}"
    )


# ---------------------------------------------------------------------------
# Property: force_release_to_free — ForceRelease always lands in Free
# ---------------------------------------------------------------------------


@given(state=_all_states)
def test_force_release_always_lands_in_free(state: FsmState) -> None:
    """Hypothesis mirror of ``force_release_to_free`` in the model.

    Exhaustive over the three reachable FSM states; every one must
    transition to Free under ForceRelease.  The watchdog's
    post-condition is therefore unconditional on starting state.
    """
    result = transition(state, ForceRelease())
    assert isinstance(result, Free), (
        f"force_release_to_free violated: transition({type(state).__name__}, "
        f"ForceRelease()) = {result!r}, expected Free()"
    )


# ---------------------------------------------------------------------------
# Property: unconditional_liveness at the *trace* level
# ---------------------------------------------------------------------------


@given(events=st.lists(_all_events, min_size=0, max_size=50))
def test_force_release_lands_in_free_after_arbitrary_trace(
    events: list[object],
) -> None:
    """Trace-level mirror of ``unconditional_liveness``.

    From the initial Free state, walk an arbitrary event sequence
    (applying only successful transitions, mirroring the runtime).
    Whatever state we end up in, firing ForceRelease must land us in
    Free.

    This is the strongest runtime form of the watchdog's guarantee:
    no matter what interleaving of acquires, releases, preempts, and
    even other ForceReleases happened first, the watchdog's next
    ForceRelease unconditionally returns the FSM to a usable state.
    """
    current: FsmState = Free()
    for ev in events:
        result = transition(current, ev)
        if result is not None:
            current = result

    final = transition(current, ForceRelease())
    assert isinstance(final, Free), (
        f"unconditional_liveness violated at trace level: "
        f"after {len(events)} events ending in state "
        f"{type(current).__name__!r}, transition(_, ForceRelease()) "
        f"= {final!r}, expected Free()"
    )


# ---------------------------------------------------------------------------
# Property: ForceRelease is the only event that succeeds in every state
# ---------------------------------------------------------------------------


@given(state=_all_states)
def test_force_release_is_the_universal_release(state: FsmState) -> None:
    """ForceRelease is the *only* event that succeeds in every state —
    every other event is rejected from at least one state.

    Mirrors the design intent of ``unconditional_liveness``:
    ForceRelease is structurally distinguished from the voluntary
    releases.  ``WorkerRelease`` is rejected from Free and
    OwnedByHandler; ``HandlerRelease`` from Free and OwnedByWorker;
    ``WorkerAcquire`` and ``HandlerAcquire`` from owned states;
    ``Preempt`` from Free and OwnedByHandler.  Only ForceRelease
    yields ``Some _`` for every state.
    """
    # ForceRelease succeeds.
    assert transition(state, ForceRelease()) is not None

    # No other event succeeds in *every* state.  We assert this by
    # checking that for each non-ForceRelease event, at least one
    # state rejects it — the witness varies by event but it always
    # exists.
    other_events = [
        WorkerAcquire(),
        HandlerAcquire(),
        WorkerRelease(),
        HandlerRelease(),
        Preempt(),
    ]
    all_states = [Free(), OwnedByWorker(), OwnedByHandler()]
    for ev in other_events:
        rejected_somewhere = any(transition(s, ev) is None for s in all_states)
        assert rejected_somewhere, (
            f"event {type(ev).__name__!r} unexpectedly succeeds in every "
            f"state — that would weaken ForceRelease's distinguishing role"
        )
