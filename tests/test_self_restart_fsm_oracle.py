"""Regression tests for the self_restart FSM oracle.

Each test section corresponds to a proved invariant from
``models/self_restart.v``.  The tests exercise the extracted
``transition`` function directly and verify that
``WebhookHandler._restart_fsm_transition`` raises ``AssertionError``
on any rejected transition — enforcing the fail-closed contract.

Proved invariants exercised:

  ``sync_before_teardown``        — SyncFail from Syncing yields Aborted,
    never StoppingWorkers.  Workers are never stopped when the runner sync
    fails — the process keeps running its current code version intact.

  ``abort_is_terminal``           — every event is rejected from Aborted.
    A failed sync ends the restart episode; no teardown step can fire after
    an abort.

  ``workers_before_children``     — ChildrenKilled is rejected from
    StoppingWorkers.  Claude subprocesses are only killed after all workers
    have exited, preventing the subprocess-orphan scenario from #829.

  ``exit_requires_full_teardown`` — ChildrenKilled (the event that reaches
    Exiting) is rejected from every state except KillingChildren.  The
    process can only reach Exiting via the full sync → stop → kill sequence.

  ``exiting_is_terminal``         — every event is rejected from Exiting.
    Once os._exit(75) fires there is no subsequent coordination state.

Field lesson covered:

  Before self_restart.v, the teardown order (sync → stop workers → kill
  children) was enforced only by code sequencing in _self_restart.  The FSM
  makes this structural: SyncFail → Aborted means StoppingWorkers is
  unreachable on the abort path, and ChildrenKilled is rejected from
  StoppingWorkers so killing can only fire after WorkersStopped.
  A double-trigger (two webhooks racing) now raises AssertionError on the
  second call rather than tearing down workers a second time.
"""

import pytest

from fido.rocq.self_restart import (
    Aborted,
    ChildrenKilled,
    Event,
    Exiting,
    KillingChildren,
    Running,
    State,
    StoppingWorkers,
    SyncFail,
    Syncing,
    SyncOk,
    TriggerRestart,
    WorkersStopped,
    transition,
)
from fido.server import WebhookHandler

# ---------------------------------------------------------------------------
# Minimal handler subclass for oracle tests
#
# BaseHTTPRequestHandler.__init__ requires a socket, address, and server.
# Since we only need to exercise _restart_fsm_transition — which reads and
# writes the class-level _restart_fsm_state attribute via type(self) — we
# subclass WebhookHandler and skip the HTTP plumbing entirely.  The
# _restart_fsm_state assignment in _restart_fsm_transition writes to
# _Handler._restart_fsm_state (shadowing WebhookHandler), so these tests
# are isolated from the production class and from each other.
# ---------------------------------------------------------------------------


class _Handler(WebhookHandler):
    """Minimal WebhookHandler subclass for oracle tests."""

    def __init__(self) -> None:
        pass  # skip BaseHTTPRequestHandler HTTP plumbing


@pytest.fixture(autouse=True)
def _reset_fsm_state() -> None:
    """Reset the handler's FSM state to Running() before each test."""
    _Handler._restart_fsm_state = Running()


def _handler() -> _Handler:
    return _Handler()


# ---------------------------------------------------------------------------
# Invariant: sync_before_teardown
#
# SyncFail from Syncing yields Aborted — never StoppingWorkers.  Workers are
# never stopped when the runner sync fails.  This is the machine-checked form
# of the comment in _self_restart: "Sync runner BEFORE tearing down the
# worker.  If the sync fails we log and return without touching the running
# workers."
# ---------------------------------------------------------------------------


def test_sync_fail_yields_aborted() -> None:
    """SyncFail from Syncing yields Aborted — abort, not teardown.

    sync_before_teardown: the runner sync failing must end the episode in
    Aborted, never in StoppingWorkers or any teardown state.  Workers are
    untouched when the sync fails.
    """
    result = transition(Syncing(), SyncFail())
    assert isinstance(result, Aborted), (
        f"sync_before_teardown violated: SyncFail from Syncing "
        f"yielded {type(result).__name__!r} instead of Aborted"
    )


def test_sync_fail_does_not_yield_stopping_workers() -> None:
    """SyncFail never yields StoppingWorkers — workers untouched on abort.

    sync_before_teardown: the only path to StoppingWorkers is SyncOk from
    Syncing.  SyncFail must never produce StoppingWorkers.
    """
    result = transition(Syncing(), SyncFail())
    assert not isinstance(result, StoppingWorkers), (
        "sync_before_teardown violated: SyncFail from Syncing reached StoppingWorkers"
    )


def test_sync_ok_yields_stopping_workers() -> None:
    """SyncOk from Syncing yields StoppingWorkers — the sole path to teardown.

    The only route to worker teardown is via SyncOk.  This is the structural
    guarantee that workers are never stopped without a prior successful sync.
    """
    result = transition(Syncing(), SyncOk())
    assert isinstance(result, StoppingWorkers), (
        f"sync_before_teardown violated: SyncOk from Syncing "
        f"yielded {type(result).__name__!r} instead of StoppingWorkers"
    )


# ---------------------------------------------------------------------------
# Invariant: abort_is_terminal
#
# Every event is rejected from Aborted.  A failed sync ends the restart
# episode; no teardown step (stopping workers, killing children, or exiting)
# is accepted.  The process keeps running its current code version.
# ---------------------------------------------------------------------------


def test_abort_is_terminal_complete() -> None:
    """All five events are rejected from Aborted — abort is the end of the episode.

    abort_is_terminal: once sync fails the process is back to normal
    operation.  No event can advance beyond Aborted: workers stay up,
    children stay alive, and the process stays running.
    """
    s = Aborted()
    for event in [
        TriggerRestart(),
        SyncOk(),
        SyncFail(),
        WorkersStopped(),
        ChildrenKilled(),
    ]:
        result = transition(s, event)
        assert result is None, (
            f"abort_is_terminal violated: {type(event).__name__} accepted from Aborted"
        )


def test_workers_stopped_rejected_from_aborted() -> None:
    """WorkersStopped is rejected from Aborted — workers were never stopped.

    abort_is_terminal: the sync failed, so the worker-stop sequence never
    ran.  WorkersStopped arriving from Aborted would be a false report —
    it is always refused.
    """
    assert transition(Aborted(), WorkersStopped()) is None


def test_children_killed_rejected_from_aborted() -> None:
    """ChildrenKilled is rejected from Aborted — children were never killed.

    abort_is_terminal: the abort path never reaches KillingChildren, so
    ChildrenKilled is structurally impossible from Aborted.
    """
    assert transition(Aborted(), ChildrenKilled()) is None


# ---------------------------------------------------------------------------
# Invariant: workers_before_children
#
# ChildrenKilled is rejected from StoppingWorkers.  Claude subprocesses are
# only killed after all worker threads have exited — never before.  This is
# the machine-checked form of the ordering in _self_restart: stop_and_join
# and stop_all are called before _fn_kill_active_children.
# Skipping WorkersStopped and jumping straight to ChildrenKilled is
# structurally impossible in the FSM.
# ---------------------------------------------------------------------------


def test_children_killed_rejected_from_stopping_workers() -> None:
    """ChildrenKilled is rejected from StoppingWorkers — kill after stop, not before.

    workers_before_children: the subprocess-kill step can only fire after
    all workers have exited (WorkersStopped).  ChildrenKilled from
    StoppingWorkers is always rejected, preventing the #829 scenario where
    a subprocess is killed while its worker thread is still alive.
    """
    result = transition(StoppingWorkers(), ChildrenKilled())
    assert result is None, (
        "workers_before_children violated: ChildrenKilled accepted from StoppingWorkers"
    )


def test_workers_stopped_advances_to_killing_children() -> None:
    """WorkersStopped from StoppingWorkers yields KillingChildren — correct ordering.

    The one valid path through StoppingWorkers requires WorkersStopped first.
    Only after all workers have exited can the subprocess-kill step begin.
    """
    result = transition(StoppingWorkers(), WorkersStopped())
    assert isinstance(result, KillingChildren), (
        f"workers_before_children violated: WorkersStopped from StoppingWorkers "
        f"yielded {type(result).__name__!r} instead of KillingChildren"
    )


def test_children_killed_only_valid_from_killing_children() -> None:
    """ChildrenKilled is accepted only from KillingChildren — no shortcut.

    The process can only kill children when it is in KillingChildren state,
    which is only reachable via SyncOk → StoppingWorkers → WorkersStopped.
    Every other state rejects ChildrenKilled.
    """
    assert isinstance(transition(KillingChildren(), ChildrenKilled()), Exiting)
    assert transition(Running(), ChildrenKilled()) is None
    assert transition(Syncing(), ChildrenKilled()) is None
    assert transition(StoppingWorkers(), ChildrenKilled()) is None
    assert transition(Aborted(), ChildrenKilled()) is None
    assert transition(Exiting(), ChildrenKilled()) is None


# ---------------------------------------------------------------------------
# Invariant: exit_requires_full_teardown
#
# ChildrenKilled (the event that reaches Exiting) is rejected from every
# state except KillingChildren.  The process can only exit after sync
# succeeded, workers stopped, and children were killed — all three steps
# are structural prerequisites.
# ---------------------------------------------------------------------------


def test_children_killed_rejected_from_running() -> None:
    """ChildrenKilled rejected from Running — cannot exit without starting the sequence."""
    assert transition(Running(), ChildrenKilled()) is None


def test_children_killed_rejected_from_syncing() -> None:
    """ChildrenKilled rejected from Syncing — sync not yet resolved."""
    assert transition(Syncing(), ChildrenKilled()) is None


def test_children_killed_rejected_from_exiting() -> None:
    """ChildrenKilled rejected from Exiting — process already exiting."""
    assert transition(Exiting(), ChildrenKilled()) is None


def test_children_killed_accepted_from_killing_children() -> None:
    """ChildrenKilled from KillingChildren yields Exiting — full teardown complete.

    exit_requires_full_teardown: this is the only valid path to Exiting.
    KillingChildren is reached only via SyncOk → StoppingWorkers →
    WorkersStopped, so reaching Exiting requires all three prior steps.
    """
    result = transition(KillingChildren(), ChildrenKilled())
    assert isinstance(result, Exiting), (
        f"exit_requires_full_teardown violated: ChildrenKilled from KillingChildren "
        f"yielded {type(result).__name__!r} instead of Exiting"
    )


# ---------------------------------------------------------------------------
# Invariant: exiting_is_terminal
#
# Every event is rejected from Exiting.  Once os._exit(75) fires the process
# terminates immediately — there is no subsequent coordination state.
# ---------------------------------------------------------------------------


def test_exiting_is_terminal_complete() -> None:
    """All five events are rejected from Exiting — process exit is terminal.

    exiting_is_terminal: once ChildrenKilled advances to Exiting, the process
    calls os._exit(75) and no further event is possible.  The FSM enforces
    that no event is accepted from Exiting.
    """
    s = Exiting()
    for event in [
        TriggerRestart(),
        SyncOk(),
        SyncFail(),
        WorkersStopped(),
        ChildrenKilled(),
    ]:
        result = transition(s, event)
        assert result is None, (
            f"exiting_is_terminal violated: {type(event).__name__} accepted from Exiting"
        )


# ---------------------------------------------------------------------------
# Complete lifecycle paths
#
# End-to-end tests of the two valid multi-step paths through the FSM.
# ---------------------------------------------------------------------------


def test_happy_path() -> None:
    """Running → Syncing → StoppingWorkers → KillingChildren → Exiting.

    The complete successful restart lifecycle: trigger detected, sync
    succeeds, workers stopped, children killed, process exits with code 75.
    Every step advances the FSM and no step is skippable.
    """
    s: State = Running()

    s = transition(s, TriggerRestart())  # type: ignore[assignment]
    assert isinstance(s, Syncing)

    s = transition(s, SyncOk())  # type: ignore[assignment]
    assert isinstance(s, StoppingWorkers)

    s = transition(s, WorkersStopped())  # type: ignore[assignment]
    assert isinstance(s, KillingChildren)

    s = transition(s, ChildrenKilled())  # type: ignore[assignment]
    assert isinstance(s, Exiting)


def test_abort_path() -> None:
    """Running → Syncing → Aborted — sync failure aborts without teardown.

    The sync-failure path: trigger detected, sync fails, episode ends in
    Aborted.  Workers and children are never touched on this path.
    """
    s: State = Running()

    s = transition(s, TriggerRestart())  # type: ignore[assignment]
    assert isinstance(s, Syncing)

    s = transition(s, SyncFail())  # type: ignore[assignment]
    assert isinstance(s, Aborted)


# ---------------------------------------------------------------------------
# Exhaustive state×event matrix
#
# All 6×5 = 30 transition pairs verified against the table in self_restart.v.
# ---------------------------------------------------------------------------


def test_all_transitions_exhaustive() -> None:
    """Every state×event pair has a deterministic transition result.

    Walks the full 6×5 matrix and asserts that the output type matches the
    transition table in ``models/self_restart.v``.  Any future change to the
    Rocq model that adds or removes a valid transition will be caught here.
    """
    expected: dict[tuple[type[State], type[Event]], type[State] | None] = {
        # Running
        (Running, TriggerRestart): Syncing,
        (Running, SyncOk): None,
        (Running, SyncFail): None,
        (Running, WorkersStopped): None,
        (Running, ChildrenKilled): None,
        # Syncing
        (Syncing, TriggerRestart): None,
        (Syncing, SyncOk): StoppingWorkers,
        (Syncing, SyncFail): Aborted,
        (Syncing, WorkersStopped): None,
        (Syncing, ChildrenKilled): None,
        # StoppingWorkers
        (StoppingWorkers, TriggerRestart): None,
        (StoppingWorkers, SyncOk): None,
        (StoppingWorkers, SyncFail): None,
        (StoppingWorkers, WorkersStopped): KillingChildren,
        (StoppingWorkers, ChildrenKilled): None,
        # KillingChildren
        (KillingChildren, TriggerRestart): None,
        (KillingChildren, SyncOk): None,
        (KillingChildren, SyncFail): None,
        (KillingChildren, WorkersStopped): None,
        (KillingChildren, ChildrenKilled): Exiting,
        # Exiting
        (Exiting, TriggerRestart): None,
        (Exiting, SyncOk): None,
        (Exiting, SyncFail): None,
        (Exiting, WorkersStopped): None,
        (Exiting, ChildrenKilled): None,
        # Aborted
        (Aborted, TriggerRestart): None,
        (Aborted, SyncOk): None,
        (Aborted, SyncFail): None,
        (Aborted, WorkersStopped): None,
        (Aborted, ChildrenKilled): None,
    }
    for (state_cls, event_cls), expected_cls in expected.items():
        result = transition(state_cls(), event_cls())
        if expected_cls is None:
            assert result is None, (
                f"expected None for ({state_cls.__name__}, {event_cls.__name__}), "
                f"got {type(result).__name__}"
            )
        else:
            assert isinstance(result, expected_cls), (
                f"expected {expected_cls.__name__} for "
                f"({state_cls.__name__}, {event_cls.__name__}), "
                f"got {type(result).__name__ if result is not None else 'None'}"
            )


# ---------------------------------------------------------------------------
# _restart_fsm_transition crash-loud behaviour
#
# Verify that WebhookHandler._restart_fsm_transition raises AssertionError on
# invalid events rather than silently no-oping.  This is the fail-closed
# contract: a coordination violation surfaces as an immediate crash rather
# than silent state divergence.
# ---------------------------------------------------------------------------


def test_fsm_transition_crashes_on_invalid_event() -> None:
    """_restart_fsm_transition raises AssertionError when the FSM rejects an event.

    Starts the handler in Running and fires SyncOk — rejected because Running
    only accepts TriggerRestart.  The oracle must raise AssertionError
    immediately, enforcing the fail-closed contract.
    """
    h = _handler()
    # Default state is Running; SyncOk is rejected from Running.
    with pytest.raises(AssertionError, match="self_restart FSM"):
        h._restart_fsm_transition(SyncOk())  # pyright: ignore[reportPrivateUsage]


def test_fsm_transition_includes_state_and_event_in_error() -> None:
    """AssertionError message names the rejected state and event.

    The crash message must include the state name and the event name so the
    engineer can immediately identify which invariant was violated.
    """
    h = _handler()
    with pytest.raises(AssertionError, match="Running") as exc_info:
        h._restart_fsm_transition(SyncOk())  # pyright: ignore[reportPrivateUsage]
    assert "SyncOk" in str(exc_info.value)


def test_fsm_transition_raises_on_double_trigger() -> None:
    """Second TriggerRestart raises AssertionError — double-trigger is rejected.

    TriggerRestart is only valid from Running.  A second TriggerRestart while
    already Syncing (or any other non-Running state) raises immediately,
    preventing a racing webhook from tearing down workers a second time.
    """
    h = _handler()
    h._restart_fsm_transition(TriggerRestart())  # pyright: ignore[reportPrivateUsage]
    # Now in Syncing — TriggerRestart again is rejected.
    with pytest.raises(AssertionError, match="self_restart FSM"):
        h._restart_fsm_transition(TriggerRestart())  # pyright: ignore[reportPrivateUsage]


def test_fsm_transition_updates_state_on_valid_event() -> None:
    """_restart_fsm_transition updates _restart_fsm_state on valid transitions.

    The class-level state is updated after each accepted event, so subsequent
    calls see the advanced state rather than the initial Running state.
    """
    h = _handler()
    assert isinstance(_Handler._restart_fsm_state, Running)

    h._restart_fsm_transition(TriggerRestart())  # pyright: ignore[reportPrivateUsage]
    assert isinstance(_Handler._restart_fsm_state, Syncing)

    h._restart_fsm_transition(SyncOk())  # pyright: ignore[reportPrivateUsage]
    assert isinstance(_Handler._restart_fsm_state, StoppingWorkers)

    h._restart_fsm_transition(WorkersStopped())  # pyright: ignore[reportPrivateUsage]
    assert isinstance(_Handler._restart_fsm_state, KillingChildren)

    h._restart_fsm_transition(ChildrenKilled())  # pyright: ignore[reportPrivateUsage]
    assert isinstance(_Handler._restart_fsm_state, Exiting)


def test_fsm_transition_abort_path_advances_state() -> None:
    """_restart_fsm_transition drives the abort path: Running → Syncing → Aborted.

    Validates that the oracle fires correctly on the sync-failure branch —
    the state advances to Aborted rather than to any teardown state.
    """
    h = _handler()
    h._restart_fsm_transition(TriggerRestart())  # pyright: ignore[reportPrivateUsage]
    assert isinstance(_Handler._restart_fsm_state, Syncing)

    h._restart_fsm_transition(SyncFail())  # pyright: ignore[reportPrivateUsage]
    assert isinstance(_Handler._restart_fsm_state, Aborted)


def test_fsm_transition_returns_new_state() -> None:
    """_restart_fsm_transition returns the new state on success.

    The return value is the next FSM state, matching what restart_fsm.transition
    would return directly.
    """
    h = _handler()
    new_state = h._restart_fsm_transition(TriggerRestart())  # pyright: ignore[reportPrivateUsage]
    assert isinstance(new_state, Syncing)
