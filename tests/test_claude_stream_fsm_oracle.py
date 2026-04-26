"""Regression tests for the claude_session stream-protocol FSM oracle.

Each test section corresponds to a specific coordination bug.  The tests
construct the pre-fix protocol-violation sequence and assert that the FSM
extracted from ``models/claude_session.v`` rejects it — so if the invariant
is ever removed or weakened, this file breaks before the bug can recur.

Proved invariants exercised:

  ``single_writer``                      — Send rejected from all non-Idle states
  ``cancel_does_not_persist_across_turns`` — stale CancelFire from Idle rejected;
                                            full Cancelled→TurnReturn→Idle path
                                            returns a clean Idle with no cancel artifact
  ``empty_result_is_not_completion``     — TurnReturn from AwaitingReply yields Idle,
                                            not Cancelled; only the drain path reaches
                                            Cancelled
  ``drain_terminates``                   — Send rejected from Draining;
                                            TurnReturn always exits Draining in one step

Bugs covered:
  #973 — leftover cancel flag aborted handler's first prompt
  #975 — switch_model hung with _in_turn=True after preempt-cancelled drain
  #979 — cross-thread stdin writes corrupted the claude protocol stream
"""

from pathlib import Path

import pytest

from fido.rocq.claude_session import (
    AwaitingReply,
    CancelFire,
    Cancelled,
    Draining,
    DrainObserve,
    Idle,
    ReplyChunk,
    Send,
    Sending,
    State,
    TurnReturn,
    transition,
)

# ---------------------------------------------------------------------------
# Bug #979 — cross-thread stdin writes corrupt the claude protocol stream
#
# Root cause: _fire_worker_cancel wrote a control_request interrupt to
# claude's stdin without holding the session lock, racing with the worker
# thread's concurrent stdin write.  Partial bytes from both writes
# interleaved in the kernel pipe; claude read malformed JSON and silently
# wedged.
#
# FSM fix: single_writer — Send is rejected from every non-Idle state.
# Only one thread may be writing to stdin at a time; the FSM crashes
# loudly if a second Send arrives before the first turn completes.
# ---------------------------------------------------------------------------


def test_979_send_rejected_from_sending() -> None:
    """A second Send is rejected while the first is still in flight.

    Pre-fix race (#979): Thread A fired send() (Idle→Sending), then
    Thread B called _send_control_interrupt() — also a stdin write —
    without acquiring the session lock.  The FSM rejects the second Send
    from Sending, enforcing the single_writer invariant.
    """
    result = transition(Sending(), Send())
    assert result is None, (
        "single_writer violated: Send accepted from Sending (pre-fix #979 race)"
    )


def test_979_send_rejected_from_awaiting_reply() -> None:
    """Send is rejected while awaiting a reply.

    Mid-turn: the first reply chunk arrived (Sending→AwaitingReply) but
    the turn is still streaming.  Any concurrent stdin write is rejected.
    """
    result = transition(AwaitingReply(), Send())
    assert result is None, (
        "single_writer violated: Send accepted from AwaitingReply (pre-fix #979 race)"
    )


def test_979_send_rejected_from_draining() -> None:
    """Send is rejected during cancel drain.

    A cancel signal fired (→Draining).  No new stdin write is allowed
    until the drain completes and the session returns to Idle.
    """
    result = transition(Draining(), Send())
    assert result is None, (
        "single_writer violated: Send accepted from Draining (pre-fix #979 race)"
    )


def test_979_send_rejected_from_cancelled() -> None:
    """Send is rejected in the Cancelled state.

    The cancelled turn boundary was reached but the session has not yet
    been acknowledged back to Idle.  No new stdin write allowed.
    """
    result = transition(Cancelled(), Send())
    assert result is None, (
        "single_writer violated: Send accepted from Cancelled (pre-fix #979 race)"
    )


def test_979_send_accepted_only_from_idle() -> None:
    """Send is accepted from Idle — the only valid entry point for a new turn."""
    result = transition(Idle(), Send())
    assert isinstance(result, Sending)


# ---------------------------------------------------------------------------
# Bug #973 — leftover cancel flag aborted handler's first prompt
#
# Root cause: _cancel was session-scoped, not turn-scoped.  After the
# worker's preempted turn, _cancel remained set.  The handler thread
# acquired the session lock and its iter_events immediately saw
# _cancel=True — aborting the new, unrelated turn before it started.
#
# FSM fix: cancel_does_not_persist_across_turns — CancelFire from Idle is
# rejected (no turn to cancel).  After a cancelled turn, TurnReturn from
# Cancelled returns to a clean Idle; the next Send sees no stale cancel.
# ---------------------------------------------------------------------------


def test_973_stale_cancel_from_idle_rejected() -> None:
    """CancelFire is rejected from Idle — there is no in-flight turn.

    Pre-fix scenario (#973): worker's turn was cancelled, _cancel remained
    set.  Handler acquired the lock, but the stale cancel signal made
    iter_events fire CancelFire immediately from Idle — aborting the
    handler's first prompt.  The FSM rejects CancelFire from Idle.
    """
    result = transition(Idle(), CancelFire())
    assert result is None, (
        "cancel_does_not_persist violated: CancelFire accepted from Idle "
        "(pre-fix #973 stale cancel)"
    )


def test_973_cancelled_turn_returns_to_clean_idle() -> None:
    """Cancelled→TurnReturn→Idle leaves the FSM in a clean state.

    Demonstrates the full cancel-acknowledgement path: after a cancelled
    turn is acknowledged, the session is in Idle with no cancel artifact
    — the next Send succeeds immediately.
    """
    # Worker runs a turn that gets cancelled.
    s: State = Idle()
    s = transition(s, Send())  # type: ignore[assignment]  # Idle→Sending
    assert isinstance(s, Sending)
    s = transition(s, ReplyChunk())  # type: ignore[assignment]  # Sending→AwaitingReply
    assert isinstance(s, AwaitingReply)
    s = transition(s, CancelFire())  # type: ignore[assignment]  # AwaitingReply→Draining
    assert isinstance(s, Draining)
    s = transition(s, TurnReturn())  # type: ignore[assignment]  # Draining→Cancelled
    assert isinstance(s, Cancelled)

    # Acknowledge the cancelled turn: Cancelled→Idle.
    s = transition(s, TurnReturn())  # type: ignore[assignment]
    assert isinstance(s, Idle), (
        "cancel persisted across TurnReturn (pre-fix #973 stale cancel)"
    )

    # Handler's first turn starts from a completely clean Idle.
    s = transition(s, Send())  # type: ignore[assignment]
    assert isinstance(s, Sending), (
        "Send from Idle failed after cancel acknowledgement (pre-fix #973)"
    )


def test_973_cancel_fire_not_idempotent() -> None:
    """CancelFire cannot fire twice in the same turn — second is rejected.

    Once the cancel has fired (→Draining), a second CancelFire is
    rejected.  This prevents a stale cancel from double-cancelling a
    drain already in progress.
    """
    result = transition(Draining(), CancelFire())
    assert result is None


# ---------------------------------------------------------------------------
# Bug #975 — switch_model hung with _in_turn=True after preempt-cancelled drain
#
# Root cause: _drain_to_boundary aborted early when _cancel was still set,
# leaving _in_turn=True (here: Draining) without consuming the actual
# type=result boundary.  Then switch_model called _send_control_set_model —
# effectively another stdin write — while the prior turn was still open.
# Claude had not finished its current turn, so the control_response for
# the set_model request never arrived.  Permanent hang.
#
# FSM fix: drain_terminates — Send is rejected from Draining; the only
# valid exit is TurnReturn.  The drain is always finite.
# ---------------------------------------------------------------------------


def test_975_send_rejected_from_draining() -> None:
    """Send (e.g. switch_model's control_request) is rejected from Draining.

    Pre-fix race (#975): drain aborted early, _in_turn left True (Draining).
    switch_model then called _send_control_set_model — a stdin write — while
    the session was still draining.  The FSM rejects Send from Draining.
    """
    result = transition(Draining(), Send())
    assert result is None, (
        "drain_terminates violated: Send accepted from Draining (pre-fix #975 hang)"
    )


def test_975_drain_observe_loops_in_draining() -> None:
    """DrainObserve stays in Draining — the drain loop is well-defined.

    Any number of drain observations cycle back to Draining.  The drain
    is bounded: only TurnReturn exits.
    """
    s: State = Draining()
    for _ in range(5):
        result = transition(s, DrainObserve())
        assert isinstance(result, Draining)


def test_975_drain_terminates_via_turn_return() -> None:
    """TurnReturn is always available from Draining and exits in one step.

    Proves drain_terminates: the drain path Draining→TurnReturn→Cancelled
    is finite — there is no cycle that keeps the FSM in Draining without
    admitting a TurnReturn.
    """
    s: State = Draining()
    s = transition(s, TurnReturn())  # type: ignore[assignment]
    assert isinstance(s, Cancelled), (
        "drain_terminates violated: TurnReturn from Draining did not reach Cancelled "
        "(pre-fix #975)"
    )
    # Full acknowledgement returns to Idle — the session is clean.
    s = transition(s, TurnReturn())  # type: ignore[assignment]
    assert isinstance(s, Idle)


# ---------------------------------------------------------------------------
# Supplementary: empty_result_is_not_completion
#
# A turn that returns normally (no cancel) always reaches Idle via
# AwaitingReply, never Cancelled.  Cancelled is only reachable through
# the drain path.  This guards the semantic contract: an empty result
# does not masquerade as a successful completion (#973 symptom).
# ---------------------------------------------------------------------------


def test_normal_turn_returns_to_idle_not_cancelled() -> None:
    """TurnReturn from AwaitingReply yields Idle, not Cancelled.

    A normal (uncancelled) turn ending in AwaitingReply must reach Idle.
    The empty_result_is_not_completion invariant ensures that a TurnReturn
    without prior CancelFire cannot produce the Cancelled state.
    """
    result = transition(AwaitingReply(), TurnReturn())
    assert isinstance(result, Idle), (
        "empty_result_is_not_completion violated: "
        "TurnReturn from AwaitingReply did not reach Idle"
    )
    # Confirm Cancelled is unreachable on the normal path.
    assert not isinstance(result, Cancelled)


def test_cancelled_only_reachable_via_drain_path() -> None:
    """Cancelled is reachable only through CancelFire→Draining→TurnReturn.

    Demonstrates that the Cancelled state is not reachable from the
    normal (uncancelled) turn path.
    """
    # Normal path: no Cancelled in sight.
    s: State = Idle()
    s = transition(s, Send())  # type: ignore[assignment]
    s = transition(s, ReplyChunk())  # type: ignore[assignment]
    s = transition(s, TurnReturn())  # type: ignore[assignment]
    assert isinstance(s, Idle)
    assert not isinstance(s, Cancelled)

    # Drain path: Cancelled is reached exactly when expected.
    s = transition(s, Send())  # type: ignore[assignment]
    s = transition(s, ReplyChunk())  # type: ignore[assignment]
    s = transition(s, CancelFire())  # type: ignore[assignment]
    s = transition(s, TurnReturn())  # type: ignore[assignment]
    assert isinstance(s, Cancelled)


# ---------------------------------------------------------------------------
# _stream_transition crash-loud behaviour
#
# Verify that ClaudeSession._stream_transition raises AssertionError (not
# silently no-ops) when the FSM rejects an event.  This is the fail-closed
# contract: a protocol violation surfaces as an immediate crash rather than
# silent state divergence.
# ---------------------------------------------------------------------------


def test_stream_transition_crashes_on_invalid_event(tmp_path: Path) -> None:
    """_stream_transition raises AssertionError when the FSM rejects an event.

    Wires in a real ClaudeSession in Idle state and fires CancelFire —
    which the FSM rejects (Idle + CancelFire → None).  The oracle must
    raise AssertionError immediately, not silently accept the invalid event.
    """
    from unittest.mock import MagicMock

    from fido.claude import ClaudeSession

    system_file = tmp_path / "system.md"
    system_file.write_text("sys")
    proc = MagicMock()
    proc.pid = 1
    proc.stdin = MagicMock()
    proc.stdout = MagicMock()
    proc.stderr = MagicMock()
    proc.poll = MagicMock(return_value=None)
    fake_popen = MagicMock(return_value=proc)
    fake_selector = MagicMock(return_value=([], [], []))

    session = ClaudeSession(
        system_file, work_dir=tmp_path, popen=fake_popen, selector=fake_selector
    )
    assert isinstance(session._stream_state, Idle)

    with pytest.raises(AssertionError, match="claude_session FSM"):
        session._stream_transition(CancelFire())  # Idle + CancelFire → None

    session.stop()
