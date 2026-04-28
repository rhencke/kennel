"""Regression tests for the handler_preemption FSM oracle.

Each test section corresponds to a proved invariant from
``models/handler_preemption.v``.  The tests exercise the extracted
``transition`` function directly and verify the four machine-checked
guarantees.

Proved invariants exercised:

  ``worker_blocked_when_nonempty``      — WorkerTurnStart is rejected from
    NonEmpty; the worker must not start a new provider turn while untriaged
    webhooks are pending.

  ``handler_done_rejected_from_empty``  — HandlerDone is rejected from Empty;
    an exit_untriaged without a matching enter_untriaged is a count underflow.

  ``worker_turn_proceeds_when_empty``   — WorkerTurnStart from Empty yields
    Some Empty; when no webhooks are pending, the worker proceeds normally.

  ``webhook_arrival_always_accepted``   — WebhookArrives is accepted from both
    Empty and NonEmpty, always yielding NonEmpty.

Field lesson covered:

  Worker grinds through an in-progress task without checking for pending
  webhooks — fresh comments and CI events are starved (#1067).  The
  ``worker_blocked_when_nonempty`` invariant machine-checks that this class
  of bug cannot recur: WorkerTurnStart is always rejected from NonEmpty.
"""

from fido.rocq.handler_preemption import (
    Empty,
    HandlerDone,
    NonEmpty,
    WebhookArrives,
    WorkerTurnStart,
    transition,
)

# ---------------------------------------------------------------------------
# Invariant: worker_blocked_when_nonempty
#
# WorkerTurnStart is rejected from NonEmpty.  The worker must not start
# a new provider turn while untriaged webhooks are pending — it must
# yield at the turn boundary until the inbox drains.
# ---------------------------------------------------------------------------


def test_worker_turn_rejected_from_nonempty() -> None:
    """WorkerTurnStart is rejected from NonEmpty.

    worker_blocked_when_nonempty: the inbox has at least one untriaged
    webhook.  The worker must yield — starting a new turn would starve
    the pending handler.
    """
    result = transition(NonEmpty(), WorkerTurnStart())
    assert result is None, (
        "worker_blocked_when_nonempty violated: WorkerTurnStart accepted from NonEmpty"
    )


# ---------------------------------------------------------------------------
# Invariant: handler_done_rejected_from_empty
#
# HandlerDone is rejected from Empty.  An exit_untriaged call without a
# matching enter_untriaged is a count underflow — the runtime logs a
# warning and the FSM refuses the transition.
# ---------------------------------------------------------------------------


def test_handler_done_rejected_from_empty() -> None:
    """HandlerDone is rejected from Empty — underflow.

    handler_done_rejected_from_empty: the inbox is already empty.  A
    HandlerDone event without a preceding WebhookArrives is a bug in
    the enter/exit bookkeeping.
    """
    result = transition(Empty(), HandlerDone())
    assert result is None, (
        "handler_done_rejected_from_empty violated: HandlerDone accepted from Empty"
    )


# ---------------------------------------------------------------------------
# Invariant: worker_turn_proceeds_when_empty
#
# WorkerTurnStart from Empty yields Some Empty.  When no webhooks are
# pending, the worker proceeds to its next provider turn.
# ---------------------------------------------------------------------------


def test_worker_turn_accepted_from_empty() -> None:
    """WorkerTurnStart from Empty yields Empty.

    worker_turn_proceeds_when_empty: no untriaged webhooks pending.
    The worker starts its turn and the inbox stays empty.
    """
    result = transition(Empty(), WorkerTurnStart())
    assert isinstance(result, Empty), (
        "worker_turn_proceeds_when_empty violated: "
        f"WorkerTurnStart from Empty yielded {result!r}, expected Empty"
    )


def test_worker_turn_from_empty_preserves_state() -> None:
    """Repeated WorkerTurnStart calls from Empty always yield Empty.

    The worker can call provider_run() multiple times without any webhook
    activity and the FSM stays in the same state.
    """
    state = Empty()
    for _ in range(5):
        result = transition(state, WorkerTurnStart())
        assert isinstance(result, Empty)
        state = result


# ---------------------------------------------------------------------------
# Invariant: webhook_arrival_always_accepted
#
# WebhookArrives is accepted from both Empty and NonEmpty, always
# yielding NonEmpty.  A new webhook can always enter the inbox.
# ---------------------------------------------------------------------------


def test_webhook_arrives_from_empty() -> None:
    """WebhookArrives from Empty yields NonEmpty.

    webhook_arrival_always_accepted: first webhook enters the inbox.
    """
    result = transition(Empty(), WebhookArrives())
    assert isinstance(result, NonEmpty), (
        "webhook_arrival_always_accepted violated: "
        f"WebhookArrives from Empty yielded {result!r}, expected NonEmpty"
    )


def test_webhook_arrives_from_nonempty() -> None:
    """WebhookArrives from NonEmpty yields NonEmpty.

    webhook_arrival_always_accepted: another webhook arrives while the
    inbox already has pending handlers.
    """
    result = transition(NonEmpty(), WebhookArrives())
    assert isinstance(result, NonEmpty), (
        "webhook_arrival_always_accepted violated: "
        f"WebhookArrives from NonEmpty yielded {result!r}, expected NonEmpty"
    )


# ---------------------------------------------------------------------------
# HandlerDone from NonEmpty — stays NonEmpty
#
# The FSM abstracts {Empty, NonEmpty} without tracking the count.
# HandlerDone from NonEmpty always yields NonEmpty in the FSM;
# the runtime counter decides when to reset to Empty.
# ---------------------------------------------------------------------------


def test_handler_done_from_nonempty_stays_nonempty() -> None:
    """HandlerDone from NonEmpty yields NonEmpty.

    The FSM does not track the count — it stays NonEmpty until the
    runtime resets the state to Empty when the counter hits zero.
    """
    result = transition(NonEmpty(), HandlerDone())
    assert isinstance(result, NonEmpty), (
        f"HandlerDone from NonEmpty yielded {result!r}, expected NonEmpty"
    )


# ---------------------------------------------------------------------------
# Full lifecycle sequences
# ---------------------------------------------------------------------------


def test_full_cycle_arrive_handler_done_worker_turn() -> None:
    """Full cycle: Empty → arrive → NonEmpty → (runtime reset to Empty) →
    worker turn → Empty.

    This mirrors the runtime flow: a webhook arrives, the handler processes
    it, the inbox drains (runtime resets FSM to Empty), then the worker
    starts its turn.
    """
    state: Empty | NonEmpty = Empty()
    # Webhook arrives
    result = transition(state, WebhookArrives())
    assert isinstance(result, NonEmpty)
    # Handler finishes — FSM stays NonEmpty but runtime would reset to Empty
    result2 = transition(result, HandlerDone())
    assert isinstance(result2, NonEmpty)
    # Runtime resets to Empty (simulated)
    state = Empty()
    # Worker turn now proceeds
    result3 = transition(state, WorkerTurnStart())
    assert isinstance(result3, Empty)


def test_multiple_arrivals_then_worker_blocked() -> None:
    """Multiple WebhookArrives pile up — WorkerTurnStart is still rejected."""
    state: Empty | NonEmpty = Empty()
    for _ in range(5):
        result = transition(state, WebhookArrives())
        assert isinstance(result, NonEmpty)
        state = result
    # Worker blocked
    assert transition(state, WorkerTurnStart()) is None
