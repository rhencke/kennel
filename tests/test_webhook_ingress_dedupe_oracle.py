"""Regression tests for the webhook_ingress_dedupe FSM oracle.

Each test section corresponds to a proved invariant from
``models/webhook_ingress_dedupe.v``.  The tests exercise the extracted
``transition`` function directly and verify the machine-checked guarantees,
then exercise the ``WebhookIngressOracle`` class that wraps it.

Proved invariants exercised:

  ``exactly_once_dispatch``            — Arrive accepted from Fresh → Dispatched;
                                          rejected from Dispatched and Collapsed.

  ``redeliver_is_suppressed``          — Redeliver from Dispatched stays Dispatched;
                                          suppressed by the oracle (no double-fire).

  ``collapse_is_terminal``             — All events rejected from Collapsed; the
                                          collapsed state is a terminal sink.

  ``redeliver_requires_prior_dispatch`` — Redeliver rejected from Fresh and
                                          Collapsed; only a Dispatched slot holds a
                                          redeliverable prior delivery.
"""

import pytest

from fido.rocq.webhook_ingress_dedupe import (
    Arrive,
    Collapsed,
    CollapseReview,
    Dispatched,
    Event,
    Fresh,
    Redeliver,
    State,
    transition,
)

# ---------------------------------------------------------------------------
# Invariant: exactly_once_dispatch
#
# Arrive is accepted from Fresh (first-ever delivery) and yields Dispatched.
# Arrive is rejected from Dispatched and Collapsed — the delivery has already
# been handled or collapsed.  This machine-checks the at-most-once dispatch
# guarantee: a delivery ID is dispatched exactly once per process lifetime.
# ---------------------------------------------------------------------------


def test_exactly_once_dispatch_arrive_from_fresh() -> None:
    """Arrive from Fresh yields Dispatched — first delivery is accepted.

    exactly_once_dispatch: a delivery ID seen for the first time starts in
    Fresh.  Arrive always succeeds from Fresh, and the resulting Dispatched
    state records that the delivery has been acted upon.
    """
    result = transition(Fresh(), Arrive())
    assert isinstance(result, Dispatched), (
        f"exactly_once_dispatch violated: Arrive from Fresh "
        f"yielded {type(result).__name__!r} instead of Dispatched"
    )


def test_exactly_once_dispatch_arrive_rejected_from_dispatched() -> None:
    """Arrive is rejected from Dispatched — a second arrival for the same delivery.

    exactly_once_dispatch: the delivery was already dispatched in a prior call.
    Accepting another Arrive from Dispatched would double-fire the event and
    violate the exactly-once guarantee.
    """
    result = transition(Dispatched(), Arrive())
    assert result is None, (
        "exactly_once_dispatch violated: Arrive accepted from Dispatched"
    )


def test_exactly_once_dispatch_arrive_rejected_from_collapsed() -> None:
    """Arrive is rejected from Collapsed — collapsed deliveries cannot be re-arrived.

    exactly_once_dispatch: the delivery was collapsed (CollapseReview fired
    from Fresh) and is now in the terminal Collapsed state.  Any further
    Arrive is rejected.
    """
    result = transition(Collapsed(), Arrive())
    assert result is None, (
        "exactly_once_dispatch violated: Arrive accepted from Collapsed"
    )


# ---------------------------------------------------------------------------
# Invariant: redeliver_is_suppressed
#
# Redeliver from Dispatched stays in Dispatched.  GitHub's at-least-once
# delivery guarantee means the same delivery ID may arrive more than once.
# The FSM accepts Redeliver (keeps the state unchanged) so the oracle can
# suppress the duplicate without crashing.
# ---------------------------------------------------------------------------


def test_redeliver_is_suppressed_from_dispatched() -> None:
    """Redeliver from Dispatched stays Dispatched — duplicate delivery suppressed.

    redeliver_is_suppressed: a redelivered webhook carries the same delivery
    ID as the original.  The FSM accepts Redeliver from Dispatched and stays
    in Dispatched, signalling to the oracle that this delivery should be
    suppressed rather than re-dispatched.
    """
    result = transition(Dispatched(), Redeliver())
    assert isinstance(result, Dispatched), (
        f"redeliver_is_suppressed violated: Redeliver from Dispatched "
        f"yielded {type(result).__name__!r} instead of Dispatched"
    )


# ---------------------------------------------------------------------------
# Invariant: collapse_is_terminal
#
# All events are rejected from Collapsed.  CollapseReview fires once for a
# pull_request_review / submitted event that is handled via its inline
# comments — further arrivals, redeliveries, or collapse requests for the
# same delivery ID are all suppressed.
# ---------------------------------------------------------------------------


def test_collapse_is_terminal_arrive_rejected() -> None:
    """Arrive is rejected from Collapsed — the collapsed delivery is done.

    collapse_is_terminal: once a delivery is collapsed, no event can reopen
    it.  Arrive from Collapsed returns None so the oracle suppresses it.
    """
    result = transition(Collapsed(), Arrive())
    assert result is None, (
        "collapse_is_terminal violated: Arrive accepted from Collapsed"
    )


def test_collapse_is_terminal_redeliver_rejected() -> None:
    """Redeliver is rejected from Collapsed — no redeliver on a collapsed slot.

    collapse_is_terminal: a collapsed delivery was never dispatched (it was
    collapsed before dispatch), so Redeliver has no predecessor to re-deliver.
    """
    result = transition(Collapsed(), Redeliver())
    assert result is None, (
        "collapse_is_terminal violated: Redeliver accepted from Collapsed"
    )


def test_collapse_is_terminal_collapse_review_rejected() -> None:
    """CollapseReview is rejected from Collapsed — cannot collapse twice.

    collapse_is_terminal: the collapsed state is a terminal sink.  Even
    CollapseReview — the event that created the Collapsed state — is rejected
    from Collapsed.
    """
    result = transition(Collapsed(), CollapseReview())
    assert result is None, (
        "collapse_is_terminal violated: CollapseReview accepted from Collapsed"
    )


# ---------------------------------------------------------------------------
# Invariant: redeliver_requires_prior_dispatch
#
# Redeliver is rejected from Fresh and Collapsed.  A redelivery implies there
# was a prior delivery that was dispatched; an absent (Fresh) or collapsed slot
# has no such predecessor.
# ---------------------------------------------------------------------------


def test_redeliver_requires_prior_dispatch_fresh() -> None:
    """Redeliver is rejected from Fresh — no prior dispatch to redeliver.

    redeliver_requires_prior_dispatch: the delivery ID is appearing for the
    first time (Fresh).  There is no prior dispatch, so a Redeliver is
    semantically impossible and the FSM rejects it.
    """
    result = transition(Fresh(), Redeliver())
    assert result is None, (
        "redeliver_requires_prior_dispatch violated: Redeliver accepted from Fresh"
    )


def test_redeliver_requires_prior_dispatch_collapsed() -> None:
    """Redeliver is rejected from Collapsed — collapsed deliveries were never dispatched.

    redeliver_requires_prior_dispatch: a collapsed delivery went
    Fresh→Collapsed without ever becoming Dispatched.  There is no prior
    dispatch to redeliver, so Redeliver is rejected from Collapsed.
    """
    result = transition(Collapsed(), Redeliver())
    assert result is None, (
        "redeliver_requires_prior_dispatch violated: Redeliver accepted from Collapsed"
    )


# ---------------------------------------------------------------------------
# Supplementary: CollapseReview accepted from Fresh
#
# CollapseReview from Fresh yields Collapsed — the pull_request_review /
# submitted event is collapsed on first delivery.  This is the sole path to
# the Collapsed terminal state.
# ---------------------------------------------------------------------------


def test_collapse_review_from_fresh_yields_collapsed() -> None:
    """CollapseReview from Fresh yields Collapsed — the only path to Collapsed.

    A pull_request_review / submitted delivery fires CollapseReview on its
    first arrival.  The FSM accepts it from Fresh and moves to the terminal
    Collapsed state so the oracle suppresses the review-level event.
    """
    result = transition(Fresh(), CollapseReview())
    assert isinstance(result, Collapsed), (
        f"CollapseReview from Fresh yielded {type(result).__name__!r} instead of Collapsed"
    )


def test_collapse_review_rejected_from_dispatched() -> None:
    """CollapseReview is rejected from Dispatched.

    A delivery that was already dispatched cannot be retroactively collapsed.
    """
    result = transition(Dispatched(), CollapseReview())
    assert result is None, (
        "CollapseReview accepted from Dispatched — should be rejected"
    )


# ---------------------------------------------------------------------------
# Exhaustive state × event matrix
#
# All 9 (state, event) pairs — verifies the full transition table so no arm
# is accidentally unreachable and any future Rocq model change is caught here.
# ---------------------------------------------------------------------------


def test_all_transitions_exhaustive() -> None:
    """Every state×event pair has a deterministic transition result.

    Walks the full 3×3 matrix and asserts that the output type matches the
    transition table in ``models/webhook_ingress_dedupe.v``.  Any future
    change to the Rocq model that adds or removes a valid transition will
    be caught here.
    """
    expected: dict[tuple[type[State], type[Event]], type[State] | None] = {
        # Fresh
        (Fresh, Arrive): Dispatched,
        (Fresh, Redeliver): None,
        (Fresh, CollapseReview): Collapsed,
        # Dispatched
        (Dispatched, Arrive): None,
        (Dispatched, Redeliver): Dispatched,
        (Dispatched, CollapseReview): None,
        # Collapsed
        (Collapsed, Arrive): None,
        (Collapsed, Redeliver): None,
        (Collapsed, CollapseReview): None,
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
# WebhookIngressOracle integration tests
#
# Verify the check_dispatch() method drives the FSM correctly for the common
# scenarios and that the crash-on-violation behaviour is upheld.
# ---------------------------------------------------------------------------


def test_oracle_normal_dispatch_flow() -> None:
    """First delivery is accepted; same delivery ID again is suppressed.

    Normal flow: check_dispatch returns a non-None state on the first call
    (first arrival → Dispatched), then returns None on the second call
    (Redeliver → suppressed).
    """
    from fido.events import WebhookIngressOracle

    oracle = WebhookIngressOracle()

    first = oracle.check_dispatch("owner/repo", "delivery-1")
    assert first is not None, "first delivery should be accepted"
    assert isinstance(first, Dispatched)

    second = oracle.check_dispatch("owner/repo", "delivery-1")
    assert second is None, "duplicate delivery should be suppressed"


def test_oracle_double_fire_collapse_review() -> None:
    """CollapseReview suppresses a pull_request_review / submitted delivery.

    When collapse_review=True the oracle fires CollapseReview from Fresh →
    Collapsed and returns None to suppress the review-level event.
    """
    from fido.events import WebhookIngressOracle

    oracle = WebhookIngressOracle()

    result = oracle.check_dispatch(
        "owner/repo", "delivery-review-1", collapse_review=True
    )
    assert result is None, (
        "CollapseReview delivery should be suppressed (oracle returned non-None)"
    )


def test_oracle_different_delivery_ids_tracked_independently() -> None:
    """Different delivery IDs are tracked independently within the same repo.

    Two distinct delivery IDs both start Fresh.  Each is dispatched on its
    first arrival independently; neither suppresses the other.
    """
    from fido.events import WebhookIngressOracle

    oracle = WebhookIngressOracle()

    first = oracle.check_dispatch("owner/repo", "delivery-A")
    assert first is not None, "delivery-A first arrival should be accepted"

    second = oracle.check_dispatch("owner/repo", "delivery-B")
    assert second is not None, "delivery-B first arrival should be accepted"

    # Redispatching delivery-A should now be suppressed; delivery-B still independent.
    first_again = oracle.check_dispatch("owner/repo", "delivery-A")
    assert first_again is None, "delivery-A re-arrival should be suppressed"

    second_again = oracle.check_dispatch("owner/repo", "delivery-B")
    assert second_again is None, "delivery-B re-arrival should be suppressed"


def test_oracle_different_repos_tracked_independently() -> None:
    """The same delivery ID in different repos is tracked independently.

    repo-A and repo-B each have their own FSM table; a dispatch in repo-A
    does not affect the Fresh state for the same ID in repo-B.
    """
    from fido.events import WebhookIngressOracle

    oracle = WebhookIngressOracle()

    result_a = oracle.check_dispatch("owner/repo-a", "delivery-shared")
    assert result_a is not None, "repo-a first arrival should be accepted"

    result_b = oracle.check_dispatch("owner/repo-b", "delivery-shared")
    assert result_b is not None, (
        "repo-b first arrival should be accepted (different repo)"
    )


def test_oracle_crash_on_violation_redeliver_from_fresh() -> None:
    """WebhookIngressOracle raises AssertionError when the FSM rejects Redeliver from Fresh.

    The oracle is crash-on-violation: if internal state somehow reaches a
    configuration where Redeliver fires from Fresh, _transition must raise
    AssertionError immediately rather than silently accepting the invalid event.
    This tests the fail-closed contract by injecting a Fresh state directly
    and calling _transition with Redeliver.
    """
    from fido.events import WebhookIngressOracle

    oracle = WebhookIngressOracle()
    # Pre-seed a Fresh state directly so we can fire an invalid event.
    oracle._states["owner/repo"] = {"delivery-x": Fresh()}

    with pytest.raises(AssertionError, match="webhook_ingress_dedupe FSM"):
        oracle._transition("owner/repo", "delivery-x", Redeliver(), Fresh())


def test_oracle_crash_on_violation_arrive_from_dispatched() -> None:
    """WebhookIngressOracle raises AssertionError when Arrive fires from Dispatched.

    Arrive from Dispatched is the double-fire violation (would dispatch the
    same delivery a second time).  The oracle must crash loudly so the bug
    surfaces immediately.
    """
    from fido.events import WebhookIngressOracle

    oracle = WebhookIngressOracle()

    with pytest.raises(AssertionError, match="webhook_ingress_dedupe FSM"):
        oracle._transition("owner/repo", "delivery-y", Arrive(), Dispatched())
