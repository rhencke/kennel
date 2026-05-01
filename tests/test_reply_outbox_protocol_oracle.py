from fido.rocq import reply_outbox_protocol as oracle


def _prepared_reply() -> oracle.ProtocolState:
    state = oracle.prepare_reply(
        10,
        20,
        30,
        40,
        oracle.ReviewThreadOrigin(),
        oracle.empty_protocol_state,
    )
    assert state is not None
    return state


def _claimed_reply() -> oracle.ProtocolState:
    state = oracle.claim_outbox_effect(40, _prepared_reply())
    assert state is not None
    return state


def _posted_reply() -> oracle.ProtocolState:
    state = oracle.record_reply_posted(40, 50, _claimed_reply())
    assert state is not None
    return state


def test_claim_before_generate() -> None:
    prepared = _prepared_reply()

    assert not oracle.can_generate_reply(oracle.empty_protocol_state, 30, 20)
    assert oracle.can_generate_reply(prepared, 30, 20)
    assert not oracle.can_generate_reply(prepared, 31, 20)
    assert oracle.delivery_origin(prepared, 10) == 20


def test_duplicate_semantic_origin_is_rejected() -> None:
    prepared = _prepared_reply()

    duplicate = oracle.prepare_reply(
        11,
        20,
        31,
        41,
        oracle.ReviewThreadOrigin(),
        prepared,
    )

    assert duplicate is None


def test_claim_before_visible_reply_post() -> None:
    prepared = _prepared_reply()
    claimed = _claimed_reply()

    assert oracle.record_reply_posted(40, 50, prepared) is None
    assert isinstance(oracle.outbox_decision(prepared, 40), oracle.EmitEffect)
    assert isinstance(
        oracle.outbox_decision(claimed, 40),
        oracle.WaitForInFlightEffect,
    )


def test_visible_reply_post_is_idempotent() -> None:
    posted = _posted_reply()
    replay = oracle.record_reply_posted(40, 51, posted)

    assert replay == posted
    assert oracle.live_reply_for_origin(posted, 20) == 50
    assert oracle.effect_external(posted, 40) == 50
    assert oracle.origin_completed(posted, 20)
    assert isinstance(
        oracle.outbox_decision(posted, 40),
        oracle.ReuseDeliveredEffect,
    )


def test_deferred_issue_effect_is_idempotent() -> None:
    posted = _posted_reply()
    with_issue = oracle.prepare_deferred_issue(60, 20, 30, posted)
    assert with_issue is not None
    claimed_issue = oracle.claim_outbox_effect(60, with_issue)
    assert claimed_issue is not None
    opened = oracle.record_deferred_issue_opened(60, 70, claimed_issue)
    assert opened is not None

    replay = oracle.record_deferred_issue_opened(60, 71, opened)

    assert replay == opened
    assert oracle.live_issue_for_effect(opened, 60) == 70
    assert oracle.effect_external(opened, 60) == 70
    assert isinstance(
        oracle.outbox_decision(opened, 60),
        oracle.ReuseDeliveredEffect,
    )


def test_deferred_issue_rejects_second_effect_for_same_intent() -> None:
    posted = _posted_reply()
    with_issue = oracle.prepare_deferred_issue(60, 20, 30, posted)
    assert with_issue is not None

    duplicate = oracle.prepare_deferred_issue(61, 20, 30, with_issue)

    assert duplicate is None


def test_failed_reply_post_releases_origin_for_retry() -> None:
    claimed = _claimed_reply()

    failed = oracle.record_outbox_failure(40, claimed)

    assert failed is not None
    assert oracle.origin_claimable(failed, 20)
    assert isinstance(oracle.outbox_decision(failed, 40), oracle.RetryLaterEffect)
    retry = oracle.prepare_reply(
        12,
        20,
        32,
        42,
        oracle.ReviewThreadOrigin(),
        failed,
    )
    assert retry is not None
