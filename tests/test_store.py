import sqlite3
import threading
from contextlib import closing
from pathlib import Path

import pytest

from fido.rocq import replied_comment_claims as oracle
from fido.store import (
    FidoStore,
    ReplyOutboxEffectRecord,
    ReplyOwner,
    ReplyPromiseRecord,
    append_reply_promise_marker,
    append_reply_promise_markers,
    extract_reply_promise_ids,
)


def _oracle_promise_state(
    promises: dict[int, oracle.PromiseRow], promise_id: int
) -> oracle.PromiseState:
    return promises[promise_id].promise_state


def _oracle_claim_state(
    claims: dict[int, oracle.ClaimRow], comment_id: int
) -> oracle.ClaimState:
    return claims[comment_id].claim_state


def _oracle_owner(owner: ReplyOwner) -> oracle.ClaimOwner:
    match owner:
        case "webhook":
            return oracle.OwnerWebhook()
        case "worker":
            return oracle.OwnerWorker()
        case "recovery":
            return oracle.OwnerRecovery()


def _state_name(state: object) -> str:
    if isinstance(state, str):
        return {
            "prepared": "PromisePrepared",
            "posted": "PromisePosted",
            "acked": "PromiseAcked",
            "failed": "PromiseFailed",
            "in_progress": "ClaimInProgress",
            "completed": "ClaimCompleted",
            "retryable_failed": "ClaimRetryableFailed",
        }[state]
    return type(state).__name__


def _assert_store_matches_oracle(
    store: FidoStore,
    store_promise: ReplyPromiseRecord,
    *,
    oracle_claims: dict[int, oracle.ClaimRow],
    oracle_promises: dict[int, oracle.PromiseRow],
    oracle_promise_id: int = 1,
) -> None:
    persisted = store.promise(store_promise.promise_id)
    assert persisted is not None
    oracle_row = oracle_promises[oracle_promise_id]
    assert persisted.anchor_comment_id == oracle_row.promise_anchor_comment
    assert persisted.covered_comment_ids == tuple(oracle_row.promise_covered_comments)
    assert _state_name(persisted.state) == _state_name(oracle_row.promise_state)
    for comment_id in persisted.covered_comment_ids:
        assert (
            store.claim_state(comment_id)
            == {
                "ClaimInProgress": "in_progress",
                "ClaimCompleted": "completed",
                "ClaimRetryableFailed": "retryable_failed",
            }[_state_name(_oracle_claim_state(oracle_claims, comment_id))]
        )


def test_prepare_claim_is_atomic_and_blocks_duplicate_owner(tmp_path: Path) -> None:
    store = FidoStore(tmp_path)

    first = store.prepare_reply(
        owner="webhook", comment_type="issues", anchor_comment_id=101
    )
    second = store.prepare_reply(
        owner="worker", comment_type="issues", anchor_comment_id=101
    )

    assert first is not None
    assert second is None
    assert store.is_claimed_or_completed(101)


def test_prepare_claim_adds_anchor_when_covered_list_omits_it(tmp_path: Path) -> None:
    store = FidoStore(tmp_path)

    promise = store.prepare_reply(
        owner="webhook",
        comment_type="pulls",
        anchor_comment_id=101,
        covered_comment_ids=[102, 103],
    )

    assert promise is not None
    assert promise.covered_comment_ids == (101, 102, 103)
    stored = store.promise(promise.promise_id)
    assert stored is not None
    assert stored.covered_comment_ids == (101, 102, 103)


def test_missing_claim_and_missing_promise_are_noops(tmp_path: Path) -> None:
    store = FidoStore(tmp_path)

    assert not store.is_claimed_or_completed(999)
    assert store.claim_state(999) is None
    assert store.promise("00000000-0000-0000-0000-000000000000") is None
    store.ack_promise("00000000-0000-0000-0000-000000000000")
    store.mark_failed("00000000-0000-0000-0000-000000000000")


def test_failed_claim_becomes_retryable(tmp_path: Path) -> None:
    store = FidoStore(tmp_path)
    first = store.prepare_reply(
        owner="webhook", comment_type="pulls", anchor_comment_id=102
    )
    assert first is not None

    store.mark_failed(first.promise_id)

    failed = store.promise(first.promise_id)
    assert failed is not None
    assert failed.state == "failed"
    assert failed.next_retry_after is not None
    assert store.claim_state(102) == "retryable_failed"
    assert (
        store.prepare_reply(
            owner="recovery", comment_type="pulls", anchor_comment_id=102
        )
        is None
    )


def test_prepare_reply_allows_new_comment_in_thread_with_completed_ancestor(
    tmp_path: Path,
) -> None:
    """Regression for #1188: a new review comment whose lineage includes a
    *completed* ancestor (Fido already replied to an earlier comment in the
    same thread) must still get its own promise — the prior reply must not
    block the new one."""
    store = FidoStore(tmp_path)

    # First comment in the thread arrives, Fido replies, claim completes.
    root = store.prepare_reply(
        owner="webhook", comment_type="pulls", anchor_comment_id=2001
    )
    assert root is not None
    store.ack_promise(root.promise_id)
    assert store.claim_state(2001) == "completed"

    # New review comment arrives in the same thread. Lineage covers the
    # completed root plus the new comment. prepare_reply must succeed.
    new_reply = store.prepare_reply(
        owner="webhook",
        comment_type="pulls",
        anchor_comment_id=2003,
        covered_comment_ids=[2001, 2002, 2003],
    )

    assert new_reply is not None
    assert new_reply.anchor_comment_id == 2003


def test_prepare_reply_blocks_concurrent_in_progress_lineage_member(
    tmp_path: Path,
) -> None:
    """A concurrent in_progress claim on any covered comment still blocks —
    a sibling handler is currently coalescing this thread, so a second
    handler must back off."""
    store = FidoStore(tmp_path)

    in_flight = store.prepare_reply(
        owner="webhook", comment_type="pulls", anchor_comment_id=3001
    )
    assert in_flight is not None
    # in_flight is left in_progress (no ack, no fail).

    # A new comment lands in the same thread before the first reply
    # finishes. Its lineage includes the still-in_progress 3001.
    contended = store.prepare_reply(
        owner="worker",
        comment_type="pulls",
        anchor_comment_id=3002,
        covered_comment_ids=[3001, 3002],
    )

    assert contended is None


def test_retryable_claim_without_backoff_is_claimable(tmp_path: Path) -> None:
    store = FidoStore(tmp_path)
    promise = store.prepare_reply(
        owner="webhook", comment_type="pulls", anchor_comment_id=103
    )
    assert promise is not None

    with closing(sqlite3.connect(store.db_path)) as conn:
        conn.execute(
            """
            UPDATE comment_claims
            SET state = 'retryable_failed', next_retry_after = NULL
            WHERE comment_id = ?
            """,
            (103,),
        )
        conn.commit()

    assert not store.is_claimed_or_completed(103)


def test_ack_promise_completes_every_covered_comment(tmp_path: Path) -> None:
    store = FidoStore(tmp_path)
    promise = store.prepare_reply(
        owner="webhook",
        comment_type="pulls",
        anchor_comment_id=201,
        covered_comment_ids=[201, 202, 203],
    )
    assert promise is not None

    store.mark_posted(promise.promise_id)
    store.ack_promise(promise.promise_id)

    stored = store.promise(promise.promise_id)
    assert stored is not None
    assert stored.state == "acked"
    assert (
        store.prepare_reply(owner="worker", comment_type="pulls", anchor_comment_id=202)
        is None
    )


def test_store_prepare_reply_matches_oracle(tmp_path: Path) -> None:
    store = FidoStore(tmp_path)
    promise = store.prepare_reply(
        owner="webhook",
        comment_type="pulls",
        anchor_comment_id=601,
        covered_comment_ids=[602, 603],
    )
    assert promise is not None

    prepared = oracle.prepare_claims(
        _oracle_owner("webhook"),
        1,
        601,
        [602, 603],
        {},
        {},
    )
    assert prepared is not None
    claims, promises = prepared

    _assert_store_matches_oracle(
        store,
        promise,
        oracle_claims=claims,
        oracle_promises=promises,
    )


def test_store_mark_posted_and_ack_match_oracle(tmp_path: Path) -> None:
    store = FidoStore(tmp_path)
    promise = store.prepare_reply(
        owner="worker",
        comment_type="issues",
        anchor_comment_id=611,
        covered_comment_ids=[612],
    )
    assert promise is not None

    prepared = oracle.prepare_claims(
        _oracle_owner("worker"),
        1,
        611,
        [612],
        {},
        {},
    )
    assert prepared is not None
    claims, promises = prepared
    promises = oracle.mark_promise_posted(1, promises)
    store.mark_posted(promise.promise_id)
    _assert_store_matches_oracle(
        store,
        promise,
        oracle_claims=claims,
        oracle_promises=promises,
    )

    claims, promises = oracle.ack_promise(1, claims, promises)
    store.ack_promise(promise.promise_id)
    _assert_store_matches_oracle(
        store,
        promise,
        oracle_claims=claims,
        oracle_promises=promises,
    )


def test_store_mark_failed_matches_oracle(tmp_path: Path) -> None:
    store = FidoStore(tmp_path)
    promise = store.prepare_reply(
        owner="recovery",
        comment_type="pulls",
        anchor_comment_id=621,
        covered_comment_ids=[622],
    )
    assert promise is not None

    prepared = oracle.prepare_claims(
        _oracle_owner("recovery"),
        1,
        621,
        [622],
        {},
        {},
    )
    assert prepared is not None
    claims, promises = prepared
    claims, promises = oracle.fail_promise(1, claims, promises)

    store.mark_failed(promise.promise_id)
    _assert_store_matches_oracle(
        store,
        promise,
        oracle_claims=claims,
        oracle_promises=promises,
    )
    assert set(p.promise_id for p in store.recoverable_promises()) == {
        promise.promise_id
    }


def test_reply_promise_marker_round_trips_and_recovers(tmp_path: Path) -> None:
    store = FidoStore(tmp_path)
    promise = store.prepare_reply(
        owner="webhook", comment_type="issues", anchor_comment_id=301
    )
    assert promise is not None
    body = append_reply_promise_marker("done", promise.promise_id)

    assert extract_reply_promise_ids(body) == (promise.promise_id,)
    assert store.recover_from_bodies([body]) == 1
    stored = store.promise(promise.promise_id)
    assert stored is not None
    assert stored.state == "acked"


def test_reply_promise_marker_helpers_ignore_empty_and_duplicate_body(
    tmp_path: Path,
) -> None:
    store = FidoStore(tmp_path)
    promise = store.prepare_reply(
        owner="webhook", comment_type="issues", anchor_comment_id=302
    )
    assert promise is not None
    body = append_reply_promise_marker("done", promise.promise_id)

    assert append_reply_promise_marker(body, promise.promise_id) == body
    assert append_reply_promise_marker("done", None) == "done"
    assert extract_reply_promise_ids(None) == ()


def test_reply_promise_markers_support_many_promises(tmp_path: Path) -> None:
    store = FidoStore(tmp_path)
    first = store.prepare_reply(
        owner="webhook", comment_type="issues", anchor_comment_id=1
    )
    second = store.prepare_reply(
        owner="webhook", comment_type="issues", anchor_comment_id=2
    )
    assert first is not None
    assert second is not None

    body = append_reply_promise_markers("done", [first.promise_id, second.promise_id])

    assert extract_reply_promise_ids(body) == (first.promise_id, second.promise_id)


def test_recover_from_bodies_ignores_unknown_promise(tmp_path: Path) -> None:
    store = FidoStore(tmp_path)

    assert (
        store.recover_from_bodies(
            ["<!-- fido:reply-promise:00000000-0000-0000-0000-000000000000 -->"]
        )
        == 0
    )


def test_schema_includes_durable_store_skeleton(tmp_path: Path) -> None:
    store = FidoStore(tmp_path)
    store.ensure_schema()
    with closing(sqlite3.connect(store.db_path)) as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

    assert version == 6
    assert {
        "comment_claims",
        "reply_promises",
        "reply_promise_comments",
        "reply_artifacts",
        "reply_artifact_promises",
        "reply_outbox_effects",
        "deferred_issue_outbox",
        "command_queue",
        "pr_comment_deliveries",
        "pr_comment_queue",
        "implementation_tasks",
        "fido_state",
        "provider_sessions",
        "check_failures",
        "issue_cache_snapshots",
        "restart_metadata",
        "transition_audit_log",
    } <= tables


def test_record_deferred_issue_is_idempotent(tmp_path: Path) -> None:
    store = FidoStore(tmp_path)

    first = store.record_deferred_issue(
        idempotence_key="deferred-issue:promise-1",
        repo="owner/repo",
        title="later",
        body="body",
        issue_url="https://github.com/owner/repo/issues/1",
    )
    second = store.record_deferred_issue(
        idempotence_key="deferred-issue:promise-1",
        repo="owner/repo",
        title="later",
        body="body",
        issue_url="https://github.com/owner/repo/issues/2",
    )

    assert first.issue_url == "https://github.com/owner/repo/issues/1"
    assert second.issue_url == first.issue_url
    stored = store.deferred_issue("deferred-issue:promise-1")
    assert stored is not None
    assert stored.issue_url == first.issue_url


def test_deferred_issue_returns_none_for_unknown_key(tmp_path: Path) -> None:
    store = FidoStore(tmp_path)
    # Lookup against a key that was never enqueued — exercises the
    # ``row is None → return None`` branch.
    assert store.deferred_issue("deferred-issue:never-created") is None


def test_enqueue_pr_comment_persists_normalized_comment(tmp_path: Path) -> None:
    store = FidoStore(tmp_path)

    record = store.enqueue_pr_comment(
        delivery_id="delivery-1",
        repo="owner/repo",
        pr_number=7,
        comment_type="pulls",
        comment_id=101,
        author="rob",
        is_bot=False,
        body="please adjust this",
        github_created_at="2026-04-30T10:00:00Z",
        payload_json='{"path":"src/fido/store.py"}',
    )

    assert record.delivery_id == "delivery-1"
    assert record.repo == "owner/repo"
    assert record.pr_number == 7
    assert record.comment_type == "pulls"
    assert record.comment_id == 101
    assert record.author == "rob"
    assert record.is_bot is False
    assert record.body == "please adjust this"
    assert record.state == "pending"
    assert record.payload_json == '{"path":"src/fido/store.py"}'
    assert store.pending_pr_comments(repo="owner/repo") == [record]


def test_enqueue_pr_comment_deduplicates_delivery_id(tmp_path: Path) -> None:
    """Same ``delivery_id`` twice (GitHub redelivery) → return the existing
    record unchanged.  No body/comment-id update because GitHub guarantees
    the same delivery_id carries the same payload.
    """
    store = FidoStore(tmp_path)
    first = store.enqueue_pr_comment(
        delivery_id="delivery-1",
        repo="owner/repo",
        pr_number=7,
        comment_type="pulls",
        comment_id=101,
        author="rob",
        is_bot=False,
        body="original",
        github_created_at="2026-04-30T10:00:00Z",
    )

    second = store.enqueue_pr_comment(
        delivery_id="delivery-1",
        repo="owner/repo",
        pr_number=7,
        comment_type="pulls",
        comment_id=102,
        author="rob",
        is_bot=False,
        body="redelivery should not replace",
        github_created_at="2026-04-30T10:01:00Z",
    )

    assert second.queue_id == first.queue_id
    assert second.delivery_id == "delivery-1"
    assert second.body == "original"
    assert second.comment_id == 101
    assert second.github_created_at == "2026-04-30T10:00:00Z"
    assert store.pending_pr_comments(repo="owner/repo") == [first]


def test_enqueue_pr_comment_deduplicates_comment_identity(tmp_path: Path) -> None:
    """Different ``delivery_id`` but same comment identity (a safety-net
    re-enqueue or an edited comment) → keep the original FIFO position
    (queue_id) but refresh the queued body / payload / delivery_id with
    the latest content (per the docstring on ``enqueue_pr_comment``).
    """
    store = FidoStore(tmp_path)
    first = store.enqueue_pr_comment(
        delivery_id="delivery-1",
        repo="owner/repo",
        pr_number=7,
        comment_type="issues",
        comment_id=201,
        author="rob",
        is_bot=False,
        body="original",
        github_created_at="2026-04-30T10:00:00Z",
    )

    second = store.enqueue_pr_comment(
        delivery_id="delivery-2",
        repo="owner/repo",
        pr_number=7,
        comment_type="issues",
        comment_id=201,
        author="rob",
        is_bot=False,
        body="safety-net duplicate",
        github_created_at="2026-04-30T10:01:00Z",
    )

    # FIFO position preserved, but body/delivery_id refreshed.
    assert second.queue_id == first.queue_id
    assert second.delivery_id == "delivery-2"
    assert second.body == "safety-net duplicate"
    # github_created_at stays at the original — first.github_created_at
    # is what determines FIFO order, refreshing it would reorder the queue.
    assert second.github_created_at == first.github_created_at
    pending = store.pending_pr_comments(repo="owner/repo")
    assert len(pending) == 1
    assert pending[0].queue_id == first.queue_id
    assert pending[0].body == "safety-net duplicate"


def test_enqueue_pr_comment_updates_pending_comment_edit(tmp_path: Path) -> None:
    store = FidoStore(tmp_path)
    first = store.enqueue_pr_comment(
        delivery_id="delivery-1",
        repo="owner/repo",
        pr_number=7,
        comment_type="issues",
        comment_id=201,
        author="rob",
        is_bot=False,
        body="original",
        github_created_at="2026-04-30T10:00:00Z",
        payload_json='{"body":"original"}',
    )

    edited = store.enqueue_pr_comment(
        delivery_id="delivery-2",
        repo="owner/repo",
        pr_number=7,
        comment_type="issues",
        comment_id=201,
        author="rob",
        is_bot=False,
        body="edited before worker saw it",
        github_created_at="2026-04-30T10:05:00Z",
        payload_json='{"body":"edited"}',
    )

    assert edited.queue_id == first.queue_id
    assert edited.delivery_id == "delivery-2"
    assert edited.body == "edited before worker saw it"
    assert edited.payload_json == '{"body":"edited"}'
    assert edited.github_created_at == "2026-04-30T10:00:00Z"
    assert store.pending_pr_comments(repo="owner/repo") == [edited]


def test_enqueue_pr_comment_redelivery_does_not_regress_edit(tmp_path: Path) -> None:
    store = FidoStore(tmp_path)
    store.enqueue_pr_comment(
        delivery_id="delivery-original",
        repo="owner/repo",
        pr_number=7,
        comment_type="issues",
        comment_id=201,
        author="rob",
        is_bot=False,
        body="original",
        github_created_at="2026-04-30T10:00:00Z",
    )
    edited = store.enqueue_pr_comment(
        delivery_id="delivery-edit",
        repo="owner/repo",
        pr_number=7,
        comment_type="issues",
        comment_id=201,
        author="rob",
        is_bot=False,
        body="edited",
        github_created_at="2026-04-30T10:05:00Z",
    )

    redelivery = store.enqueue_pr_comment(
        delivery_id="delivery-original",
        repo="owner/repo",
        pr_number=7,
        comment_type="issues",
        comment_id=201,
        author="rob",
        is_bot=False,
        body="stale original redelivery",
        github_created_at="2026-04-30T10:00:00Z",
    )

    assert redelivery == edited
    assert store.pending_pr_comments(repo="owner/repo") == [edited]


def test_enqueue_pr_comment_edit_releases_retry_backoff(tmp_path: Path) -> None:
    store = FidoStore(tmp_path)
    queued = store.enqueue_pr_comment(
        delivery_id="delivery-1",
        repo="owner/repo",
        pr_number=7,
        comment_type="pulls",
        comment_id=202,
        author="rob",
        is_bot=False,
        body="original",
        github_created_at="2026-04-30T10:00:00Z",
    )
    assert store.claim_next_pr_comment(owner="worker", repo="owner/repo") is not None
    retried = store.retry_pr_comment(queued.queue_id, failure_reason="network down")
    assert retried is not None
    assert store.pending_pr_comments(repo="owner/repo") == []

    edited = store.enqueue_pr_comment(
        delivery_id="delivery-2",
        repo="owner/repo",
        pr_number=7,
        comment_type="pulls",
        comment_id=202,
        author="rob",
        is_bot=False,
        body="edited",
        github_created_at="2026-04-30T10:05:00Z",
    )

    assert edited.state == "pending"
    assert edited.next_retry_after is None
    assert edited.body == "edited"
    assert store.pending_pr_comments(repo="owner/repo") == [edited]


def test_pending_pr_comments_are_fifo_by_github_time_then_comment_id(
    tmp_path: Path,
) -> None:
    store = FidoStore(tmp_path)
    later = store.enqueue_pr_comment(
        delivery_id="delivery-later",
        repo="owner/repo",
        pr_number=7,
        comment_type="pulls",
        comment_id=300,
        author="rob",
        is_bot=False,
        body="later",
        github_created_at="2026-04-30T10:02:00Z",
    )
    earliest = store.enqueue_pr_comment(
        delivery_id="delivery-earliest",
        repo="owner/repo",
        pr_number=7,
        comment_type="pulls",
        comment_id=301,
        author="rob",
        is_bot=False,
        body="earliest",
        github_created_at="2026-04-30T10:00:00Z",
    )
    tie_breaker = store.enqueue_pr_comment(
        delivery_id="delivery-tie",
        repo="owner/repo",
        pr_number=7,
        comment_type="pulls",
        comment_id=299,
        author="rob",
        is_bot=False,
        body="same timestamp, lower id",
        github_created_at="2026-04-30T10:02:00Z",
    )

    assert store.pending_pr_comments(repo="owner/repo") == [
        earliest,
        tie_breaker,
        later,
    ]


def test_claim_next_pr_comment_marks_oldest_in_progress(tmp_path: Path) -> None:
    store = FidoStore(tmp_path)
    oldest = store.enqueue_pr_comment(
        delivery_id="delivery-oldest",
        repo="owner/repo",
        pr_number=7,
        comment_type="pulls",
        comment_id=401,
        author="rob",
        is_bot=False,
        body="oldest",
        github_created_at="2026-04-30T10:00:00Z",
    )
    newer = store.enqueue_pr_comment(
        delivery_id="delivery-newer",
        repo="owner/repo",
        pr_number=7,
        comment_type="pulls",
        comment_id=402,
        author="rob",
        is_bot=False,
        body="newer",
        github_created_at="2026-04-30T10:01:00Z",
    )

    claimed = store.claim_next_pr_comment(owner="worker", repo="owner/repo")

    assert claimed is not None
    assert claimed.queue_id == oldest.queue_id
    assert claimed.state == "in_progress"
    assert claimed.claim_owner == "worker"
    assert store.pending_pr_comments(repo="owner/repo") == [newer]


def test_recover_in_progress_pr_comments_requeues_abandoned_claim(
    tmp_path: Path,
) -> None:
    store = FidoStore(tmp_path)
    queued = store.enqueue_pr_comment(
        delivery_id="delivery-1",
        repo="owner/repo",
        pr_number=7,
        comment_type="pulls",
        comment_id=450,
        author="rob",
        is_bot=False,
        body="queued",
        github_created_at="2026-04-30T10:00:00Z",
    )
    assert store.claim_next_pr_comment(owner="worker", repo="owner/repo") is not None
    assert store.pending_pr_comments(repo="owner/repo") == []

    recovered = store.recover_in_progress_pr_comments(repo="owner/repo")

    assert len(recovered) == 1
    assert recovered[0].queue_id == queued.queue_id
    assert recovered[0].state == "pending"
    assert recovered[0].claim_owner is None
    assert recovered[0].next_retry_after is None
    assert store.pending_pr_comments(repo="owner/repo") == recovered


def test_recover_in_progress_pr_comments_is_repo_and_pr_scoped(
    tmp_path: Path,
) -> None:
    store = FidoStore(tmp_path)
    target = store.enqueue_pr_comment(
        delivery_id="delivery-target",
        repo="owner/repo",
        pr_number=7,
        comment_type="pulls",
        comment_id=451,
        author="rob",
        is_bot=False,
        body="target",
        github_created_at="2026-04-30T10:00:00Z",
    )
    other_pr = store.enqueue_pr_comment(
        delivery_id="delivery-other-pr",
        repo="owner/repo",
        pr_number=8,
        comment_type="pulls",
        comment_id=452,
        author="rob",
        is_bot=False,
        body="other pr",
        github_created_at="2026-04-30T10:01:00Z",
    )
    other_repo = store.enqueue_pr_comment(
        delivery_id="delivery-other-repo",
        repo="owner/other",
        pr_number=7,
        comment_type="pulls",
        comment_id=453,
        author="rob",
        is_bot=False,
        body="other repo",
        github_created_at="2026-04-30T10:02:00Z",
    )
    assert (
        store.claim_next_pr_comment(owner="worker", repo="owner/repo", pr_number=7)
        is not None
    )
    assert (
        store.claim_next_pr_comment(owner="worker", repo="owner/repo", pr_number=8)
        is not None
    )
    assert (
        store.claim_next_pr_comment(owner="worker", repo="owner/other", pr_number=7)
        is not None
    )

    recovered = store.recover_in_progress_pr_comments(repo="owner/repo", pr_number=7)

    assert [record.queue_id for record in recovered] == [target.queue_id]
    assert store.pending_pr_comments(repo="owner/repo", pr_number=7) == recovered
    assert store.pending_pr_comments(repo="owner/repo", pr_number=8) == []
    assert store.pending_pr_comments(repo="owner/other", pr_number=7) == []
    assert other_pr.queue_id != target.queue_id
    assert other_repo.queue_id != target.queue_id


def test_complete_pr_comment_removes_it_from_pending_fifo(tmp_path: Path) -> None:
    store = FidoStore(tmp_path)
    queued = store.enqueue_pr_comment(
        delivery_id="delivery-1",
        repo="owner/repo",
        pr_number=7,
        comment_type="issues",
        comment_id=501,
        author="rob",
        is_bot=False,
        body="queued",
        github_created_at="2026-04-30T10:00:00Z",
    )
    claimed = store.claim_next_pr_comment(owner="worker", repo="owner/repo")
    assert claimed is not None

    completed = store.complete_pr_comment(queued.queue_id)

    assert completed is not None
    assert completed.state == "completed"
    assert completed.claim_owner is None
    assert store.pending_pr_comments(repo="owner/repo") == []


def test_retry_pr_comment_sets_backoff_and_becomes_claimable_after_delay(
    tmp_path: Path,
) -> None:
    store = FidoStore(tmp_path)
    queued = store.enqueue_pr_comment(
        delivery_id="delivery-1",
        repo="owner/repo",
        pr_number=7,
        comment_type="pulls",
        comment_id=601,
        author="rob",
        is_bot=False,
        body="queued",
        github_created_at="2026-04-30T10:00:00Z",
    )
    assert store.claim_next_pr_comment(owner="worker", repo="owner/repo") is not None

    retried = store.retry_pr_comment(queued.queue_id, failure_reason="provider paused")

    assert retried is not None
    assert retried.state == "retryable_failed"
    assert retried.claim_owner is None
    assert retried.retry_count == 1
    assert retried.next_retry_after is not None
    assert store.pending_pr_comments(repo="owner/repo") == []

    with closing(sqlite3.connect(store.db_path)) as conn:
        conn.execute(
            """
            UPDATE pr_comment_queue
            SET next_retry_after = NULL
            WHERE queue_id = ?
            """,
            (queued.queue_id,),
        )
        conn.commit()

    claimable = store.pending_pr_comments(repo="owner/repo")
    assert len(claimable) == 1
    assert claimable[0].queue_id == queued.queue_id


def test_clear_pr_comment_queue_is_pr_scoped(tmp_path: Path) -> None:
    store = FidoStore(tmp_path)
    removed = store.enqueue_pr_comment(
        delivery_id="delivery-removed",
        repo="owner/repo",
        pr_number=7,
        comment_type="pulls",
        comment_id=701,
        author="rob",
        is_bot=False,
        body="removed",
        github_created_at="2026-04-30T10:00:00Z",
    )
    kept = store.enqueue_pr_comment(
        delivery_id="delivery-kept",
        repo="owner/repo",
        pr_number=8,
        comment_type="pulls",
        comment_id=702,
        author="rob",
        is_bot=False,
        body="kept",
        github_created_at="2026-04-30T10:00:00Z",
    )

    assert store.clear_pr_comment_queue(repo="owner/repo", pr_number=7) == 1

    assert store.pending_pr_comments(repo="owner/repo") == [kept]
    assert removed not in store.pending_pr_comments(repo="owner/repo")


def test_missing_pr_comment_queue_ids_are_noops(tmp_path: Path) -> None:
    store = FidoStore(tmp_path)

    assert store.claim_next_pr_comment(owner="worker", repo="owner/repo") is None
    assert store.complete_pr_comment("missing") is None
    assert store.retry_pr_comment("missing") is None


def test_record_artifact_tracks_many_promises(tmp_path: Path) -> None:
    store = FidoStore(tmp_path)
    first = store.prepare_reply(
        owner="webhook", comment_type="pulls", anchor_comment_id=501
    )
    second = store.prepare_reply(
        owner="webhook", comment_type="pulls", anchor_comment_id=502
    )
    assert first is not None
    assert second is not None

    store.record_artifact(
        artifact_comment_id=9001,
        comment_type="pulls",
        lane_key="pulls:owner/repo:7:thread:501",
        promise_ids=[first.promise_id, second.promise_id],
    )

    assert store.artifact_for_promise(first.promise_id) is not None
    artifact = store.artifact_for_promise(second.promise_id)
    assert artifact is not None
    assert artifact.artifact_comment_id == 9001
    assert artifact.promise_ids == tuple(sorted((first.promise_id, second.promise_id)))


def test_claim_reply_outbox_effect_is_durable_and_idempotent(tmp_path: Path) -> None:
    store = FidoStore(tmp_path)
    promise = store.prepare_reply(
        owner="webhook", comment_type="pulls", anchor_comment_id=501
    )
    assert promise is not None

    first = store.claim_reply_outbox_effect(
        promise_id=promise.promise_id,
        delivery_id="delivery-501",
        origin_id=501,
    )
    second = store.claim_reply_outbox_effect(
        promise_id=promise.promise_id,
        delivery_id="delivery-other",
        origin_id=999,
    )

    assert isinstance(first, ReplyOutboxEffectRecord)
    assert second == first
    assert first.delivery_id == "delivery-501"
    assert first.origin_id == 501
    assert first.state == "claimed"
    assert first.external_id is None


def test_record_reply_delivery_delivers_outbox_effect(tmp_path: Path) -> None:
    store = FidoStore(tmp_path)
    promise = store.prepare_reply(
        owner="webhook", comment_type="pulls", anchor_comment_id=502
    )
    assert promise is not None
    store.claim_reply_outbox_effect(
        promise_id=promise.promise_id,
        delivery_id="delivery-502",
        origin_id=502,
    )

    store.record_reply_delivery(
        artifact_comment_id=9003,
        comment_type="pulls",
        lane_key="pulls:owner/repo:7:thread:502",
        promise_ids=[promise.promise_id],
    )

    effect = store.reply_outbox_effect(promise.promise_id)
    assert effect is not None
    assert effect.state == "delivered"
    assert effect.external_id == 9003
    persisted = store.promise(promise.promise_id)
    assert persisted is not None
    assert persisted.state == "posted"


def test_record_artifact_ignores_empty_promise_ids(tmp_path: Path) -> None:
    store = FidoStore(tmp_path)

    store.record_artifact(
        artifact_comment_id=9002,
        comment_type="pulls",
        lane_key="pulls:owner/repo:7:thread:501",
        promise_ids=[],
    )

    assert store.artifact_for_promise("00000000-0000-0000-0000-000000000000") is None


def test_artifact_for_promise_returns_none_when_missing(tmp_path: Path) -> None:
    assert (
        FidoStore(tmp_path).artifact_for_promise("00000000-0000-0000-0000-000000000000")
        is None
    )


def test_schema_rejects_newer_user_version(tmp_path: Path) -> None:
    store = FidoStore(tmp_path)
    store.db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(store.db_path)) as conn:
        conn.execute("PRAGMA user_version = 999")
        conn.commit()

    with pytest.raises(RuntimeError, match="unsupported fido.db schema version 999"):
        store.ensure_schema()


def test_schema_upgrades_v2_database_to_current_store(tmp_path: Path) -> None:
    store = FidoStore(tmp_path)
    store.db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(store.db_path)) as conn:
        conn.execute("PRAGMA user_version = 2")
        conn.commit()

    store.ensure_schema()

    with closing(sqlite3.connect(store.db_path)) as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        pr_comment_queue = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = 'pr_comment_queue'
            """
        ).fetchone()
        deferred_issue_outbox = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = 'deferred_issue_outbox'
            """
        ).fetchone()
        delivery_table = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = 'pr_comment_deliveries'
            """
        ).fetchone()
    # Schema is bumped on every additive migration; the test is anchored
    # on the production constant so a future migration doesn't require
    # touching it.
    from fido.store import _SCHEMA_VERSION  # pyright: ignore[reportPrivateUsage]

    assert version == _SCHEMA_VERSION
    assert pr_comment_queue is not None
    assert delivery_table is not None
    assert deferred_issue_outbox is not None


def test_concurrent_prepare_has_one_winner(tmp_path: Path) -> None:
    store = FidoStore(tmp_path)
    barrier = threading.Barrier(2)
    results = []

    def claim(owner: ReplyOwner) -> None:
        barrier.wait()
        results.append(
            store.prepare_reply(
                owner=owner, comment_type="pulls", anchor_comment_id=401
            )
        )

    threads = [
        threading.Thread(target=claim, args=("webhook",)),
        threading.Thread(target=claim, args=("worker",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sum(result is not None for result in results) == 1


def test_transaction_rolls_back_on_exception(tmp_path: Path) -> None:
    store = FidoStore(tmp_path)

    try:
        with store._transaction() as conn:  # pyright: ignore[reportPrivateUsage]
            conn.execute(
                """
                INSERT INTO fido_state (key, value_json, updated_at)
                VALUES (?, ?, ?)
                """,
                (
                    "rollback-test",
                    "{}",
                    "2026-04-23T00:00:00+00:00",
                ),
            )
            raise RuntimeError("force rollback")
    except RuntimeError:
        pass

    with closing(sqlite3.connect(store.db_path)) as conn:
        row = conn.execute(
            "SELECT key FROM fido_state WHERE key = ?", ("rollback-test",)
        ).fetchone()
    assert row is None


def test_extracted_oracle_matches_claim_lifecycle() -> None:
    prepared = oracle.prepare_claims(oracle.OwnerWebhook(), 1, 10, [11], {}, {})
    assert prepared is not None
    claims, promises = prepared

    assert not oracle.comment_claimable(claims, 10)
    assert (
        oracle.prepare_claims(oracle.OwnerWorker(), 2, 11, [], claims, promises) is None
    )

    failed_claims, failed_promises = oracle.fail_promise(1, claims, promises)
    assert oracle.comment_claimable(failed_claims, 10)

    retried = oracle.prepare_claims(
        oracle.OwnerRecovery(), 3, 10, [11], failed_claims, failed_promises
    )
    assert retried is not None
    retry_claims, retry_promises = retried
    completed_claims, _ = oracle.ack_promise(3, retry_claims, retry_promises)
    assert oracle.claim_completed(completed_claims, 10)
    assert oracle.claim_completed(completed_claims, 11)


def test_extracted_oracle_matches_recovery_lifecycle() -> None:
    prepared = oracle.prepare_claims(oracle.OwnerWebhook(), 1, 10, [11], {}, {})
    assert prepared is not None
    claims, promises = prepared

    assert oracle.promise_recoverable(oracle.PromisePrepared())
    marker_claims, marker_promises = oracle.recover_promise(
        1, oracle.SeenPromiseMarker(), claims, promises
    )
    assert oracle.claim_completed(marker_claims, 10)
    assert oracle.claim_completed(marker_claims, 11)
    assert not oracle.promise_recoverable(_oracle_promise_state(marker_promises, 1))

    posted_claims, posted_promises = oracle.recover_promise(
        1, oracle.ReplayPosted(), claims, promises
    )
    assert oracle.claim_completed(posted_claims, 10)
    assert oracle.claim_completed(posted_claims, 11)
    assert not oracle.promise_recoverable(_oracle_promise_state(posted_promises, 1))

    failed_claims, failed_promises = oracle.recover_promise(
        1, oracle.AnchorDeleted(), claims, promises
    )
    assert oracle.comment_claimable(failed_claims, 10)
    assert oracle.promise_recoverable(_oracle_promise_state(failed_promises, 1))

    failed_again_claims, failed_again_promises = oracle.recover_promise(
        1, oracle.ReplayFailed(), claims, promises
    )
    assert oracle.comment_claimable(failed_again_claims, 10)
    assert oracle.promise_recoverable(_oracle_promise_state(failed_again_promises, 1))

    same_claims, same_promises = oracle.recover_promise(
        1, oracle.WrongPullRequest(), claims, promises
    )
    assert same_claims == claims
    assert same_promises == promises
