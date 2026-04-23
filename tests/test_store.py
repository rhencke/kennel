import sqlite3
import threading
from contextlib import closing
from pathlib import Path

import pytest

from fido.rocq import replied_comment_claims as oracle
from fido.store import (
    FidoStore,
    ReplyOwner,
    ReplyPromiseRecord,
    append_reply_promise_marker,
    append_reply_promise_markers,
    extract_reply_promise_ids,
)


def _oracle_promise_state(promises: object, promise_id: int) -> object:
    return promises[promise_id].promise_state  # type: ignore[index]


def _oracle_claim_state(claims: object, comment_id: int) -> object:
    return claims[comment_id].claim_state  # type: ignore[index]


def _oracle_owner(owner: ReplyOwner) -> object:
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
    oracle_claims: dict[int, object],
    oracle_promises: dict[int, object],
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

    assert version == 2
    assert {
        "comment_claims",
        "reply_promises",
        "reply_promise_comments",
        "reply_artifacts",
        "reply_artifact_promises",
        "command_queue",
        "implementation_tasks",
        "fido_state",
        "provider_sessions",
        "check_failures",
        "issue_cache_snapshots",
        "restart_metadata",
        "transition_audit_log",
    } <= tables


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
