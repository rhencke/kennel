import sqlite3
import threading
from contextlib import closing
from pathlib import Path

from fido.reply_store import (
    ReplyOwner,
    ReplyStore,
    append_reply_promise_marker,
    extract_reply_promise_ids,
)
from fido.rocq import replied_comment_claims as oracle


def test_prepare_claim_is_atomic_and_blocks_duplicate_owner(tmp_path: Path) -> None:
    store = ReplyStore(tmp_path)

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
    store = ReplyStore(tmp_path)

    promise = store.prepare_reply(
        owner="webhook",
        comment_type="pulls",
        anchor_comment_id=101,
        covered_comment_ids=[102, 103],
    )

    assert promise is not None
    assert promise.covered_comment_ids == (101, 102, 103)


def test_missing_claim_and_missing_promise_are_noops(tmp_path: Path) -> None:
    store = ReplyStore(tmp_path)

    assert not store.is_claimed_or_completed(999)
    assert store.claim_state(999) is None
    assert store.promise("00000000-0000-0000-0000-000000000000") is None
    store.ack_promise("00000000-0000-0000-0000-000000000000")
    store.mark_failed("00000000-0000-0000-0000-000000000000")


def test_failed_claim_becomes_retryable(tmp_path: Path) -> None:
    store = ReplyStore(tmp_path)
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
    store = ReplyStore(tmp_path)
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
    store = ReplyStore(tmp_path)
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


def test_reply_promise_marker_round_trips_and_recovers(tmp_path: Path) -> None:
    store = ReplyStore(tmp_path)
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
    store = ReplyStore(tmp_path)
    promise = store.prepare_reply(
        owner="webhook", comment_type="issues", anchor_comment_id=302
    )
    assert promise is not None
    body = append_reply_promise_marker("done", promise.promise_id)

    assert append_reply_promise_marker(body, promise.promise_id) == body
    assert append_reply_promise_marker("done", None) == "done"
    assert extract_reply_promise_ids(None) == ()


def test_recover_from_bodies_ignores_unknown_promise(tmp_path: Path) -> None:
    store = ReplyStore(tmp_path)

    assert (
        store.recover_from_bodies(
            ["<!-- fido:reply-promise:00000000-0000-0000-0000-000000000000 -->"]
        )
        == 0
    )


def test_schema_includes_durable_store_skeleton(tmp_path: Path) -> None:
    store = ReplyStore(tmp_path)
    store.ensure_schema()
    with closing(sqlite3.connect(store.db_path)) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

    assert {
        "comment_claims",
        "reply_promises",
        "command_queue",
        "implementation_tasks",
        "fido_state",
        "provider_sessions",
        "check_failures",
        "issue_cache_snapshots",
        "restart_metadata",
        "transition_audit_log",
    } <= tables


def test_concurrent_prepare_has_one_winner(tmp_path: Path) -> None:
    store = ReplyStore(tmp_path)
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
    store = ReplyStore(tmp_path)

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
