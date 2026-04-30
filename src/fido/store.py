"""Repo-local SQLite state store."""

import re
import sqlite3
import uuid
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

ReplyOwner = Literal["webhook", "worker", "recovery"]
ClaimState = Literal["in_progress", "completed", "retryable_failed"]
PromiseState = Literal["prepared", "posted", "acked", "failed"]
PRCommentQueueState = Literal["pending", "in_progress", "completed", "retryable_failed"]

REPLY_PROMISE_MARKER_PREFIX = "fido:reply-promise:"
_PROMISE_MARKER_RE = re.compile(r"<!--\s*fido:reply-promise:([0-9a-fA-F-]{36})\s*-->")
_SCHEMA_VERSION = 5


@dataclass(frozen=True)
class ReplyPromiseRecord:
    """One durable outbound reply attempt."""

    promise_id: str
    owner: ReplyOwner
    comment_type: str
    anchor_comment_id: int
    covered_comment_ids: tuple[int, ...]
    state: PromiseState
    retry_count: int
    next_retry_after: str | None


@dataclass(frozen=True)
class ReplyArtifactRecord:
    """One visible GitHub reply artifact that may cover many promises."""

    artifact_comment_id: int
    comment_type: str
    lane_key: str
    promise_ids: tuple[str, ...]


@dataclass(frozen=True)
class DeferredIssueRecord:
    """One deferred tracking issue opened for an outbound reply promise."""

    idempotence_key: str
    repo: str
    title: str
    body: str
    issue_url: str


@dataclass(frozen=True)
class PRCommentQueueRecord:
    """One durable queued PR comment waiting for triage."""

    queue_id: str
    delivery_id: str
    repo: str
    pr_number: int
    comment_type: str
    comment_id: int
    author: str
    is_bot: bool
    body: str
    github_created_at: str
    state: PRCommentQueueState
    claim_owner: str | None
    retry_count: int
    next_retry_after: str | None
    payload_json: str


def append_reply_promise_marker(body: str, promise_id: str | None) -> str:
    """Append one hidden recovery marker to a GitHub reply body."""
    return append_reply_promise_markers(
        body, () if promise_id is None else (promise_id,)
    )


def append_reply_promise_markers(body: str, promise_ids: Iterable[str]) -> str:
    """Append hidden recovery markers for every covered promise id."""
    normalized = tuple(
        dict.fromkeys(promise_id for promise_id in promise_ids if promise_id)
    )
    if not normalized:
        return body
    existing = set(extract_reply_promise_ids(body))
    missing = [
        promise_id for promise_id in normalized if promise_id.lower() not in existing
    ]
    if not missing:
        return body
    markers = "\n".join(
        f"<!-- {REPLY_PROMISE_MARKER_PREFIX}{promise_id} -->" for promise_id in missing
    )
    return f"{body.rstrip()}\n\n{markers}"


def extract_reply_promise_ids(body: str | None) -> tuple[str, ...]:
    """Return hidden Fido reply-promise ids embedded in *body*."""
    if not body:
        return ()
    return tuple(match.group(1).lower() for match in _PROMISE_MARKER_RE.finditer(body))


class FidoStore:
    """Repo-local SQLite state and coordination store.

    Reply claim/promise tables are the first live runtime users.  Generic state
    rows share the same database so future durable state has one home.
    """

    def __init__(self, work_dir: Path) -> None:
        self._work_dir = work_dir
        self._db_path = work_dir / ".git" / "fido.db"

    @property
    def db_path(self) -> Path:
        """Path to the repo-local SQLite database."""
        return self._db_path

    def prepare_reply(
        self,
        *,
        owner: ReplyOwner,
        comment_type: str,
        anchor_comment_id: int,
        covered_comment_ids: Iterable[int] | None = None,
    ) -> ReplyPromiseRecord | None:
        """Atomically claim covered comment ids and create a prepared promise.

        Returns ``None`` when any covered id is already completed or currently
        owned by another in-progress promise.
        """
        covered = _normalize_comment_ids(covered_comment_ids or (anchor_comment_id,))
        if anchor_comment_id not in covered:
            covered = (anchor_comment_id, *covered)
        now = _utcnow()
        promise_id = str(uuid.uuid4())
        with self._transaction() as conn:
            existing = conn.execute(
                """
                SELECT comment_id, state, next_retry_after
                FROM comment_claims
                WHERE comment_id IN ({})
                """.format(",".join("?" for _ in covered)),
                covered,
            ).fetchall()
            for row in existing:
                if row["state"] in {"completed", "in_progress"}:
                    return None
                next_retry_after = row["next_retry_after"]
                if next_retry_after and next_retry_after > now:
                    return None

            conn.execute(
                """
                INSERT INTO reply_promises (
                    promise_id, owner, comment_type, anchor_comment_id,
                    state, retry_count, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, 'prepared', 0, ?, ?)
                """,
                (
                    promise_id,
                    owner,
                    comment_type,
                    anchor_comment_id,
                    now,
                    now,
                ),
            )
            conn.executemany(
                """
                INSERT INTO reply_promise_comments (
                    promise_id, position, comment_id
                )
                VALUES (?, ?, ?)
                """,
                (
                    (promise_id, position, comment_id)
                    for position, comment_id in enumerate(covered)
                ),
            )
            conn.executemany(
                """
                INSERT INTO comment_claims (
                    comment_id, owner, state, promise_id, retry_count,
                    created_at, updated_at
                )
                VALUES (?, ?, 'in_progress', ?, 0, ?, ?)
                ON CONFLICT(comment_id) DO UPDATE SET
                    owner = excluded.owner,
                    state = 'in_progress',
                    promise_id = excluded.promise_id,
                    updated_at = excluded.updated_at
                """,
                ((comment_id, owner, promise_id, now, now) for comment_id in covered),
            )
        return ReplyPromiseRecord(
            promise_id=promise_id,
            owner=owner,
            comment_type=comment_type,
            anchor_comment_id=anchor_comment_id,
            covered_comment_ids=covered,
            state="prepared",
            retry_count=0,
            next_retry_after=None,
        )

    def mark_posted(self, promise_id: str) -> None:
        """Mark a prepared promise as having reached GitHub."""
        self._set_promise_state(promise_id, "posted")

    def record_artifact(
        self,
        *,
        artifact_comment_id: int,
        comment_type: str,
        lane_key: str,
        promise_ids: Iterable[str],
    ) -> None:
        """Record one visible reply artifact and the promises it covers."""
        covered = tuple(
            dict.fromkeys(promise_id for promise_id in promise_ids if promise_id)
        )
        if not covered:
            return
        now = _utcnow()
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO reply_artifacts (
                    artifact_comment_id, comment_type, lane_key, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(artifact_comment_id) DO UPDATE SET
                    comment_type = excluded.comment_type,
                    lane_key = excluded.lane_key,
                    updated_at = excluded.updated_at
                """,
                (artifact_comment_id, comment_type, lane_key, now, now),
            )
            conn.executemany(
                """
                INSERT OR REPLACE INTO reply_artifact_promises (
                    artifact_comment_id, promise_id
                )
                VALUES (?, ?)
                """,
                ((artifact_comment_id, promise_id) for promise_id in covered),
            )

    def artifact_for_promise(self, promise_id: str) -> ReplyArtifactRecord | None:
        """Return the visible reply artifact covering *promise_id*, if any."""
        self.ensure_schema()
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT a.artifact_comment_id, a.comment_type, a.lane_key
                FROM reply_artifacts AS a
                JOIN reply_artifact_promises AS ap
                  ON ap.artifact_comment_id = a.artifact_comment_id
                WHERE ap.promise_id = ?
                """,
                (promise_id,),
            ).fetchone()
            if row is None:
                return None
            promise_rows = conn.execute(
                """
                SELECT promise_id
                FROM reply_artifact_promises
                WHERE artifact_comment_id = ?
                ORDER BY promise_id
                """,
                (int(row["artifact_comment_id"]),),
            ).fetchall()
        return ReplyArtifactRecord(
            artifact_comment_id=int(row["artifact_comment_id"]),
            comment_type=row["comment_type"],
            lane_key=row["lane_key"],
            promise_ids=tuple(
                str(promise_row["promise_id"]) for promise_row in promise_rows
            ),
        )

    def deferred_issue(self, idempotence_key: str) -> DeferredIssueRecord | None:
        """Return a previously opened deferred issue for *idempotence_key*."""
        self.ensure_schema()
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT *
                FROM deferred_issue_outbox
                WHERE idempotence_key = ?
                """,
                (idempotence_key,),
            ).fetchone()
        if row is None:
            return None
        return DeferredIssueRecord(
            idempotence_key=row["idempotence_key"],
            repo=row["repo"],
            title=row["title"],
            body=row["body"],
            issue_url=row["issue_url"],
        )

    def record_deferred_issue(
        self,
        *,
        idempotence_key: str,
        repo: str,
        title: str,
        body: str,
        issue_url: str,
    ) -> DeferredIssueRecord:
        """Persist a successfully opened deferred issue idempotently."""
        now = _utcnow()
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO deferred_issue_outbox (
                    idempotence_key, repo, title, body, issue_url, created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(idempotence_key) DO UPDATE SET
                    issue_url = deferred_issue_outbox.issue_url,
                    updated_at = excluded.updated_at
                """,
                (idempotence_key, repo, title, body, issue_url, now, now),
            )
            row = conn.execute(
                """
                SELECT *
                FROM deferred_issue_outbox
                WHERE idempotence_key = ?
                """,
                (idempotence_key,),
            ).fetchone()
            assert row is not None
        return DeferredIssueRecord(
            idempotence_key=row["idempotence_key"],
            repo=row["repo"],
            title=row["title"],
            body=row["body"],
            issue_url=row["issue_url"],
        )

    def enqueue_pr_comment(
        self,
        *,
        delivery_id: str,
        repo: str,
        pr_number: int,
        comment_type: str,
        comment_id: int,
        author: str,
        is_bot: bool,
        body: str,
        github_created_at: str,
        payload_json: str = "{}",
    ) -> PRCommentQueueRecord:
        """Durably enqueue one normalized PR comment, idempotently.

        ``delivery_id`` deduplicates GitHub redelivery.  The comment identity
        also stays unique so a safety-net enqueue with a distinct delivery id
        cannot create duplicate triage work for the same GitHub comment.  When
        a new delivery for an already-pending comment arrives, keep the
        original FIFO position but replace the queued body/payload with the
        latest GitHub content.
        """
        now = _utcnow()
        queue_id = str(uuid.uuid4())
        with self._transaction() as conn:
            existing = self._pr_comment_by_delivery(conn, delivery_id)
            if existing is not None:
                return self._pr_comment_record_from_row(existing)
            existing = self._pr_comment_by_comment(
                conn, repo, pr_number, comment_type, comment_id
            )
            if existing is not None:
                return self._refresh_pr_comment_record(
                    conn,
                    existing,
                    delivery_id=delivery_id,
                    author=author,
                    is_bot=is_bot,
                    body=body,
                    payload_json=payload_json,
                )
            conn.execute(
                """
                INSERT INTO pr_comment_queue (
                    queue_id, delivery_id, repo, pr_number, comment_type,
                    comment_id, author, is_bot, body, github_created_at,
                    state, retry_count, payload_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?)
                """,
                (
                    queue_id,
                    delivery_id,
                    repo,
                    int(pr_number),
                    comment_type,
                    int(comment_id),
                    author,
                    1 if is_bot else 0,
                    body,
                    github_created_at,
                    payload_json,
                    now,
                    now,
                ),
            )
            row = self._pr_comment_by_queue_id(conn, queue_id)
            assert row is not None
            self._record_pr_comment_delivery(conn, delivery_id, queue_id)
            return self._pr_comment_record_from_row(row)

    def _refresh_pr_comment_record(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        delivery_id: str,
        author: str,
        is_bot: bool,
        body: str,
        payload_json: str,
    ) -> PRCommentQueueRecord:
        """Update a duplicate queued comment with the latest editable fields."""
        state = str(row["state"])
        if state == "completed":
            return self._pr_comment_record_from_row(row)
        now = _utcnow()
        if state == "in_progress":
            conn.execute(
                """
                UPDATE pr_comment_queue
                SET delivery_id = ?, author = ?, is_bot = ?, body = ?,
                    payload_json = ?, updated_at = ?
                WHERE queue_id = ?
                """,
                (
                    delivery_id,
                    author,
                    1 if is_bot else 0,
                    body,
                    payload_json,
                    now,
                    row["queue_id"],
                ),
            )
        else:
            conn.execute(
                """
                UPDATE pr_comment_queue
                SET delivery_id = ?, author = ?, is_bot = ?, body = ?,
                    state = 'pending', claim_owner = NULL, next_retry_after = NULL,
                    failure_reason = NULL, payload_json = ?, updated_at = ?
                WHERE queue_id = ?
                """,
                (
                    delivery_id,
                    author,
                    1 if is_bot else 0,
                    body,
                    payload_json,
                    now,
                    row["queue_id"],
                ),
            )
        updated = self._pr_comment_by_queue_id(conn, row["queue_id"])
        assert updated is not None
        self._record_pr_comment_delivery(conn, delivery_id, row["queue_id"])
        return self._pr_comment_record_from_row(updated)

    def pending_pr_comments(
        self, *, repo: str | None = None, pr_number: int | None = None
    ) -> list[PRCommentQueueRecord]:
        """Return retryable pending comments in FIFO order."""
        self.ensure_schema()
        where = ["state IN ('pending', 'retryable_failed')"]
        params: list[object] = []
        if repo is not None:
            where.append("repo = ?")
            params.append(repo)
        if pr_number is not None:
            where.append("pr_number = ?")
            params.append(int(pr_number))
        where.append("(next_retry_after IS NULL OR next_retry_after <= ?)")
        params.append(_utcnow())
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM pr_comment_queue
                WHERE {" AND ".join(where)}
                ORDER BY github_created_at, comment_id, queue_id
                """,
                tuple(params),
            ).fetchall()
        return [self._pr_comment_record_from_row(row) for row in rows]

    def has_pending_pr_comments(self, repo: str) -> bool:
        """Return whether *repo* has any currently retryable queued comments."""
        return bool(self.pending_pr_comments(repo=repo))

    def claim_next_pr_comment(
        self, *, owner: str, repo: str | None = None, pr_number: int | None = None
    ) -> PRCommentQueueRecord | None:
        """Claim the oldest currently retryable PR comment for processing."""
        with self._transaction() as conn:
            where = ["state IN ('pending', 'retryable_failed')"]
            params: list[object] = []
            if repo is not None:
                where.append("repo = ?")
                params.append(repo)
            if pr_number is not None:
                where.append("pr_number = ?")
                params.append(int(pr_number))
            where.append("(next_retry_after IS NULL OR next_retry_after <= ?)")
            params.append(_utcnow())
            row = conn.execute(
                f"""
                SELECT *
                FROM pr_comment_queue
                WHERE {" AND ".join(where)}
                ORDER BY github_created_at, comment_id, queue_id
                LIMIT 1
                """,
                tuple(params),
            ).fetchone()
            if row is None:
                return None
            now = _utcnow()
            conn.execute(
                """
                UPDATE pr_comment_queue
                SET state = 'in_progress', claim_owner = ?, updated_at = ?
                WHERE queue_id = ?
                """,
                (owner, now, row["queue_id"]),
            )
            claimed = self._pr_comment_by_queue_id(conn, row["queue_id"])
            assert claimed is not None
            return self._pr_comment_record_from_row(claimed)

    def recover_in_progress_pr_comments(
        self, *, repo: str | None = None, pr_number: int | None = None
    ) -> list[PRCommentQueueRecord]:
        """Requeue comments abandoned by a worker crash or restart."""
        with self._transaction() as conn:
            where = ["state = 'in_progress'"]
            params: list[object] = []
            if repo is not None:
                where.append("repo = ?")
                params.append(repo)
            if pr_number is not None:
                where.append("pr_number = ?")
                params.append(int(pr_number))
            rows = conn.execute(
                f"""
                SELECT *
                FROM pr_comment_queue
                WHERE {" AND ".join(where)}
                ORDER BY github_created_at, comment_id, queue_id
                """,
                tuple(params),
            ).fetchall()
            if not rows:
                return []
            now = _utcnow()
            recovered: list[PRCommentQueueRecord] = []
            for row in rows:
                conn.execute(
                    """
                    UPDATE pr_comment_queue
                    SET state = 'pending', claim_owner = NULL,
                        next_retry_after = NULL, failure_reason = NULL,
                        updated_at = ?
                    WHERE queue_id = ?
                    """,
                    (now, row["queue_id"]),
                )
                updated = self._pr_comment_by_queue_id(conn, row["queue_id"])
                assert updated is not None
                recovered.append(self._pr_comment_record_from_row(updated))
            return recovered

    def complete_pr_comment(self, queue_id: str) -> PRCommentQueueRecord | None:
        """Mark a queued PR comment completed."""
        return self._set_pr_comment_state(
            queue_id,
            state="completed",
            claim_owner=None,
            retry_count=None,
            next_retry_after=None,
        )

    def retry_pr_comment(
        self, queue_id: str, *, failure_reason: str | None = None
    ) -> PRCommentQueueRecord | None:
        """Mark a queued PR comment retryable with exponential backoff."""
        with self._transaction() as conn:
            row = self._pr_comment_by_queue_id(conn, queue_id)
            if row is None:
                return None
            retry_count = int(row["retry_count"]) + 1
            next_retry_after = _retry_after(retry_count)
            now = _utcnow()
            conn.execute(
                """
                UPDATE pr_comment_queue
                SET state = 'retryable_failed', claim_owner = NULL,
                    retry_count = ?, next_retry_after = ?, failure_reason = ?,
                    updated_at = ?
                WHERE queue_id = ?
                """,
                (retry_count, next_retry_after, failure_reason, now, queue_id),
            )
            retried = self._pr_comment_by_queue_id(conn, queue_id)
            assert retried is not None
            return self._pr_comment_record_from_row(retried)

    def clear_pr_comment_queue(self, *, repo: str, pr_number: int) -> int:
        """Delete queued comment state for a PR after merge/close cleanup."""
        with self._transaction() as conn:
            cursor = conn.execute(
                """
                DELETE FROM pr_comment_queue
                WHERE repo = ? AND pr_number = ?
                """,
                (repo, int(pr_number)),
            )
            return cursor.rowcount

    def ack_promise(self, promise_id: str) -> None:
        """Complete a promise and every covered raw comment id."""
        now = _utcnow()
        with self._transaction() as conn:
            row = self._promise_row(conn, promise_id)
            if row is None:
                return
            covered = self._covered_comments(conn, promise_id)
            conn.execute(
                """
                UPDATE reply_promises
                SET state = 'acked', updated_at = ?
                WHERE promise_id = ?
                """,
                (now, promise_id),
            )
            conn.executemany(
                """
                UPDATE comment_claims
                SET state = 'completed', updated_at = ?, next_retry_after = NULL
                WHERE comment_id = ?
                """,
                ((now, int(comment_id)) for comment_id in covered),
            )

    def mark_failed(self, promise_id: str) -> None:
        """Mark a promise and its claims retryable with exponential backoff."""
        now_dt = datetime.now(tz=UTC)
        now = now_dt.isoformat()
        with self._transaction() as conn:
            row = self._promise_row(conn, promise_id)
            if row is None:
                return
            retry_count = int(row["retry_count"]) + 1
            next_retry_after = _retry_after(retry_count)
            covered = self._covered_comments(conn, promise_id)
            conn.execute(
                """
                UPDATE reply_promises
                SET state = 'failed', retry_count = ?, next_retry_after = ?,
                    updated_at = ?
                WHERE promise_id = ?
                """,
                (retry_count, next_retry_after, now, promise_id),
            )
            conn.executemany(
                """
                UPDATE comment_claims
                SET state = 'retryable_failed', retry_count = ?,
                    next_retry_after = ?, updated_at = ?
                WHERE comment_id = ?
                """,
                (
                    (retry_count, next_retry_after, now, int(comment_id))
                    for comment_id in covered
                ),
            )

    def is_claimed_or_completed(self, comment_id: int) -> bool:
        """Return true when *comment_id* should not be handled again now."""
        self.ensure_schema()
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT state, next_retry_after
                FROM comment_claims
                WHERE comment_id = ?
                """,
                (comment_id,),
            ).fetchone()
        if row is None:
            return False
        if row["state"] in {"completed", "in_progress"}:
            return True
        next_retry_after = row["next_retry_after"]
        return bool(next_retry_after and next_retry_after > _utcnow())

    def claim_state(self, comment_id: int) -> ClaimState | None:
        """Return the durable claim state for tests and diagnostics."""
        self.ensure_schema()
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT state FROM comment_claims WHERE comment_id = ?",
                (comment_id,),
            ).fetchone()
        return row["state"] if row is not None else None

    def recover_from_bodies(self, bodies: Iterable[str | None]) -> int:
        """Ack stale promises whose hidden marker appears in GitHub bodies."""
        recovered = 0
        for body in bodies:
            for promise_id in extract_reply_promise_ids(body):
                before = self.promise(promise_id)
                self.ack_promise(promise_id)
                after = self.promise(promise_id)
                if before is not None and after is not None and before.state != "acked":
                    recovered += 1
        return recovered

    def recoverable_promises(self) -> list[ReplyPromiseRecord]:
        """Return non-acked promises that recovery should reconcile."""
        self.ensure_schema()
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM reply_promises
                WHERE state IN ('prepared', 'posted', 'failed')
                ORDER BY created_at, promise_id
                """
            ).fetchall()
            return [self._record_from_row(conn, row) for row in rows]

    def promise(self, promise_id: str) -> ReplyPromiseRecord | None:
        """Return one promise record."""
        self.ensure_schema()
        with closing(self._connect()) as conn:
            row = self._promise_row(conn, promise_id)
            return self._record_from_row(conn, row) if row is not None else None

    def ensure_schema(self) -> None:
        """Create the D-lane durable store schema if needed."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                version = int(conn.execute("PRAGMA user_version").fetchone()[0])
                if version > _SCHEMA_VERSION:
                    raise RuntimeError(
                        f"unsupported fido.db schema version {version}; "
                        f"expected <= {_SCHEMA_VERSION}"
                    )
                if version < _SCHEMA_VERSION:
                    for statement in _schema_statements():
                        conn.execute(statement)
                    conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            except Exception:
                conn.rollback()
                raise
            conn.commit()

    def _set_promise_state(self, promise_id: str, state: PromiseState) -> None:
        now = _utcnow()
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE reply_promises
                SET state = ?, updated_at = ?
                WHERE promise_id = ?
                """,
                (state, now, promise_id),
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _transaction(self) -> Any:
        return _FidoTransaction(self)

    def _promise_row(
        self, conn: sqlite3.Connection, promise_id: str
    ) -> sqlite3.Row | None:
        return conn.execute(
            "SELECT * FROM reply_promises WHERE promise_id = ?", (promise_id,)
        ).fetchone()

    def _covered_comments(
        self, conn: sqlite3.Connection, promise_id: str
    ) -> tuple[int, ...]:
        rows = conn.execute(
            """
            SELECT comment_id
            FROM reply_promise_comments
            WHERE promise_id = ?
            ORDER BY position
            """,
            (promise_id,),
        ).fetchall()
        return tuple(int(row["comment_id"]) for row in rows)

    def _record_from_row(
        self, conn: sqlite3.Connection, row: sqlite3.Row
    ) -> ReplyPromiseRecord:
        return ReplyPromiseRecord(
            promise_id=row["promise_id"],
            owner=cast(ReplyOwner, row["owner"]),
            comment_type=row["comment_type"],
            anchor_comment_id=int(row["anchor_comment_id"]),
            covered_comment_ids=self._covered_comments(conn, row["promise_id"]),
            state=cast(PromiseState, row["state"]),
            retry_count=int(row["retry_count"]),
            next_retry_after=row["next_retry_after"],
        )

    def _set_pr_comment_state(
        self,
        queue_id: str,
        *,
        state: PRCommentQueueState,
        claim_owner: str | None,
        retry_count: int | None,
        next_retry_after: str | None,
    ) -> PRCommentQueueRecord | None:
        now = _utcnow()
        with self._transaction() as conn:
            row = self._pr_comment_by_queue_id(conn, queue_id)
            if row is None:
                return None
            conn.execute(
                """
                UPDATE pr_comment_queue
                SET state = ?, claim_owner = ?, retry_count = ?,
                    next_retry_after = ?, updated_at = ?
                WHERE queue_id = ?
                """,
                (
                    state,
                    claim_owner,
                    int(row["retry_count"]) if retry_count is None else retry_count,
                    next_retry_after,
                    now,
                    queue_id,
                ),
            )
            updated = self._pr_comment_by_queue_id(conn, queue_id)
            assert updated is not None
            return self._pr_comment_record_from_row(updated)

    def _pr_comment_by_delivery(
        self, conn: sqlite3.Connection, delivery_id: str
    ) -> sqlite3.Row | None:
        row = conn.execute(
            "SELECT * FROM pr_comment_queue WHERE delivery_id = ?",
            (delivery_id,),
        ).fetchone()
        if row is not None:
            return row
        return conn.execute(
            """
            SELECT pr_comment_queue.*
            FROM pr_comment_deliveries
            JOIN pr_comment_queue USING (queue_id)
            WHERE pr_comment_deliveries.delivery_id = ?
            """,
            (delivery_id,),
        ).fetchone()

    def _record_pr_comment_delivery(
        self, conn: sqlite3.Connection, delivery_id: str, queue_id: str
    ) -> None:
        now = _utcnow()
        conn.execute(
            """
            INSERT OR IGNORE INTO pr_comment_deliveries (
                delivery_id, queue_id, created_at
            )
            VALUES (?, ?, ?)
            """,
            (delivery_id, queue_id, now),
        )

    def _pr_comment_by_comment(
        self,
        conn: sqlite3.Connection,
        repo: str,
        pr_number: int,
        comment_type: str,
        comment_id: int,
    ) -> sqlite3.Row | None:
        return conn.execute(
            """
            SELECT *
            FROM pr_comment_queue
            WHERE repo = ? AND pr_number = ? AND comment_type = ? AND comment_id = ?
            """,
            (repo, int(pr_number), comment_type, int(comment_id)),
        ).fetchone()

    def _pr_comment_by_queue_id(
        self, conn: sqlite3.Connection, queue_id: str
    ) -> sqlite3.Row | None:
        return conn.execute(
            "SELECT * FROM pr_comment_queue WHERE queue_id = ?",
            (queue_id,),
        ).fetchone()

    def _pr_comment_record_from_row(self, row: sqlite3.Row) -> PRCommentQueueRecord:
        return PRCommentQueueRecord(
            queue_id=row["queue_id"],
            delivery_id=row["delivery_id"],
            repo=row["repo"],
            pr_number=int(row["pr_number"]),
            comment_type=row["comment_type"],
            comment_id=int(row["comment_id"]),
            author=row["author"],
            is_bot=bool(row["is_bot"]),
            body=row["body"],
            github_created_at=row["github_created_at"],
            state=cast(PRCommentQueueState, row["state"]),
            claim_owner=row["claim_owner"],
            retry_count=int(row["retry_count"]),
            next_retry_after=row["next_retry_after"],
            payload_json=row["payload_json"],
        )


class _FidoTransaction:
    def __init__(self, store: FidoStore) -> None:
        self._store = store
        self._conn: sqlite3.Connection | None = None

    def __enter__(self) -> sqlite3.Connection:
        self._store.ensure_schema()
        self._conn = self._store._connect()  # pyright: ignore[reportPrivateUsage]
        self._conn.execute("BEGIN IMMEDIATE")
        return self._conn

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        assert self._conn is not None
        if exc_type is None:
            self._conn.commit()
        else:
            self._conn.rollback()
        self._conn.close()


def _normalize_comment_ids(comment_ids: Iterable[int]) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(comment_id) for comment_id in comment_ids))


def _utcnow() -> str:
    return datetime.now(tz=UTC).isoformat()


def _retry_after(retry_count: int) -> str:
    delay = min(3600, 2 ** min(retry_count, 12))
    return (datetime.now(tz=UTC) + timedelta(seconds=delay)).isoformat()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS comment_claims (
    comment_id INTEGER PRIMARY KEY,
    owner TEXT NOT NULL CHECK(owner IN ('webhook', 'worker', 'recovery')),
    state TEXT NOT NULL CHECK(state IN (
        'in_progress', 'completed', 'retryable_failed'
    )),
    promise_id TEXT NOT NULL,
    retry_count INTEGER NOT NULL DEFAULT 0,
    next_retry_after TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reply_promises (
    promise_id TEXT PRIMARY KEY,
    owner TEXT NOT NULL CHECK(owner IN ('webhook', 'worker', 'recovery')),
    comment_type TEXT NOT NULL,
    anchor_comment_id INTEGER NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('prepared', 'posted', 'acked', 'failed')),
    retry_count INTEGER NOT NULL DEFAULT 0,
    next_retry_after TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reply_promise_comments (
    promise_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    comment_id INTEGER NOT NULL,
    PRIMARY KEY(promise_id, comment_id),
    UNIQUE(promise_id, position),
    FOREIGN KEY(promise_id) REFERENCES reply_promises(promise_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS reply_artifacts (
    artifact_comment_id INTEGER PRIMARY KEY,
    comment_type TEXT NOT NULL,
    lane_key TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reply_artifact_promises (
    artifact_comment_id INTEGER NOT NULL,
    promise_id TEXT NOT NULL,
    PRIMARY KEY(artifact_comment_id, promise_id),
    FOREIGN KEY(artifact_comment_id) REFERENCES reply_artifacts(artifact_comment_id)
      ON DELETE CASCADE,
    FOREIGN KEY(promise_id) REFERENCES reply_promises(promise_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS deferred_issue_outbox (
    idempotence_key TEXT PRIMARY KEY,
    repo TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    issue_url TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS command_queue (
    command_id TEXT PRIMARY KEY,
    idempotence_key TEXT NOT NULL UNIQUE,
    priority INTEGER NOT NULL,
    state TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    claim_owner TEXT,
    lease_until TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    blocked_reason TEXT,
    failure_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pr_comment_queue (
    queue_id TEXT PRIMARY KEY,
    delivery_id TEXT NOT NULL UNIQUE,
    repo TEXT NOT NULL,
    pr_number INTEGER NOT NULL,
    comment_type TEXT NOT NULL CHECK(comment_type IN ('issues', 'pulls')),
    comment_id INTEGER NOT NULL,
    author TEXT NOT NULL,
    is_bot INTEGER NOT NULL CHECK(is_bot IN (0, 1)),
    body TEXT NOT NULL,
    github_created_at TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN (
        'pending', 'in_progress', 'completed', 'retryable_failed'
    )),
    claim_owner TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    next_retry_after TEXT,
    failure_reason TEXT,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(repo, pr_number, comment_type, comment_id)
);

CREATE TABLE IF NOT EXISTS pr_comment_deliveries (
    delivery_id TEXT PRIMARY KEY,
    queue_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(queue_id) REFERENCES pr_comment_queue(queue_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_pr_comment_queue_pending
ON pr_comment_queue(state, next_retry_after, github_created_at, comment_id);

CREATE INDEX IF NOT EXISTS idx_pr_comment_queue_pr
ON pr_comment_queue(repo, pr_number, github_created_at, comment_id);

CREATE TABLE IF NOT EXISTS implementation_tasks (
    task_id TEXT PRIMARY KEY,
    position INTEGER NOT NULL,
    state TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fido_state (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS provider_sessions (
    provider_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    issue_number INTEGER,
    pr_number INTEGER,
    recovery_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL,
    PRIMARY KEY(provider_id, session_id)
);

CREATE TABLE IF NOT EXISTS check_failures (
    check_name TEXT PRIMARY KEY,
    status_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS issue_cache_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS restart_metadata (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transition_audit_log (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    subject TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _schema_statements() -> tuple[str, ...]:
    return tuple(
        statement.strip() for statement in _SCHEMA.split(";") if statement.strip()
    )
