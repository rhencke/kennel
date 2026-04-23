"""SQLite-backed reply claim and promise coordination."""

import json
import re
import sqlite3
import threading
import uuid
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

ReplyOwner = Literal["webhook", "worker", "recovery"]
ClaimState = Literal["in_progress", "completed", "retryable_failed"]
PromiseState = Literal["prepared", "posted", "acked", "failed"]

REPLY_PROMISE_MARKER_PREFIX = "fido:reply-promise:"
_PROMISE_MARKER_RE = re.compile(r"<!--\s*fido:reply-promise:([0-9a-fA-F-]{36})\s*-->")
_SCHEMA_LOCK = threading.Lock()


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


def append_reply_promise_marker(body: str, promise_id: str | None) -> str:
    """Append a hidden recovery marker to a GitHub reply body."""
    if not promise_id:
        return body
    marker = f"<!-- {REPLY_PROMISE_MARKER_PREFIX}{promise_id} -->"
    if marker in body:
        return body
    return f"{body.rstrip()}\n\n{marker}"


def extract_reply_promise_ids(body: str | None) -> tuple[str, ...]:
    """Return hidden Fido reply-promise ids embedded in *body*."""
    if not body:
        return ()
    return tuple(match.group(1).lower() for match in _PROMISE_MARKER_RE.finditer(body))


class ReplyStore:
    """Repo-local SQLite coordination store.

    The D1 claim/promise tables are live runtime truth.  The broader tables are
    created now so later D-lane issues can refine/prove them without changing
    the database identity or migration root.
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
                    covered_comment_ids_json, state, retry_count, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, 'prepared', 0, ?, ?)
                """,
                (
                    promise_id,
                    owner,
                    comment_type,
                    anchor_comment_id,
                    json.dumps(list(covered)),
                    now,
                    now,
                ),
            )
            for comment_id in covered:
                conn.execute(
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
                    (comment_id, owner, promise_id, now, now),
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

    def ack_promise(self, promise_id: str) -> None:
        """Complete a promise and every covered raw comment id."""
        now = _utcnow()
        with self._transaction() as conn:
            row = self._promise_row(conn, promise_id)
            if row is None:
                return
            covered = tuple(json.loads(row["covered_comment_ids_json"]))
            conn.execute(
                """
                UPDATE reply_promises
                SET state = 'acked', updated_at = ?
                WHERE promise_id = ?
                """,
                (now, promise_id),
            )
            for comment_id in covered:
                conn.execute(
                    """
                    UPDATE comment_claims
                    SET state = 'completed', updated_at = ?, next_retry_after = NULL
                    WHERE comment_id = ?
                    """,
                    (now, int(comment_id)),
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
            delay = min(3600, 2 ** min(retry_count, 12))
            next_retry_after = (now_dt + timedelta(seconds=delay)).isoformat()
            covered = tuple(json.loads(row["covered_comment_ids_json"]))
            conn.execute(
                """
                UPDATE reply_promises
                SET state = 'failed', retry_count = ?, next_retry_after = ?,
                    updated_at = ?
                WHERE promise_id = ?
                """,
                (retry_count, next_retry_after, now, promise_id),
            )
            for comment_id in covered:
                conn.execute(
                    """
                    UPDATE comment_claims
                    SET state = 'retryable_failed', retry_count = ?,
                        next_retry_after = ?, updated_at = ?
                    WHERE comment_id = ?
                    """,
                    (retry_count, next_retry_after, now, int(comment_id)),
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
        return [self._record_from_row(row) for row in rows]

    def promise(self, promise_id: str) -> ReplyPromiseRecord | None:
        """Return one promise record."""
        self.ensure_schema()
        with closing(self._connect()) as conn:
            row = self._promise_row(conn, promise_id)
        return self._record_from_row(row) if row is not None else None

    def ensure_schema(self) -> None:
        """Create the D-lane durable store schema if needed."""
        with _SCHEMA_LOCK:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            with closing(self._connect()) as conn:
                conn.executescript(_SCHEMA)

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
        return _ReplyTransaction(self)

    def _promise_row(
        self, conn: sqlite3.Connection, promise_id: str
    ) -> sqlite3.Row | None:
        return conn.execute(
            "SELECT * FROM reply_promises WHERE promise_id = ?", (promise_id,)
        ).fetchone()

    def _record_from_row(self, row: sqlite3.Row) -> ReplyPromiseRecord:
        return ReplyPromiseRecord(
            promise_id=row["promise_id"],
            owner=row["owner"],
            comment_type=row["comment_type"],
            anchor_comment_id=int(row["anchor_comment_id"]),
            covered_comment_ids=tuple(
                int(comment_id)
                for comment_id in json.loads(row["covered_comment_ids_json"])
            ),
            state=row["state"],
            retry_count=int(row["retry_count"]),
            next_retry_after=row["next_retry_after"],
        )


class _ReplyTransaction:
    def __init__(self, store: ReplyStore) -> None:
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
    covered_comment_ids_json TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('prepared', 'posted', 'acked', 'failed')),
    retry_count INTEGER NOT NULL DEFAULT 0,
    next_retry_after TEXT,
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
