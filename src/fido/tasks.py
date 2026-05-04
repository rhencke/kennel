"""Shared task file operations with flock-based locking."""

import fcntl
import json
import logging
import random
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

from fido.claude import ClaudeClient
from fido.github import GitHub
from fido.prompts import Prompts
from fido.provider import ProviderAgent
from fido.rocq import pr_body_task_store as task_store_oracle
from fido.rocq import task_queue_rescope as rescope_oracle
from fido.rocq import thread_auto_resolve as thread_resolve_oracle
from fido.state import (
    JsonFileStore,
    State,
    _resolve_git_dir,  # pyright: ignore[reportPrivateUsage]
)
from fido.types import (
    ActiveIssue,
    ActivePR,
    ClosedPR,
    RescopeIntent,
    TaskStatus,
    TaskType,
)

log = logging.getLogger(__name__)

# Maximum number of nudge retries when Opus proposes duplicate task titles.
_RESCOPE_MAX_NUDGES = 3


def _task_kind_for_oracle(task: dict[str, Any]) -> task_store_oracle.TaskKind:
    title_upper = task.get("title", "").upper()
    if title_upper.startswith("ASK:"):
        return task_store_oracle.TaskAsk()
    if title_upper.startswith("DEFER:"):
        return task_store_oracle.TaskDefer()
    if title_upper.startswith("CI FAILURE:") or task.get("type") == TaskType.CI:
        return task_store_oracle.TaskCI()
    if task.get("type") == TaskType.THREAD:
        return task_store_oracle.TaskThread()
    return task_store_oracle.TaskSpec()


def _task_status_for_oracle(task: dict[str, Any]) -> task_store_oracle.TaskStatus:
    match task.get("status", TaskStatus.PENDING):
        case TaskStatus.COMPLETED | "completed":
            return task_store_oracle.StatusCompleted()
        case TaskStatus.BLOCKED | "blocked":
            return task_store_oracle.StatusBlocked()
        case _:
            return task_store_oracle.StatusPending()


def _task_source_comment_for_oracle(task: dict[str, Any]) -> int | None:
    comment_id = (task.get("thread") or {}).get("comment_id")
    if comment_id is None:
        return None
    return int(comment_id)


def _thread_task_status_for_oracle(
    task: dict[str, Any],
) -> thread_resolve_oracle.TaskStatus:
    match task.get("status", TaskStatus.PENDING):
        case TaskStatus.COMPLETED | "completed":
            return thread_resolve_oracle.StatusCompleted()
        case TaskStatus.BLOCKED | "blocked":
            return thread_resolve_oracle.StatusBlocked()
        case _:
            return thread_resolve_oracle.StatusPending()


def _thread_task_for_auto_resolve_oracle(
    task: dict[str, Any],
) -> thread_resolve_oracle.ThreadTask | None:
    comment_id = (task.get("thread") or {}).get("comment_id")
    if comment_id is None:
        return None
    return thread_resolve_oracle.ThreadTask(
        thread_task_comment=int(comment_id),
        thread_task_status=_thread_task_status_for_oracle(task),
    )


def thread_tasks_for_auto_resolve_oracle(
    task_list: list[dict[str, Any]],
) -> list[thread_resolve_oracle.ThreadTask]:
    tasks: list[thread_resolve_oracle.ThreadTask] = []
    for task in task_list:
        oracle_task = _thread_task_for_auto_resolve_oracle(task)
        if oracle_task is not None:
            tasks.append(oracle_task)
    return tasks


def thread_comment_author_for_auto_resolve_oracle(
    login: str,
    *,
    fido_logins: frozenset[str],
    owner: str = "",
    collaborators: frozenset[str] = frozenset(),
    allowed_bots: frozenset[str] = frozenset(),
) -> thread_resolve_oracle.ThreadCommentAuthor:
    if login.lower() in fido_logins:
        return thread_resolve_oracle.CommentByFido()
    if login == owner or login in collaborators:
        return thread_resolve_oracle.CommentByActionable()
    if login in allowed_bots or login.endswith("[bot]"):
        return thread_resolve_oracle.CommentByBot()
    return thread_resolve_oracle.CommentIgnored()


def review_thread_for_auto_resolve_oracle(
    node: dict[str, Any],
    gh_user: str,
    *,
    owner: str = "",
    collaborators: frozenset[str] = frozenset(),
    allowed_bots: frozenset[str] = frozenset(),
) -> thread_resolve_oracle.ReviewThread:
    comments: list[thread_resolve_oracle.ThreadComment] = []
    for comment in node.get("comments", {}).get("nodes", []):
        database_id = comment.get("databaseId")
        if database_id is None:
            continue
        author = (comment.get("author") or {}).get("login", "")
        comments.append(
            thread_resolve_oracle.ThreadComment(
                thread_comment_id=int(database_id),
                thread_comment_author=thread_comment_author_for_auto_resolve_oracle(
                    str(author),
                    fido_logins=frozenset({gh_user.lower()}),
                    owner=owner,
                    collaborators=collaborators,
                    allowed_bots=allowed_bots,
                ),
            )
        )
    return thread_resolve_oracle.ReviewThread(
        review_thread_resolved=bool(node.get("isResolved", False)),
        review_thread_comments=comments,
    )


def _review_thread_contains_comment(
    node: dict[str, Any],
    comment_id: int,
) -> bool:
    for comment in node.get("comments", {}).get("nodes", []):
        if comment.get("databaseId") == comment_id:
            return True
    return False


def _thread_lineage_key(thread: dict[str, Any] | None) -> str | None:
    if not thread:
        return None
    key = thread.get("lineage_key")
    return str(key) if key else None


def _thread_lineage_comment_ids(thread: dict[str, Any] | None) -> list[int]:
    if not thread:
        return []
    comment_ids = thread.get("lineage_comment_ids")
    raw_ids = (
        comment_ids if isinstance(comment_ids, list) else [thread.get("comment_id")]
    )
    lineage: list[int] = []
    for comment_id in raw_ids:
        if not isinstance(comment_id, int | str):
            continue
        try:
            value = int(comment_id)
        except TypeError, ValueError:
            continue
        if value > 0 and value not in lineage:
            lineage.append(value)
    return lineage


def _merge_thread_lineage(
    existing_thread: dict[str, Any], new_thread: dict[str, Any]
) -> bool:
    """Merge related source comment ids into an existing task thread."""
    merged = _thread_lineage_comment_ids(existing_thread)
    changed = False
    for comment_id in _thread_lineage_comment_ids(new_thread):
        if comment_id not in merged:
            merged.append(comment_id)
            changed = True
    if changed:
        existing_thread["lineage_comment_ids"] = merged
    if not existing_thread.get("lineage_key") and new_thread.get("lineage_key"):
        existing_thread["lineage_key"] = new_thread["lineage_key"]
        changed = True
    return changed


def _task_store_for_oracle(
    task_list: list[dict[str, Any]],
) -> tuple[task_store_oracle.TaskStore, dict[int, dict[str, Any]]]:
    tasks_by_oracle_id: dict[int, dict[str, Any]] = {}
    rows: dict[int, task_store_oracle.TaskRow] = {}
    order: list[int] = []
    for oracle_id, task in enumerate(task_list, 1):
        order.append(oracle_id)
        tasks_by_oracle_id[oracle_id] = task
        rows[oracle_id] = task_store_oracle.TaskRow(
            title=task.get("title", ""),
            description=task.get("description", ""),
            kind=_task_kind_for_oracle(task),
            status=_task_status_for_oracle(task),
            source_comment=_task_source_comment_for_oracle(task),
        )
    return task_store_oracle.TaskStore(order, rows), tasks_by_oracle_id


def _rescope_task_kind_for_oracle(task: dict[str, Any]) -> rescope_oracle.TaskKind:
    title_upper = task.get("title", "").upper()
    if title_upper.startswith("ASK:"):
        return rescope_oracle.TaskAsk()
    if title_upper.startswith("DEFER:"):
        return rescope_oracle.TaskDefer()
    if title_upper.startswith("CI FAILURE:") or task.get("type") == TaskType.CI:
        return rescope_oracle.TaskCI()
    if task.get("type") == TaskType.THREAD:
        return rescope_oracle.TaskThread()
    return rescope_oracle.TaskSpec()


def _rescope_task_status_for_oracle(
    task: dict[str, Any],
) -> rescope_oracle.TaskStatus:
    match task.get("status", TaskStatus.PENDING):
        case TaskStatus.COMPLETED | "completed":
            return rescope_oracle.StatusCompleted()
        case TaskStatus.BLOCKED | "blocked":
            return rescope_oracle.StatusBlocked()
        case _:
            return rescope_oracle.StatusPending()


def _rescope_task_source_comment_for_oracle(task: dict[str, Any]) -> int | None:
    comment_id = (task.get("thread") or {}).get("comment_id")
    if comment_id is None:
        return None
    return int(comment_id)


def _rescope_state_for_oracle(
    task_list: list[dict[str, Any]],
) -> tuple[
    dict[str, int],
    dict[int, dict[str, Any]],
    list[int],
    dict[int, rescope_oracle.TaskRow],
]:
    ids_by_task_id: dict[str, int] = {}
    tasks_by_oracle_id: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    rows: dict[int, rescope_oracle.TaskRow] = {}
    for oracle_id, task in enumerate(task_list, 1):
        task_id = task["id"]
        ids_by_task_id[task_id] = oracle_id
        tasks_by_oracle_id[oracle_id] = task
        order.append(oracle_id)
        rows[oracle_id] = rescope_oracle.TaskRow(
            title=task.get("title", ""),
            description=task.get("description", ""),
            kind=_rescope_task_kind_for_oracle(task),
            status=_rescope_task_status_for_oracle(task),
            source_comment=_rescope_task_source_comment_for_oracle(task),
        )
    return ids_by_task_id, tasks_by_oracle_id, order, rows


def _rescope_snapshot_order_for_oracle(
    current: list[dict[str, Any]],
    snapshot_ids: frozenset[str],
    ids_by_task_id: dict[str, int],
) -> list[int]:
    return [
        ids_by_task_id[task["id"]]
        for task in current
        if task["id"] in snapshot_ids and task.get("status") != TaskStatus.COMPLETED
    ]


def _rescope_releases_for_oracle(
    current: list[dict[str, Any]],
    ordered_items: list[dict[str, Any]],
    snapshot_ids: frozenset[str],
    ids_by_task_id: dict[str, int],
) -> list[rescope_oracle.RescopeRelease]:
    ordered_by_id: dict[str, dict[str, Any]] = {}
    for item in ordered_items:
        if item.get("id") and item["id"] not in ordered_by_id:
            ordered_by_id[item["id"]] = item
    releases: list[rescope_oracle.RescopeRelease] = []
    for task in current:
        task_id = task["id"]
        if task_id not in snapshot_ids or task.get("status") == TaskStatus.COMPLETED:
            continue
        oracle_id = ids_by_task_id[task_id]
        item = ordered_by_id.get(task_id)
        if item is None:
            decision: rescope_oracle.RescopeOp = rescope_oracle.CompleteTask(oracle_id)
        elif "description" in item and item["description"] != task.get(
            "description", ""
        ):
            decision = rescope_oracle.RewriteTask(
                oracle_id,
                task.get("title", ""),
                item["description"],
            )
        else:
            decision = rescope_oracle.KeepTask(oracle_id)
        releases.append(
            rescope_oracle.RescopeRelease(rescope_oracle.ReleaseACT(), decision)
        )
    return releases


def _materialize_rescope_oracle_result(
    oracle_order: list[int],
    oracle_rows: dict[int, rescope_oracle.TaskRow],
    tasks_by_oracle_id: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    materialized: list[dict[str, Any]] = []
    for oracle_id in oracle_order:
        task = dict(tasks_by_oracle_id[oracle_id])
        row = oracle_rows[oracle_id]
        task["title"] = row.title
        task["description"] = row.description
        if isinstance(row.status, rescope_oracle.StatusCompleted):
            task["status"] = str(TaskStatus.COMPLETED)
        elif isinstance(row.status, rescope_oracle.StatusBlocked):
            task["status"] = str(TaskStatus.BLOCKED)
        else:
            task["status"] = task.get("status", str(TaskStatus.PENDING))
        materialized.append(task)
    return materialized


def _assert_rescope_matches_oracle(
    current: list[dict[str, Any]],
    ordered_items: list[dict[str, Any]],
    snapshot_ids: frozenset[str],
    result: list[dict[str, Any]],
) -> None:
    ids_by_task_id, tasks_by_oracle_id, current_order, rows = _rescope_state_for_oracle(
        current
    )
    snapshot_order = _rescope_snapshot_order_for_oracle(
        current, snapshot_ids, ids_by_task_id
    )
    releases = _rescope_releases_for_oracle(
        current, ordered_items, snapshot_ids, ids_by_task_id
    )
    oracle_order, oracle_rows = rescope_oracle.apply_batched_rescope(
        snapshot_order, current_order, rows, releases
    )
    expected = _materialize_rescope_oracle_result(
        oracle_order, oracle_rows, tasks_by_oracle_id
    )
    if result != expected:
        raise AssertionError("rescope result diverged from Rocq oracle")


def _apply_reorder_with_oracle(
    current: list[dict[str, Any]],
    ordered_items: list[dict[str, Any]],
    snapshot_ids: frozenset[str],
) -> list[dict[str, Any]]:
    ids_by_task_id, tasks_by_oracle_id, current_order, rows = _rescope_state_for_oracle(
        current
    )
    snapshot_order = _rescope_snapshot_order_for_oracle(
        current, snapshot_ids, ids_by_task_id
    )
    releases = _rescope_releases_for_oracle(
        current, ordered_items, snapshot_ids, ids_by_task_id
    )
    oracle_order, oracle_rows = rescope_oracle.apply_batched_rescope(
        snapshot_order, current_order, rows, releases
    )
    return _materialize_rescope_oracle_result(
        oracle_order, oracle_rows, tasks_by_oracle_id
    )


def _task_file(work_dir: Path) -> Path:
    return work_dir / ".git" / "fido" / "tasks.json"


def _locked(path: Path, write: bool = False) -> "_TaskFileLock":  # noqa: ARG001
    """Context manager: flock the task file."""
    return _TaskFileLock(path)


class _TaskFileLock:
    def __init__(self, path: Path) -> None:
        self._path = path
        self.fd: IO[str] | None = None

    def __enter__(self) -> "_TaskFileLock":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.touch(exist_ok=True)
        self.fd = open(self._path, "r+")
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *_: object) -> None:
        if self.fd:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            self.fd.close()

    def _fd(self) -> IO[str]:
        assert self.fd is not None, "Lock used outside context manager"
        return self.fd

    def read(self) -> list[dict[str, Any]]:
        fd = self._fd()
        fd.seek(0)
        text = fd.read().strip()
        if not text:
            return []
        try:
            result = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"corrupt tasks.json: {e}") from e
        for t in result:
            if "type" not in t:
                raise ValueError(f"task {t.get('id', '?')} missing required type field")
        return result

    def write(self, tasks: list[dict[str, Any]]) -> None:
        fd = self._fd()
        fd.seek(0)
        fd.truncate()
        json.dump(tasks, fd, indent=2)
        fd.flush()


def _format_work_queue(task_list: list[dict[str, Any]]) -> str:
    """Format a task list into work-queue markdown.

    Priority order: CI failures → everything else (preserving list order).
    Completed tasks appear in a collapsible ``<details>`` section.
    Each line includes a ``<!-- type:X -->`` HTML comment for round-tripping.
    """
    store, tasks_by_oracle_id = _task_store_for_oracle(task_list)
    projected_rows = task_store_oracle.project_task_store(store)
    pending: list[tuple[str, str]] = []
    completed: list[tuple[str, str]] = []

    def _fmt(row: task_store_oracle.PRBodyRow) -> tuple[str, str]:
        task = tasks_by_oracle_id[row.task]
        title = row.title
        url = (task.get("thread") or {}).get("url", "")
        task_type = task.get("type", TaskType.SPEC)
        display = f"[{title}]({url})" if url else title
        return display, task_type

    for row in projected_rows:
        display = _fmt(row)
        if isinstance(row.status, task_store_oracle.PRCompleted):
            completed.append(display)
        else:
            pending.append(display)

    lines: list[str] = []
    for i, (display, task_type) in enumerate(pending):
        suffix = " **→ next**" if i == 0 else ""
        lines.append(f"- [ ] {display}{suffix} <!-- type:{task_type} -->")

    if completed:
        lines.append("")
        lines.append(f"<details><summary>Completed ({len(completed)})</summary>")
        lines.append("")
        for display, task_type in completed:
            lines.append(f"- [x] {display} <!-- type:{task_type} -->")
        lines.append("</details>")

    return "\n".join(lines)


def _apply_queue_to_body(body: str, queue: str) -> str:
    """Replace the WORK_QUEUE_START/END section in a PR body with *queue*.

    Returns *body* unchanged if the markers are absent.
    """
    start_marker = "<!-- WORK_QUEUE_START -->"
    end_marker = "<!-- WORK_QUEUE_END -->"
    start = body.find(start_marker)
    end = body.find(end_marker)
    if start == -1 or end == -1:
        return body
    start += len(start_marker)
    return body[:start] + "\n" + queue + "\n" + body[end:]


def _auto_complete_ask_tasks(
    work_dir: Path,
    gh: GitHub,
    repo: str,
    pr_number: int | str,
) -> None:
    """Mark pending ASK tasks complete when their review thread is resolved."""
    tasks = Tasks(work_dir)
    task_list = tasks.list()
    ask_tasks = [
        t
        for t in task_list
        if t.get("status") == TaskStatus.PENDING
        and t.get("title", "").upper().startswith("ASK:")
        and t.get("thread")
    ]
    if not ask_tasks:
        return

    owner, repo_name = repo.split("/", 1)
    nodes = gh.get_review_threads(owner, repo_name, pr_number)

    resolved_ids: set[int] = set()
    for node in nodes:
        if node["isResolved"]:
            comments = node["comments"]["nodes"]
            if comments and comments[0].get("databaseId"):
                resolved_ids.add(int(comments[0]["databaseId"]))

    for task in ask_tasks:
        comment_id = (task.get("thread") or {}).get("comment_id")
        if comment_id and int(comment_id) in resolved_ids:
            log.info(
                "sync_tasks: ASK task thread resolved — completing: %s", task["title"]
            )
            tasks.complete_by_id(task["id"])


@contextmanager
def pr_body_lock(work_dir: Path) -> Iterator[None]:
    """Blocking exclusive lock on the PR-body sync.lock file.

    Acquires LOCK_EX (blocking, not LOCK_NB) so callers wait rather than
    skip.  Use to serialize any full-body PR edit against sync_tasks, which
    also acquires this same lock (with LOCK_NB).  Prevents a description
    rewrite from overwriting a concurrent work-queue sync.
    """
    git_dir = _resolve_git_dir(work_dir)
    fido_dir = git_dir / "fido"
    fido_dir.mkdir(parents=True, exist_ok=True)
    lock_path = fido_dir / "sync.lock"
    fd = open(lock_path, "w")  # noqa: SIM115
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fd.close()


def sync_tasks(
    work_dir: Path,
    gh: GitHub,
    *,
    blocking: bool = False,
    _resolve_git_dir_fn: Callable[[Path], Path] = _resolve_git_dir,
    _auto_complete_ask_tasks_fn: Callable[..., None] = _auto_complete_ask_tasks,
) -> None:
    """Sync tasks.json → PR body work queue.

    Protected by a flock so concurrent calls don't race.  By default
    (``blocking=False``) a concurrent sync causes this call to silently skip.
    Pass ``blocking=True`` at authoritative call sites (e.g. post-completion)
    to wait for the lock instead — this guarantees the PR body converges even
    if a background sync holds the lock with stale data.
    """
    try:
        git_dir = _resolve_git_dir_fn(work_dir)
    except subprocess.CalledProcessError:
        log.warning("sync_tasks: could not resolve git dir for %s", work_dir)
        return

    fido_dir = git_dir / "fido"
    fido_dir.mkdir(parents=True, exist_ok=True)
    sync_lock_path = fido_dir / "sync.lock"
    sync_lock_fd = open(sync_lock_path, "w")  # noqa: SIM115
    if blocking:
        fcntl.flock(sync_lock_fd, fcntl.LOCK_EX)
    else:
        try:
            fcntl.flock(sync_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log.info("sync_tasks: another sync running — skipping")
            sync_lock_fd.close()
            return

    try:
        state = State(fido_dir).load()
        issue = state.get("issue")
        if issue is None:
            log.info("sync_tasks: no current issue — nothing to sync")
            return

        repo = gh.get_repo_info(cwd=work_dir)
        user = gh.get_user()

        pr_data = gh.find_pr(repo, issue, user)
        if pr_data is None or pr_data.get("state") != "OPEN":
            log.info("sync_tasks: no open PR for issue #%s — nothing to sync", issue)
            return

        pr_number = pr_data["number"]
        _auto_complete_ask_tasks_fn(work_dir, gh, repo, pr_number)

        task_list = Tasks(work_dir).list()
        if not task_list:
            log.info("sync_tasks: no tasks — nothing to sync")
            return

        queue = _format_work_queue(task_list)
        log.info("sync_tasks: syncing task list → PR #%s", pr_number)

        body = gh.get_pr_body(repo, pr_number)

        has_start = "WORK_QUEUE_START" in body
        has_end = "WORK_QUEUE_END" in body
        if not has_start and not has_end:
            log.info(
                "sync_tasks: PR #%s has no work queue markers — skipping",
                pr_number,
            )
            return
        if not has_start or not has_end:
            log.warning(
                "sync_tasks: PR #%s has incomplete work queue markers "
                "(start=%s end=%s) — skipping",
                pr_number,
                has_start,
                has_end,
            )
            return

        new_body = _apply_queue_to_body(body, queue)
        if new_body == body:
            log.info(
                "sync_tasks: PR #%s work queue already up to date — no change",
                pr_number,
            )
            return
        gh.edit_pr_body(repo, pr_number, new_body)
        log.info("sync_tasks: PR #%s work queue synced", pr_number)
    finally:
        sync_lock_fd.close()


def _parse_reorder_response(raw: str) -> list[dict[str, Any]] | None:
    """Parse the Opus rescope response into a list of task items.

    Uses a :class:`json.JSONDecoder` consume loop — scans *raw* for ``{``,
    attempts :meth:`~json.JSONDecoder.raw_decode` from that position, and
    advances past the decoded span on success or skips the character on
    failure.  Returns the first ``"tasks"`` list found in any decoded object,
    or ``None`` if no valid response is found.
    """
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(raw):
        brace = raw.find("{", pos)
        if brace == -1:
            break
        try:
            obj, end = decoder.raw_decode(raw, brace)
        except json.JSONDecodeError:
            pos = brace + 1
            continue
        if isinstance(obj, dict):
            tasks = obj.get("tasks")
            if isinstance(tasks, list):
                return tasks
        pos = end
    return None


def _make_new_tasks_from_opus(
    ordered_items: list[dict[str, Any]],
    snapshot_ids: frozenset[str],
    current: list[dict[str, Any]] | None = None,
    intents: list[RescopeIntent] | None = None,
) -> list[dict[str, Any]]:
    """Create fresh task dicts for items Opus returned with a null or absent id.

    Items with a non-null id (whether in the snapshot or not) are not treated
    as new — the caller handles snapshot ids via the oracle and ignores
    unrecognised string ids (unchanged behaviour).  Only items where the
    ``"id"`` key is absent or explicitly ``null`` are promoted to new tasks.

    Each new task receives a fresh timestamp-random ID, ``status: "pending"``,
    and ``type: "spec"`` unless Opus specified a different type.  Items with
    blank titles are silently skipped.

    Dedup against post-snapshot thread tasks (#1337): when *current* and
    *intents* are provided, any rescope intent whose ``comment_id`` is already
    covered by a thread task added since the snapshot was taken is treated as
    "already serviced" — the entry-boundary ``create_task`` path produced the
    thread task while Opus was thinking, so any null-id item Opus emits for
    the same intent is a duplicate.  We suppress one null-id item per covered
    intent (in arrival order) to keep at most one task per intent.
    """
    covered_intents = 0
    if current is not None and intents:
        post_snapshot_lineage: set[int] = set()
        for task in current:
            if task.get("id") in snapshot_ids:
                continue
            for cid in _thread_lineage_comment_ids(task.get("thread")):
                post_snapshot_lineage.add(cid)
        for intent in intents:
            if intent.comment_id in post_snapshot_lineage:
                covered_intents += 1

    new_tasks: list[dict[str, Any]] = []
    skipped = 0
    for item in ordered_items:
        if "id" in item and item["id"] is not None:
            continue  # has an id — handled by oracle or ignored as unknown
        title = (item.get("title") or "").strip()
        if not title:
            continue
        if skipped < covered_intents:
            log.info(
                "rescope: dropping duplicate new task %r — already serviced "
                "by a post-snapshot thread task (#1337)",
                title[:80],
            )
            skipped += 1
            continue
        task: dict[str, Any] = {
            "id": f"{int(time.time() * 1000)}-{random.randint(0, 9999):04d}",
            "title": title,
            "type": item.get("type") or "spec",
            "description": item.get("description") or "",
            "status": str(TaskStatus.PENDING),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        new_tasks.append(task)
    return new_tasks


def _find_duplicate_titles(ordered_items: list[dict[str, Any]]) -> list[str]:
    """Return non-empty titles that appear more than once in *ordered_items*.

    Each duplicated title is listed exactly once in the result, in the order
    of its first repeated occurrence.
    """
    seen: set[str] = set()
    duplicates: list[str] = []
    for item in ordered_items:
        title = item.get("title") or ""
        if not title:
            continue
        if title in seen and title not in duplicates:
            duplicates.append(title)
        seen.add(title)
    return duplicates


def _apply_reorder(
    current: list[dict[str, Any]],
    ordered_items: list[dict[str, Any]],
    original_ids: frozenset[str] = frozenset(),
    intents: list[RescopeIntent] | None = None,
) -> list[dict[str, Any]]:
    """Apply Opus-synthesised items to the current task list.

    Rules (in priority order):
    - CI tasks always come first.
    - Non-CI pending tasks follow the snapped queue order for existing tasks;
      new tasks (null/absent id in Opus output) are appended after existing
      pending tasks, CI-first within the new batch.
    - Pending/in_progress tasks that Opus omits are marked completed; the caller
      detects affected in-progress tasks and signals an abort so the worker picks
      the new next task.
    - Tasks added after the original snapshot (IDs not in the snapshot) are
      appended at the end so they are never silently dropped.
    - Completed tasks are always preserved at the end in their original order.
    - Description is updated from Opus's output; title and thread anchor are
      immutable task identity and are preserved.
    - Opus-returned IDs outside the snapshot or duplicated are ignored.
    - Opus-returned items with a null or absent id are treated as new tasks
      and inserted after existing pending tasks (before completed tasks).
    """
    snapshot_ids = original_ids or frozenset(
        t["id"] for t in current if t.get("status") != TaskStatus.COMPLETED
    )
    oracle_result = _apply_reorder_with_oracle(current, ordered_items, snapshot_ids)
    _assert_rescope_matches_oracle(current, ordered_items, snapshot_ids, oracle_result)

    new_tasks = _make_new_tasks_from_opus(
        ordered_items, snapshot_ids, current=current, intents=intents
    )
    if not new_tasks:
        return oracle_result

    # Merge new tasks into oracle result: CI tasks first (in both groups),
    # then non-CI pending (oracle then new), then completed at end.
    completed_status = str(TaskStatus.COMPLETED)
    oracle_pending = [t for t in oracle_result if t.get("status") != completed_status]
    oracle_completed = [t for t in oracle_result if t.get("status") == completed_status]

    ci_new = [t for t in new_tasks if t.get("type") == "ci"]
    non_ci_new = [t for t in new_tasks if t.get("type") != "ci"]

    ci_oracle = [t for t in oracle_pending if t.get("type") == "ci"]
    non_ci_oracle = [t for t in oracle_pending if t.get("type") != "ci"]

    return ci_oracle + ci_new + non_ci_oracle + non_ci_new + oracle_completed


def _compute_thread_changes(
    original: list[dict[str, Any]],
    result: list[dict[str, Any]],
    original_ids: frozenset[str],
) -> list[dict[str, Any]]:
    """Return change records for thread tasks that were completed or modified.

    Only tasks in *original_ids* (those Opus knew about) with a ``thread``
    attachment are reported.  Already-completed tasks are excluded.

    Each record is one of:
    - ``{"task": ..., "kind": "completed"}`` — Opus omitted or marked it done
    - ``{"task": ..., "kind": "modified", "new_title": ..., "new_description": ...}``

    Note: the Rocq model (``TaskCompleted`` vs ``TaskCancelled``) distinguishes
    explicit completion from omission.  Here both map to ``"completed"`` because
    ``_apply_reorder`` normalises omitted tasks into completed rows — the
    ``r is None`` case cannot fire through the current ``reorder_tasks`` path.
    A future change could propagate the distinction (e.g. via a marker field on
    the task dict set by ``_apply_reorder``) so the reply body distinguishes
    "done" from "cancelled".
    """
    result_by_id = {t["id"]: t for t in result}
    changes: list[dict[str, Any]] = []
    for t in original:
        if t["id"] not in original_ids:
            continue
        if not t.get("thread"):
            continue
        if t.get("status") == TaskStatus.COMPLETED:
            continue
        tid = t["id"]
        r = result_by_id.get(tid)
        if r is None or r.get("status") == TaskStatus.COMPLETED:
            changes.append({"task": t, "kind": "completed"})
        elif r.get("title") != t.get("title") or r.get("description") != t.get(
            "description"
        ):
            changes.append(
                {
                    "task": t,
                    "kind": "modified",
                    "new_title": r.get("title", ""),
                    "new_description": r.get("description", ""),
                }
            )
    return changes


def reorder_tasks(
    work_dir: Path,
    commit_summary: str,
    *,
    intents: list[RescopeIntent] | None = None,
    agent: ProviderAgent | None = None,
    prompts: Prompts | None = None,
    issue: ActiveIssue | None = None,
    pr: ActivePR | None = None,
    prior_attempts: list[ClosedPR] | None = None,
    _on_changes: Callable[[list[dict[str, Any]]], None] | None = None,
    _on_inprogress_affected: Callable[[str], None] | None = None,
    _on_done: Callable[[], None] | None = None,
) -> None:
    """Reorder pending tasks by Opus dependency analysis.

    Reads the task list, asks Opus to reorder/rewrite/drop tasks based on
    dependency analysis and recent commits, then atomically writes the result.

    The task list is read twice: once before the Opus call (to build the
    prompt) and once inside the write-lock (to pick up any tasks added while
    Opus was thinking).  Tasks added in that window are preserved at the end
    of the list rather than silently dropped.

    CI tasks always stay first; completed tasks are always preserved.
    An empty or unparseable Opus response leaves the task list unchanged.

    If *_on_changes* is provided and any thread tasks were completed or modified,
    it is called with a list of change records (see :func:`_compute_thread_changes`).

    If *_on_inprogress_affected* is provided and the currently in-progress task
    is marked completed or modified by Opus, it is called with the affected
    task's id so the caller can abort the running worker (targeted at that
    task) and restart on the new next task.  When the in-progress task is
    modified its status is reset to ``pending`` so the worker loop picks it
    up again with the updated description.

    If *_on_done* is provided, it is called after a successful reorder write so
    callers can trigger follow-up work (e.g. rewriting the PR description).
    """
    task_list = Tasks(work_dir).list()
    if not task_list:
        log.info("reorder_tasks: no tasks — skipping")
        return

    if agent is None:
        agent = ClaudeClient()
    if prompts is None:
        prompts = Prompts("")

    original_ids = frozenset(t["id"] for t in task_list)
    prompt = prompts.rescope_prompt(
        task_list,
        commit_summary,
        issue=issue,
        pr=pr,
        prior_attempts=prior_attempts,
        intents=intents,
    )
    raw = agent.run_turn(prompt, model=agent.voice_model)
    if not raw:
        log.warning("reorder_tasks: Opus returned empty response — skipping")
        return

    ordered_items = _parse_reorder_response(raw)
    if ordered_items is None:
        log.warning("reorder_tasks: could not parse Opus response — skipping")
        return

    # Nudge Opus up to _RESCOPE_MAX_NUDGES times if it proposed duplicate
    # titles.  Each turn runs in the same conversation so the model sees its
    # prior responses and the remaining-attempt count in each nudge.
    # If duplicates remain after all nudges, _apply_reorder still preserves
    # immutable task titles while applying any description/order changes.
    for nudge_attempt in range(_RESCOPE_MAX_NUDGES):
        duplicates = _find_duplicate_titles(ordered_items)
        if not duplicates:
            break
        attempts_remaining = _RESCOPE_MAX_NUDGES - nudge_attempt - 1
        log.warning(
            "reorder_tasks: Opus proposed duplicate titles %r — nudging "
            "(attempt %d/%d, %d remaining after this)",
            duplicates,
            nudge_attempt + 1,
            _RESCOPE_MAX_NUDGES,
            attempts_remaining,
        )
        nudge = prompts.rescope_duplicate_nudge(
            duplicates, attempts_remaining=attempts_remaining
        )
        nudge_raw = agent.run_turn(nudge, model=agent.voice_model)
        if not nudge_raw:
            log.warning(
                "reorder_tasks: empty response after duplicate nudge — "
                "proceeding with fallback"
            )
            break
        nudge_items = _parse_reorder_response(nudge_raw)
        if nudge_items is None:
            log.warning(
                "reorder_tasks: unparseable response after duplicate nudge — "
                "proceeding with fallback"
            )
            break
        ordered_items = nudge_items

    path = _task_file(work_dir)
    inprogress_affected = False
    with _locked(path, write=True) as lock:
        current = lock.read()
        inprogress = next(
            (t for t in current if t.get("status") == TaskStatus.IN_PROGRESS), None
        )
        result = _apply_reorder(current, ordered_items, original_ids, intents=intents)
        if inprogress is not None:
            inprogress_in_result = next(
                (t for t in result if t["id"] == inprogress["id"]), None
            )
            if (
                inprogress_in_result is None
                or inprogress_in_result.get("status") == TaskStatus.COMPLETED
            ):
                # Opus omitted the in-progress task — marked completed.
                inprogress_affected = True
                log.info(
                    "reorder_tasks: in-progress task marked completed by Opus: %s",
                    inprogress.get("title", "")[:60],
                )
            elif inprogress_in_result.get("title") != inprogress.get(
                "title"
            ) or inprogress_in_result.get("description") != inprogress.get(
                "description"
            ):
                # Opus modified the in-progress task — reset to pending.
                inprogress_affected = True
                inprogress_in_result["status"] = str(TaskStatus.PENDING)
                log.info(
                    "reorder_tasks: in-progress task modified by Opus — reset to pending: %s",
                    inprogress_in_result.get("title", "")[:60],
                )
        lock.write(result)

    if _on_changes is not None:
        changes = _compute_thread_changes(current, result, original_ids)
        if changes:
            _on_changes(changes)

    if inprogress_affected and _on_inprogress_affected is not None:
        assert inprogress is not None  # inprogress_affected is True
        _on_inprogress_affected(str(inprogress["id"]))

    log.info("reorder_tasks: applied reorder — %d tasks", len(result))

    if _on_done is not None:
        _on_done()


def sync_tasks_background(
    work_dir: Path,
    gh: GitHub,
    *,
    _start: Callable[[threading.Thread], None] = threading.Thread.start,
) -> None:
    """Launch :func:`sync_tasks` in a daemon background thread."""
    t = threading.Thread(
        target=sync_tasks,
        args=(work_dir, gh),
        name=f"sync-{work_dir.name}",
        daemon=True,
    )
    _start(t)


class Tasks(JsonFileStore):
    """Encapsulates task file operations for a single worker directory.

    Abstracts all file access so callers never touch the filesystem directly.
    Instantiate with the work_dir path and inject wherever tasks are needed.

    Inherits :meth:`~JsonFileStore.modify` for atomic read-modify-write of
    the entire task list.
    """

    def __init__(self, work_dir: Path) -> None:
        self._work_dir = work_dir

    @property
    def _data_path(self) -> Path:
        return _task_file(self._work_dir)

    def _default(self) -> list[dict[str, Any]]:
        return []

    def _validate(self, data: object) -> None:
        """Ensure every task has a ``type`` field."""
        assert isinstance(data, list), "tasks.json must hold a list"
        for t in data:
            assert isinstance(t, dict), "task entries must be JSON objects"
            if "type" not in t:
                raise ValueError(f"task {t.get('id', '?')} missing required type field")

    def list(self) -> list[dict[str, Any]]:
        """Return all tasks."""
        with _locked(self._data_path) as lock:
            return lock.read()

    def add(
        self,
        title: str,
        task_type: TaskType,
        description: str = "",
        status: TaskStatus = TaskStatus.PENDING,
        thread: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a task. Returns the new (or existing duplicate) task."""
        if not isinstance(task_type, TaskType):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                f"task_type must be TaskType, got {type(task_type).__name__}"
            )
        title = " ".join(title.split())
        task: dict[str, Any] = {
            "id": f"{int(time.time() * 1000)}-{random.randint(0, 9999):04d}",
            "title": title,
            "type": str(task_type),
            "description": description,
            "status": str(status),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if thread:
            task["thread"] = thread
        comment_id = (thread or {}).get("comment_id")
        lineage_key = _thread_lineage_key(thread)
        with _locked(self._data_path, write=True) as lock:
            existing = lock.read()
            for t in existing:
                existing_thread = t.get("thread") or {}
                existing_lineage_key = _thread_lineage_key(existing_thread)
                if lineage_key is not None and lineage_key == existing_lineage_key:
                    if _merge_thread_lineage(existing_thread, thread or {}):
                        t["thread"] = existing_thread
                        lock.write(existing)
                    log.info(
                        "task already exists for lineage %s (status: %s)",
                        lineage_key,
                        t["status"],
                    )
                    return t
                if comment_id is not None:
                    # Never re-create a task for the same comment, regardless of status.
                    if existing_thread.get("comment_id") == comment_id:
                        if _merge_thread_lineage(existing_thread, thread or {}):
                            t["thread"] = existing_thread
                            lock.write(existing)
                        log.info(
                            "task already exists for comment_id %s (status: %s)",
                            comment_id,
                            t["status"],
                        )
                        return t
                elif t["status"] == TaskStatus.PENDING and t["title"] == title:
                    log.info("task already exists: %s", title[:80])
                    return t
            existing.append(task)
            lock.write(existing)
        log.info("task added: %s", title[:80])
        return task

    def complete_by_id(self, task_id: str) -> dict[str, Any] | None:
        """Mark a task completed. Returns its thread dict or None.

        Returns ``None`` silently if no matching task is found.
        """
        with _locked(self._data_path, write=True) as lock:
            tasks = lock.read()
            for t in tasks:
                if t["id"] == task_id and t["status"] != TaskStatus.COMPLETED:
                    t["status"] = str(TaskStatus.COMPLETED)
                    lock.write(tasks)
                    log.info("task completed (id=%s): %s", task_id, t["title"][:80])
                    return t.get("thread")
        return None

    def complete_with_resolve(
        self,
        task_id: str,
        gh: GitHub,
        *,
        collaborators: frozenset[str] = frozenset(),
        allowed_bots: frozenset[str] = frozenset(),
    ) -> None:
        """Mark a task completed and resolve its review thread if we posted last.

        Combines :meth:`complete_by_id` with the per-task thread-resolve logic
        so both the worker and CLI share one path.  If the task has no thread
        metadata, or we are not the last commenter, the resolve step is skipped
        silently.

        Always triggers a background :func:`sync_tasks` after the completion
        so the PR body checkbox flips even when the worker loop doesn't run
        another sync between this completion and the PR-ready/merge step
        (#988).
        """
        thread = self.complete_by_id(task_id)
        sync_tasks_background(self._work_dir, gh)
        if not thread:
            return
        repo = thread.get("repo", "")
        pr = thread.get("pr")
        comment_id = thread.get("comment_id")
        if not (repo and pr and comment_id):
            return
        try:
            us = gh.get_user()
            owner, repo_name = repo.split("/", 1)
            threads = gh.get_review_threads(owner, repo_name, pr)
            pending_tasks = thread_tasks_for_auto_resolve_oracle(self.list())
            for t in threads:
                if _review_thread_contains_comment(t, int(comment_id)):
                    decision = thread_resolve_oracle.resolution_decision(
                        review_thread_for_auto_resolve_oracle(
                            t,
                            us,
                            owner=owner,
                            collaborators=collaborators,
                            allowed_bots=allowed_bots,
                        ),
                        pending_tasks,
                    )
                    if not isinstance(
                        decision, thread_resolve_oracle.ResolveReviewThread
                    ):
                        log.info("thread has pending same-thread work — not resolving")
                        return
                    gh.resolve_thread(t["id"])
                    log.info("thread resolved: %s", t["id"])
                    return
        except Exception as exc:  # noqa: BLE001
            log.warning("thread resolution skipped: %s", exc)

    def has_pending_for_comment(self, comment_id: int | str) -> bool:
        """Return True if any pending task references *comment_id*."""
        cid = int(comment_id)
        with _locked(self._data_path) as lock:
            for t in lock.read():
                if t.get("status") == TaskStatus.PENDING:
                    if int((t.get("thread") or {}).get("comment_id", -1)) == cid:
                        return True
        return False

    def remove(self, task_id: str) -> bool:
        """Remove a task. Returns True if found."""
        with _locked(self._data_path, write=True) as lock:
            tasks = lock.read()
            new_tasks = [t for t in tasks if t["id"] != task_id]
            if len(new_tasks) < len(tasks):
                lock.write(new_tasks)
                return True
        return False

    def update(self, task_id: str, status: TaskStatus) -> bool:
        """Update a task's status. Returns True if found."""
        with _locked(self._data_path, write=True) as lock:
            tasks = lock.read()
            for t in tasks:
                if t["id"] == task_id:
                    t["status"] = str(status)
                    lock.write(tasks)
                    log.info("task %s → %s", task_id, status)
                    return True
        return False

    def unblock_tasks(self) -> int:
        """Transition all BLOCKED tasks back to PENDING.

        Called when a new PR comment arrives so the worker can re-evaluate
        whether it is still blocked.  Returns the number of tasks unblocked.
        """
        count = 0
        with _locked(self._data_path, write=True) as lock:
            task_list = lock.read()
            for t in task_list:
                if t.get("status") == TaskStatus.BLOCKED:
                    t["status"] = str(TaskStatus.PENDING)
                    count += 1
            if count:
                lock.write(task_list)
        if count:
            log.info("unblocked %d task(s)", count)
        return count
