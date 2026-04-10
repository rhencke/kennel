"""Shared task file operations with flock-based locking."""

from __future__ import annotations

import fcntl
import json
import logging
import random
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from kennel.github import GitHub
from kennel.state import _resolve_git_dir, load_state
from kennel.types import TaskStatus, TaskType

log = logging.getLogger(__name__)


def _task_file(work_dir: Path) -> Path:
    return work_dir / ".git" / "fido" / "tasks.json"


def _locked(path: Path, write: bool = False):
    """Context manager: flock the task file."""

    class Lock:
        def __init__(self):
            self.fd = None

        def __enter__(self):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)
            self.fd = open(path, "r+")
            fcntl.flock(self.fd, fcntl.LOCK_EX)
            return self

        def __exit__(self, *_):
            if self.fd:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
                self.fd.close()

        def read(self) -> list[dict[str, Any]]:
            self.fd.seek(0)
            text = self.fd.read().strip()
            if not text:
                return []
            try:
                result = json.loads(text)
            except json.JSONDecodeError:
                log.warning("corrupt tasks.json — resetting")
                return []
            for t in result:
                if "type" not in t:
                    raise ValueError(
                        f"task {t.get('id', '?')} missing required type field"
                    )
            return result

        def write(self, tasks: list[dict[str, Any]]) -> None:
            self.fd.seek(0)
            self.fd.truncate()
            json.dump(tasks, self.fd, indent=2)
            self.fd.flush()

    return Lock()


def add_task(
    work_dir: Path,
    title: str,
    task_type: TaskType,
    description: str = "",
    status: TaskStatus = TaskStatus.PENDING,
    thread: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Add a task to the shared task file. Returns the new task.

    task_type: mandatory — one of TaskType.CI, TaskType.THREAD, TaskType.SPEC.
    thread: optional {repo, pr, comment_id, review_id} for comment/review tasks.
    """
    if not isinstance(task_type, TaskType):
        raise TypeError(f"task_type must be TaskType, got {type(task_type).__name__}")
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
    path = _task_file(work_dir)
    with _locked(path, write=True) as lock:
        existing = lock.read()
        for t in existing:
            if comment_id is not None:
                # Never re-create a task for the same comment, regardless of status.
                if (t.get("thread") or {}).get("comment_id") == comment_id:
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


def update_task(work_dir: Path, task_id: str, status: TaskStatus) -> bool:
    """Update a task's status. Returns True if found."""
    path = _task_file(work_dir)
    with _locked(path, write=True) as lock:
        tasks = lock.read()
        for t in tasks:
            if t["id"] == task_id:
                t["status"] = str(status)
                lock.write(tasks)
                log.info("task %s → %s", task_id, status)
                return True
    return False


def list_tasks(work_dir: Path) -> list[dict[str, Any]]:
    """Read all tasks."""
    path = _task_file(work_dir)
    with _locked(path) as lock:
        return lock.read()


def complete_by_id(work_dir: Path, task_id: str) -> dict[str, Any] | None:
    """Mark the task with the given ID as completed.

    Returns the task's thread dict if it had one, else None.
    Returns None silently if no matching task is found.
    """
    path = _task_file(work_dir)
    with _locked(path, write=True) as lock:
        tasks = lock.read()
        for t in tasks:
            if t["id"] == task_id and t["status"] != TaskStatus.COMPLETED:
                t["status"] = str(TaskStatus.COMPLETED)
                lock.write(tasks)
                log.info("task completed (id=%s): %s", task_id, t["title"][:80])
                return t.get("thread")
    return None


def has_pending_tasks_for_comment(work_dir: Path, comment_id: int | str) -> bool:
    """Return True if any pending task references the given comment_id."""
    cid = int(comment_id)
    path = _task_file(work_dir)
    with _locked(path) as lock:
        for t in lock.read():
            if t.get("status") == TaskStatus.PENDING:
                if int((t.get("thread") or {}).get("comment_id", -1)) == cid:
                    return True
    return False


def remove_task(work_dir: Path, task_id: str) -> bool:
    """Remove a task. Returns True if found."""
    path = _task_file(work_dir)
    with _locked(path, write=True) as lock:
        tasks = lock.read()
        new_tasks = [t for t in tasks if t["id"] != task_id]
        if len(new_tasks) < len(tasks):
            lock.write(new_tasks)
            return True
    return False


def _format_work_queue(task_list: list[dict[str, Any]]) -> str:
    """Format a task list into work-queue markdown.

    Priority order: CI failures → comment-originated → others.
    Completed tasks appear in a collapsible ``<details>`` section.
    Each line includes a ``<!-- type:X -->`` HTML comment for round-tripping.
    """
    ci_pending: list[tuple[str, str]] = []
    comment_pending: list[tuple[str, str]] = []
    other_pending: list[tuple[str, str]] = []
    completed: list[tuple[str, str]] = []

    def _fmt(t: dict[str, Any]) -> str:
        title = t.get("title", "")
        url = (t.get("thread") or {}).get("url", "")
        return f"[{title}]({url})" if url else title

    for t in task_list:
        status = t.get("status", TaskStatus.PENDING)
        task_type = t.get("type", TaskType.SPEC)
        display = _fmt(t)
        if status == TaskStatus.COMPLETED:
            completed.append((display, task_type))
        elif status in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS):
            title = t.get("title", "")
            if title.startswith("CI failure:"):
                ci_pending.append((display, task_type))
            elif t.get("thread"):
                comment_pending.append((display, task_type))
            else:
                other_pending.append((display, task_type))

    pending = ci_pending + comment_pending + other_pending
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
    *,
    _list_tasks=list_tasks,
    _complete_by_id=complete_by_id,
) -> None:
    """Mark pending ASK tasks complete when their review thread is resolved."""
    task_list = _list_tasks(work_dir)
    ask_tasks = [
        t
        for t in task_list
        if t.get("status") == TaskStatus.PENDING
        and t.get("title", "").upper().startswith("ASK:")
        and t.get("thread")
    ]
    if not ask_tasks:
        return

    try:
        owner, repo_name = repo.split("/", 1)
        threads_data = gh.get_review_threads(owner, repo_name, pr_number)
    except Exception:
        log.exception("sync_tasks: failed to fetch review threads for ASK resolution")
        return

    resolved_ids: set[int] = set()
    for node in (
        threads_data.get("data", {})
        .get("repository", {})
        .get("pullRequest", {})
        .get("reviewThreads", {})
        .get("nodes", [])
    ):
        if node.get("isResolved"):
            comments = node.get("comments", {}).get("nodes", [])
            if comments and comments[0].get("databaseId"):
                resolved_ids.add(int(comments[0]["databaseId"]))

    for task in ask_tasks:
        comment_id = (task.get("thread") or {}).get("comment_id")
        if comment_id and int(comment_id) in resolved_ids:
            log.info(
                "sync_tasks: ASK task thread resolved — completing: %s", task["title"]
            )
            _complete_by_id(work_dir, task["id"])


def sync_tasks(
    work_dir: Path,
    gh: GitHub,
    *,
    _resolve_git_dir_fn=_resolve_git_dir,
    _list_tasks=list_tasks,
    _auto_complete_ask_tasks_fn=_auto_complete_ask_tasks,
) -> None:
    """Sync tasks.json → PR body work queue.

    Python replacement for sync-tasks.sh.  Protected by a flock so concurrent
    calls silently skip rather than race.  Re-runs if tasks.json changes while
    the body is being updated.
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
    try:
        fcntl.flock(sync_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log.info("sync_tasks: another sync running — skipping")
        sync_lock_fd.close()
        return

    try:
        state = load_state(fido_dir)
        issue = state.get("issue")
        if issue is None:
            log.info("sync_tasks: no current issue — nothing to sync")
            return

        try:
            repo = gh.get_repo_info(cwd=work_dir)
            user = gh.get_user()
        except Exception:
            log.exception("sync_tasks: failed to get repo info or user")
            return

        pr_data = gh.find_pr(repo, issue, user)
        if pr_data is None or pr_data.get("state") != "OPEN":
            log.info("sync_tasks: no open PR for issue #%s — nothing to sync", issue)
            return

        pr_number = pr_data["number"]
        _auto_complete_ask_tasks_fn(work_dir, gh, repo, pr_number)

        task_list = _list_tasks(work_dir)
        if not task_list:
            log.info("sync_tasks: no tasks — nothing to sync")
            return

        queue = _format_work_queue(task_list)
        log.info("sync_tasks: syncing task list → PR #%s", pr_number)

        try:
            body = gh.get_pr_body(repo, pr_number)
        except Exception:
            log.exception("sync_tasks: failed to get PR body")
            return

        if "WORK_QUEUE_START" not in body:
            log.info(
                "sync_tasks: PR #%s has no work queue markers — skipping",
                pr_number,
            )
            return

        new_body = _apply_queue_to_body(body, queue)
        try:
            gh.edit_pr_body(repo, pr_number, new_body)
            log.info("sync_tasks: PR #%s work queue synced", pr_number)
        except Exception:
            log.exception("sync_tasks: failed to update PR body")
    finally:
        sync_lock_fd.close()


def sync_tasks_background(
    work_dir: Path, gh: GitHub, *, _start=threading.Thread.start
) -> None:
    """Launch :func:`sync_tasks` in a daemon background thread."""
    t = threading.Thread(
        target=sync_tasks,
        args=(work_dir, gh),
        name=f"sync-{work_dir.name}",
        daemon=True,
    )
    _start(t)
