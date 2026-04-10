"""Shared task file operations with flock-based locking."""

from __future__ import annotations

import fcntl
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

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
