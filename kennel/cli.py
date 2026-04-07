"""kennel task CLI — add/complete/list tasks in the shared task file."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from kennel import tasks as _tasks_mod
from kennel.github import GitHub

log = logging.getLogger(__name__)


class Cmd:
    """CLI command handler with injectable dependencies for testability."""

    def __init__(self, *, github: GitHub | None = None, tasks=_tasks_mod) -> None:
        if github is None:
            github = GitHub()

        self._github = github
        self._tasks = tasks

    def _resolve_thread_if_ours(self, thread: dict) -> None:
        """Resolve the review thread if the last reply came from us."""
        repo = thread.get("repo", "")
        pr = thread.get("pr")
        comment_id = thread.get("comment_id")
        if not (repo and pr and comment_id):
            return

        try:
            us = self._github.get_user()
            comments = self._github.get_pull_comments(repo, pr)
            thread_comments = sorted(
                [
                    c
                    for c in comments
                    if c.get("id") == comment_id
                    or c.get("in_reply_to_id") == comment_id
                ],
                key=lambda c: c.get("created_at", ""),
            )
            if not thread_comments:
                return
            last_author = thread_comments[-1].get("user", {}).get("login", "")
            if last_author != us:
                log.info("thread has new replies from %s — not resolving", last_author)
                return

            owner, repo_name = repo.split("/", 1)
            data = self._github.get_review_threads(owner, repo_name, pr)
            threads = data["data"]["repository"]["pullRequest"]["reviewThreads"][
                "nodes"
            ]
            for t in threads:
                if t.get("isResolved"):
                    continue
                nodes = t.get("comments", {}).get("nodes", [])
                if nodes and nodes[0].get("databaseId") == comment_id:
                    self._github.resolve_thread(t["id"])
                    log.info("thread resolved: %s", t["id"])
                    return
        except Exception as exc:  # noqa: BLE001
            log.warning("thread resolution skipped: %s", exc)

    def add(self, work_dir: Path, title: str, description: str) -> None:
        self._tasks.add_task(work_dir, title=title, description=description)

    def complete(self, work_dir: Path, title: str) -> None:
        thread = self._tasks.complete_by_title(work_dir, title)
        if thread:
            self._resolve_thread_if_ours(thread)

    def list(self, work_dir: Path) -> None:
        result = self._tasks.list_tasks(work_dir)
        print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kennel task",
        description="Manage kennel task list for a git repo.",
    )
    parser.add_argument("work_dir", type=Path, help="Path to the git working directory")

    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Add a task")
    p_add.add_argument("title", help="Task title")
    p_add.add_argument(
        "description", nargs="?", default="", help="Optional description"
    )

    p_complete = sub.add_parser("complete", help="Mark a task completed")
    p_complete.add_argument("title", help="Task title")

    sub.add_parser("list", help="List all tasks as JSON")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cmd = Cmd()

    match args.command:
        case "add":
            cmd.add(args.work_dir, args.title, args.description)
        case "complete":
            cmd.complete(args.work_dir, args.title)
        case "list":
            cmd.list(args.work_dir)
        case _:
            raise AssertionError(f"unreachable: unknown command {args.command!r}")
