"""kennel task CLI — add/complete/list tasks in the shared task file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kennel import github, tasks


def _resolve_thread_if_ours(thread: dict) -> None:
    """Resolve the review thread if the last reply came from us."""
    repo = thread.get("repo", "")
    pr = thread.get("pr")
    comment_id = thread.get("comment_id")
    if not (repo and pr and comment_id):
        return

    try:
        us = github.get_user()
        comments = github.get_pull_comments(repo, pr)
        thread_comments = sorted(
            [
                c
                for c in comments
                if c.get("id") == comment_id or c.get("in_reply_to_id") == comment_id
            ],
            key=lambda c: c.get("created_at", ""),
        )
        if not thread_comments:
            return
        last_author = thread_comments[-1].get("user", {}).get("login", "")
        if last_author != us:
            print(
                f"thread has new replies from {last_author} — not resolving",
                file=sys.stderr,
            )
            return

        owner, repo_name = repo.split("/", 1)
        data = github.get_review_threads(owner, repo_name, pr)
        threads = data["data"]["repository"]["pullRequest"]["reviewThreads"]["nodes"]
        for t in threads:
            if t.get("isResolved"):
                continue
            nodes = t.get("comments", {}).get("nodes", [])
            if nodes and nodes[0].get("databaseId") == comment_id:
                github.resolve_thread(t["id"])
                print(f"thread resolved: {t['id']}", file=sys.stderr)
                return
    except Exception as exc:  # noqa: BLE001
        print(f"thread resolution skipped: {exc}", file=sys.stderr)


def cmd_add(work_dir: Path, title: str, description: str) -> None:
    tasks.add_task(work_dir, title=title, description=description)


def cmd_complete(work_dir: Path, title: str) -> None:
    thread = tasks.complete_by_title(work_dir, title)
    if thread:
        _resolve_thread_if_ours(thread)


def cmd_list(work_dir: Path) -> None:
    result = tasks.list_tasks(work_dir)
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

    if args.command == "add":
        cmd_add(args.work_dir, args.title, args.description)
    elif args.command == "complete":
        cmd_complete(args.work_dir, args.title)
    elif args.command == "list":
        cmd_list(args.work_dir)
