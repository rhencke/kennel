"""fido task CLI — add/complete/list tasks in the shared task file."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from fido.github import GitHub
from fido.tasks import Tasks
from fido.types import TaskType

log = logging.getLogger(__name__)


class Cmd:
    """CLI command handler with injectable dependencies for testability."""

    def __init__(self, *, github: GitHub) -> None:
        self._github = github

    def add(
        self,
        work_dir: Path,
        task_type: TaskType,
        title: str,
        description: str,
        comment_id: int | None = None,
        repo: str | None = None,
        pr: int | None = None,
    ) -> dict[str, Any]:
        thread: dict[str, Any] | None = None
        if comment_id is not None:
            thread = {"comment_id": comment_id}
            if repo:
                thread["repo"] = repo
            if pr is not None:
                thread["pr"] = pr
        task = Tasks(work_dir).add(
            title=title,
            task_type=task_type,
            description=description,
            thread=thread,
        )
        print(json.dumps(task))
        return task

    def complete(self, work_dir: Path, task_id: str) -> None:
        Tasks(work_dir).complete_with_resolve(task_id, self._github)

    def list(self, work_dir: Path) -> None:
        result = Tasks(work_dir).list()
        print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fido task",
        description="Manage fido task list for a git repo.",
    )
    parser.add_argument("work_dir", type=Path, help="Path to the git working directory")

    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Add a task")
    p_add.add_argument(
        "task_type",
        type=TaskType,
        choices=list(TaskType),
        help="Task type (ci, thread, spec)",
    )
    p_add.add_argument("title", help="Task title")
    p_add.add_argument(
        "description", nargs="?", default="", help="Optional description"
    )
    p_add.add_argument(
        "--comment-id", type=int, default=None, help="PR review comment database ID"
    )
    p_add.add_argument("--repo", default=None, help="Repo in owner/name form")
    p_add.add_argument("--pr", type=int, default=None, help="PR number")

    p_complete = sub.add_parser("complete", help="Mark a task completed")
    p_complete.add_argument("task_id", help="Task ID")

    sub.add_parser("list", help="List all tasks as JSON")

    return parser


def main(argv: list[str] | None = None, *, _GitHub: type[GitHub] = GitHub) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cmd = Cmd(github=_GitHub())

    match args.command:
        case "add":
            cmd.add(
                args.work_dir,
                args.task_type,
                args.title,
                args.description,
                args.comment_id,
                args.repo,
                args.pr,
            )
        case "complete":
            cmd.complete(args.work_dir, args.task_id)
        case "list":
            cmd.list(args.work_dir)
        case _:
            raise AssertionError(f"unreachable: unknown command {args.command!r}")


if __name__ == "__main__":  # pragma: no cover
    main()
