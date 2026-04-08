"""Top-level kennel entry point — dispatches to 'serve' or 'task'."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv

    # TODO: remove this compat shim once shell scripts are fully removed
    if args and args[0] == "task":
        from kennel.cli import main as task_main

        task_main(args[1:])
    elif args and args[0] == "status":
        from pathlib import Path
        from types import SimpleNamespace

        from kennel.config import RepoConfig
        from kennel.status import collect, format_status

        repos: dict[str, RepoConfig] = {}
        for spec in args[1:]:
            name, path_str = spec.split(":", 1)
            repos[name] = RepoConfig(
                name=name, work_dir=Path(path_str).expanduser().resolve()
            )
        print(format_status(collect(SimpleNamespace(repos=repos))))
    elif args and args[0] == "sync-tasks":
        from pathlib import Path

        from kennel.github import GitHub
        from kennel.worker import sync_tasks

        work_dir = Path(args[1]) if len(args) > 1 else Path.cwd()
        sync_tasks(work_dir, GitHub())
    else:
        from kennel.server import run as server_run

        server_run()
