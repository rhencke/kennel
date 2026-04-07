"""Top-level kennel entry point — dispatches to 'serve' or 'task' subcommands."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv

    if args and args[0] == "task":
        from kennel.cli import main as task_main

        task_main(args[1:])
    else:
        from kennel.server import run

        run()
