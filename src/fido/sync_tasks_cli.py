"""CLI wrapper for syncing repo tasks to GitHub."""

import sys
from pathlib import Path

from fido.github import GitHub


def main(argv: list[str] | None = None, *, _GitHub: type[GitHub] = GitHub) -> None:
    from fido.tasks import sync_tasks

    args = sys.argv[1:] if argv is None else argv
    work_dir = Path(args[0]) if args else Path.cwd()
    sync_tasks(work_dir, _GitHub())
