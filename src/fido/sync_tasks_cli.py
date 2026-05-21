"""CLI wrapper for syncing repo tasks to GitHub."""

import sys
from collections.abc import Callable
from pathlib import Path

from fido.github import GitHub


def main(
    argv: list[str] | None = None,
    *,
    _GitHub: type[GitHub] = GitHub,
    _sync_tasks: Callable[..., None] | None = None,
) -> None:
    if _sync_tasks is None:
        from fido.tasks import sync_tasks as _sync_tasks  # pragma: no cover

    args = sys.argv[1:] if argv is None else argv
    work_dir = Path(args[0]) if args else Path.cwd()
    _sync_tasks(work_dir, _GitHub())


if __name__ == "__main__":  # pragma: no cover
    main()
