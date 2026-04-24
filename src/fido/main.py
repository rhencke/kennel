"""Top-level fido entry point — dispatches to 'serve' or 'task'."""

from __future__ import annotations

import sys


def main(
    argv: list[str] | None = None,
    *,
    _GitHub: type | None = None,
) -> None:
    args = sys.argv[1:] if argv is None else argv

    # TODO: remove this compat shim once shell scripts are fully removed
    if args and args[0] == "task":
        from fido.cli import main as task_main
        from fido.github import GitHub

        task_main(args[1:], _GitHub=_GitHub or GitHub)
    elif args and args[0] == "gh-status":
        from fido.gh_status import main as gh_status_main

        gh_status_main(args[1:])
    elif args and args[0] == "chat":
        from fido.chat import main as chat_main

        chat_main(args[1:])
    elif args and args[0] == "sync-tasks":
        from fido.github import GitHub
        from fido.sync_tasks_cli import main as sync_tasks_main

        sync_tasks_main(args[1:], _GitHub=_GitHub or GitHub)
    else:
        from fido.server import run as server_run

        server_run()
