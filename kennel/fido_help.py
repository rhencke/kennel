"""Help text for the root ``fido`` launcher."""

from __future__ import annotations


def main() -> None:
    print(
        """usage: ./fido <command> [args...]

Commands:
  help              Show this help.
  up [--detach]    Run the kennel server.
  down              Stop the detached kennel server container.
  status            Print kennel status.
  task              Manage repo task files.
  gh-status         Set FidoCanCode's GitHub profile status.
  chat              Start an interactive persona session.
  sync-tasks        Sync repo tasks to GitHub.
  tests             Run pytest through the project test entry point.
  traceback         Annotate extracted Python tracebacks.

Any other command is passed through to `uv run` unchanged."""
    )
