"""Help text for the root ``fido`` launcher."""

from __future__ import annotations


def main() -> None:
    print(
        """usage: ./fido <command> [args...]

Commands:
  help              Show this help.
  up                Run and supervise the fido server in the foreground.
  down              Stop the named fido server container.
  ci                Build CI checks and runtime image cache.
  gen-workflows     Regenerate GitHub Actions workflows from buildx.
  prune             Prune BuildKit cache, bounded by FIDO_BUILDKIT_KEEP_STORAGE.
  make-rocq         Generate Rocq-extracted Python through buildx.
  status            Print fido status.
  task              Manage repo task files.
  gh-status         Set FidoCanCode's GitHub profile status.
  chat              Start an interactive persona session.
  sync-tasks        Sync repo tasks to GitHub.
  tests             Run pytest through the project test entry point.
  traceback         Annotate extracted Python tracebacks.
  repl              Open a Python REPL for a Rocq model with OCaml compare.
  ruff              Run ruff through containerized uv.
  pyright           Run pyright through containerized uv.
  pytest            Run pytest through containerized uv."""
    )
