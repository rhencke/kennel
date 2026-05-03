"""Help text for the root ``fido`` launcher."""


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
  chat              Start a host-side interactive persona session with system Claude.
  sync-tasks        Sync repo tasks to GitHub.
  tests             Run pytest through the project test entry point.
  traceback         Annotate extracted Python tracebacks.
  repl              Open a Python REPL for a Rocq model with OCaml compare.
  rocq-lsp          Run a stdio LSP server for Rocq model files.
  lsp               Query Rocq navigation, actions, graph, rename, and tokens.
  ruff              Run ruff from the prebuilt container toolchain.
  pyright           Run pyright from the prebuilt container toolchain.
  pytest            Run pytest from the prebuilt container toolchain.
  smoke             Run the Codex app-server smoke test."""
    )


if __name__ == "__main__":  # pragma: no cover
    main()
