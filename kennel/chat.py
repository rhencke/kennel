"""kennel chat — interactive claude session with persona and runner-clone context."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, MutableMapping
from pathlib import Path

_RUNNER_CLONE = Path.home() / "home-runner"
_PERSONA_FILE = Path(__file__).resolve().parent.parent / "sub" / "persona.md"


def run(
    argv: list[str],
    *,
    persona_file: Path = _PERSONA_FILE,
    runner_clone: Path = _RUNNER_CLONE,
    chdir: Callable[[Path], None] = os.chdir,
    execvp: Callable[[str, list[str]], None] = os.execvp,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Start an interactive claude session with the persona and runner-clone context.

    Defaults argv to ['/remote-control'] when empty.  Fails with a message on
    stderr and exits 1 if the persona file is missing.  Otherwise exec(2)s into
    ``nice -n19 claude``, replacing the current process.
    """
    if environ is None:
        environ = os.environ

    if not persona_file.exists():
        print(f"persona file not found: {persona_file}", file=sys.stderr)
        sys.exit(1)

    persona = persona_file.read_text()
    args = argv if argv else ["/remote-control"]

    chdir(runner_clone)
    environ["CLAUDE_CODE_NO_FLICKER"] = "1"

    execvp(
        "nice",
        [
            "nice",
            "-n19",
            "claude",
            "--permission-mode=bypassPermissions",
            "--continue",
            "--append-system-prompt",
            persona,
            *args,
        ],
    )
