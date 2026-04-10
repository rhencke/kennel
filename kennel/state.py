"""State file and git-dir utilities shared between worker and tasks."""

from __future__ import annotations

import fcntl
import json
import subprocess
from pathlib import Path
from typing import IO, Any


def _state_lock(fido_dir: Path, exclusive: bool = False) -> IO[str]:
    """Open and flock state.lock in fido_dir. Caller must close the returned fd."""
    lock_path = fido_dir / "state.lock"
    lock_path.touch(exist_ok=True)
    lock_fd = open(lock_path)  # noqa: SIM115
    fcntl.flock(lock_fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
    return lock_fd


def load_state(fido_dir: Path) -> dict[str, Any]:
    """Load state.json from fido_dir, returning an empty dict if absent."""
    with _state_lock(fido_dir):
        state_path = fido_dir / "state.json"
        if not state_path.exists():
            return {}
        return json.loads(state_path.read_text())


def save_state(fido_dir: Path, state: dict[str, Any]) -> None:
    """Write state to state.json in fido_dir."""
    with _state_lock(fido_dir, exclusive=True):
        (fido_dir / "state.json").write_text(json.dumps(state))


def clear_state(fido_dir: Path) -> None:
    """Remove state.json from fido_dir (no-op if absent)."""
    with _state_lock(fido_dir, exclusive=True):
        (fido_dir / "state.json").unlink(missing_ok=True)


def _resolve_git_dir(work_dir: Path, *, _run=subprocess.run) -> Path:
    """Return the absolute .git directory for *work_dir*."""
    result = _run(
        ["git", "rev-parse", "--absolute-git-dir"],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())
