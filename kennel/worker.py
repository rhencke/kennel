"""Fido worker — runs one iteration of the work loop for a single repo."""

from __future__ import annotations

import fcntl
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from kennel import claude, hooks
from kennel.github import GitHub
from kennel.prompts import status_prompt, status_system_prompt

log = logging.getLogger("kennel")


class LockHeld(Exception):
    """Raised when the fido lock is already held by another process."""


@dataclass
class RepoContext:
    """GitHub repo metadata discovered at worker startup."""

    repo: str  # "owner/repo"
    owner: str
    repo_name: str
    gh_user: str  # authenticated GitHub username
    default_branch: str


@dataclass
class WorkerContext:
    work_dir: Path
    git_dir: Path
    fido_dir: Path
    lock_fd: IO[str]


def _sub_dir() -> Path:
    """Return the path to the sub/ skill-instructions directory."""
    return Path(__file__).parent.parent / "sub"


def _sync_script() -> Path:
    """Return the path to sync-tasks.sh."""
    return Path(__file__).parent.parent / "sync-tasks.sh"


def resolve_git_dir(work_dir: Path) -> Path:
    """Return the absolute .git directory for the repo at work_dir."""
    result = subprocess.run(
        ["git", "rev-parse", "--absolute-git-dir"],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def acquire_lock(fido_dir: Path) -> IO[str]:
    """Acquire the fido lock file exclusively (non-blocking).

    Returns the open file object (must stay open to hold the lock).
    Raises LockHeld if another fido is already running.
    """
    fido_dir.mkdir(parents=True, exist_ok=True)
    lock_path = fido_dir / "lock"
    fd = open(lock_path, "w")  # noqa: SIM115
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fd.close()
        raise LockHeld("another fido is running")
    return fd


def create_context(work_dir: Path) -> WorkerContext:
    """Build a WorkerContext for work_dir, acquiring the fido lock.

    Raises LockHeld if the lock is already held.
    """
    git_dir = resolve_git_dir(work_dir)
    fido_dir = git_dir / "fido"
    lock_fd = acquire_lock(fido_dir)
    return WorkerContext(
        work_dir=work_dir,
        git_dir=git_dir,
        fido_dir=fido_dir,
        lock_fd=lock_fd,
    )


def discover_repo_context(work_dir: Path, gh: GitHub) -> RepoContext:
    """Discover repo metadata for work_dir using the GitHub API."""
    repo = gh.get_repo_info(cwd=work_dir)
    owner, repo_name = repo.split("/", 1)
    gh_user = gh.get_user()
    default_branch = gh.get_default_branch(cwd=work_dir)
    return RepoContext(
        repo=repo,
        owner=owner,
        repo_name=repo_name,
        gh_user=gh_user,
        default_branch=default_branch,
    )


def create_compact_script(fido_dir: Path) -> Path:
    """Write the PostCompact hook script that re-reads skill instructions.

    Returns the path to the created script.
    """
    sub_dir = _sub_dir()
    script_path = fido_dir / "compact.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\n"
        "printf '[fido PostCompact] Re-reading skill instructions"
        " after context compression.\\n\\n'\n"
        f'for f in "{sub_dir}"/*.md; do\n'
        '  printf \'## %s\\n\\n\' "$(basename "$f")"\n'
        '  cat "$f"\n'
        "  printf '\\n\\n'\n"
        "done\n"
    )
    script_path.chmod(0o755)
    return script_path


def setup_hooks(work_dir: Path, fido_dir: Path) -> tuple[str, str]:
    """Set up PostCompact and PostToolUse hooks for a fido session.

    Creates the compact script, registers hooks in settings.local.json,
    and ensures the file is git-excluded.

    Returns (compact_cmd, sync_cmd).
    """
    hooks.ensure_gitexcluded(work_dir)
    compact_script = create_compact_script(fido_dir)
    compact_cmd = f"bash {compact_script}"
    sync_cmd = f"bash {_sync_script()} {work_dir} &"
    hooks.add_hooks(work_dir, compact_cmd, sync_cmd)
    return compact_cmd, sync_cmd


def teardown_hooks(
    work_dir: Path, fido_dir: Path, compact_cmd: str, sync_cmd: str
) -> None:
    """Remove hooks and the compact script created by setup_hooks."""
    hooks.remove_hooks(work_dir, compact_cmd, sync_cmd)
    (fido_dir / "compact.sh").unlink(missing_ok=True)


def set_status(gh: GitHub, what: str, busy: bool = True) -> None:
    """Set the authenticated user's GitHub status using Claude-generated text.

    Reads the persona from sub/persona.md, asks Claude to generate a two-line
    status (emoji + text), then updates the GitHub user status via GraphQL.
    Silently skips if Claude returns an empty or malformed response.
    """
    persona_path = _sub_dir() / "persona.md"
    try:
        persona = persona_path.read_text()
    except OSError:
        persona = ""

    raw = claude.generate_status(
        prompt=status_prompt(persona, what),
        system_prompt=status_system_prompt(),
    )
    if not raw:
        log.warning("set_status: claude returned empty — skipping")
        return

    lines = raw.splitlines()
    if len(lines) < 2:
        log.warning("set_status: expected 2 lines, got %d — skipping", len(lines))
        return

    emoji = lines[0].strip()
    text = lines[1].strip()[:80]
    gh.set_user_status(text, emoji, busy=busy)
    log.info("set_status: %s %s", emoji, text)


def run(work_dir: Path) -> int:
    """Run one iteration of the worker loop.

    Returns:
        0 — no more work (all done or idle)
        2 — lock held / transient failure (retry later)
    """
    try:
        ctx = create_context(work_dir)
    except LockHeld:
        log.warning("another fido is running — exiting")
        return 2
    log.info("worker started for %s (git_dir=%s)", work_dir, ctx.git_dir)

    gh = GitHub()
    repo_ctx = discover_repo_context(work_dir, gh)
    log.info(
        "repo=%s user=%s default_branch=%s",
        repo_ctx.repo,
        repo_ctx.gh_user,
        repo_ctx.default_branch,
    )

    compact_cmd, sync_cmd = setup_hooks(work_dir, ctx.fido_dir)
    try:
        # TODO: main loop implemented in subsequent tasks
        pass
    finally:
        teardown_hooks(work_dir, ctx.fido_dir, compact_cmd, sync_cmd)

    return 0
