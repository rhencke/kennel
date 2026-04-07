"""Fido worker — runs one iteration of the work loop for a single repo."""

from __future__ import annotations

import fcntl
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from kennel import claude, hooks
from kennel.github import GitHub
from kennel.prompts import Prompts, status_system_prompt

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
    """Return the path to sync-tasks.sh.

    TODO: remove once sync-tasks.sh is rewritten to Python.
    """
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
    # TODO: replace with Python sync once sync-tasks.sh is removed
    sync_cmd = f"bash {_sync_script()} {work_dir} &"
    hooks.add_hooks(work_dir, compact_cmd, sync_cmd)
    return compact_cmd, sync_cmd


def teardown_hooks(
    work_dir: Path, fido_dir: Path, compact_cmd: str, sync_cmd: str
) -> None:
    """Remove hooks and the compact script created by setup_hooks."""
    hooks.remove_hooks(work_dir, compact_cmd, sync_cmd)
    (fido_dir / "compact.sh").unlink(missing_ok=True)


def load_state(fido_dir: Path) -> dict[str, Any]:
    """Load state.json from fido_dir, returning an empty dict if absent."""
    state_path = fido_dir / "state.json"
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text())


def save_state(fido_dir: Path, state: dict[str, Any]) -> None:
    """Write state to state.json in fido_dir."""
    (fido_dir / "state.json").write_text(json.dumps(state))


def clear_state(fido_dir: Path) -> None:
    """Remove state.json from fido_dir (no-op if absent)."""
    (fido_dir / "state.json").unlink(missing_ok=True)


class Worker:
    """Fido worker for a single repository.

    Accepts ``work_dir`` and a :class:`~kennel.github.GitHub` client via the
    constructor so that tests can inject a mock without patching module-level
    names.  See :ref:`dependency-injection` in CLAUDE.md.
    """

    def __init__(self, work_dir: Path, gh: GitHub) -> None:
        self.work_dir = work_dir
        self.gh = gh

    def discover_repo_context(self) -> RepoContext:
        """Discover repo metadata for self.work_dir using the GitHub API."""
        repo = self.gh.get_repo_info(cwd=self.work_dir)
        owner, repo_name = repo.split("/", 1)
        gh_user = self.gh.get_user()
        default_branch = self.gh.get_default_branch(cwd=self.work_dir)
        return RepoContext(
            repo=repo,
            owner=owner,
            repo_name=repo_name,
            gh_user=gh_user,
            default_branch=default_branch,
        )

    def get_current_issue(self, fido_dir: Path, repo: str) -> int | None:
        """Return the current issue number from state, or None if there is none.

        If state.json records an issue that has been CLOSED on GitHub, the state
        is cleared (advancing to the next issue) and None is returned.
        """
        state = load_state(fido_dir)
        issue = state.get("issue")
        if issue is None:
            return None
        issue_data = self.gh.view_issue(repo, issue)
        if issue_data["state"] == "CLOSED":
            log.info("issue #%s: closed — advancing", issue)
            clear_state(fido_dir)
            return None
        return int(issue)

    def set_status(self, what: str, busy: bool = True) -> None:
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

        prompts = Prompts(persona)
        raw = claude.generate_status(
            prompt=prompts.status_prompt(what),
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
        self.gh.set_user_status(text, emoji, busy=busy)
        log.info("set_status: %s %s", emoji, text)

    def find_next_issue(self, fido_dir: Path, repo_ctx: RepoContext) -> int | None:
        """Find the next eligible open issue assigned to gh_user.

        An issue is eligible if it has no sub-issues or all sub-issues are CLOSED.

        On success: saves the issue number to state, sets the GitHub status, and
        returns the issue number.  When no eligible issue exists: sets a "done"
        status and returns None.
        """
        log.info("finding next eligible issue")
        issues = self.gh.find_issues(
            repo_ctx.owner, repo_ctx.repo_name, repo_ctx.gh_user
        )
        for issue in issues:
            sub_issues = issue.get("subIssues", {}).get("nodes", [])
            if not sub_issues or all(si["state"] == "CLOSED" for si in sub_issues):
                number = issue["number"]
                title = issue["title"]
                log.info("starting issue #%s: %s", number, title)
                save_state(fido_dir, {"issue": number})
                self.set_status(f"Picking up issue #{number}: {title}")
                return number

        log.info(
            "no eligible issues assigned to %s in %s",
            repo_ctx.gh_user,
            repo_ctx.repo,
        )
        self.set_status("All done — no issues to fetch", busy=False)
        return None

    def post_pickup_comment(
        self, repo: str, issue: int, issue_title: str, gh_user: str
    ) -> None:
        """Post a Fido-flavoured pickup comment on the issue if not already posted.

        Checks whether gh_user has already commented; if so, skips.  Otherwise
        generates the comment via Claude (Opus, using the Fido persona) and posts it.
        Falls back to a plain-text comment if Claude returns nothing.
        """
        comments = self.gh.get_issue_comments(repo, issue)
        if any(c.get("user", {}).get("login") == gh_user for c in comments):
            log.info("issue #%s: pickup comment already exists — skipping", issue)
            return

        persona_path = _sub_dir() / "persona.md"
        try:
            persona = persona_path.read_text()
        except OSError:
            persona = ""

        prompts = Prompts(persona)
        prompt = prompts.pickup_comment_prompt(issue_title)
        msg = claude.generate_reply(prompt)
        if not msg:
            msg = f"Picking up issue: {issue_title}"

        self.gh.comment_issue(repo, issue, msg)
        log.info("posted pickup comment on issue #%s", issue)

    def run(self) -> int:
        """Run one iteration of the worker loop.

        Returns:
            0 — no more work (all done or idle)
            2 — lock held / transient failure (retry later)
        """
        try:
            ctx = create_context(self.work_dir)
        except LockHeld:
            log.warning("another fido is running — exiting")
            return 2
        log.info("worker started for %s (git_dir=%s)", self.work_dir, ctx.git_dir)

        repo_ctx = self.discover_repo_context()
        log.info(
            "repo=%s user=%s default_branch=%s",
            repo_ctx.repo,
            repo_ctx.gh_user,
            repo_ctx.default_branch,
        )

        compact_cmd, sync_cmd = setup_hooks(self.work_dir, ctx.fido_dir)
        try:
            issue = self.get_current_issue(ctx.fido_dir, repo_ctx.repo)
            if issue is None:
                issue = self.find_next_issue(ctx.fido_dir, repo_ctx)
            if issue is None:
                return 0

            issue_data = self.gh.view_issue(repo_ctx.repo, issue)
            issue_title = issue_data["title"]
            self.post_pickup_comment(
                repo_ctx.repo, issue, issue_title, repo_ctx.gh_user
            )
            # TODO: PR management, CI checks, task execution, and merge
        finally:
            teardown_hooks(self.work_dir, ctx.fido_dir, compact_cmd, sync_cmd)

        return 0


def run(work_dir: Path) -> int:
    """Run one iteration of the worker loop.

    Creates a :class:`Worker` with a live :class:`~kennel.github.GitHub` client
    and delegates to :meth:`Worker.run`.  For testing, construct ``Worker``
    directly with a mock ``gh`` instead of patching module-level names.
    """
    return Worker(work_dir, GitHub()).run()
