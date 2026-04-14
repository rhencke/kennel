"""Fido worker — runs one iteration of the work loop for a single repo."""

from __future__ import annotations

import fcntl
import json
import logging
import re
import subprocess
import threading
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Protocol

from kennel import claude, hooks, tasks
from kennel.config import RepoMembership
from kennel.github import GitHub
from kennel.prompts import Prompts, rewrite_description_prompt
from kennel.state import (
    State,
    _resolve_git_dir,
)
from kennel.tasks import Tasks
from kennel.types import TaskStatus, TaskType

_CI_LOG_TAIL = 200  # max lines of failure log to include in the CI prompt

log = logging.getLogger(__name__)

_thread_repo: threading.local = threading.local()


class RepoContextFilter(logging.Filter):
    """Inject the current worker thread's repo name into every log record.

    Set ``_thread_repo.repo_name`` before entering a worker loop to tag all
    log records emitted on that thread with the repo's short name.  Records
    from threads that never set the context default to ``"-"``.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.repo_name = getattr(_thread_repo, "repo_name", "-")  # type: ignore[attr-defined]
        return True


class RepoNameFilter(logging.Filter):
    """Only pass log records whose repo_name matches *short_name*.

    Intended for per-repo file handlers so each log file receives only
    records from that repo's worker thread.  Must be applied after
    :class:`RepoContextFilter` has injected ``repo_name`` onto the record.
    """

    def __init__(self, short_name: str) -> None:
        super().__init__()
        self.short_name = short_name

    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, "repo_name", "-") == self.short_name


class ActivityReporter(Protocol):
    """Structural protocol satisfied by WorkerRegistry.

    Workers use this to report their activity and query the full registry
    snapshot for status generation, without a direct import of WorkerRegistry
    (which would create a circular dependency).
    """

    def report_activity(self, repo_name: str, what: str, busy: bool) -> None: ...

    def get_all_activities(self) -> list[Any]: ...

    def status_update(self) -> AbstractContextManager[None]: ...


class LockHeld(Exception):
    """Raised when the fido lock is already held by another process."""


@dataclass
class RepoContext:
    """GitHub repo metadata discovered at worker startup."""

    repo: str  # "owner/repo"
    owner: str  # URL slug (used in API paths); NOT the human reviewer
    repo_name: str
    gh_user: str  # authenticated GitHub username (the bot itself)
    default_branch: str
    membership: RepoMembership = field(default_factory=RepoMembership)

    @property
    def collaborators(self) -> frozenset[str]:
        """Shortcut for ``self.membership.collaborators``."""
        return self.membership.collaborators


@dataclass
class WorkerContext:
    work_dir: Path
    git_dir: Path
    fido_dir: Path
    lock_fd: IO[str]

    def __enter__(self) -> WorkerContext:
        return self

    def __exit__(self, *args: object) -> None:
        self.lock_fd.close()


def _sub_dir() -> Path:
    """Return the path to the sub/ skill-instructions directory."""
    return Path(__file__).parent.parent / "sub"


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


def build_prompt(fido_dir: Path, subskill: str, context: str) -> tuple[Path, Path]:
    """Write system and prompt files for a sub-Claude session.

    The system file contains ``persona.md`` and ``<subskill>.md`` joined by a
    blank line (matching bash ``printf '%s\\n\\n%s\\n' "$PERSONA" "$skill"``).
    The prompt file contains the context string.

    Returns ``(system_file, prompt_file)`` where both live in *fido_dir*.
    """
    sub = _sub_dir()
    persona = (sub / "persona.md").read_text().rstrip()
    skill = (sub / f"{subskill}.md").read_text().rstrip()
    system_file = fido_dir / "system"
    prompt_file = fido_dir / "prompt"
    system_file.write_text(f"{persona}\n\n{skill}\n")
    prompt_file.write_text(f"{context}\n")
    return system_file, prompt_file


def _sanitize_status_text(text: str) -> str:
    """Strip leading/trailing whitespace and collapse newlines to a single space."""
    return re.sub(r"\s*\n\s*", " ", text).strip()


def _sanitize_slug(raw: str, fallback: str) -> str:
    """Sanitize a branch name slug: lowercase, hyphens only, max 40 chars.

    If the sanitized result is shorter than 3 characters, falls back to a
    slug derived from *fallback* (with any ``(closes #N)`` suffix stripped).
    """
    slug = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")[:40]
    if len(slug) < 3:
        clean = re.sub(r"\(closes\s*#\d+\)", "", fallback, flags=re.IGNORECASE)
        slug = re.sub(r"[^a-z0-9]+", "-", clean.lower()).strip("-")[:40]
    return slug


def claude_start(
    fido_dir: Path,
    model: str = "claude-opus-4-6",
    timeout: int = 300,
    cwd: Path | str | None = None,
) -> str:
    """Start a new sub-Claude session from fido_dir/system and fido_dir/prompt.

    Returns the session_id string (empty string on failure).
    """
    system_file = fido_dir / "system"
    prompt_file = fido_dir / "prompt"
    output = claude.print_prompt_from_file(
        system_file, prompt_file, model, timeout, cwd=cwd
    )
    return claude.extract_session_id(output)


def claude_run(
    fido_dir: Path,
    session_id: str = "",
    model: str = "claude-sonnet-4-6",
    timeout: int = 300,
    cwd: Path | str | None = None,
) -> tuple[str, str]:
    """Continue or start a sub-Claude session, streaming progress as JSON.

    If *session_id* is non-empty the existing session is resumed via
    ``claude --resume``.  Otherwise a new session is started from
    *fido_dir/system* and *fido_dir/prompt*.

    Returns ``(session_id, raw_output)`` where *raw_output* is the full
    stream-json text produced by the claude CLI.
    """
    prompt_file = fido_dir / "prompt"
    if session_id:
        output = claude.resume_session(session_id, prompt_file, model, timeout, cwd=cwd)
        return session_id, output
    system_file = fido_dir / "system"
    output = claude.print_prompt_from_file(
        system_file, prompt_file, model, timeout, cwd=cwd
    )
    new_session_id = claude.extract_session_id(output)
    return new_session_id, output


def _pick_next_task(task_list: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the highest-priority eligible pending task, or ``None``.

    Filters out completed tasks and those prefixed with ``ask:`` or ``defer:``
    (case-insensitive).  Among the remaining candidates the priority order is:

    1. Tasks with ``type`` == ``TaskType.CI``
    2. Everything else (first in list wins, including thread tasks)
    """
    pending = [
        t
        for t in task_list
        if t.get("status") == TaskStatus.PENDING
        and not t.get("title", "").lower().startswith("ask:")
        and not t.get("title", "").lower().startswith("defer:")
    ]
    if not pending:
        return None
    for t in pending:
        if t.get("type") == TaskType.CI:
            return t
    return pending[0]


def _has_pending_asks(task_list: list[dict[str, Any]]) -> bool:
    """Return True if any pending task is an open question (ASK: prefix)."""
    return any(
        t.get("status") == TaskStatus.PENDING
        and t.get("title", "").lower().startswith("ask:")
        for t in task_list
    )


_DECISIVE_REVIEW_STATES = {"APPROVED", "CHANGES_REQUESTED"}


def latest_decisive_review(
    owner_reviews: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the most recent APPROVED or CHANGES_REQUESTED review, or None.

    COMMENTED reviews are non-decisive and do not affect the merge/re-request
    decision, so they are excluded.
    """
    decisive = [r for r in owner_reviews if r.get("state") in _DECISIVE_REVIEW_STATES]
    return decisive[-1] if decisive else None


def should_rerequest_review(
    owner_reviews: list[dict[str, Any]],
    commits: list[dict[str, Any]],
) -> bool:
    """Return True if fido should re-request review from the owner.

    True when the latest decisive owner review is CHANGES_REQUESTED and either
    no timestamps are available or the review pre-dates the latest commit
    (meaning new work has been pushed that addresses the feedback).
    """
    latest_review = latest_decisive_review(owner_reviews)
    if latest_review is None or latest_review.get("state") != "CHANGES_REQUESTED":
        return False
    review_at = latest_review.get("submittedAt", "")
    latest_commit_date = max((c.get("committedDate", "") for c in commits), default="")
    if review_at and latest_commit_date and review_at > latest_commit_date:
        return False
    return True


def ci_ready_for_review(
    checks: list[dict[str, Any]],
    required_checks: list[str],
) -> bool:
    """Return True if CI is clear to request review.

    True when ``required_checks`` is empty (no branch protection), or when
    every required check name has a corresponding check with state ``SUCCESS``.
    """
    if not required_checks:
        return True
    states = {c["name"]: c["state"] for c in checks}
    return all(states.get(name) == "SUCCESS" for name in required_checks)


_BODY_TAG_RE = re.compile(r"<body>\s*(.*?)\s*</body>", re.DOTALL | re.IGNORECASE)


def _extract_body(raw: str | None) -> str:
    """Extract content between <body>...</body> tags from Opus output.

    Returns the extracted content stripped of whitespace, or "" if no body
    tags were found or the raw input was empty.  This enforces the output
    contract in :func:`kennel.prompts.rewrite_description_prompt` — Opus is
    told to wrap its output in body tags so we can strip any preamble or
    trailing commentary.
    """
    if not raw:
        return ""
    m = _BODY_TAG_RE.search(raw)
    if not m:
        return ""
    return m.group(1).strip()


def _write_pr_description(
    gh: Any,
    repo: str,
    pr_number: int,
    issue: int,
    task_list: list[dict[str, Any]],
    existing_body: str = "",
    *,
    _print_prompt=None,
) -> None:
    """Write or rewrite the PR description.

    Handles both initial PR creation (``existing_body=""``; builds work-queue
    section from *task_list*) and post-rescope rewrites (``existing_body``
    contains the current body; preserves the rest section after the ``---``
    divider).

    Generates the description via Opus and writes it back via
    ``gh.edit_pr_body``.  Raises ``ValueError`` when the existing body has no
    ``---`` divider (rewrite precondition) or when Opus returns no
    ``<body>``-tagged content.
    """
    if _print_prompt is None:
        _print_prompt = claude.print_prompt

    divider = "\n\n---\n\n"

    # For a rewrite, only proceed when the divider is present so we know
    # where the description section ends.  For initial write (empty body)
    # skip this guard and build the rest section fresh.
    if existing_body and divider not in existing_body:
        raise ValueError(
            f"_write_pr_description: no --- divider in PR #{pr_number} body"
        )

    # Preserve the existing rest section or build the work-queue from scratch.
    if divider in existing_body:
        rest = existing_body.split(divider, 1)[1]
    else:
        pending = [t for t in task_list if t.get("status") == TaskStatus.PENDING]
        next_task = _pick_next_task(task_list)
        if pending:
            lines = [
                f"- [ ] {t['title']}{' **→ next**' if t is next_task else ''}"
                for t in pending
            ]
            queue = "\n".join(lines)
        else:
            queue = "<!-- no tasks yet -->"
        rest = f"## Work queue\n\n<!-- WORK_QUEUE_START -->\n{queue}\n<!-- WORK_QUEUE_END -->"

    prompt = rewrite_description_prompt(existing_body, task_list)
    raw = _print_prompt(prompt, "claude-opus-4-6", timeout=30)
    new_desc = _extract_body(raw)
    if not new_desc:
        raise ValueError(
            f"_write_pr_description: Opus returned no <body> content for PR #{pr_number}"
            f" (raw={str(raw or '')[:200]!r})"
        )

    # Ensure "Fixes #N" is always present (Opus preserves it for rewrites via
    # prompt rules; for initial writes we append it here).
    if f"Fixes #{issue}" not in new_desc:
        new_desc = f"{new_desc.rstrip()}\n\nFixes #{issue}."

    new_body = f"{new_desc.strip()}{divider}{rest}"
    gh.edit_pr_body(repo, pr_number, new_body)
    log.info("_write_pr_description: PR #%s description written", pr_number)


class Worker:
    """Fido worker for a single repository.

    Accepts ``work_dir`` and a :class:`~kennel.github.GitHub` client via the
    constructor so that tests can inject a mock without patching module-level
    names.  See :ref:`dependency-injection` in CLAUDE.md.
    """

    def __init__(
        self,
        work_dir: Path,
        gh: GitHub,
        abort_task: threading.Event | None = None,
        repo_name: str = "",
        registry: ActivityReporter | None = None,
        membership: RepoMembership | None = None,
        _tasks: Tasks | None = None,
    ) -> None:
        self.work_dir = work_dir
        self.gh = gh
        self._abort_task = abort_task if abort_task is not None else threading.Event()
        self._repo_name = repo_name
        self._registry = registry
        self._membership = membership if membership is not None else RepoMembership()
        self._tasks = _tasks if _tasks is not None else Tasks(work_dir)

    def resolve_git_dir(self, *, _run=subprocess.run) -> Path:
        """Return the absolute .git directory for self.work_dir."""
        return _resolve_git_dir(self.work_dir, _run=_run)

    def create_context(self, *, _run=subprocess.run) -> WorkerContext:
        """Build a WorkerContext for self.work_dir, acquiring the fido lock.

        Raises LockHeld if the lock is already held.
        """
        git_dir = self.resolve_git_dir(_run=_run)
        fido_dir = git_dir / "fido"
        lock_fd = acquire_lock(fido_dir)
        return WorkerContext(
            work_dir=self.work_dir,
            git_dir=git_dir,
            fido_dir=fido_dir,
            lock_fd=lock_fd,
        )

    def setup_hooks(self, fido_dir: Path) -> tuple[str, str]:
        """Set up PostCompact and PostToolUse hooks for a fido session.

        Creates the compact script, registers hooks in settings.local.json,
        and ensures the file is git-excluded.

        Returns (compact_cmd, sync_cmd).
        """
        hooks.ensure_gitexcluded(self.work_dir)
        compact_script = create_compact_script(fido_dir)
        compact_cmd = f"bash {compact_script}"
        sync_cmd = f"uv run kennel sync-tasks {self.work_dir} &"
        hooks.add_hooks(self.work_dir, compact_cmd, sync_cmd)
        return compact_cmd, sync_cmd

    def teardown_hooks(self, fido_dir: Path, compact_cmd: str, sync_cmd: str) -> None:
        """Remove hooks and the compact script created by setup_hooks."""
        hooks.remove_hooks(self.work_dir, compact_cmd, sync_cmd)
        (fido_dir / "compact.sh").unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Business logic
    # ------------------------------------------------------------------

    def discover_repo_context(self) -> RepoContext:
        """Discover repo metadata for self.work_dir using the GitHub API.

        The membership (collaborators list) is taken from ``self._membership``,
        which is populated at kennel server startup — no per-iteration API
        call for collaborators.
        """
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
            membership=self._membership,
        )

    def get_current_issue(self, fido_dir: Path, repo: str) -> int | None:
        """Return the current issue number from state, or None if there is none.

        If state.json records an issue that has been CLOSED on GitHub, the state
        is cleared (advancing to the next issue) and None is returned.
        """
        state_obj = State(fido_dir)
        issue = state_obj.load().get("issue")
        if issue is None:
            return None
        issue_data = self.gh.view_issue(repo, issue)
        if issue_data["state"] == "CLOSED":
            log.info("issue #%s: closed — advancing", issue)
            state_obj.clear()
            return None
        return int(issue)

    def set_status(
        self,
        what: str,
        busy: bool = True,
        *,
        _generate_status_with_session=claude.generate_status_with_session,
        _generate_status_emoji=claude.generate_status_emoji,
        _resume_status=claude.resume_status,
        _sub_dir_fn=_sub_dir,
    ) -> None:
        """Set the authenticated user's GitHub status using Claude-generated text.

        Makes two separate Opus calls: the first generates short status text
        (≤80 chars), the second picks an emoji.  If the text exceeds 80
        characters, retries up to 3 times in the same session to shorten it,
        then truncates as a last resort.  Falls back to ``what[:80]`` if
        Claude returns an empty response for the text call.
        """
        persona_path = _sub_dir_fn() / "persona.md"
        try:
            persona = persona_path.read_text()
        except OSError:
            persona = ""

        prompts = Prompts(persona)

        ctx = (
            self._registry.status_update()
            if self._registry is not None
            else nullcontext()
        )
        with ctx:
            if self._registry is not None:
                self._registry.report_activity(self._repo_name, what, busy)
                activities = [
                    (a.repo_name, a.what, a.busy)
                    for a in self._registry.get_all_activities()
                ]
            else:
                activities = [(self.work_dir.name, what, busy)]

            # Call 1: generate status text
            text, session_id = _generate_status_with_session(
                prompt=prompts.status_text_prompt(activities),
                system_prompt=prompts.status_text_system_prompt(),
            )
            if not text:
                log.warning(
                    "set_status: claude returned empty — using plain-text fallback"
                )
                text = what[:80]
                session_id = ""

            text = _sanitize_status_text(text)

            for _ in range(3):
                if len(text) <= 80 or not session_id:
                    break
                retry_raw = _resume_status(
                    session_id,
                    f"The status text is {len(text)} characters — please shorten it to 80 characters or fewer.",
                )
                if not retry_raw:
                    break
                text = retry_raw.strip()

            if len(text) > 80:
                text = text[:80]

            # Call 2: generate emoji
            emoji = _generate_status_emoji(
                prompt=prompts.status_emoji_prompt(text),
                system_prompt=prompts.status_emoji_system_prompt(),
            )
            if not emoji:
                emoji = ":dog:"

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
                State(fido_dir).save({"issue": number})
                self.set_status(f"Picking up issue #{number}: {title}")
                return number

        log.info(
            "no eligible issues assigned to %s in %s",
            repo_ctx.gh_user,
            repo_ctx.repo,
        )
        self.set_status("All done — no issues to fetch", busy=False)
        return None

    def _git(
        self, args: list[str], check: bool = True, *, _run=subprocess.run
    ) -> subprocess.CompletedProcess[str]:
        """Run a git command in self.work_dir."""
        return _run(
            ["git", *args],
            cwd=self.work_dir,
            capture_output=True,
            text=True,
            check=check,
        )

    def git_clean(self) -> None:
        """Discard uncommitted changes and untracked files in the work tree.

        Runs ``git checkout -- .`` to restore tracked files to HEAD, then
        ``git clean -fd`` to remove untracked files and directories.  Only
        uncommitted changes are affected — pushed commits are never reverted.
        """
        self._git(["checkout", "--", "."])
        result = self._git(["clean", "-fd"])
        cleaned = result.stdout.strip()
        if cleaned:
            log.info("git clean removed:\n%s", cleaned)
        else:
            log.info("git clean: nothing to remove")

    def find_or_create_pr(
        self,
        fido_dir: Path,
        repo_ctx: RepoContext,
        issue: int,
        issue_title: str,
        issue_body: str = "",
    ) -> tuple[int, str]:
        """Find or create the branch and draft PR for *issue*.

        Returns ``(pr_number, slug)`` for an open or freshly-created PR.
        Raises ``RuntimeError`` if setup produces no tasks.

        Workflow:
        - **Existing closed PR**: ignore it and create a fresh PR.
        - **Existing open PR**: check out the branch; run the setup sub-Claude
          if tasks.json is empty (planning not yet done).
        - **No PR**: generate a slug via Claude Haiku, create branch, push,
          run setup sub-Claude, build the PR body (description + work queue),
          then create the draft PR.
        """
        remote = "origin"
        request = f"{issue_title} (closes #{issue})"

        existing = self.gh.find_pr(repo_ctx.repo, issue, repo_ctx.gh_user)

        if existing is not None:
            pr_number = existing["number"]
            slug = existing["headRefName"]
            # Open PR — resume
            log.info("resuming PR #%s on branch %s", pr_number, slug)
            self._git(["fetch", remote])
            try:
                self._git(["checkout", slug])
            except subprocess.CalledProcessError:
                self._git(["checkout", "-b", slug, "--track", f"{remote}/{slug}"])
            task_list = self._tasks.list()
            if not task_list:
                # Try seeding from PR body first (recovers from state reset)
                self.seed_tasks_from_pr_body(repo_ctx.repo, pr_number)
                task_list = self._tasks.list()
            if not task_list:
                log.info("PR #%s has no tasks — running setup", pr_number)
                context = (
                    f"Request: {request}\n"
                    f"Repo: {repo_ctx.repo}\n"
                    f"Branch: {slug}\n"
                    f"PR: {pr_number}\n"
                    f"Fork remote: {remote}\n"
                    f"Upstream: {remote}/{repo_ctx.default_branch}\n"
                    f"Work dir: {self.work_dir}"
                )
                build_prompt(fido_dir, "setup", context)
                session_id = claude_start(fido_dir, cwd=self.work_dir)
                log.info("setup session: %s", session_id)
                with State(fido_dir).modify() as state:
                    state["setup_session_id"] = session_id
                if not self._tasks.list():
                    raise RuntimeError(f"setup produced no tasks for PR #{pr_number}")
            log.info(
                "PR: #%s  https://github.com/%s/pull/%s",
                pr_number,
                repo_ctx.repo,
                pr_number,
            )
            return pr_number, slug

        # Generate branch slug via Haiku
        raw_slug = claude.generate_branch_name(
            "Output ONLY a git branch name: 2-4 lowercase words separated by"
            " hyphens, no issue numbers, summarising this request."
            " No explanation, no punctuation, just the branch name."
            f"\n\nRequest: {request}"
        )
        slug = _sanitize_slug(raw_slug, request)
        log.info("new branch: %s", slug)

        # Create branch from default, push
        # Always start fresh from default branch — delete existing branch if present
        self._git(["fetch", remote])
        self._git(["checkout", repo_ctx.default_branch], check=False)
        self._git(["branch", "-D", slug], check=False)
        self._git(["push", remote, "--delete", slug], check=False)
        self._git(["checkout", "-b", slug, f"{remote}/{repo_ctx.default_branch}"])
        self._git(["commit", "--allow-empty", "-m", "wip: start"])
        self._git(["push", "-u", remote, slug])

        # Run setup sub-Claude (plans tasks before PR is created)
        log.info("running setup (pre-PR)")
        context = (
            f"Request: {request}\n"
            f"Repo: {repo_ctx.repo}\n"
            f"Branch: {slug}\n"
            f"Fork remote: {remote}\n"
            f"Upstream: {remote}/{repo_ctx.default_branch}\n"
            f"Work dir: {self.work_dir}"
        )
        build_prompt(fido_dir, "setup", context)
        session_id = claude_start(fido_dir, cwd=self.work_dir)
        log.info("setup session: %s", session_id)
        with State(fido_dir).modify() as state:
            state["setup_session_id"] = session_id

        if not self._tasks.list():
            raise RuntimeError("setup produced no tasks")

        # Create draft PR, then write the description using the same function
        # used for post-rescope rewrites so both paths share one code path.
        url = self.gh.create_pr(
            repo_ctx.repo,
            request,
            f"Fixes #{issue}.",
            repo_ctx.default_branch,
            slug,
        )
        pr_number = int(url.rstrip("/").split("/")[-1])
        _write_pr_description(
            self.gh, repo_ctx.repo, pr_number, issue, self._tasks.list()
        )
        task_count = len(
            [t for t in self._tasks.list() if t.get("status") == "pending"]
        )
        log.info("PR: #%s opened with %d tasks", pr_number, task_count)
        log.info("PR: #%s  %s", pr_number, url)

        return pr_number, slug

    # ------------------------------------------------------------------
    # CI failure handling
    # ------------------------------------------------------------------

    def _extract_run_id(self, link: str) -> str:
        """Extract the GitHub Actions run ID from a check URL.

        Handles URLs like ``https://github.com/owner/repo/actions/runs/12345/job/67890``.
        Returns an empty string if no run ID is found.
        """
        m = re.search(r"runs/(\d+)", link)
        return m.group(1) if m else ""

    def _filter_ci_threads(
        self,
        nodes: list[dict[str, Any]],
        gh_user: str,
        check_name: str,
    ) -> list[dict[str, Any]]:
        """Return unresolved review threads relevant to a CI failure.

        A thread is included when:
        - it is not resolved,
        - the last commenter is not *gh_user* (i.e. awaiting a response), and
        - at least one comment body contains the check name, "ci", "lint", or
          "format" (case-insensitive).
        """
        keywords = {check_name.lower(), "ci", "lint", "format"}
        result = []
        for node in nodes:
            if node["isResolved"]:
                continue
            comments = node["comments"]["nodes"]
            if not comments:
                continue
            last = comments[-1]
            if (last.get("author") or {}).get("login") == gh_user:
                continue
            bodies = " ".join(c["body"].lower() for c in comments)
            if not any(kw in bodies for kw in keywords):
                continue
            result.append(
                {
                    "first_author": (comments[0].get("author") or {}).get("login", ""),
                    "first_body": comments[0]["body"],
                    "last_author": (last.get("author") or {}).get("login", ""),
                    "last_body": last["body"],
                    "url": comments[0]["url"],
                }
            )
        return result

    def handle_ci(
        self,
        fido_dir: Path,
        repo_ctx: RepoContext,
        pr_number: int,
        slug: str,
    ) -> bool:
        """Check for failing CI checks and run the ci sub-Claude to fix them.

        Returns ``True`` if a CI failure was detected and handled (the caller
        should re-run the work loop immediately).  Returns ``False`` when all
        checks are passing or no checks exist.

        On a failure:
        1. Sets the GitHub user status.
        2. Fetches the run failure log (last ``_CI_LOG_TAIL`` lines).
        3. Collects CI-related unresolved review threads.
        4. Builds the ``ci`` sub-Claude prompt and runs Claude.
        5. Marks the ``CI failure: <check>`` task complete.
        6. Triggers a background sync of the work queue.
        """
        log.info("checking: ci")
        checks = self.gh.pr_checks(repo_ctx.repo, pr_number)
        failing = next(
            (c for c in checks if c.get("state") in ("FAILURE", "ERROR")),
            None,
        )
        if failing is None:
            return False

        check_name = failing["name"]
        log.info("CI failing: %s", check_name)
        self.set_status(f"Fixing CI: {check_name} on PR #{pr_number}")

        run_id = self._extract_run_id(failing.get("link", ""))
        if run_id:
            raw_log = self.gh.get_run_log(repo_ctx.repo, run_id)
            lines = raw_log.splitlines()
            failure_log = "\n".join(lines[-_CI_LOG_TAIL:])
        else:
            failure_log = ""

        ci_threads = self._filter_ci_threads(
            self.gh.get_review_threads(repo_ctx.owner, repo_ctx.repo_name, pr_number),
            repo_ctx.gh_user,
            check_name,
        )

        context = (
            f"PR: {pr_number}\n"
            f"Repo: {repo_ctx.repo}\n"
            f"Branch: {slug}\n"
            f"Upstream: origin/{repo_ctx.default_branch}\n"
            f"Work dir: {self.work_dir}\n"
            f"Failing check: {check_name}\n"
            f"\nFailure log (last {_CI_LOG_TAIL} lines):\n{failure_log}\n"
            f"\nReview threads related to this CI failure"
            f" (JSON — may be empty):\n{json.dumps(ci_threads)}"
        )
        build_prompt(fido_dir, "ci", context)
        session_id, _ = claude_run(fido_dir, cwd=self.work_dir)
        log.info("CI fix done (session=%s)", session_id)

        # CI failures have no task entry — no complete call needed
        tasks.sync_tasks(self.work_dir, self.gh)
        return True

    def _filter_threads(
        self,
        nodes: list[dict[str, Any]],
        gh_user: str,
        collaborators: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        """Return unresolved review threads for the comments sub-Claude.

        A thread is included when:
        - it is not resolved,
        - it has at least one comment,
        - the last commenter is not *gh_user* (awaiting a response), and
        - the last commenter is either in *collaborators* or ends with ``[bot]``.
        """
        result = []
        for node in nodes:
            if node["isResolved"]:
                continue
            comments = node["comments"]["nodes"]
            if not comments:
                continue
            first_comment = comments[0]
            last_comment = comments[-1]
            last_author = (last_comment.get("author") or {}).get("login", "")
            if last_author == gh_user:
                continue
            if last_author not in collaborators and not last_author.endswith("[bot]"):
                continue
            first_login = (first_comment.get("author") or {}).get("login", "")
            result.append(
                {
                    "id": node["id"],
                    "is_bot": first_login.endswith("[bot]"),
                    "first_author": first_login,
                    "first_db_id": first_comment.get("databaseId"),
                    "first_body": first_comment["body"],
                    "last_author": last_author,
                    "last_body": last_comment["body"],
                    "url": first_comment["url"],
                    "total": len(comments),
                }
            )
        return result

    def resolve_addressed_threads(
        self,
        repo_ctx: RepoContext,
        pr_number: int,
    ) -> bool:
        """Resolve unresolved threads where gh_user posted the last reply.

        These are threads fido already acknowledged; the work has since been
        done, so the thread can be closed.

        Returns ``True`` if at least one thread was resolved.
        """
        nodes = self.gh.get_review_threads(
            repo_ctx.owner, repo_ctx.repo_name, pr_number
        )
        resolved_any = False
        for node in nodes:
            if node["isResolved"]:
                continue
            comments = node["comments"]["nodes"]
            if not comments:
                continue
            last_author = (comments[-1].get("author") or {}).get("login", "")
            if last_author != repo_ctx.gh_user:
                continue
            comment_ids = [
                c.get("databaseId") for c in comments if c.get("databaseId") is not None
            ]
            if any(self._tasks.has_pending_for_comment(cid) for cid in comment_ids):
                log.info(
                    "skipping resolve for thread %s — pending sibling tasks remain",
                    node["id"],
                )
                continue
            thread_id = node["id"]
            self.gh.resolve_thread(thread_id)
            log.info("resolved thread %s", thread_id)
            resolved_any = True
        return resolved_any

    def handle_threads(
        self,
        fido_dir: Path,
        repo_ctx: RepoContext,
        pr_number: int,
        slug: str,
    ) -> bool:
        """Check for unresolved review threads and run the comments sub-Claude.

        Returns ``True`` if unresolved threads were found and handled.  Returns
        ``False`` if there are no actionable threads.
        """
        log.info("checking: threads")
        threads = self._filter_threads(
            self.gh.get_review_threads(repo_ctx.owner, repo_ctx.repo_name, pr_number),
            repo_ctx.gh_user,
            repo_ctx.collaborators,
        )
        if not threads:
            return False
        log.info("unresolved threads: %d", len(threads))
        context = (
            f"PR: {pr_number}\n"
            f"Repo: {repo_ctx.repo}\n"
            f"Owner: {repo_ctx.owner}\n"
            f"Repo name: {repo_ctx.repo_name}\n"
            f"Branch: {slug}\n"
            f"Upstream: origin/{repo_ctx.default_branch}\n"
            f"Work dir: {self.work_dir}\n"
            f"GitHub user: {repo_ctx.gh_user}\n"
            f"\nUnresolved threads (JSON):\n{json.dumps({'threads': threads})}"
        )
        build_prompt(fido_dir, "comments", context)
        session_id, _ = claude_run(fido_dir, cwd=self.work_dir)
        log.info("threads done (session=%s)", session_id)
        tasks.sync_tasks_background(self.work_dir, self.gh)
        return True

    def ensure_pushed(self, remote: str, slug: str) -> bool | None:
        """Ensure the branch is pushed to *remote*.

        Compares the local HEAD SHA to ``remote/slug``.  If they already
        match, returns ``None`` immediately (no push needed — already in sync).
        Otherwise attempts ``git push`` and returns ``True`` on success,
        ``False`` if the push fails.
        """
        local = self._git(["rev-parse", "HEAD"]).stdout.strip()
        remote_ref = f"{remote}/{slug}"
        result = self._git(["rev-parse", remote_ref], check=False)
        if result.returncode == 0 and result.stdout.strip() == local:
            return None
        log.info("pushing %s to %s", slug, remote)
        push = self._git(["push", "-u", remote, slug], check=False)
        if push.returncode != 0:
            log.warning("push failed: %s", push.stderr.strip())
            return False
        return True

    def _squash_wip_commit(self, remote: str, slug: str, default_branch: str) -> bool:
        """Drop the empty 'wip: start' sentinel if it is the branch root.

        Checks the first commit on the branch (relative to *remote*'s
        *default_branch*).  If its message is ``wip: start`` and it carries no
        file changes, rebases to remove it and force-pushes, returning ``True``.
        Returns ``False`` if the sentinel is not present (first commit has a
        different message, branch has no commits beyond the base, or any git
        operation fails).
        """
        # Find the merge-base between HEAD and the remote default branch
        base = self._git(
            ["merge-base", "HEAD", f"{remote}/{default_branch}"],
            check=False,
        )
        if base.returncode != 0:
            return False
        base_sha = base.stdout.strip()

        # List commits on branch in reverse (oldest first)
        log_result = self._git(
            ["log", "--format=%H %s", f"{base_sha}..HEAD", "--reverse"],
            check=False,
        )
        if log_result.returncode != 0 or not log_result.stdout.strip():
            return False

        first_line = log_result.stdout.strip().splitlines()[0]
        wip_sha, _, subject = first_line.partition(" ")
        if subject != "wip: start":
            return False

        # Confirm it is truly empty (no file diff vs its parent)
        diff = self._git(["diff-tree", "--no-commit-id", "-r", wip_sha], check=False)
        if diff.returncode != 0 or diff.stdout.strip():
            return False

        # Drop the wip commit by rebasing everything after it onto base_sha
        rebase = self._git(["rebase", "--onto", base_sha, wip_sha, slug], check=False)
        if rebase.returncode != 0:
            log.warning("squash wip: start rebase failed: %s", rebase.stderr.strip())
            return False

        # Force-push the cleaned branch
        push = self._git(
            ["push", "--force-with-lease", "-u", remote, slug], check=False
        )
        if push.returncode != 0:
            log.warning("squash wip: start force-push failed: %s", push.stderr.strip())
            return False

        log.info("squashed wip: start sentinel from %s", slug)
        return True

    def _cleanup_aborted_task(
        self, fido_dir: Path, task_id: str, task_title: str
    ) -> None:
        """Discard uncommitted changes and remove task after an abort signal.

        Called when ``self._abort_task`` is set mid-execution.  Runs
        ``git_clean`` to restore the working tree, removes the task from the
        queue, clears ``current_task_id`` from state, resets the abort event,
        and syncs the PR work queue.
        """
        log.info("task aborted: %s", task_title)
        self.git_clean()
        self._tasks.remove(task_id)
        with State(fido_dir).modify() as state:
            state.pop("current_task_id", None)
        self._abort_task.clear()
        tasks.sync_tasks(self.work_dir, self.gh)

    def execute_task(
        self,
        fido_dir: Path,
        repo_ctx: RepoContext,
        pr_number: int,
        slug: str,
    ) -> bool:
        """Pick and execute the next pending task via the task sub-Claude.

        Priority order: CI-failure tasks first, then thread-originated tasks,
        then all others.  Skips tasks whose titles begin with ``ask:`` or
        ``defer:`` (case-insensitive).

        Returns ``True`` if a task was executed, ``False`` when no eligible
        task was found.
        """
        log.info("checking: tasks")
        task_list = self._tasks.list()
        task = _pick_next_task(task_list)
        if task is None:
            return False
        task_title = task["title"]
        log.info("task: %s", task_title)
        self.set_status(f"Working on: {task_title}")
        thread = task.get("thread") or {}
        context_parts = [
            f"PR: {pr_number}",
            f"Repo: {repo_ctx.repo}",
            f"Branch: {slug}",
            f"Upstream: origin/{repo_ctx.default_branch}",
            f"\nTask title: {task_title}",
        ]
        if thread.get("comment_id"):
            context_parts.append(
                f"\nThis task originated from a PR review comment."
                f"\nThread comment_id: {thread['comment_id']}"
                f"\nPR: {thread.get('pr', pr_number)}"
            )
            if thread.get("url"):
                context_parts.append(f"Thread URL: {thread['url']}")
        context = "\n".join(context_parts)
        build_prompt(fido_dir, "task", context)
        head_before = self._git(["rev-parse", "HEAD"]).stdout.strip()
        with State(fido_dir).modify() as state:
            setup_session_id = state.get("setup_session_id", "")
            state["current_task_id"] = task["id"]
        session_id, output = claude_run(
            fido_dir, session_id=setup_session_id, cwd=self.work_dir
        )
        log.info("task done (session=%s)", session_id)
        head_after = self._git(["rev-parse", "HEAD"]).stdout.strip()

        if self._abort_task.is_set():
            self._cleanup_aborted_task(fido_dir, task["id"], task_title)
            return True

        # Resume loop: let Claude cook until commits appear
        attempt = 0
        while head_before == head_after:
            # If the task was completed externally (e.g. via `kennel task
            # complete`) while we were waiting, stop retrying — the work is
            # already recorded as done and no commits will ever appear.
            current_task_list = self._tasks.list()
            if not any(
                t["id"] == task["id"] and t.get("status") == TaskStatus.PENDING
                for t in current_task_list
            ):
                log.info(
                    "task externally completed — stopping retry (id=%s)", task["id"]
                )
                break
            attempt += 1
            if session_id:
                log.info(
                    "task produced no commits — resuming session (attempt %d)",
                    attempt,
                )
                session_id, output = claude_run(
                    fido_dir, session_id=session_id, cwd=self.work_dir
                )
            else:
                log.info(
                    "task produced no commits — starting fresh session (attempt %d)",
                    attempt,
                )
                build_prompt(fido_dir, "task", context)
                session_id, output = claude_run(fido_dir, cwd=self.work_dir)
            log.info("task resume done (session=%s)", session_id)
            head_after = self._git(["rev-parse", "HEAD"]).stdout.strip()

            if self._abort_task.is_set():
                self._cleanup_aborted_task(fido_dir, task["id"], task_title)
                return True

        if session_id:
            with State(fido_dir).modify() as state:
                state["setup_session_id"] = session_id

        self._squash_wip_commit("origin", slug, repo_ctx.default_branch)
        pushed = self.ensure_pushed("origin", slug)
        if pushed is not False:
            self._tasks.complete_by_id(task["id"])
            with State(fido_dir).modify() as state:
                state.pop("current_task_id", None)
            tasks.sync_tasks(self.work_dir, self.gh)
        return True

    def seed_tasks_from_pr_body(self, repo: str, pr_number: int) -> None:
        """Seed tasks.json from the PR body work-queue markers if tasks.json is empty.

        Extracts ONLY unchecked task items (``- [ ] ...``) between
        ``WORK_QUEUE_START`` and ``WORK_QUEUE_END`` markers and adds them via
        :func:`~kennel.tasks.add_task`.  Completed (``- [x]``) tasks are
        skipped — the work is already in the git history; re-creating them
        as pending would send fido into a "no commits produced" retry loop.
        Lines without a ``<!-- type:X -->`` comment are skipped with a
        warning (e.g. stale multi-line task bodies from older PR bodies).

        No-op if tasks.json is already non-empty, or if no markers /
        unchecked items are found.
        """
        if self._tasks.list():
            return  # already have tasks — nothing to seed
        pr_data = self.gh.get_pr(repo, pr_number)
        body = pr_data.get("body") or ""
        match = re.search(
            r"<!-- WORK_QUEUE_START -->(.*?)<!-- WORK_QUEUE_END -->",
            body,
            re.DOTALL,
        )
        if not match:
            return
        parsed: list[tuple[str, TaskType]] = []
        for line in match.group(1).splitlines():
            m = re.match(r"^- \[ \] (.+)$", line)
            if not m:
                continue
            rest = m.group(1)
            type_match = re.search(r"<!-- type:(\w+) -->", rest)
            if not type_match:
                log.warning("skipping task line without type marker: %r", line)
                continue
            raw_type = type_match.group(1)
            task_type = TaskType(raw_type)
            title = re.sub(r"\s*<!-- type:\w+ -->\s*", "", rest)
            title = re.sub(r"\s*\*\*→ next\*\*\s*", "", title).strip()
            if not title:
                continue
            parsed.append((title, task_type))
        if not parsed:
            return
        for title, task_type in parsed:
            self._tasks.add(title, task_type)
        log.info("seeded %d tasks from PR body", len(parsed))

    def post_pickup_comment(
        self, repo: str, issue: int, issue_title: str, gh_user: str
    ) -> None:
        """Post a Fido-flavoured pickup comment on the issue if not already posted.

        Checks whether gh_user has commented since the issue was last opened
        (handles reopened issues).  Otherwise generates the comment via Claude
        (Opus, using the Fido persona) and posts it.
        Falls back to a plain-text comment if Claude returns nothing.
        """
        issue_data = self.gh.view_issue(repo, issue)
        issue_created = issue_data.get("created_at", "")
        # For reopened issues, use the most recent open event
        events = self.gh.get_issue_events(repo, issue)
        last_opened = issue_created
        for e in events:
            if e.get("event") == "reopened":
                last_opened = e.get("created_at", last_opened)

        comments = self.gh.get_issue_comments(repo, issue)
        has_recent_comment = any(
            c.get("user", {}).get("login") == gh_user
            and c.get("created_at", "") >= last_opened
            for c in comments
        )
        if has_recent_comment:
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

    def handle_promote_merge(
        self,
        fido_dir: Path,
        repo_ctx: RepoContext,
        pr_number: int,
        slug: str,
        issue: int,
    ) -> int:
        """Handle promote/merge at the end of a work iteration.

        Called when CI, review feedback, threads, and pending tasks are all
        clear.  Checks the review state and either merges, enables auto-merge,
        re-requests review, promotes a draft PR, requests review (once CI
        passes), or sets the idle status.

        Review is only requested when all required CI checks are passing (or
        when there are no required checks on the target branch).

        Returns:
            0 — no more work (auto-merge enabled, changes-requested,
                setup not complete, waiting for CI, or review in progress)
            1 — did work (merged and cleaned up, or promoted draft;
                caller should re-run immediately)
        """
        log.info("PR #%s: checking review status", pr_number)
        reviews_data = self.gh.get_reviews(repo_ctx.repo, pr_number)
        reviews = reviews_data.get("reviews", [])
        commits = reviews_data.get("commits", [])
        is_draft = reviews_data.get("isDraft", False)
        requested_reviewers = reviews_data.get("requestedReviewers", [])

        collab_reviews = [
            r
            for r in reviews
            if r.get("author", {}).get("login") in repo_ctx.collaborators
        ]
        _latest_decisive = latest_decisive_review(collab_reviews)
        latest_state = (
            _latest_decisive.get("state", "NONE") if _latest_decisive else "NONE"
        )
        is_approved = latest_state == "APPROVED"
        task_list = self._tasks.list()
        pending = [t for t in task_list if t.get("status") == TaskStatus.PENDING]

        # Merge only if: approved + not draft + no pending tasks
        if is_approved and not is_draft and not pending:
            pr_info = self.gh.get_pr(repo_ctx.repo, pr_number)
            merge_state = pr_info.get("mergeStateStatus", "")
            if merge_state == "BLOCKED":
                log.info(
                    "PR #%s approved but merge blocked — enabling auto-merge",
                    pr_number,
                )
                self.gh.pr_merge(repo_ctx.repo, pr_number, squash=True, auto=True)
                return 0
            log.info("PR #%s approved by %s — merging", pr_number, repo_ctx.owner)
            self.gh.pr_merge(repo_ctx.repo, pr_number, squash=True)
            (fido_dir / "tasks.json").write_text("[]")
            State(fido_dir).clear()
            self._git(["checkout", repo_ctx.default_branch])
            self._git(
                ["pull", "origin", repo_ctx.default_branch, "--ff-only"], check=False
            )
            self._git(["branch", "-d", slug], check=False)
            self._git(["push", "origin", "--delete", slug], check=False)
            self.set_status(f"Merged PR #{pr_number}! Issue #{issue} done")
            return 1

        if _has_pending_asks(task_list):
            log.info(
                "PR #%s: open questions pending — deferring ready/review", pr_number
            )
            return 0

        if latest_state == "CHANGES_REQUESTED":
            if not should_rerequest_review(collab_reviews, commits):
                log.info(
                    "PR #%s: CHANGES_REQUESTED review newer than latest commit — skipping re-request",
                    pr_number,
                )
                return 0
            missing = sorted(repo_ctx.collaborators - set(requested_reviewers))
            if missing:
                checks = self.gh.pr_checks(repo_ctx.repo, pr_number)
                required = self.gh.get_required_checks(
                    repo_ctx.repo, repo_ctx.default_branch
                )
                if ci_ready_for_review(checks, required):
                    log.info(
                        "PR #%s: changes requested — all addressed, CI passing — re-requesting review from %s",
                        pr_number,
                        ", ".join(missing),
                    )
                    self.gh.add_pr_reviewers(repo_ctx.repo, pr_number, missing)
                else:
                    log.info(
                        "PR #%s: changes requested — all addressed, but CI not yet passing — deferring re-request",
                        pr_number,
                    )
            return 0

        if is_draft:
            completed = [
                t for t in task_list if t.get("status") == TaskStatus.COMPLETED
            ]
            if not completed:
                log.info(
                    "PR #%s: no tasks completed — not promoting (setup may have failed)",
                    pr_number,
                )
                return 0
            if pending:
                log.info(
                    "PR #%s: %d tasks still pending — not promoting yet",
                    pr_number,
                    len(pending),
                )
                return 0
            log.info("PR #%s: work complete — marking ready", pr_number)
            self.gh.pr_ready(repo_ctx.repo, pr_number)
            missing = sorted(repo_ctx.collaborators - set(requested_reviewers))
            if missing:
                checks = self.gh.pr_checks(repo_ctx.repo, pr_number)
                required = self.gh.get_required_checks(
                    repo_ctx.repo, repo_ctx.default_branch
                )
                if ci_ready_for_review(checks, required):
                    log.info(
                        "PR #%s: CI passing — requesting review from %s",
                        pr_number,
                        ", ".join(missing),
                    )
                    self.gh.add_pr_reviewers(repo_ctx.repo, pr_number, missing)
                else:
                    log.info(
                        "PR #%s: CI not yet passing — deferring review request",
                        pr_number,
                    )
            return 1

        missing = sorted(repo_ctx.collaborators - set(requested_reviewers))
        if missing and latest_state == "NONE":
            checks = self.gh.pr_checks(repo_ctx.repo, pr_number)
            required = self.gh.get_required_checks(
                repo_ctx.repo, repo_ctx.default_branch
            )
            if ci_ready_for_review(checks, required):
                log.info(
                    "PR #%s: CI passing — requesting review from %s",
                    pr_number,
                    ", ".join(missing),
                )
                self.gh.add_pr_reviewers(repo_ctx.repo, pr_number, missing)
            else:
                log.info(
                    "PR #%s: CI not yet passing — waiting before requesting review",
                    pr_number,
                )
            return 0

        log.info("PR #%s: no work", pr_number)
        self.set_status("Napping — waiting for work", busy=False)
        return 0

    def run(self) -> int:
        """Run one iteration of the worker loop.

        Returns:
            0 — no more work (all done or idle)
            1 — did work (caller should re-run immediately)
            2 — lock held / transient failure (retry later)
        """
        try:
            ctx = self.create_context()
        except LockHeld:
            log.warning("another fido is running — exiting")
            return 2

        with ctx:
            log.info("worker started for %s (git_dir=%s)", self.work_dir, ctx.git_dir)

            repo_ctx = self.discover_repo_context()
            log.info(
                "repo=%s user=%s default_branch=%s",
                repo_ctx.repo,
                repo_ctx.gh_user,
                repo_ctx.default_branch,
            )

            compact_cmd, sync_cmd = self.setup_hooks(ctx.fido_dir)
            try:
                issue = self.get_current_issue(ctx.fido_dir, repo_ctx.repo)
                if issue is None:
                    issue = self.find_next_issue(ctx.fido_dir, repo_ctx)
                if issue is None:
                    return 0

                issue_data = self.gh.view_issue(repo_ctx.repo, issue)
                issue_title = issue_data["title"]
                issue_body = issue_data.get("body", "") or ""
                self.post_pickup_comment(
                    repo_ctx.repo, issue, issue_title, repo_ctx.gh_user
                )
                pr_number, slug = self.find_or_create_pr(
                    ctx.fido_dir, repo_ctx, issue, issue_title, issue_body
                )
                self.seed_tasks_from_pr_body(repo_ctx.repo, pr_number)
                if self.handle_ci(ctx.fido_dir, repo_ctx, pr_number, slug):
                    return 1
                if self.handle_threads(ctx.fido_dir, repo_ctx, pr_number, slug):
                    return 1
                if self.execute_task(ctx.fido_dir, repo_ctx, pr_number, slug):
                    self.resolve_addressed_threads(repo_ctx, pr_number)
                    return 1
                return self.handle_promote_merge(
                    ctx.fido_dir, repo_ctx, pr_number, slug, issue
                )
            finally:
                self.teardown_hooks(ctx.fido_dir, compact_cmd, sync_cmd)


_IDLE_TIMEOUT = 60.0  # seconds to wait when there was no work to do
_RETRY_TIMEOUT = 5.0  # seconds to wait when the lock was contended


class WorkerThread(threading.Thread):
    """Daemon thread that repeatedly calls :class:`Worker` for one repo.

    Loop semantics:
    - ``Worker.run()`` returns 1 → did work, loop immediately.
    - ``Worker.run()`` returns 0 → idle, wait up to ``_IDLE_TIMEOUT``.
    - ``Worker.run()`` returns 2 → lock held, wait up to ``_RETRY_TIMEOUT``.
    - Unexpected exception → propagates, killing the thread (watchdog restarts).

    Call :meth:`wake` to interrupt any wait early (e.g. when a webhook arrives).
    Call :meth:`stop` to request a clean shutdown.
    """

    def __init__(
        self,
        work_dir: Path,
        repo_name: str,
        gh: GitHub,
        registry: ActivityReporter | None = None,
        membership: RepoMembership | None = None,
    ) -> None:
        super().__init__(name=f"worker-{work_dir.name}", daemon=True)
        self.work_dir = work_dir
        self._repo_name = repo_name
        self._gh = gh
        self._registry = registry
        self._membership = membership if membership is not None else RepoMembership()
        self._wake = threading.Event()
        self._abort_task = threading.Event()
        self._stop = False
        self.crash_error: str | None = None

    def wake(self) -> None:
        """Signal the thread to wake up and check for work immediately."""
        self._wake.set()

    def abort_task(self) -> None:
        """Signal the worker to abort the current task after claude_run returns."""
        self._abort_task.set()
        self._wake.set()

    def stop(self) -> None:
        """Request the thread to exit after the current iteration."""
        self._stop = True
        self._wake.set()

    def run(self) -> None:
        """Main loop — runs until :meth:`stop` is called."""
        _thread_repo.repo_name = self._repo_name.split("/")[-1]
        try:
            while not self._stop:
                result = Worker(
                    self.work_dir,
                    self._gh,
                    self._abort_task,
                    self._repo_name,
                    self._registry,
                    self._membership,
                ).run()

                if result == 1:
                    # Did work — loop immediately without waiting.
                    continue

                timeout = _RETRY_TIMEOUT if result == 2 else _IDLE_TIMEOUT
                self._wake.wait(timeout=timeout)
                self._wake.clear()
        except Exception as exc:
            self.crash_error = f"{type(exc).__name__}: {exc}"
            log.exception("WorkerThread %s: unexpected error", self.name)
            raise


def run(work_dir: Path) -> int:
    """Run one iteration of the worker loop.

    Creates a :class:`Worker` with a live :class:`~kennel.github.GitHub` client
    and delegates to :meth:`Worker.run`.  For testing, construct ``Worker``
    directly with a mock ``gh`` instead of patching module-level names.
    """
    return Worker(work_dir, GitHub()).run()
