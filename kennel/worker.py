"""Fido worker — runs one iteration of the work loop for a single repo."""

from __future__ import annotations

import fcntl
import json
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from kennel import claude, hooks, tasks
from kennel.github import GitHub
from kennel.prompts import Prompts

_CI_LOG_TAIL = 200  # max lines of failure log to include in the CI prompt
_SHORTCODE_RE = re.compile(r"^(:[a-z0-9_+\-]+:)\s*(.*)", re.DOTALL)

log = logging.getLogger(__name__)


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
    model: str = "claude-sonnet-4-6",
    timeout: int = 300,
) -> str:
    """Start a new sub-Claude session from fido_dir/system and fido_dir/prompt.

    Returns the session_id string (empty string on failure).
    """
    system_file = fido_dir / "system"
    prompt_file = fido_dir / "prompt"
    output = claude.print_prompt_from_file(system_file, prompt_file, model, timeout)
    return claude.extract_session_id(output)


def claude_run(
    fido_dir: Path,
    session_id: str = "",
    model: str = "claude-sonnet-4-6",
    timeout: int = 300,
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
        output = claude.resume_session(session_id, prompt_file, model, timeout)
        return session_id, output
    system_file = fido_dir / "system"
    output = claude.print_prompt_from_file(system_file, prompt_file, model, timeout)
    new_session_id = claude.extract_session_id(output)
    return new_session_id, output


def _pick_next_task(task_list: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the highest-priority eligible pending task, or ``None``.

    Filters out completed tasks and those prefixed with ``ask:`` or ``defer:``
    (case-insensitive).  Among the remaining candidates the priority order is:

    1. Tasks whose title starts with ``CI failure:``
    2. Tasks that carry a ``thread`` payload (comment-originated)
    3. Everything else (first in list wins)
    """
    pending = [
        t
        for t in task_list
        if t.get("status") == "pending"
        and not t.get("title", "").lower().startswith("ask:")
        and not t.get("title", "").lower().startswith("defer:")
    ]
    if not pending:
        return None
    for t in pending:
        if t.get("title", "").startswith("CI failure:"):
            return t
    for t in pending:
        if t.get("thread") is not None:
            return t
    return pending[0]


class Worker:
    """Fido worker for a single repository.

    Accepts ``work_dir`` and a :class:`~kennel.github.GitHub` client via the
    constructor so that tests can inject a mock without patching module-level
    names.  See :ref:`dependency-injection` in CLAUDE.md.
    """

    def __init__(self, work_dir: Path, gh: GitHub) -> None:
        self.work_dir = work_dir
        self.gh = gh

    def resolve_git_dir(self) -> Path:
        """Return the absolute .git directory for self.work_dir."""
        result = subprocess.run(
            ["git", "rev-parse", "--absolute-git-dir"],
            cwd=self.work_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip())

    def create_context(self) -> WorkerContext:
        """Build a WorkerContext for self.work_dir, acquiring the fido lock.

        Raises LockHeld if the lock is already held.
        """
        git_dir = self.resolve_git_dir()
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
        # TODO: replace with Python sync once sync-tasks.sh is removed
        sync_cmd = f"bash {_sync_script()} {self.work_dir} &"
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
            system_prompt=prompts.status_system_prompt(),
        )
        if not raw:
            log.warning("set_status: claude returned empty — skipping")
            return

        lines = raw.splitlines()
        if len(lines) >= 2:
            emoji = lines[0].strip()
            text = lines[1].strip()[:80]
        else:
            # Single-line output — extract :shortcode: prefix if present,
            # otherwise fall back to :dog: and use the whole line as text.
            m = _SHORTCODE_RE.match(raw.strip())
            if m:
                emoji = m.group(1)
                text = m.group(2).strip()[:80]
            else:
                emoji = ":dog:"
                text = raw.strip()[:80]
            if not text:
                log.warning("set_status: no status text extracted — skipping")
                return

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

    def _git(
        self, args: list[str], check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        """Run a git command in self.work_dir."""
        return subprocess.run(
            ["git", *args],
            cwd=self.work_dir,
            capture_output=True,
            text=True,
            check=check,
        )

    def _build_pr_body(self, request: str, issue: int) -> str:
        """Build the draft PR body: generated description + work-queue section.

        Reads the Fido persona from ``sub/persona.md`` and generates a 2-3
        sentence description via Claude (Opus).  Falls back to plain text if
        Claude returns nothing.  Appends the pending task list (read from
        tasks.json) inside the ``WORK_QUEUE_START/END`` markers used by
        sync-tasks.sh.
        """
        persona_path = _sub_dir() / "persona.md"
        try:
            persona = persona_path.read_text()
        except OSError:
            persona = ""

        plain = f"Working on: {request}. Implementation in progress."
        system_prompt = (
            "You are a GitHub PR description writer."
            " Write a 2-3 sentence description suitable for a GitHub PR body."
            " No markdown headers."
            f" The last line must be a blank line followed by 'Fixes #{issue}.'"
            " on its own line."
        )
        desc = claude.print_prompt_json(
            prompt=f"{persona}\n\nWrite a 2-3 sentence pull request description"
            f" for: {plain}",
            key="description",
            model="claude-opus-4-6",
            system_prompt=system_prompt,
        )
        if not desc:
            desc = plain

        task_list = tasks.list_tasks(self.work_dir)
        pending = [t for t in task_list if t.get("status") == "pending"]
        next_task = _pick_next_task(task_list)
        if pending:
            lines = []
            for t in pending:
                marker = " **→ next**" if t is next_task else ""
                lines.append(f"- [ ] {t['title']}{marker}")
            queue = "\n".join(lines)
        else:
            queue = "<!-- no tasks yet -->"

        return (
            f"{desc}\n\n---\n\n## Work queue\n\n"
            f"<!-- WORK_QUEUE_START -->\n{queue}\n<!-- WORK_QUEUE_END -->"
        )

    def find_or_create_pr(
        self,
        fido_dir: Path,
        repo_ctx: RepoContext,
        issue: int,
        issue_title: str,
    ) -> tuple[int, str] | None:
        """Find or create the branch and draft PR for *issue*.

        Returns ``(pr_number, slug)`` for an open or freshly-created PR,
        or ``None`` if the issue was already resolved (PR merged and issue
        closed).

        Workflow:
        - **Existing merged PR**: close the issue, clear state, return None.
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
            state = existing["state"]
            pr_number = existing["number"]
            slug = existing["headRefName"]

            if state == "MERGED":
                log.info("PR #%s already merged — closing issue #%s", pr_number, issue)
                self.gh.close_issue(repo_ctx.repo, issue)
                clear_state(fido_dir)
                return None

            if state != "CLOSED":
                # Open PR — resume
                log.info("resuming PR #%s on branch %s", pr_number, slug)
                self._git(["fetch", remote])
                try:
                    self._git(["checkout", slug])
                except subprocess.CalledProcessError:
                    self._git(["checkout", "-b", slug, "--track", f"{remote}/{slug}"])
                task_list = tasks.list_tasks(self.work_dir)
                if not task_list:
                    log.info("PR #%s has no tasks — running setup", pr_number)
                    context = (
                        f"Request: {request}\n"
                        f"Repo: {repo_ctx.repo}\n"
                        f"Branch: {slug}\n"
                        f"PR: {pr_number}\n"
                        f"Fork remote: {remote}\n"
                        f"Upstream: {remote}/{repo_ctx.default_branch}"
                    )
                    build_prompt(fido_dir, "setup", context)
                    session_id = claude_start(fido_dir)
                    log.info("setup session: %s", session_id)
                    if not tasks.list_tasks(self.work_dir):
                        log.warning(
                            "setup produced no tasks — skipping PR #%s, will retry",
                            pr_number,
                        )
                        return None
                log.info(
                    "PR: #%s  https://github.com/%s/pull/%s",
                    pr_number,
                    repo_ctx.repo,
                    pr_number,
                )
                return pr_number, slug

            # CLOSED — fall through to create a fresh PR
            log.info("PR #%s closed without merge — creating fresh PR", pr_number)

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
        self._git(["fetch", remote])
        try:
            self._git(["checkout", "-b", slug, f"{remote}/{repo_ctx.default_branch}"])
        except subprocess.CalledProcessError:
            self._git(["checkout", slug])
        self._git(["commit", "--allow-empty", "-m", "wip: start"])
        self._git(["push", "-u", remote, slug])

        # Run setup sub-Claude (plans tasks before PR is created)
        log.info("running setup (pre-PR)")
        context = (
            f"Request: {request}\n"
            f"Repo: {repo_ctx.repo}\n"
            f"Branch: {slug}\n"
            f"Fork remote: {remote}\n"
            f"Upstream: {remote}/{repo_ctx.default_branch}"
        )
        build_prompt(fido_dir, "setup", context)
        session_id = claude_start(fido_dir)
        log.info("setup session: %s", session_id)

        if not tasks.list_tasks(self.work_dir):
            log.warning("setup produced no tasks — skipping PR creation, will retry")
            return None

        # Build PR body with tasks already populated by setup
        pr_body = self._build_pr_body(request, issue)

        # Create draft PR
        url = self.gh.create_pr(
            repo_ctx.repo,
            request,
            pr_body,
            repo_ctx.default_branch,
            slug,
        )
        pr_number = int(url.rstrip("/").split("/")[-1])
        task_count = len(
            [t for t in tasks.list_tasks(self.work_dir) if t.get("status") == "pending"]
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
        threads_data: dict[str, Any],
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
        nodes = (
            threads_data.get("data", {})
            .get("repository", {})
            .get("pullRequest", {})
            .get("reviewThreads", {})
            .get("nodes", [])
        )
        result = []
        for node in nodes:
            if node.get("isResolved"):
                continue
            comments = node.get("comments", {}).get("nodes", [])
            if not comments:
                continue
            last = comments[-1]
            if last.get("author", {}).get("login") == gh_user:
                continue
            bodies = " ".join(c.get("body", "").lower() for c in comments)
            if not any(kw in bodies for kw in keywords):
                continue
            result.append(
                {
                    "first_author": comments[0].get("author", {}).get("login", ""),
                    "first_body": comments[0].get("body", ""),
                    "last_author": last.get("author", {}).get("login", ""),
                    "last_body": last.get("body", ""),
                    "url": comments[0].get("url", ""),
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

        threads_data = self.gh.get_review_threads(
            repo_ctx.owner, repo_ctx.repo_name, pr_number
        )
        ci_threads = self._filter_ci_threads(threads_data, repo_ctx.gh_user, check_name)

        context = (
            f"PR: {pr_number}\n"
            f"Repo: {repo_ctx.repo}\n"
            f"Branch: {slug}\n"
            f"Upstream: origin/{repo_ctx.default_branch}\n"
            f"Failing check: {check_name}\n"
            f"\nFailure log (last {_CI_LOG_TAIL} lines):\n{failure_log}\n"
            f"\nReview threads related to this CI failure"
            f" (JSON — may be empty):\n{json.dumps(ci_threads)}"
        )
        build_prompt(fido_dir, "ci", context)
        session_id, _ = claude_run(fido_dir)
        log.info("CI fix done (session=%s)", session_id)

        tasks.complete_by_title(self.work_dir, f"CI failure: {check_name}")
        subprocess.Popen(  # noqa: S603
            ["bash", str(_sync_script()), str(self.work_dir)],
        )
        return True

    def _filter_threads(
        self,
        threads_data: dict[str, Any],
        gh_user: str,
        owner: str,
    ) -> list[dict[str, Any]]:
        """Return unresolved review threads for the comments sub-Claude.

        A thread is included when:
        - it is not resolved,
        - it has at least one comment,
        - the last commenter is not *gh_user* (awaiting a response), and
        - the last commenter is either *owner* or ends with ``[bot]``.
        """
        nodes = (
            threads_data.get("data", {})
            .get("repository", {})
            .get("pullRequest", {})
            .get("reviewThreads", {})
            .get("nodes", [])
        )
        result = []
        for node in nodes:
            if node.get("isResolved"):
                continue
            comments = node.get("comments", {}).get("nodes", [])
            if not comments:
                continue
            first_comment = comments[0]
            last_comment = comments[-1]
            last_author = last_comment.get("author", {}).get("login", "")
            if last_author == gh_user:
                continue
            if last_author != owner and not last_author.endswith("[bot]"):
                continue
            result.append(
                {
                    "id": node.get("id", ""),
                    "is_bot": first_comment.get("author", {})
                    .get("login", "")
                    .endswith("[bot]"),
                    "first_author": first_comment.get("author", {}).get("login", ""),
                    "first_db_id": first_comment.get("databaseId"),
                    "first_body": first_comment.get("body", ""),
                    "last_author": last_author,
                    "last_body": last_comment.get("body", ""),
                    "url": first_comment.get("url", ""),
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
        threads_data = self.gh.get_review_threads(
            repo_ctx.owner, repo_ctx.repo_name, pr_number
        )
        nodes = (
            threads_data.get("data", {})
            .get("repository", {})
            .get("pullRequest", {})
            .get("reviewThreads", {})
            .get("nodes", [])
        )
        resolved_any = False
        for node in nodes:
            if node.get("isResolved"):
                continue
            comments = node.get("comments", {}).get("nodes", [])
            if not comments:
                continue
            last_author = comments[-1].get("author", {}).get("login", "")
            if last_author != repo_ctx.gh_user:
                continue
            thread_id = node.get("id", "")
            self.gh.resolve_thread(thread_id)
            log.info("resolved thread %s", thread_id)
            resolved_any = True
        return resolved_any

    def handle_review_feedback(
        self,
        fido_dir: Path,
        repo_ctx: RepoContext,
        pr_number: int,
        slug: str,
    ) -> bool:
        """Check for a CHANGES_REQUESTED review with no newer commits and handle it.

        Returns ``True`` if review feedback was found and handled.  Returns
        ``False`` if there is no actionable review.
        """
        pr_data = self.gh.get_pr(repo_ctx.repo, pr_number)
        reviews = pr_data.get("reviews", [])
        owner_reviews = [
            r for r in reviews if r.get("author", {}).get("login") == repo_ctx.owner
        ]
        if not owner_reviews:
            return False
        review = owner_reviews[-1]
        if review.get("state") != "CHANGES_REQUESTED":
            return False
        commits = pr_data.get("commits", [])
        if commits:
            last_commit_date = commits[-1].get("committedDate", "")
            if last_commit_date > review.get("submittedAt", ""):
                return False
        review_body = review.get("body", "")
        if not review_body:
            return False
        log.info("review feedback from %s", repo_ctx.owner)
        context = (
            f"PR: {pr_number}\n"
            f"Repo: {repo_ctx.repo}\n"
            f"Branch: {slug}\n"
            f"Upstream: origin/{repo_ctx.default_branch}\n"
            f"\nTask title: Address review feedback from {repo_ctx.owner}\n"
            f"Task description: {repo_ctx.owner} submitted a review requesting"
            f" changes with the following feedback. Address it, commit, and push.\n"
            f"\nReview feedback:\n{review_body}"
        )
        build_prompt(fido_dir, "task", context)
        session_id, _ = claude_run(fido_dir)
        log.info("review feedback done (session=%s)", session_id)
        tasks.complete_by_title(
            self.work_dir, f"Address review feedback from {repo_ctx.owner}"
        )
        subprocess.Popen(  # noqa: S603
            ["bash", str(_sync_script()), str(self.work_dir)],
        )
        return True

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
        threads_data = self.gh.get_review_threads(
            repo_ctx.owner, repo_ctx.repo_name, pr_number
        )
        threads = self._filter_threads(threads_data, repo_ctx.gh_user, repo_ctx.owner)
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
            f"GitHub user: {repo_ctx.gh_user}\n"
            f"\nUnresolved threads (JSON):\n{json.dumps({'threads': threads})}"
        )
        build_prompt(fido_dir, "comments", context)
        session_id, _ = claude_run(fido_dir)
        log.info("threads done (session=%s)", session_id)
        subprocess.Popen(  # noqa: S603
            ["bash", str(_sync_script()), str(self.work_dir)],
        )
        return True

    def ensure_pushed(self, remote: str, slug: str) -> bool:
        """Ensure the branch is pushed to *remote*.

        Compares the local HEAD SHA to ``remote/slug``.  If they already
        match, returns ``True`` immediately (no push needed).  Otherwise
        attempts ``git push`` and returns ``True`` on success, ``False`` if
        the push fails.
        """
        local = self._git(["rev-parse", "HEAD"]).stdout.strip()
        remote_ref = f"{remote}/{slug}"
        result = self._git(["rev-parse", remote_ref], check=False)
        if result.returncode == 0 and result.stdout.strip() == local:
            return True
        log.info("pushing %s to %s", slug, remote)
        push = self._git(["push", "-u", remote, slug], check=False)
        if push.returncode != 0:
            log.warning("push failed: %s", push.stderr.strip())
            return False
        return True

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
        task_list = tasks.list_tasks(self.work_dir)
        task = _pick_next_task(task_list)
        if task is None:
            return False
        task_title = task["title"]
        log.info("task: %s", task_title)
        self.set_status(f"Working on: {task_title}")
        context = (
            f"PR: {pr_number}\n"
            f"Repo: {repo_ctx.repo}\n"
            f"Branch: {slug}\n"
            f"Upstream: origin/{repo_ctx.default_branch}\n"
            f"\nTask title: {task_title}"
        )
        build_prompt(fido_dir, "task", context)
        session_id, _ = claude_run(fido_dir)
        log.info("task done (session=%s)", session_id)
        if not self.ensure_pushed("origin", slug):
            return True
        tasks.complete_by_title(self.work_dir, task_title)
        subprocess.Popen(  # noqa: S603
            ["bash", str(_sync_script()), str(self.work_dir)],
        )
        return True

    def seed_tasks_from_pr_body(self, repo: str, pr_number: int) -> None:
        """Seed tasks.json from the PR body work-queue markers if tasks.json is empty.

        Extracts unchecked task items (``- [ ] ...``) between
        ``WORK_QUEUE_START`` and ``WORK_QUEUE_END`` markers and adds them via
        :func:`~kennel.tasks.add_task`.  No-op if tasks.json is already
        non-empty, or if no markers / unchecked items are found.
        """
        if tasks.list_tasks(self.work_dir):
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
        task_titles = []
        for line in match.group(1).splitlines():
            m = re.match(r"^- \[ \] (.+)$", line)
            if m:
                title = re.sub(r"\s*\*\*→ next\*\*\s*", "", m.group(1)).strip()
                task_titles.append(title)
        if not task_titles:
            return
        for title in task_titles:
            tasks.add_task(self.work_dir, title)
        log.info("seeded %d tasks from PR body", len(task_titles))

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
        re-requests review, promotes a draft PR, or sets the idle status.

        Returns:
            0 — no more work (auto-merge enabled, changes-requested,
                setup not complete, or no outstanding work)
            1 — did work (merged and cleaned up, or promoted draft;
                caller should re-run immediately)
        """
        log.info("PR #%s: checking review status", pr_number)
        reviews_data = self.gh.get_reviews(repo_ctx.repo, pr_number)
        reviews = reviews_data.get("reviews", [])
        commits = reviews_data.get("commits", [])
        is_draft = reviews_data.get("isDraft", False)

        is_approved = any(
            r.get("author", {}).get("login") == repo_ctx.owner
            and r.get("state") == "APPROVED"
            for r in reviews
        )
        task_list = tasks.list_tasks(self.work_dir)
        pending = [t for t in task_list if t.get("status") == "pending"]

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
            self.gh.close_issue(repo_ctx.repo, issue)
            (fido_dir / "tasks.json").write_text("[]")
            clear_state(fido_dir)
            self._git(["checkout", repo_ctx.default_branch])
            self._git(
                ["pull", "origin", repo_ctx.default_branch, "--ff-only"], check=False
            )
            self._git(["branch", "-d", slug], check=False)
            self.set_status(f"Merged PR #{pr_number}! Issue #{issue} done")
            return 1

        owner_reviews = [
            r for r in reviews if r.get("author", {}).get("login") == repo_ctx.owner
        ]
        latest_state = (
            owner_reviews[-1].get("state", "NONE") if owner_reviews else "NONE"
        )

        if latest_state == "CHANGES_REQUESTED":
            log.info(
                "PR #%s: changes requested — all addressed, re-requesting review",
                pr_number,
            )
            self.gh.add_pr_reviewer(repo_ctx.repo, pr_number, repo_ctx.owner)
            return 0

        if is_draft:
            completed = [t for t in task_list if t.get("status") == "completed"]
            if not completed:
                log.info(
                    "PR #%s: no tasks completed — not promoting (setup may have failed)",
                    pr_number,
                )
                return 0
            log.info(
                "PR #%s: work complete — marking ready, requesting review from %s",
                pr_number,
                repo_ctx.owner,
            )
            self.gh.pr_ready(repo_ctx.repo, pr_number)
            self.gh.add_pr_reviewer(repo_ctx.repo, pr_number, repo_ctx.owner)
            return 1

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
            self.post_pickup_comment(
                repo_ctx.repo, issue, issue_title, repo_ctx.gh_user
            )
            result = self.find_or_create_pr(ctx.fido_dir, repo_ctx, issue, issue_title)
            if result is None:
                return 0
            pr_number, slug = result
            self.seed_tasks_from_pr_body(repo_ctx.repo, pr_number)
            if self.handle_ci(ctx.fido_dir, repo_ctx, pr_number, slug):
                return 1
            if self.handle_review_feedback(ctx.fido_dir, repo_ctx, pr_number, slug):
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

        return 0


def run(work_dir: Path) -> int:
    """Run one iteration of the worker loop.

    Creates a :class:`Worker` with a live :class:`~kennel.github.GitHub` client
    and delegates to :meth:`Worker.run`.  For testing, construct ``Worker``
    directly with a mock ``gh`` instead of patching module-level names.
    """
    return Worker(work_dir, GitHub()).run()
